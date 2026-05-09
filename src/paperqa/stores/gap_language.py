"""Gap-Language Extraction — find explicit gap signals in papers.

Uses Tantivy index for keyword search, clusters results with BERTopic.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from paperqa.agents.search import SearchIndex
from paperqa.stores.gap_analysis import GapLanguageResult

logger = logging.getLogger(__name__)

# Keywords that signal research gaps
GAP_KEYWORDS = [
    "future work",
    "future research",
    "limitation",
    "limitations",
    "not been studied",
    "remains unclear",
    "under-explored",
    "no study has",
    "research gap",
    "open problem",
    "yet to be",
    "warrants further",
    "needs investigation",
    "further research is needed",
    "direction for future",
    "potential limitation",
]


async def extract_gap_chunks(
    index_dir: str = "./data/tantivy_index",
    top_n: int = 20,
    qdrant_url: str | None = None,
    qdrant_collection: str | None = None,
) -> list[dict]:
    """Search Tantivy index for gap-language keywords, then fetch real text from Qdrant.

    Returns list of {pdf_hash, title, year, keyword_matched, text, page_num}.
    """
    import os

    _qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://192.168.0.28:6333")
    _collection = qdrant_collection or os.getenv("QDRANT_COLLECTION", "paperbridge_glm_v2")

    index = SearchIndex(
        fields=["file_location", "body", "title", "year"],
        index_name="paperbridge_index",
        index_directory=index_dir,
    )

    # Step 1: Find papers mentioning gap keywords via Tantivy
    gap_paper_hashes: dict[str, dict] = {}

    for keyword in GAP_KEYWORDS:
        try:
            results = await index.query(keyword, top_n=top_n)
        except Exception as e:
            logger.warning("Tantivy query '%s' failed: %s", keyword, e)
            continue

        for r in results:
            if not isinstance(r, dict):
                continue
            pdf_hash = r.get("pdf_hash")
            if not pdf_hash or pdf_hash in gap_paper_hashes:
                continue
            gap_paper_hashes[pdf_hash] = {
                "title": r.get("title", ""),
                "year": r.get("year"),
                "keyword_matched": keyword,
            }

    logger.info("Tantivy found %d papers with gap-language", len(gap_paper_hashes))

    if not gap_paper_hashes:
        return []

    # Step 2: Fetch actual gap text from Qdrant chunks of these papers
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchAny

    client = AsyncQdrantClient(url=_qdrant_url)
    gap_chunks: list[dict] = []
    seen_keys: set[str] = set()

    hash_list = list(gap_paper_hashes.keys())
    keyword_patterns = [kw.lower() for kw in GAP_KEYWORDS]

    # Scroll chunks from gap papers in batches
    batch_size = 100
    for i in range(0, len(hash_list), batch_size):
        batch_hashes = hash_list[i : i + batch_size]
        try:
            scroll_result = await client.scroll(
                collection_name=_collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="pdf_hash", match=MatchAny(any=batch_hashes))]
                ),
                limit=500,
                with_payload=True,
                with_vectors=False,
            )
            points = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result

            for pt in points:
                payload = pt.payload or {}
                text = payload.get("text", "")
                text_lower = text.lower()

                matched_keyword = None
                for kw in keyword_patterns:
                    if kw in text_lower:
                        matched_keyword = kw
                        break

                if not matched_keyword:
                    continue

                pdf_hash = payload.get("pdf_hash", "")
                page_num = payload.get("page_num", 0)
                key = f"{pdf_hash}:{page_num}:{hash(text[:80])}"
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                paper_info = gap_paper_hashes.get(pdf_hash, {})
                gap_chunks.append({
                    "pdf_hash": pdf_hash,
                    "title": paper_info.get("title", payload.get("pdf_name", "")),
                    "year": paper_info.get("year"),
                    "keyword_matched": matched_keyword,
                    "text": text[:500],
                    "page_num": page_num,
                })

        except Exception as e:
            logger.warning("Qdrant scroll for gap papers failed: %s", e)

    await client.close()

    logger.info("Found %d gap chunks with real text from %d papers",
                len(gap_chunks), len(gap_paper_hashes))
    return gap_chunks


async def cluster_gap_chunks(
    gap_chunks: list[dict],
    output_dir: str = "./data/gap_analysis/gap_language",
) -> GapLanguageResult:
    """Cluster gap chunks using BERTopic to find gap categories.

    Uses chunk titles + keywords as text for clustering.
    """
    from bertopic import BERTopic

    if not gap_chunks:
        return GapLanguageResult(gap_chunks=[], gap_categories={}, top_gaps=[])

    # Build text representations: prefer real chunk text if available, else title+keyword
    texts = []
    for chunk in gap_chunks:
        body = chunk.get("text", "")
        if body and len(body) > 20:
            texts.append(body[:300])
        else:
            texts.append(f"{chunk['title']} {chunk['keyword_matched']}")

    try:
        topic_model = BERTopic(verbose=False)
        topics, _ = topic_model.fit_transform(texts)

        topic_info = topic_model.get_topic_info()

        # Build gap categories with representative texts
        topic_to_chunks: dict[int, list[dict]] = {}
        for chunk, topic_id in zip(gap_chunks, topics):
            if topic_id == -1:
                continue
            topic_to_chunks.setdefault(topic_id, []).append(chunk)

        gap_categories: dict[int, dict] = {}
        for _, row in topic_info.iterrows():
            if row["Topic"] == -1:
                continue
            topic_id = int(row["Topic"])
            words = topic_model.get_topic(topic_id)
            word_list = []
            if words:
                for w in words:
                    if isinstance(w, (tuple, list)):
                        word_list.append(str(w[0]))
                    else:
                        word_list.append(str(w))

            rep_chunks = topic_to_chunks.get(topic_id, [])
            representative_texts = [
                c.get("text", c.get("title", ""))[:300]
                for c in rep_chunks[:5]
                if c.get("text") or c.get("title")
            ]

            gap_categories[topic_id] = {
                "label": row["Name"],
                "count": int(row["Count"]),
                "words": word_list,
                "representative_texts": representative_texts,
            }
    except Exception as e:
        logger.warning("BERTopic clustering failed for gap chunks: %s. Using flat list.", e)
        rep_texts = [
            c.get("text", c.get("title", ""))[:300]
            for c in gap_chunks[:5]
            if c.get("text") or c.get("title")
        ]
        gap_categories = {
            0: {
                "label": "all_gaps",
                "count": len(gap_chunks),
                "words": [],
                "representative_texts": rep_texts,
            },
        }

    # Top gap phrases (most common keywords)
    keyword_counts = Counter(chunk["keyword_matched"] for chunk in gap_chunks)
    top_gaps = [kw for kw, _ in keyword_counts.most_common(20)]

    # Save artifacts
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "gap_chunks.json", "w") as f:
        json.dump(gap_chunks, f, indent=2)

    with open(output_path / "categories.json", "w") as f:
        json.dump({str(k): v for k, v in gap_categories.items()}, f, indent=2)

    logger.info("Saved gap-language artifacts: %d chunks, %d categories",
                len(gap_chunks), len(gap_categories))

    return GapLanguageResult(
        gap_chunks=gap_chunks,
        gap_categories=gap_categories,
        top_gaps=top_gaps,
    )


async def run_gap_language(
    output_dir: str = "./data/gap_analysis/gap_language",
    domain_filter: "EmbeddingDomainFilter | None" = None,
    index_dir: str = "./paper-qa/data/tantivy_index",
    qdrant_url: str | None = None,
    qdrant_collection: str | None = None,
) -> GapLanguageResult:
    """Run the full gap-language pipeline: extract → filter → cluster → save.

    Args:
        output_dir: Where to save artifacts.
        domain_filter: If provided, gap chunks are filtered by embedding
                       domain relevance (title is embedded and scored).
        index_dir: Tantivy index directory.
        qdrant_url: Qdrant server URL for fetching actual gap text.
        qdrant_collection: Qdrant collection name.
    """
    gap_chunks = await extract_gap_chunks(
        index_dir=index_dir,
        qdrant_url=qdrant_url,
        qdrant_collection=qdrant_collection,
    )

    if domain_filter and domain_filter.initialized:
        titles = [ch.get("title", "") for ch in gap_chunks]
        if titles:
            vecs = await domain_filter._embed_batch(titles)
            scores = domain_filter.score_embeddings_batch(vecs)
            threshold = domain_filter.core_threshold
            before = len(gap_chunks)
            gap_chunks = [
                {**ch, "domain_score": round(float(s), 4)}
                for ch, s in zip(gap_chunks, scores)
                if float(s) >= threshold
            ]
            logger.info(
                "Domain-filtered gap chunks: %d → %d (threshold=%.2f, removed %d irrelevant)",
                before, len(gap_chunks), threshold, before - len(gap_chunks),
            )

    return await cluster_gap_chunks(gap_chunks, output_dir)
