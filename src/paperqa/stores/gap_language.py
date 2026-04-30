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
) -> list[dict]:
    """Search Tantivy index for gap-language keywords.

    Returns list of {pdf_hash, title, year, keyword_matched, file_location}.
    """
    index = SearchIndex(
        fields=["file_location", "body", "title", "year"],
        index_name="paperbridge_index",
        index_directory=index_dir,
    )

    gap_chunks: list[dict] = []
    seen_hashes: set[str] = set()

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
            if not pdf_hash or pdf_hash in seen_hashes:
                continue
            seen_hashes.add(pdf_hash)

            gap_chunks.append({
                "pdf_hash": pdf_hash,
                "title": r.get("title", ""),
                "year": r.get("year"),
                "keyword_matched": keyword,
                "file_location": r.get("file_location", ""),
            })
            logger.info("Found gap chunk: '%s' in %s", keyword, r.get("title", "")[:50])

    logger.info("Found %d unique chunks with gap-language", len(gap_chunks))
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

    # Build text representations for clustering
    texts = [f"{chunk['title']} {chunk['keyword_matched']}" for chunk in gap_chunks]

    try:
        # Fit BERTopic on gap chunks
        topic_model = BERTopic(
            verbose=False,
        )
        topics, _ = topic_model.fit_transform(texts)

        topic_info = topic_model.get_topic_info()

        # Build gap categories
        gap_categories: dict[int, dict] = {}
        for _, row in topic_info.iterrows():
            if row["Topic"] == -1:
                continue
            topic_id = int(row["Topic"])
            words = topic_model.get_topic(topic_id)
            # Handle both list[str] and list[tuple[str, float]] formats
            word_list = []
            if words:
                for w in words:
                    if isinstance(w, (tuple, list)):
                        word_list.append(str(w[0]))
                    else:
                        word_list.append(str(w))
            gap_categories[topic_id] = {
                "label": row["Name"],
                "count": int(row["Count"]),
                "words": word_list,
            }
    except Exception as e:
        logger.warning("BERTopic clustering failed for gap chunks: %s. Using flat list.", e)
        # Fallback: no clustering, just group by keyword
        gap_categories = {0: {"label": "all_gaps", "count": len(gap_chunks), "words": []}}

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
) -> GapLanguageResult:
    """Run the full gap-language pipeline: extract → cluster → save."""
    gap_chunks = await extract_gap_chunks()
    return await cluster_gap_chunks(gap_chunks, output_dir)
