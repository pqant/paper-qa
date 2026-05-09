"""OpenAlex comparison — compare corpus coverage against global taxonomy.

Queries OpenAlex API for topics related to bin packing / container loading.
Compares with BERTopic topics to find coverage gaps.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from paperqa.stores.gap_analysis import OpenAlexResult, TopicResult

logger = logging.getLogger(__name__)

import os

_DEFAULT_SEARCH_TERMS = [
    "bin packing",
    "container loading",
    "cutting stock",
    "3D loading",
    "packing problem",
    "loading problem",
    "container loading problem",
    "three-dimensional bin packing",
    "orthogonal packing",
    "stock cutting",
    "knapsack problem",
    "loading optimization",
]


def get_domain_search_terms() -> list[str]:
    """Return OpenAlex search terms from env or defaults.

    Set OPENALEX_SEARCH_TERMS as pipe-separated to override:
    OPENALEX_SEARCH_TERMS="deep learning|transformer|attention mechanism"
    """
    env = os.getenv("OPENALEX_SEARCH_TERMS")
    if env:
        return [t.strip() for t in env.split("|") if t.strip()]
    return _DEFAULT_SEARCH_TERMS


async def fetch_openalex_topics() -> list[dict]:
    """Query OpenAlex API for topics related to our domain.

    Uses /works to find related topics, then /topics/{id} for works_count.
    Returns list of {id, display_name, works_count}.
    """
    import httpx

    # Step 1: Collect unique topic IDs from work searches
    topic_info: dict[str, dict] = {}  # id → {display_name, ...}

    async with httpx.AsyncClient(timeout=60.0) as client:
        for term in get_domain_search_terms():
            try:
                url = "https://api.openalex.org/works"
                params = {
                    "search": term,
                    "per_page": 200,
                    "sort": "cited_by_count:desc",
                    "select": "id,primary_topic",
                }
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

                for work in data.get("results", []):
                    primary_topic = work.get("primary_topic")
                    if primary_topic:
                        tid = primary_topic["id"]
                        if tid not in topic_info:
                            topic_info[tid] = {
                                "id": tid,
                                "display_name": primary_topic["display_name"],
                            }

                logger.info("OpenAlex '%s': %d works, %d unique topics so far",
                            term, len(data.get("results", [])), len(topic_info))

            except Exception as e:
                logger.warning("OpenAlex query '%s' failed: %s", term, e)

        # Step 2: Fetch works_count for each topic concurrently (limited)
        topics_list: list[dict] = []
        topic_ids = list(topic_info.keys())
        semaphore = asyncio.Semaphore(10)  # max 10 concurrent requests

        async def fetch_topic(tid: str) -> dict:
            async with semaphore:
                try:
                    resp = await client.get(
                        f"https://api.openalex.org/topics/{tid}",
                        params={"select": "id,display_name,works_count"},
                    )
                    resp.raise_for_status()
                    topic = resp.json()
                    info = topic_info.get(tid, {})
                    return {
                        "id": topic["id"],
                        "display_name": topic.get("display_name", info.get("display_name", "")),
                        "works_count": topic.get("works_count", 0),
                    }
                except Exception as e:
                    logger.warning("OpenAlex topic %s fetch failed: %s", tid, e)
                    info = topic_info[tid].copy()
                    info["works_count"] = 0
                    return info

        # Fetch all topics concurrently (with semaphore limiting)
        results = await asyncio.gather(
            *(fetch_topic(tid) for tid in topic_ids),
        )
        topics_list = list(results)

    logger.info("Fetched %d OpenAlex topics with works_count", len(topics_list))
    return topics_list


async def compute_coverage_embedding(
    openalex_topics: list[dict],
    topic_result: TopicResult,
    embedding_api_base: str | None = None,
    embedding_model: str | None = None,
    similarity_threshold: float = 0.65,
) -> tuple[list[dict], list[dict]]:
    """Match BERTopic topics to OpenAlex topics via embedding cosine similarity.

    For each OpenAlex topic, find the best-matching BERTopic topic by
    embedding similarity. If no BERTopic topic matches above the threshold,
    the OpenAlex topic is considered "missing" from our corpus.

    Returns (our_coverage, missing_topics).
    """
    import httpx
    import numpy as np

    api_base = embedding_api_base or os.getenv("EMBEDDING_API_BASE", "http://192.168.0.28:8082/v1")
    model = embedding_model or os.getenv("EMBEDDING_MODEL", "hf.co/Qwen/Qwen3-Embedding-8B-GGUF:F16")

    # Build BERTopic topic labels: "keyword1 keyword2 keyword3 ..."
    bertopic_labels: list[str] = []
    bertopic_ids: list[int] = []
    for tid, words in topic_result.topics.items():
        bertopic_labels.append(" ".join(words[:8]))
        bertopic_ids.append(tid)

    if not bertopic_labels or not openalex_topics:
        logger.warning("Empty topics: BERTopic=%d, OpenAlex=%d", len(bertopic_labels), len(openalex_topics))
        return [], []

    openalex_names = [t["display_name"] for t in openalex_topics]

    # Embed both sets
    async def embed_batch(texts: list[str]) -> np.ndarray:
        async with httpx.AsyncClient(timeout=120.0) as client:
            all_vecs = []
            batch_size = 64
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                resp = await client.post(
                    f"{api_base}/embeddings",
                    json={"input": batch, "model": model},
                )
                resp.raise_for_status()
                data = resp.json()["data"]
                vecs = [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]
                all_vecs.extend(vecs)
            return np.array(all_vecs, dtype=np.float32)

    logger.info("Embedding %d BERTopic labels + %d OpenAlex topics for coverage...",
                len(bertopic_labels), len(openalex_names))

    bert_vecs = await embed_batch(bertopic_labels)
    oa_vecs = await embed_batch(openalex_names)

    # Normalize for cosine similarity
    bert_norms = bert_vecs / (np.linalg.norm(bert_vecs, axis=1, keepdims=True) + 1e-10)
    oa_norms = oa_vecs / (np.linalg.norm(oa_vecs, axis=1, keepdims=True) + 1e-10)

    # similarity matrix: (num_openalex, num_bertopic)
    sim_matrix = oa_norms @ bert_norms.T

    # Build topic→chunk count
    topic_counts: dict[int, int] = {}
    for t in topic_result.chunk_to_topic:
        if t != -1:
            topic_counts[t] = topic_counts.get(t, 0) + 1

    our_coverage: list[dict] = []
    missing_topics: list[dict] = []

    for i, topic in enumerate(openalex_topics):
        best_idx = int(np.argmax(sim_matrix[i]))
        best_sim = float(sim_matrix[i, best_idx])
        best_bertopic_id = bertopic_ids[best_idx]

        our_count = topic_counts.get(best_bertopic_id, 0) if best_sim >= similarity_threshold else 0
        global_count = topic.get("works_count", 1)
        ratio = our_count / max(global_count, 1)

        entry = {
            "topic_name": topic["display_name"],
            "our_count": our_count,
            "global_count": global_count,
            "ratio": round(ratio, 4),
            "best_match_similarity": round(best_sim, 4),
            "best_match_topic": " ".join(topic_result.topics.get(best_bertopic_id, [])[:5]),
        }
        our_coverage.append(entry)

        if best_sim < similarity_threshold:
            missing_topics.append({
                "topic_name": topic["display_name"],
                "our_count": 0,
                "global_count": global_count,
                "ratio": 0.0,
                "best_match_similarity": round(best_sim, 4),
            })

    missing_topics.sort(key=lambda x: -x.get("best_match_similarity", 0))
    our_coverage.sort(key=lambda x: x["best_match_similarity"])

    logger.info("Embedding coverage: %d/%d OpenAlex topics are missing (threshold=%.2f)",
                len(missing_topics), len(openalex_topics), similarity_threshold)
    return our_coverage, missing_topics


async def run_openalex(
    topic_result: TopicResult,
    output_dir: str = "./data/gap_analysis/openalex",
    embedding_api_base: str | None = None,
    embedding_model: str | None = None,
) -> OpenAlexResult:
    """Run OpenAlex comparison: fetch topics → embedding coverage → save."""
    global_topics = await fetch_openalex_topics()

    our_coverage, missing_topics = await compute_coverage_embedding(
        global_topics,
        topic_result,
        embedding_api_base=embedding_api_base,
        embedding_model=embedding_model,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "global_topics.json", "w") as f:
        json.dump(global_topics, f, indent=2)

    with open(output_path / "coverage.json", "w") as f:
        json.dump(our_coverage, f, indent=2)

    with open(output_path / "missing_topics.json", "w") as f:
        json.dump(missing_topics, f, indent=2)

    logger.info("Saved OpenAlex artifacts to %s", output_path)

    return OpenAlexResult(
        global_topics=global_topics,
        our_coverage=our_coverage,
        missing_topics=missing_topics,
    )
