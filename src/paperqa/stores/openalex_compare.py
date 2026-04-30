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

# Search terms for OpenAlex topic discovery
DOMAIN_SEARCH_TERMS = [
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


async def fetch_openalex_topics() -> list[dict]:
    """Query OpenAlex API for topics related to our domain.

    Uses /works to find related topics, then /topics/{id} for works_count.
    Returns list of {id, display_name, works_count}.
    """
    import httpx

    # Step 1: Collect unique topic IDs from work searches
    topic_info: dict[str, dict] = {}  # id → {display_name, ...}

    async with httpx.AsyncClient(timeout=60.0) as client:
        for term in DOMAIN_SEARCH_TERMS:
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


def compute_coverage(
    openalex_topics: list[dict],
    topic_result: TopicResult,
    threshold: float = 0.01,
) -> tuple[list[dict], list[dict]]:
    """Match BERTopic topics to OpenAlex topics and compute coverage.

    Returns (our_coverage, missing_topics).
    """
    # Build BERTopic keyword set
    bertopic_keywords: set[str] = set()
    for topic_id, words in topic_result.topics.items():
        for word in words:
            bertopic_keywords.add(word.lower())

    # Build topic→chunk count map
    topic_counts: dict[int, int] = {}
    for t in topic_result.chunk_to_topic:
        if t != -1:
            topic_counts[t] = topic_counts.get(t, 0) + 1

    our_coverage: list[dict] = []
    missing_topics: list[dict] = []

    for topic in openalex_topics:
        topic_name = topic["display_name"].lower()
        # Count how many BERTopic keywords match this OpenAlex topic
        matching_keywords = sum(1 for kw in bertopic_keywords if kw in topic_name)

        # Estimate our count from matching BERTopic topics
        our_count = 0
        for tid, words in topic_result.topics.items():
            for word in words:
                if word.lower() in topic_name:
                    our_count += topic_counts.get(tid, 0)
                    break

        global_count = topic.get("works_count", 1)
        ratio = our_count / max(global_count, 1)

        our_coverage.append({
            "topic_name": topic["display_name"],
            "our_count": our_count,
            "global_count": global_count,
            "ratio": round(ratio, 4),
            "matching_keywords": matching_keywords,
        })

        if ratio < threshold:
            missing_topics.append({
                "topic_name": topic["display_name"],
                "our_count": our_count,
                "global_count": global_count,
                "ratio": round(ratio, 4),
            })

    # Sort by ratio ascending (most missing first)
    missing_topics.sort(key=lambda x: x["ratio"])
    our_coverage.sort(key=lambda x: x["ratio"])

    logger.info("Coverage computed: %d missing topics (threshold < %.2f)",
                len(missing_topics), threshold)
    return our_coverage, missing_topics


async def run_openalex(
    topic_result: TopicResult,
    output_dir: str = "./data/gap_analysis/openalex",
) -> OpenAlexResult:
    """Run OpenAlex comparison: fetch topics → compute coverage → save."""
    # Fetch OpenAlex topics
    global_topics = await fetch_openalex_topics()

    # Compute coverage
    our_coverage, missing_topics = compute_coverage(global_topics, topic_result)

    # Save artifacts
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
