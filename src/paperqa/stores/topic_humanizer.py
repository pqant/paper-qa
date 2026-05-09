"""LLM-based topic humanization.

Converts BERTopic raw labels (e.g. "962_boxes_pallet_superboxes_trios")
into meaningful, actionable research gap descriptions.

Each topic is sent to the LLM with its keywords and sample chunk excerpts
so the LLM can infer what the cluster actually represents.

All LLM calls are sequential (max_concurrent_requests=1 constraint).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

def _get_domain_name() -> str:
    import os
    return os.environ.get("DOMAIN_NAME", "3D bin packing and container loading optimization")


_HUMANIZE_PROMPT = """You are a research domain expert in {domain_name}.

A topic clustering algorithm (BERTopic) found a topic cluster with these characteristics:

- Raw label: {topic_name}
- Top keywords: {keywords}
- Number of documents in cluster: {chunk_count}
- Sample text excerpts from papers in this cluster:
{sample_text}

Based ONLY on the keywords and sample excerpts above, generate:

1. **title**: A clear, specific research topic title (8-15 words). Must be specific enough that a researcher understands exactly what subfield this covers. Do NOT use generic phrases like "optimization methods" or "novel approaches".

2. **description**: One paragraph (2-4 sentences) explaining:
   - What this research area covers (based on the evidence above)
   - Why the small cluster size ({chunk_count} documents) might indicate a research gap
   - What aspect seems understudied based on the samples

Rules:
- Base your response ONLY on the provided keywords and sample excerpts
- Do NOT invent information not present in the input
- If the topic clearly does NOT relate to {domain_name}, set is_relevant to false
- Title should be in academic English

Return ONLY valid JSON (no markdown, no explanation):
{{"title": "...", "description": "...", "is_relevant": true/false}}"""


async def humanize_topic(
    topic_id: int,
    topic_words: list[str],
    topic_name: str,
    chunk_count: int,
    sample_chunks: list[str],
    llm_api_base: str = "http://192.168.0.28:8005/v1",
    llm_model: str = "qwen3.6-35b-a3b",
) -> dict[str, Any]:
    """Generate human-readable title and description for a BERTopic topic.

    Args:
        topic_id: BERTopic topic number.
        topic_words: Top representative words from BERTopic.
        topic_name: Raw BERTopic label (e.g. "962_boxes_pallet_superboxes_trios").
        chunk_count: Number of documents in this cluster.
        sample_chunks: 3-5 text excerpts from chunks in this cluster.
        llm_api_base: LLM server URL.
        llm_model: Model identifier.

    Returns:
        {"title": "...", "description": "...", "is_relevant": bool}
    """
    sample_text = "\n".join(
        f"  [{i+1}] {chunk[:250].strip()}" for i, chunk in enumerate(sample_chunks[:5])
    )
    if not sample_text.strip():
        sample_text = "  (no sample excerpts available)"

    prompt = _HUMANIZE_PROMPT.format(
        domain_name=_get_domain_name(),
        topic_name=topic_name,
        keywords=", ".join(topic_words[:12]),
        chunk_count=chunk_count,
        sample_text=sample_text,
    )

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{llm_api_base}/chat/completions",
                json={
                    "model": llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 4096,
                    "temperature": 0.1,
                },
            )
            response.raise_for_status()

            msg = response.json()["choices"][0]["message"]
            content = msg.get("content", "") or ""
            reasoning = msg.get("reasoning_content", "") or ""

            result = _extract_json(content) or _extract_json(reasoning)
            if result and "title" in result:
                return {
                    "title": result["title"],
                    "description": result.get("description", ""),
                    "is_relevant": result.get("is_relevant", True),
                }

    except Exception as e:
        logger.warning("Topic humanization failed for topic %d (%s): %s",
                        topic_id, topic_name[:40], e)

    return _fallback_humanize(topic_words, topic_name, chunk_count)


def _extract_json(text: str) -> dict | None:
    """Robustly extract JSON from LLM response that may contain markdown fences or thinking."""
    if not text or not text.strip():
        return None

    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    start = cleaned.rfind('{"title"')
    if start == -1:
        start = cleaned.rfind("{")
    if start == -1:
        return None

    end = cleaned.rfind("}")
    if end == -1 or end <= start:
        return None

    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        pass

    for i in range(end, start, -1):
        try:
            return json.loads(cleaned[start:i + 1])
        except json.JSONDecodeError:
            continue
    return None


def _fallback_humanize(
    topic_words: list[str],
    topic_name: str,
    chunk_count: int,
) -> dict[str, Any]:
    """Deterministic fallback when LLM is unavailable.

    Cleans BERTopic label into a readable format without LLM.
    """
    # Remove numeric prefix (e.g. "962_" from "962_boxes_pallet_superboxes_trios")
    parts = topic_name.split("_", 1)
    if len(parts) > 1 and parts[0].isdigit():
        core = parts[1]
    else:
        core = topic_name

    # Filter meaningful words
    clean_words = [
        w.strip() for w in core.split("_")
        if len(w.strip()) >= 3 and w.strip().isalpha()
    ]

    if not clean_words:
        clean_words = [w for w in topic_words if len(w) >= 3 and w.isalpha()][:6]

    title = " ".join(w.capitalize() for w in clean_words[:6])
    if not title:
        title = f"Topic Cluster {topic_name}"

    return {
        "title": title,
        "description": (
            f"Topic cluster with {chunk_count} documents. "
            f"Keywords: {', '.join(topic_words[:8])}. "
            f"LLM humanization unavailable — showing keyword-based label."
        ),
        "is_relevant": True,
    }


async def humanize_topics_batch(
    topics_data: list[dict[str, Any]],
    llm_api_base: str = "http://192.168.0.28:8005/v1",
    llm_model: str = "qwen3.6-35b-a3b",
) -> dict[str, dict[str, Any]]:
    """Humanize multiple topics sequentially.

    Args:
        topics_data: List of dicts, each with:
            - key (str) — unique identifier for result mapping
            - topic_id (int) — BERTopic topic id (for logging)
            - words (list[str])
            - name (str) — raw BERTopic label
            - chunk_count (int)
            - sample_chunks (list[str]) — text excerpts

    Returns:
        {key: {"title": ..., "description": ..., "is_relevant": ...}}
    """
    results: dict[str, dict[str, Any]] = {}

    for i, topic in enumerate(topics_data):
        key = topic.get("key", str(topic.get("topic_id", i)))
        tid = topic.get("topic_id", i)
        logger.info(
            "Humanizing topic %d/%d (key=%s): %s",
            i + 1, len(topics_data), key, topic.get("name", "")[:50],
        )
        result = await humanize_topic(
            topic_id=tid,
            topic_words=topic["words"],
            topic_name=topic["name"],
            chunk_count=topic["chunk_count"],
            sample_chunks=topic.get("sample_chunks", []),
            llm_api_base=llm_api_base,
            llm_model=llm_model,
        )
        results[key] = result

    logger.info("Humanized %d topics", len(results))
    return results
