"""BERTopic pipeline — discover topics from domain-filtered Qdrant corpus.

Flow:
  1. Initialize EmbeddingDomainFilter (embed 10 anchor queries once)
  2. Scroll Qdrant in batches WITH vectors
  3. Score each batch with cosine similarity — keep only passing chunks
  4. PCA 4096→256 → UMAP 256→10 → HDBSCAN → topic labels
  5. Save all artifacts + filtering metadata to disk
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import AsyncQdrantClient

from paperqa.stores.domain_filter import EmbeddingDomainFilter
from paperqa.stores.gap_analysis import TopicResult
from paperqa.stores.paperbridge_store import parse_pdf_name

logger = logging.getLogger(__name__)


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars for json.dump."""
    if isinstance(obj, dict):
        out: dict[Any, Any] = {}
        for k, v in obj.items():
            if isinstance(k, np.integer):
                k = int(k)
            elif isinstance(k, np.floating):
                k = float(k)
            out[k] = _json_safe(v)
        return out
    if isinstance(obj, list | tuple):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


@dataclass
class FilterStats:
    """Statistics from the domain filtering stage."""

    total_scrolled: int = 0
    total_passed: int = 0
    total_rejected: int = 0
    core_domain: int = 0
    adjacent: int = 0
    cross_domain: int = 0
    irrelevant: int = 0
    score_min: float = 1.0
    score_max: float = 0.0
    score_mean: float = 0.0
    elapsed_seconds: float = 0.0
    thresholds: dict[str, float] = field(default_factory=dict)
    # When PIPELINE_MAX_SCROLL_POINTS is set, scroll stops early (partial corpus sample).
    scroll_capped: bool = False
    scroll_cap_limit: int | None = None


async def extract_filtered_chunks(
    domain_filter: EmbeddingDomainFilter,
    collection_name: str = "paperbridge_glm_v2",
    qdrant_url: str = "http://192.168.0.28:6333",
    batch_size: int = 500,
    min_relevance: float | None = None,
    output_dir: str | None = None,
    progress_callback: Any = None,
) -> tuple[list[str], list[np.ndarray], list[int], list[float], list[str], FilterStats]:
    """Scroll Qdrant and filter chunks by domain relevance in real-time.

    Instead of loading all chunks into memory, each batch is scored
    immediately and only passing chunks are retained.

    Args:
        domain_filter: Initialized EmbeddingDomainFilter.
        collection_name: Qdrant collection.
        qdrant_url: Qdrant server URL.
        batch_size: Chunks per scroll request.
        min_relevance: Score threshold (defaults to filter's adjacent_threshold).
        output_dir: If set, persist filtering metadata for checkpoint/resume.
        progress_callback: Optional callable(scrolled, passed, total_estimate)
                           for real-time UI progress updates.

    Returns:
        (texts, embeddings, years, scores, pdf_names, filter_stats)
        Only chunks above min_relevance are included.
    """
    t0 = time.time()
    threshold = min_relevance if min_relevance is not None else domain_filter.adjacent_threshold

    client = AsyncQdrantClient(url=qdrant_url)

    # Get total count for progress estimation
    collection_info = await client.get_collection(collection_name)
    total_estimate = collection_info.points_count or 0

    texts: list[str] = []
    embeddings: list[np.ndarray] = []
    years: list[int] = []
    scores: list[float] = []
    pdf_names: list[str] = []

    stats = FilterStats(thresholds={
        "core": domain_filter.core_threshold,
        "adjacent": domain_filter.adjacent_threshold,
        "cross_domain": domain_filter.cross_domain_threshold,
        "applied": threshold,
    })

    offset = None
    seen_offsets: set[str] = set()
    all_scores_for_stats: list[float] = []

    _cap_raw = os.getenv("PIPELINE_MAX_SCROLL_POINTS", "").strip()
    max_scroll_points: int | None = None
    if _cap_raw.isdigit():
        max_scroll_points = int(_cap_raw)
        logger.info(
            "PIPELINE_MAX_SCROLL_POINTS=%d — scroll will stop after this many raw points "
            "(partial sample; unset env for full corpus)",
            max_scroll_points,
        )

    while True:
        points, next_offset = await client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=["pdf_name", "text", "page_num"],
            with_vectors=True,
        )
        if not points:
            break

        batch_vecs: list[np.ndarray] = []
        batch_texts: list[str] = []
        batch_years: list[int] = []
        batch_pdfs: list[str] = []

        for point in points:
            payload = point.payload
            text = payload.get("text", "")
            if not text.strip():
                continue

            vec = np.array(point.vector, dtype=np.float32)
            meta = parse_pdf_name(payload.get("pdf_name", ""))

            batch_vecs.append(vec)
            batch_texts.append(text)
            batch_years.append(meta.get("year") or 2000)
            batch_pdfs.append(payload.get("pdf_name", ""))

        if batch_vecs:
            batch_scores = domain_filter.score_embeddings_batch(batch_vecs)

            for i, s in enumerate(batch_scores):
                s_val = float(s)
                all_scores_for_stats.append(s_val)

                cat = domain_filter._classify_score(s_val)
                if cat == "core_domain":
                    stats.core_domain += 1
                elif cat == "adjacent":
                    stats.adjacent += 1
                elif cat == "cross_domain":
                    stats.cross_domain += 1
                else:
                    stats.irrelevant += 1

                if s_val >= threshold:
                    texts.append(batch_texts[i])
                    embeddings.append(batch_vecs[i])
                    years.append(batch_years[i])
                    scores.append(s_val)
                    pdf_names.append(batch_pdfs[i])

        stats.total_scrolled += len(points)

        if max_scroll_points is not None and stats.total_scrolled >= max_scroll_points:
            stats.scroll_capped = True
            stats.scroll_cap_limit = max_scroll_points
            logger.info(
                "Scroll cap reached (%d raw points); continuing pipeline on %d filtered chunks",
                max_scroll_points,
                len(texts),
            )
            break

        if stats.total_scrolled % 5000 == 0:
            logger.info(
                "Scrolled %d / ~%d — %d passed (%.1f%%)",
                stats.total_scrolled,
                total_estimate,
                len(texts),
                100 * len(texts) / max(1, stats.total_scrolled),
            )

        if progress_callback:
            progress_callback(stats.total_scrolled, len(texts), total_estimate)

        if next_offset is None:
            break

        offset_str = str(next_offset)
        if offset_str in seen_offsets:
            break
        seen_offsets.add(offset_str)
        offset = next_offset

    stats.total_passed = len(texts)
    stats.total_rejected = stats.total_scrolled - stats.total_passed
    stats.elapsed_seconds = round(time.time() - t0, 1)

    if all_scores_for_stats:
        arr = np.array(all_scores_for_stats)
        stats.score_min = round(float(np.min(arr)), 4)
        stats.score_max = round(float(np.max(arr)), 4)
        stats.score_mean = round(float(np.mean(arr)), 4)

    logger.info(
        "Domain filtering complete: %d / %d chunks passed (%.1f%%) in %.0fs",
        stats.total_passed,
        stats.total_scrolled,
        100 * stats.total_passed / max(1, stats.total_scrolled),
        stats.elapsed_seconds,
    )

    await client.close()

    if output_dir:
        _save_filter_stats(stats, Path(output_dir))

    return texts, embeddings, years, scores, pdf_names, stats


def fit_bertopic(
    texts: list[str],
    embeddings: list[np.ndarray],
    years: list[int],
    output_dir: str = "./data/gap_analysis/bertopic",
) -> TopicResult:
    """Fit BERTopic on pre-filtered, pre-computed embeddings.

    Pipeline: PCA 4096→256 → UMAP 256→10 → HDBSCAN → ClassTF-IDF labels.
    """
    from bertopic import BERTopic
    from sklearn.decomposition import PCA
    from umap import UMAP
    import hdbscan
    import pandas as pd

    embeddings_array = np.array(embeddings, dtype=np.float32)
    years_array = np.array(years, dtype=np.int32)

    logger.info("Running PCA 4096→256 on %d chunks...", len(texts))
    pca = PCA(n_components=256, random_state=42)
    reduced = pca.fit_transform(embeddings_array)

    logger.info("Running UMAP 256→10...")
    umap_model = UMAP(
        n_components=10,
        metric="cosine",
        random_state=42,
        min_dist=0.0,
        n_neighbors=15,
    )
    embeddings_reduced = umap_model.fit_transform(reduced)

    logger.info("Running HDBSCAN clustering...")
    cluster_model = hdbscan.HDBSCAN(
        min_cluster_size=20,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    logger.info("Fitting BERTopic...")
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        calculate_probabilities=False,
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(texts, embeddings=embeddings_reduced)

    topic_info_df = topic_model.get_topic_info()

    topics_array = np.array(topics)
    valid_mask = topics_array != -1
    num_topics = len(topic_model.get_topics())
    logger.info(
        "BERTopic found %d topics (%d outliers)",
        num_topics,
        int(np.sum(~valid_mask)),
    )

    # Temporal distribution
    valid_topics = topics_array[valid_mask]
    valid_years = years_array[valid_mask]
    temporal_data: dict[int, dict] = {}
    for t, y in zip(valid_topics, valid_years):
        t_id = int(t)
        y_id = int(y)
        if t_id not in temporal_data:
            temporal_data[t_id] = {}
        temporal_data[t_id][y_id] = temporal_data[t_id].get(y_id, 0) + 1

    sparse_topics = [
        {"topic_id": int(row["Topic"]), "count": int(row["Count"]), "name": str(row["Name"])}
        for _, row in topic_info_df.iterrows()
        if row["Topic"] != -1 and row["Count"] < 50
    ]

    topic_dict: dict[int, list[str]] = {}
    for row in topic_info_df.iterrows():
        if row[1]["Topic"] == -1:
            continue
        tid = int(row[1]["Topic"])
        words = topic_model.get_topic(tid)
        if words:
            word_list = []
            for w in words:
                if isinstance(w, (tuple, list)):
                    word_list.append(str(w[0]))
                else:
                    word_list.append(str(w))
            topic_dict[tid] = word_list
        else:
            topic_dict[tid] = []

    topic_info_dict = {
        "columns": list(topic_info_df.columns),
        "data": topic_info_df.to_dict(orient="records"),
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    topic_model.save(str(output_path / "bertopic_model"), serialization="pytorch")

    result = TopicResult(
        topics=topic_dict,
        topic_info=topic_info_dict,
        chunk_to_topic=list(topics),
        sparse_topics=sparse_topics,
        temporal_data=temporal_data,
    )

    metadata = {
        "num_chunks": len(texts),
        "num_topics": num_topics,
        "num_outliers": int(sum(~valid_mask)),
        "sparse_topics": len(sparse_topics),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    topic_info_df.to_csv(output_path / "topic_info.csv", index=False)

    topic_result_data = _json_safe({
        "topics": topic_dict,
        "chunk_to_topic": list(topics),
        "sparse_topics": sparse_topics,
        "topic_info": topic_info_dict,
        "temporal_data": temporal_data,
    })
    with open(output_path / "topic_result.json", "w") as f:
        json.dump(topic_result_data, f)

    logger.info("Saved BERTopic artifacts to %s", output_path)
    return result


async def run_bertopic(
    qdrant_collection: str = "paperbridge_glm_v2",
    qdrant_url: str = "http://192.168.0.28:6333",
    output_dir: str = "./data/gap_analysis/bertopic",
    domain_filter: EmbeddingDomainFilter | None = None,
    min_relevance: float | None = None,
    progress_callback: Any = None,
) -> tuple[TopicResult, FilterStats, list[float]]:
    """Run the full pipeline: init filter → extract+filter → BERTopic → save.

    Args:
        qdrant_collection: Qdrant collection name.
        qdrant_url: Qdrant server URL.
        output_dir: Directory for artifacts.
        domain_filter: Pre-initialized filter. If None, creates and initializes one.
        min_relevance: Override threshold for filtering.
        progress_callback: For UI progress updates.

    Returns:
        (TopicResult, FilterStats, chunk_scores) — FilterStats is None only if filter was skipped.
        chunk_scores: per-chunk domain relevance scores (same order as chunk_to_topic).
    """
    if domain_filter is None:
        domain_filter = EmbeddingDomainFilter()
        await domain_filter.initialize()

    texts, embeddings, years, scores, pdf_names, filter_stats = await extract_filtered_chunks(
        domain_filter=domain_filter,
        collection_name=qdrant_collection,
        qdrant_url=qdrant_url,
        output_dir=output_dir,
        min_relevance=min_relevance,
        progress_callback=progress_callback,
    )

    topic_result = fit_bertopic(texts, embeddings, years, output_dir)
    return topic_result, filter_stats, scores


# ── Persistence helpers ──────────────────────────────────────────────────────


def load_topic_result(output_dir: str = "./data/gap_analysis/bertopic") -> TopicResult:
    """Load a previously saved TopicResult from disk."""
    path = Path(output_dir) / "topic_result.json"
    with open(path) as f:
        data = json.load(f)
    topics_int_keys = {int(k): v for k, v in data["topics"].items()}
    return TopicResult(
        topics=topics_int_keys,
        chunk_to_topic=data["chunk_to_topic"],
        sparse_topics=data.get("sparse_topics", []),
        topic_info=data.get("topic_info", {}),
        temporal_data=data.get("temporal_data"),
    )


def _save_filter_stats(stats: FilterStats, output_dir: Path) -> None:
    """Save filtering statistics for checkpoint/resume."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "filter_stats.json", "w") as f:
        json.dump(asdict(stats), f, indent=2)
    logger.info("Saved filter stats to %s/filter_stats.json", output_dir)
