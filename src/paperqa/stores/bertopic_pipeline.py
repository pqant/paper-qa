"""BERTopic pipeline — discover topics from Qdrant corpus.

Uses pre-computed 4096-dim embeddings from Qdrant.
PCA 4096→256 → UMAP 256→10 → HDBSCAN → topic labels.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from qdrant_client import AsyncQdrantClient

from paperqa.stores.gap_analysis import TopicResult
from paperqa.stores.paperbridge_store import parse_pdf_name

logger = logging.getLogger(__name__)


async def extract_chunks_from_qdrant(
    collection_name: str = "paperbridge_glm_v2",
    qdrant_url: str = "http://192.168.0.28:6333",
    batch_size: int = 500,
) -> tuple[list[str], list[np.ndarray], list[int]]:
    """Scroll all chunks from Qdrant, return texts, embeddings, years.

    Returns:
        (texts, embeddings, years) where each index corresponds to the same chunk.
    """
    client = AsyncQdrantClient(url=qdrant_url)

    all_texts: list[str] = []
    all_embeddings: list[np.ndarray] = []
    all_years: list[int] = []

    offset = None
    seen_offsets: set[str] = set()
    scrolled = 0

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

        for point in points:
            payload = point.payload
            text = payload.get("text", "")
            if not text.strip():
                continue

            all_texts.append(text)
            all_embeddings.append(np.array(point.vector, dtype=np.float32))

            meta = parse_pdf_name(payload.get("pdf_name", ""))
            all_years.append(meta.get("year") or 2000)

        scrolled += len(points)
        if scrolled % 2000 == 0:
            logger.info("Scrolled %d points (%d valid chunks)", scrolled, len(all_texts))

        if next_offset is None:
            break

        offset_str = str(next_offset)
        if offset_str in seen_offsets:
            break
        seen_offsets.add(offset_str)
        offset = next_offset

    logger.info("Extracted %d chunks from Qdrant", len(all_texts))
    return all_texts, all_embeddings, all_years


def fit_bertopic(
    texts: list[str],
    embeddings: list[np.ndarray],
    years: list[int],
    output_dir: str = "./data/gap_analysis/bertopic",
) -> TopicResult:
    """Fit BERTopic on pre-computed embeddings.

    Pipeline: PCA 4096→256 → UMAP 256→10 → HDBSCAN → ClassTF-IDF labels.
    """
    from bertopic import BERTopic
    from sklearn.decomposition import PCA
    from umap import UMAP
    import hdbscan
    import pandas as pd

    embeddings_array = np.array(embeddings, dtype=np.float32)
    years_array = np.array(years, dtype=np.int32)

    # PCA dimensionality reduction (memory optimization)
    logger.info("Running PCA 4096→256...")
    pca = PCA(n_components=256, random_state=42)
    reduced = pca.fit_transform(embeddings_array)

    # UMAP dimensionality reduction
    logger.info("Running UMAP 256→10...")
    umap_model = UMAP(
        n_components=10,
        metric="cosine",
        random_state=42,
        min_dist=0.0,
        n_neighbors=15,
    )
    embeddings_reduced = umap_model.fit_transform(reduced)

    # HDBSCAN clustering — euclidean on UMAP-reduced space
    # (UMAP already handled cosine; euclidean on 10-dim output is standard)
    logger.info("Running HDBSCAN clustering...")
    cluster_model = hdbscan.HDBSCAN(
        min_cluster_size=20,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # Fit BERTopic with pre-computed reduced embeddings
    logger.info("Fitting BERTopic...")
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        calculate_probabilities=False,
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(texts, embeddings=embeddings_reduced)

    # Get topic info
    topic_info_df = topic_model.get_topic_info()

    # Filter out -1 (outliers) for analysis
    topics_array = np.array(topics)
    valid_mask = topics_array != -1
    num_topics = len(topic_model.get_topics())
    logger.info("BERTopic found %d topics (%d outliers)",
                num_topics, int(np.sum(~valid_mask)))

    # Build temporal data
    valid_topics = topics_array[valid_mask]
    valid_years = years_array[valid_mask]
    temporal_data: dict[int, dict] = {}
    for t, y in zip(valid_topics, valid_years):
        if t not in temporal_data:
            temporal_data[t] = {}
        temporal_data[t][int(y)] = temporal_data[t].get(int(y), 0) + 1

    # Identify sparse topics (<50 docs)
    sparse_topics = [
        {"topic_id": row["Topic"], "count": row["Count"], "name": row["Name"]}
        for _, row in topic_info_df.iterrows()
        if row["Topic"] != -1 and row["Count"] < 50
    ]

    # Build topic dict — BERTopic 0.17+ uses get_topic(topic_id)
    # Normalize to plain string lists for downstream compatibility
    topic_dict: dict[int, list[str]] = {}
    for row in topic_info_df.iterrows():
        if row[1]["Topic"] == -1:
            continue
        tid = int(row[1]["Topic"])
        words = topic_model.get_topic(tid)
        if words:
            # get_topic() returns list of strings in BERTopic 0.17+
            topic_dict[tid] = [str(w) for w in words]
        else:
            topic_dict[tid] = []

    # Convert topic_info to dict
    topic_info_dict = {
        "columns": list(topic_info_df.columns),
        "data": topic_info_df.to_dict(orient="records"),
    }

    # Save artifacts
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    topic_model.save(str(output_path / "bertopic_model"), serialization="pytorch")

    # Save metadata
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

    # Save topic info as CSV
    topic_info_df.to_csv(output_path / "topic_info.csv", index=False)

    logger.info("Saved BERTopic artifacts to %s", output_path)
    return result


async def run_bertopic(
    qdrant_collection: str = "paperbridge_glm_v2",
    output_dir: str = "./data/gap_analysis/bertopic",
) -> TopicResult:
    """Run the full BERTopic pipeline: extract → fit → save."""
    texts, embeddings, years = await extract_chunks_from_qdrant(
        collection_name=qdrant_collection,
    )
    return fit_bertopic(texts, embeddings, years, output_dir)
