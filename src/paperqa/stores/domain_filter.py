"""Embedding-based domain relevance filter.

Uses cosine similarity between chunk embeddings and domain anchor embeddings.
No hardcoded keyword lists — matching is purely semantic via the embedding model.

Domain anchors are natural-language descriptions of the target research area.
They get embedded once at initialization and cached for the session lifetime.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_DOMAIN_ANCHORS: list[str] = [
    "three-dimensional bin packing problem with optimization algorithms and heuristics",
    "container loading and cargo arrangement for space utilization in logistics",
    "cutting stock problem and strip packing with guillotine or non-guillotine cuts",
    "knapsack problem variants including multi-dimensional and bounded knapsack",
    "palletization and pallet loading optimization in warehouse operations",
    "vehicle and truck cargo loading with sequence and stability constraints",
    "rectangle and irregular shape nesting and packing for manufacturing",
    "packing heuristics such as best-fit first-fit bottom-left skyline algorithms",
    "box and carton packing with rotation orientation and weight distribution",
    "orthogonal packing placement strategies and layout optimization",
]


def get_domain_anchors() -> list[str]:
    """Return domain anchors from env or defaults.

    Set DOMAIN_ANCHORS as a pipe-separated string to override, e.g.:
    DOMAIN_ANCHORS="deep learning for NLP|transformer architectures|..."
    """
    env = os.getenv("DOMAIN_ANCHORS")
    if env:
        return [a.strip() for a in env.split("|") if a.strip()]
    return _DEFAULT_DOMAIN_ANCHORS


class EmbeddingDomainFilter:
    """Semantic domain relevance filter powered by embedding cosine similarity.

    Two modes of operation:
        1. score_embedding(vector) — sync, pure math, zero I/O.
           Use when chunk embeddings are already available (e.g. from Qdrant).
        2. score_text(text) — async, calls embedding API.
           Use for small volumes (topic labels, gap sentences).

    Thresholds are auto-calibrated via `calibrate()` or set manually.
    """

    def __init__(
        self,
        embedding_api_base: str | None = None,
        embedding_model: str | None = None,
        core_threshold: float | None = None,
        adjacent_threshold: float | None = None,
        cross_domain_threshold: float | None = None,
    ):
        self._api_base = embedding_api_base or os.getenv(
            "EMBEDDING_API_BASE", "http://192.168.0.28:8082/v1"
        )
        self._model = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "hf.co/Qwen/Qwen3-Embedding-8B-GGUF:F16"
        )
        self.core_threshold = core_threshold or float(
            os.getenv("DOMAIN_CORE_THRESHOLD", "0.55")
        )
        self.adjacent_threshold = adjacent_threshold or float(
            os.getenv("DOMAIN_ADJACENT_THRESHOLD", "0.47")
        )
        self.cross_domain_threshold = cross_domain_threshold or float(
            os.getenv("DOMAIN_CROSS_DOMAIN_THRESHOLD", "0.39")
        )

        self._anchor_matrix: np.ndarray | None = None
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    # ── Initialization ──────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Embed domain anchors and cache the normalized matrix."""
        if self._initialized:
            return

        anchors = get_domain_anchors()
        logger.info(
            "Initializing EmbeddingDomainFilter — embedding %d anchors via %s ...",
            len(anchors),
            self._api_base,
        )
        raw = await self._embed_batch(anchors)
        mat = np.array(raw, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._anchor_matrix = mat / norms
        self._initialized = True
        logger.info(
            "EmbeddingDomainFilter ready (anchor matrix %s)", self._anchor_matrix.shape
        )

    def initialize_from_vectors(self, anchor_vectors: list[list[float]]) -> None:
        """Initialize with pre-computed anchor vectors (for testing / caching)."""
        mat = np.array(anchor_vectors, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._anchor_matrix = mat / norms
        self._initialized = True

    # ── Embedding API ───────────────────────────────────────────────────────

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via the OpenAI-compatible embeddings API."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._api_base}/embeddings",
                json={"input": texts, "model": self._model},
            )
            resp.raise_for_status()
            data = resp.json()
            items = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in items]

    async def embed_single(self, text: str) -> list[float]:
        """Embed a single text. Useful for callers that need the raw vector."""
        result = await self._embed_batch([text])
        return result[0]

    # ── Scoring ─────────────────────────────────────────────────────────────

    def _ensure_ready(self) -> None:
        if not self._initialized or self._anchor_matrix is None:
            raise RuntimeError(
                "DomainFilter not initialized. Call `await filter.initialize()` first."
            )

    def score_embedding(self, embedding: list[float] | np.ndarray) -> float:
        """Domain relevance score from a pre-existing embedding vector.

        Hot path — called per chunk. Pure numpy math, zero I/O.
        Returns max cosine similarity across all domain anchors (0.0 – 1.0).
        """
        self._ensure_ready()
        vec = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return 0.0
        vec = vec / norm
        similarities = self._anchor_matrix @ vec
        return round(float(np.max(similarities)), 4)

    def score_embeddings_batch(
        self, embeddings: list[list[float]] | np.ndarray
    ) -> np.ndarray:
        """Vectorized scoring for many embeddings at once.

        Returns array of scores, one per input embedding.
        Much faster than calling score_embedding() in a loop.
        """
        self._ensure_ready()
        mat = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms
        # (N, D) @ (A, D).T → (N, A) → max over anchors → (N,)
        sims = mat @ self._anchor_matrix.T
        return np.round(np.max(sims, axis=1), 4)

    async def score_text(self, text: str) -> float:
        """Domain relevance score from raw text (embeds on-the-fly)."""
        self._ensure_ready()
        raw = await self._embed_batch([text])
        return self.score_embedding(raw[0])

    # ── Classification ──────────────────────────────────────────────────────

    def classify_embedding(self, embedding: list[float] | np.ndarray) -> str:
        """Classify a chunk's domain category from its embedding.

        Returns: "core_domain" | "adjacent" | "cross_domain" | "irrelevant"
        """
        score = self.score_embedding(embedding)
        return self._classify_score(score)

    async def classify_text(self, text: str) -> str:
        """Classify domain category from raw text."""
        score = await self.score_text(text)
        return self._classify_score(score)

    def _classify_score(self, score: float) -> str:
        if score >= self.core_threshold:
            return "core_domain"
        if score >= self.adjacent_threshold:
            return "adjacent"
        if score >= self.cross_domain_threshold:
            return "cross_domain"
        return "irrelevant"

    # ── Bulk filtering ──────────────────────────────────────────────────────

    def filter_chunks(
        self,
        chunks: list[dict[str, Any]],
        embedding_key: str = "vector",
        min_relevance: float | None = None,
    ) -> tuple[list[dict], list[dict], list[float]]:
        """Partition chunks into relevant / cross-domain / irrelevant.

        Args:
            chunks: Each dict must contain the embedding under `embedding_key`.
            embedding_key: Key for the embedding vector in each chunk dict.
            min_relevance: Score threshold for "relevant" (defaults to adjacent_threshold).

        Returns:
            (relevant, cross_domain, scores) where scores[i] corresponds to chunks[i].
        """
        threshold = min_relevance if min_relevance is not None else self.adjacent_threshold

        vecs = [ch[embedding_key] for ch in chunks if ch.get(embedding_key) is not None]
        if not vecs:
            return [], [], [0.0] * len(chunks)

        all_scores = self.score_embeddings_batch(vecs)

        relevant: list[dict] = []
        cross_domain: list[dict] = []
        score_list: list[float] = []
        vi = 0

        for chunk in chunks:
            if chunk.get(embedding_key) is None:
                score_list.append(0.0)
                continue
            s = float(all_scores[vi])
            vi += 1
            score_list.append(s)

            if s >= threshold:
                relevant.append(chunk)
            elif s >= self.cross_domain_threshold:
                cross_domain.append(chunk)

        logger.info(
            "Domain filter: %d relevant (≥%.2f), %d cross-domain, %d irrelevant (of %d)",
            len(relevant),
            threshold,
            len(cross_domain),
            len(chunks) - len(relevant) - len(cross_domain),
            len(chunks),
        )
        return relevant, cross_domain, score_list

    # ── Calibration ─────────────────────────────────────────────────────────

    async def calibrate(
        self,
        positive_texts: list[str],
        negative_texts: list[str],
        margin: float = 0.05,
    ) -> dict[str, float]:
        """Auto-calibrate thresholds from labeled examples.

        Embeds positive (domain-relevant) and negative (irrelevant) texts,
        computes their score distributions, and sets thresholds to maximize
        separation.

        Args:
            positive_texts: Texts known to be domain-relevant.
            negative_texts: Texts known to be irrelevant.
            margin: Safety margin added/subtracted from the boundary.

        Returns:
            Dict with the calibrated threshold values.
        """
        self._ensure_ready()

        pos_vecs = await self._embed_batch(positive_texts)
        neg_vecs = await self._embed_batch(negative_texts)

        pos_scores = self.score_embeddings_batch(pos_vecs)
        neg_scores = self.score_embeddings_batch(neg_vecs)

        pos_min = float(np.min(pos_scores))
        neg_max = float(np.max(neg_scores))

        boundary = (pos_min + neg_max) / 2.0

        self.core_threshold = round(boundary + margin * 2, 3)
        self.adjacent_threshold = round(boundary, 3)
        self.cross_domain_threshold = round(boundary - margin * 2, 3)

        result = {
            "core_threshold": self.core_threshold,
            "adjacent_threshold": self.adjacent_threshold,
            "cross_domain_threshold": self.cross_domain_threshold,
            "positive_score_range": f"{float(np.min(pos_scores)):.4f} – {float(np.max(pos_scores)):.4f}",
            "negative_score_range": f"{float(np.min(neg_scores)):.4f} – {float(np.max(neg_scores)):.4f}",
            "boundary": round(boundary, 4),
        }

        logger.info("Calibrated thresholds: %s", result)
        return result


# ── Module-level convenience ────────────────────────────────────────────────

_default_filter: EmbeddingDomainFilter | None = None


async def get_domain_filter(**kwargs: Any) -> EmbeddingDomainFilter:
    """Get or create the module-level singleton filter."""
    global _default_filter
    if _default_filter is None:
        _default_filter = EmbeddingDomainFilter(**kwargs)
        await _default_filter.initialize()
    return _default_filter


def reset_domain_filter() -> None:
    """Reset the singleton (useful for tests)."""
    global _default_filter
    _default_filter = None
