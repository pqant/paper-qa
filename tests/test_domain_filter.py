"""Tests for EmbeddingDomainFilter — embedding-based domain relevance scoring."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from paperqa.stores.domain_filter import (
    EmbeddingDomainFilter,
    reset_domain_filter,
)


def _make_vector(dim: int = 4096, direction: list[float] | None = None) -> list[float]:
    """Create a normalized vector. If direction is given, use it (padded to dim)."""
    if direction:
        v = np.zeros(dim, dtype=np.float32)
        v[: len(direction)] = direction
    else:
        rng = np.random.default_rng(42)
        v = rng.standard_normal(dim).astype(np.float32)
    v = v / np.linalg.norm(v)
    return v.tolist()


@pytest.fixture
def filter_with_anchors():
    """Create a filter initialized with synthetic anchor vectors."""
    filt = EmbeddingDomainFilter(
        core_threshold=0.55,
        adjacent_threshold=0.47,
        cross_domain_threshold=0.39,
    )
    rng = np.random.default_rng(123)
    anchor_vecs = []
    for i in range(10):
        v = np.zeros(4096, dtype=np.float32)
        v[i * 10 : i * 10 + 10] = rng.standard_normal(10)
        v = v / np.linalg.norm(v)
        anchor_vecs.append(v.tolist())
    filt.initialize_from_vectors(anchor_vecs)
    return filt


class TestInitialization:
    def test_initialize_from_vectors(self, filter_with_anchors):
        assert filter_with_anchors.initialized is True

    def test_not_initialized_raises(self):
        filt = EmbeddingDomainFilter()
        with pytest.raises(RuntimeError, match="not initialized"):
            filt.score_embedding([0.0] * 4096)

    def test_double_init_from_vectors(self, filter_with_anchors):
        filter_with_anchors.initialize_from_vectors([[1.0] + [0.0] * 4095])
        assert filter_with_anchors.initialized


class TestScoring:
    def test_identical_vector_scores_one(self, filter_with_anchors):
        anchor = filter_with_anchors._anchor_matrix[0].tolist()
        score = filter_with_anchors.score_embedding(anchor)
        assert abs(score - 1.0) < 0.01

    def test_orthogonal_vector_scores_low(self, filter_with_anchors):
        v = np.zeros(4096, dtype=np.float32)
        v[3000:3010] = 1.0
        v = v / np.linalg.norm(v)
        score = filter_with_anchors.score_embedding(v.tolist())
        assert score < 0.3

    def test_zero_vector_scores_zero(self, filter_with_anchors):
        score = filter_with_anchors.score_embedding([0.0] * 4096)
        assert score == 0.0

    def test_score_in_range(self, filter_with_anchors):
        v = _make_vector(4096)
        score = filter_with_anchors.score_embedding(v)
        assert 0.0 <= score <= 1.0

    def test_batch_matches_single(self, filter_with_anchors):
        vecs = [_make_vector(4096, direction=[float(i)]) for i in range(1, 6)]
        batch = filter_with_anchors.score_embeddings_batch(vecs)
        singles = [filter_with_anchors.score_embedding(v) for v in vecs]
        np.testing.assert_allclose(batch, singles, atol=1e-3)


class TestClassification:
    def test_core_domain(self, filter_with_anchors):
        anchor = filter_with_anchors._anchor_matrix[0].tolist()
        cat = filter_with_anchors.classify_embedding(anchor)
        assert cat == "core_domain"

    def test_irrelevant(self, filter_with_anchors):
        v = np.zeros(4096, dtype=np.float32)
        v[3000:3010] = 1.0
        v = v / np.linalg.norm(v)
        cat = filter_with_anchors.classify_embedding(v.tolist())
        assert cat == "irrelevant"

    def test_classify_score_boundaries(self, filter_with_anchors):
        assert filter_with_anchors._classify_score(0.60) == "core_domain"
        assert filter_with_anchors._classify_score(0.50) == "adjacent"
        assert filter_with_anchors._classify_score(0.40) == "cross_domain"
        assert filter_with_anchors._classify_score(0.30) == "irrelevant"


class TestFilterChunks:
    def test_separates_relevant_from_irrelevant(self, filter_with_anchors):
        relevant_vec = filter_with_anchors._anchor_matrix[0].tolist()
        irrelevant_vec = np.zeros(4096, dtype=np.float32)
        irrelevant_vec[3000:3010] = 1.0
        irrelevant_vec = (irrelevant_vec / np.linalg.norm(irrelevant_vec)).tolist()

        chunks = [
            {"id": "relevant", "vector": relevant_vec},
            {"id": "irrelevant", "vector": irrelevant_vec},
        ]
        rel, cross, scores = filter_with_anchors.filter_chunks(chunks)

        rel_ids = [c["id"] for c in rel]
        assert "relevant" in rel_ids
        assert "irrelevant" not in rel_ids
        assert len(scores) == 2
        assert scores[0] > scores[1]

    def test_empty_input(self, filter_with_anchors):
        rel, cross, scores = filter_with_anchors.filter_chunks([])
        assert rel == []
        assert cross == []
        assert scores == []

    def test_missing_vector_key(self, filter_with_anchors):
        chunks = [{"id": "no_vec"}]
        rel, cross, scores = filter_with_anchors.filter_chunks(chunks)
        assert len(scores) == 1
        assert scores[0] == 0.0


class TestAsyncMethods:
    @pytest.mark.asyncio
    async def test_initialize_calls_embed_api(self):
        mock_embeddings = [np.random.randn(4096).tolist() for _ in range(10)]
        filt = EmbeddingDomainFilter()

        with patch.object(filt, "_embed_batch", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = mock_embeddings
            await filt.initialize()

        assert filt.initialized
        mock_embed.assert_called_once()

    @pytest.mark.asyncio
    async def test_score_text_embeds_and_scores(self, filter_with_anchors):
        anchor_vec = filter_with_anchors._anchor_matrix[0].tolist()

        with patch.object(
            filter_with_anchors, "_embed_batch", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = [anchor_vec]
            score = await filter_with_anchors.score_text("bin packing problem")

        assert score > 0.9
        mock_embed.assert_called_once()


class TestResetSingleton:
    def test_reset(self):
        reset_domain_filter()


class TestCalibration:
    @pytest.mark.asyncio
    async def test_calibrate_adjusts_thresholds(self, filter_with_anchors):
        pos_vecs = [filter_with_anchors._anchor_matrix[i].tolist() for i in range(3)]
        neg_vec = np.zeros(4096, dtype=np.float32)
        neg_vec[3000:3010] = 1.0
        neg_vecs = [(neg_vec / np.linalg.norm(neg_vec)).tolist()]

        with patch.object(
            filter_with_anchors, "_embed_batch", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.side_effect = [pos_vecs, neg_vecs]
            result = await filter_with_anchors.calibrate(
                positive_texts=["a", "b", "c"],
                negative_texts=["d"],
            )

        assert "core_threshold" in result
        assert "boundary" in result
        assert filter_with_anchors.core_threshold > filter_with_anchors.adjacent_threshold
        assert filter_with_anchors.adjacent_threshold > filter_with_anchors.cross_domain_threshold
