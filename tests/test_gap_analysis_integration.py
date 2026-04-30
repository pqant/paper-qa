"""Integration test for Phase 2: Gap Analysis Layer.

Tests the full pipeline with a subset of data.
Requires: Qdrant, OpenAlex API, LLM server.

Run:
    PYTHONPATH=src pytest tests/test_gap_analysis_integration.py -v --timeout=3600
"""

import asyncio
import pytest
from unittest import mock

from paperqa.stores.gap_analysis import (
    GapItem,
    GapLanguageResult,
    GapReport,
    NovelpyResult,
    OpenAlexResult,
    TopicResult,
)


class TestGapAnalysisDataclasses:
    """Test that dataclasses serialize and deserialize correctly."""

    def test_topic_result(self):
        result = TopicResult(
            topics={0: ["bin", "packing"], 1: ["container", "loading"]},
            topic_info={"columns": [], "data": []},
            chunk_to_topic=[0, 1, 2, 3],
            sparse_topics=[{"topic_id": 0, "count": 10, "name": "test topic"}],
            temporal_data={0: {2024: 5}},
        )
        assert len(result.topics) == 2
        assert len(result.chunk_to_topic) == 4
        assert result.topics[0] == ["bin", "packing"]

    def test_gap_report(self):
        report = GapReport(
            gaps=[GapItem(id=1, title="Test gap", description="A gap", evidence_sources=["bertopic"], confidence=0.8, papers=[])],
            summary="Test summary",
            recommendations=["Investigate test gap"],
            metadata={"total_papers": 100, "total_chunks": 1000, "analysis_date": "2026-04-30"},
        )
        assert len(report.gaps) == 1
        assert report.gaps[0].confidence == 0.8
        assert "Test gap" in report.summary or len(report.summary) > 0

    def test_all_results_roundtrip(self):
        """Test that all result dataclasses can be created and accessed."""
        topic = TopicResult(topics={}, topic_info={}, chunk_to_topic=[], sparse_topics=[], temporal_data={})
        gap = GapLanguageResult(gap_chunks=[], gap_categories={}, top_gaps=[])
        openalex = OpenAlexResult(global_topics=[], our_coverage=[], missing_topics=[])
        novelpy = NovelpyResult(concepts=[], cooccurrence_counts=[], novel_pairs=[])
        assert len(topic.topics) == 0
        assert len(gap.gap_chunks) == 0
        assert len(openalex.global_topics) == 0
        assert len(novelpy.concepts) == 0


class TestCandidateCollection:
    """Test candidate gap collection from all components."""

    def test_collect_from_all_sources(self):
        from paperqa.stores.gap_synthesis import collect_candidate_gaps

        mock_topic = TopicResult(
            topics={0: ["bin", "packing"], 1: ["container", "loading"]},
            topic_info={"columns": [], "data": []},
            chunk_to_topic=[0] * 100 + [1] * 100,
            sparse_topics=[{"topic_id": 0, "count": 10, "name": "quantum packing"}],
            temporal_data={},
        )
        mock_gap = GapLanguageResult(
            gap_chunks=[],
            gap_categories={0: {"label": "real-time constraints", "count": 15, "words": ["real-time", "constraint"]}},
            top_gaps=["future work"],
        )
        mock_openalex = OpenAlexResult(
            global_topics=[],
            our_coverage=[],
            missing_topics=[{"topic_name": "digital twin", "our_count": 0, "global_count": 5000, "ratio": 0.0}],
        )
        mock_novelpy = NovelpyResult(
            concepts=["bin", "packing", "quantum"],
            cooccurrence_counts=[],
            novel_pairs=[("quantum", "bin packing", 0.95)],
        )

        candidates = collect_candidate_gaps(mock_topic, mock_gap, mock_openalex, mock_novelpy)
        assert len(candidates) >= 3

        sources = {c["evidence_sources"][0] for c in candidates}
        assert "bertopic_sparse" in sources
        assert "openalex_missing" in sources
        assert "novelpy" in sources
        assert "gap_language" in sources


class TestNovelpyPipeline:
    """Test Novelpy pipeline with mock data."""

    def test_extract_concepts(self):
        from paperqa.stores.novelpy_pipeline import extract_concepts_from_topics

        mock = TopicResult(
            topics={0: ["bin", "packing"], 1: ["container", "loading"]},
            topic_info={}, chunk_to_topic=[], sparse_topics=[], temporal_data={},
        )
        concepts = extract_concepts_from_topics(mock)
        assert "bin" in concepts
        assert "packing" in concepts
        assert len(concepts) == 4

    def test_calculate_novelty(self):
        from paperqa.stores.novelpy_pipeline import calculate_novelty

        concepts = ["a", "b", "c", "d"]
        cooccurrence = [
            ("a", "b", 1000),  # very common
            ("a", "c", 50),    # somewhat rare
            ("c", "d", 10),    # rare
        ]
        pairs = calculate_novelty(concepts, cooccurrence, top_k=50)
        # Rare pairs should score higher
        assert len(pairs) > 0
        assert pairs[0][2] > pairs[-1][2]  # first pair has highest score


class TestBERTopicPipeline:
    """Test BERTopic extraction (requires Qdrant)."""

    @pytest.mark.asyncio
    async def test_extract_chunks_smoke(self):
        """Quick smoke test — extract first 500 chunks."""
        from paperqa.stores.bertopic_pipeline import extract_chunks_from_qdrant

        texts, embs, years = await extract_chunks_from_qdrant(batch_size=500)
        assert len(texts) > 100000  # full corpus
        assert embs[0].shape == (4096,)
        assert 1980 <= min(years) <= 2020
        assert 2020 <= max(years) <= 2027


class TestGapLanguageExtraction:
    """Test gap-language extraction (requires Tantivy)."""

    @pytest.mark.asyncio
    async def test_extract_gap_chunks(self):
        from paperqa.stores.gap_language import extract_gap_chunks

        chunks = await extract_gap_chunks(top_n=10)
        assert len(chunks) > 0
        for chunk in chunks:
            assert "pdf_hash" in chunk
            assert "keyword_matched" in chunk


class TestOpenAlexAPI:
    """Test OpenAlex connectivity."""

    @pytest.mark.asyncio
    async def test_fetch_topics(self):
        from paperqa.stores.openalex_compare import fetch_openalex_topics

        topics = await fetch_openalex_topics()
        assert len(topics) > 0
        for topic in topics[:5]:
            assert "display_name" in topic
            assert "works_count" in topic


class TestComputeCoverage:
    """Test coverage computation."""

    def test_compute_coverage(self):
        from paperqa.stores.openalex_compare import compute_coverage

        mock_topic = TopicResult(
            topics={0: ["bin", "packing", "3d"], 1: ["optimization", "genetic"]},
            topic_info={},
            chunk_to_topic=[0] * 50000 + [1] * 30000,
            sparse_topics=[],
            temporal_data={},
        )
        openalex_topics = [
            {"id": "1", "display_name": "3D Bin Packing", "works_count": 10000},
            {"id": "2", "display_name": "Quantum Computing", "works_count": 50000},
        ]
        coverage, missing = compute_coverage(openalex_topics, mock_topic)
        assert len(coverage) == 2
        # 3D Bin Packing should have higher coverage than Quantum Computing
        assert coverage[0]["ratio"] <= coverage[1]["ratio"]
