"""
Gap Evidence Collection System

Collects detailed, verifiable evidence for gap candidates.
Fully deterministic - no LLM involvement.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvidenceChunk:
    """A single chunk of text supporting a gap claim."""

    text: str
    pdf_hash: str
    pdf_name: str
    page_num: int
    relevance_score: float
    embedding: list[float] | None = None


@dataclass
class DetailedEvidence:
    """Comprehensive evidence collection for a gap candidate."""

    # Core statistics
    total_chunks: int = 0
    unique_papers: int = 0
    unique_years: list[int] = field(default_factory=list)

    # Supporting material
    supporting_chunks: list[EvidenceChunk] = field(default_factory=list)
    top_relevant_papers: list[dict] = field(default_factory=list)

    # Temporal analysis
    yearly_distribution: dict[int, int] = field(default_factory=dict)

    # Co-citation analysis
    co_cited_topics: list[str] = field(default_factory=list)

    # Coverage metrics
    coverage_ratio: float = 0.0
    domain_percentage: float = 0.0

    # Metadata
    collection_timestamp: str = ""
    qdrant_collection: str = ""


class EvidenceCollector:
    """
    Collects detailed evidence for gap candidates.

    All methods are fully deterministic - uses Qdrant search and
    statistical analysis only. No LLM involved.
    """

    def __init__(
        self,
        qdrant_url: str = "http://192.168.0.28:6333",
        qdrant_collection: str = "paperbridge_glm_v2",
        embedding_url: str = "http://192.168.0.28:8082",
    ):
        self.qdrant_url = qdrant_url
        self.qdrant_collection = qdrant_collection
        self.embedding_url = embedding_url
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from qdrant_client import AsyncQdrantClient
            self._client = AsyncQdrantClient(url=self.qdrant_url)
        return self._client

    async def close(self) -> None:
        """Close the underlying Qdrant client connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def collect_evidence_for_gap(
        self,
        gap: "GapCandidate",  # Type hint as string to avoid circular import
        embedding_model,
        max_chunks: int = 50,
    ) -> DetailedEvidence:
        """
        Collect comprehensive evidence for a gap candidate.

        Args:
            gap: The gap candidate to collect evidence for
            embedding_model: Model for computing query embeddings
            max_chunks: Maximum number of chunks to retrieve

        Returns:
            DetailedEvidence with all supporting material
        """
        evidence = DetailedEvidence(
            collection_timestamp="",  # Will be set by caller
            qdrant_collection=self.qdrant_collection,
        )

        # Step 1: Build search queries from gap's core terms
        queries = self._build_search_queries(gap)
        logger.info(f"Built {len(queries)} search queries for gap: {gap.title}")

        # Step 2: Search Qdrant for each query
        all_chunks: list[EvidenceChunk] = []
        seen_chunk_keys: set[str] = set()

        for query in queries:
            chunks = await self._search_chunks(
                query=query,
                embedding_model=embedding_model,
                k=max_chunks // len(queries) + 5,
            )
            for chunk in chunks:
                key = f"{chunk.pdf_hash}:{chunk.page_num}:{hash(chunk.text[:100])}"
                if key not in seen_chunk_keys:
                    seen_chunk_keys.add(key)
                    all_chunks.append(chunk)

        # Step 3: Deduplicate and sort by relevance
        all_chunks.sort(key=lambda c: -c.relevance_score)
        all_chunks = all_chunks[:max_chunks]

        # Step 4: Compute statistics
        evidence.total_chunks = len(all_chunks)
        evidence.unique_papers = len(set(c.pdf_hash for c in all_chunks))
        evidence.supporting_chunks = all_chunks

        # Step 5: Extract temporal distribution
        year_counts: dict[int, int] = {}
        for chunk in all_chunks:
            year = self._extract_year_from_pdf_name(chunk.pdf_name)
            if year:
                year_counts[year] = year_counts.get(year, 0) + 1

        evidence.yearly_distribution = dict(sorted(year_counts.items()))
        evidence.unique_years = sorted(year_counts.keys())

        # Step 6: Compute coverage metrics
        total_corpus_size = await self._get_corpus_size()
        evidence.coverage_ratio = len(all_chunks) / max(total_corpus_size, 1)
        evidence.domain_percentage = 100.0 * evidence.coverage_ratio

        # Step 7: Get top relevant papers
        paper_scores: dict[str, float] = {}
        for chunk in all_chunks:
            if chunk.pdf_hash not in paper_scores:
                paper_scores[chunk.pdf_hash] = chunk.relevance_score
            else:
                paper_scores[chunk.pdf_hash] = max(
                    paper_scores[chunk.pdf_hash],
                    chunk.relevance_score
                )

        evidence.top_relevant_papers = [
            {"pdf_hash": h, "relevance_score": s}
            for h, s in sorted(paper_scores.items(), key=lambda x: -x[1])[:20]
        ]

        logger.info(
            f"Collected evidence: {evidence.total_chunks} chunks, "
            f"{evidence.unique_papers} papers for gap '{gap.title}'"
        )

        return evidence

    def _build_search_queries(self, gap: "GapCandidate") -> list[str]:
        """
        Build search queries from gap's core terms.

        Creates multiple query variations to improve recall.
        """
        queries = []

        # Primary query: core terms joined
        if gap.core_terms:
            queries.append(" ".join(gap.core_terms))

        # Individual term queries
        for term in gap.core_terms:
            if len(term) > 3:  # Skip very short terms
                queries.append(term)

        # Type-specific queries
        if gap.gap_type == "novel_combination":
            # For concept pairs, search for each concept separately
            if len(gap.core_terms) >= 2:
                queries.append(f"{gap.core_terms[0]} {gap.core_terms[1]}")

        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            q_lower = q.lower()
            if q_lower not in seen:
                seen.add(q_lower)
                unique_queries.append(q)

        return unique_queries[:6]  # Limit to 6 queries max

    async def _search_chunks(
        self,
        query: str,
        embedding_model,
        k: int = 20,
    ) -> list[EvidenceChunk]:
        """
        Search Qdrant for chunks relevant to the query.

        Uses the provided embedding model to embed the query,
        then performs cosine similarity search.
        """
        # Embed the query
        try:
            embedding_model.set_mode(getattr(__import__('lmi', fromlist=['EmbeddingModes']), 'EmbeddingModes', None)
                                     and __import__('lmi', fromlist=['EmbeddingModes']).EmbeddingModes.QUERY
                                     or object())
        except (AttributeError, TypeError, ImportError):
            pass

        query_embedding = (await embedding_model.embed_documents([query]))[0]

        # Query Qdrant
        results = await self.client.query_points(
            collection_name=self.qdrant_collection,
            query=query_embedding,
            limit=k,
            with_payload=True,
            with_vectors=False,
        )

        min_chars = int(os.getenv("EVIDENCE_MIN_CHUNK_CHARS", "60"))

        # Convert to EvidenceChunk objects
        chunks = []
        for point in results.points:
            payload = point.payload
            raw = (payload.get("text") or "").strip()
            if len(raw) < min_chars:
                continue
            chunks.append(EvidenceChunk(
                text=raw[:500],
                pdf_hash=payload.get("pdf_hash", ""),
                pdf_name=payload.get("pdf_name", ""),
                page_num=payload.get("page_num", 0),
                relevance_score=point.score,
            ))

        return chunks

    def _extract_year_from_pdf_name(self, pdf_name: str) -> int | None:
        """
        Extract year from PDF filename.

        Expected format: {year}__{doi}__{title}.pdf
        Example: 2025__arxiv_2503.08863__Title.pdf
        """
        try:
            parts = pdf_name.replace(".pdf", "").split("__")
            if parts and parts[0].isdigit():
                return int(parts[0])
        except (ValueError, IndexError):
            pass
        return None

    async def _get_corpus_size(self) -> int:
        """Get total number of points in the collection."""
        info = await self.client.get_collection(self.qdrant_collection)
        return info.points_count or 0

    async def get_papers_metadata(
        self,
        pdf_hashes: list[str],
    ) -> dict[str, dict[str, Any]]:
        """
        Get metadata for specific papers.

        Args:
            pdf_hashes: List of PDF hashes to fetch metadata for

        Returns:
            Dictionary mapping pdf_hash to metadata dict
        """
        metadata = {}

        # Scroll through Qdrant to find matching papers
        seen_hashes: set[str] = set()
        offset = None

        target_hashes = set(pdf_hashes)

        while len(seen_hashes & target_hashes) < len(target_hashes):
            points, next_offset = await self.client.scroll(
                collection_name=self.qdrant_collection,
                limit=500,
                offset=offset,
                with_payload=["pdf_hash", "pdf_name", "pdf_path"],
                with_vectors=False,
            )

            if not points:
                break

            for point in points:
                pdf_hash = point.payload.get("pdf_hash", "")
                if pdf_hash in target_hashes and pdf_hash not in metadata:
                    metadata[pdf_hash] = {
                        "pdf_hash": pdf_hash,
                        "pdf_name": point.payload.get("pdf_name", ""),
                        "pdf_path": point.payload.get("pdf_path", ""),
                    }
                    seen_hashes.add(pdf_hash)

            if next_offset is None:
                break
            offset = next_offset

        return metadata
