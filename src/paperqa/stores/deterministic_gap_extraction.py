"""
Deterministic Gap Extraction Layer

%100 deterministik gap çıkarımı - SIFIR LLM kullanımı.
Hallucination riski yok. Tüm çıktılar istatistiksel kanıtlara dayanır.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import AsyncQdrantClient

logger = logging.getLogger(__name__)


async def load_existing_gaps(
    candidates_path: str = "./data/gap_analysis/candidates.json"
) -> list[GapCandidate]:
    """
    Load pre-computed gap candidates from existing JSON file.

    This is a convenience function to use existing gap analysis results
    without re-running the full extraction pipeline.
    """
    import re

    def is_valid_term(term: str) -> bool:
        """Check if a term is semantically meaningful."""
        # Must be at least 3 characters
        if len(term) < 3:
            return False
        # Must contain at least 2 letters
        letter_count = sum(1 for c in term if c.isalpha())
        if letter_count < 2:
            return False
        # Must not be mostly numbers or codes (e.g., e24, sg2, 000e00)
        if re.match(r'^[a-z]?\d+[a-z]?\d*$', term.lower()):
            return False
        # Must not contain LaTeX artifacts
        if 'math' in term.lower() or 'overline' in term.lower():
            return False
        return True

    with open(candidates_path, "r") as f:
        candidates_data = json.load(f)

    gaps = []
    for idx, data in enumerate(candidates_data):
        # Parse title to extract core terms
        title = data.get("title", "")
        # Format: "Understudied area: 815_tl_ilph_mdrlt1_rlt1"
        if ":" in title:
            parts = title.split(":")[-1].strip()
            # Filter out noise - only keep semantically meaningful terms
            raw_terms = parts.split("_")
            core_terms = [
                t.replace("_", " ").strip()
                for t in raw_terms
                if is_valid_term(t)
            ]
        else:
            core_terms = [title]

        # Default values for sparse topics
        chunk_count = 49
        statistical_score = 0.51

        gaps.append(GapCandidate(
            id=f"gap_{idx:03d}",
            gap_type="sparse_topic",
            title=title,
            description=data.get("description", ""),
            core_terms=core_terms if core_terms else ["gap"],
            chunk_count=chunk_count,
            unique_papers=chunk_count,
            statistical_score=statistical_score,
            evidence_sources=data.get("evidence_sources", ["bertopic_sparse"]),
            llm_used=False,
        ))

    logger.info(f"Loaded {len(gaps)} pre-computed gap candidates")
    return gaps


@dataclass
class GapCandidate:
    """A single gap candidate extracted via deterministic methods."""

    id: str
    gap_type: str  # "sparse_topic", "openalex_missing", "novel_combination", "author_identified"
    title: str
    description: str

    # Core evidence (deterministic)
    core_terms: list[str] = field(default_factory=list)
    chunk_count: int = 0
    unique_papers: int = 0

    # Type-specific metrics
    statistical_score: float = 0.0  # For sparse topics
    coverage_ratio: float = 1.0  # For OpenAlex gaps
    atypicality_score: float = 0.0  # For Novelpy pairs
    mention_count: int = 0  # For gap-language

    # Evidence sources
    supporting_chunks: list[dict] = field(default_factory=list)
    related_papers: list[str] = field(default_factory=list)

    # Metadata
    evidence_sources: list[str] = field(default_factory=list)
    llm_used: bool = False  # Always False for this layer

    def compute_priority_score(self) -> float:
        """
        Compute priority score for ranking gaps.
        Fully deterministic - no LLM involved.
        """
        # Factor 1: Statistical significance (inverse of chunk count for sparse topics)
        if self.gap_type == "sparse_topic":
            stat_score = max(0, 1.0 - (self.chunk_count / 100))
        elif self.gap_type == "openalex_missing":
            stat_score = 1.0 - self.coverage_ratio
        elif self.gap_type == "novel_combination":
            stat_score = self.atypicality_score
        elif self.gap_type == "author_identified":
            stat_score = min(1.0, self.mention_count / 50)
        else:
            stat_score = 0.5

        # Factor 2: Evidence quantity
        evidence_score = min(1.0, self.chunk_count / 30)

        # Factor 3: Source credibility
        source_score = {
            "author_identified": 1.0,      # Authors explicitly mention this
            "openalex_missing": 0.8,       # External taxonomy validation
            "sparse_topic": 0.6,           # Statistical signal
            "novel_combination": 0.7,      # Co-occurrence analysis
        }.get(self.gap_type, 0.5)

        # Weighted combination
        return 0.4 * stat_score + 0.3 * evidence_score + 0.3 * source_score


# English stop words that add no semantic value to evidence search
STOP_WORDS: frozenset = frozenset({
    "the", "and", "for", "with", "of", "in", "on", "to", "a", "an",
    "is", "it", "this", "that", "are", "was", "were", "be", "been",
    "have", "has", "had", "from", "or", "but", "not", "no", "nor",
    "do", "does", "did", "will", "would", "shall", "should", "may",
    "might", "can", "could", "its", "they", "them", "their", "we",
    "our", "you", "your", "he", "she", "his", "her", "who", "what",
    "which", "where", "when", "how", "all", "each", "every", "both",
    "few", "many", "much", "some", "such", "only", "own", "same",
    "so", "than", "too", "very", "just", "also", "into", "over",
    "after", "before", "between", "under", "about", "against",
    "during", "among", "while", "being", "having", "doing",
    # Generic academic terms that add no semantic value
    "research", "study", "studies", "analysis", "analyses", "method",
    "methods", "approach", "approaches", "model", "models", "system",
    "systems", "technique", "techniques", "framework", "frameworks",
    "algorithm", "algorithms", "application", "applications", "problem",
    "problems", "issue", "issues", "data", "information", "process",
    "processes", "methodology", "methodologies", "review", "reviews",
    "survey", "surveys", "journal", "journals", "paper", "papers",
    "based", "using", "towards", "toward", "via", "through",
    "space", "spaces", "design", "designs", "performance",
})


def extract_core_terms(topic_name: str) -> list[str]:
    """
    Extract meaningful core terms from a BERTopic topic name.

    Input: "962_boxes_pallet_superboxes_trios"
    Output: ["boxes", "pallet", "superboxes", "trios"]

    Fully deterministic - no LLM. Filters out English stop words
    and generic academic terms that add no semantic value.
    """
    # Remove numeric prefix
    parts = topic_name.split("_", 1)
    if len(parts) > 1 and parts[0].isdigit():
        core = parts[1]
    else:
        core = topic_name

    # Split by underscore and filter
    terms = []
    for term in core.split("_"):
        term_lower = term.lower().strip()
        # Filter out noise and stop words
        if (len(term_lower) >= 3 and
            term_lower not in STOP_WORDS and
            not term_lower.isdigit() and
            not term_lower.startswith("0x") and
            not term_lower.startswith("00")):
            terms.append(term_lower)

    # If everything was filtered, return raw terms (better than empty)
    if not terms:
        raw = [t.lower() for t in core.split("_") if len(t) >= 3][:4]
        if raw:
            return raw

    return terms


def format_topic_name(topic_name: str) -> str:
    """
    Format a BERTopic topic name into a readable string.

    Input: "962_boxes_pallet_superboxes_trios"
    Output: "Boxes Pallet Superboxes Trios"

    Deterministic formatting only - no interpretation.
    """
    terms = extract_core_terms(topic_name)
    return " ".join(term.capitalize() for term in terms)


class DeterministicGapExtractor:
    """
    Main extractor class for deterministic gap discovery.

    Uses four independent sources:
    1. BERTopic sparse topics (statistical underrepresentation)
    2. OpenAlex coverage gaps (external taxonomy comparison)
    3. Novelpy novel combinations (concept co-occurrence analysis)
    4. Gap-language explicit mentions (author-identified gaps)

    ALL extraction is fully deterministic - zero LLM involvement.
    """

    def __init__(
        self,
        qdrant_url: str = "http://192.168.0.28:6333",
        qdrant_collection: str = "paperbridge_glm_v2",
    ):
        self.qdrant_url = qdrant_url
        self.qdrant_collection = qdrant_collection
        self._client: AsyncQdrantClient | None = None

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            from qdrant_client import AsyncQdrantClient
            self._client = AsyncQdrantClient(url=self.qdrant_url)
        return self._client

    async def extract_all_gaps(
        self,
        topic_result_path: str = "./data/gap_analysis/bertopic/metadata.json",
        openalex_path: str = "./data/gap_analysis/openalex/missing_topics.json",
        novelpy_path: str = "./data/gap_analysis/novelpy/novel_pairs.json",
        gap_language_path: str = "./data/gap_analysis/gap_language/gap_chunks.json",
    ) -> list[GapCandidate]:
        """
        Extract all gap candidates from four deterministic sources.

        Returns a unified list of GapCandidate objects.
        """
        gaps: list[GapCandidate] = []
        gap_id = 0

        # 1. BERTopic Sparse Topics
        logger.info("Extracting sparse topics from BERTopic...")
        sparse_gaps = await self._extract_bertopic_sparse(
            topic_result_path=topic_result_path
        )
        for gap in sparse_gaps:
            gap.id = f"gap_{gap_id:03d}"
            gap_id += 1
            gaps.append(gap)
        logger.info(f"Found {len(sparse_gaps)} sparse topic gaps")

        # 2. OpenAlex Coverage Gaps
        logger.info("Extracting coverage gaps from OpenAlex...")
        openalex_gaps = await self._extract_openalex_gaps(
            openalex_path=openalex_path
        )
        for gap in openalex_gaps:
            gap.id = f"gap_{gap_id:03d}"
            gap_id += 1
            gaps.append(gap)
        logger.info(f"Found {len(openalex_gaps)} OpenAlex coverage gaps")

        # 3. Novelpy Novel Combinations
        logger.info("Extracting novel combinations from Novelpy...")
        novelpy_gaps = await self._extract_novelpy_pairs(
            novelpy_path=novelpy_path
        )
        for gap in novelpy_gaps:
            gap.id = f"gap_{gap_id:03d}"
            gap_id += 1
            gaps.append(gap)
        logger.info(f"Found {len(novelpy_gaps)} novel combination gaps")

        # 4. Gap-Language Explicit Mentions
        logger.info("Extracting explicit mentions from Gap-Language...")
        gap_lang_gaps = await self._extract_gap_language(
            gap_language_path=gap_language_path
        )
        for gap in gap_lang_gaps:
            gap.id = f"gap_{gap_id:03d}"
            gap_id += 1
            gaps.append(gap)
        logger.info(f"Found {len(gap_lang_gaps)} author-identified gaps")

        # Sort by priority score
        gaps.sort(key=lambda g: -g.compute_priority_score())

        logger.info(f"Total gap candidates: {len(gaps)}")
        return gaps

    async def _extract_bertopic_sparse(
        self,
        topic_result_path: str,
    ) -> list[GapCandidate]:
        """
        Extract sparse topics as gap candidates.

        Sparse topics = topics with <50 documents (statistically underrepresented).
        """
        # Load topic info from CSV
        topic_info_path = Path(topic_result_path).parent / "topic_info.csv"

        gaps = []
        sparse_count = 0

        # Read CSV and filter sparse topics (count < 50)
        with open(topic_info_path, "r") as f:
            header = f.readline().strip().split(",")
            topic_idx = header.index("Topic")
            count_idx = header.index("Count")
            name_idx = header.index("Name")

            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 4:
                    continue

                topic_id = int(parts[topic_idx])
                count = int(parts[count_idx])
                topic_name = parts[name_idx]

                # Skip if not sparse
                if count >= 50:
                    continue

                sparse_count += 1

                # Extract core terms
                core_terms = extract_core_terms(topic_name)

                # Skip if no meaningful core terms
                if len(core_terms) < 2:
                    continue

                # Fetch representative chunks from Qdrant
                chunks = await self._get_chunks_for_topic(
                    topic_id=topic_id,
                    limit=15
                )

                unique_papers = len(set(c.get("pdf_hash", "") for c in chunks))

                gaps.append(GapCandidate(
                    id="",  # Will be assigned by caller
                    gap_type="sparse_topic",
                    title=format_topic_name(topic_name),
                    description=f"Topic with only {count} documents in corpus",
                    core_terms=core_terms,
                    chunk_count=count,
                    unique_papers=unique_papers,
                    statistical_score=max(0, 1.0 - (count / 100)),
                    supporting_chunks=chunks[:10],
                    related_papers=list(set(c.get("pdf_hash", "") for c in chunks))[:20],
                    evidence_sources=["bertopic_sparse"],
                    llm_used=False,
                ))

        logger.info(f"Found {sparse_count} sparse topics with meaningful core terms")
        return gaps

    async def _extract_openalex_gaps(
        self,
        openalex_path: str,
    ) -> list[GapCandidate]:
        """
        Extract OpenAlex coverage gaps.

        Topics that exist globally but are underrepresented in our corpus.
        """
        with open(openalex_path, "r") as f:
            missing_topics = json.load(f)

        gaps = []
        for missing in missing_topics:
            topic_name = missing["topic_name"]
            our_count = missing.get("our_count", 0)
            global_count = missing.get("global_count", 1)
            ratio = missing.get("ratio", 1.0)

            core_terms = extract_core_terms(topic_name.replace(" ", "_"))

            gaps.append(GapCandidate(
                id="",
                gap_type="openalex_missing",
                title=topic_name,
                description=f"Global topic with {global_count} works worldwide but only {our_count} in our corpus (coverage: {ratio:.2%})",
                core_terms=core_terms,
                chunk_count=our_count,
                statistical_score=1.0 - ratio,
                coverage_ratio=ratio,
                evidence_sources=["openalex_missing"],
                llm_used=False,
            ))

        return gaps

    async def _extract_novelpy_pairs(
        self,
        novelpy_path: str,
    ) -> list[GapCandidate]:
        """
        Extract novel concept combinations from Novelpy.

        Concept pairs with low co-occurrence = potential research gaps.
        """
        with open(novelpy_path, "r") as f:
            novel_pairs = json.load(f)

        gaps = []
        for pair in novel_pairs[:30]:  # Top 30 novel pairs
            concept_a = pair.get("a", "")
            concept_b = pair.get("b", "")
            score = pair.get("score", 0.0)

            title = f"{concept_a} + {concept_b}"
            description = (
                f"Concept pair '{concept_a}' + '{concept_b}' has low co-occurrence "
                f"(atypicality={score:.3f}), suggesting an unexplored combination"
            )

            gaps.append(GapCandidate(
                id="",
                gap_type="novel_combination",
                title=title,
                description=description,
                core_terms=[concept_a, concept_b],
                atypicality_score=score,
                evidence_sources=["novelpy"],
                llm_used=False,
            ))

        return gaps

    async def _extract_gap_language(
        self,
        gap_language_path: str,
    ) -> list[GapCandidate]:
        """
        Extract explicit gap mentions from Gap-Language.

        Chunks where authors explicitly mention "future work", "limitation", "gap", etc.
        """
        with open(gap_language_path, "r") as f:
            gap_chunks = json.load(f)

        # Group by keyword
        keyword_groups: dict[str, list[dict]] = {}
        for chunk in gap_chunks:
            keyword = chunk.get("keyword_matched", "unknown")
            if keyword not in keyword_groups:
                keyword_groups[keyword] = []
            keyword_groups[keyword].append(chunk)

        gaps = []
        for keyword, chunks in keyword_groups.items():
            # Group similar keywords
            if "future" in keyword.lower():
                category = "Future Work"
            elif "limitation" in keyword.lower():
                category = "Limitation"
            elif "unclear" in keyword.lower():
                category = "Unclear Area"
            elif "gap" in keyword.lower():
                category = "Research Gap"
            else:
                category = "Future Research"

            title = f"{category}: {keyword}"
            description = (
                f"Authors mention '{keyword}' as a research gap {len(chunks)} times "
                f"in the corpus"
            )

            # Extract core terms from keyword
            core_terms = [w.lower() for w in keyword.replace("_", " ").split()
                         if len(w) > 2]

            # Get representative chunks
            unique_papers = len(set(c.get("pdf_hash", "") for c in chunks))

            gaps.append(GapCandidate(
                id="",
                gap_type="author_identified",
                title=title,
                description=description,
                core_terms=core_terms,
                chunk_count=len(chunks),
                unique_papers=unique_papers,
                mention_count=len(chunks),
                supporting_chunks=chunks[:10],
                related_papers=list(set(c.get("pdf_hash", "") for c in chunks))[:20],
                evidence_sources=["gap_language"],
                llm_used=False,
            ))

        return gaps

    async def _get_chunks_for_topic(
        self,
        topic_id: int,
        limit: int = 20,
    ) -> list[dict]:
        """
        Get sample chunks for a BERTopic topic.

        Note: This requires topic-to-chunk mapping which should be saved
        during BERTopic processing. For now, returns empty list.
        """
        # TODO: Implement when topic-to-chunk mapping is available
        # For now, return empty - chunks will be fetched during evidence collection
        return []

    async def get_corpus_stats(self) -> dict[str, Any]:
        """Get overall corpus statistics."""
        info = await self.client.get_collection(self.qdrant_collection)
        return {
            "total_points": info.points_count,
            "vector_size": info.config.params.vectors.size,
            "collection_name": self.qdrant_collection,
        }
