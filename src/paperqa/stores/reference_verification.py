"""
Reference Verification System

Verifies that generated references actually exist in the corpus.
Prevents LLM hallucination of fake citations.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Reference:
    """A reference/citation with verification status."""

    id: int
    citation: str
    title: str | None = None
    authors: list[str] | None = None
    year: int | None = None
    journal: str | None = None
    doi: str | None = None

    # Verification status
    verified: bool = False
    matched_paper_hash: str | None = None
    confidence: float = 0.0
    warning: str | None = None


class ReferenceVerifier:
    """
    Verifies references against the corpus.

    Ensures that citations in generated papers actually exist
    in the underlying document collection.
    """

    def __init__(
        self,
        qdrant_url: str = "http://192.168.0.28:6333",
        qdrant_collection: str = "paperbridge_glm_v2",
    ):
        self.qdrant_url = qdrant_url
        self.qdrant_collection = qdrant_collection
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from qdrant_client import AsyncQdrantClient
            self._client = AsyncQdrantClient(url=self.qdrant_url)
        return self._client

    async def verify_references(
        self,
        references: list[Reference | dict],
        corpus_limit: int = 1000,
    ) -> list[Reference]:
        """
        Verify a list of references against the corpus.

        Args:
            references: List of Reference objects or dicts
            corpus_limit: Maximum number of corpus entries to load for matching

        Returns:
            List of Reference objects with verification status
        """
        # Load corpus metadata for matching
        logger.info(f"Loading {corpus_limit} corpus entries for reference verification...")
        corpus_metadata = await self._load_corpus_metadata(corpus_limit)

        verified_references = []

        for ref in references:
            # Convert dict to Reference if needed
            if isinstance(ref, dict):
                ref_obj = self._dict_to_reference(ref)
            else:
                ref_obj = ref

            # Verify this reference
            await self._verify_single_reference(ref_obj, corpus_metadata)
            verified_references.append(ref_obj)

        # Log summary
        verified_count = sum(1 for r in verified_references if r.verified)
        logger.info(
            f"Reference verification complete: {verified_count}/{len(verified_references)} verified"
        )

        return verified_references

    def _dict_to_reference(self, ref_dict: dict) -> Reference:
        """Convert a dict to a Reference object."""
        citation = ref_dict.get("citation", ref_dict.get("title", ""))

        # Try to extract year from citation
        year = None
        year_match = re.search(r"\b(19|20)\d{2}\b", citation)
        if year_match:
            year = int(year_match.group())

        return Reference(
            id=ref_dict.get("id", 0),
            citation=citation,
            title=ref_dict.get("title"),
            year=year,
            verified=False,
        )

    async def _verify_single_reference(
        self,
        ref: Reference,
        corpus_metadata: dict[str, dict[str, Any]],
    ) -> None:
        """
        Verify a single reference against corpus metadata.

        Uses multiple matching strategies:
        1. Exact DOI match
        2. Title similarity
        3. Author + Year combination
        """
        # Strategy 1: Try to extract and match DOI
        doi = self._extract_doi_from_citation(ref.citation)
        if doi:
            matched = await self._match_by_doi(doi, corpus_metadata)
            if matched:
                ref.verified = True
                ref.matched_paper_hash = matched["pdf_hash"]
                ref.confidence = 0.95
                return

        # Strategy 2: Match by title similarity
        title_match = await self._match_by_title(ref.citation, corpus_metadata)
        if title_match and title_match["confidence"] > 0.7:
            ref.verified = True
            ref.matched_paper_hash = title_match["pdf_hash"]
            ref.confidence = title_match["confidence"]
            return

        # Strategy 3: Match by year and keywords
        year_match = await self._match_by_year_keywords(ref, corpus_metadata)
        if year_match and year_match["confidence"] > 0.6:
            ref.verified = True
            ref.matched_paper_hash = year_match["pdf_hash"]
            ref.confidence = year_match["confidence"]
            return

        # No match found
        ref.verified = False
        ref.warning = (
            "Reference not found in corpus. Please verify externally "
            "or remove if unverifiable."
        )
        ref.confidence = 0.0

    def _extract_doi_from_citation(self, citation: str) -> str | None:
        """Extract DOI from a citation string."""
        # Look for DOI patterns
        doi_patterns = [
            r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b",  # Standard DOI format
            r"\bd oi\.org/(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b",  # DOI URL
        ]

        for pattern in doi_patterns:
            match = re.search(pattern, citation, re.IGNORECASE)
            if match:
                return match.group(0)

        return None

    async def _match_by_doi(
        self,
        doi: str,
        corpus_metadata: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Match reference by DOI."""
        # Normalize DOI for comparison
        normalized_doi = doi.lower().replace("doi:", "").strip()

        for pdf_hash, meta in corpus_metadata.items():
            meta_doi = meta.get("doi", "").lower().replace("doi:", "").strip()
            if normalized_doi == meta_doi:
                return {"pdf_hash": pdf_hash, "confidence": 0.95}

        return None

    async def _match_by_title(
        self,
        citation: str,
        corpus_metadata: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Match reference by title similarity."""
        # Extract potential title from citation
        # Heuristic: Title is often between author/year and journal
        title_text = self._extract_title_from_citation(citation)

        if not title_text:
            return None

        best_match = None
        best_score = 0.0

        for pdf_hash, meta in corpus_metadata.items():
            meta_title = meta.get("title", "")
            if not meta_title:
                continue

            # Compute similarity score
            score = self._compute_title_similarity(title_text, meta_title)

            if score > best_score:
                best_score = score
                best_match = {"pdf_hash": pdf_hash, "confidence": score}

        return best_match

    async def _match_by_year_keywords(
        self,
        ref: Reference,
        corpus_metadata: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Match reference by year and keyword overlap."""
        if not ref.year:
            return None

        # Extract keywords from citation
        keywords = self._extract_keywords_from_citation(ref.citation)

        best_match = None
        best_score = 0.0

        for pdf_hash, meta in corpus_metadata.items():
            meta_year = meta.get("year")
            if not meta_year:
                continue

            # Check year match (allow 1-year tolerance)
            if abs(meta_year - ref.year) > 1:
                continue

            # Compute keyword overlap score
            meta_title = meta.get("title", "").lower()
            keyword_matches = sum(1 for kw in keywords if kw in meta_title)
            score = keyword_matches / max(len(keywords), 1)

            if score > best_score:
                best_score = score
                best_match = {"pdf_hash": pdf_hash, "confidence": 0.5 + score * 0.5}

        return best_match

    def _extract_title_from_citation(self, citation: str) -> str:
        """
        Extract title from a citation string.

        Heuristic: Title is often the part between author/year and journal.
        """
        # Remove common citation prefixes
        cleaned = citation
        for prefix in ["et al.", "Authors:", "Reference:"]:
            cleaned = cleaned.replace(prefix, "")

        # Try to extract title (heuristic)
        # Often title is the main text before journal name
        parts = cleaned.split(".")
        if len(parts) > 1:
            # Assume last part is journal, middle parts are title
            title_parts = parts[1:-1] if len(parts) > 2 else parts[:-1]
            return " ".join(title_parts).strip()

        return cleaned[:100]  # Fallback: first 100 chars

    def _extract_keywords_from_citation(self, citation: str) -> list[str]:
        """Extract meaningful keywords from a citation."""
        # Remove common words and punctuation
        words = re.findall(r"\b[a-zA-Z]{3,}\b", citation.lower())

        # Filter out common stop words
        stop_words = {
            "the", "and", "for", "with", "from", "that", "this", "from",
            "paper", "study", "research", "article", "journal", "on", "of"
        }

        keywords = [w for w in words if w not in stop_words]

        # Return unique keywords, limited to top 10
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
                if len(unique_keywords) >= 10:
                    break

        return unique_keywords

    def _compute_title_similarity(self, title1: str, title2: str) -> float:
        """
        Compute similarity between two titles.

        Uses simple token overlap as a proxy for similarity.
        """
        # Normalize: lowercase, remove punctuation
        def normalize(text: str) -> set[str]:
            words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
            return set(words)

        words1 = normalize(title1)
        words2 = normalize(title2)

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    async def _load_corpus_metadata(
        self,
        limit: int,
    ) -> dict[str, dict[str, Any]]:
        """
        Load metadata from corpus for reference matching.

        Returns dict mapping pdf_hash to metadata.
        """
        metadata = {}
        offset = None
        seen = 0

        while seen < limit:
            points, next_offset = await self.client.scroll(
                collection_name=self.qdrant_collection,
                limit=min(500, limit - seen),
                offset=offset,
                with_payload=["pdf_hash", "pdf_name", "pdf_path"],
                with_vectors=False,
            )

            if not points:
                break

            for point in points:
                payload = point.payload
                pdf_hash = payload.get("pdf_hash", "")
                if pdf_hash and pdf_hash not in metadata:
                    # Parse metadata from pdf_name
                    meta = self._parse_pdf_metadata(payload)
                    metadata[pdf_hash] = meta

                seen += 1

            if next_offset is None:
                break
            offset = next_offset

        logger.info(f"Loaded metadata for {len(metadata)} papers")
        return metadata

    def _parse_pdf_metadata(self, payload: dict) -> dict[str, Any]:
        """Parse metadata from PDF payload."""
        pdf_name = payload.get("pdf_name", "")

        # Extract year, DOI, title from filename format
        # Format: {year}__{doi_encoded}__{title}.pdf
        parts = pdf_name.replace(".pdf", "").split("__")

        year = None
        doi = None
        title = pdf_name

        if len(parts) >= 1 and parts[0].isdigit():
            year = int(parts[0])

        if len(parts) >= 2:
            doi = parts[1].replace("_", "/")

        if len(parts) >= 3:
            title = parts[2]

        return {
            "pdf_hash": payload.get("pdf_hash", ""),
            "pdf_name": pdf_name,
            "pdf_path": payload.get("pdf_path", ""),
            "year": year,
            "doi": doi,
            "title": title,
        }


def parse_paper_title_from_context(title: str) -> str:
    """
    Parse and clean a paper title from context.

    Removes common prefixes and cleans up formatting.
    """
    # Remove common prefixes
    prefixes = [
        "The ", "A ", "An ", "Study of ", "Research on ",
        "Paper: ", "Title: ", "Reference: "
    ]

    cleaned = title
    for prefix in prefixes:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):]
            break

    # Trim to reasonable length
    if len(cleaned) > 300:
        cleaned = cleaned[:300]

    return cleaned.strip()
