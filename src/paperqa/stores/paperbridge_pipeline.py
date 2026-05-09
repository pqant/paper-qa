"""
PaperBridge Paper-Oriented Gap Analysis Pipeline

Main orchestrator for the paper-oriented gap analysis pipeline.
Combines deterministic extraction with controlled LLM assessment.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HumanReviewItem:
    """A gap candidate prepared for human review."""

    # Gap information
    gap_id: str
    gap_type: str
    title: str
    description: str
    core_terms: list[str]

    # Evidence
    chunk_count: int
    unique_papers: int
    supporting_chunks: list[dict]
    yearly_distribution: dict[int, int]

    # LLM Assessment (if available)
    llm_assessment: dict | None = None

    # Human input (filled during review)
    human_verdict: str | None = None  # "approve", "reject", "needs_more_research"
    paper_type: str | None = None
    priority: int = 0  # 1-5
    notes: str = ""
    proposed_methodology: str = ""
    target_venue: str = ""

    # Computed score
    priority_score: float = 0.0


@dataclass
class PipelineResult:
    """Result of the gap analysis pipeline."""

    # Gaps prepared for human review
    review_items: list[HumanReviewItem]

    # All gap candidates (including those not sent for review)
    all_gaps: list[dict] = field(default_factory=list)

    # Statistics
    stats: dict[str, Any] = field(default_factory=dict)

    # Timestamp
    timestamp: str = ""

    def save_to_json(self, output_path: str) -> None:
        """Save pipeline result to JSON file."""
        data = {
            "timestamp": self.timestamp,
            "stats": self.stats,
            "review_items": [
                {
                    "gap_id": item.gap_id,
                    "gap_type": item.gap_type,
                    "title": item.title,
                    "description": item.description,
                    "core_terms": item.core_terms,
                    "chunk_count": item.chunk_count,
                    "unique_papers": item.unique_papers,
                    "yearly_distribution": item.yearly_distribution,
                    "llm_assessment": item.llm_assessment,
                    "priority_score": item.priority_score,
                }
                for item in self.review_items
            ],
            "all_gaps": self.all_gaps,
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved pipeline result to {output_path}")


async def run_paper_oriented_gap_pipeline(
    qdrant_collection: str = "paperbridge_glm_v2",
    output_dir: str = "./data/gap_analysis/pipeline_output",
    llm_api_base: str = "http://localhost:8081/v1",
    qdrant_url: str = "http://localhost:6333",
    max_review_items: int = 30,
) -> PipelineResult:
    """
    Run the complete paper-oriented gap analysis pipeline.

    Pipeline stages:
    1. Deterministic gap extraction (zero LLM)
    2. Evidence collection for each gap
    3. LLM paper worthiness assessment (controlled, verified)
    4. Priority ranking and selection for human review

    Args:
        qdrant_collection: Qdrant collection name
        output_dir: Directory to save results
        llm_api_base: LLM server endpoint
        max_review_items: Maximum number of gaps to prepare for human review

    Returns:
        PipelineResult with review items and statistics
    """
    from paperqa.stores.deterministic_gap_extraction import (
        DeterministicGapExtractor,
        load_existing_gaps,
    )
    from paperqa.stores.gap_assessment import assess_gap_worthiness
    from paperqa.stores.gap_evidence import EvidenceCollector
    from paperqa.stores.reference_verification import ReferenceVerifier

    logger.info("Starting paper-oriented gap analysis pipeline...")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()

    # === STAGE 1: Deterministic Gap Extraction ===
    logger.info("Stage 1: Loading pre-computed gap candidates...")

    # Use pre-computed gaps from existing analysis (much faster)
    raw_gaps = await load_existing_gaps(
        candidates_path="./data/gap_analysis/candidates.json"
    )
    logger.info(f"Loaded {len(raw_gaps)} gap candidates")

    # === STAGE 2: Evidence Collection ===
    logger.info("Stage 2: Collecting evidence for gaps...")

    # Build embedding model (reuse existing PaperBridge setup)
    from lmi.embeddings import LiteLLMEmbeddingModel

    embedding_model = LiteLLMEmbeddingModel(
        name="ollama/hf.co/Qwen/Qwen3-Embedding-8B-GGUF:F16",
        config={"kwargs": {"api_base": "http://localhost:8081/v1"}},
    )

    evidence_collector = EvidenceCollector(
        qdrant_url=qdrant_url,
        qdrant_collection=qdrant_collection,
    )

    gap_with_evidence = []
    for gap in raw_gaps:
        try:
            evidence = await evidence_collector.collect_evidence_for_gap(
                gap=gap,
                embedding_model=embedding_model,
                max_chunks=50,
            )
            gap_with_evidence.append((gap, evidence))
        except Exception as e:
            logger.warning(f"Failed to collect evidence for gap '{gap.title}': {e}")
            continue

    logger.info(f"Collected evidence for {len(gap_with_evidence)} gaps")

    # === STAGE 3: LLM Assessment (Controlled) ===
    logger.info("Stage 3: LLM paper worthiness assessment...")

    assessed_gaps = []
    for gap, evidence in gap_with_evidence:
        try:
            assessment = await assess_gap_worthiness(
                gap=gap,
                evidence=evidence,
                llm_api_base=llm_api_base,
            )
            assessed_gaps.append((gap, evidence, assessment))
        except Exception as e:
            logger.warning(f"LLM assessment failed for gap '{gap.title}': {e}")
            # Add with conservative assessment
            from paperqa.stores.gap_assessment import PaperWorthinessAssessment
            assessed_gaps.append((
                gap,
                evidence,
                PaperWorthinessAssessment(
                    paper_worthy="insufficient_evidence",
                    confidence=0.0,
                    reasoning=f"Assessment failed: {str(e)}",
                ),
            ))

    logger.info(f"Completed LLM assessment for {len(assessed_gaps)} gaps")

    # === STAGE 4: Priority Ranking ===
    logger.info("Stage 4: Ranking gaps by priority...")

    # Calculate priority scores
    for gap, evidence, assessment in assessed_gaps:
        # Base score from deterministic calculation
        base_score = gap.compute_priority_score()

        # Adjust based on LLM assessment
        if assessment.paper_worthy == "worthy":
            llm_bonus = assessment.confidence * 0.3
        elif assessment.paper_worthy == "not_worthy":
            llm_bonus = -0.3
        else:
            llm_bonus = 0.0

        # Adjust based on evidence quantity
        evidence_bonus = min(0.2, evidence.unique_papers / 100)

        gap.priority_score = base_score + llm_bonus + evidence_bonus

    # Sort by priority score
    assessed_gaps.sort(key=lambda x: -x[0].priority_score)

    # === STAGE 5: Prepare for Human Review ===
    logger.info("Stage 5: Preparing for human review...")

    review_items = []
    for gap, evidence, assessment in assessed_gaps[:max_review_items]:
        llm_assessment_dict = None
        if assessment:
            llm_assessment_dict = {
                "paper_worthy": assessment.paper_worthy,
                "confidence": assessment.confidence,
                "recommended_paper_type": assessment.recommended_paper_type,
                "reasoning": assessment.reasoning,
                "concerns": assessment.concerns,
                "suggested_focus": assessment.suggested_focus,
            }

        item = HumanReviewItem(
            gap_id=gap.id,
            gap_type=gap.gap_type,
            title=gap.title,
            description=gap.description,
            core_terms=gap.core_terms,
            chunk_count=evidence.total_chunks,
            unique_papers=evidence.unique_papers,
            supporting_chunks=[
                {"text": c.text[:200], "pdf_hash": c.pdf_hash, "score": c.relevance_score}
                for c in evidence.supporting_chunks[:10]
            ],
            yearly_distribution=evidence.yearly_distribution,
            llm_assessment=llm_assessment_dict,
            priority_score=gap.priority_score,
        )
        review_items.append(item)

    # === Generate Statistics ===
    stats = {
        "total_candidates": len(raw_gaps),
        "with_evidence": len(gap_with_evidence),
        "assessed": len(assessed_gaps),
        "for_human_review": len(review_items),
        "llm_used_in_extraction": False,
        "llm_used_in_assessment": True,
        "assessment_breakdown": {
            "worthy": sum(1 for _, _, a in assessed_gaps if a.paper_worthy == "worthy"),
            "not_worthy": sum(1 for _, _, a in assessed_gaps if a.paper_worthy == "not_worthy"),
            "insufficient_evidence": sum(
                1 for _, _, a in assessed_gaps if a.paper_worthy == "insufficient_evidence"
            ),
        },
    }

    # === Save Results ===
    result = PipelineResult(
        review_items=review_items,
        all_gaps=[
            {
                "id": g.id,
                "type": g.gap_type,
                "title": g.title,
                "description": g.description,
                "priority_score": g.priority_score,
            }
            for g, _, _ in assessed_gaps
        ],
        stats=stats,
        timestamp=timestamp,
    )

    result.save_to_json(output_path / "pipeline_result.json")

    logger.info("Pipeline complete!")
    logger.info(f"Statistics: {stats}")

    return result


# Convenience function for direct import
async def run_gap_analysis_for_papers(
    output_dir: str = "./data/gap_analysis/pipeline_output",
    llm_api_base: str = "http://192.168.0.28:8005/v1",
) -> PipelineResult:
    """
    Convenience function to run gap analysis for paper production.

    Args:
        output_dir: Directory to save results
        llm_api_base: LLM server endpoint

    Returns:
        PipelineResult ready for human review
    """
    return await run_paper_oriented_gap_pipeline(
        output_dir=output_dir,
        llm_api_base=llm_api_base,
        max_review_items=30,
    )
