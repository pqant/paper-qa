"""Phase 2: Gap Analysis Layer — orchestrator + shared dataclasses.

Orchestrates 5 components:
1. BERTopic — topic clustering (corpus chunks → 50-200 topics)
2. Gap-Language — explicit gap signals from authors
3. OpenAlex — external taxonomy comparison
4. Novelpy — novelty scoring
5. LLM Synthesis — validate + produce report
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TopicResult:
    """Output from BERTopic topic clustering."""
    topics: dict[int, list[str]]  # topic_id → [top_words] (BERTopic get_topic() output)
    topic_info: dict[str, Any]  # pandas DataFrame to dict: columns topic_id, count, name, avg_year
    chunk_to_topic: list[int]  # each chunk → topic_id
    sparse_topics: list[dict]  # topics with <50 docs
    temporal_data: dict[int, dict]  # topic_id → {year: count}


@dataclass
class GapLanguageResult:
    """Output from gap-language extraction."""
    gap_chunks: list[dict]  # {pdf_hash, text, page_num, keyword_matched}
    gap_categories: dict[int, dict]  # topic_id → {label, count, representative_texts}
    top_gaps: list[str]  # top gap phrases sorted by count


@dataclass
class OpenAlexResult:
    """Output from OpenAlex taxonomy comparison."""
    global_topics: list[dict]  # {name, works_count, sub_fields}
    our_coverage: list[dict]  # {topic_name, our_count, global_count, ratio}
    missing_topics: list[dict]  # topics with coverage < threshold


@dataclass
class NovelpyResult:
    """Output from Novelpy novelty scoring."""
    concepts: list[str]  # unique concepts
    cooccurrence_counts: list[tuple[str, str, int]]  # (concept_a, concept_b, count)
    novel_pairs: list[tuple[str, str, float]]  # (concept_a, concept_b, atypicality_score)


@dataclass
class GapItem:
    """A single confirmed research gap."""
    id: int
    title: str
    description: str
    evidence_sources: list[str]  # component names that found this gap
    confidence: float  # 0.0-1.0
    papers: list[str]  # related paper citations


@dataclass
class GapReport:
    """Final gap analysis report."""
    gaps: list[GapItem]
    summary: str  # executive summary
    recommendations: list[str]  # suggested research directions
    metadata: dict[str, Any]  # total_papers, total_chunks, analysis_date, components


async def run_gap_analysis(
    qdrant_collection: str = "paperbridge_glm_v2",
    output_dir: str = "./data/gap_analysis/",
    llm_api_base: str = "http://192.168.0.28:8005/v1",
) -> GapReport:
    """Run gap analysis via the enriched report pipeline.

    Delegates to enriched_report.generate_enriched_report() which runs
    all 8 stages (BERTopic, domain filter, gap-language, OpenAlex,
    Novelpy, candidate building, topic humanization, evidence+assessment).

    Returns a GapReport for backward compatibility.
    """
    from paperqa.stores.enriched_report import generate_enriched_report

    logger.info("=== Starting enriched gap analysis pipeline ===")

    enriched = await generate_enriched_report(
        qdrant_collection=qdrant_collection,
        llm_api_base=llm_api_base,
        output_dir=output_dir,
    )

    # Convert enriched report to legacy GapReport for backward compatibility
    gap_items = []
    for i, gap in enumerate(enriched.gaps):
        gap_items.append(GapItem(
            id=i + 1,
            title=gap.title,
            description=gap.description,
            evidence_sources=gap.evidence_sources,
            confidence=gap.priority_score,
            papers=[p.get("pdf_hash", "") for p in gap.top_relevant_papers[:10]],
        ))

    report = GapReport(
        gaps=gap_items,
        summary=enriched.summary,
        recommendations=[
            f"Investigate: {g.title} — {g.description}"
            for g in enriched.gaps[:10]
        ],
        metadata=enriched.metadata,
    )

    logger.info("Gap analysis complete: %d gaps identified", len(report.gaps))
    return report
