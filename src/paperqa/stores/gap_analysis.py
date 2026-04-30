"""Phase 2: Gap Analysis Layer — orchestrator + shared dataclasses.

Orchestrates 5 components:
1. BERTopic — topic clustering (175K chunks → 50-200 topics)
2. Gap-Language — explicit gap signals from authors
3. OpenAlex — external taxonomy comparison
4. Novelpy — novelty scoring
5. LLM Synthesis — validate + produce report
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TopicResult:
    """Output from BERTopic topic clustering."""
    topics: dict[int, list[str]]  # topic_id → [top_words]
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
    """Run the full gap analysis pipeline.

    Executes 5 components in order. Saves intermediate results to disk.
    Returns the final GapReport.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. BERTopic — Topic Clustering
    logger.info("=== Phase 2.1: BERTopic topic clustering ===")
    from paperqa.stores.bertopic_pipeline import run_bertopic

    topic_result = await run_bertopic(
        qdrant_collection=qdrant_collection,
        output_dir=str(output_path / "bertopic"),
    )
    logger.info("BERTopic complete: %d topics", len(topic_result.topics))

    # 2. Gap-Language Extraction
    logger.info("=== Phase 2.2: Gap-language extraction ===")
    from paperqa.stores.gap_language import run_gap_language

    gap_lang_result = await run_gap_language(
        output_dir=str(output_path / "gap_language"),
    )
    logger.info("Gap-language complete: %d gap chunks, %d categories",
                len(gap_lang_result.gap_chunks), len(gap_lang_result.gap_categories))

    # 3. OpenAlex Comparison
    logger.info("=== Phase 2.3: OpenAlex taxonomy comparison ===")
    from paperqa.stores.openalex_compare import run_openalex

    openalex_result = await run_openalex(
        topic_result=topic_result,
        output_dir=str(output_path / "openalex"),
    )
    logger.info("OpenAlex complete: %d global topics, %d missing",
                len(openalex_result.global_topics), len(openalex_result.missing_topics))

    # 4. Novelpy Scoring
    logger.info("=== Phase 2.4: Novelpy novelty scoring ===")
    from paperqa.stores.novelpy_pipeline import run_novelpy

    novelpy_result = await run_novelpy(
        topic_result=topic_result,
        output_dir=str(output_path / "novelpy"),
    )
    logger.info("Novelpy complete: %d concepts, %d novel pairs",
                len(novelpy_result.concepts), len(novelpy_result.novel_pairs))

    # 5. LLM Synthesis
    logger.info("=== Phase 2.5: LLM gap synthesis ===")
    from paperqa.stores.gap_synthesis import synthesize_gaps

    report = await synthesize_gaps(
        topic_result=topic_result,
        gap_lang_result=gap_lang_result,
        openalex_result=openalex_result,
        novelpy_result=novelpy_result,
        output_dir=str(output_path / "report"),
        llm_api_base=llm_api_base,
    )
    logger.info("Gap analysis complete: %d gaps identified", len(report.gaps))

    return report
