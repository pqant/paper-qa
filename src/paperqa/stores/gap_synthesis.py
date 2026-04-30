"""LLM Gap Synthesis — validate gap candidates and produce final report.

Collects candidates from BERTopic, OpenAlex, Novelpy, Gap-Language.
Validates each candidate with paperbridge_agent_query.
Synthesizes a structured report.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

from paperqa.stores.agent import paperbridge_agent_query
from paperqa.stores.gap_analysis import (
    GapItem,
    GapLanguageResult,
    GapReport,
    NovelpyResult,
    OpenAlexResult,
    TopicResult,
)

logger = logging.getLogger(__name__)


def collect_candidate_gaps(
    topic_result: TopicResult,
    gap_lang_result: GapLanguageResult,
    openalex_result: OpenAlexResult,
    novelpy_result: NovelpyResult,
) -> list[dict]:
    """Collect gap candidates from all 4 components.

    Returns list of {title, description, evidence_sources}.
    """
    candidates: list[dict] = []
    seen_titles: set[str] = set()

    # 1. Sparse BERTopic topics (understudied areas)
    for sparse in topic_result.sparse_topics:
        title = f"Understudied area: {sparse['name']}"
        if title not in seen_titles:
            seen_titles.add(title)
            candidates.append({
                "title": title,
                "description": f"Topic '{sparse['name']}' has only {sparse['count']} documents in corpus of 175K chunks.",
                "evidence_sources": ["bertopic_sparse"],
            })

    # 2. Missing OpenAlex topics (globally important, locally absent)
    for missing in openalex_result.missing_topics[:20]:
        title = f"Missing global topic: {missing['topic_name']}"
        if title not in seen_titles:
            seen_titles.add(title)
            candidates.append({
                "title": title,
                "description": f"OpenAlex topic '{missing['topic_name']}' has {missing['global_count']} global works but coverage ratio {missing['ratio']}.",
                "evidence_sources": ["openalex_missing"],
            })

    # 3. Novel concept pairs
    for c_a, c_b, score in novelpy_result.novel_pairs[:20]:
        title = f"Novel combination: {c_a} + {c_b}"
        if title not in seen_titles:
            seen_titles.add(title)
            candidates.append({
                "title": title,
                "description": f"Concept pair '{c_a}' + '{c_b}' has low co-occurrence (atypicality={score}), suggesting an unexplored research direction.",
                "evidence_sources": ["novelpy"],
            })

    # 4. Gap-language categories
    for topic_id, cat in gap_lang_result.gap_categories.items():
        title = f"Author-identified gap: {cat['label']}"
        if title not in seen_titles:
            seen_titles.add(title)
            candidates.append({
                "title": title,
                "description": f"Authors mention '{cat['label']}' as a gap {cat['count']} times. Keywords: {', '.join(cat.get('words', [])[:5])}.",
                "evidence_sources": ["gap_language"],
            })

    logger.info("Collected %d gap candidates from 4 components", len(candidates))
    return candidates


async def validate_candidate(
    candidate: dict,
    llm_api_base: str = "http://192.168.0.28:8005/v1",
) -> dict:
    """Validate a gap candidate with paperbridge_agent_query.

    Returns {title, description, evidence_sources, answer, is_confirmed, contexts}.
    """
    query = f"What research exists on {candidate['title']} in 3D bin packing or container loading?"

    result = await paperbridge_agent_query(
        query=query,
        llm_api_base=llm_api_base,
        evidence_skip_summary=True,
    )

    answer = result.session.answer
    # Simple heuristic: if answer is very short or contains uncertainty signals
    is_confirmed = (
        len(answer) < 50
        or "insufficient" in answer.lower()
        or "no evidence" in answer.lower()
        or "no studies" in answer.lower()
        or "cannot answer" in answer.lower()
    )

    return {
        "title": candidate["title"],
        "description": candidate["description"],
        "evidence_sources": candidate["evidence_sources"],
        "answer": answer[:500],
        "is_confirmed": is_confirmed,
        "contexts": len(result.session.contexts),
    }


async def synthesize_gaps(
    topic_result: TopicResult,
    gap_lang_result: GapLanguageResult,
    openalex_result: OpenAlexResult,
    novelpy_result: NovelpyResult,
    output_dir: str = "./data/gap_analysis/report",
    llm_api_base: str = "http://192.168.0.28:8005/v1",
    max_candidates: int = 20,
) -> GapReport:
    """Validate gap candidates and produce final report.

    Args:
        max_candidates: Maximum number of candidates to validate with LLM
            (due to LLM parallel=1 constraint, keep this low).
    """
    # Collect candidates
    candidates = collect_candidate_gaps(
        topic_result, gap_lang_result, openalex_result, novelpy_result,
    )
    candidates = candidates[:max_candidates]

    # Validate each candidate (sequential due to parallel=1)
    confirmed_gaps: list[GapItem] = []
    gap_id = 0

    for i, candidate in enumerate(candidates):
        logger.info("Validating candidate %d/%d: %s",
                    i + 1, len(candidates), candidate["title"])
        result = await validate_candidate(candidate, llm_api_base)

        if result["is_confirmed"]:
            gap_id += 1
            confidence = min(1.0, len(result["evidence_sources"]) * 0.3 + 0.3)
            confirmed_gaps.append(GapItem(
                id=gap_id,
                title=result["title"],
                description=result["description"],
                evidence_sources=result["evidence_sources"],
                confidence=round(confidence, 2),
                papers=[],
            ))

    # Sort by confidence descending
    confirmed_gaps.sort(key=lambda x: -x.confidence)

    # Generate executive summary
    summary = (
        f"Analysis of 9,987 papers ({len(topic_result.chunk_to_topic):,} chunks) "
        f"in 3D container loading / bin packing identified {len(confirmed_gaps)} "
        f"actionable research gaps across {len(topic_result.topics)} discovered topics. "
        f"Gaps were validated using {len(candidates)} candidate queries against the corpus."
    )

    # Recommendations
    recommendations = [
        f"Investigate: {gap.title} — {gap.description}"
        for gap in confirmed_gaps[:10]
    ]

    report = GapReport(
        gaps=confirmed_gaps,
        summary=summary,
        recommendations=recommendations,
        metadata={
            "total_papers": 9987,
            "total_chunks": len(topic_result.chunk_to_topic),
            "total_topics": len(topic_result.topics),
            "analysis_date": str(date.today()),
            "components": ["bertopic", "gap_language", "openalex", "novelpy", "llm_synthesis"],
            "candidates_validated": len(candidates),
            "confirmed_gaps": len(confirmed_gaps),
        },
    )

    # Save artifacts
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # JSON report
    report_json = {
        "summary": report.summary,
        "gaps": [
            {
                "id": g.id,
                "title": g.title,
                "description": g.description,
                "evidence_sources": g.evidence_sources,
                "confidence": g.confidence,
            }
            for g in report.gaps
        ],
        "recommendations": report.recommendations,
        "metadata": report.metadata,
    }
    with open(output_path / "report.json", "w") as f:
        json.dump(report_json, f, indent=2)

    # Markdown report
    md_lines = [
        "# Gap Analysis Report",
        "",
        report.summary,
        "",
        "## Research Gaps",
        "",
    ]
    for gap in report.gaps:
        md_lines.append(f"### {gap.id}. {gap.title}")
        md_lines.append("")
        md_lines.append(f"**Confidence:** {gap.confidence}")
        md_lines.append(f"**Evidence sources:** {', '.join(gap.evidence_sources)}")
        md_lines.append("")
        md_lines.append(gap.description)
        md_lines.append("")

    md_lines.append("## Recommendations")
    md_lines.append("")
    for i, rec in enumerate(report.recommendations):
        md_lines.append(f"{i + 1}. {rec}")
    md_lines.append("")

    with open(output_path / "report.md", "w") as f:
        f.write("\n".join(md_lines))

    logger.info("Saved gap analysis report: %d gaps, %s",
                len(confirmed_gaps), output_path)

    return report
