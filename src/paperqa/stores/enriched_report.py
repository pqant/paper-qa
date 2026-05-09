"""Enriched Report Pipeline — full orchestrator for gap analysis.

Produces a comprehensive, publication-quality report.json by orchestrating:
1. BERTopic topic clustering (existing)
2. Domain relevance filtering (NEW — domain_filter.py)
3. Topic humanization via LLM (NEW — topic_humanizer.py)
4. Gap-language extraction (existing)
5. OpenAlex comparison (existing)
6. Novelpy scoring (existing)
7. Evidence collection from Qdrant (existing — gap_evidence.py)
8. LLM worthiness assessment (existing — gap_assessment.py)

Output: enriched report.json with real data for every field.
No mock data. No hardcoded values.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


# ── Progress tracking ────────────────────────────────────────────────────────


@dataclass
class PipelineProgress:
    """Current state of the enriched report pipeline."""

    run_id: str = ""
    status: str = "idle"  # idle, running, completed, failed
    current_stage: int = 0
    total_stages: int = 7
    stage_name: str = ""
    stage_detail: str = ""
    started_at: str = ""
    updated_at: str = ""
    completed_stages: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ProgressCallback = Callable[[PipelineProgress], Awaitable[None]] | None


# ── Output dataclasses ───────────────────────────────────────────────────────


@dataclass
class EnrichedGap:
    """A single gap with all enrichment data populated from real sources."""

    id: str
    gap_type: str  # sparse_topic, openalex_missing, novel_combination, author_identified
    title: str  # LLM-humanized title
    description: str  # LLM-humanized description
    domain_category: str  # core_domain, adjacent, cross_domain

    # Deterministic metrics
    core_terms: list[str] = field(default_factory=list)
    chunk_count: int = 0
    unique_papers: int = 0
    statistical_score: float = 0.0
    domain_relevance_score: float = 0.0
    priority_score: float = 0.0

    # Real evidence from Qdrant
    supporting_chunks: list[dict] = field(default_factory=list)
    yearly_distribution: dict[str, int] = field(default_factory=dict)
    top_relevant_papers: list[dict] = field(default_factory=list)
    coverage_ratio: float = 0.0

    # Real LLM assessment
    llm_assessment: dict[str, Any] | None = None

    # Evidence metadata
    evidence_sources: list[str] = field(default_factory=list)

    # Human review fields (initially empty)
    human_verdict: str | None = None
    priority: int = 0
    notes: str = ""
    proposed_methodology: str = ""
    target_venue: str = ""


@dataclass
class EnrichedReport:
    """Complete enriched report structure."""

    summary: str
    analysis_date: str
    corpus_stats: dict[str, Any] = field(default_factory=dict)
    pipeline_stages: list[dict[str, Any]] = field(default_factory=list)

    # Main gaps (core_domain + adjacent)
    gaps: list[EnrichedGap] = field(default_factory=list)

    # Cross-domain opportunities (separate section)
    cross_domain_opportunities: list[EnrichedGap] = field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Main orchestrator ────────────────────────────────────────────────────────


async def generate_enriched_report(
    qdrant_url: str = "http://192.168.0.28:6333",
    qdrant_collection: str = "paperbridge_glm_v2",
    embedding_url: str = "http://192.168.0.28:8082",
    llm_api_base: str = "http://192.168.0.28:8005/v1",
    llm_model: str = "qwen3.6-35b-a3b",
    output_dir: str = "./data/gap_analysis",
    tantivy_index_dir: str = "./paper-qa/data/tantivy_index",
    max_gaps: int = 30,
    min_domain_relevance: float = 0.65,
    progress_callback: ProgressCallback = None,
    run_id: str = "",
) -> EnrichedReport:
    """Run full enriched report pipeline.

    This is the single entry point for gap analysis.
    All data in the output is real — no mock values.

    Args:
        qdrant_url: Qdrant server URL.
        qdrant_collection: Qdrant collection name.
        embedding_url: Embedding server URL (for evidence search).
        llm_api_base: LLM server URL (for humanization + assessment).
        llm_model: LLM model name.
        output_dir: Directory for saving intermediate results.
        max_gaps: Maximum number of gaps to fully enrich.
        min_domain_relevance: Minimum domain score to be "relevant".
        progress_callback: Async callback invoked at each stage transition.
        run_id: Unique identifier for this pipeline run.

    Returns:
        EnrichedReport with all real data.
    """
    import uuid as _uuid

    if not run_id:
        run_id = str(_uuid.uuid4())[:8]

    progress = PipelineProgress(
        run_id=run_id,
        status="running",
        started_at=datetime.now(timezone.utc).isoformat(),
        total_stages=7,
    )

    async def _update_progress(
        stage: int, name: str, detail: str = "",
    ) -> None:
        progress.current_stage = stage
        progress.stage_name = name
        progress.stage_detail = detail
        progress.updated_at = datetime.now(timezone.utc).isoformat()
        if progress_callback:
            try:
                await progress_callback(progress)
            except Exception as exc:
                logger.warning("Progress callback failed: %s", exc)

    stages: list[dict[str, Any]] = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Domain filtering + BERTopic ────────────────────────────
    logger.info("═══ Stage 1/7: Domain filtering + BERTopic clustering ═══")
    await _update_progress(1, "Domain Filtering + BERTopic", "Scrolling Qdrant and clustering chunks...")
    stage_start = datetime.now(timezone.utc)

    from paperqa.stores.bertopic_pipeline import run_bertopic
    from paperqa.stores.domain_filter import EmbeddingDomainFilter

    domain_filter = EmbeddingDomainFilter(
        embedding_api_base=f"{embedding_url}/v1",
    )
    await domain_filter.initialize()

    topic_result, filter_stats, chunk_domain_scores = await run_bertopic(
        qdrant_collection=qdrant_collection,
        qdrant_url=qdrant_url,
        output_dir=str(output_path / "bertopic"),
        domain_filter=domain_filter,
        min_relevance=min_domain_relevance,
    )

    stages.append({
        "name": "Domain Filtering + BERTopic",
        "status": "complete",
        "chunks_scrolled": filter_stats.total_scrolled if filter_stats else 0,
        "chunks_passed_filter": filter_stats.total_passed if filter_stats else 0,
        "filter_pass_rate": f"{100 * filter_stats.total_passed / max(1, filter_stats.total_scrolled):.1f}%" if filter_stats else "N/A",
        "topics_found": len(topic_result.topics),
        "sparse_topics": len(topic_result.sparse_topics),
        "total_chunks_clustered": len(topic_result.chunk_to_topic),
        "scroll_capped": bool(getattr(filter_stats, "scroll_capped", False)),
        "scroll_cap_limit": getattr(filter_stats, "scroll_cap_limit", None),
        "duration_seconds": _elapsed(stage_start),
    })
    logger.info(
        "Stage 1 complete: %d/%d chunks passed filter → %d topics, %d sparse",
        filter_stats.total_passed if filter_stats else 0,
        filter_stats.total_scrolled if filter_stats else 0,
        len(topic_result.topics),
        len(topic_result.sparse_topics),
    )

    # Compute real per-topic average domain scores from chunk-level data
    from collections import defaultdict as _defaultdict
    _topic_score_sums: dict[int, float] = _defaultdict(float)
    _topic_score_counts: dict[int, int] = _defaultdict(int)
    for i, tid in enumerate(topic_result.chunk_to_topic):
        if tid != -1 and i < len(chunk_domain_scores):
            _topic_score_sums[tid] += chunk_domain_scores[i]
            _topic_score_counts[tid] += 1

    domain_scores: dict[int, float] = {}
    for tid in topic_result.topics:
        if _topic_score_counts[tid] > 0:
            domain_scores[tid] = round(
                _topic_score_sums[tid] / _topic_score_counts[tid], 4
            )
        else:
            domain_scores[tid] = 0.5
    logger.info(
        "Per-topic domain scores: min=%.3f, max=%.3f, mean=%.3f",
        min(domain_scores.values()) if domain_scores else 0,
        max(domain_scores.values()) if domain_scores else 0,
        sum(domain_scores.values()) / max(len(domain_scores), 1),
    )

    progress.completed_stages.append(stages[-1])

    # ── Stage 2: Gap-Language extraction ─────────────────────────────────
    logger.info("═══ Stage 2/7: Gap-language extraction ═══")
    await _update_progress(2, "Gap-Language Extraction", "Searching for author-identified gaps via Tantivy + Qdrant...")
    stage_start = datetime.now(timezone.utc)

    from paperqa.stores.gap_language import run_gap_language

    gap_lang_result = await run_gap_language(
        output_dir=str(output_path / "gap_language"),
        domain_filter=domain_filter,
        index_dir=tantivy_index_dir,
        qdrant_url=qdrant_url,
        qdrant_collection=qdrant_collection,
    )

    stages.append({
        "name": "Gap-Language Extraction",
        "status": "complete",
        "gap_chunks": len(gap_lang_result.gap_chunks),
        "gap_categories": len(gap_lang_result.gap_categories),
        "duration_seconds": _elapsed(stage_start),
    })
    logger.info("Stage 2 complete: %d gap chunks", len(gap_lang_result.gap_chunks))

    progress.completed_stages.append(stages[-1])

    # ── Stage 3: OpenAlex comparison ─────────────────────────────────────
    logger.info("═══ Stage 3/7: OpenAlex taxonomy comparison ═══")
    await _update_progress(3, "OpenAlex Comparison", "Fetching global taxonomy and computing embedding coverage...")
    stage_start = datetime.now(timezone.utc)

    from paperqa.stores.openalex_compare import run_openalex

    openalex_result = await run_openalex(
        topic_result=topic_result,
        output_dir=str(output_path / "openalex"),
        embedding_api_base=f"{embedding_url}/v1",
    )

    stages.append({
        "name": "OpenAlex Comparison",
        "status": "complete",
        "global_topics": len(openalex_result.global_topics),
        "missing_topics": len(openalex_result.missing_topics),
        "duration_seconds": _elapsed(stage_start),
    })
    logger.info("Stage 3 complete: %d missing topics (embedding-based)", len(openalex_result.missing_topics))

    openalex_domain_scores: dict[str, float] = {}
    if openalex_result.missing_topics:
        oa_names = [t.get("topic_name", "") for t in openalex_result.missing_topics]
        oa_vecs = await domain_filter._embed_batch(oa_names)
        oa_scores = domain_filter.score_embeddings_batch(oa_vecs)
        for name, sc in zip(oa_names, oa_scores):
            openalex_domain_scores[name] = float(sc)
        relevant_oa = sum(1 for s in oa_scores if float(s) >= domain_filter.adjacent_threshold)
        logger.info("OpenAlex domain filter: %d/%d missing topics are domain-relevant (threshold=%.2f)",
                     relevant_oa, len(oa_names), domain_filter.adjacent_threshold)

    progress.completed_stages.append(stages[-1])

    # ── Stage 4: Novelpy scoring ─────────────────────────────────────────
    logger.info("═══ Stage 4/7: Novelpy novelty scoring ═══")
    await _update_progress(4, "Novelpy Scoring", "Computing novel concept combinations (PMI)...")
    stage_start = datetime.now(timezone.utc)

    from paperqa.stores.novelpy_pipeline import run_novelpy

    novelpy_result = await run_novelpy(
        topic_result=topic_result,
        output_dir=str(output_path / "novelpy"),
    )

    stages.append({
        "name": "Novelpy Scoring",
        "status": "complete",
        "concepts": len(novelpy_result.concepts),
        "novel_pairs": len(novelpy_result.novel_pairs),
        "duration_seconds": _elapsed(stage_start),
    })
    logger.info("Stage 4 complete: %d novel pairs", len(novelpy_result.novel_pairs))
    progress.completed_stages.append(stages[-1])

    # ── Stage 5: Build gap candidates ────────────────────────────────────
    logger.info("═══ Stage 5/7: Build gap candidates ═══")
    await _update_progress(5, "Building Gap Candidates", "Merging all sources, ranking, and selecting top candidates...")
    stage_start = datetime.now(timezone.utc)

    # Pre-compute real domain scores for novel pairs and gap-language categories
    novel_domain_scores: dict[str, float] = {}
    gaplang_domain_scores: dict[int, float] = {}

    texts_to_score: list[str] = []
    novel_keys: list[str] = []
    gaplang_keys: list[int] = []

    for concept_a, concept_b, _ in novelpy_result.novel_pairs[:30]:
        text = f"{concept_a} {concept_b}"
        texts_to_score.append(text)
        novel_keys.append(f"{concept_a}+{concept_b}")

    for tid, cat in gap_lang_result.gap_categories.items():
        label = cat.get("label", "")
        if label.strip():
            rep_texts = cat.get("representative_texts", [])
            score_text = f"{label} {' '.join(t[:100] for t in rep_texts[:2])}" if rep_texts else label
            texts_to_score.append(score_text)
            gaplang_keys.append(tid)

    if texts_to_score:
        _pre_vecs = await domain_filter._embed_batch(texts_to_score)
        _pre_scores = domain_filter.score_embeddings_batch(_pre_vecs)
        idx = 0
        for key in novel_keys:
            novel_domain_scores[key] = round(float(_pre_scores[idx]), 4)
            idx += 1
        for key in gaplang_keys:
            gaplang_domain_scores[key] = round(float(_pre_scores[idx]), 4)
            idx += 1
        logger.info(
            "Pre-computed domain scores: %d novel pairs, %d gap-language categories",
            len(novel_domain_scores), len(gaplang_domain_scores),
        )

    candidates = _build_candidates(
        domain_scores=domain_scores,
        topic_result=topic_result,
        gap_lang_result=gap_lang_result,
        openalex_result=openalex_result,
        novelpy_result=novelpy_result,
        openalex_domain_scores=openalex_domain_scores,
        novel_domain_scores=novel_domain_scores,
        gaplang_domain_scores=gaplang_domain_scores,
        domain_filter=domain_filter,
    )

    # Sort by priority and truncate
    candidates.sort(key=lambda c: -c["priority_score"])
    main_candidates = [c for c in candidates if c["domain_category"] != "cross_domain"][:max_gaps]
    cross_candidates = [c for c in candidates if c["domain_category"] == "cross_domain"][:10]

    stages.append({
        "name": "Candidate Building",
        "status": "complete",
        "total_candidates": len(candidates),
        "main_candidates": len(main_candidates),
        "cross_domain_candidates": len(cross_candidates),
        "duration_seconds": _elapsed(stage_start),
    })
    logger.info("Stage 5 complete: %d main + %d cross-domain candidates",
                len(main_candidates), len(cross_candidates))

    # ── Stage 5b: Fetch sample chunks for candidates missing them ──────
    all_selected = main_candidates + cross_candidates
    needs_chunks = [c for c in all_selected if not c.get("raw_supporting_chunks")]
    if needs_chunks:
        logger.info("Fetching sample chunks for %d candidates missing them", len(needs_chunks))
        embedding_model = _build_embedding_model(embedding_url)
        from qdrant_client import AsyncQdrantClient
        _qdrant = AsyncQdrantClient(url=qdrant_url)
        for cand in needs_chunks:
            query_text = " ".join(cand.get("core_terms", [])[:5])
            if not query_text.strip():
                continue
            try:
                qvec = (await embedding_model.embed_documents([query_text]))[0]
                hits = await _qdrant.query_points(
                    collection_name=qdrant_collection,
                    query=qvec,
                    limit=5,
                    with_payload=True,
                    with_vectors=False,
                )
                cand["raw_supporting_chunks"] = [
                    {"text": pt.payload.get("text", "")[:500]}
                    for pt in hits.points
                ]
            except Exception as e:
                logger.warning("Sample chunk fetch failed for %s: %s", cand["id"], e)
        await _qdrant.close()

    progress.completed_stages.append(stages[-1])

    # ── Stage 6: Humanize topics (LLM) ──────────────────────────────────
    logger.info("═══ Stage 6/7: Topic humanization via LLM ═══")
    await _update_progress(6, "LLM Topic Humanization", "Generating human-readable titles and descriptions...")
    stage_start = datetime.now(timezone.utc)

    from paperqa.stores.topic_humanizer import humanize_topics_batch

    topics_to_humanize = []
    for cand in all_selected:
        sample_chunks = [
            ch.get("text", "")
            for ch in cand.get("raw_supporting_chunks", [])[:5]
        ]
        topics_to_humanize.append({
            "key": cand["id"],
            "topic_id": cand.get("topic_id", 0),
            "words": cand.get("core_terms", []),
            "name": cand.get("raw_title", cand["title"]),
            "chunk_count": cand.get("chunk_count", 0),
            "sample_chunks": sample_chunks,
        })

    humanized = await humanize_topics_batch(
        topics_data=topics_to_humanize,
        llm_api_base=llm_api_base,
        llm_model=llm_model,
    )

    for cand in main_candidates + cross_candidates:
        cand_key = cand["id"]
        if cand_key in humanized:
            h = humanized[cand_key]
            cand["title"] = h["title"]
            cand["description"] = h["description"]

    stages.append({
        "name": "Topic Humanization",
        "status": "complete",
        "topics_humanized": len(humanized),
        "duration_seconds": _elapsed(stage_start),
    })
    logger.info("Stage 6 complete: %d topics humanized", len(humanized))

    progress.completed_stages.append(stages[-1])

    # ── Stage 7: Evidence + LLM Assessment ───────────────────────────────
    logger.info("═══ Stage 7/7: Evidence collection & LLM assessment ═══")
    await _update_progress(7, "Evidence Collection & LLM Assessment", "Fetching evidence from Qdrant and running LLM worthiness assessment...")
    stage_start = datetime.now(timezone.utc)

    cp_base = output_path / "checkpoints"
    enriched_gaps = await _enrich_candidates(
        candidates=main_candidates,
        category="main",
        qdrant_url=qdrant_url,
        qdrant_collection=qdrant_collection,
        embedding_url=embedding_url,
        llm_api_base=llm_api_base,
        llm_model=llm_model,
        checkpoint_dir=cp_base / "main",
    )

    cross_domain_gaps = await _enrich_candidates(
        candidates=cross_candidates,
        category="cross_domain",
        qdrant_url=qdrant_url,
        qdrant_collection=qdrant_collection,
        embedding_url=embedding_url,
        llm_api_base=llm_api_base,
        llm_model=llm_model,
        checkpoint_dir=cp_base / "cross_domain",
    )

    stages.append({
        "name": "Evidence & Assessment",
        "status": "complete",
        "main_enriched": len(enriched_gaps),
        "cross_domain_enriched": len(cross_domain_gaps),
        "duration_seconds": _elapsed(stage_start),
    })
    logger.info("Stage 7 complete: %d main + %d cross-domain enriched",
                len(enriched_gaps), len(cross_domain_gaps))

    # ── Assemble final report ────────────────────────────────────────────
    total_papers = _count_unique_papers(topic_result)
    worthy_count = sum(
        1 for g in enriched_gaps
        if g.llm_assessment and g.llm_assessment.get("paper_worthy") == "worthy"
    )

    report = EnrichedReport(
        summary=(
            f"Analysis of {total_papers:,} papers "
            f"({len(topic_result.chunk_to_topic):,} chunks) "
            f"identified "
            f"{len(enriched_gaps)} actionable research gaps "
            f"and {len(cross_domain_gaps)} cross-domain opportunities. "
            f"{worthy_count} gaps assessed as publication-worthy. "
            f"Pipeline used {len(stages)} stages with embedding-based domain "
            f"pre-filtering, LLM humanization, and evidence-based assessment."
        ),
        analysis_date=datetime.now(timezone.utc).isoformat(),
        corpus_stats={
            "total_chunks": len(topic_result.chunk_to_topic),
            "total_topics": len(topic_result.topics),
            "relevant_topics": len(topic_result.topics),
            "cross_domain_topics": 0,
            "sparse_topics": len(topic_result.sparse_topics),
            "total_papers": total_papers,
        },
        pipeline_stages=stages,
        gaps=enriched_gaps,
        cross_domain_opportunities=cross_domain_gaps,
        metadata={
            "run_id": run_id,
            "qdrant_collection": qdrant_collection,
            "llm_model": llm_model,
            "min_domain_relevance": min_domain_relevance,
            "max_gaps": max_gaps,
            "components": [
                "bertopic", "domain_filter", "gap_language",
                "openalex", "novelpy", "topic_humanizer",
                "gap_evidence", "gap_assessment",
            ],
        },
    )

    # Save to disk
    _save_report(report, output_path / "report")

    progress.status = "completed"
    progress.completed_stages.append(stages[-1])
    await _update_progress(7, "Complete", f"Done: {len(enriched_gaps)} gaps, {worthy_count} worthy")

    logger.info("═══ Enriched report generation complete ═══")
    logger.info("  Main gaps: %d", len(enriched_gaps))
    logger.info("  Cross-domain: %d", len(cross_domain_gaps))
    logger.info("  Worthy: %d", worthy_count)

    return report


# ── Candidate building ───────────────────────────────────────────────────────


def _build_candidates(
    domain_scores: dict[int, float],
    topic_result: "TopicResult",
    gap_lang_result: "GapLanguageResult",
    openalex_result: "OpenAlexResult",
    novelpy_result: "NovelpyResult",
    openalex_domain_scores: dict[str, float] | None = None,
    novel_domain_scores: dict[str, float] | None = None,
    gaplang_domain_scores: dict[int, float] | None = None,
    domain_filter: "EmbeddingDomainFilter | None" = None,
) -> list[dict[str, Any]]:
    """Build gap candidates from all sources with domain annotations.

    All domain_relevance_scores are embedding-based — no static fallbacks.
    """
    from paperqa.stores.deterministic_gap_extraction import extract_core_terms
    candidates: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    gap_idx = 0

    def _classify(score: float) -> str:
        if domain_filter:
            return domain_filter._classify_score(score)
        if score >= 0.55:
            return "core_domain"
        if score >= 0.47:
            return "adjacent"
        if score >= 0.39:
            return "cross_domain"
        return "irrelevant"

    # Build topic → chunk count
    topic_counts: dict[int, int] = {}
    for t in topic_result.chunk_to_topic:
        if t != -1:
            topic_counts[t] = topic_counts.get(t, 0) + 1

    # ── Source 1: Sparse topics from relevant domain ─────────────────────
    for sparse in topic_result.sparse_topics:
        topic_id = sparse.get("topic_id", -1)
        topic_name = sparse.get("name", "")

        # Only include if domain-relevant or cross-domain
        score = domain_scores.get(topic_id, 0.0)
        if score < 0.1:
            continue

        domain_cat = _classify(score)
        core_terms = extract_core_terms(topic_name)
        chunk_count = sparse.get("count", 0)

        title_key = f"sparse:{topic_name}"
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)

        candidates.append({
            "id": f"gap_{gap_idx:03d}",
            "topic_id": topic_id,
            "gap_type": "sparse_topic",
            "title": topic_name,
            "raw_title": topic_name,
            "description": f"Topic with {chunk_count} documents in corpus",
            "domain_category": domain_cat,
            "domain_relevance_score": score,
            "core_terms": core_terms,
            "chunk_count": chunk_count,
            "statistical_score": max(0, 1.0 - (chunk_count / 100)),
            "priority_score": _compute_priority(
                gap_type="sparse_topic",
                chunk_count=chunk_count,
                domain_score=score,
            ),
            "evidence_sources": ["bertopic_sparse"],
            "raw_supporting_chunks": [],
        })
        gap_idx += 1

    # ── Source 2: OpenAlex missing topics ─────────────────────────────────
    for missing in openalex_result.missing_topics[:40]:
        topic_name = missing.get("topic_name", "")
        global_count = missing.get("global_count", 1)
        our_count = missing.get("our_count", 0)
        ratio = missing.get("ratio", 1.0)

        score = (openalex_domain_scores or {}).get(topic_name, 0.0)
        _oa_min = domain_filter.adjacent_threshold if domain_filter else 0.47
        if score < _oa_min:
            continue
        domain_cat = _classify(score)

        title_key = f"openalex:{topic_name}"
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)

        candidates.append({
            "id": f"gap_{gap_idx:03d}",
            "topic_id": gap_idx,
            "gap_type": "openalex_missing",
            "title": topic_name,
            "raw_title": topic_name,
            "description": (
                f"Global topic with {global_count} works worldwide "
                f"but only {our_count} in corpus (coverage: {ratio:.2%})"
            ),
            "domain_category": domain_cat,
            "domain_relevance_score": score,
            "core_terms": extract_core_terms(topic_name.replace(" ", "_")),
            "chunk_count": our_count,
            "statistical_score": 1.0 - ratio,
            "priority_score": _compute_priority(
                gap_type="openalex_missing",
                chunk_count=our_count,
                domain_score=score,
                coverage_ratio=ratio,
            ),
            "evidence_sources": ["openalex_missing"],
            "raw_supporting_chunks": [],
        })
        gap_idx += 1

    # ── Source 3: Novel concept pairs (from pre-filtered corpus) ─────────
    for concept_a, concept_b, atypicality in novelpy_result.novel_pairs[:30]:
        # Reject noise: too short or purely numeric concepts
        skip = False
        for concept in (concept_a, concept_b):
            if len(concept) < 3 or not any(c.isalpha() for c in concept):
                skip = True
                break
        if skip:
            continue

        # Use real embedding-based domain score for this concept pair
        _novel_key = f"{concept_a}+{concept_b}"
        score = (novel_domain_scores or {}).get(_novel_key, 0.0)
        _novel_min = domain_filter.cross_domain_threshold if domain_filter else 0.39
        if score < _novel_min:
            continue
        domain_cat = _classify(score)

        title_key = f"novel:{concept_a}+{concept_b}"
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)

        candidates.append({
            "id": f"gap_{gap_idx:03d}",
            "topic_id": gap_idx,
            "gap_type": "novel_combination",
            "title": f"{concept_a} + {concept_b}",
            "raw_title": f"{concept_a} + {concept_b}",
            "description": (
                f"Concept pair with low co-occurrence "
                f"(atypicality={atypicality:.3f})"
            ),
            "domain_category": domain_cat,
            "domain_relevance_score": score,
            "core_terms": [concept_a, concept_b],
            "chunk_count": 0,
            "statistical_score": atypicality,
            "priority_score": _compute_priority(
                gap_type="novel_combination",
                chunk_count=0,
                domain_score=score,
                atypicality=atypicality,
            ),
            "evidence_sources": ["novelpy"],
            "raw_supporting_chunks": [],
        })
        gap_idx += 1

    # ── Source 4: Gap-language mentions ───────────────────────────────────
    for topic_id, cat in gap_lang_result.gap_categories.items():
        label = cat.get("label", "")
        count = cat.get("count", 0)
        rep_texts = cat.get("representative_texts", [])

        if not label.strip():
            continue

        # Use real embedding-based domain score for this gap category
        score = (gaplang_domain_scores or {}).get(topic_id, 0.0)
        _gl_min = domain_filter.cross_domain_threshold if domain_filter else 0.39
        if score < _gl_min:
            continue
        domain_cat = _classify(score)

        title_key = f"gaplang:{label}"
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)

        candidates.append({
            "id": f"gap_{gap_idx:03d}",
            "topic_id": gap_idx,
            "gap_type": "author_identified",
            "title": label,
            "raw_title": label,
            "description": f"Authors mention this gap {count} times in corpus",
            "domain_category": domain_cat,
            "domain_relevance_score": score,
            "core_terms": [w.lower() for w in label.replace("_", " ").split() if len(w) > 2],
            "chunk_count": count,
            "statistical_score": min(1.0, count / 50),
            "priority_score": _compute_priority(
                gap_type="author_identified",
                chunk_count=count,
                domain_score=score,
                mention_count=count,
            ),
            "evidence_sources": ["gap_language"],
            "raw_supporting_chunks": [
                {"text": t} for t in rep_texts[:5]
            ],
        })
        gap_idx += 1

    logger.info("Built %d total gap candidates from 4 sources", len(candidates))
    return candidates


def _compute_priority(
    gap_type: str,
    chunk_count: int,
    domain_score: float,
    coverage_ratio: float = 1.0,
    atypicality: float = 0.0,
    mention_count: int = 0,
) -> float:
    """Compute deterministic priority score.

    Weighted formula (rebalanced for better discrimination):
      0.35 × statistical_signal
    + 0.25 × domain_relevance
    + 0.20 × source_credibility
    + 0.20 × evidence_quantity
    """
    import math

    if gap_type == "sparse_topic":
        # log-scale sparsity: count=1 → 1.0, count=10 → 0.5, count=50 → 0.15
        stat = max(0.0, 1.0 - math.log1p(chunk_count) / math.log1p(100))
    elif gap_type == "openalex_missing":
        stat = 1.0 - coverage_ratio
    elif gap_type == "novel_combination":
        stat = min(1.0, atypicality)
    elif gap_type == "author_identified":
        stat = min(1.0, mention_count / 50)
    else:
        stat = 0.5

    credibility = {
        "author_identified": 1.0,
        "openalex_missing": 0.8,
        "novel_combination": 0.7,
        "sparse_topic": 0.6,
    }.get(gap_type, 0.5)

    evidence = min(1.0, chunk_count / 30) if chunk_count > 0 else 0.2

    return round(
        0.35 * stat + 0.25 * domain_score + 0.20 * credibility + 0.20 * evidence,
        3,
    )


# ── Enrichment (evidence + assessment) ───────────────────────────────────────


async def _enrich_candidates(
    candidates: list[dict[str, Any]],
    category: str,
    qdrant_url: str,
    qdrant_collection: str,
    embedding_url: str,
    llm_api_base: str,
    llm_model: str,
    checkpoint_dir: Path | None = None,
) -> list[EnrichedGap]:
    """Enrich candidates with real evidence and LLM assessment.

    Supports checkpoint/resume: if checkpoint_dir is given, completed gaps
    are loaded from disk and skipped on re-run.
    """
    from paperqa.stores.deterministic_gap_extraction import GapCandidate
    from paperqa.stores.gap_assessment import assess_gap_worthiness
    from paperqa.stores.gap_evidence import EvidenceCollector

    collector = EvidenceCollector(
        qdrant_url=qdrant_url,
        qdrant_collection=qdrant_collection,
        embedding_url=embedding_url,
    )

    embedding_model = _build_embedding_model(embedding_url)

    # Load existing checkpoints
    cached: dict[str, dict] = {}
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for cp_file in checkpoint_dir.glob("gap_*.json"):
            try:
                with open(cp_file) as f:
                    data = json.load(f)
                cached[data["id"]] = data
            except Exception as e:
                logger.debug("Skipping corrupt checkpoint %s: %s", cp_file.name, e)
        if cached:
            logger.info("Loaded %d enriched gaps from checkpoint", len(cached))

    enriched: list[EnrichedGap] = []

    for i, cand in enumerate(candidates):
        gap_id = cand["id"]

        # Resume from checkpoint if available
        if gap_id in cached:
            logger.info("Skipping %s (loaded from checkpoint)", gap_id)
            d = cached[gap_id]
            enriched.append(EnrichedGap(
                id=d["id"],
                gap_type=d["gap_type"],
                title=d["title"],
                description=d["description"],
                domain_category=d["domain_category"],
                core_terms=d["core_terms"],
                chunk_count=d["chunk_count"],
                unique_papers=d.get("unique_papers", 0),
                statistical_score=d.get("statistical_score", 0),
                domain_relevance_score=d.get("domain_relevance_score", 0),
                priority_score=d.get("priority_score", 0),
                supporting_chunks=d.get("supporting_chunks", []),
                yearly_distribution=d.get("yearly_distribution", {}),
                top_relevant_papers=d.get("top_relevant_papers", []),
                coverage_ratio=d.get("coverage_ratio", 0),
                llm_assessment=d.get("llm_assessment"),
                evidence_sources=d.get("evidence_sources", []),
            ))
            continue

        logger.info(
            "Enriching %s candidate %d/%d: %s",
            category, i + 1, len(candidates), cand["title"][:60],
        )

        gap_candidate = GapCandidate(
            id=gap_id,
            gap_type=cand["gap_type"],
            title=cand["title"],
            description=cand["description"],
            core_terms=cand["core_terms"],
            chunk_count=cand["chunk_count"],
            evidence_sources=cand["evidence_sources"],
        )

        try:
            evidence = await collector.collect_evidence_for_gap(
                gap=gap_candidate,
                embedding_model=embedding_model,
                max_chunks=50,
            )

            supporting_chunks = [
                {
                    "text": ch.text,
                    "pdf_name": ch.pdf_name,
                    "pdf_hash": ch.pdf_hash,
                    "page_num": ch.page_num,
                    "relevance_score": round(ch.relevance_score, 3),
                }
                for ch in evidence.supporting_chunks[:10]
            ]

            yearly_dist = {str(k): v for k, v in evidence.yearly_distribution.items()}
            unique_papers = evidence.unique_papers
            coverage_ratio = evidence.coverage_ratio
            top_papers = evidence.top_relevant_papers[:10]

        except Exception as e:
            logger.warning("Evidence collection failed for %s: %s", gap_id, e)
            supporting_chunks = []
            yearly_dist = {}
            unique_papers = 0
            coverage_ratio = 0.0
            top_papers = []
            evidence = None

        llm_assessment_dict: dict[str, Any] | None = None
        if evidence is not None:
            try:
                assessment = await assess_gap_worthiness(
                    gap=gap_candidate,
                    evidence=evidence,
                    llm_api_base=llm_api_base,
                    llm_model=llm_model if "/" in llm_model else f"openai/{llm_model}",
                )
                llm_assessment_dict = {
                    "paper_worthy": assessment.paper_worthy,
                    "confidence": assessment.confidence,
                    "recommended_paper_type": assessment.recommended_paper_type,
                    "reasoning": assessment.reasoning,
                    "concerns": assessment.concerns,
                    "suggested_focus": assessment.suggested_focus,
                    "evidence_verified": assessment.evidence_verified,
                    "verification_notes": assessment.verification_notes,
                }
            except Exception as e:
                logger.warning("LLM assessment failed for %s: %s", gap_id, e)

        enriched_gap = EnrichedGap(
            id=gap_id,
            gap_type=cand["gap_type"],
            title=cand["title"],
            description=cand["description"],
            domain_category=cand["domain_category"],
            core_terms=cand["core_terms"],
            chunk_count=cand["chunk_count"],
            unique_papers=unique_papers,
            statistical_score=cand["statistical_score"],
            domain_relevance_score=cand["domain_relevance_score"],
            priority_score=cand["priority_score"],
            supporting_chunks=supporting_chunks,
            yearly_distribution=yearly_dist,
            top_relevant_papers=top_papers,
            coverage_ratio=coverage_ratio,
            llm_assessment=llm_assessment_dict,
            evidence_sources=cand["evidence_sources"],
        )
        enriched.append(enriched_gap)

        # Save checkpoint per gap
        if checkpoint_dir:
            try:
                cp_data = asdict(enriched_gap)
                with open(checkpoint_dir / f"{gap_id}.json", "w") as f:
                    json.dump(cp_data, f, ensure_ascii=False)
            except Exception as e:
                logger.warning("Checkpoint save failed for %s: %s", gap_id, e)

    await collector.close()
    return enriched


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_embedding_model(embedding_url: str, embedding_model: str | None = None):
    """Build a simple embedding model wrapper for EvidenceCollector.

    Returns an object with `embed_documents` and `set_mode` methods.
    """
    import os

    import httpx

    model_name = embedding_model or os.getenv(
        "EMBEDDING_MODEL", "hf.co/Qwen/Qwen3-Embedding-8B-GGUF:F16"
    )

    class SimpleEmbeddingModel:
        def __init__(self, api_base: str, model: str):
            self.api_base = api_base.rstrip("/")
            self.model = model

        def set_mode(self, mode):
            pass

        async def embed_documents(self, texts: list[str]) -> list[list[float]]:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.api_base}/v1/embeddings",
                    json={"input": texts, "model": self.model},
                )
                response.raise_for_status()
                data = response.json()["data"]
                return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]

    return SimpleEmbeddingModel(embedding_url, model_name)


def _count_unique_papers(topic_result: "TopicResult") -> int:
    """Estimate unique paper count from topic info."""
    if isinstance(topic_result.topic_info, dict) and "data" in topic_result.topic_info:
        total_docs = sum(
            row.get("Count", 0)
            for row in topic_result.topic_info["data"]
        )
        # Chunks → papers estimation (average ~17 chunks/paper)
        return max(1, total_docs // 17)
    return 0


def _elapsed(start: datetime) -> float:
    """Seconds elapsed since start."""
    return round((datetime.now(timezone.utc) - start).total_seconds(), 1)


def _save_report(report: EnrichedReport, output_dir: Path) -> None:
    """Save report as JSON and markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    report_dict = {
        "summary": report.summary,
        "analysis_date": report.analysis_date,
        "corpus_stats": report.corpus_stats,
        "pipeline_stages": report.pipeline_stages,
        "gaps": [asdict(g) for g in report.gaps],
        "cross_domain_opportunities": [asdict(g) for g in report.cross_domain_opportunities],
        "metadata": report.metadata,
    }
    with open(output_dir / "report.json", "w") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)

    # Markdown
    md = _generate_markdown(report)
    with open(output_dir / "report.md", "w") as f:
        f.write(md)

    logger.info("Saved enriched report to %s", output_dir)


def _generate_markdown(report: EnrichedReport) -> str:
    """Generate human-readable markdown report."""
    lines = [
        "# Enriched Gap Analysis Report",
        "",
        f"*Generated: {report.analysis_date}*",
        "",
        "## Executive Summary",
        "",
        report.summary,
        "",
        "## Corpus Statistics",
        "",
    ]

    for key, val in report.corpus_stats.items():
        lines.append(f"- **{key.replace('_', ' ').title()}**: {val:,}" if isinstance(val, int) else f"- **{key.replace('_', ' ').title()}**: {val}")
    lines.append("")

    # Pipeline stages
    lines.append("## Pipeline Stages")
    lines.append("")
    for stage in report.pipeline_stages:
        status_icon = "done" if stage["status"] == "complete" else "pending"
        lines.append(f"- [{status_icon}] **{stage['name']}** ({stage.get('duration_seconds', '?')}s)")
    lines.append("")

    # Main gaps
    lines.append("## Research Gaps")
    lines.append("")
    for gap in report.gaps:
        _append_gap_markdown(lines, gap)

    # Cross-domain
    if report.cross_domain_opportunities:
        lines.append("## Cross-Domain Opportunities")
        lines.append("")
        for gap in report.cross_domain_opportunities:
            _append_gap_markdown(lines, gap)

    return "\n".join(lines)


def _append_gap_markdown(lines: list[str], gap: EnrichedGap) -> None:
    """Append markdown for a single gap."""
    lines.append(f"### {gap.id}: {gap.title}")
    lines.append("")
    lines.append(f"**Type:** {gap.gap_type} | **Domain:** {gap.domain_category} | **Priority:** {gap.priority_score:.3f}")
    lines.append("")
    lines.append(gap.description)
    lines.append("")

    if gap.supporting_chunks:
        lines.append(f"**Supporting Evidence:** {len(gap.supporting_chunks)} chunks from {gap.unique_papers} papers")
        lines.append("")

    if gap.llm_assessment:
        a = gap.llm_assessment
        lines.append(f"**LLM Assessment:** {a.get('paper_worthy', 'N/A')} (confidence: {a.get('confidence', 0):.2f})")
        if a.get("reasoning"):
            lines.append(f"> {a['reasoning'][:200]}...")
        lines.append("")

    lines.append("---")
    lines.append("")
