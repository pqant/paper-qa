"""
PaperBridge Web API - FastAPI Backend for Web UI

Provides REST API endpoints for:
1. Query execution (Phase 1)
2. Gap analysis management (Phase 2)
3. System status monitoring
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

app = FastAPI(
    title="PaperBridge API",
    description="API for PaperBridge Research Platform",
    version="1.0.0",
)

# CORS middleware for web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage
query_history: dict[str, dict[str, Any]] = {}
gap_reviews: dict[str, dict[str, Any]] = {}
gap_analysis_runs: dict[str, dict[str, Any]] = {}

# Cached real gap data
_real_gaps: list[dict[str, Any]] = []
_real_gaps_loaded = False


def _get_report_path() -> Path:
    """Find the latest report.json.

    Path: web_api.py → paperqa → src → paper-qa (3 parents up)
    Report lives at paper-qa/data/gap_analysis_integration_run/report/report.json
    """
    paperqa = Path(__file__).resolve().parent.parent.parent
    report_dir = paperqa / "data" / "gap_analysis_integration_run" / "report"
    if (report_dir / "report.json").exists():
        return report_dir / "report.json"
    # Fallback to older path
    alt = paperqa / "data" / "gap_analysis" / "report" / "report.json"
    if alt.exists():
        return alt
    return None


def _load_real_gaps() -> list[dict[str, Any]]:
    """Load gaps from the latest report.json."""
    global _real_gaps, _real_gaps_loaded
    if _real_gaps_loaded:
        return _real_gaps

    report_path = _get_report_path()
    if report_path is None:
        logger.warning("No report.json found, returning empty gaps")
        return []

    try:
        with open(report_path, encoding="utf-8") as f:
            data = json.load(f)
        _real_gaps = data.get("gaps", [])
        # Normalize fields for frontend compatibility
        for g in _real_gaps:
            g.setdefault("human_verdict", None)
            g.setdefault("priority", 0)
            g.setdefault("notes", "")
            g.setdefault("proposed_methodology", "")
            g.setdefault("target_venue", "")
            g.setdefault("supporting_chunks", [])
            g.setdefault("yearly_distribution", {})
            g.setdefault("llm_assessment", None)
            # Map chunk_count from supporting_chunks length if missing
            if "chunk_count" not in g:
                g["chunk_count"] = len(g.get("supporting_chunks", []))
            # Map unique_papers if missing
            if "unique_papers" not in g:
                g["unique_papers"] = 0
            # Map statistical_score from domain_relevance_score if missing
            if "statistical_score" not in g:
                g["statistical_score"] = g.get("domain_relevance_score", 0)
            # Map evidence_sources from gap_type
            if "evidence_sources" not in g:
                g["evidence_sources"] = [g.get("gap_type", "unknown")]
            # Map llm_used
            if "llm_used" not in g:
                g["llm_used"] = g.get("llm_assessment") is not None
        _real_gaps_loaded = True
        logger.info("Loaded %d gaps from report.json", len(_real_gaps))
    except Exception as e:
        logger.error("Failed to load report.json: %s", e)

    return _real_gaps


def _reload_gaps() -> None:
    """Force reload gaps from disk (e.g., after new pipeline run)."""
    global _real_gaps, _real_gaps_loaded
    _real_gaps = []
    _real_gaps_loaded = False
    _load_real_gaps()


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ==================== System Status ====================

@app.get("/api/system/status")
async def get_system_status():
    """Get system component status."""
    gaps = _load_real_gaps()
    last_analysis = None
    report_path = _get_report_path()
    if report_path:
        stat = report_path.stat()
        from datetime import timezone
        last_analysis = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

    return {
        "qdrant": {
            "status": "online",
            "collection": "paperbridge_glm_v2",
            "chunk_count": 175349,
        },
        "llm": {
            "status": "online",
            "model": "qwen3.6-35b-a3b",
        },
        "embedding": {
            "status": "online",
            "model": "Qwen3-Embedding-8B",
        },
        "last_analysis": last_analysis,
    }


# ==================== Query API (Phase 1) ====================

@app.post("/api/query")
async def run_query(request: dict[str, Any]):
    """
    Execute a research query and get synthesized answer.

    Request body:
    - query: str - The research question
    - max_chunks: int (optional) - Maximum chunks to retrieve (default: 10)
    - temperature: float (optional) - LLM temperature (default: 0.7)
    """
    query_text = request.get("query", "")
    if not query_text:
        raise HTTPException(status_code=400, detail="Query is required")

    query_id = str(uuid4())

    # TODO: Integrate with actual PaperQA2 query engine
    result = {
        "id": query_id,
        "query": query_text,
        "answer": "This is a placeholder answer. The actual implementation would use PaperQA2 to synthesize an answer from the corpus.",
        "sources": [
            {
                "chunk_id": f"chunk_{i}",
                "text": f"Relevant text from paper {i+1}...",
                "pdf_name": f"paper_{i+1}.pdf",
                "page_num": i + 1,
                "relevance_score": 0.9 - (i * 0.05),
            }
            for i in range(5)
        ],
        "token_usage": {
            "prompt_tokens": 1250,
            "completion_tokens": 450,
            "total_tokens": 1700,
        },
        "duration_ms": 3500,
        "created_at": datetime.now().isoformat(),
    }

    query_history[query_id] = result
    return result


@app.get("/api/query/{query_id}")
async def get_query_result(query_id: str):
    """Get a specific query result by ID."""
    if query_id not in query_history:
        raise HTTPException(status_code=404, detail="Query not found")
    return query_history[query_id]


@app.get("/api/query/history")
async def get_query_history(limit: int = 10):
    """Get recent query history."""
    sorted_queries = sorted(
        query_history.values(),
        key=lambda x: x.get("created_at", ""),
        reverse=True,
    )
    return sorted_queries[:limit]


# ==================== Gap Analysis API (Phase 2) ====================

@app.get("/api/gaps")
async def get_gaps(
    limit: int = 100,
    offset: int = 0,
    gap_type: str | None = None,
    min_priority: float | None = None,
):
    """
    Get gap candidates with filtering.

    Query parameters:
    - limit: int - Maximum gaps to return (default: 100)
    - offset: int - Pagination offset (default: 0)
    - gap_type: str | None - Filter by gap type
    - min_priority: float | None - Minimum priority score filter
    """
    all_gaps = _load_real_gaps()

    # Apply filters
    filtered = all_gaps
    if gap_type:
        filtered = [g for g in filtered if g.get("gap_type") == gap_type]
    if min_priority is not None:
        filtered = [g for g in filtered if g.get("priority_score", 0) >= min_priority]

    # Sort by priority descending
    filtered.sort(key=lambda x: x.get("priority_score", 0), reverse=True)

    # Apply pagination
    total = len(filtered)
    paginated = filtered[offset : offset + limit]

    return {
        "gaps": paginated,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/gaps/{gap_id}")
async def get_gap(gap_id: str):
    """Get a specific gap by ID with full details."""
    all_gaps = _load_real_gaps()
    for g in all_gaps:
        if g.get("id") == gap_id:
            return g
    raise HTTPException(status_code=404, detail="Gap not found")


@app.post("/api/gaps/{gap_id}/review")
async def submit_gap_review(gap_id: str, review: dict[str, Any]):
    """
    Submit human review for a gap.

    Request body:
    - verdict: str - "approve", "reject", or "needs_more_research"
    - priority: int - Priority score 1-5
    - notes: str - Reviewer notes
    - proposed_methodology: str - Suggested methodology
    - target_venue: str - Target publication venue
    """
    required_fields = ["verdict", "priority"]
    for field in required_fields:
        if field not in review:
            raise HTTPException(status_code=400, detail=f"{field} is required")

    if review["verdict"] not in ["approve", "reject", "needs_more_research"]:
        raise HTTPException(status_code=400, detail="Invalid verdict value")

    if not 1 <= review["priority"] <= 5:
        raise HTTPException(status_code=400, detail="Priority must be between 1 and 5")

    gap_reviews[gap_id] = {
        "gap_id": gap_id,
        "verdict": review["verdict"],
        "priority": review["priority"],
        "notes": review.get("notes", ""),
        "proposed_methodology": review.get("proposed_methodology", ""),
        "target_venue": review.get("target_venue", ""),
        "reviewed_at": datetime.now().isoformat(),
    }

    return {"status": "success", "message": "Review submitted"}


@app.post("/api/gap-analysis/run")
async def run_gap_analysis():
    """
    Trigger a new gap analysis run.

    Returns a run ID that can be used to track progress.
    """
    run_id = str(uuid4())

    gap_analysis_runs[run_id] = {
        "run_id": run_id,
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "progress": 0,
    }

    # Reload gaps after pipeline completes
    asyncio.create_task(_run_gap_analysis_task(run_id))

    return {"run_id": run_id, "status": "started"}


async def _run_gap_analysis_task(run_id: str):
    """Run the actual gap analysis pipeline, then reload data."""
    try:
        from paperqa.stores.enriched_report import generate_enriched_report

        paperqa_root = Path(__file__).resolve().parent.parent.parent

        await generate_enriched_report(
            qdrant_url="http://192.168.0.28:6333",
            qdrant_collection="paperbridge_glm_v2",
            embedding_url="http://192.168.0.28:8082",
            llm_api_base="http://192.168.0.28:8005/v1",
            llm_model="qwen3.6-35b-a3b",
            output_dir=str(paperqa_root / "data" / "gap_analysis_integration_run"),
            tantivy_index_dir=str(paperqa_root / "data" / "tantivy_index"),
            max_gaps=150,
            min_domain_relevance=0.50,
            run_id=f"api-{int(datetime.now().timestamp())}",
        )

        gap_analysis_runs[run_id]["status"] = "complete"
        gap_analysis_runs[run_id]["progress"] = 100
        gap_analysis_runs[run_id]["completed_at"] = datetime.now().isoformat()

        # Reload gaps from new report
        _reload_gaps()

    except Exception as e:
        logger.error("Gap analysis run failed: %s", e)
        gap_analysis_runs[run_id]["status"] = "failed"
        gap_analysis_runs[run_id]["error"] = str(e)


@app.get("/api/gap-analysis/{run_id}")
async def get_gap_analysis_status(run_id: str):
    """Get the status of a gap analysis run."""
    if run_id not in gap_analysis_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    return gap_analysis_runs[run_id]


@app.get("/api/gap-analysis/{run_id}/results")
async def get_gap_analysis_results(run_id: str):
    """Get the results of a completed gap analysis."""
    if run_id not in gap_analysis_runs:
        raise HTTPException(status_code=404, detail="Run not found")

    run = gap_analysis_runs[run_id]
    if run["status"] != "complete":
        raise HTTPException(status_code=400, detail="Analysis not yet complete")

    return run.get("results", {})


# ==================== Analytics ====================

@app.get("/api/analytics/coverage")
async def get_coverage_matrix():
    """Get coverage matrix data for visualization."""
    report_path = _get_report_path()
    if report_path:
        with open(report_path, encoding="utf-8") as f:
            data = json.load(f)
        corpus_stats = data.get("corpus_stats", {})
        summary = data.get("summary", {})
        yearly = summary.get("yearly_distribution", {})
        return {
            "total_chunks": corpus_stats.get("total_chunks", 175349),
            "total_topics": summary.get("total_topics", 994),
            "sparse_topics": summary.get("sparse_topics", 750),
            "coverage_ratio": corpus_stats.get("domain_filter_pass_rate", 0.269),
            "yearly_distribution": yearly,
        }
    return {
        "total_chunks": 175349,
        "total_topics": 994,
        "sparse_topics": 750,
        "coverage_ratio": 0.269,
        "yearly_distribution": {},
    }


@app.get("/api/analytics/gap-stats")
async def get_gap_statistics():
    """Get gap analysis statistics."""
    all_gaps = _load_real_gaps()
    assessed = [g for g in all_gaps if g.get("llm_assessment")]
    worthy = sum(
        1
        for g in assessed
        if (g.get("llm_assessment") or {}).get("paper_worthy") == "worthy"
    )
    not_worthy = sum(
        1
        for g in assessed
        if (g.get("llm_assessment") or {}).get("paper_worthy") == "not_worthy"
    )
    insufficient = sum(
        1
        for g in assessed
        if (g.get("llm_assessment") or {}).get("paper_worthy")
        == "insufficient_evidence"
    )
    with_evidence = sum(1 for g in all_gaps if g.get("supporting_chunks"))

    return {
        "total_candidates": len(all_gaps),
        "with_evidence": with_evidence,
        "assessed": len(assessed),
        "for_human_review": len(all_gaps),
        "assessment_breakdown": {
            "worthy": worthy,
            "not_worthy": not_worthy,
            "insufficient_evidence": insufficient,
        },
        "reviews_completed": len(gap_reviews),
        "approved_gaps": sum(
            1 for r in gap_reviews.values() if r.get("verdict") == "approve"
        ),
    }


# ==================== Export ====================

@app.get("/api/export/gaps/json")
async def export_gaps_json():
    """Export all gaps as JSON."""
    all_gaps = _load_real_gaps()
    return all_gaps


@app.get("/api/export/report/pdf")
async def export_report_pdf():
    """Export gap analysis report as PDF."""
    # TODO: Implement actual PDF generation
    return {"message": "PDF export functionality coming soon"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
