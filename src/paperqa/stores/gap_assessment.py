"""
Gap Paper Worthiness Assessment

Uses LLM to assess whether a gap candidate has potential for paper publication.
CRITICAL: LLM provides RECOMMENDATION only, NOT final decisions.
All assessments are verified against concrete evidence.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PaperWorthinessAssessment:
    """LLM's assessment of a gap's paper potential."""

    # Assessment result
    paper_worthy: str  # "worthy", "not_worthy", "insufficient_evidence"

    # Confidence in the assessment (0.0-1.0)
    confidence: float = 0.0

    # Recommended paper type
    recommended_paper_type: str | None = None  # "survey", "algorithm", "application", "theory"

    # LLM's reasoning (must cite specific evidence)
    reasoning: str = ""

    # Potential concerns
    concerns: list[str] = None

    # Suggested focus for the paper
    suggested_focus: str | None = None

    # Gap explanation — what's actually missing in the literature
    gap_explanation: str | None = None

    # Verification status
    evidence_verified: bool = False
    verification_notes: str | None = None

    def __post_init__(self):
        if self.concerns is None:
            self.concerns = []


async def assess_gap_worthiness(
    gap: "GapCandidate",  # Type hint as string to avoid circular import
    evidence: "DetailedEvidence",
    llm_api_base: str,
    llm_model: str = "",
) -> PaperWorthinessAssessment:
    """
    Assess whether a gap candidate has potential for paper publication.

    CRITICAL DESIGN DECISIONS:
    1. LLM provides RECOMMENDATION only, NOT final decisions
    2. Assessment is based ONLY on provided evidence
    3. LLM must CITE specific evidence in reasoning
    4. Assessment is verified against evidence constraints

    Args:
        gap: The gap candidate to assess
        evidence: Detailed evidence collected for this gap
        llm_api_base: LLM server endpoint
        llm_model: Model to use for assessment

    Returns:
        PaperWorthinessAssessment with LLM's recommendation
    """
    if not llm_model:
        import os
        llm_model = os.getenv("LLM_MODEL", "qwen3.6-35b-a3b")
    if "/" not in llm_model:
        llm_model = f"openai/{llm_model}"

    prompt = _build_assessment_prompt(gap, evidence)

    try:
        response = await _call_llm(
            prompt=prompt,
            llm_api_base=llm_api_base,
            llm_model=llm_model,
            temperature=0.1,  # Low temperature for consistency
            max_tokens=2000,
        )

        # Parse response — LLM may wrap JSON in markdown fences or preamble
        assessment_data = _extract_json(response)
        if assessment_data is None:
            raise ValueError(f"Could not extract JSON from LLM response ({len(response)} chars)")

        assessment = PaperWorthinessAssessment(
            paper_worthy=assessment_data.get("paper_worthy", "insufficient_evidence"),
            confidence=assessment_data.get("confidence", 0.0),
            recommended_paper_type=assessment_data.get("recommended_paper_type"),
            reasoning=assessment_data.get("reasoning", ""),
            concerns=assessment_data.get("concerns", []),
            suggested_focus=assessment_data.get("suggested_focus"),
            gap_explanation=assessment_data.get("gap_explanation"),
        )

        # Verify assessment against evidence (passes gap for domain check)
        _verify_assessment_against_evidence(assessment, evidence, gap)

        return assessment

    except Exception as e:
        logger.warning(f"LLM assessment failed for gap '{gap.title}': {e}")
        # Return conservative assessment on failure
        return PaperWorthinessAssessment(
            paper_worthy="insufficient_evidence",
            confidence=0.0,
            reasoning=f"Assessment failed: {str(e)}",
            evidence_verified=False,
            verification_notes="LLM call failed - manual review required",
        )


def _build_assessment_prompt(
    gap: "GapCandidate",
    evidence: "DetailedEvidence",
) -> str:
    """Build the LLM assessment prompt."""

    # Format evidence summary — show ALL supporting chunks (up to 15) for real differentiation
    evidence_summary = f"""
EVIDENCE SUMMARY:
- Gap Type: {gap.gap_type}
- Supporting Chunks: {evidence.total_chunks}
- Unique Papers: {evidence.unique_papers}
- Coverage Ratio: {evidence.coverage_ratio:.4f}
- Year Span: {min(evidence.unique_years) if evidence.unique_years else 'N/A'}-{max(evidence.unique_years) if evidence.unique_years else 'N/A'}
- Yearly Distribution: {evidence.yearly_distribution}

TOP RELEVANT PAPERS (by highest chunk score):
"""

    for i, paper in enumerate(evidence.top_relevant_papers[:10], 1):
        evidence_summary += f"\n{i}. [{paper.get('pdf_hash', '?')[:12]}...] score={paper.get('relevance_score', 0):.3f}"

    evidence_summary += "\n\nSAMPLE EVIDENCE CHUNKS (top 15 by relevance):\n"

    for i, chunk in enumerate(evidence.supporting_chunks[:15], 1):
        evidence_summary += f"\n[{i}] score={chunk.relevance_score:.3f} | {chunk.pdf_name[:30]} p.{chunk.page_num}\n    {chunk.text[:300]}...\n"

    if not evidence.supporting_chunks:
        evidence_summary += "\nNo supporting chunks found in corpus.\n"

    # Build domain relevance check
    domain_terms = get_domain_name().lower()
    core_terms_str = " ".join(gap.core_terms).lower()

    # Build the full prompt with explicit scoring rubric and calibration examples
    prompt = f"""You are a senior academic reviewer at a top-tier conference (GECCO, ALIOR, PACK) specializing in {get_domain_name()}.

CRITICAL: Your assessment MUST be about {get_domain_name()}. If the gap is about a different field (electricity, drones, biology, etc), mark it "not_worthy" with confidence 0.9.

Evaluate this research gap for paper publication potential.

GAP INFORMATION:
Type: {gap.gap_type}
Title: {gap.title}
Description: {gap.description}
Core Terms: {gap.core_terms}

{evidence_summary}

---

CALIBRATION EXAMPLES (use these to calibrate your scoring):

Example A: "3D EBP with item rotation and multi-orientation" — 32 papers, 2015-2025, all chunks discuss rotation constraints
→ worthy, confidence=0.92, reasoning="Large corpus, specific problem, active research"

Example B: "Genetic algorithms for VRP" — 12 papers, 2010-2018, chunks mention VRP but not loading constraints
→ not_worthy, confidence=0.80, reasoning="Already well-studied area, no clear gap, stale"

Example C: "Quantum computing for logistics" — 4 papers, 2022-2024, chunks are too theoretical
→ insufficient_evidence, confidence=0.60, reasoning="Only 4 papers, too speculative"

Example D: "UAV energy optimization" — 25 papers, 2018-2024, excellent evidence but WRONG DOMAIN
→ not_worthy, confidence=0.95, reasoning="Excellent topic but not bin packing/container loading"

---

SCORING RUBRIC — calibrate using these tiers:

| Confidence | Meaning | When to use |
|------------|---------|-------------|
| 0.90-1.00  | Excellent | >=20 papers, 5+ year span, VERY specific problem (e.g., "3D EBP with rotation + stacking") |
| 0.80-0.89  | Very good | >=15 papers, specific problem, recent activity |
| 0.70-0.79  | Good | >=10 papers, specific problem, in-domain. This is the MOST COMMON tier for worthy gaps. |
| 0.55-0.69  | Marginal | 5-9 papers, OR problem too broad, OR already partially addressed |
| 0.40-0.54  | Weak | <5 papers, too vague |
| 0.00-0.39  | Not viable | Off-domain, no evidence, already solved |

PAPER_WORTHY DECISION:
- "worthy" -> confidence >= 0.70 AND >= 10 unique papers AND directly about {get_domain_name()}
- "not_worthy" -> wrong domain (confidence=0.95), already well-studied with no gap, or problem too narrow/solved
- "insufficient_evidence" -> < 10 papers, OR no supporting chunks, OR cannot verify from evidence

CRITICAL CALIBRATION: A gap with exactly 10 papers, specific to our domain, and recent activity should get confidence=0.72-0.75. Don't default everything to 0.82.

CRITERIA (score each 1-5):
1. DOMAIN_RELEVANCE: Is this ACTUALLY about {get_domain_name()}? If NO → not_worthy, confidence=0.9
2. SPECIFICITY: Concrete problem? ("3D EBP with rotation" > "optimization approaches")
3. EVIDENCE: >=10 unique papers in a real research area?
4. ACTIONABILITY: Could a PhD student start tomorrow?
5. TIMELINESS: Recent activity within last 3 years?

IMPORTANT:
- Score domain relevance FIRST. If gap is not about {get_domain_name()}, return not_worthy immediately.
- Use the FULL confidence range. DO NOT cluster everything at 0.75-0.78.
- Cite specific chunk numbers [1],[2] in reasoning.

Return ONLY a valid JSON object, no preamble, no explanation, no markdown:
{{
    "paper_worthy": "worthy" or "not_worthy" or "insufficient_evidence",
    "confidence": 0.0-1.0,
    "recommended_paper_type": "survey" or "algorithm" or "application" or "theory" or null,
    "reasoning": "Step 1: Is this about {get_domain_name()}? Step 2: Which rubric tier? Step 3: Cite chunks [1],[2].",
    "concerns": ["Wrong domain? Too few papers? Stale? Too broad? Already solved?"],
    "suggested_focus": "If worthy: exact title/angle. Be specific.",
    "gap_explanation": "Literatürde tam olarak NE eksik? Hangi spesifik problem çözülmüş değil? 2-3 cümlede açıkla."
}}
"""
    return prompt


def _verify_assessment_against_evidence(
    assessment: PaperWorthinessAssessment,
    evidence: "DetailedEvidence",
    gap: "GapCandidate | None" = None,
) -> None:
    """
    Verify LLM assessment against evidence constraints.

    This prevents LLM from making claims that contradict the evidence.
    Also enforces domain relevance — non-domain gaps cannot be "worthy".
    """
    notes = []

    # Rule 0: Domain relevance check — worthy gaps MUST be about our domain
    if assessment.paper_worthy == "worthy" and gap is not None:
        domain_ok = _check_domain_relevance(gap, assessment)
        if not domain_ok:
            assessment.paper_worthy = "not_worthy"
            assessment.confidence = min(assessment.confidence, 0.5)
            notes.append(
                "Assessment changed to 'not_worthy': gap appears to be outside "
                f"target domain ({get_domain_name()})."
            )

    # Rule 1: If < 10 papers, cannot be "worthy" with high confidence
    if evidence.unique_papers < 10 and assessment.paper_worthy == "worthy":
        notes.append(
            f"Assessment marked 'worthy' but only {evidence.unique_papers} papers found. "
            "Consider lowering confidence or changing to 'insufficient_evidence'."
        )
        assessment.confidence = min(assessment.confidence, 0.5)

    # Rule 2: If no supporting chunks, must be "insufficient_evidence"
    if evidence.total_chunks == 0 and assessment.paper_worthy == "worthy":
        assessment.paper_worthy = "insufficient_evidence"
        notes.append(
            "Assessment changed to 'insufficient_evidence' due to no supporting chunks."
        )

    # Rule 3: Very low coverage ratio suggests limited foundation
    if evidence.coverage_ratio < 0.001 and assessment.confidence > 0.7:
        notes.append(
            f"Very low coverage ratio ({evidence.coverage_ratio:.4f}) but high confidence "
            f"({assessment.confidence:.2f}). Consider recalibrating."
        )

    # Rule 4: Check reasoning cites evidence
    if assessment.reasoning:
        has_evidence_reference = (
            "chunk" in assessment.reasoning.lower() or
            "paper" in assessment.reasoning.lower() or
            "evidence" in assessment.reasoning.lower() or
            "study" in assessment.reasoning.lower()
        )
        if not has_evidence_reference:
            notes.append(
                "Reasoning does not explicitly cite evidence. Consider adding specific references."
            )

    assessment.evidence_verified = len(notes) == 0
    assessment.verification_notes = "; ".join(notes) if notes else None


def _check_domain_relevance(gap: "GapCandidate", assessment: PaperWorthinessAssessment) -> bool:
    """Deterministic domain check. Returns True if gap appears to be about our domain."""
    domain_name = get_domain_name().lower()
    # Extract key domain terms
    domain_keywords = [
        "bin", "pack", "packing", "container", "load", "loading",
        "pallet", "palletizing", "knapsack", "cutting", "stock",
        "3d", "three-dimensional", "warehouse", "stowage",
    ]

    # Check gap title, description, core terms, and reasoning
    text_to_check = (
        gap.title.lower() + " " +
        gap.description.lower() + " " +
        " ".join(t.lower() for t in gap.core_terms) + " " +
        assessment.reasoning.lower()
    )

    # Count how many domain keywords appear
    matches = sum(1 for kw in domain_keywords if kw in text_to_check)

    # If 0 matches, it's almost certainly off-domain
    return matches >= 1


async def _call_llm(
    prompt: str,
    llm_api_base: str,
    llm_model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call LLM and return response."""
    import litellm

    # OpenAI-compatible local servers (LM Studio, vLLM) require a non-empty api_key
    # for the HTTP client even when the server ignores it.
    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("LITELLM_OPENAI_API_KEY")
        or "sk-local-not-used"
    )

    response = await litellm.acompletion(
        model=llm_model,
        api_base=llm_api_base,
        api_key=api_key,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=4000,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "assessment",
                "schema": {
                    "type": "object",
                    "properties": {
                        "paper_worthy": {"type": "string", "enum": ["worthy", "not_worthy", "insufficient_evidence"]},
                        "confidence": {"type": "number"},
                        "recommended_paper_type": {"type": ["string", "null"]},
                        "reasoning": {"type": "string"},
                        "concerns": {"type": "array", "items": {"type": "string"}},
                        "suggested_focus": {"type": ["string", "null"]},
                        "gap_explanation": {"type": ["string", "null"]},
                    },
                    "required": ["paper_worthy", "confidence", "reasoning", "concerns", "gap_explanation"],
                },
                "strict": True,
            },
        },
    )

    msg = response["choices"][0]["message"]
    content = msg.get("content", "") or ""
    reasoning = msg.get("reasoning_content", "") or ""

    # Combine both fields — reasoning models put JSON in reasoning_content
    combined = f"{reasoning}\n\n{content}" if reasoning and content else (content or reasoning)
    return combined


def _extract_json(text: str) -> dict | None:
    """Extract a JSON object from LLM output that may contain preamble, markdown fences, or thinking."""
    import re

    if not text or not text.strip():
        return None

    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.replace("```", "")

    # Strategy 1: Find the last occurrence of "paper_worthy" and look for a
    # JSON object that contains it. This handles models that output reasoning
    # first, then JSON at the end.
    last_pw = cleaned.rfind('"paper_worthy"')
    if last_pw != -1:
        # Find the opening brace before "paper_worthy"
        brace_before = cleaned.rfind("{", 0, last_pw + 1)
        if brace_before != -1:
            # Find matching closing brace
            depth = 0
            in_string = False
            escape_next = False
            for i in range(brace_before, len(cleaned)):
                ch = cleaned[i]
                if escape_next:
                    escape_next = False
                    continue
                if ch == '\\' and in_string:
                    escape_next = True
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                if not in_string:
                    if ch == '{' or ch == '[':
                        depth += 1
                    elif ch == '}' or ch == ']':
                        depth -= 1
                        if depth == 0:
                            try:
                                obj = json.loads(cleaned[brace_before:i + 1])
                                if isinstance(obj, dict) and "paper_worthy" in obj:
                                    return obj
                            except (json.JSONDecodeError, ValueError):
                                pass
                            break

    # Strategy 2: Try all opening braces as fallback
    brace_start = cleaned.find('{')
    while brace_start != -1:
        depth = 0
        in_string = False
        escape_next = False
        for i in range(brace_start, len(cleaned)):
            ch = cleaned[i]
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
            if not in_string:
                if ch == '{' or ch == '[':
                    depth += 1
                elif ch == '}' or ch == ']':
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(cleaned[brace_start:i + 1])
                            if isinstance(obj, dict) and "paper_worthy" in obj:
                                return obj
                        except (json.JSONDecodeError, ValueError):
                            pass
                        break
        brace_start = cleaned.find('{', brace_start + 1)

    return None


def get_domain_name() -> str:
    """Get the domain name for the assessment context.

    Set DOMAIN_NAME env var to override (e.g. "deep learning for medical imaging").
    """
    return os.environ.get(
        "DOMAIN_NAME",
        "3D bin packing and container loading optimization",
    )
