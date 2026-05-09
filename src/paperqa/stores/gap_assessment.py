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
        )

        # Verify assessment against evidence
        _verify_assessment_against_evidence(assessment, evidence)

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

    # Format evidence summary
    evidence_summary = f"""
EVIDENCE SUMMARY:
- Gap Type: {gap.gap_type}
- Supporting Chunks: {evidence.total_chunks}
- Unique Papers: {evidence.unique_papers}
- Coverage Ratio: {evidence.coverage_ratio:.4f}

SAMPLE EVIDENCE (first 3 chunks):
"""

    for i, chunk in enumerate(evidence.supporting_chunks[:3], 1):
        evidence_summary += f"\n{i}. [Score: {chunk.relevance_score:.3f}] {chunk.text[:200]}...\n"

    if not evidence.supporting_chunks:
        evidence_summary += "\nNo supporting chunks found in corpus.\n"

    # Build the full prompt
    prompt = f"""You are a senior academic reviewer specializing in {get_domain_name()}.

Evaluate this research gap for paper publication potential.

GAP INFORMATION:
Type: {gap.gap_type}
Title/Core: {gap.title}
Description: {gap.description}

{evidence_summary}

CRITERIA FOR EVALUATION:
1. Specificity: Is this gap specific enough for a focused paper?
2. Evidence Foundation: Is there sufficient literature (at least 10 relevant papers)?
3. Actionability: Can researchers build something new based on this gap?
4. Novelty: Is this gap genuinely underexplored or already well-studied?

IMPORTANT RULES:
- Base your assessment ONLY on the provided evidence
- DO NOT infer or hallucinate information not present in the evidence
- If evidence is insufficient, state "insufficient_evidence"
- Cite specific evidence numbers in your reasoning
- Be conservative - if unsure, recommend "insufficient_evidence"

Return ONLY a valid JSON object, no preamble, no explanation, no markdown:
{{
    "paper_worthy": "worthy" or "not_worthy" or "insufficient_evidence",
    "confidence": 0.0-1.0,
    "recommended_paper_type": "survey" or "algorithm" or "application" or "theory" or null,
    "reasoning": "Your assessment reasoning, citing specific evidence numbers",
    "concerns": ["List potential issues with this gap as paper topic"],
    "suggested_focus": "Recommended angle for the paper if worthy"
}}
"""
    return prompt


def _verify_assessment_against_evidence(
    assessment: PaperWorthinessAssessment,
    evidence: "DetailedEvidence",
) -> None:
    """
    Verify LLM assessment against evidence constraints.

    This prevents LLM from making claims that contradict the evidence.
    """
    notes = []

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
        max_tokens=max_tokens,
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
