"""Novelpy pipeline — discover novel concept combinations.

Extracts concepts from BERTopic topics, builds co-occurrence matrix,
calculates atypicality scores to find unexplored combinations.

Note: Uses custom co-occurrence proxy instead of novelpy library
(novelpy's spacy dependency does not build on Python 3.14).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from paperqa.stores.gap_analysis import NovelpyResult, TopicResult

logger = logging.getLogger(__name__)


def extract_concepts_from_topics(topic_result: TopicResult) -> list[str]:
    """Extract unique concepts from BERTopic topic labels.

    Uses top words from each topic as concepts.
    Filters out noise: numbers, LaTeX artifacts, short codes.
    """
    concepts: set[str] = set()

    # Regex patterns to filter out noise
    import re

    noise_patterns = [
        r'\d',                        # any digit → reject
        r'^math',
        r'^varepsilon',
        r'^delta',
        r'^og[a-z]',
        r'dagger$',
        r'rangle$',
        r'^text[a-z]',
        r'^ab[a-z]{3,}fit',
    ]
    noise_regex = re.compile('|'.join(noise_patterns))

    stopwords = {
        "the", "and", "for", "with", "from", "that", "this", "are", "was",
        "were", "been", "being", "have", "has", "had", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall", "its",
        "our", "their", "your", "his", "her", "not", "but", "also", "into",
        "each", "all", "any", "both", "more", "most", "other", "some", "such",
        "than", "too", "very", "just", "about", "above", "after", "again",
        "between", "through", "during", "before", "under", "over", "then",
        "here", "there", "when", "where", "how", "what", "which", "who",
        "whom", "these", "those", "only", "same", "using", "based", "used",
        "case", "new", "one", "two", "three", "first", "second", "per",
        "via", "non", "pre", "use", "set", "number", "approach", "method",
        "problem", "results", "proposed", "paper", "model", "fig", "table",
        "section", "chapter", "show", "shows", "shown", "see", "note",
        "let", "given", "respectively", "thus", "hence", "therefore",
        "value", "values", "total", "time", "size", "type", "order",
        "right", "left", "end", "best", "test", "data", "large", "small",
        "different", "optimal", "upper", "lower", "maximum", "minimum",
        "following", "corresponding", "obtained", "considered", "according",
    }

    def is_valid_concept(word: str) -> bool:
        word_lower = word.lower().strip()
        if len(word_lower) < 4:
            return False
        if not word_lower.isalpha():
            return False
        if noise_regex.search(word_lower):
            return False
        if word_lower in stopwords:
            return False
        return True

    concept_topics: dict[str, set[int]] = defaultdict(set)
    for topic_id, words in topic_result.topics.items():
        for word in words:
            if is_valid_concept(word):
                concept_topics[word.lower().strip()].add(topic_id)

    for c, tids in concept_topics.items():
        if len(tids) >= 2:
            concepts.add(c)

    return sorted(concepts)


def build_cooccurrence_from_chunks(
    topic_result: TopicResult,
    valid_concepts: set[str] | None = None,
) -> list[tuple[str, str, int]]:
    """Build concept co-occurrence from chunk-to-topic assignments.

    Concepts that appear in the same topic co-occur.
    Returns list of (concept_a, concept_b, count).
    """
    concept_to_topics: dict[str, set[int]] = defaultdict(set)
    for topic_id, words in topic_result.topics.items():
        for word in words:
            w = word.lower().strip()
            if valid_concepts and w not in valid_concepts:
                continue
            concept_to_topics[w].add(topic_id)

    topic_counts: dict[int, int] = {}
    for t in topic_result.chunk_to_topic:
        if t != -1:
            topic_counts[t] = topic_counts.get(t, 0) + 1

    cooccurrence: dict[tuple[str, str], int] = defaultdict(int)
    concept_list = sorted(concept_to_topics.keys())

    for i, c_a in enumerate(concept_list):
        for c_b in concept_list[i + 1:]:
            shared = concept_to_topics[c_a] & concept_to_topics[c_b]
            if shared:
                count = sum(topic_counts.get(tid, 0) for tid in shared)
                cooccurrence[(c_a, c_b)] = count

    result = [(a, b, count) for (a, b), count in cooccurrence.items()]
    result.sort(key=lambda x: -x[2])

    logger.info("Built co-occurrence: %d pairs from %d concepts",
                len(result), len(concept_list))
    return result


def calculate_novelty(
    concepts: list[str],
    cooccurrence: list[tuple[str, str, int]],
    top_k: int = 50,
) -> list[tuple[str, str, float]]:
    """Calculate atypicality via negative PMI (Pointwise Mutual Information).

    PMI = log2(P(a,b) / (P(a) * P(b)))
    Low PMI → pair co-occurs less than expected → novel combination.

    We invert PMI so higher score = more novel.
    """
    import math

    if not cooccurrence:
        return []

    concept_freq: dict[str, int] = defaultdict(int)
    total_cooc = 0
    for c_a, c_b, count in cooccurrence:
        concept_freq[c_a] += count
        concept_freq[c_b] += count
        total_cooc += count

    if total_cooc == 0:
        return []

    novel_pairs = []
    for c_a, c_b, count in cooccurrence:
        if count < 5:
            continue
        p_ab = count / total_cooc
        p_a = concept_freq[c_a] / total_cooc
        p_b = concept_freq[c_b] / total_cooc
        if p_a == 0 or p_b == 0:
            continue
        pmi = math.log2(p_ab / (p_a * p_b))
        novelty = -pmi
        if novelty > 0:
            novel_pairs.append((c_a, c_b, round(novelty, 4)))

    novel_pairs.sort(key=lambda x: -x[2])
    return novel_pairs[:top_k]


async def run_novelpy(
    topic_result: TopicResult,
    output_dir: str = "./data/gap_analysis/novelpy",
) -> NovelpyResult:
    """Run Novelpy pipeline: extract concepts → co-occurrence → novelty → save."""
    # Extract concepts
    concepts = extract_concepts_from_topics(topic_result)
    logger.info("Extracted %d concepts from BERTopic", len(concepts))

    concept_set = set(concepts)
    cooccurrence = build_cooccurrence_from_chunks(topic_result, valid_concepts=concept_set)

    # Calculate novelty
    novel_pairs = calculate_novelty(concepts, cooccurrence)

    # Save artifacts
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "concepts.json", "w") as f:
        json.dump(concepts, f, indent=2)

    with open(output_path / "cooccurrence.json", "w") as f:
        json.dump([{"a": a, "b": b, "count": c} for a, b, c in cooccurrence], f, indent=2)

    with open(output_path / "novel_pairs.json", "w") as f:
        json.dump([{"a": a, "b": b, "score": s} for a, b, s in novel_pairs], f, indent=2)

    logger.info("Saved Novelpy artifacts: %d concepts, %d novel pairs",
                len(concepts), len(novel_pairs))

    return NovelpyResult(
        concepts=concepts,
        cooccurrence_counts=cooccurrence,
        novel_pairs=novel_pairs,
    )
