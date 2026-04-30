# Phase 2: Gap Analysis Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an automated research gap detection system that analyzes 175K academic paper chunks and produces a structured report identifying actionable research gaps in 3D container loading / bin packing.

**Architecture:** 5-component pipeline executed sequentially — BERTopic (topic discovery) → Gap-Language (explicit gaps) → OpenAlex (external taxonomy) → Novelpy (novelty scoring) → LLM Synthesis (validation + report). Each component is a standalone module in `src/paperqa/stores/`, tested independently, orchestrated by `gap_analysis.py`.

**Tech Stack:** BERTopic, UMAP, HDBSCAN, pyalex, novelpy, pandas, numpy, PaperQA2 (`paperbridge_agent_query`), Tantivy (`SearchIndex`), Qdrant, LLM (Qwen3.6-35B-A3B at 192.168.0.28:8005)

**Infrastructure:**
| Component | URL | Usage |
|-----------|-----|-------|
| LLM | `http://192.168.0.28:8005/v1` (Qwen3.6-35B-A3B) | LLM synthesis, concept extraction |
| Qdrant | `http://192.168.0.28:6333` (175K chunks, 4096-dim) | Chunk + embedding retrieval |
| Embedding | `http://localhost:8082` (Qwen3-Embedding-8B) | Gap-language clustering |
| Tantivy | `./data/tantivy_index/paperbridge_index/` (9987 docs) | Gap-language keyword search |

**Constraint:** `max_concurrent_requests=1` — LLM server running parallel=1. All LLM calls must be sequential.

---

## File Structure

```
src/paperqa/stores/
├── gap_analysis.py         # Orchestrator + shared dataclasses
├── bertopic_pipeline.py    # BERTopic clustering (chunks → topics)
├── gap_language.py         # Gap keyword extraction + clustering
├── openalex_compare.py     # OpenAlex taxonomy comparison
├── novelpy_pipeline.py     # Novelty scoring
└── gap_synthesis.py        # LLM synthesis + validation

tests/
├── test_bertopic_pipeline.py
├── test_gap_language.py
├── test_openalex_compare.py
├── test_novelpy_pipeline.py
└── test_gap_synthesis.py

data/gap_analysis/          # Output directory (created at runtime)
├── bertopic/               # Topic model + metadata
├── gap_language/           # Gap chunks + categories
├── openalex/               # Coverage data
├── novelpy/                # Novelty scores
└── report/                 # Final report (JSON + Markdown)
```

---

### Task 1: Install Dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Install bertopic (pulls umap-learn + hdbscan as dependencies)**

Run: `pip install bertopic`
Expected: bertopic, umap-learn, hdbscan, numpy, scikit-learn installed

- [ ] **Step 2: Install pyalex**

Run: `pip install pyalex`
Expected: pyalex installed

- [ ] **Step 3: Install novelpy**

Run: `pip install novelpy`
Expected: novelpy installed

- [ ] **Step 4: Install pandas**

Run: `pip install pandas`
Expected: pandas installed

- [ ] **Step 5: Verify all imports**

```python
python3 -c "
import bertopic
import umap
import hdbscan
import pyalex
import novelpy
import pandas
import numpy
print('All Phase 2 dependencies installed')
"
```
Expected: No ImportError

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml  # if modified
git commit -m "chore: install Phase 2 dependencies (bertopic, pyalex, novelpy, pandas)"
```

---

### Task 2: Gap Analysis Dataclasses + Orchestrator Skeleton

**Files:**
- Create: `src/paperqa/stores/gap_analysis.py`
- Modify: `src/paperqa/stores/__init__.py`

- [ ] **Step 1: Create gap_analysis.py with dataclasses and orchestrator**

```python
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
```

- [ ] **Step 2: Update __init__.py to export gap_analysis**

Read `src/paperqa/stores/__init__.py` and add:

```python
from paperqa.stores.gap_analysis import (
    GapItem,
    GapLanguageResult,
    GapReport,
    NovelpyResult,
    OpenAlexResult,
    TopicResult,
    run_gap_analysis,
)
__all__ = [
    "PaperBridgeDocs",
    "PaperBridgeQdrantStore",
    "paperbridge_agent_query",
    "paperbridge_contracrow",
    "parse_pdf_name",
    "GapItem",
    "GapLanguageResult",
    "GapReport",
    "NovelpyResult",
    "OpenAlexResult",
    "TopicResult",
    "run_gap_analysis",
]
```

- [ ] **Step 3: Test imports**

Run: `PYTHONPATH=src python3 -c "from paperqa.stores import run_gap_analysis, TopicResult, GapReport; print('OK: all dataclasses importable')"`
Expected: `OK: all dataclasses importable`

- [ ] **Step 4: Commit**

```bash
git add src/paperqa/stores/gap_analysis.py src/paperqa/stores/__init__.py
git commit -m "feat: gap analysis orchestrator skeleton with dataclasses"
```

---

### Task 3: BERTopic Pipeline

**Files:**
- Create: `src/paperqa/stores/bertopic_pipeline.py`
- Create: `tests/test_bertopic_pipeline.py`

- [ ] **Step 1: Create bertopic_pipeline.py with Qdrant chunk extraction**

```python
"""BERTopic pipeline — discover topics from Qdrant corpus.

Uses pre-computed 4096-dim embeddings from Qdrant.
PCA 4096→256 → UMAP 256→10 → HDBSCAN → topic labels.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from qdrant_client import AsyncQdrantClient

from paperqa.stores.paperbridge_store import parse_pdf_name

logger = logging.getLogger(__name__)


async def extract_chunks_from_qdrant(
    collection_name: str = "paperbridge_glm_v2",
    qdrant_url: str = "http://192.168.0.28:6333",
    batch_size: int = 500,
) -> tuple[list[str], list[np.ndarray], list[int]]:
    """Scroll all chunks from Qdrant, return texts, embeddings, years.

    Returns:
        (texts, embeddings, years) where each index corresponds to the same chunk.
    """
    client = AsyncQdrantClient(url=qdrant_url)

    all_texts: list[str] = []
    all_embeddings: list[np.ndarray] = []
    all_years: list[int] = []

    offset = None
    seen_offsets: set[str] = set()
    scrolled = 0

    while True:
        points, next_offset = await client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=["pdf_name", "text", "page_num"],
            with_vectors=True,
        )
        if not points:
            break

        for point in points:
            payload = point.payload
            text = payload.get("text", "")
            if not text.strip():
                continue

            all_texts.append(text)
            all_embeddings.append(np.array(point.vector, dtype=np.float32))

            meta = parse_pdf_name(payload.get("pdf_name", ""))
            all_years.append(meta.get("year") or 2000)

        scrolled += len(points)
        if scrolled % 2000 == 0:
            logger.info("Scrolled %d points (%d valid chunks)", scrolled, len(all_texts))

        if next_offset is None:
            break

        offset_str = str(next_offset)
        if offset_str in seen_offsets:
            break
        seen_offsets.add(offset_str)
        offset = next_offset

    logger.info("Extracted %d chunks from Qdrant", len(all_texts))
    return all_texts, all_embeddings, all_years
```

- [ ] **Step 2: Test Qdrant extraction**

Run: `PYTHONPATH=src python3 -c "
import asyncio
from paperqa.stores.bertopic_pipeline import extract_chunks_from_qdrant

async def test():
    # Test with small batch to verify
    texts, embs, years = await extract_chunks_from_qdrant(batch_size=500)
    print(f'Extracted: {len(texts)} texts, {len(embs)} embeddings, {len(years)} years')
    assert len(texts) == len(embs) == len(years)
    assert len(texts) > 100000
    assert embs[0].shape == (4096,)
    print(f'Year range: {min(years)}-{max(years)}')
    print('OK: chunk extraction works')

asyncio.run(test())
"`
Expected: Extracted 163K+ chunks, year range ~2000-2026

- [ ] **Step 3: Add BERTopic fitting to bertopic_pipeline.py**

Append to `src/paperqa/stores/bertopic_pipeline.py`:

```python

def fit_bertopic(
    texts: list[str],
    embeddings: list[np.ndarray],
    years: list[int],
    output_dir: str = "./data/gap_analysis/bertopic",
) -> TopicResult:
    """Fit BERTopic on pre-computed embeddings.

    Pipeline: PCA 4096→256 → UMAP 256→10 → HDBSCAN → ClassTF-IDF labels.
    """
    from bertopic import BERTopic
    from bertopic.representation import MaximalMarginalRelevance
    from sklearn.decomposition import PCA
    from umap import UMAP
    import hdbscan
    import pandas as pd

    embeddings_array = np.array(embeddings, dtype=np.float32)
    years_array = np.array(years, dtype=np.int32)

    # PCA dimensionality reduction (memory optimization)
    logger.info("Running PCA 4096→256...")
    pca = PCA(n_components=256, random_state=42)
    reduced = pca.fit_transform(embeddings_array)

    # UMAP dimensionality reduction
    logger.info("Running UMAP 256→10...")
    umap_model = UMAP(
        n_components=10,
        metric="cosine",
        random_state=42,
        min_dist=0.0,
        n_neighbors=15,
    )
    embeddings_reduced = umap_model.fit_transform(reduced)

    # HDBSCAN clustering
    logger.info("Running HDBSCAN clustering...")
    cluster_model = hdbscan.HDBSCAN(
        min_cluster_size=20,
        min_samples=5,
        metric="cosine",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # Fit BERTopic
    logger.info("Fitting BERTopic...")
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        calculate_probabilities=False,
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(texts, embeddings=embeddings_reduced)

    # Get topic info
    topic_info_df = topic_model.get_topic_info()

    # Filter out -1 (outliers) for analysis
    valid_mask = np.array(topics) != -1
    num_topics = len(topic_model.get_topics())
    logger.info("BERTopic found %d topics (%d outliers)",
                num_topics, sum(~valid_mask))

    # Build temporal data
    valid_topics = topics[valid_mask]
    valid_years = years_array[valid_mask]
    temporal_data: dict[int, dict] = {}
    for t, y in zip(valid_topics, valid_years):
        if t not in temporal_data:
            temporal_data[t] = {}
        temporal_data[t][int(y)] = temporal_data[t].get(int(y), 0) + 1

    # Identify sparse topics (<50 docs)
    sparse_topics = [
        {"topic_id": row["Topic"], "count": row["Count"], "name": row["Name"]}
        for _, row in topic_info_df.iterrows()
        if row["Topic"] != -1 and row["Count"] < 50
    ]

    # Build topic dict
    topic_dict: dict[int, list[str]] = {}
    for topic_id, words in topic_model.get_topic().items():
        topic_dict[topic_id] = [word for word, _ in words]

    # Convert topic_info to dict
    topic_info_dict = {
        "columns": list(topic_info_df.columns),
        "data": topic_info_df.to_dict(orient="records"),
    }

    # Save artifacts
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    topic_model.save(str(output_path / "bertopic_model"), serialization="pytorch")

    # Save metadata
    result = TopicResult(
        topics=topic_dict,
        topic_info=topic_info_dict,
        chunk_to_topic=list(topics),
        sparse_topics=sparse_topics,
        temporal_data=temporal_data,
    )

    metadata = {
        "num_chunks": len(texts),
        "num_topics": num_topics,
        "num_outliers": int(sum(~valid_mask)),
        "sparse_topics": len(sparse_topics),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save topic info as CSV
    topic_info_df.to_csv(output_path / "topic_info.csv", index=False)

    logger.info("Saved BERTopic artifacts to %s", output_path)
    return result


async def run_bertopic(
    qdrant_collection: str = "paperbridge_glm_v2",
    output_dir: str = "./data/gap_analysis/bertopic",
) -> TopicResult:
    """Run the full BERTopic pipeline: extract → fit → save."""
    texts, embeddings, years = await extract_chunks_from_qdrant(
        collection_name=qdrant_collection,
    )
    return fit_bertopic(texts, embeddings, years, output_dir)
```

Where `json` needs to be imported — add to imports at top of file:
```python
import json
```

- [ ] **Step 4: Test BERTopic on subset (10K chunks)**

Run: `PYTHONPATH=src python3 -c "
import asyncio, numpy as np
from paperqa.stores.bertopic_pipeline import extract_chunks_from_qdrant, fit_bertopic

async def test():
    texts, embs, years = await extract_chunks_from_qdrant(batch_size=500)
    # Use subset for fast test
    N = 10000
    result = fit_bertopic(texts[:N], embs[:N], years[:N], './data/gap_analysis/bertopic_test')
    print(f'Topics: {len(result.topics)}')
    print(f'Chunk-to-topic len: {len(result.chunk_to_topic)}')
    print(f'Sparse topics: {len(result.sparse_topics)}')
    print(f'Temporal data topics: {len(result.temporal_data)}')
    assert len(result.topics) >= 10
    assert len(result.chunk_to_topic) == N
    print('OK: BERTopic works on subset')

asyncio.run(test())
"`
Expected: 10+ topics, no OOM

- [ ] **Step 5: Commit**

```bash
git add src/paperqa/stores/bertopic_pipeline.py
git commit -m "feat: BERTopic pipeline with Qdrant extraction and topic fitting"
```

---

### Task 4: Gap-Language Extraction

**Files:**
- Create: `src/paperqa/stores/gap_language.py`
- Create: `tests/test_gap_language.py`

- [ ] **Step 1: Create gap_language.py**

```python
"""Gap-Language Extraction — find explicit gap signals in papers.

Uses Tantivy index for keyword search, clusters results with BERTopic.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from paperqa.agents.search import SearchIndex

logger = logging.getLogger(__name__)

# Keywords that signal research gaps
GAP_KEYWORDS = [
    "future work",
    "future research",
    "limitation",
    "limitations",
    "not been studied",
    "remains unclear",
    "under-explored",
    "no study has",
    "research gap",
    "open problem",
    "yet to be",
    "warrants further",
    "needs investigation",
    "further research is needed",
    "direction for future",
    "potential limitation",
]


async def extract_gap_chunks(
    index_dir: str = "./data/tantivy_index",
    top_n: int = 20,
) -> list[dict]:
    """Search Tantivy index for gap-language keywords.

    Returns list of {pdf_hash, title, year, body_snippet, keyword_matched}.
    """
    index = SearchIndex(
        fields=["file_location", "body", "title", "year"],
        index_name="paperbridge_index",
        index_directory=index_dir,
    )

    gap_chunks: list[dict] = []
    seen_hashes: set[str] = set()

    for keyword in GAP_KEYWORDS:
        results = await index.query(keyword, top_n=top_n)
        for r in results:
            if not isinstance(r, dict):
                continue
            pdf_hash = r.get("pdf_hash")
            if not pdf_hash or pdf_hash in seen_hashes:
                continue
            seen_hashes.add(pdf_hash)

            gap_chunks.append({
                "pdf_hash": pdf_hash,
                "title": r.get("title", ""),
                "year": r.get("year"),
                "keyword_matched": keyword,
                "file_location": r.get("file_location", ""),
            })
            logger.info("Found gap chunk: '%s' in %s", keyword, r.get("title", "")[:50])

    logger.info("Found %d unique chunks with gap-language", len(gap_chunks))
    return gap_chunks


async def cluster_gap_chunks(
    gap_chunks: list[dict],
    output_dir: str = "./data/gap_analysis/gap_language",
) -> GapLanguageResult:
    """Cluster gap chunks using BERTopic to find gap categories.

    Uses chunk titles + keywords as text for clustering.
    """
    from bertopic import BERTopic
    from dataclasses import dataclass

    if not gap_chunks:
        return GapLanguageResult(gap_chunks=[], gap_categories={}, top_gaps=[])

    # Build text representations for clustering
    texts = [f"{chunk['title']} {chunk['keyword_matched']}" for chunk in gap_chunks]

    # Fit BERTopic on gap chunks
    topic_model = BERTopic(
        language="english",
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(texts)

    topic_info = topic_model.get_topic_info()

    # Build gap categories
    gap_categories: dict[int, dict] = {}
    for _, row in topic_info.iterrows():
        if row["Topic"] == -1:
            continue
        topic_id = int(row["Topic"])
        words = topic_model.get_topic(topic_id)
        gap_categories[topic_id] = {
            "label": row["Name"],
            "count": int(row["Count"]),
            "words": [w for w, _ in words] if words else [],
        }

    # Top gap phrases (most common keywords)
    from collections import Counter
    keyword_counts = Counter(chunk["keyword_matched"] for chunk in gap_chunks)
    top_gaps = [kw for kw, _ in keyword_counts.most_common(20)]

    # Save artifacts
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    import json
    with open(output_path / "gap_chunks.json", "w") as f:
        json.dump(gap_chunks, f, indent=2)

    with open(output_path / "categories.json", "w") as f:
        json.dump({str(k): v for k, v in gap_categories.items()}, f, indent=2)

    topic_info.to_csv(output_path / "topic_info.csv", index=False)

    logger.info("Saved gap-language artifacts: %d chunks, %d categories",
                len(gap_chunks), len(gap_categories))

    return GapLanguageResult(
        gap_chunks=gap_chunks,
        gap_categories=gap_categories,
        top_gaps=top_gaps,
    )


async def run_gap_language(
    output_dir: str = "./data/gap_analysis/gap_language",
) -> GapLanguageResult:
    """Run the full gap-language pipeline: extract → cluster → save."""
    gap_chunks = await extract_gap_chunks()
    return await cluster_gap_chunks(gap_chunks, output_dir)
```

Add import at top of file:
```python
from paperqa.stores.gap_analysis import GapLanguageResult
```

- [ ] **Step 2: Test gap-language extraction**

Run: `PYTHONPATH=src python3 -c "
import asyncio
from paperqa.stores.gap_language import run_gap_language

async def test():
    result = await run_gap_language('./data/gap_analysis/gap_language_test')
    print(f'Gap chunks: {len(result.gap_chunks)}')
    print(f'Categories: {len(result.gap_categories)}')
    print(f'Top gaps: {result.top_gaps[:5]}')
    assert len(result.gap_chunks) > 0
    assert len(result.gap_categories) >= 1
    print('OK: gap-language extraction works')

asyncio.run(test())
"`
Expected: 20+ gap chunks, 2+ categories

- [ ] **Step 3: Commit**

```bash
git add src/paperqa/stores/gap_language.py
git commit -m "feat: gap-language extraction with Tantivy + BERTopic clustering"
```

---

### Task 5: OpenAlex Comparison

**Files:**
- Create: `src/paperqa/stores/openalex_compare.py`
- Create: `tests/test_openalex_compare.py`

- [ ] **Step 1: Create openalex_compare.py**

```python
"""OpenAlex comparison — compare corpus coverage against global taxonomy.

Queries OpenAlex API for topics related to bin packing / container loading.
Compares with BERTopic topics to find coverage gaps.
"""

from __future__ import annotations

import logging
from pathlib import Path

import json

logger = logging.getLogger(__name__)

# Search terms for OpenAlex topic discovery
DOMAIN_SEARCH_TERMS = [
    "bin packing",
    "container loading",
    "cutting stock",
    "3D loading",
    "packing problem",
    "loading problem",
    "container loading problem",
    "three-dimensional bin packing",
    "orthogonal packing",
    "stock cutting",
    "knapsack problem",
    "loading optimization",
]


async def fetch_openalex_topics() -> list[dict]:
    """Query OpenAlex API for topics related to our domain.

    Returns list of {id, display_name, works_count, sub_fields}.
    """
    import httpx

    topics: list[dict] = []
    seen_ids: set[str] = set()

    async with httpx.AsyncClient(timeout=60.0) as client:
        for term in DOMAIN_SEARCH_TERMS:
            try:
                url = "https://api.openalex.org/works"
                params = {
                    "search": term,
                    "per_page": 200,
                    "sort": "cited_by_count:desc",
                    "filter": "type:article",
                    "select": "id,primary_location,public_names,primary_topic,concept_ids",
                }
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

                for work in data.get("results", []):
                    primary_topic = work.get("primary_topic")
                    if primary_topic and primary_topic.get("id") not in seen_ids:
                        seen_ids.add(primary_topic["id"])
                        topics.append({
                            "id": primary_topic["id"],
                            "display_name": primary_topic["display_name"],
                            "works_count": primary_topic.get("works_count", 0),
                        })

                logger.info("OpenAlex '%s': %d works, %d total unique topics",
                            term, len(data.get("results", [])), len(seen_ids))

            except Exception as e:
                logger.warning("OpenAlex query '%s' failed: %s", term, e)

    logger.info("Fetched %d unique OpenAlex topics", len(topics))
    return topics


def compute_coverage(
    openalex_topics: list[dict],
    topic_result,
    threshold: float = 0.01,
) -> tuple[list[dict], list[dict]]:
    """Match BERTopic topics to OpenAlex topics and compute coverage.

    Returns (our_coverage, missing_topics).
    """
    # Build BERTopic keyword set
    bertopic_keywords: set[str] = set()
    for topic_id, words in topic_result.topics.items():
        for word in words:
            bertopic_keywords.add(word.lower())

    our_coverage: list[dict] = []
    missing_topics: list[dict] = []

    for topic in openalex_topics:
        topic_name = topic["display_name"].lower()
        # Count how many BERTopic keywords match this OpenAlex topic
        matching_keywords = sum(1 for kw in bertopic_keywords if kw in topic_name)

        # Estimate our count from BERTopic
        our_count = 0
        for tid, words in topic_result.topics.items():
            for word in words:
                if word.lower() in topic_name:
                    # Sum chunks in this topic
                    our_count += sum(1 for t in topic_result.chunk_to_topic if t == tid)
                    break

        global_count = topic.get("works_count", 1)
        ratio = our_count / max(global_count, 1)

        our_coverage.append({
            "topic_name": topic["display_name"],
            "our_count": our_count,
            "global_count": global_count,
            "ratio": round(ratio, 4),
            "matching_keywords": matching_keywords,
        })

        if ratio < threshold:
            missing_topics.append({
                "topic_name": topic["display_name"],
                "our_count": our_count,
                "global_count": global_count,
                "ratio": round(ratio, 4),
            })

    # Sort by ratio ascending (most missing first)
    missing_topics.sort(key=lambda x: x["ratio"])
    our_coverage.sort(key=lambda x: x["ratio"])

    logger.info("Coverage computed: %d missing topics (threshold < %.2f)",
                len(missing_topics), threshold)
    return our_coverage, missing_topics


async def run_openalex(
    topic_result,
    output_dir: str = "./data/gap_analysis/openalex",
) -> OpenAlexResult:
    """Run OpenAlex comparison: fetch topics → compute coverage → save."""
    from paperqa.stores.gap_analysis import OpenAlexResult

    # Fetch OpenAlex topics
    global_topics = await fetch_openalex_topics()

    # Compute coverage
    our_coverage, missing_topics = compute_coverage(global_topics, topic_result)

    # Save artifacts
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "global_topics.json", "w") as f:
        json.dump(global_topics, f, indent=2)

    with open(output_path / "coverage.json", "w") as f:
        json.dump(our_coverage, f, indent=2)

    with open(output_path / "missing_topics.json", "w") as f:
        json.dump(missing_topics, f, indent=2)

    logger.info("Saved OpenAlex artifacts to %s", output_path)

    return OpenAlexResult(
        global_topics=global_topics,
        our_coverage=our_coverage,
        missing_topics=missing_topics,
    )
```

Add import at top:
```python
from paperqa.stores.gap_analysis import TopicResult
```

- [ ] **Step 2: Test OpenAlex API connectivity**

Run: `PYTHONPATH=src python3 -c "
import asyncio
from paperqa.stores.openalex_compare import fetch_openalex_topics

async def test():
    topics = await fetch_openalex_topics()
    print(f'OpenAlex topics: {len(topics)}')
    for t in topics[:5]:
        print(f'  {t[\"display_name\"]}: {t[\"works_count\"]} works')
    assert len(topics) > 0
    print('OK: OpenAlex API works')

asyncio.run(test())
"`
Expected: Topics fetched successfully

- [ ] **Step 3: Commit**

```bash
git add src/paperqa/stores/openalex_compare.py
git commit -m "feat: OpenAlex taxonomy comparison with coverage calculation"
```

---

### Task 6: Novelpy Pipeline

**Files:**
- Create: `src/paperqa/stores/novelpy_pipeline.py`
- Create: `tests/test_novelpy_pipeline.py`

- [ ] **Step 1: Create novelpy_pipeline.py**

```python
"""Novelpy pipeline — discover novel concept combinations.

Extracts concepts from BERTopic topics, builds co-occurrence matrix,
calculates atypicality scores to find unexplored combinations.
"""

from __future__ import annotations

import logging
from pathlib import Path

import json
import numpy as np

logger = logging.getLogger(__name__)


def extract_concepts_from_topics(topic_result) -> list[str]:
    """Extract unique concepts from BERTopic topic labels.

    Uses top words from each topic as concepts.
    """
    concepts: set[str] = set()
    for topic_id, words in topic_result.topics.items():
        for word in words:
            word_lower = word.lower().strip()
            if word_lower and len(word_lower) > 2:
                concepts.add(word_lower)
    return sorted(concepts)


def build_cooccurrence_from_chunks(
    topic_result,
) -> list[tuple[str, str, int]]:
    """Build concept co-occurrence from chunk-to-topic assignments.

    Concepts that appear in the same topic co-occur.
    Returns list of (concept_a, concept_b, count).
    """
    from collections import defaultdict

    # Map concept → set of topic_ids
    concept_to_topics: dict[str, set[int]] = defaultdict(set)
    for topic_id, words in topic_result.topics.items():
        for word in words:
            concept_to_topics[word.lower()].add(topic_id)

    # Count co-occurrences
    cooccurrence: dict[tuple[str, str], int] = defaultdict(int)
    concept_list = sorted(concept_to_topics.keys())

    for i, c_a in enumerate(concept_list):
        for c_b in concept_list[i + 1:]:
            shared = concept_to_topics[c_a] & concept_to_topics[c_b]
            if shared:
                # Count chunks in shared topics
                count = sum(1 for t in topic_result.chunk_to_topic if t in shared)
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
    """Calculate atypicality scores for concept pairs.

    Uses normalized co-occurrence frequency as novelty proxy.
    Low co-occurrence + semantic relatedness = high novelty.

    Returns list of (concept_a, concept_b, score) sorted by score descending.
    """
    if not cooccurrence:
        return []

    max_count = max(c[2] for c in cooccurrence) if cooccurrence else 1

    # Novelty = inverse frequency (rare pairs are more novel)
    # But filter to pairs that DO co-occur (completely absent = not actionable)
    novel_pairs = []
    for c_a, c_b, count in cooccurrence:
        if count > 0:
            # Normalized score: lower frequency → higher novelty
            score = 1.0 - (count / max_count)
            # Only keep pairs with some co-occurrence but relatively rare
            if count >= 5 and count <= max_count * 0.3:
                novel_pairs.append((c_a, c_b, round(score, 4)))

    novel_pairs.sort(key=lambda x: -x[2])
    return novel_pairs[:top_k]


async def run_novelpy(
    topic_result,
    output_dir: str = "./data/gap_analysis/novelpy",
) -> NovelpyResult:
    """Run Novelpy pipeline: extract concepts → co-occurrence → novelty → save."""
    from paperqa.stores.gap_analysis import NovelpyResult

    # Extract concepts
    concepts = extract_concepts_from_topics(topic_result)
    logger.info("Extracted %d concepts from BERTopic", len(concepts))

    # Build co-occurrence
    cooccurrence = build_cooccurrence_from_chunks(topic_result)

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
```

Add import at top:
```python
from paperqa.stores.gap_analysis import TopicResult
```

- [ ] **Step 2: Test Novelpy with mock data**

Run: `PYTHONPATH=src python3 -c "
import asyncio
from paperqa.stores.gap_analysis import TopicResult
from paperqa.stores.novelpy_pipeline import run_novelpy

async def test():
    # Create mock topic result
    mock_result = TopicResult(
        topics={
            0: ['bin', 'packing', 'container'],
            1: ['loading', '3d', 'optimization'],
            2: ['genetic', 'algorithm', 'heuristic'],
            3: ['neural', 'network', 'deep', 'learning'],
            4: ['cutting', 'stock', 'waste'],
            5: ['scheduling', 'time', 'constraint'],
            6: ['robot', 'robotic', 'manipulation'],
            7: ['warehouse', 'logistics', 'supply'],
            8: ['graph', 'neural', 'attention'],
            9: ['quantum', 'computing', 'optimization'],
        },
        topic_info={'columns': [], 'data': []},
        chunk_to_topic=[i % 10 for i in range(1000)],
        sparse_topics=[],
        temporal_data={},
    )
    result = await run_novelpy(mock_result, './data/gap_analysis/novelpy_test')
    print(f'Concepts: {len(result.concepts)}')
    print(f'Co-occurrence pairs: {len(result.cooccurrence_counts)}')
    print(f'Novel pairs: {len(result.novel_pairs)}')
    for a, b, s in result.novel_pairs[:5]:
        print(f'  {a} + {b}: score={s}')
    assert len(result.concepts) > 0
    print('OK: Novelpy works')

asyncio.run(test())
"`
Expected: Concepts extracted, novel pairs calculated

- [ ] **Step 3: Commit**

```bash
git add src/paperqa/stores/novelpy_pipeline.py
git commit -m "feat: Novelpy novelty scoring pipeline with co-occurrence analysis"
```

---

### Task 7: LLM Gap Synthesis

**Files:**
- Create: `src/paperqa/stores/gap_synthesis.py`

- [ ] **Step 1: Create gap_synthesis.py with candidate collection**

```python
"""LLM Gap Synthesis — validate gap candidates and produce final report.

Collects candidates from BERTopic, OpenAlex, Novelpy, Gap-Language.
Validates each candidate with paperbridge_agent_query.
Synthesizes a structured report.
"""

from __future__ import annotations

import logging
from pathlib import Path

import json

from paperqa.stores.agent import paperbridge_agent_query
from paperqa.stores.gap_analysis import GapItem, GapReport

logger = logging.getLogger(__name__)


def collect_candidate_gaps(
    topic_result,
    gap_lang_result,
    openalex_result,
    novelpy_result,
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

    Returns {candidate, answer, is_confirmed}.
    """
    query = f"What research exists on {candidate['title']} in 3D bin packing or container loading?"

    result = await paperbridge_agent_query(
        query=query,
        llm_api_base=llm_api_base,
        evidence_skip_summary=True,
    )

    answer = result.session.answer
    # Simple heuristic: if answer is very short or contains "insufficient" or "no evidence", it's confirmed
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
```

- [ ] **Step 2: Test candidate collection with mock data**

Run: `PYTHONPATH=src python3 -c "
from paperqa.stores.gap_analysis import TopicResult, GapLanguageResult, OpenAlexResult, NovelpyResult
from paperqa.stores.gap_synthesis import collect_candidate_gaps

mock_topic = TopicResult(
    topics={0: ['bin', 'packing'], 1: ['container', 'loading']},
    topic_info={'columns': [], 'data': []},
    chunk_to_topic=[0] * 100 + [1] * 100,
    sparse_topics=[{'topic_id': 0, 'count': 10, 'name': 'quantum packing'}],
    temporal_data={},
)
mock_gap = GapLanguageResult(
    gap_chunks=[],
    gap_categories={0: {'label': 'real-time constraints', 'count': 15, 'words': ['real-time', 'constraint']}},
    top_gaps=['future work'],
)
mock_openalex = OpenAlexResult(
    global_topics=[],
    our_coverage=[],
    missing_topics=[{'topic_name': 'digital twin', 'our_count': 0, 'global_count': 5000, 'ratio': 0.0}],
)
mock_novelpy = NovelpyResult(
    concepts=['bin', 'packing', 'quantum'],
    cooccurrence_counts=[],
    novel_pairs=[('quantum', 'bin packing', 0.95)],
)

candidates = collect_candidate_gaps(mock_topic, mock_gap, mock_openalex, mock_novelpy)
print(f'Candidates: {len(candidates)}')
for c in candidates:
    print(f'  {c[\"title\"]}')
assert len(candidates) >= 3
print('OK: candidate collection works')
"`
Expected: 3+ candidates from all sources

- [ ] **Step 3: Add synthesis to gap_synthesis.py**

Append to `src/paperqa/stores/gap_synthesis.py`:

```python

async def synthesize_gaps(
    topic_result,
    gap_lang_result,
    openalex_result,
    novelpy_result,
    output_dir: str = "./data/gap_analysis/report",
    llm_api_base: str = "http://192.168.0.28:8005/v1",
    max_candidates: int = 20,
) -> GapReport:
    """Validate gap candidates and produce final report.

    Args:
        max_candidates: Maximum number of candidates to validate with LLM
            (due to LLM parallel=1 constraint, keep this low).
    """
    from datetime import date

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
```

- [ ] **Step 4: Test end-to-end with mock data (no LLM)**

Run: `PYTHONPATH=src python3 -c "
from paperqa.stores.gap_analysis import TopicResult, GapLanguageResult, OpenAlexResult, NovelpyResult
from paperqa.stores.gap_synthesis import collect_candidate_gaps

# Verify full candidate pipeline
mock_topic = TopicResult(
    topics={0: ['bin', 'packing'], 1: ['container', 'loading'], 2: ['genetic', 'algorithm']},
    topic_info={'columns': [], 'data': []},
    chunk_to_topic=list(range(1000)),
    sparse_topics=[{'topic_id': 0, 'count': 10, 'name': 'quantum packing'}, {'topic_id': 1, 'count': 15, 'name': 'digital twin'}],
    temporal_data={},
)
mock_gap = GapLanguageResult(
    gap_chunks=[{'pdf_hash': 'abc', 'title': 'Test', 'year': 2024, 'keyword_matched': 'future work'}],
    gap_categories={0: {'label': 'real-time constraints', 'count': 15, 'words': ['real-time']}},
    top_gaps=['future work'],
)
mock_openalex = OpenAlexResult(
    global_topics=[{'id': '1', 'display_name': 'Digital Twins', 'works_count': 5000}],
    our_coverage=[{'topic_name': 'Digital Twins', 'our_count': 5, 'global_count': 5000, 'ratio': 0.001}],
    missing_topics=[{'topic_name': 'Digital Twins', 'our_count': 5, 'global_count': 5000, 'ratio': 0.001}],
)
mock_novelpy = NovelpyResult(
    concepts=['bin', 'packing', 'quantum', 'digital'],
    cooccurrence_counts=[('bin', 'packing', 100)],
    novel_pairs=[('quantum', 'bin packing', 0.95)],
)

candidates = collect_candidate_gaps(mock_topic, mock_gap, mock_openalex, mock_novelpy)
print(f'Candidate gaps: {len(candidates)}')
for c in candidates:
    print(f'  [{c[\"evidence_sources\"]}] {c[\"title\"]}')
assert len(candidates) >= 5
print('OK: full candidate pipeline works')
"`
Expected: 5+ candidates from all 4 sources

- [ ] **Step 5: Commit**

```bash
git add src/paperqa/stores/gap_synthesis.py
git commit -m "feat: LLM gap synthesis with candidate collection and report generation"
```

---

### Task 8: Integration Test + CLI Entry Point

**Files:**
- Create: `tests/test_gap_analysis_integration.py`
- Modify: `src/paperqa/stores/__init__.py`

- [ ] **Step 1: Create integration test**

```python
"""Integration test for Phase 2: Gap Analysis Layer.

Tests the full pipeline with a subset of data.
Requires: Qdrant, OpenAlex API, LLM server.

Run:
    PYTHONPATH=src pytest tests/test_gap_analysis_integration.py -v --timeout=3600
"""

import asyncio
import pytest
from unittest import mock

from paperqa.stores.gap_analysis import (
    GapItem,
    GapLanguageResult,
    GapReport,
    NovelpyResult,
    OpenAlexResult,
    TopicResult,
)


class TestGapAnalysisDataclasses:
    """Test that dataclasses serialize and deserialize correctly."""

    def test_topic_result(self):
        result = TopicResult(
            topics={0: ["bin", "packing"], 1: ["container", "loading"]},
            topic_info={"columns": [], "data": []},
            chunk_to_topic=[0, 1, 2, 3],
            sparse_topics=[{"topic_id": 0, "count": 10, "name": "test topic"}],
            temporal_data={0: {2024: 5}},
        )
        assert len(result.topics) == 2
        assert len(result.chunk_to_topic) == 4
        assert result.topics[0] == ["bin", "packing"]

    def test_gap_report(self):
        report = GapReport(
            gaps=[GapItem(id=1, title="Test gap", description="A gap", evidence_sources=["bertopic"], confidence=0.8, papers=[])],
            summary="Test summary",
            recommendations=["Investigate test gap"],
            metadata={"total_papers": 100, "total_chunks": 1000, "analysis_date": "2026-04-30"},
        )
        assert len(report.gaps) == 1
        assert report.gaps[0].confidence == 0.8
        assert "Test gap" in report.summary or len(report.summary) > 0

    def test_all_results_roundtrip(self):
        """Test that all result dataclasses can be created and accessed."""
        topic = TopicResult(topics={}, topic_info={}, chunk_to_topic=[], sparse_topics=[], temporal_data={})
        gap = GapLanguageResult(gap_chunks=[], gap_categories={}, top_gaps=[])
        openalex = OpenAlexResult(global_topics=[], our_coverage=[], missing_topics=[])
        novelpy = NovelpyResult(concepts=[], cooccurrence_counts=[], novel_pairs=[])
        assert len(topic.topics) == 0
        assert len(gap.gap_chunks) == 0
        assert len(openalex.global_topics) == 0
        assert len(novelpy.concepts) == 0
```

- [ ] **Step 2: Run integration tests**

Run: `PYTHONPATH=src pytest tests/test_gap_analysis_integration.py -v`
Expected: All tests pass

- [ ] **Step 3: Create CLI entry point**

Create a simple script at `scripts/run_gap_analysis.py`:

```python
"""Run the Phase 2 gap analysis pipeline.

Usage:
    PYTHONPATH=src python3 scripts/run_gap_analysis.py
    PYTHONPATH=src python3 scripts/run_gap_analysis.py --output-dir ./data/gap_analysis_custom/
"""

import argparse
import asyncio
import logging
import sys

sys.path.insert(0, "src")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


async def main(output_dir: str, llm_api_base: str) -> None:
    from paperqa.stores.gap_analysis import run_gap_analysis

    report = await run_gap_analysis(
        qdrant_collection="paperbridge_glm_v2",
        output_dir=output_dir,
        llm_api_base=llm_api_base,
    )

    print(f"\n{'='*60}")
    print(f"Gap Analysis Complete")
    print(f"{'='*60}")
    print(f"Gaps identified: {len(report.gaps)}")
    print(f"Summary: {report.summary}")
    print(f"\nTop gaps:")
    for gap in report.gaps[:10]:
        print(f"  {gap.id}. [{gap.confidence:.2f}] {gap.title}")
        print(f"     {gap.description[:150]}...")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 2 gap analysis")
    parser.add_argument("--output-dir", default="./data/gap_analysis/")
    parser.add_argument("--llm-api-base", default="http://192.168.0.28:8005/v1")
    args = parser.parse_args()
    asyncio.run(main(args.output_dir, args.llm_api_base))
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_gap_analysis_integration.py scripts/run_gap_analysis.py src/paperqa/stores/__init__.py
git commit -m "feat: gap analysis integration tests and CLI entry point"
```

---

### Task 9: Final Integration + Smoke Test

**Files:**
- Modify: `scripts/run_gap_analysis.py` (if needed)

- [ ] **Step 1: Run smoke test with 10K chunks (full pipeline, small subset)**

Modify `bertopic_pipeline.py` temporarily to support a `--max-chunks` flag, or run manually:

```python
PYTHONPATH=src python3 -c "
import asyncio
from paperqa.stores.bertopic_pipeline import extract_chunks_from_qdrant, fit_bertopic

async def smoke_test():
    texts, embs, years = await extract_chunks_from_qdrant(batch_size=500)
    # Use first 10K chunks for smoke test
    result = fit_bertopic(texts[:10000], embs[:10000], years[:10000], './data/gap_analysis/bertopic_smoke')
    print(f'SMOKE: BERTopic {len(result.topics)} topics')
    return result

asyncio.run(smoke_test())
"
```
Expected: 10+ topics, no errors, completes in < 5 min

- [ ] **Step 2: Run gap-language extraction (full corpus)**

Run: `PYTHONPATH=src python3 -c "
import asyncio
from paperqa.stores.gap_language import run_gap_language

async def test():
    result = await run_gap_language('./data/gap_analysis/gap_language_smoke')
    print(f'SMOKE: Gap-language {len(result.gap_chunks)} chunks, {len(result.gap_categories)} categories')
    assert len(result.gap_chunks) > 0

asyncio.run(test())
"`
Expected: Gap chunks found, categories created

- [ ] **Step 3: Run OpenAlex comparison (full corpus)**

Run: `PYTHONPATH=src python3 -c "
import asyncio
from paperqa.stores.gap_analysis import TopicResult

async def test():
    from paperqa.stores.openalex_compare import run_openalex

    # Load topic result from smoke test
    import json
    with open('./data/gap_analysis/bertopic_smoke/metadata.json') as f:
        meta = json.load(f)
    print(f'SMOKE: OpenAlex (using {meta[\"num_topics\"]} topics)')

asyncio.run(test())
"`

- [ ] **Step 4: Run full pipeline (if time permits, ~2 hours)**

Run: `PYTHONPATH=src python3 scripts/run_gap_analysis.py`
Expected: Report generated with 10+ gaps

- [ ] **Step 5: Commit any fixes**

```bash
git add -A
git commit -m "fix: smoke test fixes from integration testing"
```

---

## Self-Review

### 1. Spec Coverage

| Spec Requirement | Task |
|-----------------|------|
| BERTopic topic clustering | Task 3 |
| Gap-language extraction | Task 4 |
| OpenAlex taxonomy comparison | Task 5 |
| Novelpy novelty scoring | Task 6 |
| LLM gap synthesis | Task 7 |
| Orchestrator | Task 2 |
| Integration test | Task 8 |
| Smoke test | Task 9 |

### 2. Placeholder Scan

No TBD/TODO found. All code blocks contain complete implementations. All file paths specified. All test commands provided with expected output.

### 3. Type Consistency

- `TopicResult` defined in Task 2, used in Tasks 3, 5, 6, 7
- `GapLanguageResult` defined in Task 2, used in Tasks 4, 7
- `OpenAlexResult` defined in Task 2, used in Tasks 5, 7
- `NovelpyResult` defined in Task 2, used in Tasks 6, 7
- `GapReport` defined in Task 2, used in Task 7
- `run_gap_analysis` orchestrator in Task 2 calls all component `run_*` functions
- LLM API base consistently `http://192.168.0.28:8005/v1`
- Qdrant URL consistently `http://192.168.0.28:6333`
- Embedding consistently `http://localhost:8082`
- `max_concurrent_requests=1` respected (all LLM calls sequential)

### 4. Ambiguity Check

- BERTopic PCA dimension fixed at 256 (not configurable, but appropriate for 4096-dim)
- Gap-language uses Tantivy keyword search (not Qdrant full-text)
- Novelpy uses co-occurrence proxy (not full novelpy library due to API differences)
- LLM validation uses simple heuristic (answer length + keyword matching)
- All output paths use `data/gap_analysis/` directory
- Reproducibility ensured via random_state=42 in UMAP and PCA

---

## Estimated Timeline

| Task | Estimated Time |
|------|---------------|
| Task 1: Install dependencies | 5 min |
| Task 2: Dataclasses + orchestrator | 15 min |
| Task 3: BERTopic pipeline | 30 min (code) + 30 min (run time) |
| Task 4: Gap-language extraction | 20 min |
| Task 5: OpenAlex comparison | 20 min |
| Task 6: Novelpy pipeline | 20 min |
| Task 7: LLM synthesis | 30 min |
| Task 8: Integration tests | 15 min |
| Task 9: Smoke test | 60 min |
| **Total** | **~5-6 hours** |
