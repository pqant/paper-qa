"""PaperBridge agent pipeline — mirrors PaperQA2's fake agent using our stores.

Provides:
- paperbridge_agent_query(): full agent pipeline (search → evidence → answer)
- paperbridge_contracrow(): contradiction detection via ContraCrow prompts
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from lmi import Embeddable, EmbeddingModel, EmbeddingModes, LiteLLMModel
from lmi.embeddings import LiteLLMEmbeddingModel

from paperqa.agents.helpers import litellm_get_search_query
from paperqa.agents.models import AgentStatus, AnswerResponse
from paperqa.agents.search import SearchIndex
from paperqa.docs import Docs
from paperqa.llms import VectorStore
from paperqa.settings import Settings
from paperqa.types import Doc, DocDetails, Text

from .paperbridge_store import PaperBridgeDocs, PaperBridgeQdrantStore, parse_pdf_name

logger = logging.getLogger(__name__)


def _build_llm_config(
    llm: str = "openai/qwen3.6-35b-a3b",
    api_base: str = "http://192.168.0.28:8005/v1",
    temperature: float = 0.0,
    timeout: int = 1800,
) -> dict:
    """Build LiteLLM model_list config for local LLM."""
    import os

    os.environ.setdefault("OPENAI_API_KEY", "sk-local-dummy")
    return {
        "name": llm,
        "model_list": [
            {
                "model_name": llm,
                "litellm_params": {
                    "model": llm,
                    "api_key": os.environ.get("OPENAI_API_KEY", "sk-local-dummy"),
                    "api_base": api_base,
                    "temperature": temperature,
                    "timeout": timeout,
                },
            }
        ],
    }


def _build_embedding_config(
    api_base: str = "http://localhost:8082",
) -> LiteLLMEmbeddingModel:
    """Build embedding model pointing to our Ollama server."""
    return LiteLLMEmbeddingModel(
        name="ollama/hf.co/Qwen/Qwen3-Embedding-8B-GGUF:F16",
        config={"kwargs": {"api_base": api_base}},
    )


def _build_settings(
    llm: str = "openai/qwen3.6-35b-a3b",
    llm_api_base: str = "http://192.168.0.28:8005/v1",
    temperature: float = 0.0,
    evidence_k: int = 5,
    evidence_skip_summary: bool = False,
) -> Settings:
    """Build a complete Settings object for our infrastructure."""
    llm_config = _build_llm_config(llm, llm_api_base, temperature)
    settings = Settings(
        llm=llm,
        summary_llm=llm,
        llm_config=llm_config,
        summary_llm_config=llm_config,
        temperature=temperature,
    )
    settings.answer.evidence_k = evidence_k
    settings.answer.evidence_skip_summary = evidence_skip_summary
    settings.answer.evidence_retrieval = True
    settings.answer.answer_max_sources = 5
    settings.answer.max_concurrent_requests = 1  # LLM parallel=1
    return settings


async def _populate_docs_from_search(
    docs: PaperBridgeDocs,
    store: PaperBridgeQdrantStore,
    search_index: SearchIndex | None,
    query: str,
    llm_model: LiteLLMModel,
    embedding_model: EmbeddingModel,
    search_count: int = 3,
    semantic_k: int = 15,
    max_chunks_per_paper: int = 10,
) -> None:
    """Populate docs.texts with chunks from relevant papers.

    Uses LLM-proposed search queries + Tantivy keyword search + Qdrant semantic search.

    Args:
        docs: PaperBridgeDocs with docs registry already loaded.
        store: PaperBridgeQdrantStore for Qdrant access.
        search_index: Optional Tantivy index for keyword search.
        query: Original query/question.
        llm_model: LLM for generating search queries.
        embedding_model: Embedding model for semantic search.
        search_count: Number of LLM search queries to generate.
        semantic_k: Number of semantic results to retrieve per query.
        max_chunks_per_paper: Max chunks to load per PDF (limits context window).
    """
    # Step 1: Get LLM-proposed search queries
    search_queries = await litellm_get_search_query(query, llm=llm_model, count=search_count)
    logger.info("LLM search queries: %s", search_queries)

    # Step 2: Collect relevant PDF hashes from all sources
    relevant_hashes: set[str] = set()

    # 2a: Semantic search for each query
    for sq in search_queries:
        texts, _ = await store.similarity_search(sq, k=semantic_k, embedding_model=embedding_model)
        for t in texts:
            relevant_hashes.add(t.doc.dockey)

    # 2b: Tantivy keyword search (if index available)
    if search_index:
        try:
            for sq in search_queries:
                results = await search_index.query(sq, top_n=8)
                for r in results:
                    if isinstance(r, dict) and "pdf_hash" in r:
                        relevant_hashes.add(r["pdf_hash"])
        except Exception:
            logger.warning("Tantivy search failed, continuing with semantic only", exc_info=True)

    # Filter to only hashes we have in docs registry
    relevant_hashes = relevant_hashes & set(docs.docs.keys())
    logger.info("Found %d unique relevant PDFs (in registry)", len(relevant_hashes))
    if not relevant_hashes:
        logger.warning("No relevant PDFs found for query: %s", query)
        return

    # Step 3: Fetch chunks for relevant PDFs from Qdrant
    chunks = await store.query_chunks_by_pdf_hashes(relevant_hashes)

    # Step 4: Convert to Text objects, limiting chunks per paper
    chunks_per_paper: dict[str, int] = {}
    added_texts = 0
    for payload, vector in chunks:
        pdf_hash = payload.get("pdf_hash")
        if not pdf_hash or pdf_hash not in docs.docs:
            continue

        current_count = chunks_per_paper.get(pdf_hash, 0)
        if current_count >= max_chunks_per_paper:
            continue

        doc = docs.docs[pdf_hash]
        meta = parse_pdf_name(payload.get("pdf_name", ""))
        text = Text(
            text=payload.get("text", ""),
            name=f"{meta['docname']} pages {payload.get('page_num', 0)}-{payload.get('page_num', 0)}",
            media=[],
            doc=doc,
        )
        text.embedding = vector
        docs.texts.append(text)
        chunks_per_paper[pdf_hash] = current_count + 1
        added_texts += 1

    logger.info("Added %d text chunks to Docs (from %d papers)", added_texts, len(chunks_per_paper))


async def paperbridge_agent_query(
    query: str,
    settings: Settings | None = None,
    embedding_model: EmbeddingModel | None = None,
    search_index: SearchIndex | None = None,
    qdrant_collection: str = "paperbridge_glm_v2",
    llm_api_base: str = "http://192.168.0.28:8005/v1",
    evidence_skip_summary: bool = False,
) -> AnswerResponse:
    """Run the full agent pipeline: search → evidence → answer → complete.

    Mirrors PaperQA2's fake agent flow using our PaperBridge stores.

    Args:
        query: Question to answer.
        settings: Settings object. If None, built from defaults.
        embedding_model: Embedding model. If None, created automatically.
        search_index: Optional Tantivy index for keyword search.
        qdrant_collection: Qdrant collection name.
        llm_api_base: LLM API base URL.
        evidence_skip_summary: Skip evidence summarization (faster, less accurate).

    Returns:
        AnswerResponse with session and status.
    """
    if settings is None:
        settings = _build_settings(
            llm_api_base=llm_api_base,
            evidence_skip_summary=evidence_skip_summary,
        )

    if embedding_model is None:
        embedding_model = _build_embedding_config()

    llm_model = settings.get_llm()

    # Create Docs and Store
    store = PaperBridgeQdrantStore(collection_name=qdrant_collection)
    docs = PaperBridgeDocs(texts_index=store)

    # Load docs registry from Qdrant
    await docs.load_docs_from_qdrant(batch_size=200)
    logger.info("Loaded %d docs from Qdrant", len(docs.docs))

    # Populate docs.texts with chunks from relevant papers
    await _populate_docs_from_search(
        docs=docs,
        store=store,
        search_index=search_index,
        query=query,
        llm_model=llm_model,
        embedding_model=embedding_model,
    )

    # Gather evidence
    session = await docs.aget_evidence(
        query=query,
        settings=settings,
        embedding_model=embedding_model,
    )
    logger.info("Gathered %d evidence contexts", len(session.contexts))

    # Generate answer using the session with gathered evidence
    session = await docs.aquery(
        query=session,
        settings=settings,
        embedding_model=embedding_model,
    )

    status = AgentStatus.SUCCESS if session.answer and len(session.answer) > 10 else AgentStatus.UNSURE

    return AnswerResponse(session=session, status=status)


async def paperbridge_contracrow(
    claim: str,
    settings: Settings | None = None,
    embedding_model: EmbeddingModel | None = None,
    search_index: SearchIndex | None = None,
    llm_api_base: str = "http://192.168.0.28:8005/v1",
) -> AnswerResponse:
    """Check if a claim is supported, contradicted, or has insufficient evidence.

    Uses ContraCrow-style prompts for contradiction detection.

    Args:
        claim: The claim to verify against the corpus.
        settings: Settings object. If None, built with ContraCrow-style prompts.
        embedding_model: Embedding model. If None, created automatically.
        search_index: Optional Tantivy index for keyword search.
        llm_api_base: LLM API base URL.

    Returns:
        AnswerResponse with contradiction assessment.
    """
    if settings is None:
        settings = _build_settings(
            llm_api_base=llm_api_base,
            evidence_k=15,
            evidence_skip_summary=True,
        )
        # Override QA prompt for contradiction detection
        settings.prompts.qa = (
            "Determine if the claim below is supported or contradicted by the context below.\n\n"
            "{context}\n\n"
            "----\n\n"
            "Claim: {question}\n\n"
            "Evaluate whether the claim is supported, contradicted, or if there is "
            "insufficient evidence. For each part of your response, cite sources using "
            "citation keys (pqac-1234abcd) from the context only.\n\n"
            "Provide your reasoning, then conclude with one of:\n"
            "- SUPPORT: The context clearly supports the claim\n"
            "- CONTRADICT: The context contradicts the claim\n"
            "- INSUFFICIENT_EVIDENCE: The context does not contain enough information\n\n"
            "Be specific and cite evidence from the context."
        )
        settings.prompts.summary = (
            "Provide a summary of the text below that could help determine "
            "if a claim is supported or contradicted.\n\n"
            "Text:\n{text}\n\n"
            "Summary of information relevant to evaluating claims ({summary_length}):"
        )

    return await paperbridge_agent_query(
        query=claim,
        settings=settings,
        embedding_model=embedding_model,
        search_index=search_index,
        llm_api_base=llm_api_base,
        evidence_skip_summary=True,
    )
