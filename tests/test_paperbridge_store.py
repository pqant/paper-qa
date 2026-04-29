"""Integration tests for PaperBridgeQdrantStore + PaperBridgeDocs.

Tests the full PaperQA2 pipeline against our PaperBridge Qdrant corpus.
Requires:
  - Qdrant at http://192.168.0.28:6333 (paperbridge_glm_v2 collection)
  - Ollama embedding server at http://localhost:8082
  - llama.cpp LLM at http://192.168.0.64:8080 (for aquery test)

Run:
  PYTHONPATH=src pytest tests/test_paperbridge_store.py -v --timeout=600
"""

import asyncio
from unittest import mock

import pytest

from lmi.embeddings import LiteLLMEmbeddingModel

from paperqa.stores import PaperBridgeDocs, PaperBridgeQdrantStore, parse_pdf_name
from paperqa.settings import Settings


# --- parse_pdf_name tests ---


class TestParsePDFName:
    def test_standard_filename(self):
        """Year__DOI__Title.pdf format."""
        result = parse_pdf_name("2025__arxiv_2503.08863__Improved Approximation Algorithms.pdf")
        assert result["year"] == 2025
        assert result["doi"] == "arxiv/2503.08863"
        assert result["title"] == "Improved Approximation Algorithms"
        assert result["docname"] == "Improved Approximation Algorithms"
        assert "2025" in result["citation"]

    def test_filename_without_doi(self):
        """Fallback when no DOI segment."""
        result = parse_pdf_name("2024__Title Only.pdf")
        assert result["year"] == 2024
        assert result["doi"] is None
        assert result["title"] == "Title Only"

    def test_filename_without_year(self):
        """Non-numeric year segment."""
        result = parse_pdf_name("no_year__doi__Title.pdf")
        assert result["year"] is None
        assert result["doi"] == "doi"


# --- PaperBridgeQdrantStore tests ---


class TestPaperBridgeQdrantStore:
    @pytest.mark.asyncio
    async def test_read_only_raises_on_add(self):
        store = PaperBridgeQdrantStore()
        with pytest.raises(NotImplementedError, match="read-only"):
            await store.add_texts_and_embeddings([])

    @pytest.mark.asyncio
    async def test_clear_is_noop(self):
        """clear() should not raise on read-only store."""
        store = PaperBridgeQdrantStore()
        store.clear()  # Should be silent

    @pytest.mark.asyncio
    async def test_len_returns_nonzero(self):
        """__len__ must return non-zero so aget_evidence doesn't early-return."""
        store = PaperBridgeQdrantStore()
        assert len(store) > 0

    @pytest.mark.asyncio
    async def test_collection_count(self):
        store = PaperBridgeQdrantStore()
        count = await store._collection_count()
        assert count > 0
        assert count > 10000  # Should have ~163K chunks

    @pytest.mark.asyncio
    async def test_similarity_search_returns_texts(self):
        store = PaperBridgeQdrantStore()
        emb_model = LiteLLMEmbeddingModel(
            name="ollama/hf.co/Qwen/Qwen3-Embedding-8B-GGUF:F16",
            config={"kwargs": {"api_base": "http://localhost:8082"}},
        )
        texts, scores = await store.similarity_search(
            "transformer attention mechanism",
            k=5,
            embedding_model=emb_model,
        )
        assert len(texts) == 5
        assert len(scores) == 5
        # Scores should be in descending order
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]
        # Texts should have doc references
        for t in texts:
            assert t.text  # Has content
            assert t.doc.docname  # Has doc reference
            assert hasattr(t, "embedding")  # Has embedding from Qdrant


# --- PaperBridgeDocs tests ---


class TestPaperBridgeDocs:
    @pytest.mark.asyncio
    async def test_load_docs_from_qdrant(self):
        docs = PaperBridgeDocs(texts_index=PaperBridgeQdrantStore())
        await docs.load_docs_from_qdrant(batch_size=200)
        assert len(docs.docs) > 0
        assert len(docs.docnames) > 0
        # Should have ~192 unique PDFs
        assert len(docs.docs) > 100

    @pytest.mark.asyncio
    async def test_load_docs_deduplicates(self):
        """Multiple chunks from same PDF should produce one DocDetails."""
        docs = PaperBridgeDocs(texts_index=PaperBridgeQdrantStore())
        await docs.load_docs_from_qdrant(batch_size=200)
        # All dockeys should be unique
        assert len(docs.docs) == len(set(docs.docs.keys()))

    @pytest.mark.asyncio
    async def test_retrieve_texts(self):
        emb_model = LiteLLMEmbeddingModel(
            name="ollama/hf.co/Qwen/Qwen3-Embedding-8B-GGUF:F16",
            config={"kwargs": {"api_base": "http://localhost:8082"}},
        )
        docs = PaperBridgeDocs(texts_index=PaperBridgeQdrantStore())
        await docs.load_docs_from_qdrant(batch_size=200)

        texts = await docs.retrieve_texts(
            query="attention mechanism in transformers",
            k=5,
            embedding_model=emb_model,
        )
        assert len(texts) == 5
        for t in texts:
            assert t.text
            assert t.doc.docname

    @pytest.mark.asyncio
    async def test_aget_evidence(self):
        """Full evidence retrieval pipeline."""
        emb_model = LiteLLMEmbeddingModel(
            name="ollama/hf.co/Qwen/Qwen3-Embedding-8B-GGUF:F16",
            config={"kwargs": {"api_base": "http://localhost:8082"}},
        )
        docs = PaperBridgeDocs(texts_index=PaperBridgeQdrantStore())
        await docs.load_docs_from_qdrant(batch_size=200)

        settings = Settings()
        settings.answer.evidence_k = 5
        settings.answer.evidence_skip_summary = True
        settings.answer.evidence_retrieval = True

        session = await docs.aget_evidence(
            query="What is multi-head attention in transformers?",
            settings=settings,
            embedding_model=emb_model,
        )
        assert len(session.contexts) > 0
        for ctx in session.contexts:
            assert ctx.text.text
            assert ctx.text.doc.docname

    @pytest.mark.asyncio
    async def test_aquery(self):
        """Full query -> evidence -> answer pipeline with LLM."""
        emb_model = LiteLLMEmbeddingModel(
            name="ollama/hf.co/Qwen/Qwen3-Embedding-8B-GGUF:F16",
            config={"kwargs": {"api_base": "http://localhost:8082"}},
        )
        docs = PaperBridgeDocs(texts_index=PaperBridgeQdrantStore())
        await docs.load_docs_from_qdrant(batch_size=200)

        settings = Settings()
        settings.answer.evidence_k = 3
        settings.answer.evidence_skip_summary = True
        settings.answer.evidence_retrieval = True
        settings.llm = "openai/Qwen3.6-27B"
        settings.summary_llm = "openai/Qwen3.6-27B"
        settings.llm_config = {
            "name": "openai/Qwen3.6-27B",
            "model_list": [
                {
                    "model_name": "openai/Qwen3.6-27B",
                    "litellm_params": {
                        "model": "openai/Qwen3.6-27B",
                        "api_base": "http://192.168.0.64:8080/v1",
                        "temperature": 0.0,
                        "timeout": 600,
                    },
                }
            ],
        }
        settings.summary_llm_config = settings.llm_config

        # Mock OPENAI_API_KEY if not set (required by openai/ prefix)
        import os
        orig_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "sk-dummy"
        try:
            session = await docs.aquery(
                query="What is multi-head attention in transformers?",
                settings=settings,
                embedding_model=emb_model,
            )
        finally:
            if orig_key is not None:
                os.environ["OPENAI_API_KEY"] = orig_key
            else:
                os.environ.pop("OPENAI_API_KEY", None)

        assert len(session.answer) > 100  # Should have a real answer
        assert len(session.contexts) > 0  # Should have evidence
