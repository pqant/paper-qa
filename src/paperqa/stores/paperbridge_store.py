"""PaperBridge Qdrant adapter — read-only VectorStore for existing corpus."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Sequence

from lmi import Embeddable, EmbeddingModel, EmbeddingModes
from pydantic import ConfigDict, Field
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue, Record

from paperqa.docs import Docs
from paperqa.llms import VectorStore
from paperqa.types import Doc, DocDetails, Text

logger = logging.getLogger(__name__)


def parse_pdf_name(pdf_name: str) -> dict[str, Any]:
    """Extract metadata from PaperBridge filename convention.

    Filename format: {year}__{doi_encoded}__{title}.pdf
    Example: 2025__arxiv_2503.08863__Improved Approximation Algorithms.pdf

    Args:
        pdf_name: Filename from Qdrant payload.

    Returns:
        Dict with keys: year, doi, title, citation, docname.
    """
    name = pdf_name.replace(".pdf", "")
    parts = name.split("__", 2)
    year = int(parts[0]) if parts[0].isdigit() else None
    doi_raw = parts[1] if len(parts) > 1 else None
    doi = doi_raw.replace("_", "/") if doi_raw else None
    title = parts[2].strip() if len(parts) > 2 else name
    citation = f"{title}, {year}." if year else title
    return {
        "year": year,
        "doi": doi,
        "title": title,
        "citation": citation,
        "docname": title[:80],
    }


class PaperBridgeQdrantStore(VectorStore):
    """Read-only VectorStore adapter for PaperBridge's Qdrant collection.

    Reads from our Qdrant payload schema and converts to PaperQA2 Text/DocDetails
    objects on the fly during similarity_search. Never writes to Qdrant.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Any = Field(default=None)
    collection_name: str = "paperbridge_glm_v2"

    def __init__(self, **data):
        super().__init__(**data)
        if self.client is None:
            self.client = AsyncQdrantClient(url="http://192.168.0.28:6333")

    # -- Read-only enforcement --

    async def add_texts_and_embeddings(self, texts) -> None:
        """READ-ONLY: Do not add to our collection."""
        raise NotImplementedError("PaperBridgeQdrantStore is read-only")

    def clear(self) -> None:
        """READ-ONLY: Do not clear our collection."""
        pass

    # -- Size helpers — override parent's __len__ which checks texts_hashes --

    def __len__(self) -> int:
        """Return approximate size. Parent's __len__ checks texts_hashes (always empty for us)."""
        # We return a non-zero sentinel so aget_evidence's emptiness check doesn't
        # early-return. The real count is available via _collection_count().
        return 1

    async def _collection_count(self) -> int:
        """Return the number of points in the Qdrant collection."""
        info = await self.client.get_collection(self.collection_name)
        return info.points_count or 0

    # -- Payload → Text conversion --

    def _payload_to_text(
        self, payload: dict, vector: list[float], doc: DocDetails
    ) -> Text:
        """Convert a Qdrant payload to a PaperQA2 Text object.

        Args:
            payload: Our Qdrant chunk payload.
            vector: The embedding vector from Qdrant.
            doc: Pre-constructed DocDetails (shared across chunks of same PDF).

        Returns:
            Text object compatible with PaperQA2 pipeline.
        """
        meta = parse_pdf_name(payload["pdf_name"])
        text = Text(
            text=payload["text"],
            name=f"{meta['docname']} pages {payload['page_num']}-{payload['page_num']}",
            media=[],
            doc=doc,
        )
        text.embedding = vector
        return text

    def _payload_to_doc(self, payload: dict) -> DocDetails:
        """Convert a Qdrant payload to a PaperQA2 DocDetails object.

        Args:
            payload: Our Qdrant chunk payload.

        Returns:
            DocDetails object representing the PDF.
        """
        meta = parse_pdf_name(payload["pdf_name"])
        # Exclude 'docname' and 'citation' from overwrite — DocDetails' validator
        # would regenerate them from bibtex metadata, which we don't have.
        return DocDetails(
            docname=meta["docname"],
            dockey=payload["pdf_hash"],
            citation=meta["citation"],
            content_hash=payload["pdf_hash"],
            year=meta["year"],
            doi=meta["doi"],
            title=meta["title"],
            file_location=payload.get("pdf_path"),
            fields_to_overwrite_from_metadata=set(),
        )

    # -- Core VectorStore method --

    async def similarity_search(
        self,
        query: str,
        k: int,
        embedding_model: EmbeddingModel,
    ) -> tuple[Sequence[Embeddable], list[float]]:
        """Search our Qdrant collection, return PaperQA2 Text objects.

        Args:
            query: Search query string.
            k: Number of results to return.
            embedding_model: Model to embed the query (must produce 4096-dim vectors).

        Returns:
            Tuple of (list of Text objects, list of similarity scores).
        """
        # Embed the query — respect embedding modes (QUERY vs DOCUMENT)
        try:
            embedding_model.set_mode(EmbeddingModes.QUERY)
        except (AttributeError, TypeError):
            pass

        query_embedding = (await embedding_model.embed_documents([query]))[0]

        try:
            embedding_model.set_mode(EmbeddingModes.DOCUMENT)
        except (AttributeError, TypeError):
            pass

        # Query Qdrant
        results = await self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=k,
            with_payload=True,
            with_vectors=True,
        )

        # Convert to Text objects, deduplicating Docs by pdf_hash
        doc_cache: dict[str, DocDetails] = {}
        texts: list[Text] = []
        scores: list[float] = []

        for point in results.points:
            payload = point.payload
            pdf_hash = payload["pdf_hash"]

            if pdf_hash not in doc_cache:
                doc_cache[pdf_hash] = self._payload_to_doc(payload)

            text = self._payload_to_text(payload, point.vector, doc_cache[pdf_hash])
            texts.append(text)
            scores.append(point.score)

        return texts, scores

    async def query_chunks_by_pdf_hashes(
        self,
        pdf_hashes: set[str],
        batch_size: int = 500,
    ) -> list[tuple[dict, list[float]]]:
        """Retrieve all chunks for given PDF hashes from Qdrant.

        Args:
            pdf_hashes: Set of SHA-256 hashes to fetch chunks for.
            batch_size: Scroll batch size.

        Returns:
            List of (payload, vector) tuples for matching chunks.
        """
        from qdrant_client.http.models import Filter, MatchAny

        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="pdf_hash",
                    match=MatchAny(any=list(pdf_hashes)),
                )
            ]
        )

        all_chunks: list[tuple[dict, list[float]]] = []
        offset = None
        seen_offsets: set[str] = set()

        while True:
            points, next_offset = await self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                scroll_filter=qdrant_filter,
                with_payload=["pdf_name", "pdf_hash", "pdf_path", "text", "page_num"],
                with_vectors=True,
            )
            if not points:
                break

            for point in points:
                all_chunks.append((point.payload, point.vector))

            if next_offset is None:
                break

            offset_str = str(next_offset)
            if offset_str in seen_offsets:
                break
            seen_offsets.add(offset_str)
            offset = next_offset

        logger.info("Fetched %d chunks for %d PDF hashes", len(all_chunks), len(pdf_hashes))
        return all_chunks

    async def similarity_search_with_filter(
        self,
        query: str,
        k: int,
        embedding_model: EmbeddingModel,
        pdf_hashes: set[str] | None = None,
    ) -> tuple[Sequence[Embeddable], list[float]]:
        """Like similarity_search but optionally filter to specific PDFs.

        Args:
            query: Search query string.
            k: Number of results.
            embedding_model: Embedding model.
            pdf_hashes: Optional set of PDF hashes to filter results to.
        """
        try:
            embedding_model.set_mode(EmbeddingModes.QUERY)
        except (AttributeError, TypeError):
            pass

        query_embedding = (await embedding_model.embed_documents([query]))[0]

        try:
            embedding_model.set_mode(EmbeddingModes.DOCUMENT)
        except (AttributeError, TypeError):
            pass

        qdrant_filter = None
        if pdf_hashes:
            from qdrant_client.http.models import Filter, MatchAny

            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="pdf_hash",
                        match=MatchAny(any=list(pdf_hashes)),
                    )
                ]
            )

        results = await self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=k,
            query_filter=qdrant_filter,
            with_payload=True,
            with_vectors=True,
        )

        doc_cache: dict[str, DocDetails] = {}
        texts: list[Text] = []
        scores: list[float] = []

        for point in results.points:
            payload = point.payload
            pdf_hash = payload["pdf_hash"]

            if pdf_hash not in doc_cache:
                doc_cache[pdf_hash] = self._payload_to_doc(payload)

            text = self._payload_to_text(payload, point.vector, doc_cache[pdf_hash])
            texts.append(text)
            scores.append(point.score)

        return texts, scores


class PaperBridgeDocs(Docs):
    """PaperQA2 Docs subclass that reads from PaperBridge Qdrant corpus.

    Overrides retrieve_texts() to bypass _build_texts_index() (which would try to
    embed all 163K chunks again) and queries our Qdrant store directly.
    """

    async def load_docs_from_qdrant(
        self,
        batch_size: int = 1000,
    ) -> None:
        """Populate self.docs and self.docnames by sequentially scrolling Qdrant.

        This is a one-time operation that extracts unique PDF metadata from our
        Qdrant collection. Called before any queries. Uses sequential scrolling
        to ensure all documents are loaded (parallel batch approach has race conditions).

        Args:
            batch_size: Number of points to scroll per batch.
        """
        store = self.texts_index
        if not isinstance(store, PaperBridgeQdrantStore):
            raise ValueError("texts_index must be a PaperBridgeQdrantStore")

        collection_info = await store.client.get_collection(store.collection_name)
        total_points = collection_info.points_count or 0
        logger.info("Loading docs from Qdrant collection with %d points", total_points)

        seen_hashes: set[str] = set()
        scrolled = 0
        offset = None
        seen_offsets: set[str] = set()

        while True:
            points, next_offset = await store.client.scroll(
                collection_name=store.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=["pdf_name", "pdf_hash", "pdf_path"],
                with_vectors=False,
            )
            if not points:
                break

            for point in points:
                payload = point.payload
                pdf_hash = payload.get("pdf_hash")
                if not pdf_hash or pdf_hash in seen_hashes:
                    continue
                seen_hashes.add(pdf_hash)

                meta = parse_pdf_name(payload["pdf_name"])
                doc = DocDetails(
                    docname=meta["docname"],
                    dockey=pdf_hash,
                    citation=meta["citation"],
                    content_hash=pdf_hash,
                    year=meta["year"],
                    doi=meta["doi"],
                    title=meta["title"],
                    file_location=payload.get("pdf_path"),
                    fields_to_overwrite_from_metadata=set(),
                )
                self.docs[doc.dockey] = doc
                self.docnames.add(doc.docname)

            scrolled += len(points)
            logger.info(
                "Scrolled %d points, loaded %d unique docs", scrolled, len(self.docs)
            )

            if next_offset is None:
                break

            offset_str = str(next_offset)
            if offset_str in seen_offsets:
                logger.warning("Detected repeated offset %s, stopping", offset_str)
                break
            seen_offsets.add(offset_str)
            offset = next_offset

        logger.info("Loaded %d unique docs from Qdrant", len(self.docs))

    async def retrieve_texts(
        self,
        query: str,
        k: int,
        settings=None,
        embedding_model=None,
        partitioning_fn=None,
    ) -> list[Text]:
        """Retrieve texts by querying our Qdrant store directly.

        Bypasses _build_texts_index() since our texts are already embedded in Qdrant.

        Args:
            query: Search query string.
            k: Number of results.
            settings: PaperQA2 settings (used for mmr_lambda).
            embedding_model: Embedding model for the query.
            partitioning_fn: Unused (our store doesn't support partitioning).

        Returns:
            List of Text objects with doc references.
        """
        from paperqa.settings import get_settings

        resolved_settings = get_settings(settings)
        if embedding_model is None:
            embedding_model = resolved_settings.get_embedding_model()

        # Set MMR lambda on our store (in case we add MMR support later)
        self.texts_index.mmr_lambda = resolved_settings.texts_index_mmr_lambda

        # Query our store directly — texts come back with embeddings from Qdrant
        texts, _scores = await self.texts_index.similarity_search(
            query, k, embedding_model
        )

        # Filter out deleted dockeys (same as parent class)
        texts = [t for t in texts if t.doc.dockey not in self.deleted_dockeys]
        return list(texts)[:k]
