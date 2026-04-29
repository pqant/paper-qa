"""Build Tantivy full-text index from PaperBridge Qdrant corpus.

Groups chunks by PDF, combines text into "body", extracts metadata from filenames.
One-time build for ~192 unique PDFs (~163K chunks).

Usage:
    python3 scripts/build_tantivy_index.py
    python3 scripts/build_tantivy_index.py --dry-run
    python3 scripts/build_tantivy_index.py --index-dir ./data/tantivy_index
"""

import argparse
import asyncio
import logging
import os
from collections import defaultdict

from paperqa.agents.search import SearchIndex
from paperqa.stores import parse_pdf_name
from qdrant_client import AsyncQdrantClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FIELDS = ["file_location", "body", "title", "year"]
INDEX_NAME = "paperbridge_index"


async def build_index(
    index_dir: str = "./data/tantivy_index",
    dry_run: bool = False,
    batch_size: int = 500,
) -> None:
    client = AsyncQdrantClient(url="http://192.168.0.28:6333")
    collection = "paperbridge_glm_v2"

    info = await client.get_collection(collection)
    total = info.points_count or 0
    logger.info("Qdrant collection: %d points", total)

    # Group chunks by pdf_hash
    pdf_chunks: dict[str, list[dict]] = defaultdict(list)
    pdf_meta_cache: dict[str, dict] = {}

    # Scroll all points with text
    offset = None
    scrolled = 0
    seen_offsets: set[str] = set()
    while True:
        points, next_offset = await client.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=offset,
            with_payload=["pdf_name", "pdf_hash", "pdf_path", "text", "page_num"],
            with_vectors=False,
        )
        if not points:
            break

        # Safety: stop if next_offset is None (end of collection)
        if next_offset is None:
            # Process this last batch without continuing
            pass

        for point in points:
            payload = point.payload
            pdf_hash = payload.get("pdf_hash")
            if not pdf_hash:
                continue

            pdf_chunks[pdf_hash].append({
                "text": payload.get("text", ""),
                "page_num": payload.get("page_num", 0),
            })

            if pdf_hash not in pdf_meta_cache:
                pdf_meta_cache[pdf_hash] = {
                    "pdf_name": payload.get("pdf_name", ""),
                    "pdf_path": payload.get("pdf_path", ""),
                }

        scrolled += len(points)
        logger.info("Scrolled %d points (%d unique PDFs)", scrolled, len(pdf_chunks))

        if next_offset is None:
            break

        # Safety: detect infinite loop
        offset_str = str(next_offset)
        if offset_str in seen_offsets:
            logger.warning("Detected repeated offset %s, stopping", offset_str)
            break
        seen_offsets.add(offset_str)
        offset = next_offset

    logger.info(
        "Scrolled %d points, %d unique PDFs", scrolled, len(pdf_chunks)
    )

    if dry_run:
        logger.info("Dry run, not building index")
        return

    # Build Tantivy index (use absolute path to avoid CWD issues)
    index = SearchIndex(
        fields=FIELDS,
        index_name=INDEX_NAME,
        index_directory=os.path.abspath(index_dir),
    )

    added = 0
    errors = 0
    for pdf_hash, chunks in pdf_meta_cache.items():
        meta = parse_pdf_name(chunks.get("pdf_name", ""))
        body_text = "\n".join(
            c["text"]
            for c in sorted(pdf_chunks.get(pdf_hash, []), key=lambda x: x["page_num"])
            if c["text"]
        )

        if not body_text.strip():
            continue

        index_doc = {
            "file_location": chunks.get("pdf_path", f"pdf_{pdf_hash}"),
            "body": body_text,
            "title": meta.get("title", ""),
            "year": str(meta.get("year") or ""),
        }

        try:
            # Store metadata as document so query() can load it back
            doc_metadata = {
                "title": meta.get("title", ""),
                "year": meta.get("year"),
                "doi": meta.get("doi"),
                "file_location": chunks.get("pdf_path", ""),
                "pdf_hash": pdf_hash,
            }
            await index.add_document(index_doc, document=doc_metadata)
            added += 1
            if added % 10 == 0:
                logger.info("Added %d docs", added)
        except Exception:
            errors += 1
            logger.exception("Failed to index %s", meta.get("title", pdf_hash))

    # Commit changes
    if index.changed:
        await index.save_index()
        logger.info("Index saved")

    count = await index.count
    logger.info("Index complete: %d docs indexed, %d errors", added, errors)
    logger.info("Index document count: %d", count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", default="./data/tantivy_index")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    asyncio.run(build_index(args.index_dir, args.dry_run, args.batch_size))
