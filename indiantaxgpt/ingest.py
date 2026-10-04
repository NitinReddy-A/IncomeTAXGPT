"""Build the knowledge base: load PDFs, split them into chunks, embed and upsert to Pinecone.

Usage:
    python -m indiantaxgpt.ingest              # index everything under DATA_DIR
    python -m indiantaxgpt.ingest --dry-run    # only load and chunk, print stats
    python -m indiantaxgpt.ingest --reset      # rebuild the index from scratch
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from indiantaxgpt.config import ConfigError, Settings

logger = logging.getLogger(__name__)

UPSERT_BATCH_SIZE = 64


def find_pdfs(data_dir: Path) -> list[Path]:
    if not data_dir.is_dir():
        raise ConfigError(f"Data directory {data_dir} does not exist")
    pdfs = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() == ".pdf")
    if not pdfs:
        raise ConfigError(f"No PDF files found in {data_dir}. See data/README.md.")
    return pdfs


def load_pdfs(data_dir: Path) -> list[Document]:
    """Load every page of every PDF under ``data_dir`` (recursively).

    Only the metadata needed for citations is kept, and ``source`` is stored relative to
    ``data_dir`` so chunk IDs stay stable across machines.
    """
    from pypdf import PdfReader

    pages: list[Document] = []
    for pdf in find_pdfs(data_dir):
        source = pdf.relative_to(data_dir).as_posix()
        reader = PdfReader(pdf)
        pages.extend(
            Document(page_content=page.extract_text() or "", metadata={"source": source, "page": i})
            for i, page in enumerate(reader.pages)
        )
        logger.info("Loaded %s (%d pages)", source, len(reader.pages))
    return pages


def split_documents(
    docs: Sequence[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(list(docs))
    # Scanned pages without a text layer come through as whitespace.
    return [chunk for chunk in chunks if chunk.page_content.strip()]


def chunk_id(chunk: Document) -> str:
    """Deterministic ID, so re-running ingestion updates vectors instead of duplicating them."""
    meta = chunk.metadata
    key = "|".join(
        [
            str(meta.get("source", "")),
            str(meta.get("page", "")),
            str(meta.get("start_index", "")),
            chunk.page_content,
        ]
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def ingest(settings: Settings, *, dry_run: bool = False, reset: bool = False) -> int:
    """Run the full pipeline and return the number of chunks indexed (or that would be)."""
    pages = load_pdfs(settings.data_dir)
    chunks = split_documents(pages, settings.chunk_size, settings.chunk_overlap)
    logger.info("Split %d pages into %d chunks", len(pages), len(chunks))
    if dry_run:
        return len(chunks)

    from langchain_pinecone import PineconeVectorStore

    from indiantaxgpt.embeddings import load_embeddings
    from indiantaxgpt.vectorstore import ensure_index

    settings.require_pinecone_key()
    embeddings = load_embeddings(settings.embedding_model)
    dimension = len(embeddings.embed_query("dimension probe"))
    index = ensure_index(settings, dimension, recreate=reset)

    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
    vectorstore.add_documents(
        chunks, ids=[chunk_id(c) for c in chunks], batch_size=UPSERT_BATCH_SIZE
    )
    logger.info("Upserted %d chunks into index %r", len(chunks), settings.pinecone_index_name)
    return len(chunks)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="indiantaxgpt-ingest",
        description="Index tax PDFs into Pinecone for IndianTaxGPT.",
    )
    parser.add_argument(
        "--data-dir", type=Path, help="Folder containing PDFs (default: DATA_DIR or ./data)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Load and chunk only; don't touch Pinecone"
    )
    parser.add_argument(
        "--reset", action="store_true", help="Delete and recreate the index before upserting"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        settings = Settings.from_env()
        if args.data_dir is not None:
            settings = dataclasses.replace(settings, data_dir=args.data_dir)
        count = ingest(settings, dry_run=args.dry_run, reset=args.reset)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    verb = "would index" if args.dry_run else "indexed"
    print(f"Done: {verb} {count} chunks from {settings.data_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
