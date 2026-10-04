"""Pinecone index management and the LangChain vector store wrapper around it."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from indiantaxgpt.config import ConfigError, Settings

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone
    from pinecone.db_data import Index

logger = logging.getLogger(__name__)


def _client(settings: Settings) -> Pinecone:
    from pinecone import Pinecone

    return Pinecone(api_key=settings.require_pinecone_key())


def ensure_index(settings: Settings, dimension: int, *, recreate: bool = False) -> Index:
    """Create the serverless index if needed, check its dimension, and return a handle to it."""
    from pinecone import ServerlessSpec

    pc = _client(settings)
    name = settings.pinecone_index_name
    exists = name in pc.list_indexes().names()

    if exists and recreate:
        logger.info("Deleting existing index %r", name)
        pc.delete_index(name)
        exists = False

    if not exists:
        logger.info(
            "Creating index %r (dimension=%d, %s/%s)",
            name,
            dimension,
            settings.pinecone_cloud,
            settings.pinecone_region,
        )
        pc.create_index(
            name=name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
        return pc.Index(name)

    existing_dimension = pc.describe_index(name).dimension
    if existing_dimension != dimension:
        raise ConfigError(
            f"Index {name!r} has dimension {existing_dimension}, but the embedding model "
            f"produces {dimension}. Re-run ingestion with --reset or use a different index name."
        )
    return pc.Index(name)


def load_vectorstore(settings: Settings, embeddings: Embeddings) -> PineconeVectorStore:
    """Connect to an existing index. Raises ``ConfigError`` if it hasn't been built yet."""
    from langchain_pinecone import PineconeVectorStore

    pc = _client(settings)
    name = settings.pinecone_index_name
    if name not in pc.list_indexes().names():
        raise ConfigError(
            f"Pinecone index {name!r} does not exist yet. "
            "Run `python -m indiantaxgpt.ingest` to build it."
        )
    return PineconeVectorStore(index=pc.Index(name), embedding=embeddings)
