"""Sentence-embedding model used for both indexing and querying."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


def load_embeddings(model_name: str) -> Embeddings:
    """Load a sentence-transformers model. It is downloaded from Hugging Face on first use."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )
