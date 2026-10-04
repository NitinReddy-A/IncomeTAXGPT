"""Runtime configuration, read from environment variables and an optional ``.env`` file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """All tunable settings for ingestion and question answering.

    Relative paths are resolved against the current working directory, so run the
    app and the CLI tools from the repository root.
    """

    # Vector database
    pinecone_api_key: str | None = None
    pinecone_index_name: str = "indiantaxgpt"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Embeddings (computed locally)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Local LLM
    llm_model_path: Path = Path("models/llama-2-7b-chat.Q4_K_M.gguf")
    llm_model_type: str = "llama"
    llm_max_new_tokens: int = 512
    llm_temperature: float = 0.1
    llm_context_length: int = 4096

    # Ingestion and retrieval
    data_dir: Path = Path("data")
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retriever_k: int = 4

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ConfigError("CHUNK_SIZE must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ConfigError("CHUNK_OVERLAP must be between 0 and CHUNK_SIZE")
        if self.retriever_k < 1:
            raise ConfigError("RETRIEVER_K must be at least 1")
        if not 0.0 <= self.llm_temperature <= 2.0:
            raise ConfigError("LLM_TEMPERATURE must be between 0 and 2")

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> Settings:
        """Build settings from the environment, loading ``env_file`` first if it exists.

        Variables already set in the environment take precedence over the file.
        """
        if env_file is not None and Path(env_file).is_file():
            load_dotenv(env_file, override=False)

        defaults = cls()
        return cls(
            pinecone_api_key=os.environ.get("PINECONE_API_KEY", "").strip() or None,
            pinecone_index_name=_env_str("PINECONE_INDEX_NAME", defaults.pinecone_index_name),
            pinecone_cloud=_env_str("PINECONE_CLOUD", defaults.pinecone_cloud),
            pinecone_region=_env_str("PINECONE_REGION", defaults.pinecone_region),
            embedding_model=_env_str("EMBEDDING_MODEL", defaults.embedding_model),
            llm_model_path=Path(_env_str("LLM_MODEL_PATH", str(defaults.llm_model_path))),
            llm_model_type=_env_str("LLM_MODEL_TYPE", defaults.llm_model_type),
            llm_max_new_tokens=_env_int("LLM_MAX_NEW_TOKENS", defaults.llm_max_new_tokens),
            llm_temperature=_env_float("LLM_TEMPERATURE", defaults.llm_temperature),
            llm_context_length=_env_int("LLM_CONTEXT_LENGTH", defaults.llm_context_length),
            data_dir=Path(_env_str("DATA_DIR", str(defaults.data_dir))),
            chunk_size=_env_int("CHUNK_SIZE", defaults.chunk_size),
            chunk_overlap=_env_int("CHUNK_OVERLAP", defaults.chunk_overlap),
            retriever_k=_env_int("RETRIEVER_K", defaults.retriever_k),
        )

    def require_pinecone_key(self) -> str:
        if not self.pinecone_api_key:
            raise ConfigError(
                "PINECONE_API_KEY is not set. Copy .env.example to .env and add your key "
                "from https://app.pinecone.io."
            )
        return self.pinecone_api_key
