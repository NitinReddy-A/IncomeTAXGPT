from pathlib import Path

import pytest

from indiantaxgpt.config import ConfigError, Settings


def test_defaults_when_environment_is_empty():
    settings = Settings.from_env()

    assert settings.pinecone_api_key is None
    assert settings.pinecone_index_name == "indiantaxgpt"
    assert settings.llm_model_path == Path("models/llama-2-7b-chat.Q4_K_M.gguf")
    assert settings.retriever_k == 4


def test_reads_environment_variables(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "  abc123  ")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "tax-2026")
    monkeypatch.setenv("RETRIEVER_K", "6")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.3")
    monkeypatch.setenv("DATA_DIR", "docs/pdfs")

    settings = Settings.from_env()

    assert settings.pinecone_api_key == "abc123"
    assert settings.pinecone_index_name == "tax-2026"
    assert settings.retriever_k == 6
    assert settings.llm_temperature == 0.3
    assert settings.data_dir == Path("docs/pdfs")


def test_loads_dotenv_file_without_overriding_real_environment(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("PINECONE_API_KEY=from-file\nPINECONE_INDEX_NAME=from-file\n")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "from-env")

    settings = Settings.from_env()

    assert settings.pinecone_api_key == "from-file"
    assert settings.pinecone_index_name == "from-env"


def test_blank_values_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "")
    monkeypatch.setenv("CHUNK_SIZE", " ")

    settings = Settings.from_env()

    assert settings.pinecone_api_key is None
    assert settings.chunk_size == 1000


def test_invalid_number_raises_config_error(monkeypatch):
    monkeypatch.setenv("RETRIEVER_K", "four")

    with pytest.raises(ConfigError, match="RETRIEVER_K"):
        Settings.from_env()


@pytest.mark.parametrize(
    "overrides",
    [
        {"chunk_size": 0},
        {"chunk_size": 100, "chunk_overlap": 100},
        {"retriever_k": 0},
        {"llm_temperature": 3.0},
    ],
)
def test_rejects_invalid_values(overrides):
    with pytest.raises(ConfigError):
        Settings(**overrides)


def test_require_pinecone_key():
    with pytest.raises(ConfigError, match="PINECONE_API_KEY"):
        Settings().require_pinecone_key()
    assert Settings(pinecone_api_key="k").require_pinecone_key() == "k"
