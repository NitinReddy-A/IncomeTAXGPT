import pytest

from indiantaxgpt.config import ConfigError, Settings
from indiantaxgpt.llm import LocalLLM, load_llm


class FakeModel:
    """Mimics the call signature of a ctransformers model."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.calls = []

    def __call__(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return iter(self.tokens)


def test_invoke_joins_streamed_tokens_and_passes_settings():
    model = FakeModel(["Section", " 80C", " applies."])
    llm = LocalLLM(model=model, max_new_tokens=128, temperature=0.2)

    assert llm.invoke("prompt text") == "Section 80C applies."
    ((prompt, kwargs),) = model.calls
    assert prompt == "prompt text"
    assert kwargs["max_new_tokens"] == 128
    assert kwargs["temperature"] == 0.2
    assert kwargs["stream"] is True


def test_stream_yields_each_token():
    llm = LocalLLM(model=FakeModel(["a", "b", "c"]))

    assert list(llm.stream("p")) == ["a", "b", "c"]


def test_load_llm_requires_model_file(tmp_path):
    settings = Settings(llm_model_path=tmp_path / "missing.gguf")

    with pytest.raises(ConfigError, match="Model file not found"):
        load_llm(settings)
