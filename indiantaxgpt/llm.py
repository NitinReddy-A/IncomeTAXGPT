"""Local LLM. Inference runs on the CPU through ctransformers, wrapped as a LangChain LLM."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk

from indiantaxgpt.config import ConfigError, Settings


class LocalLLM(LLM):
    """LangChain adapter for a ctransformers model, with token streaming.

    ``model`` is any callable with the ctransformers ``LLM.__call__`` signature, which keeps
    this class easy to test without loading a multi-gigabyte model.
    """

    model: Any
    max_new_tokens: int = 512
    temperature: float = 0.1

    @property
    def _llm_type(self) -> str:
        return "ctransformers"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"max_new_tokens": self.max_new_tokens, "temperature": self.temperature}

    def _stream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        tokens = self.model(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            stop=stop,
            stream=True,
        )
        for token in tokens:
            chunk = GenerationChunk(text=token)
            if run_manager is not None:
                run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        return "".join(chunk.text for chunk in self._stream(prompt, stop, run_manager, **kwargs))


def load_llm(settings: Settings) -> LocalLLM:
    """Load the quantized GGUF/GGML model file configured in ``settings``."""
    model_path = settings.llm_model_path
    if not model_path.is_file():
        raise ConfigError(
            f"Model file not found at {model_path}. "
            "Download one with `python -m indiantaxgpt.download_model` "
            "or point LLM_MODEL_PATH at an existing file."
        )

    from ctransformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        model_type=settings.llm_model_type,
        context_length=settings.llm_context_length,
    )
    return LocalLLM(
        model=model,
        max_new_tokens=settings.llm_max_new_tokens,
        temperature=settings.llm_temperature,
    )
