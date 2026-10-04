import pytest
from langchain_core.documents import Document
from langchain_core.language_models import FakeListLLM

from indiantaxgpt.assistant import (
    NO_CONTEXT_ANSWER,
    Source,
    TaxAssistant,
    build_assistant,
    format_context,
    to_sources,
)
from indiantaxgpt.config import ConfigError, Settings
from indiantaxgpt.prompts import build_prompt
from tests.conftest import StaticRetriever


class RecordingLLM(FakeListLLM):
    """Fake LLM that also keeps the prompts it was given."""

    prompts: list[str] = []

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        self.prompts.append(prompt)
        return super()._call(prompt, stop, run_manager, **kwargs)


def test_ask_returns_answer_with_deduplicated_sources(tax_docs):
    llm = RecordingLLM(responses=["  You can claim up to Rs 1.5 lakh under Section 80C.  "])
    assistant = TaxAssistant(retriever=StaticRetriever(docs=tax_docs), llm=llm)

    answer = assistant.ask("How much can I claim under 80C?")

    assert answer.text == "You can claim up to Rs 1.5 lakh under Section 80C."
    assert [s.label for s in answer.sources] == [
        "income-tax-act.pdf, page 42",
        "advance-tax.pdf, page 3",
    ]


def test_prompt_contains_question_and_numbered_context(tax_docs):
    llm = RecordingLLM(responses=["ok"])
    assistant = TaxAssistant(retriever=StaticRetriever(docs=tax_docs), llm=llm)

    assistant.ask("  When is advance tax due?  ")

    (prompt,) = llm.prompts
    assert prompt.startswith("[INST] <<SYS>>")
    assert "Question: When is advance tax due? [/INST]" in prompt
    assert "[1] income-tax-act.pdf, page 42" in prompt
    assert "[3] advance-tax.pdf, page 3" in prompt
    assert "15 June, 15 September" in prompt


def test_no_documents_skips_the_llm():
    llm = RecordingLLM(responses=["should not be used"])
    assistant = TaxAssistant(retriever=StaticRetriever(docs=[]), llm=llm)

    answer = assistant.ask("What is the tax on agricultural income?")

    assert answer.text == NO_CONTEXT_ANSWER
    assert answer.sources == []
    assert llm.prompts == []


def test_empty_question_is_rejected(tax_docs):
    assistant = TaxAssistant(
        retriever=StaticRetriever(docs=tax_docs), llm=FakeListLLM(responses=["x"])
    )

    with pytest.raises(ValueError):
        assistant.ask("   ")


def test_stream_answer_yields_text(tax_docs):
    assistant = TaxAssistant(
        retriever=StaticRetriever(docs=tax_docs), llm=FakeListLLM(responses=["Streamed answer"])
    )

    assert "".join(assistant.stream_answer("q", tax_docs)) == "Streamed answer"


def test_sources_handle_missing_metadata_and_long_text():
    docs = [Document(page_content="word " * 200, metadata={})]

    (source,) = to_sources(docs)

    assert source == Source(title="Unknown source", page=None, excerpt=source.excerpt)
    assert source.label == "Unknown source"
    assert source.excerpt.endswith("…")
    assert len(source.excerpt) <= 301


def test_sources_strip_windows_paths():
    docs = [Document(page_content="x", metadata={"source": "C:\\tax\\rules.pdf", "page": "0"})]

    assert to_sources(docs)[0].label == "rules.pdf, page 1"


def test_format_context_without_page():
    context = format_context([Document(page_content=" Some text ", metadata={"source": "a.pdf"})])

    assert context == "[1] a.pdf\nSome text"


def test_prompt_variables():
    assert set(build_prompt().input_variables) == {"context", "question"}


def test_build_assistant_requires_pinecone_key_before_loading_models():
    with pytest.raises(ConfigError, match="PINECONE_API_KEY"):
        build_assistant(Settings())


def test_build_assistant_reports_missing_model_file(tmp_path):
    settings = Settings(pinecone_api_key="k", llm_model_path=tmp_path / "missing.gguf")

    with pytest.raises(ConfigError, match="download_model"):
        build_assistant(settings)
