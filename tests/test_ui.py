from pathlib import Path

import pytest
import streamlit as st
from langchain_core.language_models import FakeListLLM
from streamlit.testing.v1 import AppTest

from indiantaxgpt import ui
from indiantaxgpt.assistant import TaxAssistant
from indiantaxgpt.config import ConfigError
from tests.conftest import StaticRetriever

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(autouse=True)
def clear_streamlit_cache():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def run_app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=30).run()


def test_shows_setup_help_when_not_configured():
    app = run_app()

    assert not app.exception
    assert "PINECONE_API_KEY" in app.error[0].value
    assert any("To get started" in md.value for md in app.markdown)


def test_shows_config_errors_from_environment(monkeypatch):
    monkeypatch.setenv("RETRIEVER_K", "zero")

    app = run_app()

    assert not app.exception
    assert "RETRIEVER_K" in app.error[0].value


def test_answers_a_question_with_sources(monkeypatch, tax_docs):
    assistant = TaxAssistant(
        retriever=StaticRetriever(docs=tax_docs),
        llm=FakeListLLM(responses=["Up to Rs 1.5 lakh under Section 80C."]),
    )
    monkeypatch.setattr(ui, "build_assistant", lambda settings: assistant)

    app = run_app()
    assert len(app.button) == len(ui.EXAMPLE_QUESTIONS) + 1  # examples + clear conversation

    app.chat_input[0].set_value("How much can I claim under 80C?").run()

    assert not app.exception
    assert len(app.chat_message) == 2
    rendered = " ".join(md.value for md in app.markdown)
    assert "Up to Rs 1.5 lakh under Section 80C." in rendered
    assert "**income-tax-act.pdf, page 42**" in rendered
    assert app.session_state.messages[-1]["content"] == "Up to Rs 1.5 lakh under Section 80C."


def test_example_question_button_asks_it(monkeypatch, tax_docs):
    retriever = StaticRetriever(docs=tax_docs, queries=[])
    assistant = TaxAssistant(retriever=retriever, llm=FakeListLLM(responses=["Answer"]))
    monkeypatch.setattr(ui, "build_assistant", lambda settings: assistant)

    app = run_app()
    app.button(key="example-3").click().run()

    assert not app.exception
    assert retriever.queries == [ui.EXAMPLE_QUESTIONS[3]]
    assert [m["role"] for m in app.session_state.messages] == ["user", "assistant"]


def test_runtime_failure_shows_error_instead_of_crashing(monkeypatch, tax_docs):
    class BrokenRetriever(StaticRetriever):
        def _get_relevant_documents(self, query, *, run_manager):
            raise ConnectionError("pinecone unreachable")

    assistant = TaxAssistant(retriever=BrokenRetriever(docs=[]), llm=FakeListLLM(responses=["x"]))
    monkeypatch.setattr(ui, "build_assistant", lambda settings: assistant)

    app = run_app()
    app.chat_input[0].set_value("anything").run()

    assert not app.exception
    assert "Something went wrong" in app.error[0].value


def test_missing_index_is_reported(monkeypatch):
    def fail(settings):
        raise ConfigError("Pinecone index 'indiantaxgpt' does not exist yet.")

    monkeypatch.setattr(ui, "build_assistant", fail)

    app = run_app()

    assert "does not exist yet" in app.error[0].value
