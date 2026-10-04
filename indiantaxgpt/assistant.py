"""The question-answering pipeline: retrieve relevant passages, then generate a grounded answer."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain_core.output_parsers import StrOutputParser

from indiantaxgpt.prompts import build_prompt

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.language_models import BaseLanguageModel
    from langchain_core.prompts import BasePromptTemplate
    from langchain_core.retrievers import BaseRetriever

    from indiantaxgpt.config import Settings

NO_CONTEXT_ANSWER = (
    "I couldn't find anything relevant to that in the indexed documents. "
    "Try rephrasing the question, or add documents that cover the topic and re-run ingestion."
)

_EXCERPT_CHARS = 300


@dataclass(frozen=True)
class Source:
    """A document passage an answer was based on."""

    title: str
    page: int | None
    excerpt: str

    @property
    def label(self) -> str:
        return f"{self.title}, page {self.page}" if self.page is not None else self.title


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)


def _source_title(metadata: dict[str, Any]) -> str:
    source = str(metadata.get("source") or "").replace("\\", "/")
    return source.rsplit("/", 1)[-1] or "Unknown source"


def _page_number(metadata: dict[str, Any]) -> int | None:
    # PDF loaders store a zero-based page index; people expect one-based page numbers.
    page = metadata.get("page")
    try:
        return int(page) + 1
    except (TypeError, ValueError):
        return None


def to_sources(docs: Sequence[Document]) -> list[Source]:
    """Convert retrieved documents into de-duplicated citations, keeping retrieval order."""
    sources: list[Source] = []
    seen: set[tuple[str, int | None]] = set()
    for doc in docs:
        title, page = _source_title(doc.metadata), _page_number(doc.metadata)
        if (title, page) in seen:
            continue
        seen.add((title, page))
        excerpt = " ".join(doc.page_content.split())
        if len(excerpt) > _EXCERPT_CHARS:
            excerpt = excerpt[:_EXCERPT_CHARS].rsplit(" ", 1)[0] + "…"
        sources.append(Source(title=title, page=page, excerpt=excerpt))
    return sources


def format_context(docs: Sequence[Document]) -> str:
    """Render retrieved passages as numbered blocks for the prompt."""
    blocks = []
    for i, doc in enumerate(docs, start=1):
        title, page = _source_title(doc.metadata), _page_number(doc.metadata)
        header = f"[{i}] {title}" + (f", page {page}" if page is not None else "")
        blocks.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)


class TaxAssistant:
    """Answers questions using passages from a retriever and a language model."""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLanguageModel,
        prompt: BasePromptTemplate | None = None,
    ) -> None:
        self._retriever = retriever
        self._chain = (prompt or build_prompt()) | llm | StrOutputParser()

    def retrieve(self, question: str) -> list[Document]:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")
        return self._retriever.invoke(question)

    def stream_answer(self, question: str, docs: Sequence[Document]) -> Iterator[str]:
        """Yield the answer for ``question`` token by token, grounded in ``docs``."""
        yield from self._chain.stream(
            {"context": format_context(docs), "question": question.strip()}
        )

    def ask(self, question: str) -> Answer:
        docs = self.retrieve(question)
        if not docs:
            return Answer(text=NO_CONTEXT_ANSWER)
        text = "".join(self.stream_answer(question, docs)).strip()
        return Answer(text=text, sources=to_sources(docs))


def build_assistant(settings: Settings) -> TaxAssistant:
    """Wire up embeddings, the Pinecone index and the local LLM from ``settings``."""
    from indiantaxgpt.embeddings import load_embeddings
    from indiantaxgpt.llm import load_llm
    from indiantaxgpt.vectorstore import load_vectorstore

    # Check the cheap prerequisites before spending time loading models.
    settings.require_pinecone_key()
    llm = load_llm(settings)
    embeddings = load_embeddings(settings.embedding_model)
    vectorstore = load_vectorstore(settings, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.retriever_k})
    return TaxAssistant(retriever=retriever, llm=llm)
