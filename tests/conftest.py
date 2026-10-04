from __future__ import annotations

import os
from unittest import mock

import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from indiantaxgpt.config import Settings

ENV_VARS = [
    "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME",
    "PINECONE_CLOUD",
    "PINECONE_REGION",
    "EMBEDDING_MODEL",
    "LLM_MODEL_PATH",
    "LLM_MODEL_TYPE",
    "LLM_MAX_NEW_TOKENS",
    "LLM_TEMPERATURE",
    "LLM_CONTEXT_LENGTH",
    "DATA_DIR",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "RETRIEVER_K",
]


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    """Run every test in an empty directory with no IndianTaxGPT variables set.

    ``os.environ`` is restored afterwards, including anything ``load_dotenv`` wrote.
    """
    with mock.patch.dict(os.environ):
        for name in ENV_VARS:
            os.environ.pop(name, None)
        monkeypatch.chdir(tmp_path)
        yield


class StaticRetriever(BaseRetriever):
    """Returns the same documents for every query and records what was asked."""

    docs: list[Document]
    queries: list[str] = []

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        self.queries.append(query)
        return list(self.docs)


@pytest.fixture
def tax_docs() -> list[Document]:
    return [
        Document(
            page_content="Section 80C allows a deduction of up to Rs 1,50,000 for specified "
            "investments such as PPF, ELSS and life insurance premiums.",
            metadata={"source": "acts/income-tax-act.pdf", "page": 41},
        ),
        Document(
            page_content="The deduction under section 80C is available to individuals and HUFs.",
            metadata={"source": "acts/income-tax-act.pdf", "page": 41},
        ),
        Document(
            page_content="Advance tax is payable in instalments by 15 June, 15 September, "
            "15 December and 15 March.",
            metadata={"source": "guides/advance-tax.pdf", "page": 2},
        ),
    ]


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(pinecone_api_key="test-key", data_dir=tmp_path / "data")


def make_pdf(pages: list[str]) -> bytes:
    """Build a minimal valid PDF with one line of Helvetica text per page."""
    font_id = 3 + 2 * len(pages)
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids ["
        + b" ".join(f"{3 + 2 * i} 0 R".encode() for i in range(len(pages)))
        + f"] /Count {len(pages)} >>".encode(),
    ]
    for i, text in enumerate(pages):
        content_id = 4 + 2 * i
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>".encode()
        )
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
    out += f"startxref\n{xref_at}\n%%EOF\n".encode()
    return bytes(out)
