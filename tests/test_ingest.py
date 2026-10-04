import dataclasses

import pytest
from langchain_core.documents import Document

from indiantaxgpt import ingest
from indiantaxgpt.config import ConfigError
from tests.conftest import make_pdf


def write_pdf(path, pages):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(make_pdf(pages))


def test_load_pdfs_reads_nested_files_with_relative_sources(tmp_path):
    data = tmp_path / "data"
    write_pdf(data / "acts" / "act.pdf", ["Section 80C deductions", "Section 80D health"])
    write_pdf(data / "guide.PDF", ["Filing your return"])
    (data / "notes.txt").write_text("ignored")

    pages = ingest.load_pdfs(data)

    assert [(p.metadata["source"], p.metadata["page"]) for p in pages] == [
        ("acts/act.pdf", 0),
        ("acts/act.pdf", 1),
        ("guide.PDF", 0),
    ]
    assert "Section 80C deductions" in pages[0].page_content


def test_missing_or_empty_data_dir_raises(tmp_path):
    with pytest.raises(ConfigError, match="does not exist"):
        ingest.find_pdfs(tmp_path / "nope")

    (tmp_path / "empty").mkdir()
    with pytest.raises(ConfigError, match="No PDF files"):
        ingest.find_pdfs(tmp_path / "empty")


def test_split_documents_keeps_metadata_and_drops_blank_chunks():
    pages = [
        Document(page_content="income tax " * 100, metadata={"source": "a.pdf", "page": 0}),
        Document(page_content="   \n  ", metadata={"source": "a.pdf", "page": 1}),
    ]

    chunks = ingest.split_documents(pages, chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(len(c.page_content) <= 200 for c in chunks)
    assert all(c.metadata["source"] == "a.pdf" and c.metadata["page"] == 0 for c in chunks)
    assert all("start_index" in c.metadata for c in chunks)


def test_chunk_ids_are_stable_and_unique():
    pages = [Document(page_content="section " * 300, metadata={"source": "a.pdf", "page": 0})]

    first = [ingest.chunk_id(c) for c in ingest.split_documents(pages, 200, 20)]
    second = [ingest.chunk_id(c) for c in ingest.split_documents(pages, 200, 20)]

    assert first == second
    assert len(set(first)) == len(first)


def test_dry_run_does_not_touch_pinecone(settings, monkeypatch):
    write_pdf(settings.data_dir / "act.pdf", ["Section 10 exemptions"])
    settings = dataclasses.replace(settings, pinecone_api_key=None)

    assert ingest.ingest(settings, dry_run=True) == 1


def test_cli_dry_run(tmp_path, capsys):
    write_pdf(tmp_path / "pdfs" / "act.pdf", ["Section 87A rebate", "Section 139 returns"])

    exit_code = ingest.main(["--data-dir", str(tmp_path / "pdfs"), "--dry-run"])

    assert exit_code == 0
    assert "would index 2 chunks" in capsys.readouterr().out


def test_cli_reports_config_errors(tmp_path, capsys):
    exit_code = ingest.main(["--data-dir", str(tmp_path / "missing")])

    assert exit_code == 1
    assert "does not exist" in capsys.readouterr().err
