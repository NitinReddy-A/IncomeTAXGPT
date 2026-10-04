from types import SimpleNamespace

import pinecone
import pytest
from langchain_core.embeddings import FakeEmbeddings
from langchain_pinecone import PineconeVectorStore

from indiantaxgpt.config import ConfigError, Settings
from indiantaxgpt.vectorstore import ensure_index, load_vectorstore

RealPinecone = pinecone.Pinecone


class FakeControlPlane:
    """Stands in for ``pinecone.Pinecone``: tracks indexes and the calls made to it."""

    def __init__(self, indexes: dict[str, int]):
        self.indexes = dict(indexes)
        self.calls: list[tuple] = []

    def client(self, api_key):
        assert api_key == "test-key"
        plane = self

        class Client:
            def list_indexes(self):
                return SimpleNamespace(names=lambda: list(plane.indexes))

            def describe_index(self, name):
                return SimpleNamespace(dimension=plane.indexes[name])

            def create_index(self, name, dimension, metric, spec):
                plane.calls.append(("create", name, dimension, metric, spec.cloud, spec.region))
                plane.indexes[name] = dimension

            def delete_index(self, name):
                plane.calls.append(("delete", name))
                del plane.indexes[name]

            def Index(self, name):  # noqa: N802 - mirrors the Pinecone client API
                # A real data-plane handle; constructing it makes no network calls.
                return RealPinecone(api_key=api_key).Index(
                    name, host=f"https://{name}.example.pinecone.io"
                )

        return Client()


@pytest.fixture
def control_plane(monkeypatch):
    def install(indexes):
        plane = FakeControlPlane(indexes)
        monkeypatch.setattr(pinecone, "Pinecone", lambda api_key: plane.client(api_key))
        return plane

    return install


@pytest.fixture
def settings():
    return Settings(pinecone_api_key="test-key", pinecone_index_name="tax")


def test_creates_missing_index(control_plane, settings):
    plane = control_plane({})

    index = ensure_index(settings, 384)

    assert plane.calls == [("create", "tax", 384, "cosine", "aws", "us-east-1")]
    assert index.config.host == "https://tax.example.pinecone.io"


def test_reuses_existing_index_with_matching_dimension(control_plane, settings):
    plane = control_plane({"tax": 384})

    ensure_index(settings, 384)

    assert plane.calls == []


def test_dimension_mismatch_is_reported(control_plane, settings):
    control_plane({"tax": 768})

    with pytest.raises(ConfigError, match="dimension 768"):
        ensure_index(settings, 384)


def test_recreate_deletes_then_creates(control_plane, settings):
    plane = control_plane({"tax": 768})

    ensure_index(settings, 384, recreate=True)

    assert [call[0] for call in plane.calls] == ["delete", "create"]
    assert plane.indexes == {"tax": 384}


def test_load_vectorstore_requires_existing_index(control_plane, settings):
    control_plane({})

    with pytest.raises(ConfigError, match="indiantaxgpt.ingest"):
        load_vectorstore(settings, FakeEmbeddings(size=384))


def test_load_vectorstore_wraps_index(control_plane, settings):
    control_plane({"tax": 384})

    store = load_vectorstore(settings, FakeEmbeddings(size=384))

    assert isinstance(store, PineconeVectorStore)
    assert store.as_retriever(search_kwargs={"k": 3}).search_kwargs == {"k": 3}
