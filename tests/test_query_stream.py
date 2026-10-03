import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.generating.models import ContextWindow
from src.retrieving.models import RetrievalResult

CAPACITY = 2


class FakeGenerator:
    def prepare(self, query, retrieval_result, chat_history=None):
        return SimpleNamespace(context_window=ContextWindow(), prompt="p", diagnostic=None)

    async def stream(self, prepared, call):
        for piece in ("Hello", " world"):
            call.text += piece
            yield piece
        call.prompt_tokens, call.completion_tokens = 10, 5


class FakeRetriever:
    def __init__(self, error=None):
        self.error = error

    async def retrieve(self, query, top_k, tenant_id, pipeline_logger=None, **kwargs):
        if self.error:
            raise self.error
        return RetrievalResult(query=query, top_k=top_k, latency_ms=1.0, chunks=[])


class FakeResources:
    """Stands in for RetrievalResources: the same fake answers as dense and as sparse retriever."""

    def __init__(self, error=None):
        self.retriever = FakeRetriever(error)

    def retrievers(self, index_id=None):
        return self.retriever, self.retriever


def documents_with(count):
    documents = MagicMock()
    documents.document_count.return_value = count
    return documents


@pytest.fixture
def wired(app_state):
    app_state.generator = FakeGenerator()
    app_state.retrieval = FakeResources()
    app_state.evaluator = MagicMock()
    app_state.documents = documents_with(3)
    return app_state


def stream(client, tenant_key, tenant="tenant-1"):
    response = client.post(
        "/query/stream",
        json={"query": "what is this?"},
        headers={"X-API-Key": tenant_key(tenant)},
    )
    events = [json.loads(line[6:]) for line in response.text.split("\n\n") if line.startswith("data: ")]
    return response, events


def test_stream_delivers_tokens_sources_and_done(client, wired, tenant_key):
    response, events = stream(client, tenant_key)

    assert response.status_code == 200
    assert "".join(e["content"] for e in events if e["type"] == "token") == "Hello world"
    assert [e["type"] for e in events][-2:] == ["sources", "done"]


def test_capacity_is_returned_after_a_completed_stream(client, wired, tenant_key):
    stream(client, tenant_key)
    assert wired.query_semaphore._value == CAPACITY


def test_stream_works_when_no_document_store_is_configured(client, wired, tenant_key):
    """Regression: token_generator was only defined inside `if registry:`, raising NameError."""
    wired.documents = None
    response, events = stream(client, tenant_key)

    assert response.status_code == 200
    assert events[-1]["type"] == "done"


def test_capacity_is_returned_when_no_document_store_is_configured(client, wired, tenant_key):
    """Regression: the semaphore was leaked permanently on this path."""
    wired.documents = None
    for _ in range(CAPACITY + 2):
        stream(client, tenant_key)
    assert wired.query_semaphore._value == CAPACITY


def test_capacity_is_returned_for_an_empty_workspace(client, wired, tenant_key):
    wired.documents = documents_with(0)
    response, events = stream(client, tenant_key)

    assert "no documents" in events[0]["content"]
    assert wired.query_semaphore._value == CAPACITY


def test_capacity_is_returned_when_retrieval_fails(client, wired, tenant_key):
    wired.retrieval = FakeResources(error=RuntimeError("vector store down"))
    response, _ = stream(client, tenant_key)

    assert response.status_code == 500
    assert wired.query_semaphore._value == CAPACITY


def test_an_unauthenticated_stream_is_a_401_not_a_stream_that_says_so(client, wired):
    response = client.post("/query/stream", json={"query": "hi"})
    assert response.status_code == 401
    assert "/register" not in response.text
