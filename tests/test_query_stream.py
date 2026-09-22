import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.generating.models import ContextWindow
from src.retrieving.models import RetrievalResult

CAPACITY = 2


class FakeGenerator:
    def __init__(self):
        self.llm_client = SimpleNamespace(last_prompt_tokens=10, last_completion_tokens=5)
        self.context_builder = SimpleNamespace(build=lambda chunks: ContextWindow())

    async def generate(self, query, **kwargs):
        for piece in ("Hello", " world"):
            yield piece


class FakeRetriever:
    def __init__(self, error=None):
        self.error = error

    async def retrieve(self, query, top_k, tenant_id, pipeline_logger=None, **kwargs):
        if self.error:
            raise self.error
        return RetrievalResult(query=query, top_k=top_k, latency_ms=1.0, chunks=[])


def registry_with_docs(count):
    registry = MagicMock()
    registry.get_doc_count.return_value = count
    return registry


@pytest.fixture
def wired(app_state):
    app_state.generator = FakeGenerator()
    app_state.retriever = FakeRetriever()
    app_state.reranker = None
    app_state.evaluator = MagicMock()
    app_state.registry = registry_with_docs(3)
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


def test_stream_works_when_no_registry_is_configured(client, wired, tenant_key):
    """Regression: token_generator was only defined inside `if registry:`, raising NameError."""
    wired.registry = None
    response, events = stream(client, tenant_key)

    assert response.status_code == 200
    assert events[-1]["type"] == "done"


def test_capacity_is_returned_when_no_registry_is_configured(client, wired, tenant_key):
    """Regression: the semaphore was leaked permanently on this path."""
    wired.registry = None
    for _ in range(CAPACITY + 2):
        stream(client, tenant_key)
    assert wired.query_semaphore._value == CAPACITY


def test_capacity_is_returned_for_an_empty_workspace(client, wired, tenant_key):
    wired.registry = registry_with_docs(0)
    response, events = stream(client, tenant_key)

    assert "no documents" in events[0]["content"]
    assert wired.query_semaphore._value == CAPACITY


def test_capacity_is_returned_when_retrieval_fails(client, wired, tenant_key):
    wired.retriever = FakeRetriever(error=RuntimeError("vector store down"))
    response, _ = stream(client, tenant_key)

    assert response.status_code == 500
    assert wired.query_semaphore._value == CAPACITY


def test_unauthenticated_stream_is_told_to_ask_for_a_key(client, wired):
    response = client.post("/query/stream", json={"query": "hi"})
    assert "administrator" in response.text
    assert "/register" not in response.text
