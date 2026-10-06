"""Helpers and fixtures shared by the embedding tests."""
import json

import httpx
import pytest

from src.embedding import embedder as embedder_module
from src.config import get_settings
from src.embedding import providers
from src.embedding.providers import build_embedder
from src.db.schema import EMBEDDING_DIMENSION


VECTOR = [0.0] * EMBEDDING_DIMENSION


class FakeServer:
    """Answers embedding requests through httpx.MockTransport and records their bodies."""

    def __init__(self, responses=None):
        self.bodies = []
        self.responses = list(responses or [])

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        if self.responses:
            return self.responses.pop(0)
        texts = body.get("input", [])
        return httpx.Response(200, json={"data": [{"embedding": VECTOR, "index": i} for i in range(len(texts))], "usage": {"total_tokens": 7}})


async def no_sleep(seconds):
    return None


def voyage(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    providers._windows.clear()
    return build_embedder(get_settings())


@pytest.fixture
def server(monkeypatch):
    fake = FakeServer()
    real_client = httpx.AsyncClient
    monkeypatch.setattr(embedder_module.httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(fake.handler), **kw))
    monkeypatch.setattr(embedder_module.asyncio, "sleep", no_sleep)
    return fake
