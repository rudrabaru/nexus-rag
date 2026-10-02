"""Embedding providers: pacing, request splitting, retries, and the one-model-per-index rule."""
import json
from types import SimpleNamespace

import httpx
import pytest

from src.config import get_settings
from src.embedding import providers
from src.embedding.generator import EmbeddingGenerator, embedding_input
from src.embedding.pacing import WINDOW_SECONDS, RateWindow
from src.embedding.providers import EmbeddingError, build_embedder
from src.retrieving.dense import DenseRetriever
from src.registry.schema import EMBEDDING_DIMENSION

VECTOR = [0.0] * EMBEDDING_DIMENSION


# ── Pacing ───────────────────────────────────────────────────────────────────

def test_rate_window_admits_up_to_the_request_limit_then_waits_for_the_oldest_to_expire():
    window = RateWindow(requests_per_minute=3, tokens_per_minute=10_000)
    assert [window.reserve(10, now=t) for t in (0.0, 1.0, 2.0)] == [0.0, 0.0, 0.0]
    assert window.reserve(10, now=3.0) == pytest.approx(WINDOW_SECONDS - 3.0)
    assert window.reserve(10, now=60.0) == 0.0  # the request at t=0 has left the window


def test_rate_window_enforces_the_token_limit():
    window = RateWindow(requests_per_minute=100, tokens_per_minute=10_000)
    assert window.reserve(8_000, now=0.0) == 0.0
    assert window.reserve(3_000, now=1.0) > 0  # would exceed 10K tokens within the minute


def test_a_request_larger_than_the_token_window_still_goes_out_alone():
    window = RateWindow(requests_per_minute=3, tokens_per_minute=10_000)
    assert window.reserve(50_000, now=0.0) == 0.0


# ── Provider wire formats and behaviour ─────────────────────────────────────

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


@pytest.fixture
def server(monkeypatch):
    fake = FakeServer()
    real_client = httpx.AsyncClient
    monkeypatch.setattr(providers.httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(fake.handler), **kw))
    monkeypatch.setattr(providers.asyncio, "sleep", _no_sleep)
    return fake


async def _no_sleep(seconds):
    return None


def voyage(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    providers._windows.clear()
    return build_embedder(get_settings())


def test_the_default_index_is_voyage_4():
    assert build_embedder(get_settings()).index_id == "voyage:voyage-4"


def test_an_index_id_selects_its_own_provider_and_model():
    embedder = build_embedder(get_settings(), "jina:jina-embeddings-v3")
    assert (embedder.provider, embedder.model) == ("jina", "jina-embeddings-v3")
    with pytest.raises(ValueError):
        build_embedder(get_settings(), "qdrant:whatever")


async def test_queries_and_documents_are_embedded_asymmetrically(server, monkeypatch):
    embedder = voyage(monkeypatch)
    await embedder.aembed(["q"], "query")
    await embedder.aembed(["d"], "document")
    assert [b["input_type"] for b in server.bodies] == ["query", "document"]
    assert all(b["output_dimension"] == EMBEDDING_DIMENSION for b in server.bodies)


def test_texts_are_split_so_no_request_exceeds_the_token_window(monkeypatch):
    """A single request larger than the TPM window could never be admitted by the provider."""
    embedder = voyage(monkeypatch, VOYAGE_TPM="1000")
    groups = embedder._request_groups(["x" * 1500] * 4)  # ~500 estimated tokens each
    assert [len(g) for g in groups] == [2, 2]


async def test_a_rate_limited_request_is_retried(server, monkeypatch):
    server.responses = [httpx.Response(429, headers={"retry-after": "1"}, json={"detail": "slow down"})]
    batch = await voyage(monkeypatch).aembed(["a"], "document")
    assert len(batch.vectors) == 1 and len(server.bodies) == 2


async def test_a_bad_key_fails_immediately_without_retrying(server, monkeypatch):
    server.responses = [httpx.Response(401, json={"detail": "bad key"})]
    with pytest.raises(EmbeddingError) as exc:
        await voyage(monkeypatch).aembed(["a"], "document")
    assert exc.value.retryable is False and len(server.bodies) == 1


async def test_vectors_of_the_wrong_width_are_refused(server, monkeypatch):
    server.responses = [httpx.Response(200, json={"data": [{"embedding": [0.1] * 512, "index": 0}], "usage": {}})]
    with pytest.raises(EmbeddingError, match="512"):
        await voyage(monkeypatch).aembed(["a"], "document")


# ── Generator and retriever wiring ──────────────────────────────────────────

def make_chunk(text="hello", heading_path=("Guide", "Setup")):
    from src.chunking.metadata import ChunkMetadata

    return ChunkMetadata(
        chunk_id="c0", source_url="u", source_document="Doc", title="T", heading_path=list(heading_path),
        chunk_text=text, token_count=1, 
        document_version="v", chunk_version="v", tenant_id="t", doc_id="d",
    )


def test_the_embedding_input_carries_the_document_and_heading_path():
    assert embedding_input(make_chunk()) == "[Doc > Guide > Setup]\nhello"


def test_embedded_chunks_record_their_index(server, monkeypatch):
    embedder = voyage(monkeypatch)
    embedded, failed = EmbeddingGenerator(embedder).generate_embeddings([make_chunk()])
    assert failed == [] and embedded[0].index_id == "voyage:voyage-4" and embedded[0].embedding_model == "voyage-4"
    assert server.bodies[0]["input_type"] == "document"


def test_a_failed_embedding_call_marks_the_batch_failed_with_its_reason(server, monkeypatch):
    server.responses = [httpx.Response(401, json={"detail": "bad key"})]
    generator = EmbeddingGenerator(voyage(monkeypatch))
    embedded, failed = generator.generate_embeddings([make_chunk(), make_chunk(), make_chunk(text="  ")])
    assert embedded == [] and failed == [0, 1]  # the empty chunk is skipped, not failed
    assert "401" in generator.last_error


def test_a_retriever_refuses_a_query_embedder_from_another_index(monkeypatch):
    store = SimpleNamespace(index_id="jina:jina-embeddings-v3")
    with pytest.raises(ValueError, match="not comparable"):
        DenseRetriever(store, voyage(monkeypatch))
