"""Pacing, provider wire formats and behaviour."""
import httpx
import pytest
from src.config import get_settings
from src.embedding.pacing import WINDOW_SECONDS, RateWindow
from src.embedding.embedder import EmbeddingError
from src.embedding.providers import build_embedder
from src.db.schema import EMBEDDING_DIMENSION
from tests.support.embedding import voyage


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


def test_the_default_index_is_voyage_4():
    assert build_embedder(get_settings()).index_id == "voyage:voyage-4"


def test_an_index_id_selects_its_own_provider_and_model():
    embedder = build_embedder(get_settings(), "ollama:bge-m3")
    assert (embedder.provider, embedder.model) == ("ollama", "bge-m3")
    with pytest.raises(ValueError):
        build_embedder(get_settings(), "qdrant:whatever")
    with pytest.raises(ValueError):  # the Jina embedding provider was removed
        build_embedder(get_settings(), "jina:jina-embeddings-v3")


async def test_cloudflare_sends_the_account_model_and_token_to_its_openai_compatible_endpoint(server, monkeypatch):
    seen = []
    real_handler = server.handler
    server.handler = lambda request: seen.append(request) or real_handler(request)
    for key, value in {"EMBEDDING_PROVIDER": "cloudflare", "CLOUDFLARE_ACCOUNT_ID": "acc123", "CLOUDFLARE_API_TOKEN": "cf-token"}.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    embedder = build_embedder(get_settings())
    assert embedder.index_id == "cloudflare:bge-m3"
    batch = await embedder.aembed(["a", "b"], "document")
    assert len(batch.vectors) == 2 and batch.tokens == 7
    request = seen[0]
    assert str(request.url) == "https://api.cloudflare.com/client/v4/accounts/acc123/ai/v1/embeddings"
    assert request.headers["authorization"] == "Bearer cf-token"
    assert server.bodies[0] == {"model": "@cf/baai/bge-m3", "input": ["a", "b"]}


async def test_queries_and_documents_are_embedded_asymmetrically(server, monkeypatch):
    embedder = voyage(monkeypatch)
    await embedder.aembed(["q"], "query")
    await embedder.aembed(["d"], "document")
    assert [b["input_type"] for b in server.bodies] == ["query", "document"]
    assert all(b["output_dimension"] == EMBEDDING_DIMENSION for b in server.bodies)


def test_texts_are_split_so_no_request_exceeds_the_token_window(monkeypatch):
    """A single request larger than the TPM window could never be admitted by the provider."""
    embedder = voyage(monkeypatch, VOYAGE_TPM="1000")
    groups = embedder.group_indices(["x" * 1500] * 4)  # ~500 estimated tokens each
    assert groups == [[0, 1], [2, 3]]


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
