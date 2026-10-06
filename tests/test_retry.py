"""The shared retry loop, and what the HTTP rerankers do with it."""
import httpx
import pytest

from src import retry
from src.embedding.pacing import RateWindow
from src.retrieving.rerankers import RerankError
from src.retrieving.rerankers.jina import JinaReranker
from src.retrieving.rerankers.voyage import VoyageReranker
from src.retry import RetryableError, backoff_seconds, retry_async
from tests.builders import retrieved_chunk as chunk


@pytest.fixture
def slept(monkeypatch):
    waits = []

    async def record(seconds):
        waits.append(seconds)

    monkeypatch.setattr(retry.asyncio, "sleep", record)
    return waits


async def test_a_transient_failure_is_retried_with_growing_backoff_then_succeeds(slept):
    attempts = []

    async def send():
        attempts.append(1)
        if len(attempts) < 3:
            raise RetryableError(RuntimeError("busy"))
        return "ok"

    assert await retry_async(send, attempts=4) == "ok"
    assert slept == [1.0, 2.0]


async def test_the_error_of_the_last_attempt_is_raised_when_attempts_run_out(slept):
    async def send():
        raise RetryableError(ValueError("still busy"))

    with pytest.raises(ValueError, match="still busy"):
        await retry_async(send, attempts=2)
    assert len(slept) == 1


async def test_an_error_that_is_not_retryable_propagates_at_once(slept):
    calls = []

    async def send():
        calls.append(1)
        raise PermissionError("bad key")

    with pytest.raises(PermissionError):
        await retry_async(send, attempts=5)
    assert len(calls) == 1 and slept == []


async def test_the_providers_own_delay_overrides_the_backoff_and_is_capped(slept):
    async def send():
        raise RetryableError(RuntimeError("limit"), delay=45.0 if not slept else 10_000.0)

    with pytest.raises(RuntimeError):
        await retry_async(send, attempts=3)
    assert slept == [45.0, retry.MAX_BACKOFF_SECONDS]


def test_backoff_doubles_and_jitter_only_adds():
    assert [backoff_seconds(n) for n in range(4)] == [1.0, 2.0, 4.0, 8.0]
    assert 4.0 <= backoff_seconds(2, jitter=True) < 5.0


# ── The HTTP rerankers ───────────────────────────────────────────────────────

def serve(monkeypatch, module, responses):
    requests = []

    def handler(request):
        requests.append(request)
        return responses.pop(0)

    real = httpx.AsyncClient
    monkeypatch.setattr(f"src.retrieving.rerankers.{module}.httpx.AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    return requests


async def test_jina_does_not_retry_a_client_error(monkeypatch, slept):
    requests = serve(monkeypatch, "jina", [httpx.Response(401, json={"detail": "bad key"})])
    with pytest.raises(RerankError, match="401"):
        await JinaReranker("key").rerank("q", [chunk(text='text')], top_k=1)
    assert len(requests) == 1 and slept == []


async def test_jina_retries_a_server_error(monkeypatch, slept):
    ok = httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.9}], "usage": {"total_tokens": 10}})
    requests = serve(monkeypatch, "jina", [httpx.Response(503, json={}), ok])
    result = await JinaReranker("key").rerank("q", [chunk(text='text')], top_k=1)
    assert len(requests) == 2 and result.chunks[0].similarity_score == 0.9


async def test_voyage_refuses_a_pool_that_cannot_fit_its_token_limit_before_sending_anything(monkeypatch, slept):
    requests = serve(monkeypatch, "voyage", [])
    reranker = VoyageReranker("key", "https://api.voyageai.com/v1", "rerank-3", RateWindow(3, 1_000))
    with pytest.raises(RerankError, match="lower rerank_candidates"):
        await reranker.rerank("q", [chunk(text="word " * 400)] * 3, top_k=1)
    assert requests == []
