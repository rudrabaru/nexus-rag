"""The Voyage reranker: wire format, cost, its own pacing window, retries, and failures that surface."""
import httpx
import pytest

from src.config_checks import config_problems
from src.config import Settings
from src.embedding.pacing import RateWindow
from src.retrieving.config import RetrievalConfig
from src.retrieving.rerankers import build_reranker
from src.retrieving.rerankers.voyage import VoyageReranker
from tests.builders import retrieved_chunk as chunk


@pytest.fixture
def served(monkeypatch):
    """Routes the reranker's HTTP calls to a queue of canned responses and records each request body."""
    state = {"responses": [], "requests": []}

    def handler(request: httpx.Request) -> httpx.Response:
        state["requests"].append(request)
        return state["responses"].pop(0)

    real = httpx.AsyncClient
    monkeypatch.setattr("src.retrieving.rerankers.voyage.httpx.AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))

    async def no_wait(seconds):
        state.setdefault("slept", []).append(seconds)

    monkeypatch.setattr("src.retrieving.rerankers.voyage.asyncio.sleep", no_wait)
    return state


def reranker(rpm=3, tpm=10_000, model="rerank-3") -> VoyageReranker:
    return VoyageReranker("key", "https://api.voyageai.com/v1/", model, RateWindow(rpm, tpm))


OK = {"data": [{"index": 2, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.4}], "usage": {"total_tokens": 1_000_000}}


async def test_candidates_are_reordered_rescored_and_priced_by_the_tokens_voyage_counted(served):
    served["responses"].append(httpx.Response(200, json=OK))
    result = await reranker().rerank("how often", [chunk("a"), chunk("b"), chunk("c")], top_k=2)

    assert [(c.chunk_id, c.similarity_score) for c in result.chunks] == [("c", 0.9), ("a", 0.4)]
    assert result.rerank_cost_usd == pytest.approx(0.05)  # 1M tokens at rerank-3's list price
    request = served["requests"][0]
    assert str(request.url) == "https://api.voyageai.com/v1/rerank" and request.headers["authorization"] == "Bearer key"
    assert request.read() == httpx.Request("POST", "http://x", json={
        "model": "rerank-3", "query": "how often", "documents": ["text of a", "text of b", "text of c"], "top_k": 2,
    }).read()


async def test_a_rate_limited_request_waits_one_request_slot_and_is_retried(served):
    served["responses"] += [httpx.Response(429, json={"detail": "3 RPM"}), httpx.Response(200, json=OK)]
    result = await reranker(rpm=3).rerank("q", [chunk("a"), chunk("b"), chunk("c")], top_k=2)
    assert len(result.chunks) == 2 and len(served["requests"]) == 2
    assert served["slept"] == [20.0]  # 60 s / 3 requests per minute


async def test_persistent_rate_limiting_raises_instead_of_returning_the_first_stage_order(served):
    served["responses"] += [httpx.Response(429, json={"detail": "3 RPM"})] * 3
    with pytest.raises(Exception, match="voyage: HTTP 429"):
        await reranker().rerank("q", [chunk("a")], top_k=1)


async def test_a_client_error_is_final_and_not_retried(served):
    served["responses"].append(httpx.Response(401, json={"detail": "bad key"}))
    with pytest.raises(Exception, match="401"):
        await reranker().rerank("q", [chunk("a")], top_k=1)
    assert len(served["requests"]) == 1


async def test_its_pacing_window_makes_a_second_request_wait_for_a_free_slot(served, monkeypatch):
    window = RateWindow(requests_per_minute=1, tokens_per_minute=10_000)
    waited = []

    async def sleep_until_the_slot_frees(seconds):
        waited.append(seconds)
        window._sent.clear()  # a minute passing

    monkeypatch.setattr("src.retrieving.rerankers.voyage.asyncio.sleep", sleep_until_the_slot_frees)
    r = VoyageReranker("key", "https://api.voyageai.com/v1", "rerank-3", window)
    served["responses"] += [httpx.Response(200, json=OK)] * 2
    await r.rerank("q", [chunk("a"), chunk("b"), chunk("c")], top_k=2)
    assert waited == []  # the first request goes at once
    await r.rerank("q", [chunk("a"), chunk("b"), chunk("c")], top_k=2)
    assert len(waited) == 1 and 0 < waited[0] <= 60  # the second waits for the first one's slot


async def test_without_a_key_it_fails_fast_and_an_empty_pool_calls_nothing(served):
    with pytest.raises(Exception, match="VOYAGE_API_KEY"):
        await VoyageReranker("", "https://x", "rerank-3", RateWindow(3, 10_000)).rerank("q", [chunk("a")], top_k=1)
    assert (await reranker().rerank("q", [], top_k=3)).chunks == [] and served["requests"] == []


def test_voyage_is_a_selectable_reranker_with_a_shared_window_per_process():
    assert RetrievalConfig(reranker="voyage", rerank_candidates=8).reranker == "voyage"
    settings = Settings(_env_file=None, voyage_api_key="k")
    first, second = build_reranker("voyage", settings), build_reranker("voyage", settings)
    assert first.window is second.window and first.model == "rerank-3"


def test_choosing_voyage_requires_its_key(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    assert any("VOYAGE_API_KEY (RERANKER=voyage)" in p for p in config_problems(Settings(_env_file=None, reranker="voyage"), "api"))
    assert not any("RERANKER" in p for p in config_problems(Settings(_env_file=None, reranker="voyage", voyage_api_key="k"), "api"))
