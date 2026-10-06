"""Streaming: failing before the first token, never after."""
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
from src.llm.client import LLMCall, LLMClient
from src.llm.errors import GenerationError
from tests.support.llm_client import config, not_found_error, rate_limit_error, response, timeout_error


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    monkeypatch.setattr("src.llm.client.time.sleep", lambda *_: None)
    # A fresh coroutine per call — asyncio.sleep(0) can only be awaited once.
    real_sleep = asyncio.sleep  # captured first: the patch replaces asyncio.sleep itself, so the lambda must not call it by name
    monkeypatch.setattr("src.llm.client.asyncio.sleep", lambda *_: real_sleep(0))


@pytest.fixture(autouse=True)
def deterministic_cost(monkeypatch):
    monkeypatch.setattr("src.llm.client.litellm.completion_cost", lambda **kw: 0.0042)


async def fake_stream(chunks):
    for delta, usage in chunks:
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=delta))] if delta is not None else [],
            usage=usage,
        )


def usage(pt=7, ct=3):
    return SimpleNamespace(prompt_tokens=pt, completion_tokens=ct)


@pytest.mark.asyncio
async def test_stream_timeout_before_first_token_falls_back_without_retries(monkeypatch):
    calls = []

    async def acompletion(**kw):
        calls.append((kw["model"], kw["timeout"]))
        if kw["model"] == "gemini/primary-model":
            raise timeout_error()
        return fake_stream([("fallback answer", usage())])

    monkeypatch.setattr("src.llm.client.litellm.acompletion", acompletion)
    client = LLMClient(config(fallback={"provider": "groq", "model_name": "fallback-model"}))

    text = "".join([c async for c in client.call_llm_stream("hi", LLMCall(), max_retries=3)])

    assert text == "fallback answer"
    assert calls == [("gemini/primary-model", 60.0), ("groq/fallback-model", 60.0)]


@pytest.mark.asyncio
async def test_stream_falls_back_when_the_primary_fails_before_any_token(monkeypatch):
    async def primary(**kw):
        raise rate_limit_error()

    async def fallback(**kw):
        return fake_stream([("fallback ", None), ("answer", usage())])

    calls = {"primary": 0, "fallback": 0}

    async def acompletion(**kw):
        if kw["model"] == "gemini/primary-model":
            calls["primary"] += 1
            return await primary(**kw)
        calls["fallback"] += 1
        return await fallback(**kw)

    monkeypatch.setattr("src.llm.client.litellm.acompletion", acompletion)
    client = LLMClient(config(fallback={"provider": "groq", "model_name": "fallback-model"}))

    call = LLMCall()
    text = "".join([c async for c in client.call_llm_stream("hi", call, max_retries=0)])

    assert text == "fallback answer" == call.text
    assert calls["fallback"] == 1
    assert (call.prompt_tokens, call.completion_tokens) == (7, 3)
    assert (call.provider, call.model) == ("groq", "fallback-model")


@pytest.mark.asyncio
async def test_stream_never_falls_back_after_the_first_token_is_yielded(monkeypatch):
    """A client that has already received tokens cannot be silently handed a second,
    unrelated answer from a different model."""
    async def acompletion(**kw):
        async def gen():
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="partial "))], usage=None)
            raise rate_limit_error()

        return gen()

    fallback_called = MagicMock()
    monkeypatch.setattr("src.llm.client.litellm.acompletion", acompletion)
    client = LLMClient(config(fallback={"provider": "groq", "model_name": "fallback-model"}))
    client._fallback_client.call_llm_stream = fallback_called

    chunks = []
    with pytest.raises(GenerationError):  # the failure is an error, not more answer text
        async for piece in client.call_llm_stream("hi", LLMCall(), max_retries=0):
            chunks.append(piece)

    assert chunks == ["partial "]
    fallback_called.assert_not_called()


@pytest.mark.asyncio
async def test_successful_stream_populates_usage_and_cost_from_the_final_chunk(monkeypatch):
    async def acompletion(**kw):
        return fake_stream([("hel", None), ("lo", usage(pt=12, ct=6))])

    monkeypatch.setattr("src.llm.client.litellm.acompletion", acompletion)
    client = LLMClient(config())

    call = LLMCall()
    text = "".join([c async for c in client.call_llm_stream("hi", call)])

    assert text == "hello" == call.text
    assert (call.prompt_tokens, call.completion_tokens) == (12, 6)
    assert call.cost_usd == 0.0042
    assert call.provider == "gemini"


def test_concurrent_calls_on_one_client_keep_their_own_usage(monkeypatch):
    """
    Regression: usage, cost and the serving provider were stored on the shared client, so with
    concurrent queries one request logged another's tokens and cost.
    """
    import threading

    barrier = threading.Barrier(2)

    def completion(**kw):
        prompt = kw["messages"][0]["content"]
        barrier.wait(timeout=5)  # both calls are in flight before either returns
        return response(content=prompt, prompt_tokens=len(prompt), completion_tokens=len(prompt))

    monkeypatch.setattr("src.llm.client.litellm.completion", completion)
    client = LLMClient(config())
    results = {}

    def run(prompt):
        results[prompt] = client.call_llm(prompt)

    threads = [threading.Thread(target=run, args=(p,)) for p in ("a", "bbbbbb")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert (results["a"].prompt_tokens, results["bbbbbb"].prompt_tokens) == (1, 6)


@pytest.mark.asyncio
async def test_stream_falls_back_when_the_primary_model_is_dead_not_only_when_it_is_busy(monkeypatch):
    async def acompletion(**kw):
        if kw["model"] == "gemini/primary-model":
            raise not_found_error(model="gemini/primary-model")
        return fake_stream([("fallback answer", usage())])

    monkeypatch.setattr("src.llm.client.litellm.acompletion", acompletion)
    client = LLMClient(config(fallback={"provider": "groq", "model_name": "fallback-model"}))

    text = "".join([c async for c in client.call_llm_stream("hi", LLMCall())])

    assert text == "fallback answer"


@pytest.mark.asyncio
async def test_a_stream_that_ends_without_text_is_retried_and_then_fails_not_returned_as_an_empty_answer(monkeypatch):
    calls = []

    async def acompletion(**kw):
        calls.append(1)
        return fake_stream([])

    monkeypatch.setattr("src.llm.client.litellm.acompletion", acompletion)

    with pytest.raises(GenerationError, match="EmptyResponseError"):
        async for _ in LLMClient(config()).call_llm_stream("hi", LLMCall(), max_retries=2):
            pass

    assert len(calls) == 3


@pytest.mark.asyncio
async def test_a_stream_with_no_fallback_raises_instead_of_yielding_failure_text(monkeypatch):
    async def acompletion(**kw):
        raise not_found_error(model="gemini/primary-model")

    monkeypatch.setattr("src.llm.client.litellm.acompletion", acompletion)
    chunks = []
    with pytest.raises(GenerationError, match="NotFoundError"):
        async for piece in LLMClient(config()).call_llm_stream("hi", LLMCall()):
            chunks.append(piece)
    assert chunks == []
