"""
Tests for the retry/fallback logic in src/llm/client.py.

litellm.completion / litellm.acompletion are mocked directly rather than mocking HTTP,
since they are the module's actual dependency surface. time.sleep and asyncio.sleep are
patched to no-ops so the exponential-backoff paths run instantly.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import litellm
import pytest

from src.llm.client import LLMCall, LLMClient
from src.llm.errors import GenerationError
from src.generating.models import GenerationConfig


def config(provider="gemini", model_name="primary-model", fallback=None):
    return GenerationConfig(provider=provider, model_name=model_name, fallback_config=fallback)


def response(content="hello", finish_reason="stop", prompt_tokens=10, completion_tokens=5):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def rate_limit_error(model="gemini/primary-model", provider="gemini"):
    return litellm.RateLimitError(message="429 rate limited", llm_provider=provider, model=model)


def not_found_error(model="groq/dead-model", provider="groq"):
    return litellm.NotFoundError(message="model does not exist", model=model, llm_provider=provider)


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    monkeypatch.setattr("src.llm.client.time.sleep", lambda *_: None)
    # A fresh coroutine per call — asyncio.sleep(0) can only be awaited once.
    real_sleep = asyncio.sleep  # captured first: the patch replaces asyncio.sleep itself, so the lambda must not call it by name
    monkeypatch.setattr("src.llm.client.asyncio.sleep", lambda *_: real_sleep(0))


@pytest.fixture(autouse=True)
def deterministic_cost(monkeypatch):
    monkeypatch.setattr("src.llm.client.litellm.completion_cost", lambda **kw: 0.0042)


# ── call_llm: the success path ────────────────────────────────────────────────

def test_successful_call_returns_content_tokens_and_cost(monkeypatch):
    monkeypatch.setattr("src.llm.client.litellm.completion", lambda **kw: response())
    client = LLMClient(config())

    call = client.call_llm("hi")

    assert call.text == "hello" and call.cost_known
    assert (call.prompt_tokens, call.completion_tokens) == (10, 5)
    assert call.cost_usd == 0.0042
    assert (call.provider, call.model) == ("gemini", "primary-model")


def test_model_string_is_provider_slash_model_name(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "src.llm.client.litellm.completion",
        lambda **kw: seen.update(kw) or response(),
    )
    LLMClient(config(provider="groq", model_name="openai/gpt-oss-20b")).call_llm("hi")
    assert seen["model"] == "groq/openai/gpt-oss-20b"


def test_response_schema_is_forwarded_as_response_format(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "src.llm.client.litellm.completion",
        lambda **kw: seen.update(kw) or response(),
    )

    class Schema:
        pass

    LLMClient(config()).call_llm("hi", response_schema=Schema)
    assert seen["response_format"] is Schema


# ── call_llm: retry classification ──────────────────────────────────────────

def test_a_429_on_the_primary_reaches_the_fallback():
    """The specific regression test item 6 requires: a 429 must not just retry forever
    against a dead quota — it must eventually reach the configured fallback."""
    calls = []

    def fake_completion(**kw):
        calls.append(kw["model"])
        if kw["model"] == "gemini/primary-model":
            raise rate_limit_error()
        return response(content="from fallback")

    import src.llm.client as mod
    mod.litellm.completion = fake_completion
    try:
        client = LLMClient(config(fallback={"provider": "groq", "model_name": "fallback-model"}))
        call = client.call_llm("hi", max_retries=2)
    finally:
        mod.litellm.completion = litellm.completion

    assert call.text == "from fallback"
    assert (call.provider, call.model) == ("groq", "fallback-model")  # the model that answered is what gets logged
    # 1 primary attempt + 2 retries, all against the primary, then exactly one fallback call.
    assert calls == ["gemini/primary-model"] * 3 + ["groq/fallback-model"]


def test_no_fallback_configured_returns_a_readable_error_after_retries(monkeypatch):
    monkeypatch.setattr("src.llm.client.litellm.completion", MagicMock(side_effect=rate_limit_error()))
    client = LLMClient(config())

    with pytest.raises(GenerationError, match="RateLimitError"):  # never returned as if it were an answer
        client.call_llm("hi", max_retries=2)


def test_a_dead_model_is_not_retried_locally_before_falling_back():
    """404s cannot be fixed by retrying the same request, so a dead model should waste
    zero retries against itself before moving to the fallback (the exact bug this
    replaced: llama-3.1-8b-instant's 404 previously reached no fallback at all, and a
    model that will never succeed should not eat a retry budget either)."""
    call_count = {"primary": 0, "fallback": 0}

    def fake_completion(**kw):
        if kw["model"] == "groq/dead-model":
            call_count["primary"] += 1
            raise not_found_error()
        call_count["fallback"] += 1
        return response(content="from fallback")

    import src.llm.client as mod
    mod.litellm.completion = fake_completion
    try:
        client = LLMClient(config(provider="groq", model_name="dead-model", fallback={"provider": "gemini", "model_name": "backup"}))
        answer = client.call_llm("hi", max_retries=3).text
    finally:
        mod.litellm.completion = litellm.completion

    assert answer == "from fallback"
    assert call_count == {"primary": 1, "fallback": 1}


def test_a_dead_model_with_no_fallback_fails_after_exactly_one_attempt(monkeypatch):
    mock = MagicMock(side_effect=not_found_error())
    monkeypatch.setattr("src.llm.client.litellm.completion", mock)
    client = LLMClient(config())

    with pytest.raises(GenerationError, match="NotFoundError"):
        client.call_llm("hi", max_retries=3)

    assert mock.call_count == 1


def test_empty_content_on_a_200_is_treated_as_a_retryable_error(monkeypatch):
    """Observed live: a reasoning model can exhaust its token budget on internal
    reasoning and return finish_reason='length' with content=None. That must not be
    passed downstream as if it were a real (empty) answer."""
    calls = {"n": 0}

    def fake_completion(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return response(content=None, finish_reason="length")
        return response(content="recovered")

    monkeypatch.setattr("src.llm.client.litellm.completion", fake_completion)
    client = LLMClient(config())

    answer = client.call_llm("hi", max_retries=2).text

    assert answer == "recovered"
    assert calls["n"] == 2


def test_an_unclassified_exception_fails_immediately_without_retry(monkeypatch):
    mock = MagicMock(side_effect=ValueError("something unrelated broke"))
    monkeypatch.setattr("src.llm.client.litellm.completion", mock)
    client = LLMClient(config())

    with pytest.raises(GenerationError, match="ValueError"):
        client.call_llm("hi", max_retries=3)

    assert mock.call_count == 1


def test_cost_lookup_failure_does_not_break_generation(monkeypatch):
    monkeypatch.setattr("src.llm.client.litellm.completion", lambda **kw: response())
    monkeypatch.setattr(
        "src.llm.client.litellm.completion_cost",
        MagicMock(side_effect=Exception("model not in litellm's pricing map")),
    )
    client = LLMClient(config())

    call = client.call_llm("hi")

    assert call.text == "hello"
    assert call.cost_usd == 0.0 and call.cost_known is False  # unknown is not the same as free


def timeout_error(model="gemini/primary-model"):
    return litellm.Timeout(message="request timed out", model=model, llm_provider="gemini")


def test_every_call_carries_the_configured_timeout(monkeypatch):
    """Regression: no timeout was passed, so litellm's 6000 s default applied and a hung
    provider held a query slot for up to 100 minutes (observed live on Gemini)."""
    seen = {}
    monkeypatch.setattr("src.llm.client.litellm.completion", lambda **kw: seen.update(kw) or response())
    LLMClient(GenerationConfig(request_timeout_seconds=12.0)).call_llm("hi")
    assert seen["timeout"] == 12.0


def test_a_timeout_goes_straight_to_the_fallback_without_local_retries():
    calls = []

    def fake_completion(**kw):
        calls.append(kw["model"])
        if kw["model"] == "gemini/primary-model":
            raise timeout_error()
        return response(content="from fallback")

    import src.llm.client as mod
    mod.litellm.completion = fake_completion
    try:
        client = LLMClient(config(fallback={"provider": "groq", "model_name": "fallback-model"}))
        answer = client.call_llm("hi", max_retries=3).text
    finally:
        mod.litellm.completion = litellm.completion

    assert answer == "from fallback"
    assert calls == ["gemini/primary-model", "groq/fallback-model"]


# ── call_llm_stream: fail-before-first-token only ───────────────────────────


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

async def fake_stream(chunks):
    for delta, usage in chunks:
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=delta))] if delta is not None else [],
            usage=usage,
        )


def usage(pt=7, ct=3):
    return SimpleNamespace(prompt_tokens=pt, completion_tokens=ct)


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
