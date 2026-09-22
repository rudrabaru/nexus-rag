"""
Tests for the retry/fallback logic in src/generating/llm_client.py.

litellm.completion / litellm.acompletion are mocked directly rather than mocking HTTP,
since they are the module's actual dependency surface. time.sleep and asyncio.sleep are
patched to no-ops so the exponential-backoff paths run instantly.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import litellm
import pytest

from src.generating.llm_client import LLMClient
from src.generating.models import GenerationConfig


def config(provider="gemini", model_name="primary-model", fallback=None, max_retries_field=None):
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
    monkeypatch.setattr("src.generating.llm_client.time.sleep", lambda *_: None)
    # A fresh coroutine per call — asyncio.sleep(0) can only be awaited once.
    monkeypatch.setattr("src.generating.llm_client.asyncio.sleep", lambda *_: asyncio.sleep(0))


@pytest.fixture(autouse=True)
def deterministic_cost(monkeypatch):
    monkeypatch.setattr("src.generating.llm_client.litellm.completion_cost", lambda **kw: 0.0042)


# ── call_llm: the success path ────────────────────────────────────────────────

def test_successful_call_returns_content_tokens_and_cost(monkeypatch):
    monkeypatch.setattr("src.generating.llm_client.litellm.completion", lambda **kw: response())
    client = LLMClient(config())

    answer, raw, prompt_tokens, completion_tokens = client.call_llm("hi")

    assert answer == "hello"
    assert (prompt_tokens, completion_tokens) == (10, 5)
    assert client.last_cost_usd == 0.0042
    assert client.last_served_provider == "gemini"
    assert client.last_served_model == "primary-model"


def test_model_string_is_provider_slash_model_name(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "src.generating.llm_client.litellm.completion",
        lambda **kw: seen.update(kw) or response(),
    )
    LLMClient(config(provider="groq", model_name="openai/gpt-oss-20b")).call_llm("hi")
    assert seen["model"] == "groq/openai/gpt-oss-20b"


def test_response_schema_is_forwarded_as_response_format(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "src.generating.llm_client.litellm.completion",
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

    import src.generating.llm_client as mod
    mod.litellm.completion = fake_completion
    try:
        client = LLMClient(config(fallback={"provider": "groq", "model_name": "fallback-model"}))
        answer, *_ = client.call_llm("hi", max_retries=2)
    finally:
        mod.litellm.completion = litellm.completion

    assert answer == "from fallback"
    # 1 primary attempt + 2 retries, all against the primary, then exactly one fallback call.
    assert calls == ["gemini/primary-model"] * 3 + ["groq/fallback-model"]


def test_no_fallback_configured_returns_a_readable_error_after_retries(monkeypatch):
    monkeypatch.setattr("src.generating.llm_client.litellm.completion", MagicMock(side_effect=rate_limit_error()))
    client = LLMClient(config())

    answer, raw, prompt_tokens, completion_tokens = client.call_llm("hi", max_retries=2)

    assert answer.startswith("[Generation failed: RateLimitError")
    assert (prompt_tokens, completion_tokens) == (0, 0)


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

    import src.generating.llm_client as mod
    mod.litellm.completion = fake_completion
    try:
        client = LLMClient(config(provider="groq", model_name="dead-model", fallback={"provider": "gemini", "model_name": "backup"}))
        answer, *_ = client.call_llm("hi", max_retries=3)
    finally:
        mod.litellm.completion = litellm.completion

    assert answer == "from fallback"
    assert call_count == {"primary": 1, "fallback": 1}


def test_a_dead_model_with_no_fallback_fails_after_exactly_one_attempt(monkeypatch):
    mock = MagicMock(side_effect=not_found_error())
    monkeypatch.setattr("src.generating.llm_client.litellm.completion", mock)
    client = LLMClient(config())

    answer, *_ = client.call_llm("hi", max_retries=3)

    assert "NotFoundError" in answer
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

    monkeypatch.setattr("src.generating.llm_client.litellm.completion", fake_completion)
    client = LLMClient(config())

    answer, *_ = client.call_llm("hi", max_retries=2)

    assert answer == "recovered"
    assert calls["n"] == 2


def test_an_unclassified_exception_fails_immediately_without_retry(monkeypatch):
    mock = MagicMock(side_effect=ValueError("something unrelated broke"))
    monkeypatch.setattr("src.generating.llm_client.litellm.completion", mock)
    client = LLMClient(config())

    answer, *_ = client.call_llm("hi", max_retries=3)

    assert "ValueError" in answer
    assert mock.call_count == 1


def test_cost_lookup_failure_does_not_break_generation(monkeypatch):
    monkeypatch.setattr("src.generating.llm_client.litellm.completion", lambda **kw: response())
    monkeypatch.setattr(
        "src.generating.llm_client.litellm.completion_cost",
        MagicMock(side_effect=Exception("model not in litellm's pricing map")),
    )
    client = LLMClient(config())

    answer, *_ = client.call_llm("hi")

    assert answer == "hello"
    assert client.last_cost_usd == 0.0


# ── call_llm_stream: fail-before-first-token only ───────────────────────────

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

    monkeypatch.setattr("src.generating.llm_client.litellm.acompletion", acompletion)
    client = LLMClient(config(fallback={"provider": "groq", "model_name": "fallback-model"}))

    text = "".join([c async for c in client.call_llm_stream("hi", max_retries=0)])

    assert text == "fallback answer"
    assert calls["fallback"] == 1
    assert client.last_prompt_tokens == 7 and client.last_completion_tokens == 3


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
    monkeypatch.setattr("src.generating.llm_client.litellm.acompletion", acompletion)
    client = LLMClient(config(fallback={"provider": "groq", "model_name": "fallback-model"}))
    client._fallback_client.call_llm_stream = fallback_called

    chunks = [c async for c in client.call_llm_stream("hi", max_retries=0)]

    assert chunks[0] == "partial "
    assert "[Generation failed:" in chunks[1]
    fallback_called.assert_not_called()


@pytest.mark.asyncio
async def test_successful_stream_populates_usage_and_cost_from_the_final_chunk(monkeypatch):
    async def acompletion(**kw):
        return fake_stream([("hel", None), ("lo", usage(pt=12, ct=6))])

    monkeypatch.setattr("src.generating.llm_client.litellm.acompletion", acompletion)
    client = LLMClient(config())

    text = "".join([c async for c in client.call_llm_stream("hi")])

    assert text == "hello"
    assert (client.last_prompt_tokens, client.last_completion_tokens) == (12, 6)
    assert client.last_cost_usd == 0.0042
    assert client.last_served_provider == "gemini"
