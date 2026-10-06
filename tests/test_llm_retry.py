"""Which failures are retried, and how."""
import asyncio
from unittest.mock import MagicMock
import litellm
import pytest
from src.llm.client import LLMClient
from src.llm.errors import GenerationError
from src.generating.models import GenerationConfig
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
