"""Calling the model: the success path"""
import asyncio
import pytest
from src.llm.client import LLMClient
from tests.support.llm_client import config, response


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    monkeypatch.setattr("src.llm.client.time.sleep", lambda *_: None)
    # A fresh coroutine per call — asyncio.sleep(0) can only be awaited once.
    real_sleep = asyncio.sleep  # captured first: the patch replaces asyncio.sleep itself, so the lambda must not call it by name
    monkeypatch.setattr("src.llm.client.asyncio.sleep", lambda *_: real_sleep(0))


@pytest.fixture(autouse=True)
def deterministic_cost(monkeypatch):
    monkeypatch.setattr("src.llm.client.litellm.completion_cost", lambda **kw: 0.0042)


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
