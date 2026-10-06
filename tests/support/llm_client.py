"""Helpers shared by the llm_client tests."""
from types import SimpleNamespace
import litellm
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


def timeout_error(model="gemini/primary-model"):
    return litellm.Timeout(message="request timed out", model=model, llm_provider="gemini")
