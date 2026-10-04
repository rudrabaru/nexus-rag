"""
Which model plays which role, read from settings, so no module hard-codes a model id.

    chat      answers users; falls back to LLM_CHAT_FALLBACK when that provider's key is set
    rewrite   short prompts (follow-up rewriting); falls back to the chat model
    judge     scores answers; pinned: a fallback judge would silently change the measuring instrument
    testset   writes the synthetic questions; pinned for the same reason

A role's value is 'provider/model'. The judge defaults to a different model family than the chat
model, because a judge that shares a family with the generator favours its own answers.
"""
from typing import Any, Dict

from src.config import Settings
from src.llm.config import parse_model

ROLES = ("chat", "rewrite", "judge", "testset")


def role_model(settings: Settings, role: str) -> str:
    return getattr(settings, f"llm_{role}")


def role_fields(settings: Settings, role: str) -> Dict[str, Any]:
    """LLMConfig fields for a role: its provider and model, and its fallback where the role has one."""
    provider, model_name = parse_model(role_model(settings, role))
    fields: Dict[str, Any] = {"provider": provider, "model_name": model_name}
    fallback = _fallback_spec(settings, role)
    if fallback:
        fallback_provider, fallback_model = parse_model(fallback)
        if (fallback_provider, fallback_model) != (provider, model_name) and settings.has_llm_key(fallback_provider):
            fields["fallback_config"] = {"provider": fallback_provider, "model_name": fallback_model}
    return fields


def _fallback_spec(settings: Settings, role: str) -> str:
    if role == "chat":
        return settings.llm_chat_fallback
    if role == "rewrite":
        return settings.llm_chat
    return ""
