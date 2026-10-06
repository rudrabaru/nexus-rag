"""
Whether the configuration is complete and consistent for the process that is starting.

A process refuses to start with the full list of what is wrong, instead of failing later at the
first request that needs the missing piece. What is required depends on the role: the API serves
chat and needs an admin key and an LLM key; a worker only needs the database and the embedding
provider; a command-line tool needs the database and says itself what else a command needs.
"""
import os
from typing import List, Literal

from sqlalchemy.engine import make_url

from src.config import LLM_PROVIDER_KEY_FIELDS, Settings
from src.llm.config import parse_model

Role = Literal["api", "worker", "cli"]

MIN_ADMIN_KEY_LENGTH = 32  # a person-chosen secret guarded only by a throttle needs room against guessing
EMBEDDING_PROVIDERS = ("voyage", "ollama", "cloudflare")
RERANKERS = ("flashrank", "jina", "voyage", "none")
STRATEGIES = ("dense", "sparse", "hybrid")
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")
TLS_MODES = ("require", "verify-ca", "verify-full")

# Variables that once configured Nexus. Ignoring them silently would change behaviour without a
# word (RERANKER would fall back to its default), so a process that still sets one refuses to start.
REMOVED_VARIABLES = {
    "RAG_API_KEY": "use ADMIN_API_KEY",
    "ENABLE_RERANKER": "use RERANKER=flashrank | jina | voyage | none",
    "TRUST_PROXIES": "use TRUSTED_PROXY_HOPS (the number of reverse proxies in front of the API)",
    "MISTRAL_API_KEY": "Mistral is no longer a supported provider",
    "LLM_PROVIDER": "use LLM_CHAT=provider/model (for example gemini/gemini-3.5-flash)",
    "LLM_MODEL_NAME": "use LLM_CHAT=provider/model (for example gemini/gemini-3.5-flash)",
}


def _database_problem(url: str) -> str | None:
    if not url:
        return "DATABASE_URL"
    try:
        parsed = make_url(url)
    except Exception:
        return "DATABASE_URL (not a valid postgresql:// URL)"
    if parsed.get_backend_name() not in ("postgresql", "postgres"):
        return "DATABASE_URL (must be a postgresql:// URL)"
    if (parsed.host or "") not in LOCAL_HOSTS and parsed.query.get("sslmode") not in TLS_MODES:
        return "DATABASE_URL (a remote database needs ?sslmode=require so the connection is encrypted)"
    return None


def _embedding_problems(settings: Settings) -> List[str]:
    provider = settings.embedding_provider.lower()
    if provider not in EMBEDDING_PROVIDERS:
        return [f"EMBEDDING_PROVIDER ({' | '.join(EMBEDDING_PROVIDERS)})"]
    if provider == "voyage" and not settings.voyage_api_key.get_secret_value():
        return ["VOYAGE_API_KEY (EMBEDDING_PROVIDER=voyage)"]
    if provider == "cloudflare":
        missing = [
            name for name, value in (
                ("CLOUDFLARE_ACCOUNT_ID", settings.cloudflare_account_id),
                ("CLOUDFLARE_API_TOKEN", settings.cloudflare_api_token.get_secret_value()),
            ) if not value
        ]
        return [f"{name} (EMBEDDING_PROVIDER=cloudflare)" for name in missing]
    return []


def _llm_problems(settings: Settings) -> List[str]:
    """Every role's model must be 'provider/model'; chat, which every request needs, must also have its provider's key."""
    problems = []
    for variable in ("llm_chat", "llm_chat_fallback", "llm_rewrite", "llm_judge", "llm_testset"):
        value = getattr(settings, variable)
        if not value and variable == "llm_chat_fallback":
            continue
        try:
            provider, _ = parse_model(value)
        except ValueError:
            problems.append(f"{variable.upper()} (provider/model, for example gemini/gemini-3.5-flash)")
            continue
        if variable == "llm_chat" and provider not in LLM_PROVIDER_KEY_FIELDS:
            problems.append(f"LLM_CHAT (provider must be one of {', '.join(LLM_PROVIDER_KEY_FIELDS)})")
        elif variable == "llm_chat" and not settings.has_llm_key(provider):
            problems.append(LLM_PROVIDER_KEY_FIELDS[provider].upper())
    return problems


def _retrieval_problems(settings: Settings) -> List[str]:
    problems = []
    if settings.retrieval_strategy.lower() not in STRATEGIES:
        problems.append(f"RETRIEVAL_STRATEGY ({' | '.join(STRATEGIES)})")
    reranker = settings.reranker.lower()
    if reranker not in RERANKERS:
        problems.append(f"RERANKER ({' | '.join(RERANKERS)})")
    elif reranker == "jina" and not settings.jina_api_key.get_secret_value():
        problems.append("JINA_API_KEY (RERANKER=jina)")
    elif reranker == "voyage" and not settings.voyage_api_key.get_secret_value():
        problems.append("VOYAGE_API_KEY (RERANKER=voyage)")
    return problems


def config_problems(settings: Settings, role: Role) -> List[str]:
    """Names of settings that are missing or invalid for this process role. Empty = good to start."""
    problems = [
        f"{name} is no longer supported: {advice}" for name, advice in REMOVED_VARIABLES.items() if os.environ.get(name)
    ]
    database = _database_problem(settings.database_url.get_secret_value())
    if database:
        problems.append(database)

    if role in ("api", "worker"):
        problems += _embedding_problems(settings)
    if role == "api":
        admin_key = settings.admin_api_key.get_secret_value()
        if not admin_key:
            problems.append("ADMIN_API_KEY")
        elif len(admin_key) < MIN_ADMIN_KEY_LENGTH:
            problems.append(
                f"ADMIN_API_KEY (at least {MIN_ADMIN_KEY_LENGTH} characters; "
                "generate one with: python -c \"import secrets; print(secrets.token_urlsafe(32))\")"
            )
        problems += _retrieval_problems(settings)
        problems += _llm_problems(settings)
    if role == "cli":
        problems += _retrieval_problems(settings)
    return problems
