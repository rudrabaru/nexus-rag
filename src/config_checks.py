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

Role = Literal["api", "worker", "cli"]

MIN_ADMIN_KEY_LENGTH = 32  # a person-chosen secret guarded only by a throttle needs room against guessing
EMBEDDING_PROVIDERS = ("voyage", "ollama")
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
    return []


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
        provider = settings.llm_provider.lower()
        if not settings.has_llm_key(provider):
            problems.append(LLM_PROVIDER_KEY_FIELDS[provider].upper())
    if role == "cli":
        problems += _retrieval_problems(settings)
    return problems
