import logging
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Single validated view of the environment.

    .env is loaded into os.environ once, in src/api/main.py; this class only reads
    the process environment. Modules scheduled for replacement still read os.environ
    directly for their provider keys, so those keys are declared here for
    fail-fast validation rather than as their only consumer.
    """

    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    admin_api_key: str = ""
    rag_api_key: str = ""  # legacy fallback for admin_api_key

    # Postgres (Neon) is the system of record: chunks, vectors, sparse index, documents,
    # jobs, keys and metrics. Use the direct (non-pooler) endpoint; see src/registry/engine.py.
    database_url: str = ""
    db_pool_size: int = 5

    # Read only by `python -m scripts.migrate_legacy`, which copies the legacy collection.
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "nexus_rag_collection"

    jina_api_key: str = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""
    openai_api_key: str = ""

    llm_provider: str = "gemini"
    llm_model_name: str = ""
    vision_model_name: str = ""

    enable_reranker: bool = True
    enable_query_generalisation: bool = False
    query_concurrency: int = 4
    # How many ingestion jobs one worker process runs at once (Procrastinate Worker
    # concurrency). Read only by the worker (src/jobs/worker.py); the API never runs jobs.
    worker_concurrency: int = 2

    allowed_origins: str = ""
    trust_proxies: bool = False

    @property
    def effective_admin_key(self) -> str:
        return self.admin_api_key or self.rag_api_key

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    def missing_required(self) -> List[str]:
        """Names of settings that must be set (or valid) before the API can serve traffic."""
        problems = []
        if not self.effective_admin_key:
            problems.append("ADMIN_API_KEY")
        if not self.database_url:
            problems.append("DATABASE_URL")
        if not self.jina_api_key:
            problems.append("JINA_API_KEY")

        provider_key = {"gemini": "gemini_api_key", "groq": "groq_api_key", "openai": "openai_api_key"}.get(
            self.llm_provider.lower()
        )
        if provider_key and not getattr(self, provider_key):
            problems.append(provider_key.upper())
        return problems

    def warn_on_legacy_secrets(self) -> None:
        if self.rag_api_key and not self.admin_api_key:
            logger.warning("RAG_API_KEY is deprecated; set ADMIN_API_KEY instead.")


@lru_cache
def get_settings() -> Settings:
    return Settings()
