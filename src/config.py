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

    jina_api_key: str = ""  # legacy index queries and the reranker only; page fetching is keyless
    gemini_api_key: str = ""
    groq_api_key: str = ""
    openai_api_key: str = ""

    llm_provider: str = "gemini"
    llm_model_name: str = ""

    # One index = one embedding model (src/embedding/providers.py). Ingestion writes to, and
    # queries read from, the index of this provider + model.
    embedding_provider: str = "voyage"
    embedding_model: str = ""  # empty = the provider's default model
    voyage_api_key: str = ""
    voyage_base_url: str = "https://api.voyageai.com/v1"
    # Voyage's limits for an account with no payment method on file (MongoDB Voyage docs,
    # verified 2026-09-26). Adding a payment method raises them; free tokens still apply.
    voyage_rpm: int = 3
    voyage_tpm: int = 10_000
    ollama_base_url: str = "http://localhost:11434"

    # Page fetching (src/fetching): only hosted reader APIs ever contact a third-party site.
    firecrawl_api_key: str = ""  # optional fallback reader and sitemap mapper
    fetch_allowed_domains: str = ""  # non-empty = allowlist mode (e.g. a public demo)
    fetch_denied_domains: str = (
        "facebook.com,instagram.com,x.com,twitter.com,tiktok.com,linkedin.com,"
        "whitepages.com,spokeo.com,beenverified.com,truepeoplesearch.com"
    )
    fetch_daily_page_quota: int = 200
    fetch_min_interval_seconds: float = 3.0

    # Docling parsing of uploads (src/parsing), in the worker only. The 2026-09-26 spike
    # measured up to ~8.5 s/page on 2 CPU threads: 50 pages fit the timeout even on a CPU
    # twice as slow. Longer PDFs are still ingested, as PyMuPDF text without headings.
    docling_max_pages: int = 50
    docling_timeout_seconds: int = 900

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
        return _csv(self.allowed_origins)

    @property
    def allowed_fetch_domains(self) -> List[str]:
        return [d.lower() for d in _csv(self.fetch_allowed_domains)]

    @property
    def denied_fetch_domains(self) -> List[str]:
        return [d.lower() for d in _csv(self.fetch_denied_domains)]

    def missing_required(self) -> List[str]:
        """Names of settings that must be set (or valid) before the API can serve traffic."""
        problems = []
        if not self.effective_admin_key:
            problems.append("ADMIN_API_KEY")
        if not self.database_url:
            problems.append("DATABASE_URL")

        embedding_key = {"voyage": "voyage_api_key", "jina": "jina_api_key"}.get(self.embedding_provider.lower())
        if self.embedding_provider.lower() not in ("voyage", "jina", "ollama"):
            problems.append("EMBEDDING_PROVIDER (voyage | ollama | jina)")
        elif embedding_key and not getattr(self, embedding_key):
            problems.append(f"{embedding_key.upper()} (EMBEDDING_PROVIDER={self.embedding_provider})")

        provider_key = {"gemini": "gemini_api_key", "groq": "groq_api_key", "openai": "openai_api_key"}.get(
            self.llm_provider.lower()
        )
        if provider_key and not getattr(self, provider_key):
            problems.append(provider_key.upper())
        return problems

    def warn_on_legacy_secrets(self) -> None:
        if self.rag_api_key and not self.admin_api_key:
            logger.warning("RAG_API_KEY is deprecated; set ADMIN_API_KEY instead.")


def _csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
