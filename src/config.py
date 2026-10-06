import logging
from functools import lru_cache
from typing import List, Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# The Settings field holding each LLM provider's API key. litellm reads the same variables
# (GEMINI_API_KEY, ...) from the environment when it makes the call.
LLM_PROVIDER_KEY_FIELDS = {"gemini": "gemini_api_key", "groq": "groq_api_key", "openai": "openai_api_key"}


class Settings(BaseSettings):
    """
    Single validated view of the environment.

    .env is loaded into os.environ once by src/runtime.py when a process starts; this class only
    reads the process environment. Secrets are SecretStr, so printing or capturing the settings
    (a log line, an error report) shows "**********" instead of the value. Whether the values are
    complete and consistent for a given process is src/config_checks.py's job.
    """

    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    admin_api_key: SecretStr = SecretStr("")

    # Postgres (Neon) is the system of record: chunks, vectors, sparse index, documents,
    # jobs, keys and metrics. Use the direct (non-pooler) endpoint; see src/db/engine.py.
    database_url: SecretStr = SecretStr("")
    db_pool_size: int = Field(5, ge=1, le=20)

    jina_api_key: SecretStr = SecretStr("")  # the Jina reranker only; page fetching is keyless
    gemini_api_key: SecretStr = SecretStr("")
    groq_api_key: SecretStr = SecretStr("")
    openai_api_key: SecretStr = SecretStr("")

    # Which model plays which role, as 'provider/model' (src/llm/roles.py). The judge defaults to a
    # different family than chat, because a judge that shares the generator's family favours its answers.
    llm_chat: str = "gemini/gemini-3.5-flash"
    llm_chat_fallback: str = "groq/openai/gpt-oss-20b"  # used only when that provider's key is set; empty disables it
    llm_rewrite: str = "groq/openai/gpt-oss-20b"
    llm_judge: str = "groq/openai/gpt-oss-20b"
    llm_testset: str = "gemini/gemini-3.5-flash"

    # One index = one embedding model (src/embedding/providers.py). Ingestion writes to, and
    # queries read from, the index of this provider + model.
    embedding_provider: str = "voyage"
    embedding_model: str = ""  # empty = the provider's default model
    voyage_api_key: SecretStr = SecretStr("")
    voyage_base_url: str = "https://api.voyageai.com/v1"
    # Voyage's limits for an account with no payment method on file (measured 2026-10-02: the
    # binding one is 10K tokens a minute, about 17 chunks). Adding a payment method raises them.
    voyage_rpm: int = Field(3, ge=1)
    voyage_tpm: int = Field(10_000, ge=1)
    voyage_rerank_model: str = "rerank-3"  # same card-free limit as embeddings, in a separate bucket
    ollama_base_url: str = "http://localhost:11434"

    # Page fetching (src/crawling): only hosted reader APIs ever contact a third-party site.
    firecrawl_api_key: SecretStr = SecretStr("")  # optional fallback reader and sitemap mapper
    fetch_allowed_domains: str = ""  # non-empty = allowlist mode (e.g. a public demo)
    fetch_denied_domains: str = (
        "facebook.com,instagram.com,x.com,twitter.com,tiktok.com,linkedin.com,"
        "whitepages.com,spokeo.com,beenverified.com,truepeoplesearch.com"
    )
    fetch_daily_page_quota: int = Field(200, ge=0)
    fetch_min_interval_seconds: float = Field(3.0, ge=0)

    # Docling parsing of uploads (src/parsing), in the worker only. The 2026-09-26 spike
    # measured up to ~8.5 s/page on 2 CPU threads: 50 pages fit the timeout even on a CPU
    # twice as slow. Longer PDFs are still ingested, as PyMuPDF text without headings.
    docling_max_pages: int = Field(50, ge=1)
    docling_timeout_seconds: int = Field(900, ge=1)

    # Chat's default retrieval (src/retrieving/config.py); an evaluation sets its own.
    retrieval_strategy: str = "hybrid"  # dense | sparse | hybrid
    reranker: str = "flashrank"  # the reranker a query's use_reranker flag applies: flashrank | jina | voyage | none
    flashrank_model: str = "ms-marco-TinyBERT-L-2-v2"
    flashrank_cache_dir: str = ""  # empty = ~/.cache/flashrank
    enable_query_generalisation: bool = False
    query_concurrency: int = Field(4, ge=1, le=32)
    # How many ingestion jobs one worker process runs at once (Procrastinate Worker
    # concurrency). Read only by the worker; the API never runs jobs.
    worker_concurrency: int = Field(2, ge=1, le=8)

    allowed_origins: str = ""

    # Observability (src/observability). LOG_FORMAT: auto = JSON unless stderr is a terminal.
    # Error reporting is off unless a Sentry DSN is set; it sends exceptions only, no request data.
    log_format: Literal["auto", "json", "console"] = "auto"
    sentry_dsn: SecretStr = SecretStr("")
    sentry_environment: str = "local"

    # How many reverse proxies in front of the API append to X-Forwarded-For. 0 = the peer address is
    # the client. Render's load balancer is one hop. Entries further left are client-supplied.
    trusted_proxy_hops: int = Field(0, ge=0, le=5)

    @property
    def effective_reranker(self) -> str | None:
        """The configured reranker's name, or None when reranking is off."""
        name = self.reranker.lower()
        return None if name == "none" else name

    @property
    def cors_origins(self) -> List[str]:
        return _csv(self.allowed_origins)

    @property
    def allowed_fetch_domains(self) -> List[str]:
        return [d.lower() for d in _csv(self.fetch_allowed_domains)]

    @property
    def denied_fetch_domains(self) -> List[str]:
        return [d.lower() for d in _csv(self.fetch_denied_domains)]

    def has_llm_key(self, provider: str) -> bool:
        """False when the provider is known and its key is empty. Unknown providers are left to litellm."""
        field = LLM_PROVIDER_KEY_FIELDS.get(provider.lower())
        return field is None or bool(getattr(self, field).get_secret_value())


def _csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
