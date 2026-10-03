import logging
from functools import lru_cache
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# The Settings field holding each LLM provider's API key. litellm reads the same variables
# (GEMINI_API_KEY, ...) from the environment when it makes the call.
LLM_PROVIDER_KEY_FIELDS = {
    "gemini": "gemini_api_key", "groq": "groq_api_key", "openai": "openai_api_key", "mistral": "mistral_api_key",
}


class Settings(BaseSettings):
    """
    Single validated view of the environment.

    .env is loaded into os.environ once, in src/api/main.py (and the worker entry points);
    this class only reads the process environment.
    """

    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    admin_api_key: str = ""
    rag_api_key: str = ""  # legacy fallback for admin_api_key

    # Postgres (Neon) is the system of record: chunks, vectors, sparse index, documents,
    # jobs, keys and metrics. Use the direct (non-pooler) endpoint; see src/registry/engine.py.
    database_url: str = ""
    db_pool_size: int = 5

    jina_api_key: str = ""  # the jina reranker and the legacy jina index only; page fetching is keyless
    gemini_api_key: str = ""
    groq_api_key: str = ""
    openai_api_key: str = ""
    mistral_api_key: str = ""  # bulk generation and judging: test-set generation (src/testsets), experiment models

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
    voyage_rerank_model: str = "rerank-3"  # rerank-3 is current; rerank-2.5 legacy. Same 3 RPM / 10K TPM card-free limit, separate from embeddings
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

    # Chat's default retrieval (src/retrieving/config.py); an evaluation sets its own.
    retrieval_strategy: str = "hybrid"  # dense | sparse | hybrid
    # The reranker a query's use_reranker flag applies: flashrank | jina | voyage | none.
    reranker: str = "flashrank"
    enable_reranker: Optional[bool] = None  # legacy: false means RERANKER=none
    flashrank_model: str = "ms-marco-TinyBERT-L-2-v2"
    flashrank_cache_dir: str = ""  # empty = ~/.cache/flashrank
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
    def effective_reranker(self) -> Optional[str]:
        """The configured reranker's name, or None when reranking is off."""
        if self.enable_reranker is False or self.reranker.lower() == "none":
            return None
        return self.reranker.lower()

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
        return field is None or bool(getattr(self, field))

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

        if self.retrieval_strategy.lower() not in ("dense", "sparse", "hybrid"):
            problems.append("RETRIEVAL_STRATEGY (dense | sparse | hybrid)")
        if self.reranker.lower() not in ("flashrank", "jina", "voyage", "none"):
            problems.append("RERANKER (flashrank | jina | voyage | none)")
        elif self.effective_reranker == "jina" and not self.jina_api_key:
            problems.append("JINA_API_KEY (RERANKER=jina)")
        elif self.effective_reranker == "voyage" and not self.voyage_api_key:
            problems.append("VOYAGE_API_KEY (RERANKER=voyage)")

        if not self.has_llm_key(self.llm_provider):
            problems.append(LLM_PROVIDER_KEY_FIELDS[self.llm_provider.lower()].upper())
        return problems

    def warn_on_legacy_secrets(self) -> None:
        if self.rag_api_key and not self.admin_api_key:
            logger.warning("RAG_API_KEY is deprecated; set ADMIN_API_KEY instead.")
        if self.enable_reranker is not None:
            logger.warning("ENABLE_RERANKER is deprecated; set RERANKER=flashrank | jina | voyage | none instead.")


def _csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
