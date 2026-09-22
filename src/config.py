import logging
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

MIN_SIGNING_SECRET_LENGTH = 16


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
    api_key_signing_secret: str = ""
    rag_api_key: str = ""  # legacy fallback for both secrets above

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
    ingestion_concurrency: int = 3
    query_concurrency: int = 4

    allowed_origins: str = ""
    trust_proxies: bool = False

    @property
    def effective_admin_key(self) -> str:
        return self.admin_api_key or self.rag_api_key

    @property
    def effective_signing_secret(self) -> str:
        return self.api_key_signing_secret or self.rag_api_key

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    def missing_required(self) -> List[str]:
        """Names of settings that must be set (or valid) before the API can serve traffic."""
        problems = []
        if not self.effective_admin_key:
            problems.append("ADMIN_API_KEY")
        if len(self.effective_signing_secret) < MIN_SIGNING_SECRET_LENGTH:
            problems.append(f"API_KEY_SIGNING_SECRET (min {MIN_SIGNING_SECRET_LENGTH} chars)")
        if not self.qdrant_url:
            problems.append("QDRANT_URL")
        if not self.qdrant_api_key:
            problems.append("QDRANT_API_KEY")
        if not self.jina_api_key:
            problems.append("JINA_API_KEY")

        provider_key = {"gemini": "gemini_api_key", "groq": "groq_api_key", "openai": "openai_api_key"}.get(
            self.llm_provider.lower()
        )
        if provider_key and not getattr(self, provider_key):
            problems.append(provider_key.upper())
        return problems

    def warn_on_legacy_secrets(self) -> None:
        if self.rag_api_key and not (self.admin_api_key and self.api_key_signing_secret):
            logger.warning(
                "RAG_API_KEY is being used as both the admin key and the signing secret. "
                "Set ADMIN_API_KEY and API_KEY_SIGNING_SECRET separately."
            )
        if self.effective_admin_key and self.effective_admin_key == self.effective_signing_secret:
            logger.warning("The admin key and the signing secret are identical; rotating one rotates both.")


@lru_cache
def get_settings() -> Settings:
    return Settings()
