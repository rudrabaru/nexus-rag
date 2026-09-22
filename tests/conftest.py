import asyncio
from unittest.mock import MagicMock

import dotenv
import pytest
from fastapi.testclient import TestClient

# src.api.main calls load_dotenv() on import. Tests must never read a developer's real
# .env, so it is disabled before that module is first imported.
dotenv.load_dotenv = lambda *args, **kwargs: False

from src.config import Settings, get_settings  # noqa: E402

SIGNING_SECRET = "test-signing-secret-0123456789"
ADMIN_KEY = "test-admin-key-0123456789"

STATE_KEYS = (
    "ready", "auth_store", "registry", "retriever", "reranker", "generator",
    "evaluator", "rewriter", "metrics_store", "pipeline_logger", "query_semaphore",
    "ingestion_semaphore", "embedding_generator",
)


@pytest.fixture(autouse=True)
def settings_env(monkeypatch):
    """A complete, fake environment so Settings never depends on a developer's .env."""
    for name in (*Settings.model_fields, "demo_mode"):
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)
    monkeypatch.setenv("API_KEY_SIGNING_SECRET", SIGNING_SECRET)
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.invalid")
    monkeypatch.setenv("QDRANT_API_KEY", "qdrant-key")
    monkeypatch.setenv("JINA_API_KEY", "jina-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def app_state():
    """
    The real FastAPI app with lifespan skipped (no network) and rate limiting disabled,
    so each test injects exactly the collaborators it needs on app.state.
    """
    from src.api.main import app
    from src.api.routes import ingest, query
    from src.registry.auth_store import AuthStore

    query.limiter.enabled = False
    ingest.limiter.enabled = False

    app.state.ready = True
    app.state.auth_store = AuthStore(MagicMock(), SIGNING_SECRET)
    app.state.query_semaphore = asyncio.Semaphore(2)
    yield app.state

    for key in STATE_KEYS:
        if hasattr(app.state, key):
            delattr(app.state, key)
    query.limiter.enabled = True
    ingest.limiter.enabled = True


@pytest.fixture
def client(app_state):
    from src.api.main import app

    return TestClient(app)


@pytest.fixture
def tenant_key(app_state):
    """Returns a factory issuing a valid key for a given tenant id."""
    return lambda tenant_id: app_state.auth_store.create_api_key(tenant_id)
