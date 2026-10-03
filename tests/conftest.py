import asyncio

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from src.config import Settings, get_settings  # noqa: E402
from src.registry.auth_store import AuthStore  # noqa: E402
from src.registry.schema import api_keys  # noqa: E402

ADMIN_KEY = "test-admin-key-0123456789-abcdefghijklmnop"

STATE_KEYS = (
    "ready", "auth_store", "registry", "retrieval", "generator",
    "evaluator", "rewriter", "metrics_store", "pipeline_logger", "query_semaphore",
    "job_queue",
)


@pytest.fixture(autouse=True)
def settings_env(monkeypatch):
    """A complete, fake environment so Settings never depends on a developer's .env."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setenv("ADMIN_API_KEY", ADMIN_KEY)
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@db.invalid/test?sslmode=require")
    monkeypatch.setenv("VOYAGE_API_KEY", "voyage-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def auth_engine():
    """
    In-memory SQLite holding only api_keys. AuthStore issues portable SQL, so these tests run
    the real queries. StaticPool shares one connection across the threads asyncio.to_thread uses.
    """
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    api_keys.create(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def app(auth_engine):
    """A fresh FastAPI app per test, built by the real factory."""
    from src.api.app import create_app

    return create_app()


@pytest.fixture
def app_state(app, auth_engine):
    """
    The app with lifespan skipped (no network) and rate limiting disabled, so each test injects
    exactly the collaborators it needs on app.state.
    """
    from src.api.rate_limit import auth_failures, limiter

    limiter.enabled = False
    auth_failures.reset()

    app.state.ready = True
    app.state.auth_store = AuthStore(auth_engine)
    app.state.query_semaphore = asyncio.Semaphore(2)
    yield app.state

    for key in STATE_KEYS:
        if hasattr(app.state, key):
            delattr(app.state, key)
    limiter.enabled = True
    auth_failures.reset()


@pytest.fixture
def client(app, app_state):
    return TestClient(app)


@pytest.fixture
def tenant_key(app_state):
    """Returns a factory issuing a valid key for a given tenant id."""
    return lambda tenant_id: app_state.auth_store.create_api_key(tenant_id)
