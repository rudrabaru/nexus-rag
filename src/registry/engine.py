"""
Database engines. Two engines share one DATABASE_URL and one schema (src/registry/schema.py):

- sync (psycopg 3): registry, jobs, keys, metrics and chunk writes. These already run in
  worker threads (asyncio.to_thread / run_in_executor), so a blocking driver is correct there.
- async (asyncpg): query-time dense and sparse search, which runs on the event loop.

Neon specifics:
- Idle computes are suspended after ~5 minutes and their connections are dropped, so pooled
  connections are pre-pinged and recycled.
- Use the direct endpoint. The "-pooler" endpoint is PgBouncer in transaction mode, which
  breaks LISTEN/NOTIFY and session advisory locks (both needed by the job queue) and
  asyncpg's prepared-statement cache. If a pooler URL is given anyway, the statement caches
  are disabled so queries still work.
"""
import logging
from functools import lru_cache
from typing import Dict, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import URL, Engine, make_url
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from src.config import get_settings

logger = logging.getLogger(__name__)

POOL_RECYCLE_SECONDS = 240  # below Neon's ~5 minute idle suspend
CONNECT_TIMEOUT_SECONDS = 15  # covers a cold compute start

# Query-time HNSW settings, applied when a connection opens so no search pays an extra round
# trip to set them.
# - iterative_scan: a tenant-filtered HNSW scan keeps scanning until it has LIMIT matching rows,
#   instead of filtering a fixed candidate list and returning fewer (pgvector >= 0.8.0).
#   relaxed_order may return rows slightly out of distance order; the store re-sorts by score.
# - ef_search: must be at least the largest LIMIT. The largest is the rerank candidate pool,
#   top_k * 4 with QueryRequest.top_k <= 20, i.e. 80.
HNSW_SESSION_SETTINGS = {"hnsw.iterative_scan": "relaxed_order", "hnsw.ef_search": "100"}

_SSLMODES_REQUIRING_TLS = {"require", "verify-ca", "verify-full"}


def _base_url(database_url: str) -> URL:
    url = make_url(database_url)
    if url.get_backend_name() not in ("postgresql", "postgres"):
        raise ValueError("DATABASE_URL must be a postgresql:// URL.")
    return url.set(drivername="postgresql")


def is_pooler_url(database_url: str) -> bool:
    return "-pooler" in (make_url(database_url).host or "")


def libpq_url(database_url: str) -> str:
    """The URL as plain libpq connection info, for psycopg connections outside SQLAlchemy."""
    return _base_url(database_url).render_as_string(hide_password=False)


def sync_url(database_url: str) -> URL:
    """psycopg understands libpq parameters (sslmode, channel_binding) as given."""
    return _base_url(database_url).set(drivername="postgresql+psycopg")


def async_url(database_url: str) -> URL:
    """asyncpg rejects libpq-only parameters, so sslmode is translated and channel_binding dropped."""
    url = _base_url(database_url)
    query = dict(url.query)
    sslmode = query.pop("sslmode", None)
    query.pop("channel_binding", None)
    if sslmode in _SSLMODES_REQUIRING_TLS:
        query["ssl"] = "require"
    return url.set(drivername="postgresql+asyncpg", query=query)


def session_options(session_settings: Dict[str, str]) -> str:
    """Session settings in libpq `options` form: -cname=value, space-separated."""
    return " ".join(f"-c{name}={value}" for name, value in session_settings.items())


def async_connect_args(database_url: str, extra_settings: Optional[Dict[str, str]] = None) -> dict:
    """
    asyncpg connection arguments carrying the query-time session settings.

    Settings travel in the libpq `options` startup parameter, not as individual startup
    parameters: Neon's proxy silently drops individual ones (verified 2026-09-23 — even
    work_mem was ignored) but forwards `options`. Vanilla Postgres accepts both forms.
    """
    connect_args = {"timeout": CONNECT_TIMEOUT_SECONDS}
    session_settings = {**HNSW_SESSION_SETTINGS, **(extra_settings or {})}
    if is_pooler_url(database_url):
        # PgBouncer rejects unknown startup parameters, so the settings cannot be sent.
        logger.warning(
            "DATABASE_URL uses Neon's pooler endpoint; tenant-filtered vector search runs without "
            "iterative scans and may return fewer than top_k results. Use the direct endpoint."
        )
        connect_args["statement_cache_size"] = 0
    else:
        connect_args["server_settings"] = {"options": session_options(session_settings)}
    return connect_args


@lru_cache
def get_sync_engine() -> Engine:
    settings = get_settings()
    return create_engine(
        sync_url(settings.database_url),
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_pool_size,
        pool_pre_ping=True,
        pool_recycle=POOL_RECYCLE_SECONDS,
        connect_args={"connect_timeout": CONNECT_TIMEOUT_SECONDS},
    )


@lru_cache
def get_async_engine() -> AsyncEngine:
    settings = get_settings()
    connect_args = async_connect_args(settings.database_url)
    return create_async_engine(
        async_url(settings.database_url),
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_pool_size,
        pool_pre_ping=True,
        pool_recycle=POOL_RECYCLE_SECONDS,
        connect_args=connect_args,
    )


async def dispose_engines() -> None:
    """Closes pooled connections on shutdown. Safe to call when an engine was never created."""
    if get_async_engine.cache_info().currsize:
        await get_async_engine().dispose()
        get_async_engine.cache_clear()
    if get_sync_engine.cache_info().currsize:
        get_sync_engine().dispose()
        get_sync_engine.cache_clear()
