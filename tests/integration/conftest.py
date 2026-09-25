"""
Integration fixtures: real Postgres + pgvector, from TEST_DATABASE_URL (skipped when unset).

Isolation: each session migrates into a fresh schema (nexus_test_<random>) and drops it with
CASCADE at the end. `public` stays on the search_path only because the pgvector extension's
types live there.

That fallback is exactly what makes isolation fragile: any table missing from the test schema
silently resolves to the real one in `public`. This once happened (Alembic found
public.alembic_version, skipped the migration, and the cleanup TRUNCATE emptied the real
tables), so isolation is enforced three ways:
1. Alembic's version table is pinned to the test schema.
2. The session aborts before any test runs unless every table exists in the test schema.
3. Cleanup names each table with its schema, so it cannot reach `public`.
A separate Neon branch for TEST_DATABASE_URL is still the better practice.
"""
import asyncio
import os
import sys
import uuid

import procrastinate
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from src.registry.engine import async_connect_args, async_url, libpq_url, session_options, sync_url
from src.registry.schema import metadata
from src.registry.schema_version import ALEMBIC_INI

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")


@pytest.fixture(scope="session")
def event_loop_policy():
    """
    Overrides pytest-asyncio's default for this test session only. Procrastinate's
    PsycopgConnector uses psycopg's async pool, which refuses Windows' default
    ProactorEventLoop ("Psycopg cannot use the 'ProactorEventLoop' to run in async mode").
    asyncpg (pg_async_engine, above) tolerates either loop, so scoping this to the
    integration session is enough; it does not affect src/api/main.py, which sets Proactor
    for uvicorn in the API's own process.
    """
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.get_event_loop_policy()


@pytest.fixture(scope="session")
def test_schema():
    if not TEST_DATABASE_URL:
        # Locally, skipping is fine. In CI a missing database means broken wiring, and a
        # silent skip would let the whole integration suite stop running unnoticed.
        if os.environ.get("CI"):
            pytest.fail("TEST_DATABASE_URL is not set in CI; the integration suite would silently skip.")
        pytest.skip("TEST_DATABASE_URL is not set")
    return f"nexus_test_{uuid.uuid4().hex[:10]}"


def _tables_in_schema(conn, schema: str) -> set:
    rows = conn.execute(
        text("SELECT table_name FROM information_schema.tables WHERE table_schema = :s"), {"s": schema}
    )
    return {r[0] for r in rows}


@pytest.fixture(scope="session")
def pg_engine(test_schema):
    admin = create_engine(sync_url(TEST_DATABASE_URL), poolclass=NullPool)
    with admin.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{test_schema}"'))

    engine = create_engine(
        sync_url(TEST_DATABASE_URL),
        poolclass=NullPool,
        connect_args={"options": session_options({"search_path": f"{test_schema},public"})},
    )
    try:
        config = Config(str(ALEMBIC_INI))
        config.attributes["version_table_schema"] = test_schema
        with engine.begin() as conn:
            config.attributes["connection"] = conn
            command.upgrade(config, "head")

        with engine.connect() as conn:
            missing = set(metadata.tables) - _tables_in_schema(conn, test_schema)
        if missing:
            pytest.exit(
                f"Refusing to run: tables {sorted(missing)} are not in {test_schema}, so queries "
                "would resolve to the real tables in `public`.",
                returncode=3,
            )

        yield engine
    finally:
        engine.dispose()
        with admin.begin() as conn:
            conn.execute(text(f'DROP SCHEMA "{test_schema}" CASCADE'))
        admin.dispose()


@pytest.fixture
def pg_async_engine(test_schema, pg_engine):
    """
    Built with the production connect args (plus the test schema), so the integration tests
    exercise the same session-settings path the API uses. NullPool: asyncpg connections are
    bound to the event loop that opened them, and each test has its own loop.
    """
    engine = create_async_engine(
        async_url(TEST_DATABASE_URL),
        poolclass=NullPool,
        connect_args=async_connect_args(TEST_DATABASE_URL, {"search_path": f"{test_schema},public"}),
    )
    yield engine


@pytest.fixture
async def procrastinate_app(test_schema, pg_engine):
    """
    A real Procrastinate App against the test schema (its own tables, from migration 0002,
    are not part of src.registry.schema.metadata and so are not truncated between tests).
    Tests register their own task via @app.task(...) rather than importing src.jobs.tasks's
    blueprint, which can only be namespaced into an App once per process (see tests/test_jobs.py).

    search_path is the test schema ALONE, with no `public` fallback. Procrastinate's own
    tables and functions never reference anything in `public`, and once `public` also carries
    Procrastinate's schema (as the real deployment does), including it here makes every
    procrastinate_*_v1/v2 function ambiguous between the two schemas — Postgres refuses to
    guess which one a loosely-typed call means ("function ... is not unique") — a real failure
    this caused during development, not a hypothetical one.
    """
    connector = procrastinate.PsycopgConnector(
        conninfo=libpq_url(TEST_DATABASE_URL),
        min_size=1,
        max_size=2,
        kwargs={"options": session_options({"search_path": test_schema})},
    )
    app = procrastinate.App(connector=connector)
    async with app.open_async():
        # Procrastinate's tables are outside src.registry.schema.metadata, so clean_tables does
        # not empty them: without this, an earlier test's job (and the lock it holds) blocks a
        # later test's. search_path is the test schema alone, so this cannot reach `public`.
        await app.connector.execute_query_async(
            "TRUNCATE procrastinate_events, procrastinate_periodic_defers, procrastinate_jobs, "
            "procrastinate_workers RESTART IDENTITY CASCADE"
        )
        yield app


@pytest.fixture
def clean_tables(pg_engine, test_schema):
    """Empties every table between tests. Schema-qualified, so it can never reach `public`."""
    yield
    qualified = ", ".join(f'"{test_schema}"."{name}"' for name in metadata.tables)
    with pg_engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {qualified} RESTART IDENTITY CASCADE"))
