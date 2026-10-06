"""Migrations and the schema as real Postgres sees it."""
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from src.db.schema_version import ALEMBIC_INI, assert_schema_current


pytestmark = pytest.mark.usefixtures("clean_tables")


def test_unqualified_names_resolve_to_the_throwaway_schema(pg_engine, test_schema):
    """Guards the isolation every other test depends on (see conftest)."""

    schema_of = text(
        "SELECT n.nspname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.oid = to_regclass(:t)"
    )
    with pg_engine.connect() as conn:
        for table in ("chunks", "api_keys", "alembic_version"):
            assert conn.execute(schema_of, {"t": table}).scalar() == test_schema, table


def test_database_is_at_the_code_revision(pg_engine):
    assert_schema_current(pg_engine)


def test_migrations_match_the_schema_module(pg_engine):
    """`alembic check`: autogenerate finds no difference between schema.py and the migrated database."""
    config = Config(str(ALEMBIC_INI))
    with pg_engine.connect() as conn:
        config.attributes["connection"] = conn
        command.check(config)


async def test_search_connections_receive_the_hnsw_settings(pg_async_engine):
    """
    Regression: settings sent as individual startup parameters were silently dropped by
    Neon's proxy, so tenant-filtered search would run without iterative scans.
    """

    async with pg_async_engine.connect() as conn:
        iterative = (await conn.execute(text("SELECT current_setting('hnsw.iterative_scan', true)"))).scalar()
        ef_search = (await conn.execute(text("SELECT current_setting('hnsw.ef_search', true)"))).scalar()
    assert (iterative, ef_search) == ("relaxed_order", "100")
