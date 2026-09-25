from pathlib import Path

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import create_engine, pool

from src.registry.engine import sync_url
from src.registry.schema import metadata

# Real environment variables win over .env (same rule as the API).
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from src.config import get_settings  # noqa: E402  (must read the environment after .env is loaded)


def _database_url() -> str:
    url = get_settings().database_url
    if not url:
        raise RuntimeError("DATABASE_URL is not set.")
    return sync_url(url).render_as_string(hide_password=False)


def run_migrations_offline() -> None:
    context.configure(url=_database_url(), target_metadata=metadata, literal_binds=True, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


def _include_object(obj, name, type_, reflected, compare_to) -> bool:
    """Procrastinate's tables come from its own frozen schema (migration 0002), not from schema.py."""
    return not (reflected and compare_to is None and (name or "").startswith("procrastinate_"))


def _run(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=metadata,
        compare_type=True,
        include_object=_include_object,
        # Set by the integration tests. Without it Alembic looks the version table up through
        # search_path, finds the real one in `public`, and skips creating the test schema.
        version_table_schema=context.config.attributes.get("version_table_schema"),
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    # A caller may supply its own connection (the integration tests migrate an isolated schema).
    supplied = context.config.attributes.get("connection")
    if supplied is not None:
        _run(supplied)
        return
    engine = create_engine(_database_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        _run(connection)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
