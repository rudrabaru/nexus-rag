"""
Startup guard: the running code and the database schema must be at the same migration.

Migrations are applied explicitly (`alembic upgrade head`), never by the API on boot: with more
than one instance, concurrent auto-migration races. The API only verifies and refuses to start
on a mismatch, so a skipped migration fails loudly at startup instead of at the first query
that touches a missing column.
"""
from pathlib import Path

from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy.engine import Engine

ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"


def expected_revision() -> str:
    return ScriptDirectory.from_config(Config(str(ALEMBIC_INI))).get_current_head()


def assert_schema_current(engine: Engine) -> None:
    expected = expected_revision()
    with engine.connect() as conn:
        current = MigrationContext.configure(conn).get_current_revision()
    if current != expected:
        raise RuntimeError(
            f"Database schema is at revision {current!r} but the code expects {expected!r}. "
            "Run `alembic upgrade head` against DATABASE_URL."
        )
