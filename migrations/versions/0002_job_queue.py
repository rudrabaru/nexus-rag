"""Job queue: Procrastinate's schema (frozen at 3.9.0) and the upload hand-off table.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-23
"""
from pathlib import Path
from typing import Sequence, Union

from alembic import context, op
import sqlalchemy as sa

revision: str = "0002"
down_revision: Union[str, Sequence[str], None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

PROCRASTINATE_SCHEMA = Path(__file__).resolve().parents[1] / "sql" / "procrastinate_3.9.0.sql"

DROP_PROCRASTINATE = r"""
DROP TABLE IF EXISTS procrastinate_events, procrastinate_periodic_defers, procrastinate_jobs, procrastinate_workers CASCADE;
DO $$
DECLARE f record;
BEGIN
    FOR f IN SELECT p.oid::regprocedure AS signature
               FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
              WHERE p.proname LIKE 'procrastinate\_%' AND n.nspname = current_schema()
    LOOP
        EXECUTE 'DROP FUNCTION IF EXISTS ' || f.signature || ' CASCADE';
    END LOOP;
END $$;
DROP TYPE IF EXISTS procrastinate_job_to_defer_v1, procrastinate_job_event_type, procrastinate_job_status CASCADE;
"""


def _run_script(sql: str) -> None:
    """
    Runs a multi-statement script through the raw driver connection with no parameters.
    Procrastinate's schema contains literal '%' in PL/pgSQL RAISE messages, which psycopg
    would parse as placeholders if the script went through SQLAlchemy's execute path.
    """
    if context.is_offline_mode():
        op.execute(sql)
    else:
        op.get_bind().connection.driver_connection.execute(sql)


def upgrade() -> None:
    _run_script(PROCRASTINATE_SCHEMA.read_text(encoding="utf-8"))

    op.create_table(
        "ingest_sources",
        sa.Column("job_id", sa.Text(), nullable=False),
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("content", sa.LargeBinary(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["job_id"], ["jobs.job_id"], name="fk_ingest_sources_job_id_jobs", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("job_id", name="pk_ingest_sources"),
    )
    op.create_index("ix_ingest_sources_tenant_id", "ingest_sources", ["tenant_id"])


def downgrade() -> None:
    op.drop_table("ingest_sources")
    _run_script(DROP_PROCRASTINATE)
