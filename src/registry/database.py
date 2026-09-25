from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection, Engine

from src.registry.mixins.document_store import DocumentStoreMixin
from src.registry.mixins.job_store import JobStoreMixin
from src.registry.rows import utcnow
from src.registry.schema import tenants


def add_tenant_tokens(conn: Connection, tenant_id: str, tokens: int) -> None:
    """Adds embedding tokens to the tenant's usage counter, on the caller's transaction."""
    if not tokens:
        return
    stmt = insert(tenants).values(tenant_id=tenant_id, created_at=utcnow(), total_embedding_tokens=tokens)
    conn.execute(
        stmt.on_conflict_do_update(
            index_elements=[tenants.c.tenant_id],
            set_={"total_embedding_tokens": tenants.c.total_embedding_tokens + tokens},
        )
    )


class DocumentRegistry(DocumentStoreMixin, JobStoreMixin):
    """
    Documents, ingestion jobs and pending uploads, stored in Postgres.

    Synchronous by design: every caller runs in a worker thread (asyncio.to_thread or an
    executor), where a blocking driver is correct.
    """

    def __init__(self, engine: Engine):
        self._engine = engine

    def increment_tenant_embedding_tokens(self, tenant_id: str, tokens: int) -> None:
        """Convenience wrapper around add_tenant_tokens, opening its own transaction."""
        with self._engine.begin() as conn:
            add_tenant_tokens(conn, tenant_id, tokens)
