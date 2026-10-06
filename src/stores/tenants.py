from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection

from src.db.rows import utcnow
from src.db.schema import tenants


def add_embedding_tokens(conn: Connection, tenant_id: str, tokens: int) -> None:
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
