from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from sqlalchemy import delete, func, insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Connection, Engine

from src.db.schema import fetch_log, fetched_pages

QUOTA_WINDOW = timedelta(hours=24)


def delete_fetched_pages(conn: Connection, job_id: str) -> None:
    conn.execute(delete(fetched_pages).where(fetched_pages.c.job_id == job_id))


class FetchStore:
    """Fetched pages waiting for the parse worker, and the audit log of every fetch attempt."""

    def __init__(self, engine: Engine):
        self._engine = engine

    def store_fetched_page(self, job_id: str, url: str, title: Optional[str], markdown: str, provider: str) -> None:
        """Idempotent: a retried fetch job rewrites the same (job_id, url) row."""
        stmt = pg_insert(fetched_pages).values(job_id=job_id, url=url, title=title, markdown=markdown, provider=provider)
        stmt = stmt.on_conflict_do_update(
            index_elements=[fetched_pages.c.job_id, fetched_pages.c.url],
            set_={"title": stmt.excluded.title, "markdown": stmt.excluded.markdown, "provider": stmt.excluded.provider},
        )
        with self._engine.begin() as conn:
            conn.execute(stmt)

    def fetched_urls(self, job_id: str) -> set:
        with self._engine.connect() as conn:
            return set(conn.execute(select(fetched_pages.c.url).where(fetched_pages.c.job_id == job_id)).scalars())

    def get_fetched_pages(self, job_id: str) -> List[Dict]:
        stmt = (
            select(fetched_pages.c.url, fetched_pages.c.title, fetched_pages.c.markdown)
            .where(fetched_pages.c.job_id == job_id)
            .order_by(fetched_pages.c.fetched_at, fetched_pages.c.url)
        )
        with self._engine.connect() as conn:
            return [dict(row) for row in conn.execute(stmt).mappings()]

    def log_fetch(
        self, tenant_id: str, job_id: Optional[str], url: str, outcome: str,
        provider: Optional[str] = None, detail: Optional[str] = None,
    ) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                insert(fetch_log).values(
                    tenant_id=tenant_id, job_id=job_id, url=url, outcome=outcome,
                    provider=provider, detail=(detail or "")[:500] or None,
                )
            )

    def pages_fetched_today(self, tenant_id: str, now: Optional[datetime] = None) -> int:
        """Pages successfully fetched for the tenant in the last 24 hours (the daily quota's measure)."""
        since = (now or datetime.now(timezone.utc)) - QUOTA_WINDOW
        stmt = select(func.count()).select_from(fetch_log).where(
            fetch_log.c.tenant_id == tenant_id, fetch_log.c.outcome == "fetched", fetch_log.c.created_at >= since
        )
        with self._engine.connect() as conn:
            return conn.execute(stmt).scalar_one()
