"""Job-row transitions that run inside a caller's transaction, shared by JobStore and the worker's commit."""
from typing import Any, Dict, Optional

from sqlalchemy import bindparam, delete, func, literal, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Connection

from src.db.rows import utcnow
from src.db.schema import documents, ingest_sources, jobs

TERMINAL_STATUSES = ("complete", "failed")
ACTIVE_STATUSES = ("queued", "processing")
PARTIAL_SUCCESS_DEFAULT_ERROR = "Some chunks or pages failed processing (e.g., API rate limits or crawling errors)."


def merged_metadata(new_meta: Dict[str, Any]):
    """Merges keys into jobs.metadata atomically in SQL (JSONB ||), with no read-modify-write race."""
    return func.coalesce(jobs.c.metadata, literal({}, JSONB)).op("||")(bindparam("new_meta", new_meta, type_=JSONB))


def _error_from_metadata(meta: Optional[Dict[str, Any]]) -> Optional[str]:
    if isinstance(meta, dict):
        return meta.get("error_reason") or meta.get("error")
    return None


def doc_of_job(job_id: str):
    return select(jobs.c.doc_id).where(jobs.c.job_id == job_id).scalar_subquery()


def complete_job(
    conn: Connection,
    job_id: str,
    stats: Dict[str, Any],
    status: str = "complete",
    metadata: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    """
    Marks a job complete (or partial_success) and records the document's final stats, on the
    caller's transaction. A complete job clears the document's error: a failed earlier attempt
    must not outlive a success.
    """
    now = utcnow()
    if metadata is not None:
        conn.execute(update(jobs).where(jobs.c.job_id == job_id).values(metadata=merged_metadata(metadata)))

    if error is None and status == "partial_success":
        stored = conn.execute(select(jobs.c.metadata).where(jobs.c.job_id == job_id)).scalar_one_or_none()
        error = _error_from_metadata(metadata) or _error_from_metadata(stored) or PARTIAL_SUCCESS_DEFAULT_ERROR

    conn.execute(
        update(jobs).where(jobs.c.job_id == job_id).values(status=status, progress_pct=100, finished_at=now, error=error)
    )
    conn.execute(
        update(documents)
        .where(documents.c.doc_id == doc_of_job(job_id))
        .values(status=status, updated_at=now, stats=stats, error=error)
    )


def delete_ingest_source(conn: Connection, job_id: str) -> None:
    conn.execute(delete(ingest_sources).where(ingest_sources.c.job_id == job_id))
