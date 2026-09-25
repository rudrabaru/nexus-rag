from typing import Any, Dict, Optional, Tuple

from sqlalchemy import bindparam, case, delete, func, literal, select, update
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.engine import Connection

from src.registry.rows import row_to_dict, utcnow
from src.registry.schema import documents, ingest_sources, jobs

TERMINAL_STATUSES = ("complete", "failed")
STATUSES_WITH_DOCUMENT_ERROR = ("failed", "partial_success")
PARTIAL_SUCCESS_DEFAULT_ERROR = "Some chunks or pages failed processing (e.g., API rate limits or crawling errors)."


def _merged_metadata(new_meta: Dict[str, Any]):
    """Merges keys into jobs.metadata atomically in SQL (JSONB ||), with no read-modify-write race."""
    return func.coalesce(jobs.c.metadata, literal({}, JSONB)).op("||")(bindparam("new_meta", new_meta, type_=JSONB))


def _error_from_metadata(meta: Optional[Dict[str, Any]]) -> Optional[str]:
    if isinstance(meta, dict):
        return meta.get("error_reason") or meta.get("error")
    return None


def _doc_of_job(job_id: str):
    return select(jobs.c.doc_id).where(jobs.c.job_id == job_id).scalar_subquery()


def complete_job(
    conn: Connection,
    job_id: str,
    stats: Dict[str, Any],
    status: str = "complete",
    metadata: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    """Marks a job complete (or partial_success) and records the document's final stats, on the caller's transaction."""
    now = utcnow()
    if metadata is not None:
        conn.execute(update(jobs).where(jobs.c.job_id == job_id).values(metadata=_merged_metadata(metadata)))

    if error is None:
        stored = conn.execute(select(jobs.c.metadata).where(jobs.c.job_id == job_id)).scalar_one_or_none()
        error = _error_from_metadata(metadata) or _error_from_metadata(stored)
    if error is None and status == "partial_success":
        error = PARTIAL_SUCCESS_DEFAULT_ERROR

    conn.execute(
        update(jobs)
        .where(jobs.c.job_id == job_id)
        .values(status=status, progress_pct=100, finished_at=now, error=func.coalesce(error, jobs.c.error))
    )
    conn.execute(
        update(documents)
        .where(documents.c.doc_id == _doc_of_job(job_id))
        .values(status=status, updated_at=now, stats=stats, error=func.coalesce(error, documents.c.error))
    )


def delete_ingest_source(conn: Connection, job_id: str) -> None:
    conn.execute(delete(ingest_sources).where(ingest_sources.c.job_id == job_id))


class JobStoreMixin:
    """Ingestion job rows, the document status they drive, and uploads waiting for the worker."""

    def register_job(
        self,
        job_id: str,
        doc_id: str,
        source: str,
        format: str,
        tenant_id: str,
        content_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        upload: Optional[Tuple[str, bytes]] = None,
    ) -> None:
        """
        Creates a queued job and upserts its document as pending (a complete document stays
        complete). An uploaded file (filename, bytes) is stored in the same transaction, so a
        queued job never exists without its source.
        """
        now = utcnow()
        upsert_document = insert(documents).values(
            doc_id=doc_id, tenant_id=tenant_id, source=source, format=format, status="pending",
            content_hash=content_hash, ingested_at=now, updated_at=now,
        )
        upsert_document = upsert_document.on_conflict_do_update(
            index_elements=[documents.c.doc_id],
            set_={
                "status": case((documents.c.status == "complete", "complete"), else_="pending"),
                "updated_at": now,
            },
        )
        with self._engine.begin() as conn:
            conn.execute(upsert_document)
            conn.execute(
                insert(jobs).values(
                    job_id=job_id, doc_id=doc_id, status="queued", progress_pct=0, created_at=now, metadata=metadata
                )
            )
            if upload is not None:
                filename, content = upload
                conn.execute(
                    insert(ingest_sources).values(job_id=job_id, tenant_id=tenant_id, filename=filename, content=content)
                )

    def update_job_status(
        self,
        job_id: str,
        status: str,
        progress_pct: int = 0,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        values = {"status": status, "progress_pct": progress_pct}
        if status in TERMINAL_STATUSES:
            values.update(finished_at=utcnow(), error=error)
        elif error is not None:
            values["error"] = error
        if metadata is not None:
            values["metadata"] = _merged_metadata(metadata)

        with self._engine.begin() as conn:
            conn.execute(update(jobs).where(jobs.c.job_id == job_id).values(**values))
            if status in STATUSES_WITH_DOCUMENT_ERROR and error:
                conn.execute(
                    update(documents)
                    .where(documents.c.doc_id == _doc_of_job(job_id))
                    .values(status=status, error=error, updated_at=utcnow())
                )

    def complete_job(
        self,
        job_id: str,
        stats: Dict[str, Any],
        status: str = "complete",
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Convenience wrapper around the module-level complete_job, opening its own transaction."""
        with self._engine.begin() as conn:
            complete_job(conn, job_id, stats, status=status, metadata=metadata, error=error)

    def fail_job(self, job_id: str, error: str) -> None:
        """Final failure: marks the job and its document failed and discards the pending upload, atomically."""
        now = utcnow()
        with self._engine.begin() as conn:
            conn.execute(
                update(jobs).where(jobs.c.job_id == job_id).values(status="failed", finished_at=now, error=error)
            )
            conn.execute(
                update(documents)
                .where(documents.c.doc_id == _doc_of_job(job_id))
                .values(status="failed", error=error, updated_at=now)
            )
            delete_ingest_source(conn, job_id)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._engine.connect() as conn:
            return row_to_dict(conn.execute(select(jobs).where(jobs.c.job_id == job_id)).first())

    def get_ingest_source(self, job_id: str) -> Optional[Tuple[str, bytes]]:
        """The uploaded (filename, bytes) waiting for this job, or None once committed or discarded."""
        stmt = select(ingest_sources.c.filename, ingest_sources.c.content).where(ingest_sources.c.job_id == job_id)
        with self._engine.connect() as conn:
            row = conn.execute(stmt).first()
        return (row.filename, bytes(row.content)) if row else None

    def pending_upload_bytes(self, tenant_id: str) -> int:
        """Bytes of uploads a tenant has queued but not yet processed."""
        stmt = select(func.coalesce(func.sum(func.octet_length(ingest_sources.c.content)), 0)).where(
            ingest_sources.c.tenant_id == tenant_id
        )
        with self._engine.connect() as conn:
            return int(conn.execute(stmt).scalar_one())
