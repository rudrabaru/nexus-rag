from typing import Any, Dict, Optional, Tuple

from sqlalchemy import bindparam, case, delete, func, literal, select, update
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.engine import Connection, Engine

from src.db.rows import row_to_dict, utcnow
from src.db.schema import chunks, documents, ingest_sources, jobs
from src.stores.fetches import delete_fetched_pages

TERMINAL_STATUSES = ("complete", "failed")
ACTIVE_STATUSES = ("queued", "processing")
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
    """
    Marks a job complete (or partial_success) and records the document's final stats, on the
    caller's transaction. A complete job clears the document's error: a failed earlier attempt
    must not outlive a success.
    """
    now = utcnow()
    if metadata is not None:
        conn.execute(update(jobs).where(jobs.c.job_id == job_id).values(metadata=_merged_metadata(metadata)))

    if error is None and status == "partial_success":
        stored = conn.execute(select(jobs.c.metadata).where(jobs.c.job_id == job_id)).scalar_one_or_none()
        error = _error_from_metadata(metadata) or _error_from_metadata(stored) or PARTIAL_SUCCESS_DEFAULT_ERROR

    conn.execute(
        update(jobs).where(jobs.c.job_id == job_id).values(status=status, progress_pct=100, finished_at=now, error=error)
    )
    conn.execute(
        update(documents)
        .where(documents.c.doc_id == _doc_of_job(job_id))
        .values(status=status, updated_at=now, stats=stats, error=error)
    )


def delete_ingest_source(conn: Connection, job_id: str) -> None:
    conn.execute(delete(ingest_sources).where(ingest_sources.c.job_id == job_id))


class JobStore:
    """Ingestion job rows, the document status they drive, and uploads waiting for the worker."""

    def __init__(self, engine: Engine):
        self._engine = engine

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
        complete and keeps serving queries until the new run commits). An uploaded file
        (filename, bytes) is stored in the same transaction, so a queued job never exists without
        its source.
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
                "content_hash": func.coalesce(upsert_document.excluded.content_hash, documents.c.content_hash),
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
            if status == "complete":  # e.g. a resume with nothing left to do: the document is complete too
                conn.execute(
                    update(documents)
                    .where(documents.c.doc_id == _doc_of_job(job_id))
                    .values(status="complete", error=None, updated_at=utcnow())
                )

    def fail_job(self, job_id: str, error: str) -> None:
        """
        Final failure: marks the job failed and discards its pending upload or fetched pages,
        atomically. The document is marked failed only when it holds no chunks; a document being
        re-ingested keeps serving its previous chunks, and carries the error as a note.
        """
        now = utcnow()
        has_chunks = select(chunks.c.chunk_id).where(chunks.c.doc_id == _doc_of_job(job_id)).exists()
        with self._engine.begin() as conn:
            conn.execute(
                update(jobs).where(jobs.c.job_id == job_id).values(status="failed", finished_at=now, error=error)
            )
            conn.execute(
                update(documents)
                .where(documents.c.doc_id == _doc_of_job(job_id))
                .values(status=case((has_chunks, documents.c.status), else_="failed"), error=error, updated_at=now)
            )
            delete_ingest_source(conn, job_id)
            delete_fetched_pages(conn, job_id)

    def complete_as_duplicate(self, job_id: str, duplicate_of: str) -> None:
        """
        The job's content is already indexed as another document of the same tenant. Like the
        API's upload dedup, the job is re-pointed at that document and completed, and its source
        rows are discarded. The placeholder document it registered is deleted too, unless it
        holds chunks or other jobs (a resumed document), all in one transaction.
        """
        with self._engine.begin() as conn:
            placeholder = conn.execute(select(jobs.c.doc_id).where(jobs.c.job_id == job_id)).scalar_one_or_none()
            conn.execute(
                update(jobs)
                .where(jobs.c.job_id == job_id)
                .values(
                    doc_id=duplicate_of, status="complete", progress_pct=100, finished_at=utcnow(),
                    metadata=_merged_metadata({"duplicate_of": duplicate_of}),
                )
            )
            delete_ingest_source(conn, job_id)
            delete_fetched_pages(conn, job_id)
            if placeholder and placeholder != duplicate_of:
                conn.execute(
                    delete(documents).where(
                        documents.c.doc_id == placeholder,
                        ~select(chunks.c.chunk_id).where(chunks.c.doc_id == placeholder).exists(),
                        ~select(jobs.c.job_id).where(jobs.c.doc_id == placeholder).exists(),
                    )
                )

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

    def active_job_count(self, tenant_id: str) -> int:
        """Jobs the tenant has queued or running. Bounds how much work one tenant can pile up while no worker runs."""
        stmt = (
            select(func.count())
            .select_from(jobs.join(documents, jobs.c.doc_id == documents.c.doc_id))
            .where(documents.c.tenant_id == tenant_id, jobs.c.status.in_(ACTIVE_STATUSES))
        )
        with self._engine.connect() as conn:
            return conn.execute(stmt).scalar_one()
