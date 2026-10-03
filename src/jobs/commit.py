"""Atomic commit: an ingestion's chunks, job status, tenant usage and source cleanup land in one transaction."""
from sqlalchemy.engine import Engine

from src.ingestion.embedding_worker import EmbeddingOutcome
from src.jobs.contract import IngestionRequest
from src.retrieving.chunk_writes import delete_stale_chunks, write_chunks
from src.stores.checkpoints import delete_checkpoints
from src.stores.fetches import delete_fetched_pages
from src.stores.jobs import complete_job, delete_ingest_source
from src.stores.tenants import add_embedding_tokens


def commit_ingestion(engine: Engine, request: IngestionRequest, outcome: EmbeddingOutcome) -> None:
    """
    All-or-nothing: chunks, job status, document stats and tenant usage commit together, or
    none of them do. A crash between embedding and this call leaves the job "processing";
    Procrastinate's stalled-job detection (src/jobs/recovery.py) requeues it, and re-running is
    safe because write_chunks upserts are idempotent.

    A re-ingestion (not a resume) replaces the document: its previous chunks that this run did
    not rewrite are deleted in the same transaction (only for pages this run read: a page skipped
    by the fetch quota keeps its chunks), so queries see the old document until the instant the
    new one is in place. A run with failed chunks (partial_success) does not delete
    anything: it must not leave the document with fewer chunks than it had.
    """
    error = outcome.error_reason if outcome.status == "partial_success" else None
    with engine.begin() as conn:
        write_chunks(conn, outcome.chunks)
        if not request.resume and outcome.status == "complete":
            for index_id in {c.index_id for c in outcome.chunks}:
                written = [c for c in outcome.chunks if c.index_id == index_id]
                delete_stale_chunks(
                    conn, request.tenant_id, index_id, request.doc_id,
                    [c.chunk_id for c in written], {c.source_url for c in written},
                )
        add_embedding_tokens(conn, request.tenant_id, outcome.total_tokens)
        complete_job(conn, request.job_id, outcome.stats, status=outcome.status, metadata=outcome.metadata, error=error)
        delete_ingest_source(conn, request.job_id)
        delete_fetched_pages(conn, request.job_id)
        delete_checkpoints(conn, request.job_id)
