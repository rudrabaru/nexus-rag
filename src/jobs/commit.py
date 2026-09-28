"""Atomic commit: an ingestion's chunks, job status, tenant usage and source cleanup land in one transaction."""
from sqlalchemy.engine import Engine

from src.ingestion.embedding_worker import EmbeddingOutcome
from src.registry.database import add_tenant_tokens
from src.registry.mixins.fetch_store import delete_fetched_pages
from src.registry.mixins.job_store import complete_job, delete_ingest_source
from src.retrieving.chunk_writes import write_chunks


def commit_ingestion(engine: Engine, job_id: str, tenant_id: str, outcome: EmbeddingOutcome) -> None:
    """
    All-or-nothing: chunks, job status, document stats and tenant usage commit together, or
    none of them do. A crash between embedding and this call leaves the job "processing";
    Procrastinate's stalled-job detection (src/jobs/worker.py) requeues it, and re-running is
    safe because write_chunks upserts are idempotent.
    """
    error = outcome.error_reason if outcome.status == "partial_success" else None
    with engine.begin() as conn:
        write_chunks(conn, outcome.chunks)
        add_tenant_tokens(conn, tenant_id, outcome.total_tokens)
        complete_job(conn, job_id, outcome.stats, status=outcome.status, metadata=outcome.metadata, error=error)
        delete_ingest_source(conn, job_id)
        delete_fetched_pages(conn, job_id)
