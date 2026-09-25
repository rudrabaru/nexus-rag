"""Small helpers shared by the ingestion task and the recovery sweeper (src/jobs/tasks.py)."""
import os
from functools import lru_cache
from typing import Optional

from src.ingestion.errors import UnprocessableSourceError
from src.jobs.contract import IngestionRequest
from src.observability.logger import PipelineLogger
from src.registry.database import DocumentRegistry
from src.registry.engine import get_sync_engine

# A job in one of these states has already been committed or permanently failed. A rerun
# (for example a requeue after a worker died between the atomic commit and Procrastinate
# recording success) must not touch it: the uploaded bytes are already deleted, so rerunning
# would wrongly mark a complete document failed.
FINISHED_STATUSES = ("complete", "partial_success", "failed")


@lru_cache
def pipeline_logger() -> PipelineLogger:
    """One logger per worker process, shared by every job it runs (its own background writer thread)."""
    return PipelineLogger("nexus_worker", engine=get_sync_engine())


class Progress:
    """Reports job progress as it happens, so GET /ingest/{job_id} reflects a running job."""

    def __init__(self, registry: DocumentRegistry, job_id: str):
        self.registry = registry
        self.job_id = job_id

    def set(self, pct: int, metadata: Optional[dict] = None) -> None:
        self.registry.update_job_status(self.job_id, "processing", pct, metadata=metadata)


def already_finished(registry: DocumentRegistry, job_id: str) -> Optional[str]:
    """Why this job must not run again, or None if it should run. A missing row means the document was deleted."""
    job = registry.get_job(job_id)
    if job is None:
        return "its document was deleted"
    if job["status"] in FINISHED_STATUSES:
        return f"it already finished with status {job['status']!r}"
    return None


def resolve_source(request: IngestionRequest, registry: DocumentRegistry, temp_dir: str) -> str:
    """
    The path or URL the dispatcher should read: the request's URL, or an uploaded file written
    to a worker-local temp file (fetched from ingest_sources, the API/worker hand-off).
    """
    if request.url:
        return request.url

    source = registry.get_ingest_source(request.job_id)
    if source is None:
        raise UnprocessableSourceError(f"No uploaded content found for job {request.job_id} (already processed or expired).")
    filename, content = source
    path = os.path.join(temp_dir, filename)
    with open(path, "wb") as f:
        f.write(content)
    return path
