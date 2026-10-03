"""Small helpers shared by the fetch and ingest tasks and the recovery sweepers."""
from functools import lru_cache
from typing import Optional

from src.jobs.contract import IngestionRequest
from src.observability.logger import PipelineLogger
from src.db.engine import get_sync_engine
from src.stores.jobs import JobStore

# A job in one of these states has already been committed or permanently failed. A rerun
# (for example a requeue after a worker died between the atomic commit and Procrastinate
# recording success) must not touch it: its source rows are already deleted, so rerunning
# would wrongly mark a complete document failed.
FINISHED_STATUSES = ("complete", "partial_success", "failed")


@lru_cache
def pipeline_logger() -> PipelineLogger:
    """One logger per worker process, shared by every job it runs (its own background writer thread)."""
    return PipelineLogger("nexus_worker", engine=get_sync_engine())


class Progress:
    """Reports job progress as it happens, so GET /ingest/{job_id} reflects a running job."""

    def __init__(self, jobs: JobStore, job_id: str):
        self.jobs = jobs
        self.job_id = job_id

    def set(self, pct: int, metadata: Optional[dict] = None) -> None:
        self.jobs.update_job_status(self.job_id, "processing", pct, metadata=metadata)


def already_finished(jobs: JobStore, job_id: str) -> Optional[str]:
    """Why this job must not run again, or None if it should run. A missing row means the document was deleted."""
    job = jobs.get_job(job_id)
    if job is None:
        return "its document was deleted"
    if job["status"] in FINISHED_STATUSES:
        return f"it already finished with status {job['status']!r}"
    return None


def job_tag(request: IngestionRequest) -> str:
    return f"[job={request.job_id[:8]} doc={request.doc_id[:8]}]"
