"""
Recovery of ingestion jobs orphaned by a worker that died mid-job (killed, OOM, host lost).

Procrastinate records which worker holds each running job and that worker's heartbeat, and
can list jobs whose worker's heartbeat has stopped — but it does not act on them. This does:
a job with retry budget left goes back on the queue; one that has used its budget is marked
failed, in both Procrastinate and the domain tables, so no job stays "processing" forever.

Requeuing is safe because nothing is half-written (chunks, job status and tenant usage commit
in one transaction, src/jobs/commit.py), the uploaded bytes stay in ingest_sources until that
commit, and chunk writes are idempotent upserts.
"""
import asyncio
import logging
from dataclasses import dataclass

from procrastinate import jobs as procrastinate_jobs
from procrastinate.manager import JobManager

from src.jobs.contract import INGEST_QUEUE, INGEST_TASK, MAX_RETRIES
from src.registry.database import DocumentRegistry

logger = logging.getLogger(__name__)

# Three times the worker's 10 s heartbeat interval: a heartbeat missed to a Neon compute
# resume (a few seconds) or a long GC pause is not mistaken for a dead worker. Cost: an
# orphaned job waits at least this long, plus up to a minute for the next sweep, before it
# is requeued. Validated by tests/integration/test_job_queue.py and a real `docker kill`.
STALLED_AFTER_SECONDS = 30


@dataclass
class RecoveryReport:
    requeued: int = 0
    failed: int = 0


async def recover_stalled_jobs(job_manager: JobManager, registry: DocumentRegistry) -> RecoveryReport:
    report = RecoveryReport()
    stalled = await job_manager.get_stalled_jobs(
        queue=INGEST_QUEUE, task_name=INGEST_TASK, seconds_since_heartbeat=STALLED_AFTER_SECONDS
    )
    for job in stalled:
        domain_job_id = job.task_kwargs.get("job_id")
        if job.attempts >= MAX_RETRIES:
            message = f"Worker stopped responding; gave up after {job.attempts + 1} attempts."
            await job_manager.finish_job_by_id_async(job.id, procrastinate_jobs.Status.FAILED, delete_job=False)
            if domain_job_id:
                await asyncio.to_thread(registry.fail_job, domain_job_id, message)
            report.failed += 1
            logger.error(f"[job={str(domain_job_id)[:8]}] {message}")
        else:
            await job_manager.retry_job(job)
            report.requeued += 1
            logger.warning(f"[job={str(domain_job_id)[:8]}] worker stopped responding; requeued (attempt {job.attempts + 2}).")
    return report
