"""
What every queue's task does around its work, and the sweep that recovers its dead jobs, so the
two queues cannot drift apart.

Failure handling:
- UnprocessableSourceError means retrying would produce the same result (no usable content, content
  too large, a provider that rejects the request). It is not re-raised: the job is marked failed and
  Procrastinate sees the task as having run, so it is never retried.
- Any other exception is re-raised so Procrastinate retries it with backoff. On the attempt that
  exhausts the retry budget the job is also marked failed here, so the domain status does not stay
  "processing" once Procrastinate has given up.
"""
import asyncio
import logging
from typing import Awaitable, Callable

import procrastinate

from src.db.engine import get_sync_engine
from src.errors import UnprocessableSourceError
from src.jobs.contract import MAX_RETRIES, RECOVERY_CRON, IngestionRequest
from src.jobs.recovery import recover_stalled_jobs
from src.jobs.support import already_finished
from src.stores.jobs import JobStore

logger = logging.getLogger(__name__)


async def run_with_policy(
    context: procrastinate.JobContext, request: IngestionRequest, work: Callable[[], Awaitable[None]], verb: str
) -> None:
    """Runs `work` for one job; `verb` ("ingestion", "fetching") words the messages."""
    tag = f"[job={request.job_id[:8]}]"
    jobs = JobStore(get_sync_engine())

    skip_reason = await asyncio.to_thread(already_finished, jobs, request.job_id)
    if skip_reason:
        logger.info(f"{tag} not running: {skip_reason}.")
        return

    try:
        await work()
    except UnprocessableSourceError as e:
        logger.error(f"{tag} unprocessable source, not retrying: {e}")
        await asyncio.to_thread(jobs.fail_job, request.job_id, str(e))
    except Exception as e:
        attempt = context.job.attempts + 1
        logger.error(f"{tag} attempt {attempt}/{MAX_RETRIES + 1} failed: {e}", exc_info=True)
        if context.job.attempts >= MAX_RETRIES:
            message = f"{verb.capitalize()} failed after {attempt} attempts ({type(e).__name__}); details are in the worker log."
            await asyncio.to_thread(jobs.fail_job, request.job_id, message)
        raise


def register_recovery(blueprint: procrastinate.Blueprint, queue: str, task: str, task_name: str) -> None:
    """Adds the periodic sweep that requeues (or fails, once out of retries) this queue's jobs whose worker died."""

    @blueprint.periodic(cron=RECOVERY_CRON, periodic_id=task_name.replace("_", "-"))
    @blueprint.task(name=task_name, queue=queue, lock=task_name, pass_context=True)  # lock: two workers' sweeps never overlap
    async def recover_stalled(context: procrastinate.JobContext, timestamp: int) -> None:
        report = await recover_stalled_jobs(
            context.app.job_manager, JobStore(get_sync_engine()), queue, task, own_worker_id=context.job.worker_id
        )
        if report.requeued or report.failed:
            logger.warning(f"Stalled-job sweep ({queue}): requeued={report.requeued} failed={report.failed}")
