"""Procrastinate locks and recovery of jobs orphaned by a dead worker."""
from datetime import datetime, timedelta, timezone
import pytest
from procrastinate import jobs as procrastinate_jobs
from src.jobs.contract import INGEST_QUEUE, INGEST_TASK


pytestmark = pytest.mark.usefixtures("clean_tables")


async def _defer_ingestion_held_by_a_worker(app, registry, domain_job_id, attempts=0):
    """Registers a domain job, defers its ingestion, and leaves it 'doing' on a worker."""
    registry.register_job(domain_job_id, f"doc-{domain_job_id}", "https://example.com/x", "web", "tenant-1")
    await app.configure_task(INGEST_TASK, queue=INGEST_QUEUE, lock=f"doc-{domain_job_id}").defer_async(job_id=domain_job_id)
    worker_id = await app.job_manager.register_worker()
    job = await app.job_manager.fetch_job(queues=[INGEST_QUEUE], worker_id=worker_id)
    for _ in range(attempts):  # simulate earlier retries by cycling the job through a retry
        # Backdated: retry_job stamps scheduled_at from this machine's clock, which can be a
        # moment ahead of the database's, leaving the job briefly unfetchable.
        await app.job_manager.retry_job(job, retry_at=datetime.now(timezone.utc) - timedelta(minutes=1))
        job = await app.job_manager.fetch_job(queues=[INGEST_QUEUE], worker_id=worker_id)
    return job, worker_id


async def _kill_worker(app, worker_id):
    """A dead worker is one whose heartbeat stopped: age it past the stalled threshold."""
    await app.connector.execute_query_async(
        f"UPDATE procrastinate_workers SET last_heartbeat = now() - interval '10 minutes' WHERE id = {int(worker_id)}"
    )


async def test_a_locked_job_is_not_fetched_while_its_lock_is_held(procrastinate_app):
    """
    Regression class this guards against: two ingestion jobs for the same document (e.g. a
    resume retried while the original run is still in flight) must never run concurrently,
    since both would write that document's chunks at once. src/jobs/queue.py relies on
    Procrastinate's own lock, not application code, to serialise them.
    """
    calls = []

    @procrastinate_app.task(queue="test-lock")
    def noop(**kwargs):
        calls.append(kwargs)

    await noop.configure(lock="doc-x").defer_async(n=1)
    await noop.configure(lock="doc-x").defer_async(n=2)

    worker_id = await procrastinate_app.job_manager.register_worker()
    first = await procrastinate_app.job_manager.fetch_job(queues=["test-lock"], worker_id=worker_id)
    assert first is not None and first.task_kwargs["n"] == 1

    blocked = await procrastinate_app.job_manager.fetch_job(queues=["test-lock"], worker_id=worker_id)
    assert blocked is None  # job 2 shares job 1's lock, which is still 'doing'

    await procrastinate_app.job_manager.finish_job_by_id_async(first.id, procrastinate_jobs.Status.SUCCEEDED, delete_job=False)

    second = await procrastinate_app.job_manager.fetch_job(queues=["test-lock"], worker_id=worker_id)
    assert second is not None and second.task_kwargs["n"] == 2


async def test_a_job_held_by_a_dead_worker_is_requeued(procrastinate_app, registry):
    from src.jobs.recovery import recover_stalled_jobs

    job, worker_id = await _defer_ingestion_held_by_a_worker(procrastinate_app, registry, "job-1")
    await _kill_worker(procrastinate_app, worker_id)

    report = await recover_stalled_jobs(procrastinate_app.job_manager, registry, INGEST_QUEUE, INGEST_TASK)

    assert (report.requeued, report.failed) == (1, 0)
    status = await procrastinate_app.job_manager.get_job_status_async(job.id)
    assert status == procrastinate_jobs.Status.TODO


async def test_a_job_held_by_a_live_worker_is_left_alone(procrastinate_app, registry):
    from src.jobs.recovery import recover_stalled_jobs

    job, _ = await _defer_ingestion_held_by_a_worker(procrastinate_app, registry, "job-1")  # heartbeat is fresh

    report = await recover_stalled_jobs(procrastinate_app.job_manager, registry, INGEST_QUEUE, INGEST_TASK)

    assert (report.requeued, report.failed) == (0, 0)
    assert await procrastinate_app.job_manager.get_job_status_async(job.id) == procrastinate_jobs.Status.DOING


async def test_a_job_that_has_used_its_retries_is_failed_everywhere_not_requeued_forever(procrastinate_app, registry):
    from src.jobs.contract import MAX_RETRIES
    from src.jobs.recovery import recover_stalled_jobs

    job, worker_id = await _defer_ingestion_held_by_a_worker(procrastinate_app, registry, "job-1", attempts=MAX_RETRIES)
    await _kill_worker(procrastinate_app, worker_id)

    report = await recover_stalled_jobs(procrastinate_app.job_manager, registry, INGEST_QUEUE, INGEST_TASK)

    assert (report.requeued, report.failed) == (0, 1)
    assert await procrastinate_app.job_manager.get_job_status_async(job.id) == procrastinate_jobs.Status.FAILED
    assert registry.get_job("job-1")["status"] == "failed"
    assert registry.get_document("doc-job-1")["status"] == "failed"


async def test_other_tasks_stalled_jobs_are_not_touched(procrastinate_app, registry):
    """The sweeper owns ingestion jobs only; it must not requeue an unrelated task's job."""
    from src.jobs.recovery import recover_stalled_jobs

    @procrastinate_app.task(queue="ingest")
    def unrelated(**kwargs):
        pass

    await unrelated.defer_async(x=1)
    worker_id = await procrastinate_app.job_manager.register_worker()
    other = await procrastinate_app.job_manager.fetch_job(queues=["ingest"], worker_id=worker_id)
    await _kill_worker(procrastinate_app, worker_id)

    report = await recover_stalled_jobs(procrastinate_app.job_manager, registry, INGEST_QUEUE, INGEST_TASK)

    assert (report.requeued, report.failed) == (0, 0)
    assert await procrastinate_app.job_manager.get_job_status_async(other.id) == procrastinate_jobs.Status.DOING
