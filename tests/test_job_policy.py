"""The failure policy both queues share, and the sweep's rule about its own worker's jobs."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.errors import UnprocessableSourceError
from src.jobs import policy
from src.jobs.contract import MAX_RETRIES, IngestionRequest
from src.jobs.recovery import recover_stalled_jobs


@pytest.fixture
def jobs(monkeypatch):
    store = MagicMock()
    monkeypatch.setattr(policy, "JobStore", lambda engine: store)
    monkeypatch.setattr(policy, "get_sync_engine", lambda: None)
    monkeypatch.setattr(policy, "already_finished", lambda store, job_id: None)
    return store


def context(attempts=0):
    return SimpleNamespace(job=SimpleNamespace(attempts=attempts, worker_id=1))


REQUEST = IngestionRequest(job_id="job-12345678", doc_id="doc-1", tenant_id="t", url="https://example.com/a")


async def test_an_unprocessable_source_fails_the_job_without_a_retry(jobs):
    async def work():
        raise UnprocessableSourceError("no usable content")

    await policy.run_with_policy(context(), REQUEST, work, "ingestion")  # not re-raised: Procrastinate must not retry

    jobs.fail_job.assert_called_once_with("job-12345678", "no usable content")


async def test_a_transient_failure_is_retried_and_the_job_stays_open(jobs):
    async def work():
        raise RuntimeError("connection reset")

    with pytest.raises(RuntimeError):
        await policy.run_with_policy(context(attempts=0), REQUEST, work, "fetching")

    jobs.fail_job.assert_not_called()


async def test_the_last_attempt_fails_the_job_with_a_safe_message_and_still_raises(jobs):
    async def work():
        raise RuntimeError("password=hunter2 in a driver message")

    with pytest.raises(RuntimeError):
        await policy.run_with_policy(context(attempts=MAX_RETRIES), REQUEST, work, "fetching")

    message = jobs.fail_job.call_args.args[1]
    assert message.startswith("Fetching failed after") and "RuntimeError" in message and "hunter2" not in message


async def test_a_finished_job_is_not_run_again(jobs, monkeypatch):
    monkeypatch.setattr(policy, "already_finished", lambda store, job_id: "already complete")
    work = AsyncMock()

    await policy.run_with_policy(context(), REQUEST, work, "ingestion")

    work.assert_not_called()


async def test_the_sweep_leaves_the_jobs_of_its_own_worker_alone():
    mine = SimpleNamespace(id=1, worker_id=7, attempts=0, task_kwargs={"job_id": "a"})
    theirs = SimpleNamespace(id=2, worker_id=8, attempts=0, task_kwargs={"job_id": "b"})
    manager = MagicMock()
    manager.get_stalled_jobs = AsyncMock(return_value=[mine, theirs])
    manager.retry_job = AsyncMock()

    report = await recover_stalled_jobs(manager, MagicMock(), "ingest", "nexus:ingest_document", own_worker_id=7)

    assert report.requeued == 1
    manager.retry_job.assert_awaited_once_with(theirs)
