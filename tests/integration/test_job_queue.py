"""
Behaviour that only a real Procrastinate + Postgres pairing can prove: the ingest_sources
upload hand-off, commit_ingestion's atomicity, and Procrastinate's own lock serialisation.

The job-name/queue/retry wiring itself (src.jobs.tasks.blueprint under its namespace) is
tested without a database in tests/test_jobs.py, using Procrastinate's InMemoryConnector.
"""
from datetime import datetime, timedelta, timezone

import pytest
from procrastinate import jobs as procrastinate_jobs
from sqlalchemy import select

from src.embedding.models import EmbeddedChunk
from src.ingestion.embedding_worker import EmbeddingOutcome
from src.jobs.commit import commit_ingestion
from src.registry.database import DocumentRegistry
from src.registry.schema import tenants

pytestmark = pytest.mark.usefixtures("clean_tables")


def chunk(chunk_id, tenant="tenant-1", doc_id="doc-1", text="hello world"):
    return EmbeddedChunk(
        chunk_id=chunk_id, source_url=f"https://example.com/{doc_id}", source_document=f"Doc {doc_id}", title="T",
        heading_path=[], chunk_index=0, chunk_text=text, token_count=3, char_start=0, char_end=len(text),
        document_version="v", chunk_version="v", tenant_id=tenant, doc_id=doc_id,
        embedding=[0.1] * 1024, embedding_model="test-model",
    )


@pytest.fixture
def registry(pg_engine):
    return DocumentRegistry(pg_engine)


# ── The upload hand-off (ingest_sources) ─────────────────────────────────────

def test_an_uploaded_files_bytes_are_readable_by_the_worker(registry):
    registry.register_job("job-1", "doc-1", "upload://doc-1/a.pdf", "pdf", "tenant-1", upload=("a.pdf", b"%PDF-1.4 fake"))

    filename, content = registry.get_ingest_source("job-1")
    assert (filename, content) == ("a.pdf", b"%PDF-1.4 fake")
    assert registry.pending_upload_bytes("tenant-1") == len(b"%PDF-1.4 fake")


def test_registering_a_job_with_no_upload_leaves_no_ingest_source(registry):
    registry.register_job("job-1", "doc-1", "https://example.com/doc-1", "web", "tenant-1")
    assert registry.get_ingest_source("job-1") is None


def test_failing_a_job_discards_its_pending_upload(registry):
    registry.register_job("job-1", "doc-1", "upload://doc-1/a.pdf", "pdf", "tenant-1", upload=("a.pdf", b"x" * 100))
    registry.fail_job("job-1", "boom")
    assert registry.get_ingest_source("job-1") is None
    assert registry.pending_upload_bytes("tenant-1") == 0


def test_pending_upload_bytes_only_counts_the_tenants_own_uploads(registry):
    registry.register_job("job-1", "doc-1", "upload://doc-1/a.pdf", "pdf", "tenant-1", upload=("a.pdf", b"x" * 10))
    registry.register_job("job-2", "doc-2", "upload://doc-2/b.pdf", "pdf", "tenant-2", upload=("b.pdf", b"y" * 999))
    assert registry.pending_upload_bytes("tenant-1") == 10


# ── commit_ingestion: chunks, job status, tenant usage and the upload all-or-nothing ──

def test_commit_ingestion_lands_everything_in_one_transaction(pg_engine, registry):
    registry.register_job("job-1", "doc-1", "upload://doc-1/a.pdf", "pdf", "tenant-1", upload=("a.pdf", b"raw bytes"))
    outcome = EmbeddingOutcome(chunks=[chunk("c1"), chunk("c2")], failed_indices=[], total_chunks=2, error_reason=None)

    commit_ingestion(pg_engine, "job-1", "tenant-1", outcome)

    assert registry.get_document("doc-1")["status"] == "complete"
    assert registry.get_document("doc-1")["chunk_count"] == 2
    assert registry.get_ingest_source("job-1") is None  # the hand-off row is cleaned up
    with pg_engine.connect() as conn:
        assert conn.execute(select(tenants.c.total_embedding_tokens).where(tenants.c.tenant_id == "tenant-1")).scalar_one() == 6


def test_commit_ingestion_records_partial_success_with_its_reason(pg_engine, registry):
    registry.register_job("job-1", "doc-1", "https://example.com/doc-1", "web", "tenant-1")
    outcome = EmbeddingOutcome(chunks=[chunk("c1")], failed_indices=[1], total_chunks=2, error_reason="rate limited")

    commit_ingestion(pg_engine, "job-1", "tenant-1", outcome)

    doc = registry.get_document("doc-1")
    assert doc["status"] == "partial_success"
    assert doc["error"] == "rate limited"
    assert doc["chunk_count"] == 1


def test_commit_ingestion_rolls_back_completely_on_a_bad_chunk(pg_engine, registry):
    """A chunk missing its tenant/doc (write_chunks raises) must not leave the job half-committed."""
    registry.register_job("job-1", "doc-1", "https://example.com/doc-1", "web", "tenant-1")
    bad = chunk("c1")
    bad.tenant_id = None
    outcome = EmbeddingOutcome(chunks=[bad], failed_indices=[], total_chunks=1, error_reason=None)

    with pytest.raises(ValueError):
        commit_ingestion(pg_engine, "job-1", "tenant-1", outcome)

    assert registry.get_job("job-1")["status"] == "queued"  # unchanged: the transaction rolled back
    with pg_engine.connect() as conn:
        assert conn.execute(select(tenants.c.tenant_id).where(tenants.c.tenant_id == "tenant-1")).first() is None


# ── Procrastinate against real Postgres: lock serialisation ─────────────────

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


# ── Recovery of jobs orphaned by a dead worker ───────────────────────────────

async def _defer_ingestion_held_by_a_worker(app, registry, domain_job_id, attempts=0):
    """Registers a domain job, defers its ingestion, and leaves it 'doing' on a worker."""
    from src.jobs.contract import INGEST_QUEUE, INGEST_TASK

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


async def test_a_job_held_by_a_dead_worker_is_requeued(procrastinate_app, registry):
    from src.jobs.recovery import recover_stalled_jobs

    job, worker_id = await _defer_ingestion_held_by_a_worker(procrastinate_app, registry, "job-1")
    await _kill_worker(procrastinate_app, worker_id)

    report = await recover_stalled_jobs(procrastinate_app.job_manager, registry)

    assert (report.requeued, report.failed) == (1, 0)
    status = await procrastinate_app.job_manager.get_job_status_async(job.id)
    assert status == procrastinate_jobs.Status.TODO


async def test_a_job_held_by_a_live_worker_is_left_alone(procrastinate_app, registry):
    from src.jobs.recovery import recover_stalled_jobs

    job, _ = await _defer_ingestion_held_by_a_worker(procrastinate_app, registry, "job-1")  # heartbeat is fresh

    report = await recover_stalled_jobs(procrastinate_app.job_manager, registry)

    assert (report.requeued, report.failed) == (0, 0)
    assert await procrastinate_app.job_manager.get_job_status_async(job.id) == procrastinate_jobs.Status.DOING


async def test_a_job_that_has_used_its_retries_is_failed_everywhere_not_requeued_forever(procrastinate_app, registry):
    from src.jobs.contract import MAX_RETRIES
    from src.jobs.recovery import recover_stalled_jobs

    job, worker_id = await _defer_ingestion_held_by_a_worker(procrastinate_app, registry, "job-1", attempts=MAX_RETRIES)
    await _kill_worker(procrastinate_app, worker_id)

    report = await recover_stalled_jobs(procrastinate_app.job_manager, registry)

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

    report = await recover_stalled_jobs(procrastinate_app.job_manager, registry)

    assert (report.requeued, report.failed) == (0, 0)
    assert await procrastinate_app.job_manager.get_job_status_async(other.id) == procrastinate_jobs.Status.DOING
