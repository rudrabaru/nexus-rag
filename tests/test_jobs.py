"""
The job queue contract and wiring, tested without a real database via Procrastinate's
InMemoryConnector. Behaviour that needs real Postgres (locking, the ingest_sources hand-off,
atomic commit) is in tests/integration/test_jobs.py.
"""
import procrastinate
import pytest
from procrastinate.testing import InMemoryConnector

from src.jobs.contract import INGEST_QUEUE, INGEST_TASK, INGEST_TASK_NAME, MAX_RETRIES, TASK_NAMESPACE, IngestionRequest
from src.jobs.tasks import blueprint


# ── IngestionRequest ─────────────────────────────────────────────────────────

def test_a_request_needs_exactly_one_source():
    with pytest.raises(ValueError):
        IngestionRequest(job_id="j", doc_id="d", tenant_id="t")
    with pytest.raises(ValueError):
        IngestionRequest(job_id="j", doc_id="d", tenant_id="t", url="https://a.example", filename="a.pdf")


def test_source_ref_prefers_the_url():
    request = IngestionRequest(job_id="j", doc_id="d", tenant_id="t", url="https://a.example")
    assert request.source_ref == "https://a.example"


def test_source_ref_is_a_stable_upload_uri_for_files():
    request = IngestionRequest(job_id="j", doc_id="d1", tenant_id="t", filename="report.pdf")
    assert request.source_ref == "upload://d1/report.pdf"


def test_request_round_trips_through_defer_kwargs():
    """A defer() call passes **request.model_dump(); every field must survive that trip."""
    request = IngestionRequest(job_id="j", doc_id="d", tenant_id="t", url="https://a.example", extract_visuals=True, resume=True)
    assert IngestionRequest(**request.model_dump()) == request


# ── Wiring: the API's task name resolves to the worker's registered task ────
#
# add_tasks_from mutates the blueprint it is given (renaming its tasks into the namespace),
# so it must run exactly once per process — the same constraint production code has
# (src/jobs/worker.py calls it once at import time). One module-scoped app is built here and
# shared by the tests below, rather than one per test.

@pytest.fixture(scope="module")
def worker_app() -> procrastinate.App:
    app = procrastinate.App(connector=InMemoryConnector())
    app.add_tasks_from(blueprint, namespace=TASK_NAMESPACE)
    return app


def test_the_worker_registers_the_task_under_the_name_the_api_defers_to(worker_app):
    assert INGEST_TASK in worker_app.tasks
    assert worker_app.tasks[INGEST_TASK].name == INGEST_TASK
    # INGEST_TASK_NAME + TASK_NAMESPACE must combine (":"-joined) into INGEST_TASK, or the
    # API (which only knows the string, never imports the task) would defer into the void.
    assert INGEST_TASK == f"{TASK_NAMESPACE}:{INGEST_TASK_NAME}"


async def test_a_job_deferred_by_name_alone_is_fetched_by_the_worker_app(worker_app):
    """Mirrors the real split: the API defers via configure_task(name), never importing tasks.py."""
    worker_app.connector.reset()
    deferer = worker_app.configure_task(INGEST_TASK, queue=INGEST_QUEUE, lock="doc-1")
    await deferer.defer_async(job_id="j1", doc_id="doc-1", tenant_id="t1", url="https://a.example")

    worker_id = await worker_app.job_manager.register_worker()
    job = await worker_app.job_manager.fetch_job(queues=[INGEST_QUEUE], worker_id=worker_id)
    assert job is not None
    assert job.task_name == INGEST_TASK
    assert job.lock == "doc-1"
    assert job.task_kwargs["doc_id"] == "doc-1"


def test_retry_budget_matches_the_contract(worker_app):
    task = worker_app.tasks[INGEST_TASK]
    assert task.retry_strategy.max_attempts == MAX_RETRIES


# ── Never rerun a job that already finished ─────────────────────────────────

from unittest.mock import MagicMock  # noqa: E402

from src.jobs.support import already_finished  # noqa: E402


@pytest.mark.parametrize("status", ["complete", "partial_success", "failed"])
def test_a_finished_job_is_not_rerun(status):
    """
    Regression class: a worker that dies after the atomic commit but before Procrastinate
    records success gets requeued. The uploaded bytes are gone by then, so a rerun would
    fail and mark a COMPLETE document failed.
    """
    registry = MagicMock()
    registry.get_job.return_value = {"status": status}
    assert status in already_finished(registry, "j")


@pytest.mark.parametrize("status", ["queued", "processing"])
def test_an_unfinished_job_runs(status):
    registry = MagicMock()
    registry.get_job.return_value = {"status": status}
    assert already_finished(registry, "j") is None


def test_a_job_whose_document_was_deleted_is_not_run():
    registry = MagicMock()
    registry.get_job.return_value = None
    assert "deleted" in already_finished(registry, "j")


def test_the_recovery_sweep_is_registered_as_a_periodic_task(worker_app):
    from src.jobs.contract import RECOVERY_TASK_NAME

    assert f"{TASK_NAMESPACE}:{RECOVERY_TASK_NAME}" in worker_app.tasks
    assert any(name == f"{TASK_NAMESPACE}:{RECOVERY_TASK_NAME}" for name, _ in worker_app.periodic_registry.periodic_tasks)
