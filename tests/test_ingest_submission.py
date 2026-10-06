"""Submitting an ingestion: policy first, then durable registration, upload validation and queue limits."""
import pytest
from src.jobs.contract import FETCH_QUEUE, FETCH_TASK, INGEST_QUEUE, INGEST_TASK
from src.services.errors import InvalidRequest, PayloadTooLarge, QuotaExceeded
from src.services.uploads import MAX_UPLOAD_BYTES
from src.services.ingestion_service import (
    MAX_ACTIVE_JOBS_PER_TENANT,
    MAX_PENDING_UPLOAD_BYTES,
)
from tests.support.ingest_security import Wired, resolve_to


class FakeUpload:
    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self._content = content

    async def read(self, size: int = -1) -> bytes:
        chunk, self._content = (self._content, b"") if size < 0 else (self._content[:size], self._content[size:])
        return chunk


@pytest.mark.asyncio
async def test_ingestion_rejects_a_local_path_in_the_url_field():
    wired = Wired()
    with pytest.raises(InvalidRequest):
        await wired.submit(url="/app/logs.txt")
    wired.documents.chunk_count.assert_not_called()


@pytest.mark.asyncio
async def test_ingestion_rejects_malformed_port():
    with pytest.raises(InvalidRequest):
        await Wired().submit(url="https://example.com:99999/")


@pytest.mark.asyncio
async def test_a_valid_url_is_registered_then_deferred_to_the_fetch_queue(monkeypatch):
    """The API never fetches: a URL goes to the fetch worker, which later defers ingest."""
    resolve_to(monkeypatch, "93.184.216.34")
    wired = Wired()

    response = await wired.submit(url="https://example.com/docs")

    assert response.status == "queued"
    wired.jobs.register_job.assert_called_once()
    wired.queue.configure_task.assert_called_once_with(FETCH_TASK, queue=FETCH_QUEUE, lock=wired.jobs.register_job.call_args[0][1])
    wired.queue.configure_task.return_value.defer.assert_called_once()
    deferred = wired.queue.configure_task.return_value.defer.call_args.kwargs
    assert deferred["job_id"] == response.job_id
    assert deferred["url"] == "https://example.com/docs"
    assert deferred["filename"] is None


@pytest.mark.asyncio
async def test_unsupported_file_extension_is_rejected():
    with pytest.raises(InvalidRequest):
        await Wired().submit(file=FakeUpload("a.exe", b"x"))


@pytest.mark.asyncio
async def test_oversized_upload_is_rejected():
    with pytest.raises(PayloadTooLarge):
        await Wired().submit(file=FakeUpload("a.txt", b"x" * (MAX_UPLOAD_BYTES + 1)))


@pytest.mark.asyncio
async def test_too_many_pending_uploads_are_rejected():
    """Bounds Neon's 0.5 GB storage against an offline or backlogged worker."""
    wired = Wired(pending_bytes=MAX_PENDING_UPLOAD_BYTES)
    with pytest.raises(QuotaExceeded):
        await wired.submit(file=FakeUpload("a.txt", b"x"))
    wired.jobs.register_job.assert_not_called()


@pytest.mark.asyncio
async def test_duplicate_upload_short_circuits_without_deferring_a_job():
    wired = Wired(existing_by_hash={"doc_id": "doc-1", "status": "complete", "source": "u", "format": "txt"})

    response = await wired.submit(file=FakeUpload("a.txt", b"same"))

    assert response.status == "complete"
    wired.queue.configure_task.assert_not_called()


@pytest.mark.asyncio
async def test_a_valid_upload_stores_its_bytes_with_the_job_not_on_local_disk():
    wired = Wired()

    response = await wired.submit(file=FakeUpload("a.txt", b"hello"))

    assert response.status == "queued"
    assert wired.jobs.register_job.call_args.kwargs["upload"] == ("a.txt", b"hello")
    wired.queue.configure_task.assert_called_once_with(INGEST_TASK, queue=INGEST_QUEUE, lock=wired.jobs.register_job.call_args[0][1])
    deferred = wired.queue.configure_task.return_value.defer.call_args.kwargs
    assert deferred["filename"] == "a.txt" and deferred["url"] is None


@pytest.mark.asyncio
async def test_a_tenant_with_too_much_queued_work_is_refused_until_some_finishes():
    wired = Wired(active_jobs=MAX_ACTIVE_JOBS_PER_TENANT)
    with pytest.raises(QuotaExceeded):
        await wired.submit(file=FakeUpload("a.txt", b"x"))
    wired.jobs.register_job.assert_not_called()


@pytest.mark.asyncio
async def test_re_ingesting_never_deletes_the_existing_document_up_front():
    """Workers run on demand: the old chunks must keep serving until the new run commits (src/jobs/commit.py)."""
    wired = Wired()
    await wired.submit(file=FakeUpload("a.txt", b"v2"), resume=False)
    wired.documents.delete_document.assert_not_called()


@pytest.mark.asyncio
async def test_a_job_the_queue_cannot_take_is_failed_and_reported_not_left_queued_forever():
    from src.services.errors import Unavailable

    wired = Wired()
    wired.queue.configure_task.return_value.defer.side_effect = RuntimeError("connection refused")
    with pytest.raises(Unavailable):
        await wired.submit(file=FakeUpload("a.txt", b"x"))

    failed_job_id, reason = wired.jobs.fail_job.call_args.args
    assert failed_job_id == wired.jobs.register_job.call_args.args[0]
    assert "refused" not in reason  # the cause is logged, not stored for callers
