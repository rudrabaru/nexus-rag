import socket
from unittest.mock import MagicMock

import pytest

from src.crawling.policy import check_fetchable
from src.crawling.url_policy import UnsafeUrlError, validate_public_url
from src.jobs.contract import FETCH_QUEUE, FETCH_TASK, INGEST_QUEUE, INGEST_TASK
from src.services.errors import InvalidRequest, PayloadTooLarge, QuotaExceeded
from src.services.uploads import MAX_UPLOAD_BYTES
from src.services.ingestion_service import (
    MAX_ACTIVE_JOBS_PER_TENANT,
    MAX_PENDING_UPLOAD_BYTES,
    IngestionService,
)


def resolve_to(monkeypatch, *addresses):
    infos = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, 443)) for a in addresses]
    monkeypatch.setattr("src.crawling.url_policy.socket.getaddrinfo", lambda *args, **kwargs: infos)


class FakeUpload:
    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self._content = content

    async def read(self, size: int = -1) -> bytes:
        chunk, self._content = (self._content, b"") if size < 0 else (self._content[:size], self._content[size:])
        return chunk


class Wired:
    """An IngestionService over mock stores and a mock queue (MagicMock attributes are plain sync callables, as asyncio.to_thread wants)."""

    def __init__(self, quota=0, pending_bytes=0, existing_by_hash=None, pages_fetched_today=0, active_jobs=0):
        self.documents, self.jobs, self.fetches, self.queue = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        self.documents.chunk_count.return_value = quota
        self.documents.get_document_by_hash.return_value = existing_by_hash
        self.jobs.pending_upload_bytes.return_value = pending_bytes
        self.jobs.active_job_count.return_value = active_jobs
        self.fetches.pages_fetched_today.return_value = pages_fetched_today
        self.service = IngestionService(self.queue, self.documents, self.jobs, self.fetches)

    async def submit(self, url=None, file=None, resume=False, tenant="tenant-1"):
        return await self.service.submit(tenant, url, file, resume)


# ── URL policy ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "url",
    [
        "/etc/passwd.txt",
        "logs.txt",
        "C:\\Users\\someone\\notes.md",
        "file:///etc/hosts.txt",
        "ftp://example.com/a.pdf",
        "gopher://example.com/",
        "//example.com/a.md",
        "",
    ],
)
def test_non_http_sources_are_rejected_before_any_lookup(url, monkeypatch):
    monkeypatch.setattr(
        "src.crawling.url_policy.socket.getaddrinfo",
        lambda *a, **k: pytest.fail("DNS must not be consulted for a non-http URL"),
    )
    with pytest.raises(UnsafeUrlError):
        validate_public_url(url)


@pytest.mark.parametrize(
    "address",
    ["10.0.0.5", "127.0.0.1", "169.254.169.254", "192.168.1.10", "172.16.0.1", "100.64.0.1", "0.0.0.0", "::1", "fd00::1", "::ffff:127.0.0.1"],
)
def test_hosts_resolving_to_non_public_addresses_are_rejected(address, monkeypatch):
    resolve_to(monkeypatch, address)
    with pytest.raises(UnsafeUrlError):
        validate_public_url("https://internal.example/page")


def test_one_private_address_among_public_ones_is_enough_to_reject(monkeypatch):
    resolve_to(monkeypatch, "93.184.216.34", "10.0.0.5")
    with pytest.raises(UnsafeUrlError):
        validate_public_url("https://mixed.example/")


def test_embedded_credentials_are_rejected(monkeypatch):
    resolve_to(monkeypatch, "93.184.216.34")
    with pytest.raises(UnsafeUrlError):
        validate_public_url("https://user:secret@example.com/")


def test_unresolvable_hosts_are_rejected(monkeypatch):
    def fail(*args, **kwargs):
        raise socket.gaierror("no such host")

    monkeypatch.setattr("src.crawling.url_policy.socket.getaddrinfo", fail)
    with pytest.raises(UnsafeUrlError):
        validate_public_url("https://does-not-exist.invalid/")


def test_malformed_port_is_rejected_with_validation_error():
    with pytest.raises(UnsafeUrlError):
        validate_public_url("https://example.com:99999/")


@pytest.mark.parametrize("url", ["https://example.com/docs", "http://example.com/a.pdf", "HTTPS://Example.com/x"])
def test_public_http_urls_are_accepted(url, monkeypatch):
    resolve_to(monkeypatch, "93.184.216.34")
    validate_public_url(url)


# ── Wiring: the policy runs before anything else touches the source ─────────

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


# ── Fetch policy: what the fetch worker may ask a reader API for ────────────

NO_DOMAIN_LISTS = ([], [])


def test_plain_http_is_not_fetched(monkeypatch):
    resolve_to(monkeypatch, "93.184.216.34")
    with pytest.raises(UnsafeUrlError):
        check_fetchable("http://example.com/docs", *NO_DOMAIN_LISTS)


@pytest.mark.parametrize("url", ["https://facebook.com/someone", "https://m.facebook.com/someone"])
def test_denied_domains_and_their_subdomains_are_not_fetched(url, monkeypatch):
    resolve_to(monkeypatch, "93.184.216.34")
    with pytest.raises(UnsafeUrlError):
        check_fetchable(url, [], ["facebook.com"])


def test_a_domain_that_merely_ends_with_a_denied_name_is_allowed(monkeypatch):
    resolve_to(monkeypatch, "93.184.216.34")
    check_fetchable("https://notfacebook.com/page", [], ["facebook.com"])


def test_allowlist_mode_rejects_everything_else(monkeypatch):
    resolve_to(monkeypatch, "93.184.216.34")
    check_fetchable("https://docs.python.org/3/", ["docs.python.org"], [])
    with pytest.raises(UnsafeUrlError):
        check_fetchable("https://example.com/", ["docs.python.org"], [])


def test_the_fetch_policy_still_blocks_private_addresses(monkeypatch):
    resolve_to(monkeypatch, "10.0.0.5")
    with pytest.raises(UnsafeUrlError):
        check_fetchable("https://internal.example/", *NO_DOMAIN_LISTS)


@pytest.mark.asyncio
async def test_http_urls_are_rejected_by_the_api_before_any_job_exists(monkeypatch):
    resolve_to(monkeypatch, "93.184.216.34")
    wired = Wired()
    with pytest.raises(InvalidRequest):
        await wired.submit(url="http://example.com/docs")
    wired.jobs.register_job.assert_not_called()


@pytest.mark.asyncio
async def test_a_tenant_over_its_daily_page_quota_gets_429(monkeypatch):
    from src.config import get_settings

    resolve_to(monkeypatch, "93.184.216.34")
    wired = Wired(pages_fetched_today=get_settings().fetch_daily_page_quota)
    with pytest.raises(QuotaExceeded):
        await wired.submit(url="https://example.com/docs")
    wired.jobs.register_job.assert_not_called()


# ── Wiring: a valid request registers the job durably, then defers it once ──

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


# ── File upload validation ──────────────────────────────────────────────────

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


# ── Job status is tenant-scoped ──────────────────────────────────────────────

def wire_job(app_state, job_tenant, job_exists=True):
    app_state.jobs, app_state.documents = MagicMock(), MagicMock()
    app_state.jobs.get_job.return_value = {
        "job_id": "job-1", "doc_id": "doc-1", "status": "complete",
        "progress_pct": 100, "error": None, "metadata": None,
    } if job_exists else None
    app_state.documents.get_document.return_value = {"doc_id": "doc-1", "tenant_id": job_tenant, "chunk_count": 2}


def test_job_status_requires_authentication(client, app_state):
    wire_job(app_state, "tenant-1")
    assert client.get("/v1/jobs/job-1").status_code == 401


def test_owner_can_read_their_job(client, app_state, tenant_key):
    wire_job(app_state, "tenant-1")
    response = client.get("/v1/jobs/job-1", headers={"X-API-Key": tenant_key("tenant-1")})

    assert response.status_code == 200
    assert response.json()["chunk_count"] == 2


def test_another_tenants_job_is_reported_as_missing(client, app_state, tenant_key):
    wire_job(app_state, "tenant-1")
    response = client.get("/v1/jobs/job-1", headers={"X-API-Key": tenant_key("tenant-2")})
    assert response.status_code == 404


def test_unknown_job_is_missing(client, app_state, tenant_key):
    wire_job(app_state, "tenant-1", job_exists=False)
    assert client.get("/v1/jobs/nope", headers={"X-API-Key": tenant_key("tenant-1")}).status_code == 404


# ── URL policy edge cases ────────────────────────────────────────────────────

@pytest.mark.parametrize("url", ["https://[x", "https://[::1", "https://exa mple.com:abc/"])
def test_malformed_urls_are_unsafe_url_errors_not_crashes(url):
    with pytest.raises(UnsafeUrlError):
        validate_public_url(url)


@pytest.mark.parametrize("address", ["64:ff9b::7f00:1", "64:ff9b::a00:5", "2002:7f00:1::1"])
def test_private_addresses_wrapped_in_nat64_or_6to4_are_rejected(address, monkeypatch):
    resolve_to(monkeypatch, address)
    with pytest.raises(UnsafeUrlError):
        validate_public_url("https://wrapped.example/")


def test_a_host_that_does_not_resolve_in_time_is_rejected(monkeypatch):
    import time

    from src.crawling import url_policy

    monkeypatch.setattr(url_policy, "DNS_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr("src.crawling.url_policy.socket.getaddrinfo", lambda *a, **k: time.sleep(0.5))
    with pytest.raises(UnsafeUrlError, match="in time"):
        validate_public_url("https://slow.example/")


def test_redaction_keeps_the_address_and_drops_the_secret():
    from src.crawling.url_policy import redact_url

    assert redact_url("https://a.example/docs/page?token=SECRET#frag") == "https://a.example/docs/page"


# ── Queue limits and failure handling ────────────────────────────────────────

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
