import socket
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from src.ingestion.url_policy import UnsafeUrlError, validate_public_url
from src.jobs.contract import INGEST_QUEUE, INGEST_TASK
from src.services.ingestion_service import (
    MAX_PENDING_UPLOAD_BYTES,
    MAX_UPLOAD_BYTES,
    prepare_and_queue_ingestion,
)


def resolve_to(monkeypatch, *addresses):
    infos = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, 443)) for a in addresses]
    monkeypatch.setattr("src.ingestion.url_policy.socket.getaddrinfo", lambda *args, **kwargs: infos)


class FakeUpload:
    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self._content = content

    async def read(self) -> bytes:
        return self._content


def make_registry(quota=0, pending_bytes=0, existing_by_hash=None):
    """
    registry.get_tenant_quota etc. are called through asyncio.to_thread, which expects a
    plain sync callable — MagicMock's auto-generated attributes already are one.
    """
    registry = MagicMock()
    registry.get_tenant_quota.return_value = quota
    registry.pending_upload_bytes.return_value = pending_bytes
    registry.get_document_by_hash.return_value = existing_by_hash
    return registry


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
        "src.ingestion.url_policy.socket.getaddrinfo",
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

    monkeypatch.setattr("src.ingestion.url_policy.socket.getaddrinfo", fail)
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
    registry = make_registry()
    with pytest.raises(HTTPException) as exc:
        await prepare_and_queue_ingestion(MagicMock(), registry, "tenant-1", "/app/logs.txt", None, False, False)

    assert exc.value.status_code == 400
    registry.get_tenant_quota.assert_not_called()


@pytest.mark.asyncio
async def test_ingestion_rejects_malformed_port_with_http_400():
    registry = make_registry()
    with pytest.raises(HTTPException) as exc:
        await prepare_and_queue_ingestion(MagicMock(), registry, "tenant-1", "https://example.com:99999/", None, False, False)

    assert exc.value.status_code == 400


# ── Wiring: a valid request registers the job durably, then defers it once ──

@pytest.mark.asyncio
async def test_a_valid_url_registers_the_job_before_deferring_it(monkeypatch):
    resolve_to(monkeypatch, "93.184.216.34")
    registry = make_registry()
    job_queue = MagicMock()

    response = await prepare_and_queue_ingestion(job_queue, registry, "tenant-1", "https://example.com/docs", None, False, False)

    assert response["status"] == "queued"
    registry.register_job.assert_called_once()
    job_queue.configure_task.assert_called_once_with(INGEST_TASK, queue=INGEST_QUEUE, lock=registry.register_job.call_args[0][1])
    job_queue.configure_task.return_value.defer.assert_called_once()
    deferred = job_queue.configure_task.return_value.defer.call_args.kwargs
    assert deferred["job_id"] == response["job_id"]
    assert deferred["url"] == "https://example.com/docs"
    assert deferred["filename"] is None


# ── File upload validation ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_unsupported_file_extension_is_rejected():
    registry = make_registry()
    with pytest.raises(HTTPException) as exc:
        await prepare_and_queue_ingestion(MagicMock(), registry, "tenant-1", None, FakeUpload("a.exe", b"x"), False, False)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_oversized_upload_is_rejected():
    registry = make_registry()
    upload = FakeUpload("a.txt", b"x" * (MAX_UPLOAD_BYTES + 1))
    with pytest.raises(HTTPException) as exc:
        await prepare_and_queue_ingestion(MagicMock(), registry, "tenant-1", None, upload, False, False)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_too_many_pending_uploads_are_rejected():
    """Bounds Neon's 0.5 GB storage against an offline or backlogged worker."""
    registry = make_registry(pending_bytes=MAX_PENDING_UPLOAD_BYTES)
    with pytest.raises(HTTPException) as exc:
        await prepare_and_queue_ingestion(MagicMock(), registry, "tenant-1", None, FakeUpload("a.txt", b"x"), False, False)
    assert exc.value.status_code == 429
    registry.register_job.assert_not_called()


@pytest.mark.asyncio
async def test_duplicate_upload_short_circuits_without_deferring_a_job():
    registry = make_registry(existing_by_hash={"doc_id": "doc-1", "status": "complete", "source": "u", "format": "txt"})
    job_queue = MagicMock()

    response = await prepare_and_queue_ingestion(job_queue, registry, "tenant-1", None, FakeUpload("a.txt", b"same"), False, False)

    assert response["status"] == "complete"
    job_queue.configure_task.assert_not_called()


@pytest.mark.asyncio
async def test_a_valid_upload_stores_its_bytes_with_the_job_not_on_local_disk():
    registry = make_registry()
    job_queue = MagicMock()

    response = await prepare_and_queue_ingestion(job_queue, registry, "tenant-1", None, FakeUpload("a.txt", b"hello"), False, False)

    assert response["status"] == "queued"
    assert registry.register_job.call_args.kwargs["upload"] == ("a.txt", b"hello")
    deferred = job_queue.configure_task.return_value.defer.call_args.kwargs
    assert deferred["filename"] == "a.txt" and deferred["url"] is None


# ── Job status is tenant-scoped ──────────────────────────────────────────────

def make_job_registry(job_tenant):
    registry = MagicMock()
    registry.get_job.return_value = {
        "job_id": "job-1", "doc_id": "doc-1", "status": "complete",
        "progress_pct": 100, "error": None, "metadata": None,
    }
    registry.get_document.return_value = {"doc_id": "doc-1", "tenant_id": job_tenant, "chunk_count": 2}
    return registry


def test_job_status_requires_authentication(client, app_state):
    app_state.registry = make_job_registry("tenant-1")
    assert client.get("/ingest/job-1").status_code == 401


def test_owner_can_read_their_job(client, app_state, tenant_key):
    app_state.registry = make_job_registry("tenant-1")
    response = client.get("/ingest/job-1", headers={"X-API-Key": tenant_key("tenant-1")})

    assert response.status_code == 200
    assert response.json()["chunk_count"] == 2


def test_another_tenants_job_is_reported_as_missing(client, app_state, tenant_key):
    app_state.registry = make_job_registry("tenant-1")
    response = client.get("/ingest/job-1", headers={"X-API-Key": tenant_key("tenant-2")})
    assert response.status_code == 404


def test_unknown_job_is_missing(client, app_state, tenant_key):
    registry = make_job_registry("tenant-1")
    registry.get_job.return_value = None
    app_state.registry = registry
    assert client.get("/ingest/nope", headers={"X-API-Key": tenant_key("tenant-1")}).status_code == 404
