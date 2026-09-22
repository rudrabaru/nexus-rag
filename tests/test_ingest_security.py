import socket
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from src.ingestion.url_policy import UnsafeUrlError, validate_public_url
from src.services.ingestion_service import prepare_ingestion


def resolve_to(monkeypatch, *addresses):
    infos = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, 443)) for a in addresses]
    monkeypatch.setattr("src.ingestion.url_policy.socket.getaddrinfo", lambda *args, **kwargs: infos)


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


@pytest.mark.parametrize("url", ["https://example.com/docs", "http://example.com/a.pdf", "HTTPS://Example.com/x"])
def test_public_http_urls_are_accepted(url, monkeypatch):
    resolve_to(monkeypatch, "93.184.216.34")
    validate_public_url(url)


# ── Wiring: the policy runs before anything else touches the source ─────────

@pytest.mark.asyncio
async def test_prepare_ingestion_rejects_a_local_path_in_the_url_field():
    registry = MagicMock()
    with pytest.raises(HTTPException) as exc:
        await prepare_ingestion("/app/logs.txt", None, "tenant-1", registry)

    assert exc.value.status_code == 400
    registry.get_tenant_quota.assert_not_called()


# ── Job status is tenant-scoped ──────────────────────────────────────────────

def make_registry(job_tenant):
    registry = MagicMock()
    registry.get_job.return_value = {
        "job_id": "job-1", "doc_id": "doc-1", "status": "complete",
        "progress_pct": 100, "error": None, "metadata": None,
    }
    registry.get_document.return_value = {"doc_id": "doc-1", "tenant_id": job_tenant, "chunk_ids": ["a", "b"]}
    return registry


def test_job_status_requires_authentication(client, app_state):
    app_state.registry = make_registry("tenant-1")
    assert client.get("/ingest/job-1").status_code == 401


def test_owner_can_read_their_job(client, app_state, tenant_key):
    app_state.registry = make_registry("tenant-1")
    response = client.get("/ingest/job-1", headers={"X-API-Key": tenant_key("tenant-1")})

    assert response.status_code == 200
    assert response.json()["chunk_count"] == 2


def test_another_tenants_job_is_reported_as_missing(client, app_state, tenant_key):
    app_state.registry = make_registry("tenant-1")
    response = client.get("/ingest/job-1", headers={"X-API-Key": tenant_key("tenant-2")})
    assert response.status_code == 404


def test_unknown_job_is_missing(client, app_state, tenant_key):
    registry = make_registry("tenant-1")
    registry.get_job.return_value = None
    app_state.registry = registry
    assert client.get("/ingest/nope", headers={"X-API-Key": tenant_key("tenant-1")}).status_code == 404
