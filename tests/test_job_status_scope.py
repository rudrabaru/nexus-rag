"""Job status is tenant-scoped."""
from unittest.mock import MagicMock


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
