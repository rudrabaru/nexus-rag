import asyncio
from unittest.mock import MagicMock


def test_health_is_ok_when_initialised(client):
    assert client.get("/health").status_code == 200


def test_health_fails_when_initialisation_failed(client, app_state):
    """A dead instance must fail its liveness probe, or the platform keeps routing traffic to it."""
    app_state.init_error = "Database schema is at revision None"
    try:
        response = client.get("/health")
        assert response.status_code == 503
        assert "schema" not in response.text  # the reason belongs in the server log
    finally:
        del app_state.init_error


def test_ready_is_a_503_when_initialisation_failed(client, app_state):
    """Regression: /ready answered HTTP 200 with an error body, so a probe counted it ready."""
    app_state.init_error = "boom"
    try:
        assert client.get("/ready").status_code == 503
    finally:
        del app_state.init_error


def test_overload_is_a_503_with_retry_after_not_a_200_answer(client, app_state, tenant_key):
    for name in ("generator", "retrieval", "evaluator"):
        setattr(app_state, name, MagicMock())
    app_state.query_semaphore = asyncio.Semaphore(0)
    key = tenant_key("tenant-1")

    for path in ("/query", "/query/stream"):
        response = client.post(path, json={"query": "hi"}, headers={"X-API-Key": key})
        assert response.status_code == 503
        assert response.headers["retry-after"] == "5"


def test_a_failed_start_does_not_reveal_its_reason_to_callers(client, app_state, tenant_key):
    app_state.init_error = "password authentication failed for user neondb_owner at ep-secret.neon.tech"
    del app_state.ready
    try:
        for path in ("/ready", "/health"):
            assert "neondb_owner" not in client.get(path).text
        response = client.post("/query", json={"query": "hi"}, headers={"X-API-Key": "nx_whatever"})
        assert response.status_code == 503
        assert "neondb_owner" not in response.text
    finally:
        del app_state.init_error


def test_an_unexpected_error_returns_a_reference_not_the_exception(app, app_state, tenant_key):
    from fastapi.testclient import TestClient

    class Boom:
        def retrievers(self, index_id=None):
            raise RuntimeError("SELECT * FROM chunks WHERE tenant_id = 'secret-tenant'")

    app_state.generator, app_state.retrieval, app_state.evaluator = MagicMock(), Boom(), MagicMock()
    app_state.documents = MagicMock()
    app_state.documents.document_count.return_value = 1
    response = TestClient(app, raise_server_exceptions=False).post(
        "/query", json={"query": "hi"}, headers={"X-API-Key": tenant_key("tenant-1")}
    )

    assert response.status_code == 500
    assert "SELECT" not in response.text and "secret-tenant" not in response.text
    assert "reference" in response.json()["detail"]
