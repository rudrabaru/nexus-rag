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
        assert "schema" in response.json()["message"]
    finally:
        del app_state.init_error


def test_overload_is_a_503_with_retry_after_not_a_200_answer(client, app_state, tenant_key):
    for name in ("generator", "retriever", "evaluator"):
        setattr(app_state, name, MagicMock())
    app_state.reranker = None
    app_state.query_semaphore = asyncio.Semaphore(0)
    key = tenant_key("tenant-1")

    for path in ("/query", "/query/stream"):
        response = client.post(path, json={"query": "hi"}, headers={"X-API-Key": key})
        assert response.status_code == 503
        assert response.headers["retry-after"] == "5"
