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


def test_ready_also_requires_the_database_to_answer(client, app_state):
    app_state.system = MagicMock()
    app_state.system.database_ok.return_value = True
    assert client.get("/ready").json() == {"status": "ready"}

    app_state.system.database_ok.return_value = False
    response = client.get("/ready")
    assert response.status_code == 503 and response.json() == {"status": "database_unreachable"}


def test_overload_is_a_503_with_retry_after_not_a_200_answer(client, app_state, tenant_key):
    for name in ("generator", "retrieval", "evaluator"):
        setattr(app_state, name, MagicMock())
    app_state.query_semaphore = asyncio.Semaphore(0)
    key = tenant_key("tenant-1")

    for path in ("/v1/chat", "/v1/chat/stream"):
        response = client.post(path, json={"query": "hi"}, headers={"X-API-Key": key})
        assert response.status_code == 503
        assert response.headers["retry-after"] == "5"
        assert response.json()["code"] == "unavailable"


def test_a_failed_start_does_not_reveal_its_reason_to_callers(client, app_state):
    app_state.init_error = "password authentication failed for user neondb_owner at ep-secret.neon.tech"
    del app_state.ready
    try:
        for path in ("/ready", "/health"):
            assert "neondb_owner" not in client.get(path).text
        response = client.post("/v1/chat", json={"query": "hi"}, headers={"X-API-Key": "nx_whatever"})
        assert response.status_code == 503
        assert "neondb_owner" not in response.text
    finally:
        del app_state.init_error


def wire_exploding_retrieval(app_state):
    class Boom:
        def retrievers(self, index_id=None):
            raise RuntimeError("SELECT * FROM chunks WHERE tenant_id = 'secret-tenant'")

    app_state.generator, app_state.retrieval, app_state.evaluator = MagicMock(), Boom(), MagicMock()
    app_state.documents = MagicMock()
    app_state.documents.document_count.return_value = 1


def test_an_unexpected_error_returns_the_standard_body_not_the_exception(client, app_state, tenant_key):
    wire_exploding_retrieval(app_state)
    response = client.post("/v1/chat", json={"query": "hi"}, headers={"X-API-Key": tenant_key("tenant-1")})

    assert response.status_code == 500
    assert "SELECT" not in response.text and "secret-tenant" not in response.text
    body = response.json()
    assert body["code"] == "internal_error" and body["request_id"] == response.headers["x-request-id"]


def test_every_error_has_the_same_body_and_a_request_id(client, app_state, tenant_key):
    for name in ("generator", "retrieval", "evaluator"):
        setattr(app_state, name, MagicMock())
    cases = [
        client.post("/v1/chat", json={"query": "hi"}),  # 401
        client.post("/v1/chat", json={"query": ""}, headers={"X-API-Key": tenant_key("t")}),  # 422
        client.get("/v1/nothing-here"),  # 404
        client.get("/v1/chat"),  # 405
    ]
    assert [r.status_code for r in cases] == [401, 422, 404, 405]
    assert [r.json()["code"] for r in cases] == ["unauthorized", "validation_failed", "not_found", "method_not_allowed"]
    for response in cases:
        assert set(response.json()) >= {"code", "message", "request_id"}
        assert response.json()["request_id"] == response.headers["x-request-id"]
    assert cases[1].json()["details"][0]["field"] == "body.query"


def test_a_well_formed_request_id_from_the_caller_is_kept_and_a_malformed_one_replaced(client):
    assert client.get("/health", headers={"X-Request-ID": "trace-1234567890"}).headers["x-request-id"] == "trace-1234567890"
    assert client.get("/health", headers={"X-Request-ID": "bad id with spaces"}).headers["x-request-id"] != "bad id with spaces"


def test_workers_report_whether_one_is_running_without_failing_when_none_is(client, app_state, tenant_key):
    app_state.system = MagicMock()
    app_state.system.workers_online.return_value = 0
    response = client.get("/v1/system/workers", headers={"X-API-Key": tenant_key("tenant-1")})

    assert response.status_code == 200 and response.json()["online"] == 0
    assert "Chat is unaffected" in response.json()["detail"]
