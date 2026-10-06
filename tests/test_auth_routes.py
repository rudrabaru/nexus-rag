"""Credentials on the routes: one path, always 401, never fail-open."""
from unittest.mock import MagicMock
import pytest
from src.stores.api_keys import AuthStore, hash_api_key
from tests.conftest import ADMIN_KEY


ADMIN = {"X-Admin-Key": ADMIN_KEY}


def test_open_registration_endpoint_is_gone(client):
    assert client.post("/register").status_code == 404


def test_issuing_a_key_requires_the_admin_key(client):
    assert client.post("/v1/admin/keys").status_code == 401
    assert client.post("/v1/admin/keys", headers={"X-Admin-Key": "wrong"}).status_code == 401
    assert client.post("/v1/admin/keys", headers={"X-API-Key": ADMIN_KEY}).status_code == 401


def test_admin_can_issue_a_key_that_then_authenticates(client, app_state):
    response = client.post("/v1/admin/keys", json={"tenant_id": "acme-1"}, headers=ADMIN)

    assert response.status_code == 200
    body = response.json()
    assert body["tenant_id"] == "acme-1"
    assert app_state.auth_store.validate_api_key(body["api_key"]) == "acme-1"


def test_admin_key_generates_a_tenant_id_when_none_given(client):
    response = client.post("/v1/admin/keys", headers=ADMIN)
    assert response.status_code == 200
    assert len(response.json()["tenant_id"]) == 36


def test_admin_key_rejects_an_unsafe_tenant_id(client):
    response = client.post("/v1/admin/keys", json={"tenant_id": "a_b"}, headers=ADMIN)
    assert response.status_code == 422


def test_revoked_key_gets_401(client, app_state, tenant_key):
    app_state.query_log = MagicMock()
    app_state.query_log.summary.return_value = {"total_queries": 0, "total_cost_usd": 0.0, "avg_cost_per_query_usd": 0.0, "avg_latency_ms": 0.0}
    app_state.query_log.recent_queries.return_value = []
    key = tenant_key("tenant-1")
    assert client.get("/v1/usage", headers={"X-API-Key": key}).status_code == 200

    response = client.post("/v1/admin/keys/revoke", json={"api_key": key}, headers=ADMIN)
    assert response.json() == {"revoked": 1}
    assert client.get("/v1/usage", headers={"X-API-Key": key}).status_code == 401


def test_revoke_requires_admin_and_exactly_one_target(client, tenant_key):
    key = tenant_key("tenant-1")
    assert client.post("/v1/admin/keys/revoke", json={"api_key": key}).status_code == 401
    assert client.post("/v1/admin/keys/revoke", json={}, headers=ADMIN).status_code == 422
    both = {"api_key": key, "tenant_id": "tenant-1"}
    assert client.post("/v1/admin/keys/revoke", json=both, headers=ADMIN).status_code == 422


def test_demo_mode_no_longer_bypasses_authentication(client, monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    response = client.get("/v1/usage")
    assert response.status_code == 401


def test_a_missing_key_is_401_on_every_protected_route_not_a_200_sentence(client, app_state):
    from unittest.mock import MagicMock

    for name in ("generator", "retrieval", "evaluator", "documents", "jobs", "ingestion", "rewriter", "job_queue"):
        setattr(app_state, name, MagicMock())
    for method, path in [("post", "/v1/chat"), ("post", "/v1/chat/stream"), ("post", "/v1/retrieval/compare"),
                         ("get", "/v1/usage"), ("get", "/v1/jobs/x"), ("post", "/v1/documents"),
                         ("get", "/v1/documents"), ("get", "/v1/workspace/stats"), ("get", "/v1/workspace/settings"),
                         ("put", "/v1/workspace/settings"), ("delete", "/v1/documents/x"), ("get", "/v1/system/workers")]:
        kwargs = {"json": {"query": "hi"}} if path in ("/v1/chat", "/v1/chat/stream", "/v1/retrieval/compare") else {}
        kwargs = {"json": {}} if path == "/v1/workspace/settings" and method == "put" else kwargs
        assert getattr(client, method)(path, **kwargs).status_code == 401, path


def test_a_non_ascii_admin_header_is_401_not_a_server_error(client):
    response = client.post("/v1/admin/keys", headers={"X-Admin-Key": "caf\u00e9".encode("latin-1")})
    assert response.status_code == 401


def test_keys_that_could_not_have_been_issued_never_reach_the_database(auth_engine):
    from unittest.mock import MagicMock

    engine = MagicMock()
    store = AuthStore(engine)
    for bad in ["x" * 46, "nx_short", "nx_" + "a" * 100, "sk_live_" + "a" * 38]:
        assert store.validate_api_key(bad) is None
    engine.connect.assert_not_called()


def test_a_tenant_id_with_a_trailing_newline_is_rejected(auth_engine):
    with pytest.raises(ValueError):
        AuthStore(auth_engine).create_api_key("tenant-1\n")


def test_a_lookup_that_began_before_a_revocation_cannot_cache_a_stale_answer(auth_engine):
    store = AuthStore(auth_engine)
    key = store.create_api_key("tenant-1")
    generation = store._generation
    store.revoke_api_key(key)  # a concurrent revocation lands while another request was reading the row
    store._cache_put(hash_api_key(key), "tenant-1", generation)
    assert store.validate_api_key(key) is None


def test_repeated_wrong_keys_are_throttled_per_client(client):
    for _ in range(10):
        assert client.post("/v1/admin/keys", headers={"X-Admin-Key": "wrong"}).status_code == 401
    assert client.post("/v1/admin/keys", headers={"X-Admin-Key": "wrong"}).status_code == 429
    assert client.post("/v1/admin/keys", headers=ADMIN).status_code == 429  # even the right key waits out the window


def test_a_foreign_document_cannot_be_distinguished_from_a_missing_one(client, app_state, tenant_key):
    from unittest.mock import MagicMock

    app_state.documents = MagicMock()
    app_state.documents.get_document.side_effect = lambda doc_id: (
        {"doc_id": doc_id, "tenant_id": "tenant-1"} if doc_id == "mine" else None
    )
    headers = {"X-API-Key": tenant_key("tenant-2")}
    assert client.delete("/v1/documents/mine", headers=headers).status_code == 404
    assert client.delete("/v1/documents/nonexistent", headers=headers).status_code == 404
    app_state.documents.delete_document.assert_not_called()


def test_the_admin_listing_is_explicit_and_a_missing_tenant_is_an_error(auth_engine):
    from src.stores.documents import DocumentStore

    registry = DocumentStore(auth_engine)
    for tenant in (None, ""):
        with pytest.raises(ValueError):
            registry.list_documents(tenant)
