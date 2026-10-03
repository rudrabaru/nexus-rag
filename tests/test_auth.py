import pytest
from sqlalchemy import select
from starlette.requests import Request

from src.api.rate_limit import client_ip, rate_limit_key
from src.stores.api_keys import KEY_PREFIX, AuthStore, hash_api_key
from src.db.schema import api_keys
from tests.conftest import ADMIN_KEY

ADMIN = {"X-Admin-Key": ADMIN_KEY}


def make_request(headers=None, client=("1.2.3.4", 5000)):
    scope = {
        "type": "http",
        "headers": [(k.encode(), v.encode()) for k, v in (headers or {}).items()],
        "client": client,
    }
    return Request(scope)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


# ── AuthStore ────────────────────────────────────────────────────────────────

def test_issued_key_validates_to_its_tenant(auth_engine):
    store = AuthStore(auth_engine)
    key = store.create_api_key("tenant-1")
    assert key.startswith(KEY_PREFIX)
    assert store.validate_api_key(key) == "tenant-1"


def test_only_the_hash_is_stored(auth_engine):
    key = AuthStore(auth_engine).create_api_key("tenant-1")
    with auth_engine.connect() as conn:
        row = conn.execute(select(api_keys)).mappings().one()
    assert row["key_hash"] == hash_api_key(key)
    assert key not in row.values()
    assert row["key_prefix"] == key[: len(KEY_PREFIX) + 6]


def test_keys_are_unique_per_issue(auth_engine):
    store = AuthStore(auth_engine)
    assert store.create_api_key("tenant-1") != store.create_api_key("tenant-1")


@pytest.mark.parametrize("bad", ["", None, "not-a-key", "nx_unknown", "x" * 10_000])
def test_unknown_or_malformed_keys_are_rejected(auth_engine, bad):
    assert AuthStore(auth_engine).validate_api_key(bad) is None


@pytest.mark.parametrize("bad_tenant", ["a_b", "", "has space", "x" * 65, "../etc"])
def test_unsafe_tenant_ids_cannot_be_issued(auth_engine, bad_tenant):
    with pytest.raises(ValueError):
        AuthStore(auth_engine).create_api_key(bad_tenant)


def test_revoked_key_stops_working_immediately_in_the_revoking_process(auth_engine):
    store = AuthStore(auth_engine)
    key = store.create_api_key("tenant-1")
    assert store.validate_api_key(key) == "tenant-1"  # now cached

    assert store.revoke_api_key(key) == 1
    assert store.validate_api_key(key) is None


def test_revocation_reaches_other_processes_within_the_cache_ttl(auth_engine):
    clock = FakeClock()
    issuer, other_process = AuthStore(auth_engine), AuthStore(auth_engine, cache_ttl_seconds=60, clock=clock)
    key = issuer.create_api_key("tenant-1")
    assert other_process.validate_api_key(key) == "tenant-1"

    issuer.revoke_api_key(key)
    clock.now = 59
    assert other_process.validate_api_key(key) == "tenant-1"  # bounded staleness, by design
    clock.now = 61
    assert other_process.validate_api_key(key) is None


def test_revoking_a_tenant_revokes_all_its_keys_only(auth_engine):
    store = AuthStore(auth_engine)
    a1, a2, b = store.create_api_key("tenant-a"), store.create_api_key("tenant-a"), store.create_api_key("tenant-b")

    assert store.revoke_tenant_keys("tenant-a") == 2
    assert store.validate_api_key(a1) is None
    assert store.validate_api_key(a2) is None
    assert store.validate_api_key(b) == "tenant-b"


def test_revoking_twice_reports_nothing_new(auth_engine):
    store = AuthStore(auth_engine)
    key = store.create_api_key("tenant-1")
    assert store.revoke_api_key(key) == 1
    assert store.revoke_api_key(key) == 0


# ── Routes ───────────────────────────────────────────────────────────────────

def test_open_registration_endpoint_is_gone(client):
    assert client.post("/register").status_code == 404


def test_issuing_a_key_requires_the_admin_key(client):
    assert client.post("/admin/keys").status_code == 401
    assert client.post("/admin/keys", headers={"X-Admin-Key": "wrong"}).status_code == 401
    assert client.post("/admin/keys", headers={"X-API-Key": ADMIN_KEY}).status_code == 401


def test_admin_can_issue_a_key_that_then_authenticates(client, app_state):
    response = client.post("/admin/keys", json={"tenant_id": "acme-1"}, headers=ADMIN)

    assert response.status_code == 200
    body = response.json()
    assert body["tenant_id"] == "acme-1"
    assert app_state.auth_store.validate_api_key(body["api_key"]) == "acme-1"


def test_admin_key_generates_a_tenant_id_when_none_given(client):
    response = client.post("/admin/keys", headers=ADMIN)
    assert response.status_code == 200
    assert len(response.json()["tenant_id"]) == 36


def test_admin_key_rejects_an_unsafe_tenant_id(client):
    response = client.post("/admin/keys", json={"tenant_id": "a_b"}, headers=ADMIN)
    assert response.status_code == 422


def test_revoked_key_gets_401(client, tenant_key):
    key = tenant_key("tenant-1")
    assert client.get("/logs", headers={"X-API-Key": key}).status_code == 200

    response = client.post("/admin/keys/revoke", json={"api_key": key}, headers=ADMIN)
    assert response.json() == {"revoked": 1}
    assert client.get("/logs", headers={"X-API-Key": key}).status_code == 401


def test_revoke_requires_admin_and_exactly_one_target(client, tenant_key):
    key = tenant_key("tenant-1")
    assert client.post("/admin/keys/revoke", json={"api_key": key}).status_code == 401
    assert client.post("/admin/keys/revoke", json={}, headers=ADMIN).status_code == 422
    both = {"api_key": key, "tenant_id": "tenant-1"}
    assert client.post("/admin/keys/revoke", json=both, headers=ADMIN).status_code == 422


def test_demo_mode_no_longer_bypasses_authentication(client, monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    response = client.get("/logs")
    assert response.status_code == 401


# ── Rate-limit key ───────────────────────────────────────────────────────────

def hops(monkeypatch, count):
    from src.config import get_settings

    monkeypatch.setenv("TRUSTED_PROXY_HOPS", str(count))
    get_settings.cache_clear()


def test_forwarded_headers_are_ignored_by_default():
    assert client_ip(make_request({"x-forwarded-for": "9.9.9.9"})) == "1.2.3.4"


def test_with_one_trusted_proxy_the_last_entry_is_the_client_and_earlier_ones_are_spoofable(monkeypatch):
    hops(monkeypatch, 1)
    # The client sent "6.6.6.6" itself; the proxy appended the address it actually saw.
    assert client_ip(make_request({"x-forwarded-for": "6.6.6.6, 9.9.9.9"})) == "9.9.9.9"


def test_with_two_trusted_proxies_the_client_is_second_from_the_right(monkeypatch):
    hops(monkeypatch, 2)
    assert client_ip(make_request({"x-forwarded-for": "6.6.6.6, 9.9.9.9, 10.0.0.1"})) == "9.9.9.9"


def test_a_header_shorter_than_the_trusted_chain_or_not_an_address_falls_back_to_the_peer(monkeypatch):
    hops(monkeypatch, 2)
    assert client_ip(make_request({"x-forwarded-for": "9.9.9.9"})) == "1.2.3.4"
    hops(monkeypatch, 1)
    assert client_ip(make_request({"x-forwarded-for": "not-an-ip"})) == "1.2.3.4"


def test_authenticated_callers_are_limited_per_tenant_not_per_ip():
    request = make_request({"x-forwarded-for": "9.9.9.9"})
    request.state.tenant_id = "tenant-1"
    assert rate_limit_key(request) == "tenant:tenant-1"


def test_anonymous_callers_are_limited_per_ip():
    assert rate_limit_key(make_request()) == "ip:1.2.3.4"


# ── Credentials: one path, always 401, never fail-open ──────────────────────

def test_a_missing_key_is_401_on_every_protected_route_not_a_200_sentence(client, app_state):
    from unittest.mock import MagicMock

    for name in ("generator", "retrieval", "evaluator", "documents", "jobs", "ingestion", "rewriter", "job_queue"):
        setattr(app_state, name, MagicMock())
    for method, path in [("post", "/query"), ("post", "/query/stream"), ("post", "/query/compare"),
                         ("get", "/logs"), ("get", "/ingest/x"), ("post", "/ingest"),
                         ("get", "/documents"), ("get", "/documents/stats"), ("delete", "/documents/x")]:
        kwargs = {"json": {"query": "hi"}} if path.startswith("/query") else {}
        assert getattr(client, method)(path, **kwargs).status_code == 401, path


def test_a_non_ascii_admin_header_is_401_not_a_server_error(client):
    response = client.post("/admin/keys", headers={"X-Admin-Key": "caf\u00e9".encode("latin-1")})
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
        assert client.post("/admin/keys", headers={"X-Admin-Key": "wrong"}).status_code == 401
    assert client.post("/admin/keys", headers={"X-Admin-Key": "wrong"}).status_code == 429
    assert client.post("/admin/keys", headers=ADMIN).status_code == 429  # even the right key waits out the window


def test_a_foreign_document_cannot_be_distinguished_from_a_missing_one(client, app_state, tenant_key):
    from unittest.mock import MagicMock

    app_state.documents = MagicMock()
    app_state.documents.get_document.side_effect = lambda doc_id: (
        {"doc_id": doc_id, "tenant_id": "tenant-1"} if doc_id == "mine" else None
    )
    headers = {"X-API-Key": tenant_key("tenant-2")}
    assert client.delete("/documents/mine", headers=headers).status_code == 404
    assert client.delete("/documents/nonexistent", headers=headers).status_code == 404
    app_state.documents.delete_document.assert_not_called()


def test_the_admin_listing_is_explicit_and_a_missing_tenant_is_an_error(auth_engine):
    from src.stores.documents import DocumentStore

    registry = DocumentStore(auth_engine)
    for tenant in (None, ""):
        with pytest.raises(ValueError):
            registry.list_documents(tenant)
