import pytest
from sqlalchemy import select
from starlette.requests import Request

from src.api.auth import get_rate_limit_key, get_real_ip
from src.registry.auth_store import KEY_PREFIX, AuthStore, hash_api_key
from src.registry.rows import utcnow
from src.registry.schema import api_keys
from tests.conftest import ADMIN_KEY

ADMIN = {"RAG-API-KEY": ADMIN_KEY}


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


def test_legacy_key_is_accepted_once_its_hash_is_imported(auth_engine):
    """Keys issued by the HMAC scheme keep working after migration, and become revocable."""
    legacy_key = "sk_live_tenant-1_" + "ab" * 32
    with auth_engine.begin() as conn:
        conn.execute(api_keys.insert().values(key_hash=hash_api_key(legacy_key), tenant_id="tenant-1", created_at=utcnow()))

    store = AuthStore(auth_engine)
    assert store.validate_api_key(legacy_key) == "tenant-1"
    store.revoke_api_key(legacy_key)
    assert store.validate_api_key(legacy_key) is None


# ── Routes ───────────────────────────────────────────────────────────────────

def test_open_registration_endpoint_is_gone(client):
    assert client.post("/register").status_code == 404


def test_issuing_a_key_requires_the_admin_key(client):
    assert client.post("/admin/keys").status_code == 401
    assert client.post("/admin/keys", headers={"RAG-API-KEY": "wrong"}).status_code == 401
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

def test_forwarded_headers_are_ignored_by_default():
    request = make_request({"x-forwarded-for": "9.9.9.9"})
    assert get_real_ip(request) == "1.2.3.4"


def test_forwarded_headers_are_honoured_only_when_proxies_are_trusted(monkeypatch):
    from src.config import get_settings

    monkeypatch.setenv("TRUST_PROXIES", "true")
    get_settings.cache_clear()
    assert get_real_ip(make_request({"x-forwarded-for": "9.9.9.9, 10.0.0.1"})) == "9.9.9.9"


def test_authenticated_callers_are_limited_per_tenant_not_per_ip(auth_engine):
    from fastapi import FastAPI

    app = FastAPI()
    app.state.auth_store = AuthStore(auth_engine)
    key = app.state.auth_store.create_api_key("tenant-1")

    request = make_request({"x-api-key": key, "x-forwarded-for": "9.9.9.9"})
    request.scope["app"] = app
    assert get_rate_limit_key(request) == "tenant:tenant-1"


def test_anonymous_callers_are_limited_per_ip():
    assert get_rate_limit_key(make_request()) == "ip:1.2.3.4"
