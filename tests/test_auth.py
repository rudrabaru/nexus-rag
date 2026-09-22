from unittest.mock import MagicMock

import pytest
from starlette.requests import Request

from src.api.auth import get_rate_limit_key, get_real_ip
from src.registry.auth_store import AuthStore
from tests.conftest import ADMIN_KEY, SIGNING_SECRET


def make_request(headers=None, client=("1.2.3.4", 5000)):
    scope = {
        "type": "http",
        "headers": [(k.encode(), v.encode()) for k, v in (headers or {}).items()],
        "client": client,
    }
    return Request(scope)


# ── AuthStore ────────────────────────────────────────────────────────────────

def test_issued_key_validates_to_its_tenant():
    store = AuthStore(MagicMock(), SIGNING_SECRET)
    assert store.validate_api_key(store.create_api_key("tenant-1")) == "tenant-1"


def test_key_signed_with_another_secret_is_rejected():
    key = AuthStore(MagicMock(), SIGNING_SECRET).create_api_key("tenant-1")
    assert AuthStore(MagicMock(), "a-different-secret-0123456789").validate_api_key(key) is None


def test_tampered_tenant_id_is_rejected():
    store = AuthStore(MagicMock(), SIGNING_SECRET)
    key = store.create_api_key("tenant-1")
    assert store.validate_api_key(key.replace("tenant-1", "tenant-2")) is None


@pytest.mark.parametrize("bad", ["", "sk_live_", "sk_live_a_b_c", "not-a-key", "sk_live_tenant_"])
def test_malformed_keys_are_rejected(bad):
    assert AuthStore(MagicMock(), SIGNING_SECRET).validate_api_key(bad) is None


@pytest.mark.parametrize("bad_tenant", ["a_b", "", "has space", "x" * 65, "../etc"])
def test_tenant_ids_that_break_key_parsing_cannot_be_issued(bad_tenant):
    with pytest.raises(ValueError):
        AuthStore(MagicMock(), SIGNING_SECRET).create_api_key(bad_tenant)


def test_store_refuses_to_run_without_a_secret():
    with pytest.raises(ValueError):
        AuthStore(MagicMock(), "")


# ── Routes ───────────────────────────────────────────────────────────────────

def test_open_registration_endpoint_is_gone(client):
    assert client.post("/register").status_code == 404


def test_issuing_a_key_requires_the_admin_key(client):
    assert client.post("/admin/keys").status_code == 401
    assert client.post("/admin/keys", headers={"RAG-API-KEY": "wrong"}).status_code == 401
    assert client.post("/admin/keys", headers={"X-API-Key": ADMIN_KEY}).status_code == 401


def test_admin_can_issue_a_key_that_then_authenticates(client, app_state):
    response = client.post("/admin/keys", json={"tenant_id": "acme-1"}, headers={"RAG-API-KEY": ADMIN_KEY})

    assert response.status_code == 200
    body = response.json()
    assert body["tenant_id"] == "acme-1"
    assert app_state.auth_store.validate_api_key(body["api_key"]) == "acme-1"


def test_admin_key_generates_a_tenant_id_when_none_given(client):
    response = client.post("/admin/keys", headers={"RAG-API-KEY": ADMIN_KEY})
    assert response.status_code == 200
    assert len(response.json()["tenant_id"]) == 36


def test_admin_key_rejects_an_unparseable_tenant_id(client):
    response = client.post("/admin/keys", json={"tenant_id": "a_b"}, headers={"RAG-API-KEY": ADMIN_KEY})
    assert response.status_code == 422


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


def test_authenticated_callers_are_limited_per_tenant_not_per_ip():
    request = make_request({"x-forwarded-for": "9.9.9.9"})
    request.state.tenant_id = "tenant-1"
    assert get_rate_limit_key(request) == "tenant:tenant-1"


def test_anonymous_callers_are_limited_per_ip():
    assert get_rate_limit_key(make_request()) == "ip:1.2.3.4"
