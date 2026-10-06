"""AuthStore: issue, validate, revoke."""
import pytest
from sqlalchemy import select
from src.stores.api_keys import KEY_PREFIX, AuthStore, hash_api_key
from src.db.schema import api_keys


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


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
