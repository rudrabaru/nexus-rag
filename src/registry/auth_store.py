"""
Tenant API keys: random tokens, stored only as SHA-256 hashes, individually revocable.

Why SHA-256 rather than bcrypt/argon2: keys are 256-bit random tokens, so offline guessing is
infeasible whatever the hash speed. Slow hashes exist to protect low-entropy passwords. This
is the usual practice for API tokens.

Keys issued before this store (sk_live_<tenant>_<hmac>) keep working when their hash is
imported into api_keys by `python -m scripts.migrate_legacy`; they are then revocable too.

Validation results are cached per process for CACHE_TTL_SECONDS, because every request
validates its key twice (rate-limit key and auth dependency) and each miss is a database
round trip. A revocation takes effect immediately in the process that performs it and within
the TTL in any other process.
"""
import hashlib
import re
import secrets
import threading
import time
from collections import OrderedDict
from typing import Callable, Optional, Tuple

from sqlalchemy import select, update
from sqlalchemy.engine import Engine

from src.registry.rows import utcnow
from src.registry.schema import api_keys

KEY_PREFIX = "nx_"
KEY_BYTES = 32
DISPLAY_PREFIX_LENGTH = len(KEY_PREFIX) + 6
MAX_KEY_LENGTH = 256
CACHE_TTL_SECONDS = 60.0
CACHE_MAX_ENTRIES = 10_000

# Tenant IDs appear in logs, URLs and filters, so they are restricted to a safe charset.
TENANT_ID_PATTERN = re.compile(r"^[A-Za-z0-9-]{1,64}$")


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


class AuthStore:
    def __init__(
        self, engine: Engine, cache_ttl_seconds: float = CACHE_TTL_SECONDS, clock: Callable[[], float] = time.monotonic
    ):
        self._engine = engine
        self._ttl = cache_ttl_seconds
        self._clock = clock
        self._cache: "OrderedDict[str, Tuple[Optional[str], float]]" = OrderedDict()
        self._lock = threading.Lock()

    def create_api_key(self, tenant_id: str) -> str:
        """Issues a new key for a tenant. The plaintext is returned once and never stored."""
        if not TENANT_ID_PATTERN.match(tenant_id):
            raise ValueError("tenant_id must be 1-64 characters of letters, digits or hyphens.")

        api_key = f"{KEY_PREFIX}{secrets.token_urlsafe(KEY_BYTES)}"
        with self._engine.begin() as conn:
            conn.execute(
                api_keys.insert().values(
                    key_hash=hash_api_key(api_key),
                    tenant_id=tenant_id,
                    key_prefix=api_key[:DISPLAY_PREFIX_LENGTH],
                    created_at=utcnow(),
                )
            )
        return api_key

    def validate_api_key(self, api_key: Optional[str]) -> Optional[str]:
        """Returns the tenant_id for a valid, unrevoked key, or None."""
        if not api_key or len(api_key) > MAX_KEY_LENGTH:
            return None

        key_hash = hash_api_key(api_key)
        cached = self._cache_get(key_hash)
        if cached is not None:
            return cached[0]

        stmt = select(api_keys.c.tenant_id).where(api_keys.c.key_hash == key_hash, api_keys.c.revoked_at.is_(None))
        with self._engine.connect() as conn:
            tenant_id = conn.execute(stmt).scalar_one_or_none()
        self._cache_put(key_hash, tenant_id)
        return tenant_id

    def revoke_api_key(self, api_key: str) -> int:
        return self._revoke(api_keys.c.key_hash == hash_api_key(api_key))

    def revoke_tenant_keys(self, tenant_id: str) -> int:
        return self._revoke(api_keys.c.tenant_id == tenant_id)

    def _revoke(self, condition) -> int:
        stmt = update(api_keys).where(condition, api_keys.c.revoked_at.is_(None)).values(revoked_at=utcnow())
        with self._engine.begin() as conn:
            revoked = conn.execute(stmt).rowcount
        with self._lock:
            self._cache.clear()
        return revoked

    # A bounded LRU with expiry. Negative results are cached too, so a flood of invalid keys
    # cannot turn into a flood of database queries.

    def _cache_get(self, key_hash: str) -> Optional[Tuple[Optional[str], float]]:
        with self._lock:
            entry = self._cache.get(key_hash)
            if entry is None:
                return None
            if entry[1] <= self._clock():
                del self._cache[key_hash]
                return None
            self._cache.move_to_end(key_hash)
            return entry

    def _cache_put(self, key_hash: str, tenant_id: Optional[str]) -> None:
        with self._lock:
            self._cache[key_hash] = (tenant_id, self._clock() + self._ttl)
            self._cache.move_to_end(key_hash)
            while len(self._cache) > CACHE_MAX_ENTRIES:
                self._cache.popitem(last=False)
