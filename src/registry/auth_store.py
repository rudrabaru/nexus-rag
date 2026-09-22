import hashlib
import hmac
import logging
import re
import secrets
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Tenant IDs are embedded in the key and split on "_", so they may not contain it.
TENANT_ID_PATTERN = re.compile(r"^[A-Za-z0-9-]{1,64}$")


class AuthStore:
    """
    Issues and validates tenant API keys.

    Keys are stateless HMAC-signed strings (sk_live_<tenant>_<signature>) so they survive
    the ephemeral registry disk. Because validation never consults storage, individual keys
    cannot be revoked yet; that arrives with the durable Postgres store.
    """

    def __init__(self, get_conn_func, signing_secret: str):
        if not signing_secret:
            raise ValueError("A signing secret is required to issue or validate API keys.")
        self._get_conn = get_conn_func
        self._secret = signing_secret

    def _generate_signature(self, tenant_id: str) -> str:
        return hmac.new(self._secret.encode(), tenant_id.encode(), hashlib.sha256).hexdigest()

    def validate_api_key(self, api_key: str) -> Optional[str]:
        """Returns the tenant_id for a valid key, or None."""
        if not api_key.startswith("sk_live_"):
            return None

        parts = api_key.split("_")
        if len(parts) != 4:
            return None

        tenant_id, signature = parts[2], parts[3]
        if not TENANT_ID_PATTERN.match(tenant_id):
            return None

        if secrets.compare_digest(signature, self._generate_signature(tenant_id)):
            return tenant_id
        return None

    def create_api_key(self, tenant_id: str) -> str:
        """Issues a key for a tenant. Only reachable through the admin-authenticated route."""
        if not TENANT_ID_PATTERN.match(tenant_id):
            raise ValueError("tenant_id must be 1-64 characters of letters, digits or hyphens.")

        api_key = f"sk_live_{tenant_id}_{self._generate_signature(tenant_id)}"

        # Recorded only for token accounting and creation time; authentication never reads it.
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO api_keys (key_hash, tenant_id, created_at) VALUES (?, ?, ?)",
                    (key_hash, tenant_id, now),
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to record API key creation, but the key is valid: {e}")

        return api_key
