import hashlib
import secrets
from datetime import datetime, timezone
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class AuthStore:
    """
    Manages API keys and tenant authentication using the shared registry database.
    """
    def __init__(self, get_conn_func):
        # We take a callable that returns the thread-local sqlite connection
        self._get_conn = get_conn_func

    def validate_api_key(self, api_key: str) -> Optional[str]:
        """Validates an API key and returns the associated tenant_id, or None if invalid."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT tenant_id FROM api_keys WHERE key_hash = ?", (key_hash,))
            row = cursor.fetchone()
            return row["tenant_id"] if row else None

    def create_api_key(self, tenant_id: str) -> str:
        """Generates a new API key for a tenant and stores its hash."""
        raw_key = secrets.token_hex(32)
        api_key = f"sk_live_{raw_key}"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute("INSERT INTO api_keys (key_hash, tenant_id, created_at) VALUES (?, ?, ?)", (key_hash, tenant_id, now))
            conn.commit()
        return api_key
