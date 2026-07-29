import hashlib
import hmac
import secrets
import os
from datetime import datetime, timezone
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class AuthStore:
    """
    Manages API keys and tenant authentication.
    Uses stateless HMAC-signed API keys to survive ephemeral environments.
    """
    def __init__(self, get_conn_func):
        # We take a callable that returns the thread-local sqlite connection
        self._get_conn = get_conn_func
        self._secret = os.environ.get("RAG_API_KEY", "")

    def _generate_signature(self, tenant_id: str) -> str:
        return hmac.new(self._secret.encode(), tenant_id.encode(), hashlib.sha256).hexdigest()

    def validate_api_key(self, api_key: str) -> Optional[str]:
        """Validates a stateless API key and returns the associated tenant_id, or None if invalid."""
        if not api_key.startswith("sk_live_"):
            return None
        
        parts = api_key.split("_")
        if len(parts) != 4:
            return None
            
        tenant_id = parts[2]
        signature = parts[3]
        
        expected_signature = self._generate_signature(tenant_id)
        if secrets.compare_digest(signature, expected_signature):
            return tenant_id
            
        return None

    def create_api_key(self, tenant_id: str) -> str:
        """Generates a new stateless API key for a tenant."""
        signature = self._generate_signature(tenant_id)
        api_key = f"sk_live_{tenant_id}_{signature}"
        
        # We still insert into api_keys to track embedding_tokens and creation time, 
        # but authentication no longer depends on this SQLite row existing.
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._get_conn() as conn:
                conn.execute("INSERT OR IGNORE INTO api_keys (key_hash, tenant_id, created_at) VALUES (?, ?, ?)", (key_hash, tenant_id, now))
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to record API key creation in SQLite, but key is valid: {e}")
            
        return api_key
