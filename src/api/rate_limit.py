"""
Rate limits and who they are keyed by.

One Limiter serves every route. Callers are limited per workspace once their key has been
validated (the auth dependency runs before the limit is checked), and per client address
otherwise. State is in-process, which is correct for the single instance Nexus runs as; a shared
store is needed before a second instance exists (docs/phases/phase9_production_tradeoffs.md).
"""
import ipaddress
import threading
import time
from collections import OrderedDict, deque
from typing import Deque

from fastapi import HTTPException, Request
from slowapi import Limiter

from src.config import get_settings

QUERY_LIMIT = "5/minute"  # each answer costs LLM tokens against free-tier quotas
INGEST_LIMIT = "10/minute"
READ_LIMIT = "60/minute"
ADMIN_LIMIT = "10/minute"

AUTH_FAILURES_PER_WINDOW = 10
AUTH_FAILURE_WINDOW_SECONDS = 60.0
MAX_TRACKED_CLIENTS = 10_000


def client_ip(request: Request) -> str:
    """
    The caller's address. Forwarded headers are believed only for the number of trusted proxy
    hops configured: each trusted proxy appends the address it saw, so the caller is the
    entry that many places from the right. Anything earlier is client-supplied and spoofable.
    """
    peer = request.client.host if request.client else "127.0.0.1"
    hops = get_settings().trusted_proxy_hops
    if hops <= 0:
        return peer
    forwarded = [part.strip() for part in request.headers.get("x-forwarded-for", "").split(",") if part.strip()]
    if len(forwarded) < hops:
        return peer
    try:
        return str(ipaddress.ip_address(forwarded[-hops]))
    except ValueError:
        return peer


def rate_limit_key(request: Request) -> str:
    tenant_id = getattr(request.state, "tenant_id", None)
    return f"tenant:{tenant_id}" if tenant_id else f"ip:{client_ip(request)}"


limiter = Limiter(key_func=rate_limit_key)


class AuthFailureThrottle:
    """
    Counts rejected credentials per client address and refuses a client that keeps guessing.
    Tenant keys are 256-bit and cannot be guessed, but the admin key is chosen by a person, and
    every rejected key otherwise costs a database lookup.
    """

    def __init__(self, max_failures: int = AUTH_FAILURES_PER_WINDOW, window_seconds: float = AUTH_FAILURE_WINDOW_SECONDS):
        self.max_failures = max_failures
        self.window_seconds = window_seconds
        self._failures: "OrderedDict[str, Deque[float]]" = OrderedDict()
        self._lock = threading.Lock()

    def _recent(self, client: str, now: float) -> Deque[float]:
        attempts = self._failures.get(client)
        if attempts is None:
            return deque()
        while attempts and now - attempts[0] >= self.window_seconds:
            attempts.popleft()
        return attempts

    def check(self, client: str) -> None:
        with self._lock:
            if len(self._recent(client, time.monotonic())) >= self.max_failures:
                raise HTTPException(
                    status_code=429,
                    detail="Too many rejected credentials. Wait a minute before trying again.",
                    headers={"Retry-After": str(int(self.window_seconds))},
                )

    def record(self, client: str) -> None:
        now = time.monotonic()
        with self._lock:
            attempts = self._recent(client, now)
            attempts.append(now)
            self._failures[client] = attempts
            self._failures.move_to_end(client)
            while len(self._failures) > MAX_TRACKED_CLIENTS:
                self._failures.popitem(last=False)

    def reset(self) -> None:
        with self._lock:
            self._failures.clear()


auth_failures = AuthFailureThrottle()
