"""
What we are willing to fetch, and how fast.

This is an abuse and compliance policy, not a content rule: it decides which sites a tenant
may point the platform at, never what content is kept. It exists because running a crawler
on shared hosting got the project's Hugging Face account suspended for automated traffic
(plan 2.2). The domain lists are configuration (FETCH_DENIED_DOMAINS / FETCH_ALLOWED_DOMAINS),
not code.
"""
import threading
import time
from typing import Dict, List
from urllib.parse import urlparse

from src.ingestion.url_policy import UnsafeUrlError, validate_public_url


def _matches(host: str, domains: List[str]) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def check_fetchable(url: str, allowed_domains: List[str], denied_domains: List[str]) -> None:
    """Raises UnsafeUrlError unless the URL is https, public, and permitted by the domain lists."""
    if urlparse(url.strip()).scheme.lower() != "https":
        raise UnsafeUrlError("Only https URLs are fetched.")
    validate_public_url(url)
    host = (urlparse(url.strip()).hostname or "").lower().rstrip(".")
    if _matches(host, denied_domains):
        raise UnsafeUrlError(f"Fetching from {host} is not permitted (social networks and people-search sites are denied).")
    if allowed_domains and not _matches(host, allowed_domains):
        raise UnsafeUrlError(f"Fetching from {host} is not permitted: only allowlisted domains are fetched.")


class DomainPacer:
    """
    At most one request per domain every `min_interval` seconds, per process, even though the
    request goes through a reader API: the reader fetches from the target site on our behalf,
    so bursting it through a provider is still bursting it. Thread-safe; callers sleep.
    """

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._next_slot: Dict[str, float] = {}
        self._lock = threading.Lock()

    def reserve(self, url: str) -> float:
        """Books the next free slot for the URL's domain; returns seconds to wait until it."""
        host = (urlparse(url).hostname or "").lower()
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot.get(host, 0.0))
            self._next_slot[host] = slot + self.min_interval
            return slot - now
