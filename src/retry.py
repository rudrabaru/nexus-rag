"""
One retry loop for every outbound HTTP call (embeddings, rerankers), so backoff, logging and the
final error are decided in one place.

The caller's `send` makes one attempt and either returns, raises RetryableError (this attempt failed
in a way another attempt may fix), or raises anything else (retrying cannot help, so it propagates
at once). When attempts run out, the error carried by the last RetryableError is raised.
"""
import asyncio
import logging
import random
from typing import Awaitable, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# HTTP statuses a provider uses for "try again": timeouts, rate limits and server-side failures.
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

MAX_BACKOFF_SECONDS = 90.0  # longer than a rate-limit window, so a provider's own wait is honoured in full


class RetryableError(Exception):
    """An attempt failed transiently. `error` is what to raise if it was the last attempt; `delay` overrides the backoff."""

    def __init__(self, error: Exception, delay: Optional[float] = None):
        super().__init__(str(error))
        self.error = error
        self.delay = delay


def backoff_seconds(attempt: int, jitter: bool = False) -> float:
    """1, 2, 4, ... seconds for attempt 0, 1, 2, ...; jitter spreads concurrent retries apart."""
    return float(2 ** attempt) + (random.uniform(0, 1) if jitter else 0.0)


async def retry_async(send: Callable[[], Awaitable[T]], attempts: int, label: str = "") -> T:
    for attempt in range(attempts):
        try:
            return await send()
        except RetryableError as e:
            if attempt == attempts - 1:
                raise e.error from None
            delay = min(e.delay if e.delay is not None else backoff_seconds(attempt), MAX_BACKOFF_SECONDS)
            logger.warning(f"{label} {e.error} | retry {attempt + 1}/{attempts - 1} in {delay:.1f}s")
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")
