"""
Load shedding for the answer endpoints. Each answer costs LLM tokens and holds a worker slot, so
only so many run at once; the rest are refused immediately with 503 + Retry-After instead of
queueing behind them. An overloaded server is then distinguishable from an answer, and a client or
load balancer can back off.
"""
import asyncio
from typing import Optional

from fastapi import HTTPException, Request

RETRY_AFTER_SECONDS = 5


class Slot:
    """One held unit of capacity. Releasing is idempotent, so every path that might release it can."""

    def __init__(self, semaphore: Optional[asyncio.Semaphore]):
        self._semaphore = semaphore

    def release(self) -> None:
        if self._semaphore is not None:
            semaphore, self._semaphore = self._semaphore, None
            semaphore.release()

    async def __aenter__(self) -> "Slot":
        return self

    async def __aexit__(self, *exc) -> None:
        self.release()


async def acquire_slot(request: Request) -> Slot:
    semaphore = getattr(request.app.state, "query_semaphore", None)
    if semaphore is None:
        return Slot(None)
    if semaphore.locked():
        raise HTTPException(
            status_code=503, detail="The server is at capacity. Retry shortly.",
            headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
        )
    await semaphore.acquire()
    return Slot(semaphore)
