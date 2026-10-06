"""
Client-side pacing for a provider's requests-per-minute and tokens-per-minute limits.

Why client-side: Voyage's limits for an account with no payment method are 3 RPM and 10K TPM.
Sending as fast as possible and backing off on 429s would spend most of an ingestion waiting
on penalties; pacing to the published limit spends the same wall-clock time without the
errors. Server-side 429s are still handled (src/embedding/providers.py), because the limit is
per API key and this window only sees one process's traffic.

Thread-based, not asyncio-based: every ingestion job runs in its own event loop (a
Procrastinate sync task calling asyncio.run), and an asyncio primitive shared across loops is
the bug fixed in item 3. A threading.Lock guards the bookkeeping; callers sleep with asyncio.
"""
import threading
import time
from collections import deque
from typing import Deque, Tuple

WINDOW_SECONDS = 60.0


class RateWindow:
    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self._sent: Deque[Tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def reserve(self, tokens: int, now: float = None) -> float:
        """
        Records a request of `tokens` and returns 0.0 if it may be sent now; otherwise records
        nothing and returns how many seconds to wait before asking again.
        """
        now = time.monotonic() if now is None else now
        tokens = min(tokens, self.tokens_per_minute)  # one request larger than the window still goes alone
        with self._lock:
            while self._sent and now - self._sent[0][0] >= WINDOW_SECONDS:
                self._sent.popleft()
            used = sum(t for _, t in self._sent)
            if len(self._sent) < self.requests_per_minute and used + tokens <= self.tokens_per_minute:
                self._sent.append((now, tokens))
                return 0.0
            return max(WINDOW_SECONDS - (now - self._sent[0][0]), 0.05)
