"""One embedding request path: provider-sized batches, rate-window pacing, retries and vector validation."""
import asyncio
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional

import httpx

from src.db.schema import EMBEDDING_DIMENSION
from src.embedding.pacing import WINDOW_SECONDS, RateWindow
from src.retry import RETRYABLE_STATUS, RetryableError, retry_async
from src.tokens import estimate_tokens

logger = logging.getLogger(__name__)

InputType = Literal["query", "document"]

# List prices (USD per 1M tokens), for cost reporting only. Voyage's first 200M tokens per
# model are free, so this is what the usage would cost once that grant is spent.
LIST_PRICE_PER_MILLION = {"voyage-4-large": 0.12, "voyage-4": 0.06, "voyage-4-lite": 0.02}

MAX_ATTEMPTS = 4


class EmbeddingError(RuntimeError):
    """An embedding request failed. `retryable` is False for errors retrying cannot fix (bad key, bad model)."""

    def __init__(self, message: str, retryable: bool, status: Optional[int] = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status  # the provider's HTTP status, when it answered


@dataclass
class EmbeddingBatch:
    vectors: List[List[float]]
    tokens: int


@dataclass
class Embedder:
    provider: str
    model: str
    endpoint: str
    headers: Dict[str, str]
    request_body: Callable[[List[str], InputType], dict]
    read_response: Callable[[dict], EmbeddingBatch]
    max_texts_per_request: int
    max_tokens_per_request: int
    window: Optional[RateWindow] = None
    timeout_seconds: float = 60.0

    @property
    def index_id(self) -> str:
        return f"{self.provider}:{self.model}"

    def cost_usd(self, tokens: int) -> float:
        return tokens * LIST_PRICE_PER_MILLION.get(self.model, 0.0) / 1_000_000

    def embed(self, texts: List[str], input_type: InputType) -> EmbeddingBatch:
        """Sync entry point, for callers in a worker thread with no running event loop."""
        return asyncio.run(self.aembed(texts, input_type))

    async def aembed(self, texts: List[str], input_type: InputType) -> EmbeddingBatch:
        vectors: List[List[float]] = []
        tokens = 0
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            for positions in self.group_indices(texts):
                batch = await self._post_paced(client, [texts[i] for i in positions], input_type)
                vectors.extend(batch.vectors)
                tokens += batch.tokens
        return EmbeddingBatch(vectors=vectors, tokens=tokens)

    def group_indices(self, texts: List[str]) -> List[List[int]]:
        """Positions of `texts` split into requests within the provider's per-request text and token limits."""
        token_cap = self.max_tokens_per_request
        if self.window:
            token_cap = min(token_cap, self.window.tokens_per_minute)
        groups, current, current_tokens = [], [], 0
        for position, text in enumerate(texts):
            estimate = estimate_tokens(text)
            if current and (len(current) >= self.max_texts_per_request or current_tokens + estimate > token_cap):
                groups.append(current)
                current, current_tokens = [], 0
            current.append(position)
            current_tokens += estimate
        if current:
            groups.append(current)
        return groups

    async def _post_paced(self, client: httpx.AsyncClient, texts: List[str], input_type: InputType) -> EmbeddingBatch:
        async def send() -> EmbeddingBatch:
            await self._wait_for_window(sum(estimate_tokens(t) for t in texts))
            try:
                response = await client.post(self.endpoint, headers=self.headers, json=self.request_body(texts, input_type))
            except httpx.TransportError as e:
                raise RetryableError(EmbeddingError(f"{self.provider} unreachable: {e}", retryable=True))
            if response.status_code < 400:
                return self._validated(self.read_response(response.json()), len(texts))
            retryable = response.status_code in RETRYABLE_STATUS
            error = EmbeddingError(
                f"{self.provider} embeddings HTTP {response.status_code}: {response.text[:200]}", retryable,
                status=response.status_code,
            )
            if not retryable:
                raise error
            raise RetryableError(error, delay=self._rate_limit_wait(response) if response.status_code == 429 else _retry_after(response))

        return await retry_async(send, MAX_ATTEMPTS, "EMBED")

    def _rate_limit_wait(self, response: httpx.Response) -> float:
        """
        A 429 means this minute's budget is spent, so retrying within seconds only burns attempts:
        wait for the budget to refill (a whole window), or as long as the provider asks.
        """
        waits = [_retry_after(response) or 0.0]
        waits.append(WINDOW_SECONDS if self.window else 2.0)
        return max(waits)

    async def _wait_for_window(self, tokens: int) -> None:
        if not self.window:
            return
        while (wait := self.window.reserve(tokens)) > 0:
            logger.info(f"EMBED | pacing {self.provider} to {self.window.requests_per_minute} RPM / "
                        f"{self.window.tokens_per_minute} TPM | waiting {wait:.1f}s")
            await asyncio.sleep(wait)

    def _validated(self, batch: EmbeddingBatch, expected_count: int) -> EmbeddingBatch:
        if len(batch.vectors) != expected_count:
            raise EmbeddingError(f"{self.provider} returned {len(batch.vectors)} vectors for {expected_count} texts", False)
        width = {len(v) for v in batch.vectors}
        if width != {EMBEDDING_DIMENSION}:
            raise EmbeddingError(
                f"{self.index_id} returned {sorted(width)}-dimensional vectors; the chunks.embedding "
                f"column holds {EMBEDDING_DIMENSION}. Choose a model with that output size.", retryable=False
            )
        return batch


def _retry_after(response: httpx.Response) -> Optional[float]:
    try:
        return float(response.headers.get("retry-after", ""))
    except ValueError:
        return None
