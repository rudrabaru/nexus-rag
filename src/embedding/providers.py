"""
Embedding providers behind one small interface.

An index has exactly one embedding model: vectors from different models live in different
spaces and are never compared. The index id is therefore derived from the provider and model
("voyage:voyage-4"), recorded on every chunk, and used to scope every search.

Providers differ only in wire format, so each is a plain function that builds an Embedder
(endpoint, request body, response reader, limits), not a subclass:
- voyage: hosted default. 200M free tokens per model, but 3 RPM / 10K TPM without a payment
  method, so requests are paced (src/embedding/pacing.py).
- ollama: local (bge-m3 by default, 1024-dim) for the optional GPU worker.
- jina: kept only so the prototype jina-embeddings-v3 index, which the frozen retrieval
  baselines were measured on, stays queryable. Its free allowance is a one-time grant.

Asymmetric retrieval models embed queries and documents differently (Voyage input_type,
Jina task). Using the wrong side silently lowers recall, so callers always say which.
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional

import httpx

from src.config import Settings
from src.embedding.pacing import RateWindow
from src.registry.schema import EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)

InputType = Literal["query", "document"]

DEFAULT_MODELS = {"voyage": "voyage-4", "ollama": "bge-m3", "jina": "jina-embeddings-v3"}

# List prices (USD per 1M tokens), for cost reporting only. Voyage's first 200M tokens per
# model are free, so this is what the usage would cost once that grant is spent.
LIST_PRICE_PER_MILLION = {
    "voyage-4-large": 0.12, "voyage-4": 0.06, "voyage-4-lite": 0.02, "jina-embeddings-v3": 0.02,
}

# Pacing needs a token estimate before the provider has counted anything, and the API image
# carries no tokenizer. ~3 characters per token over-estimates English (~4), so pacing errs
# toward waiting rather than toward a 429.
CHARS_PER_TOKEN_ESTIMATE = 3
MAX_ATTEMPTS = 4
MAX_BACKOFF_SECONDS = 60.0
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class EmbeddingError(RuntimeError):
    """An embedding request failed. `retryable` is False for errors retrying cannot fix (bad key, bad model)."""

    def __init__(self, message: str, retryable: bool):
        super().__init__(message)
        self.retryable = retryable


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
            for group in self._request_groups(texts):
                batch = await self._post_paced(client, group, input_type)
                vectors.extend(batch.vectors)
                tokens += batch.tokens
        return EmbeddingBatch(vectors=vectors, tokens=tokens)

    def _request_groups(self, texts: List[str]) -> List[List[str]]:
        """Splits texts into requests within the provider's per-request text and token limits."""
        token_cap = self.max_tokens_per_request
        if self.window:
            token_cap = min(token_cap, self.window.tokens_per_minute)
        groups, current, current_tokens = [], [], 0
        for text in texts:
            estimate = estimate_tokens(text)
            if current and (len(current) >= self.max_texts_per_request or current_tokens + estimate > token_cap):
                groups.append(current)
                current, current_tokens = [], 0
            current.append(text)
            current_tokens += estimate
        if current:
            groups.append(current)
        return groups

    async def _post_paced(self, client: httpx.AsyncClient, texts: List[str], input_type: InputType) -> EmbeddingBatch:
        for attempt in range(MAX_ATTEMPTS):
            await self._wait_for_window(sum(estimate_tokens(t) for t in texts))
            try:
                response = await client.post(self.endpoint, headers=self.headers, json=self.request_body(texts, input_type))
            except httpx.TransportError as e:
                error, delay = EmbeddingError(f"{self.provider} unreachable: {e}", retryable=True), 2.0 ** attempt
            else:
                if response.status_code < 400:
                    return self._validated(self.read_response(response.json()), len(texts))
                retryable = response.status_code in RETRYABLE_STATUS
                error = EmbeddingError(
                    f"{self.provider} embeddings HTTP {response.status_code}: {response.text[:200]}", retryable
                )
                if not retryable:
                    raise error
                delay = _retry_after(response) or 2.0 ** (attempt + 1)
            if attempt == MAX_ATTEMPTS - 1:
                raise error
            logger.warning(f"EMBED | {error} | retry {attempt + 1}/{MAX_ATTEMPTS - 1} in {delay:.1f}s")
            await asyncio.sleep(min(delay, MAX_BACKOFF_SECONDS))
        raise AssertionError("unreachable")

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


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


def _retry_after(response: httpx.Response) -> Optional[float]:
    try:
        return float(response.headers.get("retry-after", ""))
    except ValueError:
        return None


def _openai_style(data: dict) -> EmbeddingBatch:
    """Voyage and Jina both answer {data: [{embedding, index}], usage: {total_tokens}}."""
    rows = sorted(data["data"], key=lambda row: row.get("index", 0))
    return EmbeddingBatch(vectors=[row["embedding"] for row in rows], tokens=int(data.get("usage", {}).get("total_tokens", 0)))


# One pacing window per provider per process, shared by every Embedder built for it.
_windows: Dict[str, RateWindow] = {}


def _window(provider: str, rpm: int, tpm: int) -> RateWindow:
    if provider not in _windows:
        _windows[provider] = RateWindow(rpm, tpm)
    return _windows[provider]


def voyage_embedder(settings: Settings, model: str) -> Embedder:
    return Embedder(
        provider="voyage",
        model=model,
        endpoint=f"{settings.voyage_base_url.rstrip('/')}/embeddings",
        headers={"Authorization": f"Bearer {settings.voyage_api_key}"},
        request_body=lambda texts, input_type: {
            "input": texts, "model": model, "input_type": input_type, "output_dimension": EMBEDDING_DIMENSION,
        },
        read_response=_openai_style,
        max_texts_per_request=1000,
        max_tokens_per_request=120_000,  # the smallest per-request cap across voyage-4 models
        window=_window("voyage", settings.voyage_rpm, settings.voyage_tpm),
    )


def ollama_embedder(settings: Settings, model: str) -> Embedder:
    return Embedder(
        provider="ollama",
        model=model,
        endpoint=f"{settings.ollama_base_url.rstrip('/')}/api/embed",
        headers={},
        request_body=lambda texts, input_type: {"model": model, "input": texts},
        read_response=lambda data: EmbeddingBatch(vectors=data["embeddings"], tokens=int(data.get("prompt_eval_count", 0))),
        max_texts_per_request=32,  # bounds one request's VRAM on a 4 GB GPU
        max_tokens_per_request=32 * 8192,
        timeout_seconds=300.0,
    )


def jina_embedder(settings: Settings, model: str) -> Embedder:
    tasks = {"query": "retrieval.query", "document": "retrieval.passage"}
    return Embedder(
        provider="jina",
        model=model,
        endpoint="https://api.jina.ai/v1/embeddings",
        headers={"Authorization": f"Bearer {settings.jina_api_key}"},
        request_body=lambda texts, input_type: {"model": model, "input": texts, "task": tasks[input_type]},
        read_response=_openai_style,
        max_texts_per_request=100,
        max_tokens_per_request=100 * 8192,
    )


_BUILDERS = {"voyage": voyage_embedder, "ollama": ollama_embedder, "jina": jina_embedder}


def build_embedder(settings: Settings, index_id: Optional[str] = None) -> Embedder:
    """The embedder for `index_id` ("provider:model"), or for the configured provider when omitted."""
    if index_id:
        provider, _, model = index_id.partition(":")
    else:
        provider = settings.embedding_provider.lower()
        model = settings.embedding_model or DEFAULT_MODELS.get(provider, "")
    if provider not in _BUILDERS or not model:
        raise ValueError(f"Unknown embedding index {index_id or provider!r}; expected provider:model with provider in {sorted(_BUILDERS)}.")
    return _BUILDERS[provider](settings, model)
