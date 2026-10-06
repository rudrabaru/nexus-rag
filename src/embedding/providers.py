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
- cloudflare: Workers AI bge-m3 (1024-dim) over its OpenAI-compatible endpoint. The free plan has no payment
  method and allows about 9M tokens a day (10,000 neurons at 1,075 neurons per M tokens) at 3,000 requests
  a minute, so ingestion is not paced. Documented limits; measure before relying on them.

Asymmetric retrieval models embed queries and documents differently (Voyage input_type,
Jina task). Using the wrong side silently lowers recall, so callers always say which.
"""
import threading
from typing import Dict, Optional

from src.config import Settings
from src.db.schema import EMBEDDING_DIMENSION
from src.embedding.embedder import Embedder, EmbeddingBatch
from src.embedding.pacing import RateWindow

DEFAULT_MODELS = {"voyage": "voyage-4", "ollama": "bge-m3", "cloudflare": "bge-m3"}


def _openai_style(data: dict) -> EmbeddingBatch:
    """Voyage answers {data: [{embedding, index}], usage: {total_tokens}}."""
    rows = sorted(data["data"], key=lambda row: row.get("index", 0))
    return EmbeddingBatch(vectors=[row["embedding"] for row in rows], tokens=int(data.get("usage", {}).get("total_tokens", 0)))


# One pacing window per provider per process, shared by every Embedder built for it.
_windows: Dict[str, RateWindow] = {}


_windows_lock = threading.Lock()


def _window(provider: str, rpm: int, tpm: int) -> RateWindow:
    with _windows_lock:
        if provider not in _windows:
            _windows[provider] = RateWindow(rpm, tpm)
        return _windows[provider]


def voyage_embedder(settings: Settings, model: str) -> Embedder:
    return Embedder(
        provider="voyage",
        model=model,
        endpoint=f"{settings.voyage_base_url.rstrip('/')}/embeddings",
        headers={"Authorization": f"Bearer {settings.voyage_api_key.get_secret_value()}"},
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


def cloudflare_embedder(settings: Settings, model: str) -> Embedder:
    account = settings.cloudflare_account_id
    return Embedder(
        provider="cloudflare",
        model=model,
        endpoint=f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1/embeddings",
        headers={"Authorization": f"Bearer {settings.cloudflare_api_token.get_secret_value()}"},
        request_body=lambda texts, input_type: {"model": f"@cf/baai/{model}", "input": texts},
        read_response=_openai_style,
        # The documentation states no per-request batch or token cap. These are conservative starting
        # points (experiment): small enough to stay well inside the model's 60,000-token context per input.
        max_texts_per_request=50,
        max_tokens_per_request=100_000,
    )


_BUILDERS = {"voyage": voyage_embedder, "ollama": ollama_embedder, "cloudflare": cloudflare_embedder}


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
