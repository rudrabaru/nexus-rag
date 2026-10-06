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

Asymmetric retrieval models embed queries and documents differently (Voyage input_type,
Jina task). Using the wrong side silently lowers recall, so callers always say which.
"""
import threading
from typing import Dict, Optional

from src.config import Settings
from src.db.schema import EMBEDDING_DIMENSION
from src.embedding.embedder import Embedder, EmbeddingBatch
from src.embedding.pacing import RateWindow

DEFAULT_MODELS = {"voyage": "voyage-4", "ollama": "bge-m3"}


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


_BUILDERS = {"voyage": voyage_embedder, "ollama": ollama_embedder}


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
