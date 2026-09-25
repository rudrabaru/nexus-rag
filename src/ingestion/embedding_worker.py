"""
Embedding stage of an ingestion: embeds chunks in batches and reports the outcome.

It writes nothing. The job commits the returned chunks together with the job status in one
transaction (src/jobs/commit.py), so a crash mid-embedding leaves no partial document.
"""
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from src.embedding.models import EmbeddedChunk

logger = logging.getLogger(__name__)

BATCH_SIZE = 50


class EmbeddingUnavailableError(RuntimeError):
    """No chunk could be embedded. Transient by assumption (quota, outage), so the job is retried."""


@dataclass
class EmbeddingOutcome:
    chunks: List[EmbeddedChunk]
    failed_indices: List[int]
    total_chunks: int
    error_reason: Optional[str]

    @property
    def total_tokens(self) -> int:
        return sum(c.token_count for c in self.chunks)

    @property
    def status(self) -> str:
        return "partial_success" if self.failed_indices else "complete"

    @property
    def stats(self) -> Dict[str, int]:
        return {"total_added": len(self.chunks), "total_tokens": self.total_tokens}

    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        if not self.failed_indices:
            return None
        return {
            "failed_chunk_indices": self.failed_indices,
            "embedded": len(self.chunks),
            "total": self.total_chunks,
            "error_reason": self.error_reason,
        }


class EmbeddingWorker:
    def __init__(self, embedding_generator):
        self.embedding_generator = embedding_generator

    def embed(
        self,
        all_chunks: List[Any],
        on_progress: Optional[Callable[[int], None]] = None,
        pipeline_logger: Any = None,
        job_id: Optional[str] = None,
    ) -> EmbeddingOutcome:
        start = time.time()
        embedded: List[EmbeddedChunk] = []
        failed_indices: List[int] = []

        for i in range(0, len(all_chunks), BATCH_SIZE):
            batch = all_chunks[i:i + BATCH_SIZE]
            embedded_batch, batch_failures = self.embedding_generator.generate_embeddings(batch)
            embedded.extend(embedded_batch)
            failed_indices.extend(idx + i for idx in batch_failures)
            if on_progress:
                on_progress(min(75 + int(24 * min(i + BATCH_SIZE, len(all_chunks)) / len(all_chunks)), 99))

        error_reason = None
        if failed_indices:
            error_reason = getattr(self.embedding_generator, "last_error", None) or (
                f"{len(failed_indices)} chunks failed embedding (API rate limit or timeout)."
            )
        if all_chunks and not embedded:
            raise EmbeddingUnavailableError(error_reason or "No chunk could be embedded.")

        if pipeline_logger:
            pipeline_logger.log_event(
                "embedding_complete",
                job_id=job_id,
                chunk_count=len(embedded),
                batch_count=-(-len(all_chunks) // BATCH_SIZE),
                duration_ms=(time.time() - start) * 1000,
            )
        return EmbeddingOutcome(
            chunks=embedded, failed_indices=failed_indices, total_chunks=len(all_chunks), error_reason=error_reason
        )
