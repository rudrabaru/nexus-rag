import hashlib
import logging
from typing import Dict, List, Optional, Tuple

from src.chunking.metadata import ChunkMetadata
from src.embedding.models import EmbeddedChunk
from src.embedding.embedder import Embedder, EmbeddingError
from src.errors import EmbeddingRejectedError
from src.stores.checkpoints import Checkpoint, CheckpointStore

logger = logging.getLogger(__name__)


def embedding_input(chunk: ChunkMetadata) -> str:
    """
    The text a chunk is embedded as: its document and heading path, then its text. The
    prefix gives short chunks the context of where they sit.
    """
    return f"[{chunk.source_document} > {' > '.join(chunk.heading_path or [])}]\n{chunk.chunk_text}"


def input_hash(index_id: str, text: str) -> str:
    """Identifies exactly what was embedded and into which index: a vector is reusable only for the same pair."""
    return hashlib.sha256(f"{index_id}\x1f{text}".encode("utf-8")).hexdigest()


def apportion(total: int, weights: List[int]) -> List[int]:
    """Splits `total` across `weights` proportionally, exactly: the parts always sum to `total`."""
    weight_sum = sum(weights) or 1
    parts = [total * w // weight_sum for w in weights]
    parts[0] += total - sum(parts)
    return parts


class EmbeddingGenerator:
    """
    Embeds chunks as documents into one index, one provider request at a time. Keeps per-run state
    (last_error, provider_tokens), so one instance is used per ingestion run, never shared between
    concurrent runs.

    With a checkpoint store, every completed request is saved as it finishes and a retry of the
    same job reuses what was saved: at the card-free Voyage limit one ingestion is hours of paced
    requests, and a failure late in the run must not repeat the early part.
    """

    def __init__(self, embedder: Embedder, checkpoints: Optional[CheckpointStore] = None, job_id: Optional[str] = None):
        self.embedder = embedder
        self.last_error: Optional[str] = None
        self.provider_tokens = 0  # what the provider reported for this run's requests (reused checkpoints included)
        self._checkpoints = checkpoints if job_id else None
        self._job_id = job_id
        self._saved: Dict[str, Checkpoint] = checkpoints.load(job_id) if self._checkpoints else {}

    def generate_embeddings(self, chunks: List[ChunkMetadata]) -> Tuple[List[EmbeddedChunk], List[int]]:
        """
        Returns (embedded chunks, indices into `chunks` that failed). Empty chunks are skipped, not
        failed. A request the provider rejects for good (bad key, wrong model, wrong vector width)
        raises EmbeddingRejectedError: retrying it can only fail the same way.
        """
        positions = [i for i, c in enumerate(chunks) if c.chunk_text.strip()]
        if len(positions) < len(chunks):
            logger.warning(f"Skipped {len(chunks) - len(positions)} empty chunks before embedding.")
        if not positions:
            return [], []

        index_id = self.embedder.index_id
        texts = {i: embedding_input(chunks[i]) for i in positions}
        vectors: Dict[int, List[float]] = {}
        for i in positions:
            saved = self._saved.get(chunks[i].chunk_id)
            if saved and saved.input_hash == input_hash(index_id, texts[i]):
                vectors[i] = saved.embedding
                self.provider_tokens += saved.tokens
        if vectors:
            logger.info(f"EMBED | reusing {len(vectors)} of {len(positions)} chunks from an earlier attempt")

        todo = [i for i in positions if i not in vectors]
        failed: List[int] = []
        for group in self.embedder.group_indices([texts[i] for i in todo]):
            members = [todo[g] for g in group]
            try:
                batch = self.embedder.embed([texts[i] for i in members], "document")
            except EmbeddingError as e:
                self.last_error = str(e)
                if not e.retryable:
                    raise EmbeddingRejectedError(self._rejection(e)) from e
                logger.error(f"EMBED | {len(members)} chunks failed: {e}")
                failed.extend(members)
                continue
            except Exception as e:  # a malformed response or a dropped connection: this request fails, the run continues
                self.last_error = str(e)
                logger.error(f"EMBED | {len(members)} chunks failed: {e}")
                failed.extend(members)
                continue

            self.provider_tokens += batch.tokens
            for i, vector in zip(members, batch.vectors):
                vectors[i] = vector
            self._save(chunks, members, texts, batch.vectors, batch.tokens)

        embedded = [
            EmbeddedChunk(
                **chunks[i].model_dump(), embedding=vectors[i],
                embedding_model=self.embedder.model, index_id=index_id,
            )
            for i in positions if i in vectors
        ]
        return embedded, sorted(failed)

    def _save(self, chunks, members, texts, vectors, tokens) -> None:
        if not self._checkpoints:
            return
        index_id = self.embedder.index_id
        shares = apportion(tokens, [len(texts[i]) for i in members])
        try:
            self._checkpoints.save(self._job_id, [
                {"chunk_id": chunks[i].chunk_id, "input_hash": input_hash(index_id, texts[i]), "embedding": vector, "tokens": share}
                for i, vector, share in zip(members, vectors, shares)
            ])
        except Exception:  # losing a checkpoint costs time on a retry, never the run
            logger.exception("EMBED | could not save a checkpoint")

    def _rejection(self, error: EmbeddingError) -> str:
        reason = f"HTTP {error.status}" if error.status else "an invalid response"
        return (
            f"The embedding provider ({self.embedder.provider}) rejected the request ({reason}). "
            "Check the provider key, the model and that it outputs the width the index uses."
        )
