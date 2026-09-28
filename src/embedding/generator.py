import logging
from typing import List, Optional, Tuple

from src.chunking.metadata import ChunkMetadata
from src.embedding.models import EmbeddedChunk
from src.embedding.providers import Embedder

logger = logging.getLogger(__name__)


def embedding_input(chunk: ChunkMetadata) -> str:
    """
    The text a chunk is embedded as: its document and heading path, then its text. The
    prefix gives short chunks the context of where they sit.
    """
    return f"[{chunk.source_document} > {' > '.join(chunk.heading_path or [])}]\n{chunk.chunk_text}"


class EmbeddingGenerator:
    """
    Embeds chunks as documents into one index. Keeps per-run state (last_error), so one
    instance is used per ingestion run, never shared between concurrent runs.
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.last_error: Optional[str] = None

    def generate_embeddings(self, chunks: List[ChunkMetadata]) -> Tuple[List[EmbeddedChunk], List[int]]:
        """Returns (embedded chunks, indices into `chunks` that failed). Empty chunks are skipped, not failed."""
        positions = [i for i, c in enumerate(chunks) if c.chunk_text.strip()]
        if len(positions) < len(chunks):
            logger.warning(f"Skipped {len(chunks) - len(positions)} empty chunks before embedding.")
        if not positions:
            return [], []

        try:
            batch = self.embedder.embed([embedding_input(chunks[i]) for i in positions], "document")
        except Exception as e:  # EmbeddingError, or a malformed response: the batch fails, the run continues
            self.last_error = str(e)
            logger.error(f"EMBED | {len(positions)} chunks failed: {e}")
            return [], positions

        embedded = [
            EmbeddedChunk(
                **chunks[i].model_dump(), embedding=vector,
                embedding_model=self.embedder.model, index_id=self.embedder.index_id,
            )
            for i, vector in zip(positions, batch.vectors)
        ]
        return embedded, []
