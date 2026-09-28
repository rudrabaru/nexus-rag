from datetime import datetime, timezone
from typing import List

from pydantic import Field

from src.chunking.metadata import ChunkMetadata


class EmbeddedChunk(ChunkMetadata):
    """A chunk with its vector, and the index (provider:model) the vector belongs to."""

    embedding: List[float] = Field(..., description="The embedding vector for this chunk")
    embedding_model: str = Field(..., description="The model used to generate this embedding")
    index_id: str = Field(..., description="The embedding index this vector belongs to (provider:model)")
    embedded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="When the embedding was generated"
    )
