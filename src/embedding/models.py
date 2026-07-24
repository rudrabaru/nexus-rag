from datetime import datetime, timezone
from typing import List
from pydantic import BaseModel, Field
from src.chunking.metadata import ChunkMetadata


class EmbeddedChunk(ChunkMetadata):
    """
    Represents a chunk of content that has been embedded, including the vector.
    Inherits all metadata from ChunkMetadata.
    """

    embedding: List[float] = Field(
        ..., description="The embedding vector for this chunk"
    )
    embedding_model: str = Field(
        ..., description="The model used to generate this embedding"
    )
    embedded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="When the embedding was generated"
    )


class EmbeddingReport(BaseModel):
    """
    Summary statistics for the embedding generation phase.
    """

    model_name: str = Field(..., description="Model used for generation")
    total_chunks_processed: int = Field(
        0, description="Number of chunks successfully embedded"
    )
    total_failures: int = Field(0, description="Number of chunks that failed embedding")
    embedding_dimensions: int = Field(
        ..., description="Dimensionality of the embeddings"
    )
    avg_chunk_tokens: float = Field(
        0.0, description="Average tokens per embedded chunk"
    )
    duration_seconds: float = Field(
        0.0, description="Time taken to generate embeddings"
    )

    class Config:
        json_encoders = {float: lambda v: round(v, 2)}


class EmbeddingManifest(BaseModel):
    embedding_version: str
    source_chunk_version: str
    embedding_model: str
    embedding_dimension: int
    distance_metric: str
    normalized: bool
    total_chunk_count: int
    generation_timestamp: str


class SimilarityResult(BaseModel):
    """
    Represents the result of a similarity test.
    """

    query_chunk_id: str = Field(..., description="The chunk ID used as the query")
    query_text: str = Field(..., description="Text of the query chunk")
    nearest_neighbors: List[dict] = Field(
        ..., description="List of dicts with 'chunk_id', 'score', and 'text'"
    )
