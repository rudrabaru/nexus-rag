from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    model_name: str = Field(
        "jina-embeddings-v3", description="Name of the embedding model"
    )
    batch_size: int = Field(100, description="Batch size for generating embeddings")
    max_retries: int = Field(3, description="Maximum number of retries for embedding API calls")
    expected_dimensions: int = Field(
        1024, description="Expected dimensionality of the embedding vectors"
    )
    normalize_embeddings: bool = Field(
        True, description="Whether to normalize embeddings to unit length"
    )
    distance_metric: str = Field(
        "cosine",
        description="The distance metric the embedding space is designed for",
    )

    class Config:
        validate_assignment = True
