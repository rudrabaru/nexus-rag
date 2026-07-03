import torch
from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    """
    Configuration for embedding generation.
    """

    model_name: str = Field(
        "BAAI/bge-small-en-v1.5", description="Name of the sentence-transformers model"
    )
    batch_size: int = Field(32, description="Batch size for generating embeddings")
    output_version: str = Field(
        "v1", description="Version string for the embeddings output directory"
    )
    expected_dimensions: int = Field(
        384, description="Expected dimensionality of the embedding vectors"
    )
    device: str = Field(
        "cuda" if torch.cuda.is_available() else "cpu",
        description="Device to run the model on",
    )
    normalize_embeddings: bool = Field(
        True, description="Whether to normalize embeddings to unit length"
    )
    distance_metric: str = Field(
        "cosine",
        description="The distance metric the embedding space is designed for (e.g. cosine, dot_product, euclidean)",
    )

    class Config:
        validate_assignment = True
