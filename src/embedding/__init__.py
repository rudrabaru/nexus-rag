"""
Embeddings module: Convert textual chunks into vector representations.
"""

from .config import EmbeddingConfig
from .models import EmbeddedChunk, EmbeddingReport, SimilarityResult
from .generator import EmbeddingGenerator

__all__ = [
    "EmbeddingConfig",
    "EmbeddedChunk",
    "EmbeddingReport",
    "SimilarityResult",
    "EmbeddingGenerator",
]
