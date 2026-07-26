"""
Chunking module: Convert documents into semantically meaningful retrieval units.

This module implements the Phase 2 pipeline:
- Load processed documents
- Split on semantic boundaries
- Apply token-based sizing with overlap
- Generate rich metadata for each chunk
- Inspect and validate chunk quality
"""

from .metadata import ChunkMetadata, ChunkingConfig
from .tokenizer import TokenCounter, TokenBudget
from .chunker import DocumentChunker

__all__ = [
    "ChunkMetadata",
    "ChunkingConfig",
    "TokenCounter",
    "TokenBudget",
    "DocumentChunker",
]
