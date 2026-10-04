"""
Pydantic schemas for chunk representation and metadata.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ChunkMetadata(BaseModel):
    chunk_id: str = Field(..., description="Unique chunk identifier")
    source_url: str = Field(..., description="URL of original document")
    source_document: str = Field(..., description="Source document filename")
    title: str = Field(..., description="Original document title")

    # Structural metadata
    heading_path: List[str] = Field(
        default_factory=list, description="Hierarchy of headings leading to this chunk"
    )
    section_title: str = Field("", description="Title of the immediate section")

    chunk_text: str = Field(..., description="Chunk content")
    token_count: int = Field(..., description="Token count (for budgeting)")

    # Content flags
    starts_with_heading: bool = Field(False, description="Begins with heading?")
    contains_code: bool = Field(False, description="Chunk contains code blocks")
    contains_table: bool = Field(False, description="Chunk contains markdown tables")
    content_type: str = Field(
        "mixed",
        description="Primary content type: text, code, table, mixed",
    )
    document_version: str = Field(..., description="Version of the processed doc used")
    chunk_version: str = Field(..., description="Chunking version used")
    oversized_chunk: bool = Field(
        False, description="Is this chunk > max target size but allowed as exception"
    )
    tiny_chunk_merged: bool = Field(
        False, description="Was this chunk created by merging tiny chunks"
    )

    tenant_id: Optional[str] = Field(None, description="Tenant that owns the chunk")
    doc_id: Optional[str] = Field(None, description="Registry document this chunk belongs to")


class ChunkingConfig(BaseModel):
    """
    Configuration for document chunking.
    """

    chunk_size: int = Field(600, description="Target chunk size in tokens")
    overlap: int = Field(125, description="Overlap between chunks in tokens")
    embedding_hard_limit: int = Field(2000, description="Maximum tokens allowed before hard truncation to prevent embedding API failures")
    min_chunk_tokens: int = Field(150, description="Minimum tokens per chunk")
    max_chunk_tokens: int = Field(800, description="Maximum tokens per chunk")
    source_version: str = Field("unknown", description="Version of input docs")
    output_version: str = Field("unknown", description="Version of output chunks")

    model_config = ConfigDict(validate_assignment=True)

