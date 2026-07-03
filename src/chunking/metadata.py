"""
Pydantic schemas for chunk representation and metadata.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """
    Represents a single chunk of content with full metadata for source attribution
    and debugging.
    """

    chunk_id: str = Field(..., description="Unique chunk identifier")
    source_url: str = Field(..., description="URL of original document")
    source_document: str = Field(..., description="Source document filename")
    title: str = Field(..., description="Original document title")

    # Structural metadata
    heading_path: List[str] = Field(
        default_factory=list, description="Hierarchy of headings leading to this chunk"
    )
    section_title: str = Field("", description="Title of the immediate section")

    # Chunking metadata
    chunk_index: int = Field(..., description="Chunk position in document")
    total_chunks: int = Field(0, description="Total number of chunks in document")
    chunk_text: str = Field(..., description="Chunk content")
    token_count: int = Field(..., description="Token count (for budgeting)")
    char_start: int = Field(..., description="Start position in original doc")
    char_end: int = Field(..., description="End position in original doc")

    # Content flags
    starts_with_heading: bool = Field(False, description="Begins with heading?")
    heading: Optional[str] = Field(None, description="Heading text if present")
    contains_code: bool = Field(False, description="Chunk contains code blocks")
    code_languages: Optional[List[str]] = Field(
        None, description="Languages of code blocks"
    )
    contains_table: bool = Field(False, description="Chunk contains markdown tables")
    content_type: str = Field(
        "mixed",
        description="Primary content type: text, code, table, mixed, visual_description",
    )
    visual_asset_ref: Optional[str] = Field(
        None, description="Path to source image file"
    )
    visual_asset_type: Optional[str] = Field(
        None, description="chart|flowchart|diagram|table_image|photo"
    )
    document_version: str = Field(..., description="Version of the processed doc used")
    chunk_version: str = Field(..., description="Chunking version used")
    table_chunk: bool = Field(False, description="Is this an atomic table chunk")
    oversized_chunk: bool = Field(
        False, description="Is this chunk > max target size but allowed as exception"
    )
    tiny_chunk_merged: bool = Field(
        False, description="Was this chunk created by merging tiny chunks"
    )

    # Multi-tenancy
    visibility: str = Field("public", description="Public or private access")
    tenant_id: Optional[str] = Field(None, description="Tenant ID for private docs")

    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ChunkingConfig(BaseModel):
    """
    Configuration for document chunking.
    """

    chunk_size: int = Field(600, description="Target chunk size in tokens")
    overlap: int = Field(125, description="Overlap between chunks in tokens")
    preserve_markdown: bool = Field(True, description="Preserve markdown structure")
    min_chunk_tokens: int = Field(150, description="Minimum tokens per chunk")
    max_chunk_tokens: int = Field(800, description="Maximum tokens per chunk")
    source_version: str = Field("unknown", description="Version of input docs")
    output_version: str = Field("unknown", description="Version of output chunks")

    class Config:
        validate_assignment = True


class ChunkingReport(BaseModel):
    """
    Summary statistics for chunking operation.
    """

    total_documents: int = Field(..., description="Documents processed")
    total_chunks: int = Field(..., description="Chunks generated")
    total_tokens: int = Field(..., description="Sum of all chunk tokens")
    avg_tokens_per_chunk: float = Field(..., description="Average tokens per chunk")
    min_tokens_per_chunk: int = Field(..., description="Minimum tokens in chunk")
    max_tokens_per_chunk: int = Field(..., description="Maximum tokens in chunk")
    coverage_percent: float = Field(..., description="Content coverage %")
    processing_time_seconds: float = Field(..., description="Processing duration")
    chunks_in_target_range: int = Field(0, description="Chunks in target token range")

    class Config:
        json_encoders = {float: lambda v: round(v, 2)}
