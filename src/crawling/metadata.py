"""
Metadata schemas for crawled documents using Pydantic.
Ensures type safety and consistent data structure across the pipeline.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field


class VisualChunkDraft(BaseModel):
    """Draft representation of a visual element and its generated description."""

    text: str = Field(..., description="Generated text description of the visual")
    asset_ref: str = Field(..., description="Path to source image file")
    asset_type: str = Field(
        ..., description="chart|flowchart|diagram|table_image|photo"
    )
    page_number: Optional[int] = Field(None, description="Source page number")


class AdapterResult(BaseModel):
    """Result returned by ingestion adapters."""

    documents: List[CrawledDocument]
    visual_chunks: List[VisualChunkDraft] = Field(default_factory=list)


class CrawledDocument(BaseModel):
    """Schema for a crawled document with metadata."""

    url: str = Field(..., description="Full URL of the crawled page")
    title: Optional[str] = Field(
        None, description="Page title extracted from <title> or <h1>"
    )
    markdown_content: str = Field(
        ..., description="Cleaned markdown body content from page"
    )
    crawl_depth: int = Field(
        0, description="Depth at which this page was crawled (0=root)"
    )
    crawled_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of crawl"
    )
    source_url: Optional[str] = Field(None, description="URL that linked to this page")
    outgoing_links: List[str] = Field(
        default_factory=list, description="Links found on this page"
    )
    word_count: int = Field(0, description="Word count of markdown_content")
    status_code: Optional[int] = Field(
        None, description="HTTP status code when crawled"
    )
    error: Optional[str] = Field(None, description="Error message if crawl failed")
    raw_html: Optional[str] = Field(None, exclude=True, description="Raw HTML content")

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com/docs/overview",
                "title": "Load Balancing Overview",
                "markdown_content": "# Load Balancing Overview\n\nLoad balancing...",
                "crawl_depth": 1,
                "crawled_at": "2026-05-28T10:00:00",
                "outgoing_links": ["https://example.com/docs/features"],
                "word_count": 1500,
                "status_code": 200,
            }
        }

