"""
Metadata schemas for crawled documents using Pydantic.
Ensures type safety and consistent data structure across the pipeline.
"""

from __future__ import annotations
from datetime import datetime
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
        ..., description="Depth at which this page was crawled (0=root)"
    )
    crawled_at: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp of crawl"
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


class CrawlConfig(BaseModel):
    """Configuration for the crawler."""

    start_url: str = Field(..., description="Root URL to start crawling")
    max_depth: int = Field(3, description="Maximum crawl depth")
    max_pages: int = Field(300, description="Maximum pages to crawl")
    soft_limit_pages: int = Field(1000, description="Soft limit for logging")
    warning_limit_pages: int = Field(5000, description="Warning limit threshold")
    abort_limit_pages: int = Field(10000, description="Hard limit to abort crawl")
    allowed_domains: List[str] = Field(
        default_factory=list,
        description="List of allowed domains (if empty, use start_url domain)",
    )
    exclude_patterns: List[str] = Field(
        default_factory=lambda: [r".*\.pdf$", r".*\.jpg$", r".*\.png$", r".*\?page="],
        description="URL patterns to exclude (regex)",
    )
    timeout_seconds: int = Field(30, description="HTTP request timeout")
    delay_seconds: float = Field(
        0.5, description="Delay between requests to avoid rate limiting"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "start_url": "https://example.com/docs/overview",
                "max_depth": 3,
                "max_pages": 300,
                "allowed_domains": ["example.com"],
                "exclude_patterns": [r".*\.pdf$", r".*\?page="],
                "timeout_seconds": 30,
                "delay_seconds": 0.5,
            }
        }


class CrawlMetrics(BaseModel):
    """Metrics from a crawl run."""

    total_urls_attempted: int = 0
    total_urls_crawled: int = 0
    total_urls_failed: int = 0
    total_urls_filtered: int = 0
    total_words: int = 0
    crawl_duration_seconds: float = 0.0
    average_page_words: float = 0.0

    def __str__(self):
        return f"""
Crawl Metrics:
  - URLs Attempted: {self.total_urls_attempted}
  - URLs Successfully Crawled: {self.total_urls_crawled}
  - URLs Failed: {self.total_urls_failed}
  - URLs Filtered Out: {self.total_urls_filtered}
  - Total Words Extracted: {self.total_words:,}
  - Duration: {self.crawl_duration_seconds:.1f}s
  - Avg Words/Page: {self.average_page_words:.0f}
        """


class CrawlFailure(BaseModel):
    """Schema for a failed crawl attempt."""

    url: str
    error: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class CrawlManifestEntry(BaseModel):
    """Schema for a successful crawl attempt in the manifest."""

    url: str
    status: str = "success"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
