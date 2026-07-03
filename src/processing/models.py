from pydantic import BaseModel, Field
from typing import List, Optional
from src.crawling.metadata import CrawledDocument


class BlockMetrics(BaseModel):
    is_heading: bool = False
    is_code: bool = False
    is_table: bool = False
    is_list: bool = False
    word_count: int = 0
    link_count: int = 0
    link_density: float = 0.0
    document_frequency: float = 0.0
    position_ratio: float = 0.0
    unique_word_ratio: float = 0.0


class Block(BaseModel):
    content: str
    content_hash: str
    metrics: BlockMetrics = Field(default_factory=BlockMetrics)
    boilerplate_score: float = 0.0
    triggered_signals: List[str] = Field(default_factory=list)
    is_removed: bool = False
    removal_reason: Optional[str] = None


class ProcessingStats(BaseModel):
    words_before: int = 0
    words_after: int = 0
    retention_percentage: float = 0.0
    h1_before: int = 0
    h1_after: int = 0
    h2_before: int = 0
    h2_after: int = 0
    h3_before: int = 0
    h3_after: int = 0
    h4_before: int = 0
    h4_after: int = 0
    code_blocks_before: int = 0
    code_blocks_after: int = 0
    tables_before: int = 0
    tables_after: int = 0
    blocks_removed: int = 0
    blocks_retained: int = 0


class ProcessedDocument(CrawledDocument):
    cleaned_markdown: str = ""
    page_category: str = "guide"
    processing_stats: ProcessingStats = Field(default_factory=ProcessingStats)
    blocks: List[Block] = Field(default_factory=list, exclude=True)
