from typing import Optional

from pydantic import BaseModel, Field


class BlockMetrics(BaseModel):
    is_heading: bool = False
    is_code: bool = False
    is_table: bool = False
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
    is_removed: bool = False
    removal_reason: Optional[str] = None
