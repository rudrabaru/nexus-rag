from typing import Optional

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: str
    url: str
    title: str
    status: str
    created_at: str
    chunks: int
    error: Optional[str] = None
    stats: dict = {}


class WorkspaceStatsResponse(BaseModel):
    documents_count: int
    total_chunks: int
    total_tokens: int
