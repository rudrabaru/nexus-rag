from typing import Optional

from pydantic import BaseModel


class IngestResponse(BaseModel):
    job_id: str
    status: str  # queued | complete (the same content was already indexed)
    warning: Optional[str] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress_pct: int
    error: Optional[str] = None
    doc_id: Optional[str] = None
    chunk_count: Optional[int] = None
    metadata: Optional[dict] = None
