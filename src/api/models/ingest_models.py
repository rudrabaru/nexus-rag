from pydantic import BaseModel
from typing import Optional

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress_pct: int
    error: Optional[str] = None
    doc_id: Optional[str] = None
    chunk_count: Optional[int] = None
    metadata: Optional[dict] = None
