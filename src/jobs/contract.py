"""
The API <-> worker contract: queue and task names, retry budget and the ingestion payload.

The API imports this module and src/jobs/queue.py only, never the task implementations, so
the slim API image needs no document-parsing dependencies.
"""
from typing import Optional

from pydantic import BaseModel, model_validator

TASK_NAMESPACE = "nexus"
INGEST_TASK_NAME = "ingest_document"
INGEST_TASK = f"{TASK_NAMESPACE}:{INGEST_TASK_NAME}"
INGEST_QUEUE = "ingest"

# Sweeps for ingestion jobs orphaned by a dead worker (src/jobs/recovery.py). Procrastinate cron
# has one-minute granularity, so this is its finest setting.
RECOVERY_TASK_NAME = "recover_stalled_ingestions"
RECOVERY_CRON = "* * * * *"

# An ingestion runs at most 1 + MAX_RETRIES times: transient failures (network, database,
# a worker killed mid-job) get two more chances; a document that keeps failing is then
# marked failed instead of retrying forever.
MAX_RETRIES = 2


class IngestionRequest(BaseModel):
    """What the worker needs to run one ingestion. Uploaded bytes wait in ingest_sources."""

    job_id: str
    doc_id: str
    tenant_id: str
    url: Optional[str] = None
    filename: Optional[str] = None
    content_hash: Optional[str] = None  # uploads are hashed by the API before queueing
    extract_visuals: bool = False
    resume: bool = False

    @model_validator(mode="after")
    def exactly_one_source(self):
        if bool(self.url) == bool(self.filename):
            raise ValueError("An ingestion has exactly one source: a url or an uploaded filename.")
        return self

    @property
    def source_ref(self) -> str:
        """The stable identifier recorded on the document: the URL, or upload://<doc_id>/<filename>."""
        return self.url or f"upload://{self.doc_id}/{self.filename}"
