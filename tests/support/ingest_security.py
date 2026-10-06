"""Helpers shared by the ingest_security tests."""
import socket
from unittest.mock import MagicMock
from src.services.ingestion_service import (
    IngestionService,
)


def resolve_to(monkeypatch, *addresses):
    infos = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, 443)) for a in addresses]
    monkeypatch.setattr("src.crawling.url_policy.socket.getaddrinfo", lambda *args, **kwargs: infos)


class Wired:
    """An IngestionService over mock stores and a mock queue (MagicMock attributes are plain sync callables, as asyncio.to_thread wants)."""

    def __init__(self, quota=0, pending_bytes=0, existing_by_hash=None, pages_fetched_today=0, active_jobs=0):
        self.documents, self.jobs, self.fetches, self.queue = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        self.documents.chunk_count.return_value = quota
        self.documents.get_document_by_hash.return_value = existing_by_hash
        self.jobs.pending_upload_bytes.return_value = pending_bytes
        self.jobs.active_job_count.return_value = active_jobs
        self.fetches.pages_fetched_today.return_value = pages_fetched_today
        self.service = IngestionService(self.queue, self.documents, self.jobs, self.fetches)

    async def submit(self, url=None, file=None, resume=False, tenant="tenant-1"):
        return await self.service.submit(tenant, url, file, resume)
