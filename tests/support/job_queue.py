"""Helpers shared by the job_queue tests."""
from src.embedding.models import EmbeddedChunk
from src.jobs.contract import IngestionRequest


def chunk(chunk_id, tenant="tenant-1", doc_id="doc-1", text="hello world"):
    return EmbeddedChunk(
        chunk_id=chunk_id, source_url=f"https://example.com/{doc_id}", source_document=f"Doc {doc_id}", title="T",
        heading_path=[], chunk_text=text, token_count=3, 
        document_version="v", chunk_version="v", tenant_id=tenant, doc_id=doc_id,
        embedding=[0.1] * 1024, embedding_model="test-model", index_id="test:test-model",
    )


def request(job_id="job-1", doc_id="doc-1", tenant="tenant-1", url="https://example.com/doc-1", filename=None, resume=False):
    return IngestionRequest(job_id=job_id, doc_id=doc_id, tenant_id=tenant, url=url, filename=filename, resume=resume)
