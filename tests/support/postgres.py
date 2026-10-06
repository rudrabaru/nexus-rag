"""Helpers shared by the postgres tests."""
import numpy as np
from src.embedding.models import EmbeddedChunk
from src.db.schema import EMBEDDING_DIMENSION
from src.embedding.embedder import EmbeddingBatch


def unit_vector(i: int, j: int = None) -> list:
    """A unit vector along axis i, or between axes i and j, so nearest neighbours are known."""
    v = np.zeros(EMBEDDING_DIMENSION)
    v[i] = 1.0
    if j is not None:
        v[j] = 1.0
    return list(v / np.linalg.norm(v))


TEST_INDEX = "test:test-model"


def chunk(chunk_id, tenant="tenant-1", doc_id="doc-1", chunk_text="hello world", vector=None, index_id=TEST_INDEX):
    return EmbeddedChunk(
        chunk_id=chunk_id, source_url=f"https://example.com/{doc_id}", source_document=f"Doc {doc_id}", title="T",
        heading_path=["Guide", "Setup"], chunk_text=chunk_text, token_count=3, 
        document_version="v", chunk_version="v", tenant_id=tenant, doc_id=doc_id,
        embedding=vector or unit_vector(0), embedding_model=index_id.split(":")[1], index_id=index_id,
    )


def add_document(registry, doc_id="doc-1", tenant="tenant-1", job_id=None):
    registry.register_job(job_id or f"job-{doc_id}", doc_id, f"https://example.com/{doc_id}", "web", tenant)


class AxisEmbedder:
    """Embeds every query as unit_vector(0), in the test index; no network."""
    index_id, model = TEST_INDEX, "test-model"

    async def aembed(self, texts, input_type):
        return EmbeddingBatch(vectors=[unit_vector(0) for _ in texts], tokens=len(texts))

    def cost_usd(self, tokens):
        return 0.0
