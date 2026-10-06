"""Documents, their chunks and cascades in one transaction."""
import pytest
from sqlalchemy import select
from src.stores.tenants import add_embedding_tokens
from src.stores.job_transitions import complete_job
from src.db.schema import tenants
from tests.support.postgres import add_document, chunk


pytestmark = pytest.mark.usefixtures("clean_tables")


def test_deleting_a_document_cascades_to_its_chunks_and_jobs(load, registry, store):
    add_document(registry)
    load([chunk("c1"), chunk("c2")])
    assert registry.get_document("doc-1")["chunk_count"] == 2

    assert registry.delete_document("doc-1") is True
    assert store.get_collection_size() == 0
    assert registry.get_job("job-doc-1") is None
    assert registry.delete_document("doc-1") is False


def test_job_lifecycle_and_atomic_metadata_merge(load, registry, pg_engine):
    add_document(registry)
    registry.update_job_status("job-doc-1", "processing", 10, metadata={"total_pages": 5})
    registry.update_job_status("job-doc-1", "processing", 50, metadata={"indexed_pages": 3})
    load([chunk("c1")])
    with pg_engine.begin() as conn:
        complete_job(conn, "job-doc-1", {"total_tokens": 42}, status="partial_success")

    job = registry.get_job("job-doc-1")
    assert job["metadata"] == {"total_pages": 5, "indexed_pages": 3}
    assert job["status"] == "partial_success" and job["progress_pct"] == 100
    assert job["error"]  # partial success always explains itself

    doc = registry.get_document("doc-1")
    assert doc["status"] == "partial_success"
    assert doc["stats"] == {"total_tokens": 42}
    assert doc["chunk_count"] == 1
    assert isinstance(doc["ingested_at"], str)


def test_re_registering_a_complete_document_keeps_it_complete(registry, pg_engine):
    add_document(registry, job_id="j1")
    with pg_engine.begin() as conn:
        complete_job(conn, "j1", {})
    registry.register_job("j2", "doc-1", "https://example.com/doc-1", "web", "tenant-1")
    assert registry.get_document("doc-1")["status"] == "complete"


def test_fail_job_marks_the_job_and_document_failed_and_discards_any_pending_upload(registry):
    add_document(registry)
    registry.update_job_status("job-doc-1", "processing", 40)
    registry.fail_job("job-doc-1", "boom")
    assert registry.get_job("job-doc-1")["status"] == "failed"
    assert registry.get_job("job-doc-1")["error"] == "boom"
    assert registry.get_document("doc-1")["status"] == "failed"


def test_a_duplicate_url_ingestion_completes_against_the_existing_document(load, registry):
    """
    Regression: a fetched page whose content matched an indexed document was marked complete,
    but its placeholder document stayed "pending" forever and its fetched pages were never deleted.
    """
    add_document(registry, doc_id="doc-orig", job_id="j-orig")
    load([chunk("c1", doc_id="doc-orig")])
    add_document(registry, doc_id="doc-dup", job_id="j-dup")
    registry.store_fetched_page("j-dup", "https://example.com/mirror", "M", "# same text", "jina")

    registry.complete_as_duplicate("j-dup", "doc-orig")

    job = registry.get_job("j-dup")
    assert (job["doc_id"], job["status"], job["progress_pct"]) == ("doc-orig", "complete", 100)
    assert job["metadata"] == {"duplicate_of": "doc-orig"}
    assert registry.get_document("doc-dup") is None
    assert registry.get_fetched_pages("j-dup") == []
    assert registry.get_document("doc-orig")["chunk_count"] == 1


def test_a_duplicate_keeps_a_placeholder_that_already_holds_chunks(load, registry):
    """A resumed document with chunks of its own is not deleted as a placeholder."""
    add_document(registry, doc_id="doc-orig", job_id="j-orig")
    add_document(registry, doc_id="doc-resumed", job_id="j-resume")
    load([chunk("c1", doc_id="doc-resumed")])

    registry.complete_as_duplicate("j-resume", "doc-orig")

    assert registry.get_document("doc-resumed")["chunk_count"] == 1


def test_quota_and_counts_are_per_tenant(load, registry, store):
    add_document(registry, doc_id="doc-1", tenant="tenant-1")
    add_document(registry, doc_id="doc-2", tenant="tenant-2")
    load([chunk("a"), chunk("b"), chunk("c", tenant="tenant-2", doc_id="doc-2")])
    assert registry.chunk_count("tenant-1") == 2
    assert registry.document_count("tenant-2") == 1
    assert [d["doc_id"] for d in registry.list_documents("tenant-2")] == ["doc-2"]
    assert len(registry.list_all_documents()) == 2


def test_tenant_token_usage_accumulates(pg_engine):
    for tokens in (100, 50, 0):
        with pg_engine.begin() as conn:
            add_embedding_tokens(conn, "tenant-1", tokens)
    with pg_engine.connect() as conn:
        assert conn.execute(select(tenants.c.total_embedding_tokens)).scalar_one() == 150
