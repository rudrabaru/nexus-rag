"""The upload hand-off, committing an ingestion, and replacing a document atomically."""
import pytest
from sqlalchemy import select
from src.ingestion.embedding_worker import EmbeddingOutcome
from src.jobs.commit import commit_ingestion
from src.db.schema import tenants
from tests.support.job_queue import chunk, request


pytestmark = pytest.mark.usefixtures("clean_tables")


def chunk_of(chunk_id, url, doc_id="doc-1", text="hello world"):
    made = chunk(chunk_id, doc_id=doc_id, text=text)
    made.source_url = url
    return made


def chunk_ids(pg_engine, doc_id="doc-1"):
    from src.db.schema import chunks

    with pg_engine.connect() as conn:
        return set(conn.execute(select(chunks.c.chunk_id).where(chunks.c.doc_id == doc_id)).scalars())


def test_an_uploaded_files_bytes_are_readable_by_the_worker(registry):
    registry.register_job("job-1", "doc-1", "upload://doc-1/a.pdf", "pdf", "tenant-1", upload=("a.pdf", b"%PDF-1.4 fake"))

    filename, content = registry.get_ingest_source("job-1")
    assert (filename, content) == ("a.pdf", b"%PDF-1.4 fake")
    assert registry.pending_upload_bytes("tenant-1") == len(b"%PDF-1.4 fake")


def test_registering_a_job_with_no_upload_leaves_no_ingest_source(registry):
    registry.register_job("job-1", "doc-1", "https://example.com/doc-1", "web", "tenant-1")
    assert registry.get_ingest_source("job-1") is None


def test_failing_a_job_discards_its_pending_upload(registry):
    registry.register_job("job-1", "doc-1", "upload://doc-1/a.pdf", "pdf", "tenant-1", upload=("a.pdf", b"x" * 100))
    registry.fail_job("job-1", "boom")
    assert registry.get_ingest_source("job-1") is None
    assert registry.pending_upload_bytes("tenant-1") == 0


def test_pending_upload_bytes_only_counts_the_tenants_own_uploads(registry):
    registry.register_job("job-1", "doc-1", "upload://doc-1/a.pdf", "pdf", "tenant-1", upload=("a.pdf", b"x" * 10))
    registry.register_job("job-2", "doc-2", "upload://doc-2/b.pdf", "pdf", "tenant-2", upload=("b.pdf", b"y" * 999))
    assert registry.pending_upload_bytes("tenant-1") == 10


def test_commit_ingestion_lands_everything_in_one_transaction(pg_engine, registry):
    registry.register_job("job-1", "doc-1", "upload://doc-1/a.pdf", "pdf", "tenant-1", upload=("a.pdf", b"raw bytes"))
    outcome = EmbeddingOutcome(chunks=[chunk("c1"), chunk("c2")], failed_indices=[], total_chunks=2, error_reason=None)

    commit_ingestion(pg_engine, request(filename="a.pdf", url=None), outcome)

    assert registry.get_document("doc-1")["status"] == "complete"
    assert registry.get_document("doc-1")["chunk_count"] == 2
    assert registry.get_ingest_source("job-1") is None  # the hand-off row is cleaned up
    with pg_engine.connect() as conn:
        assert conn.execute(select(tenants.c.total_embedding_tokens).where(tenants.c.tenant_id == "tenant-1")).scalar_one() == 6


def test_commit_ingestion_discards_the_jobs_fetched_pages(pg_engine, registry):
    registry.register_job("job-1", "doc-1", "https://example.com/doc-1", "web", "tenant-1")
    registry.store_fetched_page("job-1", "https://example.com/doc-1", "T", "# Page", "jina")
    outcome = EmbeddingOutcome(chunks=[chunk("c1")], failed_indices=[], total_chunks=1, error_reason=None)

    commit_ingestion(pg_engine, request(), outcome)

    assert registry.get_fetched_pages("job-1") == []


def test_commit_ingestion_records_partial_success_with_its_reason(pg_engine, registry):
    registry.register_job("job-1", "doc-1", "https://example.com/doc-1", "web", "tenant-1")
    outcome = EmbeddingOutcome(chunks=[chunk("c1")], failed_indices=[1], total_chunks=2, error_reason="rate limited")

    commit_ingestion(pg_engine, request(), outcome)

    doc = registry.get_document("doc-1")
    assert doc["status"] == "partial_success"
    assert doc["error"] == "rate limited"
    assert doc["chunk_count"] == 1


def test_commit_ingestion_rolls_back_completely_on_a_bad_chunk(pg_engine, registry):
    """A chunk missing its tenant/doc (write_chunks raises) must not leave the job half-committed."""
    registry.register_job("job-1", "doc-1", "https://example.com/doc-1", "web", "tenant-1")
    bad = chunk("c1")
    bad.tenant_id = None
    outcome = EmbeddingOutcome(chunks=[bad], failed_indices=[], total_chunks=1, error_reason=None)

    with pytest.raises(ValueError):
        commit_ingestion(pg_engine, request(), outcome)

    assert registry.get_job("job-1")["status"] == "queued"  # unchanged: the transaction rolled back
    with pg_engine.connect() as conn:
        assert conn.execute(select(tenants.c.tenant_id).where(tenants.c.tenant_id == "tenant-1")).first() is None


def test_a_re_ingestion_replaces_the_pages_it_re_read_and_keeps_the_pages_it_did_not(pg_engine, registry):
    """Only pages this run read are replaced: a page skipped by the fetch quota keeps its chunks."""
    registry.register_job("job-1", "doc-1", "https://example.com/doc-1", "sitemap", "tenant-1")
    first = EmbeddingOutcome(
        chunks=[chunk_of("old-a1", "https://example.com/a"), chunk_of("old-a2", "https://example.com/a"),
                chunk_of("old-b1", "https://example.com/b")],
        failed_indices=[], total_chunks=3, error_reason=None,
    )
    commit_ingestion(pg_engine, request(), first)

    registry.register_job("job-2", "doc-1", "https://example.com/doc-1", "sitemap", "tenant-1")
    second = EmbeddingOutcome(chunks=[chunk_of("new-a1", "https://example.com/a")], failed_indices=[], total_chunks=1, error_reason=None)
    commit_ingestion(pg_engine, request(job_id="job-2"), second)

    assert chunk_ids(pg_engine) == {"new-a1", "old-b1"}  # page a replaced, page b untouched


def test_the_old_chunks_serve_until_the_new_run_commits(pg_engine, registry):
    registry.register_job("job-1", "doc-1", "https://example.com/doc-1", "web", "tenant-1")
    commit_ingestion(pg_engine, request(), EmbeddingOutcome(chunks=[chunk("old")], failed_indices=[], total_chunks=1, error_reason=None))

    registry.register_job("job-2", "doc-1", "https://example.com/doc-1", "web", "tenant-1")  # queued, worker not yet run

    assert chunk_ids(pg_engine) == {"old"}
    assert registry.get_document("doc-1")["status"] == "complete"


def test_a_partial_run_never_removes_old_chunks(pg_engine, registry):
    registry.register_job("job-1", "doc-1", "https://example.com/doc-1", "web", "tenant-1")
    commit_ingestion(pg_engine, request(), EmbeddingOutcome(chunks=[chunk("old1"), chunk("old2")], failed_indices=[], total_chunks=2, error_reason=None))

    registry.register_job("job-2", "doc-1", "https://example.com/doc-1", "web", "tenant-1")
    partial = EmbeddingOutcome(chunks=[chunk("new1")], failed_indices=[1], total_chunks=2, error_reason="rate limited")
    commit_ingestion(pg_engine, request(job_id="job-2"), partial)

    assert chunk_ids(pg_engine) == {"old1", "old2", "new1"}


def test_a_resume_adds_to_the_document_and_removes_nothing(pg_engine, registry):
    registry.register_job("job-1", "doc-1", "https://example.com/doc-1", "web", "tenant-1")
    commit_ingestion(pg_engine, request(), EmbeddingOutcome(chunks=[chunk("old")], failed_indices=[], total_chunks=1, error_reason=None))

    registry.register_job("job-2", "doc-1", "https://example.com/doc-1", "web", "tenant-1")
    commit_ingestion(pg_engine, request(job_id="job-2", resume=True), EmbeddingOutcome(chunks=[chunk("new")], failed_indices=[], total_chunks=1, error_reason=None))

    assert chunk_ids(pg_engine) == {"old", "new"}


def test_another_documents_chunks_are_never_touched_by_a_replacement(pg_engine, registry):
    registry.register_job("job-1", "doc-1", "https://example.com/doc-1", "web", "tenant-1")
    registry.register_job("job-9", "doc-9", "https://example.com/doc-9", "web", "tenant-1")
    commit_ingestion(pg_engine, request(job_id="job-9", doc_id="doc-9"), EmbeddingOutcome(chunks=[chunk("other", doc_id="doc-9")], failed_indices=[], total_chunks=1, error_reason=None))
    commit_ingestion(pg_engine, request(), EmbeddingOutcome(chunks=[chunk("mine")], failed_indices=[], total_chunks=1, error_reason=None))

    assert chunk_ids(pg_engine, "doc-9") == {"other"}
