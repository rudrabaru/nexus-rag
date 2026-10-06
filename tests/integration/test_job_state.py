"""Job and document state transitions."""
import pytest
from sqlalchemy import select
from src.ingestion.embedding_worker import EmbeddingOutcome
from src.jobs.commit import commit_ingestion
from src.db.schema import tenants
from tests.support.job_queue import chunk, request


pytestmark = pytest.mark.usefixtures("clean_tables")


def test_a_job_that_fails_while_re_ingesting_leaves_the_document_serving_its_old_chunks(pg_engine, registry):
    registry.register_job("job-1", "doc-1", "https://example.com/doc-1", "web", "tenant-1")
    commit_ingestion(pg_engine, request(), EmbeddingOutcome(chunks=[chunk("old")], failed_indices=[], total_chunks=1, error_reason=None))
    registry.register_job("job-2", "doc-1", "https://example.com/doc-1", "web", "tenant-1")

    registry.fail_job("job-2", "reader outage")

    doc = registry.get_document("doc-1")
    assert doc["status"] == "complete" and doc["error"] == "reader outage" and doc["chunk_count"] == 1


def test_a_successful_run_clears_an_earlier_failures_error(pg_engine, registry):
    registry.register_job("job-1", "doc-1", "https://example.com/doc-1", "web", "tenant-1")
    registry.fail_job("job-1", "reader outage")
    registry.register_job("job-2", "doc-1", "https://example.com/doc-1", "web", "tenant-1")

    commit_ingestion(pg_engine, request(job_id="job-2"), EmbeddingOutcome(chunks=[chunk("c")], failed_indices=[], total_chunks=1, error_reason=None))

    doc = registry.get_document("doc-1")
    assert doc["status"] == "complete" and doc["error"] is None


def test_completing_a_job_with_nothing_to_do_completes_its_document_too(registry):
    registry.register_job("job-1", "doc-1", "https://example.com/doc-1", "web", "tenant-1")
    registry.update_job_status("job-1", "complete", 100)
    assert registry.get_document("doc-1")["status"] == "complete"


def test_active_jobs_counts_only_this_tenants_queued_and_running_work(registry):
    registry.register_job("job-1", "doc-1", "https://example.com/1", "web", "tenant-1")
    registry.register_job("job-2", "doc-2", "https://example.com/2", "web", "tenant-1")
    registry.register_job("job-3", "doc-3", "https://example.com/3", "web", "tenant-2")
    registry.update_job_status("job-2", "processing", 10)
    registry.register_job("job-4", "doc-4", "https://example.com/4", "web", "tenant-1")
    registry.fail_job("job-4", "boom")

    assert registry.active_job_count("tenant-1") == 2
    assert registry.active_job_count("tenant-2") == 1


def test_re_registering_refreshes_the_content_hash(registry):
    registry.register_job("job-1", "doc-1", "upload://doc-1/a.txt", "txt", "tenant-1", content_hash="hash-1")
    registry.register_job("job-2", "doc-1", "upload://doc-1/a.txt", "txt", "tenant-1", content_hash="hash-2")
    assert registry.get_document("doc-1")["content_hash"] == "hash-2"


def test_the_provider_reported_token_count_is_what_the_tenant_is_charged(pg_engine, registry):
    registry.register_job("job-1", "doc-1", "https://example.com/doc-1", "web", "tenant-1")
    outcome = EmbeddingOutcome(chunks=[chunk("c1"), chunk("c2")], failed_indices=[], total_chunks=2, error_reason=None, provider_tokens=40)

    commit_ingestion(pg_engine, request(), outcome)

    with pg_engine.connect() as conn:
        assert conn.execute(select(tenants.c.total_embedding_tokens).where(tenants.c.tenant_id == "tenant-1")).scalar_one() == 40
