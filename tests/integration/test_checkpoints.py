"""Embedding checkpoints against real Postgres: halfvec round-trip, idempotent saves, and cleanup with the job."""
import pytest

from src.ingestion.embedding_worker import EmbeddingOutcome
from src.jobs.commit import commit_ingestion
from src.stores.checkpoints import CheckpointStore
from tests.integration.helpers import Stores
from tests.integration.test_job_queue import chunk, request

pytestmark = pytest.mark.usefixtures("clean_tables")


def row(chunk_id, value=0.5, input_hash="h1", tokens=3):
    return {"chunk_id": chunk_id, "input_hash": input_hash, "embedding": [value] * 1024, "tokens": tokens}


@pytest.fixture
def stores(pg_engine):
    stores = Stores(pg_engine)
    stores.register_job("job-1", "doc-1", "https://example.com/doc-1", "web", "tenant-1")
    return stores


def test_saved_vectors_come_back_for_the_same_job_only(pg_engine, stores):
    checkpoints = CheckpointStore(pg_engine)
    checkpoints.save("job-1", [row("c1"), row("c2", 0.25)])

    loaded = checkpoints.load("job-1")
    assert set(loaded) == {"c1", "c2"} and len(loaded["c1"].embedding) == 1024
    assert loaded["c1"].embedding[0] == pytest.approx(0.5, abs=1e-3) and loaded["c2"].tokens == 3
    assert checkpoints.load("job-2") == {}


def test_saving_the_same_chunk_again_replaces_it(pg_engine, stores):
    checkpoints = CheckpointStore(pg_engine)
    checkpoints.save("job-1", [row("c1", input_hash="old")])
    checkpoints.save("job-1", [row("c1", input_hash="new", tokens=9)])

    assert checkpoints.load("job-1")["c1"].input_hash == "new" and checkpoints.load("job-1")["c1"].tokens == 9


def test_committing_the_job_discards_its_checkpoints(pg_engine, stores):
    checkpoints = CheckpointStore(pg_engine)
    checkpoints.save("job-1", [row("c1")])

    commit_ingestion(pg_engine, request(), EmbeddingOutcome(chunks=[chunk("c1")], failed_indices=[], total_chunks=1, error_reason=None))

    assert checkpoints.load("job-1") == {}


def test_failing_the_job_discards_its_checkpoints(pg_engine, stores):
    checkpoints = CheckpointStore(pg_engine)
    checkpoints.save("job-1", [row("c1")])

    stores.fail_job("job-1", "boom")

    assert checkpoints.load("job-1") == {}
