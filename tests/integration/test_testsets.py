"""Test sets against real Postgres: reading a tenant's chunks, the ground-truth check, and chunk-level relevance end to end."""
import pytest

from src.config import get_settings
from src.evaluation import store
from src.evaluation.dataset import Dataset, EvaluationQuery
from src.evaluation.engine import run_experiment
from src.evaluation.ground_truth import missing_chunk_ids
from src.evaluation.report import build_report
from src.evaluation.spec import ExperimentSpec
from src.registry.database import DocumentRegistry
from src.retrieving.chunk_writes import write_chunks
from src.retrieving.pipeline import RetrievalResources
from src.testsets.sampling import group_identical, load_chunks
from tests.integration.test_postgres import TEST_INDEX, AxisEmbedder, add_document, chunk, unit_vector

pytestmark = pytest.mark.usefixtures("clean_tables")


@pytest.fixture
def corpus(pg_engine):
    registry = DocumentRegistry(pg_engine)
    for doc_id, tenant in (("doc-1", "demo"), ("doc-2", "demo"), ("doc-x", "other")):
        add_document(registry, doc_id=doc_id, tenant=tenant)
    with pg_engine.begin() as conn:
        write_chunks(conn, [
            chunk("k1", tenant="demo", doc_id="doc-1", chunk_text="rotate the signing keys", vector=unit_vector(0)),
            chunk("k2", tenant="demo", doc_id="doc-2", chunk_text="rotate the signing keys", vector=unit_vector(0, 1)),
            chunk("k3", tenant="demo", doc_id="doc-2", chunk_text="the quarterly audit", vector=unit_vector(2)),
            chunk("k4", tenant="other", doc_id="doc-x", chunk_text="another tenant's text", vector=unit_vector(3)),
            chunk("k1", tenant="demo", doc_id="doc-1", chunk_text="rotate the signing keys", vector=unit_vector(5),
                  index_id="test:second-model"),
        ])


def test_chunks_are_read_per_tenant_and_index_and_identical_texts_are_grouped(pg_engine, corpus):
    loaded = load_chunks(pg_engine, "demo", TEST_INDEX)
    assert [c.chunk_id for c in loaded] == ["k1", "k2", "k3"]  # not the other tenant's, not the second index's copy
    assert loaded[0].heading_path == ("Guide", "Setup") and loaded[0].source == "https://example.com/doc-1"

    groups = {g.chunk_ids[0]: g for g in group_identical(loaded)}
    assert sorted(groups) == ["k1", "k3"]
    assert groups["k1"].chunk_ids == ["k1", "k2"]
    assert groups["k1"].documents == ["https://example.com/doc-1", "https://example.com/doc-2"]


def test_missing_chunk_ids_are_found_within_the_tenant_only(pg_engine, corpus):
    assert missing_chunk_ids(pg_engine, "demo", ["k1", "k3"]) == []
    assert missing_chunk_ids(pg_engine, "demo", ["k1", "gone", "k4"]) == ["gone", "k4"]  # k4 belongs to another tenant
    assert missing_chunk_ids(pg_engine, "demo", []) == []


async def test_chunk_level_relevance_is_stricter_than_document_level_on_the_same_query(
    pg_engine, pg_async_engine, corpus, monkeypatch
):
    monkeypatch.setattr("src.retrieving.pipeline.build_embedder", lambda settings, index_id=None: AxisEmbedder())
    resources = RetrievalResources(get_settings(), pg_engine, pg_async_engine)
    # Every query embeds to the same vector, so dense ranks k1 (doc-1), then k2, then k3.
    # The question was written from k2; doc-1's k1 has the same text, but the query only names k2 as ground truth.
    queries = [EvaluationQuery(query="how often to rotate keys", acceptable_documents=["doc-1"], source_chunk_ids=["k2"])]

    ranks = {}
    for relevance in ("document", "chunk"):
        spec = ExperimentSpec(name=relevance, dataset="inline", tenant_id="demo", relevance=relevance,
                              trials={"dense": {"strategy": "dense"}})
        experiment_id = store.create_experiment(pg_engine, spec, Dataset(name="inline", content_hash="h", queries=queries))
        await run_experiment(pg_engine, resources, get_settings(), experiment_id, progress=lambda _: None)
        report = build_report(pg_engine, experiment_id)
        assert report["relevance"] == relevance
        ranks[relevance] = store.runs_of(pg_engine, store.trials_of(pg_engine, experiment_id)[0]["trial_id"])[0]["rank"]

    assert ranks == {"document": 1, "chunk": 2}
