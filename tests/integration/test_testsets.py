"""Test sets against real Postgres: reading a tenant's chunks, the ground-truth check, and chunk-level relevance end to end."""
import pytest

from src.config import get_settings
from src.evaluation import store
from src.evaluation.dataset import Dataset, EvaluationQuery
from src.evaluation.engine import run_experiment
from src.evaluation.ground_truth import missing_chunk_ids
from src.evaluation.report import build_report
from src.evaluation.spec import ExperimentSpec
from tests.integration.helpers import Stores
from src.retrieving.chunk_writes import write_chunks
from src.retrieving.pipeline import RetrievalResources
from src.testsets.sampling import group_identical, load_chunks
from tests.integration.test_postgres import TEST_INDEX, AxisEmbedder, add_document, chunk, unit_vector

pytestmark = pytest.mark.usefixtures("clean_tables")


@pytest.fixture
def corpus(pg_engine):
    registry = Stores(pg_engine)
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


# ── Test sets in Postgres ────────────────────────────────────────────────────

def draft_with(n=3):
    from src.testsets.models import Draft, DraftItem

    items = [DraftItem(query=f"q{i}?", reference_answer=f"a{i}", acceptable_documents=["https://example.com/doc-1"],
                       acceptable_headings=["Setup"], source_chunk_ids=[f"c{i}"], difficulty="easy", category="prose",
                       origin="synthetic", lexical_overlap=0.5, source_text="the passage") for i in range(n)]
    return Draft(meta={"model": "groq/x", "seed": 0}, items=items, abstained=["nav"])


def test_a_draft_round_trips_through_postgres_with_its_review_state(pg_engine):
    from src.stores.testsets import ACCEPTED, REJECTED, TestSetStore
    from src.testsets.repository import load_draft, save_draft

    store = TestSetStore(pg_engine)
    test_set_id = store.create("demo", "first", {"model": "groq/x", "seed": 0})
    draft = draft_with()
    draft.items[0].review_status, draft.items[1].review_status = ACCEPTED, REJECTED
    save_draft(store, test_set_id, draft)

    loaded = load_draft(store, test_set_id)
    assert [i.query for i in loaded.items] == ["q0?", "q1?", "q2?"]
    assert [i.review_status for i in loaded.items] == [ACCEPTED, REJECTED, "pending"]
    assert loaded.abstained == ["nav"] and loaded.items[0].source_text == "the passage"
    assert loaded.items[0].source_chunk_ids == ["c0"] and loaded.items[0].lexical_overlap == 0.5


def test_freezing_hashes_the_accepted_questions_and_makes_the_set_immutable(pg_engine):
    from src.stores.testsets import ACCEPTED, TestSetError, TestSetStore
    from src.testsets.repository import save_draft

    store = TestSetStore(pg_engine)
    test_set_id = store.create("demo", "first", {})
    draft = draft_with()
    draft.items[0].review_status = ACCEPTED
    save_draft(store, test_set_id, draft)

    digest = store.freeze(test_set_id)

    assert len(digest) == 64 and store.find("demo", "first")["status"] == "frozen"
    with pytest.raises(TestSetError, match="frozen"):
        save_draft(store, test_set_id, draft)
    assert [q["query"] for q in store.accepted_questions(test_set_id)] == ["q0?"]


def test_a_set_with_nothing_accepted_cannot_be_frozen_and_names_are_per_workspace(pg_engine):
    from sqlalchemy.exc import IntegrityError

    from src.stores.testsets import TestSetError, TestSetStore
    from src.testsets.repository import save_draft

    store = TestSetStore(pg_engine)
    test_set_id = store.create("demo", "first", {})
    save_draft(store, test_set_id, draft_with())
    with pytest.raises(TestSetError, match="no accepted"):
        store.freeze(test_set_id)

    store.create("other", "first", {})  # another workspace may reuse the name
    with pytest.raises(IntegrityError):
        store.create("demo", "first", {})


def test_an_experiment_resolves_a_frozen_set_by_name_and_refuses_a_draft(pg_engine):
    from src.evaluation.dataset import resolve_dataset
    from src.stores.testsets import ACCEPTED, TestSetStore
    from src.testsets.repository import save_draft

    store = TestSetStore(pg_engine)
    test_set_id = store.create("demo", "first", {})
    draft = draft_with()
    draft.items[0].review_status = draft.items[1].review_status = ACCEPTED
    save_draft(store, test_set_id, draft)

    with pytest.raises(ValueError, match="draft"):
        resolve_dataset(pg_engine, "demo", "testset:first", "chunk")
    with pytest.raises(ValueError, match="no test set"):
        resolve_dataset(pg_engine, "other", "testset:first", "chunk")

    digest = store.freeze(test_set_id)
    dataset = resolve_dataset(pg_engine, "demo", "testset:first", "chunk")
    assert dataset.content_hash == digest and [q.source_chunk_ids for q in dataset.queries] == [["c0"], ["c1"]]


def test_a_dataset_file_imports_as_a_frozen_set_and_exports_back(pg_engine, tmp_path):
    import json

    from src.stores.testsets import TestSetError, TestSetStore
    from src.testsets.repository import export_dataset, import_dataset

    source = tmp_path / "in.json"
    source.write_text(json.dumps([
        {"query": "how do I rotate keys?", "acceptable_documents": ["https://example.com/doc-1"], "difficulty": "hard"},
    ]), encoding="utf-8")
    store = TestSetStore(pg_engine)

    digest = import_dataset(store, "demo", "imported", str(source))
    assert store.find("demo", "imported")["status"] == "frozen" and len(digest) == 64
    with pytest.raises(TestSetError, match="already has"):
        import_dataset(store, "demo", "imported", str(source))

    out = tmp_path / "out.json"
    assert export_dataset(store, "demo", "imported", str(out)) == 1
    assert json.loads(out.read_text(encoding="utf-8"))[0]["query"] == "how do I rotate keys?"
