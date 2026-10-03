"""The evaluation engine end to end on real Postgres, with a fake embedder and a fake LLM (no network)."""
from types import SimpleNamespace

import litellm
import pytest
from sqlalchemy import func, select, update

from src.config import get_settings
from src.evaluation import store
from src.evaluation.dataset import Dataset, EvaluationQuery
from src.evaluation.engine import run_experiment
from src.evaluation.report import build_report
from src.evaluation.spec import ExperimentSpec
from tests.integration.helpers import Stores
from src.db.schema import runs
from src.retrieving.chunk_writes import write_chunks
from src.retrieving.pipeline import RetrievalResources
from tests.integration.test_postgres import TEST_INDEX, AxisEmbedder, add_document, chunk, unit_vector

pytestmark = pytest.mark.usefixtures("clean_tables")

QUERIES = [
    EvaluationQuery(query="signing keys rotation", acceptable_documents=["doc-1"]),
    EvaluationQuery(query="quarterly audit", acceptable_documents=["doc-2"]),
    EvaluationQuery(query="unrelated question", acceptable_documents=["doc-9"]),
]


class CountingEmbedder(AxisEmbedder):
    calls = 0

    async def aembed(self, texts, input_type):
        CountingEmbedder.calls += 1
        return await super().aembed(texts, input_type)


@pytest.fixture
def corpus(pg_engine, monkeypatch):
    registry = Stores(pg_engine)
    add_document(registry, doc_id="doc-1", tenant="demo")
    add_document(registry, doc_id="doc-2", tenant="demo")
    with pg_engine.begin() as conn:
        write_chunks(conn, [
            chunk("k1", tenant="demo", doc_id="doc-1", chunk_text="rotate the signing keys", vector=unit_vector(0)),
            chunk("k2", tenant="demo", doc_id="doc-2", chunk_text="the quarterly audit of access logs", vector=unit_vector(0, 1)),
        ])
    CountingEmbedder.calls = 0
    monkeypatch.setattr("src.retrieving.pipeline.build_embedder", lambda settings, index_id=None: CountingEmbedder())


@pytest.fixture
def resources(pg_engine, pg_async_engine):
    return RetrievalResources(get_settings(), pg_engine, pg_async_engine)


def experiment(pg_engine, trials, generation=None) -> str:
    spec = ExperimentSpec(name="t", dataset="inline", tenant_id="demo", trials=trials, generation=generation)
    return store.create_experiment(pg_engine, spec, Dataset(name="inline", content_hash="h", queries=QUERIES))


def run_count(pg_engine) -> int:
    with pg_engine.connect() as conn:
        return conn.execute(select(func.count()).select_from(runs)).scalar_one()


async def test_an_experiment_stores_every_run_reports_and_a_rerun_does_nothing(pg_engine, corpus, resources):
    experiment_id = experiment(pg_engine, {"dense": {"strategy": "dense"}, "sparse": {"strategy": "sparse"}})

    assert await run_experiment(pg_engine, resources, get_settings(), experiment_id, progress=lambda _: None) == "complete"
    assert run_count(pg_engine) == 6
    assert CountingEmbedder.calls == 3  # one embedding per query; sparse embeds nothing

    report = build_report(pg_engine, experiment_id)
    dense = report["trials"][0]
    assert dense["index_id"] == TEST_INDEX and dense["metrics"]["valid"] == 3
    # Every query embeds to the same vector, so dense ranks k1 (doc-1) first for all of them:
    # q0 hits at 1, q1 (doc-2) at 2, q2 (doc-9, not in the corpus) never.
    assert dense["metrics"]["hit_rate@1"] == pytest.approx(1 / 3)
    assert dense["metrics"]["hit_rate@3"] == pytest.approx(2 / 3)
    assert {c["candidate"] for c in report["comparisons"]} == {"sparse"}

    await run_experiment(pg_engine, resources, get_settings(), experiment_id, progress=lambda _: None)
    assert run_count(pg_engine) == 6 and CountingEmbedder.calls == 3  # resumed: nothing left to run


async def test_resume_retries_only_invalid_runs(pg_engine, corpus, resources):
    experiment_id = experiment(pg_engine, {"dense": {"strategy": "dense"}})
    await run_experiment(pg_engine, resources, get_settings(), experiment_id, progress=lambda _: None)
    with pg_engine.begin() as conn:
        conn.execute(update(runs).where(runs.c.query_index == 1).values(error="generation failed", rank=None))

    trial_id = store.trials_of(pg_engine, experiment_id)[0]["trial_id"]
    assert store.completed_queries(pg_engine, trial_id) == {0, 2}
    await run_experiment(pg_engine, resources, get_settings(), experiment_id, progress=lambda _: None)

    retried = {r["query_index"]: r for r in store.runs_of(pg_engine, trial_id)}[1]
    assert retried["error"] is None and retried["rank"] == 2  # recomputed: doc-2 is second for every query


@pytest.fixture
def fake_llm(monkeypatch):
    """Answers generation prompts, scores judge prompts, and counts both; judge_down simulates an outage."""
    calls = SimpleNamespace(generation=0, judge=0, judge_down=False)

    def completion(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        if "impartial judge" in prompt:
            if calls.judge_down:
                raise litellm.RateLimitError(message="429", llm_provider="groq", model=kwargs["model"])
            calls.judge += 1
            content = '{"score": 1.0, "reasoning": "supported"}'
        else:
            calls.generation += 1
            content = "an answer from the context"
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")], usage=usage)

    monkeypatch.setattr("src.generating.llm_client.litellm.completion", completion)
    monkeypatch.setattr("src.generating.llm_client.litellm.completion_cost", lambda **kw: 0.001)
    monkeypatch.setattr("src.generating.llm_client.time.sleep", lambda *_: None)
    return calls


GENERATION = {"model": {"provider": "gemini", "model_name": "answerer"}, "judge": {"provider": "groq", "model_name": "judge"}}


async def test_identical_prompts_are_generated_and_judged_once(pg_engine, corpus, resources, fake_llm):
    """Two configurations that retrieve the same context share one answer and one verdict per query."""
    same = {"strategy": "dense", "top_k": 2}
    experiment_id = experiment(pg_engine, {"a": same, "b": {**same, "rrf_k": 61}}, GENERATION)

    assert await run_experiment(pg_engine, resources, get_settings(), experiment_id, progress=lambda _: None) == "complete"

    assert (fake_llm.generation, fake_llm.judge) == (3, 3)
    summary = store.get_experiment(pg_engine, experiment_id)["summary"]
    assert summary["cache"] == {"generation_calls": 3, "generation_cache_hits": 3, "judge_calls": 3, "judge_cache_hits": 3}
    assert summary["judge_model"] == "groq/judge"
    b_runs = store.runs_of(pg_engine, store.trials_of(pg_engine, experiment_id)[1]["trial_id"])
    assert all(r["generation_cached"] and r["judge_cached"] and r["faithfulness"] == 1.0 for r in b_runs)
    assert all(r["generation_cost_usd"] == 0.001 for r in b_runs)  # a cached answer still carries its configuration's cost


async def test_a_judge_outage_pauses_the_experiment_and_resume_completes_it_without_duplicates(pg_engine, corpus, resources, fake_llm):
    experiment_id = experiment(pg_engine, {"dense": {"strategy": "dense"}}, GENERATION)
    fake_llm.judge_down = True

    assert await run_experiment(pg_engine, resources, get_settings(), experiment_id, progress=lambda _: None) == "paused"
    assert store.get_experiment(pg_engine, experiment_id)["status"] == "paused"

    fake_llm.judge_down = False
    generated_before_resume = fake_llm.generation
    assert await run_experiment(pg_engine, resources, get_settings(), experiment_id, progress=lambda _: None) == "complete"

    assert run_count(pg_engine) == 3
    assert fake_llm.generation == generated_before_resume  # answers came from the cache, not new calls
    trial_id = store.trials_of(pg_engine, experiment_id)[0]["trial_id"]
    assert all(r["judge_model"] == "groq/judge" and r["faithfulness"] == 1.0 for r in store.runs_of(pg_engine, trial_id))
