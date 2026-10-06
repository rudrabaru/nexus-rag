"""The engine: latency, outages, crashes and the tenant chunk count."""
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
from src.evaluation import engine as engine_module
from src.evaluation.engine import run_experiment, run_query
from src.evaluation.spec import ExperimentSpec
from src.generating.evaluator import JudgeUnavailable
from src.retrieving.models import RetrievalResult
from tests.support.eval_integrity import chunk, query


class OnePipeline:
    def __init__(self, result):
        self.result = result
        self.dense = SimpleNamespace(embedder=SimpleNamespace(cost_usd=lambda t: 0.0))

    async def run(self, text, tenant_id):
        return self.result


class EngineStore:
    def __init__(self, queries=6):
        self.experiment = {
            "spec": ExperimentSpec(
                name="e", dataset="d", tenant_id="demo", trials={"t": {"strategy": "dense"}}, concurrency=1,
                generation={"judge": {"provider": "groq", "model_name": "openai/gpt-oss-20b"}},
            ).model_dump(mode="json"),
            "queries": [query(f"q{i}").model_dump() for i in range(queries)],
        }
        self.saved, self.statuses, self.count_args = [], [], []

    def get_experiment(self, engine, experiment_id):
        return self.experiment

    def set_status(self, engine, experiment_id, status, summary=None):
        self.statuses.append((status, summary))

    def ensure_trial(self, engine, experiment_id, label, config, index_id, index_count):
        return {"trial_id": "trial", "index_count": index_count}

    def completed_queries(self, engine, trial_id):
        return set()

    def save_run(self, engine, trial_id, query_index, row):
        self.saved.append(query_index)


@pytest.fixture
def engine_env(monkeypatch):
    store = EngineStore()
    monkeypatch.setattr(engine_module, "store", store)

    def count(tenant_id=None):
        store.count_args.append(tenant_id)
        return 3

    chunk_store = SimpleNamespace(index_id="i", get_collection_size=count)
    pipeline = OnePipeline(RetrievalResult(query="q", top_k=1, latency_ms=1.0, chunks=[chunk()]))
    pipeline.dense.chunk_store = chunk_store
    monkeypatch.setattr(engine_module, "build_pipeline", lambda config, resources: pipeline)
    stage = MagicMock()
    stage.model, stage.judge.model, stage.counters = "m", "j", {}
    monkeypatch.setattr(engine_module, "GenerationStage", lambda *a, **k: stage)
    return store, stage


async def test_a_runs_latency_excludes_the_query_embedding_and_records_it_separately():
    result = RetrievalResult(query="q", top_k=1, latency_ms=120.0, embedding_latency_ms=100.0, chunks=[chunk()])
    row = await run_query(OnePipeline(result), query(), "demo", None)
    assert row["latency_ms"] == pytest.approx(20.0) and row["embedding_latency_ms"] == 100.0


async def test_a_judge_outage_stops_scheduling_after_the_first_failure_and_pauses(engine_env):
    store, stage = engine_env
    stage.run.side_effect = JudgeUnavailable("RateLimitError")
    assert await run_experiment(None, None, None, "x", progress=lambda m: None) == "paused"
    assert stage.run.call_count == 1 and store.saved == []  # the other five queries never called the dead judge
    assert store.statuses[-1][0] == "paused"


async def test_a_judge_that_can_never_work_fails_the_experiment_instead_of_pausing_it(engine_env):
    store, stage = engine_env
    stage.run.side_effect = JudgeUnavailable("NotFoundError", permanent=True)
    assert await run_experiment(None, None, None, "x", progress=lambda m: None) == "failed"
    status, summary = store.statuses[-1]
    assert status == "failed" and "NotFoundError" in summary["error"]


async def test_a_crash_leaves_the_experiment_failed_not_running_forever(engine_env):
    store, stage = engine_env
    stage.run.return_value = {}
    store.save_run = MagicMock(side_effect=RuntimeError("database gone"))
    with pytest.raises(RuntimeError):
        await run_experiment(None, None, None, "x", progress=lambda m: None)
    status, summary = store.statuses[-1]
    assert status == "failed" and "database gone" in summary["error"]


async def test_the_workspace_chunk_count_is_the_tenants_own(engine_env):
    store, stage = engine_env
    stage.run.return_value = {}
    assert await run_experiment(None, None, None, "x", progress=lambda m: None) == "complete"
    assert store.count_args == ["demo"]
