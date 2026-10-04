"""Evaluation integrity: what makes an experiment's numbers evidence, and what must make its gate fail."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from src.config import Settings
from src.evaluation import engine as engine_module
from src.evaluation import report as report_module
from src.evaluation.dataset import EvaluationQuery, integrity_problems
from src.evaluation.engine import run_experiment, run_query
from src.evaluation.generation import GenerationStage
from src.evaluation.metrics import summarize, value
from src.evaluation.report import build_report, gate_failures, regressions
from src.evaluation.significance import compare, decide
from src.evaluation.spec import ExperimentSpec, GenerationSpec
from src.generating.evaluator import JudgeUnavailable
from src.generating.models import ContextWindow
from src.llm.client import LLMCall
from src.retrieving.config import RetrievalConfig
from src.retrieving.models import RetrievalResult, RetrievedChunk
from tests.test_retrieval_pipeline import FakeRetriever, pipeline as retrieval_pipeline


def chunk(chunk_id="a", text="some text", url="https://docs.example/page") -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, source_document="Doc", source_url=url, text=text, similarity_score=0.9, token_count=10, metadata={"source_url": url})


def query(text="q", **extra) -> EvaluationQuery:
    return EvaluationQuery(query=text, acceptable_documents=["page"], **extra)


# ── Latency does not depend on which trial ran first ─────────────────────────

class OnePipeline:
    def __init__(self, result):
        self.result = result
        self.dense = SimpleNamespace(embedder=SimpleNamespace(cost_usd=lambda t: 0.0))

    async def run(self, text, tenant_id):
        return self.result


async def test_a_runs_latency_excludes_the_query_embedding_and_records_it_separately():
    result = RetrievalResult(query="q", top_k=1, latency_ms=120.0, embedding_latency_ms=100.0, chunks=[chunk()])
    row = await run_query(OnePipeline(result), query(), "demo", None)
    assert row["latency_ms"] == pytest.approx(20.0) and row["embedding_latency_ms"] == 100.0


# ── Metrics across different top_k ───────────────────────────────────────────

def test_a_hit_beyond_the_shared_cutoff_is_a_miss_for_the_trial_that_could_see_it():
    deep_trial_run = {"rank": 7}
    assert value(deep_trial_run, "mrr") == pytest.approx(1 / 7)
    assert value(deep_trial_run, "mrr", cutoff=5) == 0.0
    assert value(deep_trial_run, "hit_rate@5", cutoff=5) == 0.0 and value({"rank": 3}, "mrr", cutoff=5) == pytest.approx(1 / 3)


def test_runs_without_context_are_counted_and_left_out_of_the_faithfulness_mean():
    runs = [
        {"query_index": 0, "rank": 1, "latency_ms": 1.0, "faithfulness": 1.0},
        {"query_index": 1, "rank": 1, "latency_ms": 1.0, "empty_context": True},
    ]
    summary = summarize(runs, ["faithfulness"])
    assert summary["faithfulness"] == 1.0 and summary["empty_context"] == 1


def test_the_judge_cost_is_part_of_a_runs_cost():
    runs = [{"query_index": 0, "rank": 1, "latency_ms": 1.0, "generation_cost_usd": 0.01, "judge_cost_usd": 0.02}]
    assert summarize(runs, ["mrr"])["cost_per_query_usd"] == pytest.approx(0.03)


# ── One Holm family ──────────────────────────────────────────────────────────

def test_whether_evidence_is_sufficient_depends_on_how_many_tests_share_the_alpha():
    few_metrics = decide([compare("mrr", "a", "b", [(0, 1)] * 6)])  # one test: 6 differing queries reach p = 2/64
    assert few_metrics[0].verdict == "better"
    crowded = decide([compare(m, "a", "b", [(0, 1)] * 6) for m in ("mrr", "hit_rate@1", "hit_rate@3", "hit_rate@5")])
    assert all(c.verdict.startswith("insufficient evidence") for c in crowded)  # 4 tests need more than 6 differing queries


def test_adjusted_p_values_are_computed_across_metrics_not_within_each():
    comparisons = decide([compare("mrr", "a", "b", [(0, 1)] * 12), compare("hit_rate@5", "a", "b", [(0, 1)] * 12)])
    assert all(c.p_adjusted == pytest.approx(2 * c.p_value) or c.p_adjusted >= c.p_value for c in comparisons)
    assert comparisons[0].p_adjusted == pytest.approx(min(1.0, 2 * comparisons[0].p_value))


# ── Specs and datasets ───────────────────────────────────────────────────────

def test_a_generation_experiment_must_name_its_judge():
    with pytest.raises(ValidationError):
        GenerationSpec()
    assert GenerationSpec(judge={"provider": "groq", "model_name": "openai/gpt-oss-20b"}).model is None


def test_a_blank_acceptable_document_or_heading_is_a_dataset_problem():
    blank_doc = EvaluationQuery(query="q", acceptable_documents=["page", " "])
    blank_heading = EvaluationQuery(query="q2", acceptable_documents=["page"], acceptable_headings=[""])
    assert any("blank" in p for p in integrity_problems([blank_doc]))
    assert any("blank" in p for p in integrity_problems([blank_heading]))
    assert integrity_problems([query()]) == []


# ── The generation stage: judge choice and cache keys ────────────────────────

def stage_for(judge="groq/openai/gpt-oss-20b", model="gemini/gemini-3.5-flash"):
    spec = GenerationSpec(
        model=dict(zip(("provider", "model_name"), model.split("/", 1))),
        judge=dict(zip(("provider", "model_name"), judge.split("/", 1))),
    )
    return GenerationStage(spec, "demo", None, Settings(_env_file=None, gemini_api_key="g", groq_api_key="r"))


def test_a_model_may_not_judge_itself():
    with pytest.raises(ValueError, match="favour its own answers"):
        stage_for(judge="gemini/gemini-3.5-flash", model="gemini/gemini-3.5-flash")


class Stage:
    """A stage whose model calls are scripted and whose caches are dictionaries."""

    def __init__(self):
        self.stage = stage_for()
        self.cache = {}
        self.stage._get = lambda table, key: self.cache.get((table.name, key))
        self.stage._put = lambda table, **values: self.cache.setdefault((table.name, values["cache_key"]), values)
        self.generator_calls, self.judge_calls = 0, 0
        stage = self

        class Generator:
            def call_llm(self, prompt, **kw):
                stage.generator_calls += 1
                return LLMCall(text="an answer", prompt_tokens=5, completion_tokens=3)

        class Judge:
            def call_llm(self, prompt, **kw):
                stage.judge_calls += 1
                return LLMCall(text='{"score": 1.0, "reasoning": "supported"}', cost_usd=0.001)

        self.stage.generator.llm_client = Generator()
        self.stage.judge.llm_client = Judge()

    def run(self, text="some text", chunk_id="a"):
        return self.stage.run("q", RetrievalResult(query="q", top_k=1, latency_ms=1.0, chunks=[chunk(chunk_id, text)]))


def test_an_identical_run_is_answered_and_judged_from_the_caches():
    s = Stage()
    first, second = s.run(), s.run()
    assert (s.generator_calls, s.judge_calls) == (1, 1) and second["generation_cached"] and second["judge_cached"]
    assert first["judge_cost_usd"] == 0.001


def test_changed_chunk_text_under_the_same_chunk_id_is_not_a_cache_hit():
    s = Stage()
    s.run(text="the original text of the chunk")
    s.run(text="the re-ingested, different text")
    assert (s.generator_calls, s.judge_calls) == (2, 2)  # a chunk id can name different text after a re-ingest


def test_changing_the_judge_instrument_or_the_sampling_invalidates_cached_answers_and_verdicts():
    s = Stage()
    s.run()
    s.stage._judge_params = "another-prompt-version;t=0.0;max=4096"
    s.run()
    assert (s.generator_calls, s.judge_calls) == (1, 2)  # the answer is reused, the verdict is not
    s.stage._generation_params = "t=0.9;max=4096"
    s.run()
    assert s.generator_calls == 2


def test_a_run_with_no_context_makes_no_calls_and_says_so():
    s = Stage()
    row = s.stage.run("q", RetrievalResult(query="q", top_k=1, latency_ms=1.0, chunks=[]))
    assert row["empty_context"] is True and (s.generator_calls, s.judge_calls) == (0, 0)


# ── The report and the gate ──────────────────────────────────────────────────

def make_run(index, rank, **extra):
    return {"query_index": index, "rank": rank, "degraded": [], "error": None, "latency_ms": 1.0, **extra}


class FakeStore:
    def __init__(self, runs_by_label, top_ks, status="complete", **spec):
        spec = {"name": "e", "dataset": "d", "tenant_id": "demo", **spec,
                "trials": {label: {"strategy": "dense", "top_k": top_ks[label]} for label in runs_by_label}}
        self.experiment = {
            "experiment_id": "x", "name": "e", "tenant_id": "demo", "dataset_name": "d", "dataset_hash": "h" * 64, "status": status,
            "created_at": "", "finished_at": "", "summary": {}, "spec": spec, "queries": [{} for _ in range(20)],
        }
        self.trials = [{"trial_id": label, "label": label, "index_id": "i", "index_count": 1, "config": spec["trials"][label]} for label in runs_by_label]
        self.runs = runs_by_label

    def get_experiment(self, engine, experiment_id):
        return self.experiment

    def trials_of(self, engine, experiment_id):
        return self.trials

    def runs_of(self, engine, trial_id):
        return self.runs[trial_id]


def report_of(monkeypatch, runs, top_ks=None, **kw):
    fake = FakeStore(runs, top_ks or {label: 5 for label in runs}, **kw)
    monkeypatch.setattr(report_module, "store", fake)
    return build_report(None, "x")


def baseline_runs(n=20):
    return [make_run(i, 1 if i % 2 else 2) for i in range(n)]


def test_a_trial_whose_every_run_is_degraded_fails_the_gate_instead_of_passing_unseen(monkeypatch):
    broken = [make_run(i, None, degraded=["reranker voyage failed"]) for i in range(20)]
    report = report_of(monkeypatch, {"base": baseline_runs(), "broken": broken})
    failures = gate_failures(report)
    assert any("'broken': 0/20 runs are valid" in f for f in failures)
    assert any("could not be compared" in f for f in failures)
    assert regressions(report) == []  # the old gate looked only here, and passed


def test_a_trial_that_lost_a_few_queries_to_errors_still_passes_at_the_default_share(monkeypatch):
    candidate = baseline_runs()
    candidate[0] = make_run(0, None, error="generation failed")
    assert gate_failures(report_of(monkeypatch, {"base": baseline_runs(), "cand": candidate})) == []


def test_an_experiment_that_did_not_finish_fails_the_gate(monkeypatch):
    report = report_of(monkeypatch, {"base": baseline_runs(), "cand": baseline_runs()}, status="paused")
    assert any("paused" in f for f in gate_failures(report))


def test_the_gate_looks_only_at_the_primary_metric(monkeypatch):
    base = [make_run(i, 1) for i in range(20)]
    worse_later = [make_run(i, 2) for i in range(20)]  # mrr 1.0 -> 0.5, and hit_rate@1 1.0 -> 0.0
    report = report_of(monkeypatch, {"base": base, "cand": worse_later}, primary_metric="hit_rate@5")
    assert all(c["metric"] == "hit_rate@5" for c in regressions(report))  # hit_rate@5 did not change at all
    assert regressions(report) == []
    report = report_of(monkeypatch, {"base": base, "cand": worse_later}, primary_metric="mrr")
    assert [c["metric"] for c in regressions(report)] == ["mrr"]
    assert any("significantly worse" in f for f in gate_failures(report))


def test_trials_with_different_top_k_are_compared_at_the_smaller_one(monkeypatch):
    base = [make_run(i, 3) for i in range(20)]
    deep = [make_run(i, 3) for i in range(20)]
    deep[0] = make_run(0, 8)  # a hit beyond the other trial's top_k
    report = report_of(monkeypatch, {"base": base, "deep": deep}, top_ks={"base": 5, "deep": 10})
    assert report["comparison_top_k"] == 5
    mrr = next(c for c in report["comparisons"] if c["metric"] == "mrr")
    assert mrr["differing_queries"] == 1  # query 0: a hit at 3 for base, a miss at the shared cutoff for deep


# ── The engine: outages, crashes and the tenant's chunk count ────────────────

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


# ── The first-stage depth of a hybrid search ─────────────────────────────────

async def test_a_deeper_first_stage_feeds_fusion_without_changing_how_many_results_come_back():
    dense, sparse = FakeRetriever([f"d{i}" for i in range(10)]), FakeRetriever([f"s{i}" for i in range(10)])
    result = await retrieval_pipeline(dense, sparse, strategy="hybrid", top_k=3, fusion_depth=10).run("q", "demo")
    assert dense.calls == [10] and sparse.calls == [10] and len(result.chunks) == 3


async def test_without_a_fusion_depth_the_first_stage_is_as_deep_as_it_always_was():
    dense, sparse = FakeRetriever(["a", "b", "c"]), FakeRetriever(["a", "b", "c"])
    await retrieval_pipeline(dense, sparse, strategy="hybrid", top_k=3).run("q", "demo")
    assert dense.calls == [3] and sparse.calls == [3]


def test_a_fusion_depth_below_the_candidates_it_feeds_is_rejected():
    with pytest.raises(ValidationError):
        RetrievalConfig(strategy="hybrid", top_k=5, fusion_depth=3)
    with pytest.raises(ValidationError):
        RetrievalConfig(strategy="hybrid", top_k=3, reranker="flashrank", rerank_candidates=20, fusion_depth=10)


def test_the_context_window_type_is_unchanged_for_empty_runs():
    assert ContextWindow().included_chunks == []
