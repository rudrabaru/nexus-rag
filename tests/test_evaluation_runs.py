"""Specs, datasets and run rows."""
import pytest
from pydantic import ValidationError
from src.evaluation.dataset import EvaluationQuery, integrity_problems
from src.evaluation.engine import run_query
from src.evaluation.metrics import is_valid, value
from src.evaluation.relevance import EXACT, NONE
from src.evaluation.report import reranker_forensics
from src.evaluation.spec import ExperimentSpec
from src.retrieving.config import RetrievalConfig
from src.retrieving.models import RetrievalResult
from tests.support.evaluation import chunk, query, run


def spec(**overrides):
    fields = {"name": "e", "dataset": "d.json", "tenant_id": "demo", "trials": {"dense": {"strategy": "dense"}, "hybrid": {}}}
    return ExperimentSpec(**{**fields, **overrides})


class FakePipeline:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error
        self.dense = type("D", (), {"embedder": type("E", (), {"cost_usd": staticmethod(lambda t: t * 1e-6)})()})()

    async def run(self, text, tenant_id):
        if self.error:
            raise self.error
        return self.result


def test_the_baseline_defaults_to_the_first_trial_and_must_exist():
    assert spec().baseline == "dense"
    with pytest.raises(ValidationError):
        spec(baseline="missing")


@pytest.mark.parametrize("bad", [{"tenant_id": "ALL tenants"}, {"trials": {}}, {"trials": {"x": {"knob": 1}}}, {"concurrency": 0}])
def test_invalid_specs_are_rejected(bad):
    with pytest.raises(ValidationError):
        spec(**bad)


def test_datasets_reject_duplicate_or_empty_queries():
    assert integrity_problems([]) == ["it contains no queries"]
    problems = integrity_problems([query(), EvaluationQuery(query="Q ", acceptable_documents=["x"])])
    assert problems and "duplicates" in problems[0]


async def test_a_run_row_records_ranks_before_and_after_reranking_and_its_uncached_cost():
    result = RetrievalResult(
        query="q", top_k=2, latency_ms=12.0, embedding_tokens=0, query_embedding_tokens=9, rerank_cost_usd=0.002,
        chunks=[chunk("a", url="https://x/other"), chunk("b")], candidates=[chunk("b"), chunk("a", url="https://x/other")],
    )
    row = await run_query(FakePipeline(result), query(), "demo", None)
    assert (row["rank"], row["first_stage_rank"]) == (2, 1)
    assert [r["match"] for r in row["retrieved"]] == [NONE, EXACT]
    assert row["embedding_tokens"] == 9 and row["embedding_cost_usd"] == pytest.approx(9e-6)  # cached or not
    assert is_valid(row)


async def test_a_failed_retrieval_is_an_invalid_run_not_a_crash():
    row = await run_query(FakePipeline(error=ConnectionError("db gone")), query(), "demo", None)
    assert "db gone" in row["error"] and not is_valid(row)


def test_reranker_forensics_classify_how_the_relevant_chunk_moved():
    runs = [run(1, first_stage_rank=3), run(4, first_stage_rank=2), run(None, first_stage_rank=1), run(2, first_stage_rank=2), run(1)]
    assert reranker_forensics(runs) == {"promoted": 1, "demoted": 1, "unchanged": 1, "lost": 1}


def test_value_rejects_unknown_metrics():
    with pytest.raises(ValueError):
        value(run(1), "ndcg@5")


def test_a_trial_config_is_a_full_retrieval_config():
    assert spec().trials["hybrid"] == RetrievalConfig()
