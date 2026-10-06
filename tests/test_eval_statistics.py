"""Metrics across different top_k, one Holm family, and the first-stage depth of a hybrid search."""
import pytest
from pydantic import ValidationError
from src.evaluation.metrics import summarize, value
from src.evaluation.significance import compare, decide
from src.generating.models import ContextWindow
from src.retrieving.config import RetrievalConfig
from tests.test_retrieval_pipeline import FakeRetriever, pipeline as retrieval_pipeline


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


def test_whether_evidence_is_sufficient_depends_on_how_many_tests_share_the_alpha():
    few_metrics = decide([compare("mrr", "a", "b", [(0, 1)] * 6)])  # one test: 6 differing queries reach p = 2/64
    assert few_metrics[0].verdict == "better"
    crowded = decide([compare(m, "a", "b", [(0, 1)] * 6) for m in ("mrr", "hit_rate@1", "hit_rate@3", "hit_rate@5")])
    assert all(c.verdict.startswith("insufficient evidence") for c in crowded)  # 4 tests need more than 6 differing queries


def test_adjusted_p_values_are_computed_across_metrics_not_within_each():
    comparisons = decide([compare("mrr", "a", "b", [(0, 1)] * 12), compare("hit_rate@5", "a", "b", [(0, 1)] * 12)])
    assert all(c.p_adjusted == pytest.approx(2 * c.p_value) or c.p_adjusted >= c.p_value for c in comparisons)
    assert comparisons[0].p_adjusted == pytest.approx(min(1.0, 2 * comparisons[0].p_value))


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
