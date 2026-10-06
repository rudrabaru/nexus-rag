"""Relevance, metrics and significance."""
import numpy as np
import pytest
from pydantic import ValidationError
from src.evaluation.dataset import EvaluationQuery, integrity_problems
from src.evaluation.metrics import metric_names, percentile, summarize
from src.evaluation.relevance import EXACT, NONE, PARTIAL, judge, match
from src.evaluation.significance import (
    compare, decide, holm, paired_randomization_test, smallest_attainable_p,
)
from src.evaluation.spec import ExperimentSpec
from src.retrieving.models import RetrievedChunk
from tests.support.evaluation import chunk, query, run


def test_a_document_matches_by_substring_or_by_token_run():
    assert match(chunk("c"), query(["3.13.html"])) == EXACT
    assert match(chunk("c", url="https://example.com/whats-new/3-13"), query(["whats new 3.13"])) == EXACT
    assert match(chunk("c", url="https://example.com/3.12.html"), query(["3.13.html"])) == NONE


def test_a_heading_constraint_splits_exact_from_partial():
    q = query(headings=["PEP 594: Remove"])
    assert match(chunk("c", path=["What's New", "PEP 594: Remove dead batteries"]), q) == EXACT
    assert match(chunk("c", section="PEP 594: Removed modules"), q) == PARTIAL  # "remove" != "removed"
    assert match(chunk("c", path=["Other"]), q) == PARTIAL


def test_a_chunk_without_a_url_falls_back_to_its_document_name():
    upload = RetrievedChunk(chunk_id="c", source_document="notes-3.13.html", text="t", similarity_score=1, metadata={"source_url": None})
    assert match(upload, query()) == EXACT


def test_judgement_ranks_the_first_relevant_chunk_and_labels_each_match():
    q = query(headings=["Setup"])
    j = judge([chunk("a", url="https://x/other"), chunk("b", path=["Intro"]), chunk("c", path=["Setup"])], q)
    assert j.matches == [NONE, PARTIAL, EXACT] and j.rank == 2


def test_chunk_level_relevance_counts_only_the_named_source_chunks():
    q = query(["3.13.html"], source_chunk_ids=["c2"])
    j = judge([chunk("c1"), chunk("c2"), chunk("c3")], q, relevance="chunk")
    assert j.matches == [NONE, EXACT, NONE] and j.rank == 2
    assert judge([chunk("c1")], q).rank == 1  # document-level relevance still accepts the right document


def test_chunk_level_relevance_needs_ground_truth_on_every_query():
    with_ids, without = query(source_chunk_ids=["c1"]), EvaluationQuery(query="other", acceptable_documents=["d"])
    assert integrity_problems([with_ids, without]) == []
    assert any("source_chunk_ids" in p for p in integrity_problems([with_ids, without], relevance="chunk"))


def test_an_experiment_defaults_to_document_relevance_and_rejects_unknown_modes():
    base = {"name": "x", "dataset": "d.json", "tenant_id": "demo", "trials": {"a": {}}}
    assert ExperimentSpec(**base).relevance == "document"
    assert ExperimentSpec(**base, relevance="chunk").relevance == "chunk"
    with pytest.raises(ValidationError):
        ExperimentSpec(**base, relevance="paragraph")


def test_hit_rates_and_mrr_count_only_valid_runs():
    runs = [run(1), run(3), run(None), run(1, degraded=["dense failed"]), run(1, error="generation failed")]
    summary = summarize(runs, metric_names(5, False))
    assert (summary["queries"], summary["valid"], summary["degraded"], summary["errors"]) == (5, 3, 1, 1)
    assert summary["hit_rate@1"] == pytest.approx(1 / 3)
    assert summary["hit_rate@3"] == pytest.approx(2 / 3)
    assert summary["mrr"] == pytest.approx((1 + 1 / 3 + 0) / 3)


def test_hit_rates_beyond_top_k_are_not_reported():
    assert metric_names(3, False) == ["mrr", "hit_rate@1", "hit_rate@3"]
    assert metric_names(7, True) == ["mrr", "hit_rate@1", "hit_rate@3", "hit_rate@5", "hit_rate@7", "faithfulness"]


def test_percentiles_are_observed_values():
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.0 and percentile([5.0], 0.95) == 5.0 and percentile([], 0.5) == 0.0


def test_hit_rate_and_mrr_match_ranx():
    from ranx import Qrels, Run, evaluate

    ranks = {"q1": 1, "q2": 3, "q3": None, "q4": 2}
    qrels, retrieved = {}, {}
    for q, rank in ranks.items():
        ids = [f"{q}-{i}" for i in range(1, 6)]
        retrieved[q] = {doc: 6.0 - i for i, doc in enumerate(ids)}
        qrels[q] = {ids[rank - 1]: 1} if rank else {"never-retrieved": 1}
    expected = evaluate(Qrels(qrels), Run(retrieved), ["mrr", "hit_rate@1", "hit_rate@3"])
    ours = summarize([run(r) for r in ranks.values()], ["mrr", "hit_rate@1", "hit_rate@3"])
    for metric, score in expected.items():
        assert ours[metric] == pytest.approx(score)


def test_the_exact_test_enumerates_every_sign_pattern():
    assert paired_randomization_test([1, 1, 1, 1, 1, 1]) == pytest.approx(2 / 64)
    assert paired_randomization_test([0, 0, 0]) == 1.0
    assert paired_randomization_test([1, -1]) == 1.0


def test_the_sampled_test_agrees_with_ranx_fisher():
    from ranx.statistical_tests.fisher_randomization_test import fisher_randomization_test

    rng = np.random.default_rng(1)
    control = rng.random(40)
    treatment = control + rng.normal(0.05, 0.2, 40)
    expected, _ = fisher_randomization_test(control, treatment, n_permutations=10_000)
    assert paired_randomization_test(treatment - control) == pytest.approx(expected, abs=0.02)


def test_holm_adjusts_in_step_down_order():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_too_few_differing_queries_is_insufficient_evidence_not_no_difference():
    assert smallest_attainable_p(5) > 0.05 >= smallest_attainable_p(6)
    [c] = decide([compare("mrr", "a", "b", [(0, 1)] * 5 + [(1, 1)] * 30)])
    assert c.verdict.startswith("insufficient evidence: 5 queries differ")


def test_clear_differences_are_called_better_or_worse():
    better, worse = decide([
        compare("mrr", "a", "b", [(0, 1)] * 12),
        compare("mrr", "a", "c", [(1, 0)] * 12),
    ])
    assert (better.verdict, worse.verdict) == ("better", "worse")
    assert better.p_adjusted >= better.p_value
