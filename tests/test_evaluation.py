"""The evaluation engine's pure parts: relevance, metrics, significance, specs, datasets and run rows."""
import json

import numpy as np
import pytest
from pydantic import ValidationError

from src.evaluation.dataset import EvaluationQuery, integrity_problems, load_dataset
from src.evaluation.engine import run_query
from src.evaluation.metrics import is_valid, metric_names, percentile, summarize, value
from src.evaluation.relevance import EXACT, NONE, PARTIAL, judge, match
from src.evaluation.report import reranker_forensics
from src.evaluation.significance import (
    compare, decide, holm, paired_randomization_test, smallest_attainable_p,
)
from src.evaluation.spec import ExperimentSpec
from src.retrieving.config import RetrievalConfig
from src.retrieving.models import RetrievalResult, RetrievedChunk


def chunk(chunk_id, url="https://docs.python.org/3/whatsnew/3.13.html", section="", path=None, score=1.0):
    metadata = {"source_url": url, "section_title": section, "heading_path": json.dumps(path or [])}
    return RetrievedChunk(chunk_id=chunk_id, source_document="Doc", text="t", similarity_score=score, metadata=metadata)


def query(documents=("3.13.html",), headings=(), **extra):
    return EvaluationQuery(query="q", acceptable_documents=list(documents), acceptable_headings=list(headings), **extra)


# ── Relevance ─────────────────────────────────────────────────────────────────

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


def test_judgement_ranks_the_first_relevant_and_first_exact_chunk():
    q = query(headings=["Setup"])
    j = judge([chunk("a", url="https://x/other"), chunk("b", path=["Intro"]), chunk("c", path=["Setup"])], q)
    assert j.matches == [NONE, PARTIAL, EXACT] and (j.rank, j.exact_rank) == (2, 3)


# ── Metrics ───────────────────────────────────────────────────────────────────

def run(rank, **extra):
    return {"query_index": 0, "rank": rank, "latency_ms": 10.0, "degraded": [], "error": None, **extra}


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


# ── Significance ──────────────────────────────────────────────────────────────

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
    assert c.verdict == "insufficient evidence: only 5 queries differ"


def test_clear_differences_are_called_better_or_worse():
    better, worse = decide([
        compare("mrr", "a", "b", [(0, 1)] * 12),
        compare("mrr", "a", "c", [(1, 0)] * 12),
    ])
    assert (better.verdict, worse.verdict) == ("better", "worse")
    assert better.p_adjusted >= better.p_value


# ── Specs and datasets ────────────────────────────────────────────────────────

def spec(**overrides):
    fields = {"name": "e", "dataset": "d.json", "tenant_id": "demo", "trials": {"dense": {"strategy": "dense"}, "hybrid": {}}}
    return ExperimentSpec(**{**fields, **overrides})


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


def test_the_legacy_benchmark_format_still_loads(tmp_path):
    legacy = [{"query": "What changed in PEP 594?", "expected_topic": "t", "expected_content_type": "concept",
               "acceptable_documents": ["3.13.html"], "acceptable_headings": ["PEP 594: Remove"],
               "difficulty": "Easy", "category": "technical_docs"}]
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    dataset = load_dataset(str(path))
    assert dataset.queries[0].acceptable_headings == ["PEP 594: Remove"] and len(dataset.content_hash) == 64


# ── Run rows ──────────────────────────────────────────────────────────────────

class FakePipeline:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error
        self.dense = type("D", (), {"embedder": type("E", (), {"cost_usd": staticmethod(lambda t: t * 1e-6)})()})()

    async def run(self, text, tenant_id):
        if self.error:
            raise self.error
        return self.result


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
