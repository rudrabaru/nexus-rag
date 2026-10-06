"""The report and the regression gate."""
from src.evaluation import report as report_module
from src.evaluation.report import build_report, gate_failures, regressions


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
