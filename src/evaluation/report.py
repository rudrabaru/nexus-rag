"""
An experiment's report, computed from its stored runs (never from in-memory state, so it can be
rebuilt at any time): per-trial metrics, reranker forensics, and every trial compared with the
baseline under significance tests.
"""
from typing import Dict, List, Optional

from sqlalchemy.engine import Engine

from src.evaluation import store
from src.evaluation.dataset import SYNTHETIC
from src.evaluation.metrics import by_group, is_valid, metric_names, summarize, value
from src.evaluation.significance import compare, decide
from src.evaluation.spec import ExperimentSpec


def reranker_forensics(runs: List[dict]) -> Dict[str, int]:
    """How reranking moved the first relevant chunk, against the first-stage order it reordered."""
    moved = {"promoted": 0, "demoted": 0, "unchanged": 0, "lost": 0}
    for run in filter(is_valid, runs):
        before, after = run.get("first_stage_rank"), run.get("rank")
        if before is None:
            continue
        if after is None:
            moved["lost"] += 1  # relevant in the pool, pushed out of top_k
        elif after < before:
            moved["promoted"] += 1
        elif after > before:
            moved["demoted"] += 1
        else:
            moved["unchanged"] += 1
    return moved


def build_report(engine: Engine, experiment_id: str) -> dict:
    experiment = store.get_experiment(engine, experiment_id)
    if experiment is None:
        raise ValueError(f"No experiment {experiment_id}")
    spec = ExperimentSpec(**experiment["spec"])
    queries = experiment["queries"]
    with_generation = spec.generation is not None
    trials = store.trials_of(engine, experiment_id)
    runs = {t["label"]: store.runs_of(engine, t["trial_id"]) for t in trials}

    report = {
        "experiment": {k: experiment[k] for k in (
            "experiment_id", "name", "tenant_id", "dataset_name", "dataset_hash", "status", "created_at", "finished_at", "summary"
        )},
        "queries": len(queries),
        "synthetic": all(q.get("origin") == SYNTHETIC for q in queries),
        "relevance": spec.relevance,
        "baseline": spec.baseline,
        "alpha": spec.alpha,
        "primary_metric": spec.primary_metric,
        "min_valid": spec.min_valid,
        "comparison_top_k": None,
        "trials": [],
        "comparisons": [],
    }
    for trial in trials:
        names = metric_names(trial["config"]["top_k"], with_generation)
        entry = {
            "label": trial["label"], "index_id": trial["index_id"], "index_count": trial["index_count"],
            "config": trial["config"], "metrics": summarize(runs[trial["label"]], names),
            "by_difficulty": by_group(runs[trial["label"]], queries, "difficulty", names),
            "by_category": by_group(runs[trial["label"]], queries, "category", names),
        }
        if trial["config"].get("reranker"):
            entry["reranker"] = reranker_forensics(runs[trial["label"]])
        report["trials"].append(entry)

    if spec.baseline in runs:
        # One cutoff for every trial: a hit at rank 7 counts for a top-10 trial and not for a top-5 one.
        shared_top_k = min(t["config"]["top_k"] for t in trials)
        report["comparison_top_k"] = shared_top_k
        baseline = {r["query_index"]: r for r in runs[spec.baseline] if is_valid(r)}
        comparisons = []
        for trial in trials:
            if trial["label"] == spec.baseline:
                continue
            candidate = {r["query_index"]: r for r in runs[trial["label"]] if is_valid(r)}
            shared = sorted(baseline.keys() & candidate.keys())
            for metric in metric_names(shared_top_k, with_generation):
                paired = [(value(baseline[i], metric, shared_top_k), value(candidate[i], metric, shared_top_k)) for i in shared]
                comparisons.append(compare(metric, spec.baseline, trial["label"], [p for p in paired if None not in p]))
        report["comparisons"] = [c.as_dict() for c in decide(comparisons, spec.alpha)]
    return report


def regressions(report: dict) -> List[dict]:
    """Candidates significantly worse than the baseline on the experiment's primary metric."""
    return [c for c in report["comparisons"] if c["verdict"] == "worse" and c["metric"] == report["primary_metric"]]


def gate_failures(report: dict, min_valid: Optional[float] = None) -> List[str]:
    """
    Why a regression gate must not pass. An experiment is only evidence when it finished, enough of every
    trial's queries produced a valid run, and each trial could be compared; a trial that is entirely
    broken has no paired queries, which must read as a failure and not as "no regression".
    """
    threshold = report["min_valid"] if min_valid is None else min_valid
    failures = []
    status = report["experiment"]["status"]
    if status != "complete":
        failures.append(f"the experiment is {status}, not complete")
    for trial in report["trials"]:
        m = trial["metrics"]
        if m["queries"] == 0 or m["valid"] / m["queries"] < threshold:
            failures.append(
                f"trial {trial['label']!r}: {m['valid']}/{m['queries']} runs are valid (degraded {m['degraded']}, "
                f"errors {m['errors']}), below the required {threshold:.0%}"
            )
    compared = {c["candidate"] for c in report["comparisons"] if c["verdict"] != "no paired queries"}
    for trial in report["trials"]:
        if trial["label"] != report["baseline"] and trial["label"] not in compared:
            failures.append(f"trial {trial['label']!r} could not be compared with the baseline: no query was valid in both")
    failures += [
        f"{c['candidate']} is significantly worse than {c['baseline']} on {c['metric']} ({c['baseline_mean']:.4f} -> {c['candidate_mean']:.4f}, p adjusted {c['p_adjusted']:.4f})"
        for c in regressions(report)
    ]
    return failures


def _fmt(x) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.4f}" if abs(x) < 10 else f"{x:.0f}"
    return str(x)


def render(report: dict, details: bool = False) -> str:
    e = report["experiment"]
    lines = [
        f"Experiment {e['name']!r} ({e['experiment_id']}) - {e['status']}",
        f"  tenant {e['tenant_id']} | dataset {e['dataset_name']} (sha256 {e['dataset_hash'][:12]}) | {report['queries']} "
        f"{'SYNTHETIC ' if report['synthetic'] else ''}queries | {report['relevance']}-level relevance",
        f"  conditions: {e.get('summary') or {}}",
        "",
    ]
    for t in report["trials"]:
        m = t["metrics"]
        headline = "  ".join(f"{k}={_fmt(v)}" for k, v in m.items() if k == "mrr" or k.startswith(("hit_rate", "faithfulness")))
        lines.append(f"[{t['label']}] index {t['index_id']} ({t['index_count']} workspace chunks) | valid {m['valid']}/{m['queries']}"
                     f" (degraded {m['degraded']}, errors {m['errors']}, no context {m['empty_context']})")
        lines.append(f"  {headline}")
        lines.append(f"  latency (without query embedding) p50 {_fmt(m['latency_p50_ms'])} ms, p95 {_fmt(m['latency_p95_ms'])} ms | cost/query ${m['cost_per_query_usd']:.6f}")
        if "reranker" in t:
            lines.append(f"  reranker moved the first relevant chunk: {t['reranker']}")
        if details:
            for group, summary in t["by_difficulty"].items():
                lines.append(f"    difficulty {group}: n={summary['valid']} mrr={_fmt(summary['mrr'])}")
    if report["comparisons"]:
        lines += ["", f"Compared with baseline {report['baseline']!r} at top_k {report['comparison_top_k']} (paired randomization test, "
                      f"one Holm family over all {len(report['comparisons'])} comparisons, alpha {report['alpha']}; primary metric {report['primary_metric']}):"]
        for c in report["comparisons"]:
            lines.append(
                f"  {c['candidate']:<24} {c['metric']:<14} {_fmt(c['baseline_mean'])} -> {_fmt(c['candidate_mean'])}"
                f"  diff {_fmt(c['difference'])}  p={_fmt(c['p_value'])} adj={_fmt(c['p_adjusted'])}"
                f"  n={c['paired_queries']} differ={c['differing_queries']}  {c['verdict']}"
            )
    return "\n".join(lines)
