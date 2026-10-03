"""
An experiment's report, computed from its stored runs (never from in-memory state, so it can be
rebuilt at any time): per-trial metrics, reranker forensics, and every trial compared with the
baseline under significance tests.
"""
from typing import Dict, List

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
        shared_top_k = min(t["config"]["top_k"] for t in trials)
        baseline = {r["query_index"]: r for r in runs[spec.baseline] if is_valid(r)}
        comparisons = []
        for trial in trials:
            if trial["label"] == spec.baseline:
                continue
            candidate = {r["query_index"]: r for r in runs[trial["label"]] if is_valid(r)}
            shared = sorted(baseline.keys() & candidate.keys())
            for metric in metric_names(shared_top_k, with_generation):
                paired = [(value(baseline[i], metric), value(candidate[i], metric)) for i in shared]
                comparisons.append(compare(metric, spec.baseline, trial["label"], [p for p in paired if None not in p]))
        report["comparisons"] = [c.as_dict() for c in decide(comparisons, spec.alpha)]
    return report


def regressions(report: dict) -> List[dict]:
    return [c for c in report["comparisons"] if c["verdict"] == "worse"]


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
        lines.append(f"[{t['label']}] index {t['index_id']} ({t['index_count']} chunks) | valid {m['valid']}/{m['queries']}"
                     f" (degraded {m['degraded']}, errors {m['errors']})")
        lines.append(f"  {headline}")
        lines.append(f"  latency p50 {_fmt(m['latency_p50_ms'])} ms, p95 {_fmt(m['latency_p95_ms'])} ms | cost/query ${m['cost_per_query_usd']:.6f}")
        if "reranker" in t:
            lines.append(f"  reranker moved the first relevant chunk: {t['reranker']}")
        if details:
            for group, summary in t["by_difficulty"].items():
                lines.append(f"    difficulty {group}: n={summary['valid']} mrr={_fmt(summary['mrr'])}")
    if report["comparisons"]:
        lines += ["", f"Compared with baseline {report['baseline']!r} (paired randomization test, Holm-corrected, alpha {report['alpha']}):"]
        for c in report["comparisons"]:
            lines.append(
                f"  {c['candidate']:<24} {c['metric']:<14} {_fmt(c['baseline_mean'])} -> {_fmt(c['candidate_mean'])}"
                f"  diff {_fmt(c['difference'])}  p={_fmt(c['p_value'])} adj={_fmt(c['p_adjusted'])}"
                f"  n={c['paired_queries']} differ={c['differing_queries']}  {c['verdict']}"
            )
    return "\n".join(lines)
