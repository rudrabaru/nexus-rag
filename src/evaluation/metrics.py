"""
Per-query values and their aggregates.

- hit_rate@k: share of queries with a relevant chunk in the top k (success@k). Earlier reports
  called this "Recall@k"; the two coincide only when a query has one relevant source.
- mrr: mean of 1/rank of the first relevant chunk (0 when none).
- faithfulness: the judge's score of the answer against its context (generation runs only).

A run is valid when it ran the configuration as specified: not degraded (Phase 5) and no
generation error. Metrics are computed over valid runs; invalid ones are counted, never mixed in.
A valid run with no context has no answer and no faithfulness score: it is left out of the
faithfulness mean and counted separately (empty_context), so a trial that retrieves nothing cannot
look more faithful than one that answers.

Comparing trials whose top_k differ needs one cutoff for all of them: a hit at rank 7 is a hit for a
top-10 trial and a miss for a top-5 trial, so metrics compared across trials are computed at the
shared (smallest) top_k.
"""
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

HIT_RATE_KS = (1, 3, 5, 10)


def is_valid(run: dict) -> bool:
    return not run.get("degraded") and not run.get("error")


def reciprocal_rank(rank: Optional[int]) -> float:
    return 1.0 / rank if rank else 0.0


def value(run: dict, metric: str, cutoff: Optional[int] = None) -> Optional[float]:
    """One query's value of a metric, or None when the run has no value for it. `cutoff` ignores ranks beyond it."""
    rank = run.get("rank")
    if cutoff is not None and rank and rank > cutoff:
        rank = None
    if metric == "mrr":
        return reciprocal_rank(rank)
    if metric.startswith("hit_rate@"):
        k = int(metric.split("@")[1])
        return 1.0 if rank and rank <= k else 0.0
    if metric == "faithfulness":
        return run.get("faithfulness")
    raise ValueError(f"Unknown metric {metric!r}")


def metric_names(top_k: int, with_generation: bool) -> List[str]:
    ks = sorted({k for k in HIT_RATE_KS if k <= top_k} | {top_k})
    return ["mrr", *(f"hit_rate@{k}" for k in ks)] + (["faithfulness"] if with_generation else [])


def percentile(values: List[float], p: float) -> float:
    """Nearest-rank percentile: an observed value, never an interpolation."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


def run_cost(run: dict) -> float:
    return sum(run.get(k) or 0.0 for k in ("embedding_cost_usd", "rerank_cost_usd", "generation_cost_usd", "judge_cost_usd"))


def summarize(runs: Iterable[dict], metrics: List[str]) -> Dict[str, float]:
    runs = list(runs)
    valid = [r for r in runs if is_valid(r)]
    summary: Dict[str, float] = {
        "queries": len(runs),
        "valid": len(valid),
        "degraded": sum(1 for r in runs if r.get("degraded")),
        "errors": sum(1 for r in runs if r.get("error")),
        "empty_context": sum(1 for r in valid if r.get("empty_context")),
    }
    for metric in metrics:
        values = [v for v in (value(r, metric) for r in valid) if v is not None]
        summary[metric] = sum(values) / len(values) if values else None
    latencies = [r["latency_ms"] for r in valid]
    summary.update(
        latency_mean_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        latency_p50_ms=percentile(latencies, 0.50),
        latency_p95_ms=percentile(latencies, 0.95),
        cost_per_query_usd=sum(run_cost(r) for r in valid) / len(valid) if valid else 0.0,
    )
    return summary


def by_group(runs: Iterable[dict], queries: List[dict], field: str, metrics: List[str]) -> Dict[str, Dict[str, float]]:
    """Summaries per value of a query field (difficulty, category)."""
    groups = defaultdict(list)
    for run in runs:
        groups[queries[run["query_index"]].get(field, "unspecified")].append(run)
    return {group: summarize(members, metrics) for group, members in sorted(groups.items())}
