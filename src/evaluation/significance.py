"""
Is configuration B really better than A on this query set, or is the difference noise?

Paired randomization (sign-flip) test on per-query differences d_i = B_i - A_i: if A and B
were interchangeable, each d_i would be as likely negative as positive, so flipping signs at
random gives the null distribution of |sum d_i|. The p-value is the share of sign patterns at
least as extreme as the observed one. It assumes nothing about the distribution of the metric,
which matters here: per-query hit rates are 0/1 and reciprocal ranks are lumpy.

- Exact when at most MAX_EXACT_QUERIES queries differ (all 2^m patterns), otherwise
  PERMUTATIONS random patterns from a fixed seed, so a report is reproducible.
- Queries where A and B agree carry no information about which is better and do not change
  the statistic; what limits power is m, the number of queries that differ. The smallest
  p-value m differing queries can ever produce is 2 / 2^m, so with m < 6 no difference can
  reach p < 0.05. That case is reported as "insufficient evidence" rather than "no difference".
- Several configurations compared with one baseline are several tests; Holm's step-down
  correction keeps the chance of any false "better/worse" at alpha across one metric's family.
"""
import itertools
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

PERMUTATIONS = 10_000
MAX_EXACT_QUERIES = 13  # 2^13 = 8,192 patterns: cheaper to enumerate than to sample
SEED = 20260929
DEFAULT_ALPHA = 0.05


def smallest_attainable_p(differing: int) -> float:
    return 1.0 if differing == 0 else min(1.0, 2.0 / 2 ** differing)


def paired_randomization_test(differences: Sequence[float]) -> float:
    """Two-sided p-value for 'the mean per-query difference is zero'."""
    nonzero = np.array([d for d in differences if d != 0], dtype=float)
    m = len(nonzero)
    if m == 0:
        return 1.0
    observed = abs(nonzero.sum()) - 1e-12  # tolerance: sums of floats are compared for equality
    if m <= MAX_EXACT_QUERIES:
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=m)))
        return float(np.mean(np.abs(signs @ nonzero) >= observed))
    signs = np.random.default_rng(SEED).choice((-1.0, 1.0), size=(PERMUTATIONS, m))
    extreme = int(np.sum(np.abs(signs @ nonzero) >= observed))
    return (extreme + 1) / (PERMUTATIONS + 1)


def holm(p_values: List[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted, running_max = [0.0] * len(p_values), 0.0
    for position, i in enumerate(order):
        running_max = max(running_max, min(1.0, (len(p_values) - position) * p_values[i]))
        adjusted[i] = running_max
    return adjusted


@dataclass
class Comparison:
    metric: str
    baseline: str
    candidate: str
    paired_queries: int  # valid in both trials
    differing_queries: int
    baseline_mean: Optional[float]
    candidate_mean: Optional[float]
    difference: Optional[float]  # candidate - baseline
    p_value: Optional[float]
    p_adjusted: Optional[float] = None
    verdict: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def compare(metric: str, baseline: str, candidate: str, paired: List[tuple]) -> Comparison:
    """paired: (baseline value, candidate value) per query, both present."""
    if not paired:
        return Comparison(metric, baseline, candidate, 0, 0, None, None, None, None, verdict="no paired queries")
    a = [x for x, _ in paired]
    b = [y for _, y in paired]
    differences = [y - x for x, y in paired]
    return Comparison(
        metric, baseline, candidate,
        paired_queries=len(paired),
        differing_queries=sum(1 for d in differences if d != 0),
        baseline_mean=sum(a) / len(a),
        candidate_mean=sum(b) / len(b),
        difference=sum(differences) / len(differences),
        p_value=paired_randomization_test(differences),
    )


def decide(comparisons: List[Comparison], alpha: float = DEFAULT_ALPHA) -> List[Comparison]:
    """Applies Holm's correction within each metric's family and sets each verdict."""
    families: Dict[str, List[Comparison]] = {}
    for c in comparisons:
        if c.p_value is not None:
            families.setdefault(c.metric, []).append(c)
    for family in families.values():
        for c, adjusted in zip(family, holm([c.p_value for c in family])):
            c.p_adjusted = adjusted
            if smallest_attainable_p(c.differing_queries) > alpha:
                c.verdict = f"insufficient evidence: only {c.differing_queries} queries differ"
            elif adjusted < alpha:
                c.verdict = "better" if c.difference > 0 else "worse"
            else:
                c.verdict = "no significant difference"
    return comparisons
