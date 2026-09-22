"""
Compares an evaluation run against a frozen baseline and exits non-zero on regression.

Usage:
    python -m src.retrieving.baseline_check BASELINE.json CANDIDATE.json [--max-drop 0.0]

Both files are evaluation_metrics.json outputs of src.retrieving.eval_runner.

Threshold rationale (--max-drop, default 0.0): retrieval over a fixed index and dataset is
deterministic, and on the 38-query benchmark one changed query moves Recall@k by 0.026, so any
drop means at least one query got worse. Loosen it deliberately for configurations that call a
non-deterministic service (for example reranking). This is a stop-gap: statistical significance
testing replaces it once the evaluation engine stores per-query results.

Runs are comparable only when dataset, query count, retrieval configuration and index size all match:
metrics measured against a different corpus say nothing about a code change.

Exit codes: 0 = no regression, 1 = regression, 2 = runs are not comparable.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List

GATED_METRICS = ("recall_at_1", "recall_at_3", "recall_at_5", "mrr")
COMPARABLE_CONFIG_KEYS = ("top_k", "use_hybrid", "use_sparse", "use_reranker")


def load_run(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def not_comparable_reasons(baseline: dict, candidate: dict) -> List[str]:
    base_fp = baseline.get("reproducibility_fingerprint", {})
    cand_fp = candidate.get("reproducibility_fingerprint", {})
    reasons = []

    if base_fp.get("dataset_md5") != cand_fp.get("dataset_md5"):
        reasons.append(f"dataset differs (md5 {base_fp.get('dataset_md5')} vs {cand_fp.get('dataset_md5')})")
    if base_fp.get("index_count") != cand_fp.get("index_count"):
        reasons.append(
            f"index size differs ({base_fp.get('index_count')} vs {cand_fp.get('index_count')} chunks); "
            "refresh the baseline after a deliberate corpus change"
        )
    if baseline.get("total_queries") != candidate.get("total_queries"):
        reasons.append(f"query count differs ({baseline.get('total_queries')} vs {candidate.get('total_queries')})")

    base_cfg = base_fp.get("configuration", {})
    cand_cfg = cand_fp.get("configuration", {})
    for key in COMPARABLE_CONFIG_KEYS:
        if base_cfg.get(key) != cand_cfg.get(key):
            reasons.append(f"configuration '{key}' differs ({base_cfg.get(key)} vs {cand_cfg.get(key)})")
    return reasons


def regressions(baseline: dict, candidate: dict, max_drop: float) -> List[str]:
    found = []
    for metric in GATED_METRICS:
        before = baseline["overall_metrics"][metric]
        after = candidate["overall_metrics"][metric]
        if before - after > max_drop + 1e-9:
            found.append(f"{metric}: {before:.4f} -> {after:.4f} (drop {before - after:.4f} > allowed {max_drop})")
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fail when retrieval metrics regress against a baseline.")
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--max-drop", type=float, default=0.0)
    args = parser.parse_args(argv)

    baseline, candidate = load_run(args.baseline), load_run(args.candidate)

    reasons = not_comparable_reasons(baseline, candidate)
    if reasons:
        print("NOT COMPARABLE:\n  " + "\n  ".join(reasons))
        return 2

    problems = regressions(baseline, candidate, args.max_drop)
    if problems:
        print("REGRESSION:\n  " + "\n  ".join(problems))
        return 1

    print("OK: no metric dropped against the baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
