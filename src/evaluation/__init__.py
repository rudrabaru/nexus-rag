"""
Evaluation: run retrieval (and optionally generation) configurations over a frozen query set,
store every per-query result in Postgres, and compare configurations with significance tests.

    python -m src.evaluation run spec.json        # see docs/phases/phase6_evaluation.md
    python -m src.evaluation report <experiment_id>
"""
