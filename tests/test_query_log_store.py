"""
Cost arithmetic for per-query logging. Persistence of these values (including the provider
column that was once silently dropped) is covered against real Postgres in
tests/integration/test_postgres.py.
"""
import pytest

from src.stores.query_log import query_costs


def test_generation_cost_is_the_caller_supplied_value_not_recomputed():
    """Every cost comes from the call site (litellm, the embedder, the reranker), not a rate table."""
    costs = query_costs(embedding_cost_usd=0.0, generation_cost_usd=0.0042, rerank_cost_usd=0.0)
    assert costs["generation_cost_usd"] == 0.0042
    assert costs["total_cost_usd"] == 0.0042


def test_total_cost_sums_embedding_generation_and_rerank():
    costs = query_costs(embedding_cost_usd=0.00006, generation_cost_usd=0.01, rerank_cost_usd=0.000015)
    assert costs["embedding_cost_usd"] > 0 and costs["rerank_cost_usd"] > 0
    assert costs["total_cost_usd"] == pytest.approx(
        costs["embedding_cost_usd"] + costs["generation_cost_usd"] + costs["rerank_cost_usd"]
    )
