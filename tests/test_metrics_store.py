"""
Cost arithmetic for per-query logging. Persistence of these values (including the provider
column that was once silently dropped) is covered against real Postgres in
tests/integration/test_postgres.py.
"""
import pytest

from src.registry.metrics_store import query_costs


def test_generation_cost_is_the_caller_supplied_value_not_recomputed():
    """Generation cost comes from litellm.completion_cost() at the call site, not a rate table."""
    costs = query_costs(embedding_tokens=0, generation_cost_usd=0.0042, rerank_tokens=0)
    assert costs["generation_cost_usd"] == 0.0042
    assert costs["total_cost_usd"] == 0.0042


def test_total_cost_sums_embedding_generation_and_rerank():
    costs = query_costs(embedding_tokens=1000, generation_cost_usd=0.01, rerank_tokens=1000)
    assert costs["embedding_cost_usd"] > 0 and costs["rerank_cost_usd"] > 0
    assert costs["total_cost_usd"] == pytest.approx(
        costs["embedding_cost_usd"] + costs["generation_cost_usd"] + costs["rerank_cost_usd"]
    )
