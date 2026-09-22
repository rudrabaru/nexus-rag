"""
Regression test: MetricsStore.log_query accepted a `provider` argument that was silently
dropped before storage (the column did not exist), discovered while wiring the real
LiteLLM-based provider into src/generating/llm_client.py. Uses a real in-memory registry
so the schema (src/registry/database.py) is the actual source of truth, not a copy of it.
"""
from src.registry.database import DocumentRegistry
from src.registry.metrics_store import MetricsStore


def make_store():
    registry = DocumentRegistry(db_path=":memory:")
    return MetricsStore(registry._get_conn), registry


def test_provider_is_persisted_not_silently_dropped():
    store, registry = make_store()
    log_id = store.log_query(
        tenant_id="t1", query="q", latency_ms=100.0, tokens_used=10,
        faithfulness_score=None, details={}, provider="groq",
    )
    with registry._get_conn() as conn:
        row = conn.execute("SELECT provider FROM observability_logs WHERE log_id = ?", (log_id,)).fetchone()
    assert row["provider"] == "groq"


def test_generation_cost_is_the_caller_supplied_value_not_recomputed():
    """Generation cost now comes from litellm.completion_cost() at the call site, not
    from a rate table keyed by provider name — the same provider must not silently
    change the stored cost based on an internal table."""
    store, registry = make_store()
    log_id = store.log_query(
        tenant_id="t1", query="q", latency_ms=100.0, tokens_used=10,
        faithfulness_score=None, details={}, generation_cost_usd=0.0042, provider="gemini",
    )
    with registry._get_conn() as conn:
        row = conn.execute(
            "SELECT generation_cost_usd, total_cost_usd FROM observability_logs WHERE log_id = ?", (log_id,)
        ).fetchone()
    assert row["generation_cost_usd"] == 0.0042
    assert row["total_cost_usd"] == 0.0042  # no embedding/rerank tokens in this call


def test_total_cost_sums_embedding_generation_and_rerank():
    store, registry = make_store()
    log_id = store.log_query(
        tenant_id="t1", query="q", latency_ms=100.0, tokens_used=10, faithfulness_score=None,
        details={}, embedding_tokens=1000, generation_cost_usd=0.01, rerank_tokens=1000,
    )
    with registry._get_conn() as conn:
        row = conn.execute(
            "SELECT embedding_cost_usd, generation_cost_usd, rerank_cost_usd, total_cost_usd "
            "FROM observability_logs WHERE log_id = ?",
            (log_id,),
        ).fetchone()
    expected_total = row["embedding_cost_usd"] + row["generation_cost_usd"] + row["rerank_cost_usd"]
    assert row["total_cost_usd"] == expected_total
    assert row["generation_cost_usd"] == 0.01
