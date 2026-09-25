"""
Jina is still the active embedding/reranking provider (replaced in a later step), so
its cost is still a fixed rate here. Generation cost is no longer estimated from a
hand-written table — src/generating/llm_client.py computes it per call via
litellm.completion_cost() against litellm's maintained, per-model pricing data, and
callers pass that real figure into MetricsStore.log_query(generation_cost_usd=...).
"""

JINA_EMBEDDING_COST_PER_TOKEN = 0.00000002   # $0.02 per 1M tokens
JINA_RERANK_COST_PER_1K_TOKENS = 0.000015   # approximate
