"""
The Jina reranker is still active until item 9, so its cost is a fixed rate here. Generation
cost comes from litellm.completion_cost() per call (src/generating/llm_client.py); embedding
cost from the query embedder's list price (src/embedding/providers.py).
"""

JINA_RERANK_COST_PER_1K_TOKENS = 0.000015   # approximate
