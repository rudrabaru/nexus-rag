JINA_EMBEDDING_COST_PER_TOKEN = 0.00000002   # $0.02 per 1M tokens
JINA_RERANK_COST_PER_1K_TOKENS = 0.000015   # approximate

COST_TABLE = {
    "gemini": {
        "input_cost_per_1k": 0.000075, # gemini-2.0-flash-lite
        "output_cost_per_1k": 0.0003
    },
    "groq": {
        "input_cost_per_1k": 0.0005, # llama-3.3-70b-versatile approx
        "output_cost_per_1k": 0.0008
    }
}
