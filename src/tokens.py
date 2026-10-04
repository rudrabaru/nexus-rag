"""
The one token estimator, for the places that have no tokenizer: pacing windows, budgets for
text the chunker did not count, and usage a provider did not report.

Three characters per token over-estimates English prose (about four) and under-estimates dense
code or non-Latin text far less than four would. Over-estimating is the safe direction for every
use: a pacing window stays inside the provider's limit, and a budget is never overshot. Chunks
carry their real tiktoken count (src/chunking/tokenizer.py) and use that instead.
"""
CHARS_PER_TOKEN = 3


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)
