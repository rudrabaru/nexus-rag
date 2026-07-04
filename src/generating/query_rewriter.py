import logging
from typing import List, Dict
from src.generating.llm_client import LLMClient
from src.generating.models import GenerationConfig

logger = logging.getLogger(__name__)


class QueryRewriter:
    """
    Rewrites a conversational follow-up query into a standalone search query
    using the chat history. This massively improves retrieval relevance for
    pronoun-heavy or context-dependent queries (e.g., 'how do I configure it?').
    """

    def __init__(self, config: GenerationConfig = None):
        self.config = config or GenerationConfig()
        self.llm_client = LLMClient(self.config)

    def rewrite(self, query: str, history: List[Dict[str, str]]) -> str:
        if not history:
            return query

        history_text = ""
        # Only take the last 6 turns (3 interactions) to prevent huge prompts
        recent_history = history[-6:]
        for msg in recent_history:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            history_text += f"{role}: {content}\n"

        prompt = f"""You are a search query rewriter for a Retrieval-Augmented Generation system.
Your task is to take a conversation history and a new follow-up query, and rewrite the follow-up query into a standalone, comprehensive search query that contains all necessary context (like entity names) from the history.

CRITICAL RULES:
- Do NOT answer the query.
- Do NOT add conversational filler like "Here is the rewritten query".
- ONLY output the rewritten search query text itself.

History:
{history_text}

Follow-up query: {query}
Standalone query:"""

        try:
            # Call LLM
            answer, _, _, _ = self.llm_client.call_llm(prompt)
            rewritten = answer.strip()

            # Clean up if the LLM leaked the prompt template prefix
            if rewritten.lower().startswith("standalone query:"):
                rewritten = rewritten[len("standalone query:") :].strip()

            logger.info(f"Rewrote query from '{query}' to '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.error(f"Failed to rewrite query, falling back to original: {e}")
            return query
