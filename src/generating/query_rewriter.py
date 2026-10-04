import logging
from typing import Dict, List

from pydantic import BaseModel

from src.llm.client import LLMClient
from src.llm.errors import GenerationError
from src.llm.structured import extract_json_object

from .models import GenerationConfig

logger = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 6  # three exchanges: enough to resolve "it" and "that one" without a huge prompt


class RewrittenQuery(BaseModel):
    rewritten_query: str


GENERALISE_PROMPT = """You are a search query optimiser for a retrieval system over a collection of documents.
Rewrite the query below with the general vocabulary a document's author would use for the same idea.
- Replace specific names and proper nouns with the general term for what they are.
- Preserve the intent exactly, and add nothing the query does not ask for.
- Output ONLY a JSON object matching the requested schema.

Query: {query}
"""

REWRITE_PROMPT = """You are a search query rewriter for a Retrieval-Augmented Generation system.
Your task is to take a conversation history and a new follow-up query, and rewrite the follow-up query into a standalone, comprehensive search query that contains all necessary context (like entity names) from the history.

CRITICAL RULES:
- Do NOT answer the query.
- Output ONLY a valid JSON object matching the requested schema.

History:
{history}

Follow-up query: {query}
Standalone query:"""


class QueryRewriter:
    """
    Turns a query into the text that is searched: made standalone using the chat history (a follow-up
    like "how do I configure it?" has no meaning alone), and optionally generalised to the vocabulary
    a document would use. Both are optional steps that change retrieval, so a failure or an empty
    answer always falls back to the original query.

    Generalising is a setting (ENABLE_QUERY_GENERALISATION), off unless chosen, because it has not been measured:
    an experiment retrieves with the query as written, and an unmeasured rewrite would make chat
    serve something the evaluation never tested.
    """

    def __init__(self, config: GenerationConfig = None):
        self.config = config or GenerationConfig()
        self.llm_client = LLMClient(self.config)

    def _ask(self, prompt: str, query: str) -> str:
        try:
            call = self.llm_client.call_llm(prompt, response_schema=RewrittenQuery)
            rewritten = str(extract_json_object(call.text).get("rewritten_query", "")).strip()
        except (GenerationError, ValueError) as e:
            logger.error(f"Query rewrite failed, using the original query: {e}")
            return query
        return rewritten or query

    def generalise(self, query: str) -> str:
        rewritten = self._ask(GENERALISE_PROMPT.format(query=query), query)
        if rewritten != query:
            logger.info(f"Generalised query from '{query}' to '{rewritten}'")
        return rewritten

    def rewrite(self, query: str, history: List[Dict[str, str]]) -> str:
        if not history:
            return query
        lines = [f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}" for m in history[-MAX_HISTORY_MESSAGES:]]
        rewritten = self._ask(REWRITE_PROMPT.format(history="\n".join(lines), query=query), query)
        if rewritten != query:
            logger.info(f"Rewrote query from '{query}' to '{rewritten}'")
        return rewritten
