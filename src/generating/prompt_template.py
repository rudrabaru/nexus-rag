from .models import GenerationConfig

# System-level instruction that frames the LLM's role and constraints.
# Kept deliberately minimal to remain corpus-agnostic.
SYSTEM_PROMPT = """You are a precise technical assistant. Your task is to answer the user's question using ONLY the information provided in the context below.

Rules you must follow:
1. Base your answer exclusively on the provided context. Do not use prior knowledge.
2. When the context contains the answer, cite the source by mentioning the [Source: ...] URL.
3. If the context does not contain enough information to answer the question, respond with:
   "I don't have enough information in the provided context to answer this question."
4. Do not speculate, extrapolate, or invent information not present in the context.
5. Keep your answer concise and structured. Use bullet points or numbered steps where appropriate."""


def build_prompt(
    query: str, context_text: str, config: GenerationConfig, chat_history: list = None
) -> str:
    """
    Construct the full prompt to send to the LLM.

    The prompt has three clearly separated sections:
    1. System instruction (role + grounding rules)
    2. Optional Conversation History
    3. Retrieved context (with source citations embedded)
    4. User question

    Args:
        query: The user's natural language question.
        context_text: Assembled context string from ContextBuilder.
        config: GenerationConfig controlling citation behavior.
        chat_history: Optional list of dicts with 'role' and 'content'.

    Returns:
        A single string prompt ready to pass to any LLM provider.
    """
    citation_instruction = (
        "\nWhen answering, reference the [Source: ...] markers from the context to cite where your information came from."
        if config.cite_sources
        else ""
    )

    history_str = ""
    if chat_history:
        history_str = "\n=== CONVERSATION HISTORY ===\n"
        for msg in chat_history[-5:]:
            role = msg.get("role", "User")
            content = msg.get("content", "")
            history_str += f"{role.capitalize()}: {content}\n"
        history_str += "============================\n"

    prompt = f"""{SYSTEM_PROMPT}{citation_instruction}
{history_str}
=== CONTEXT START ===
{context_text}
=== CONTEXT END ===

Question: {query}

Answer:"""

    return prompt
