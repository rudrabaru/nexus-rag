import json
import logging
from pydantic import BaseModel

from .models import GenerationResult, GenerationConfig

logger = logging.getLogger(__name__)

class EvaluationResponse(BaseModel):
    score: float
    reasoning: str

FAITHFULNESS_PROMPT_TEMPLATE = """You are an impartial judge evaluating the faithfulness of an AI-generated answer.
You will be provided with:
1. CONTEXT: The retrieved knowledge.
2. QUESTION: The user's question.
3. ANSWER: The generated answer.

Your task is to determine if the ANSWER is fully supported by the CONTEXT.
- Score 1.0 if the answer is completely supported by the context.
- Score 0.5 if the answer is partially supported (some facts are supported, some are hallucinated or external).
- Score 0.0 if the answer is entirely hallucinated or contradicts the context.

Respond ONLY with a valid JSON object in the following format:
{{
  "score": 1.0,
  "reasoning": "A brief explanation of your score."
}}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}
"""


class FaithfulnessEvaluator:
    """
    Evaluates whether a generated answer is faithful to the provided context.
    Uses LLM-as-a-judge pattern to score and explain the assessment.
    """

    def __init__(self, config: GenerationConfig = None):
        self.config = config or GenerationConfig()
        from .llm_client import LLMClient
        self.llm_client = LLMClient(self.config)

    def evaluate(self, result: GenerationResult) -> GenerationResult:
        """
        Evaluate the faithfulness of the given generation result.
        Updates and returns the result with faithfulness_score and faithfulness_reasoning.
        """
        prompt = FAITHFULNESS_PROMPT_TEMPLATE.format(
            context=result.context_window.context_text,
            question=result.query,
            answer=result.answer,
        )

        original_temp = self.config.temperature
        self.config.temperature = 0.0
        try:
            answer_text, _, _, _ = self.llm_client.call_llm(
                prompt, response_schema=EvaluationResponse
            )
        finally:
            self.config.temperature = original_temp

        try:
            if not answer_text:
                raise ValueError("LLM returned empty response")
                
            if answer_text.startswith("[Generation failed:"):
                result.faithfulness_score = 0.0
                result.faithfulness_reasoning = answer_text
                return result
                
            try:
                evaluation = json.loads(answer_text.strip())
            except json.JSONDecodeError:
                import re
                match = re.search(r'(\{.*\})', answer_text, re.DOTALL)
                if not match:
                    raise ValueError(f"Could not extract JSON from LLM response: {answer_text}")
                evaluation = json.loads(match.group(1))
            result.faithfulness_score = float(evaluation.get("score", 0.0))
            result.faithfulness_reasoning = evaluation.get(
                "reasoning", "Failed to parse reasoning."
            )
            logger.info(
                f"Faithfulness Evaluator -> Score: {result.faithfulness_score} | Reason: {result.faithfulness_reasoning}"
            )
        except Exception as e:
            logger.error(
                f"Failed to parse faithfulness evaluator output: {e}. Raw output: {answer_text}"
            )
            result.faithfulness_score = 0.0
            result.faithfulness_reasoning = f"Parse error: {str(e)}"

        return result
