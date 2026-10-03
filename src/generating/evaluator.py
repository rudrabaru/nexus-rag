import logging
from typing import Tuple

from pydantic import BaseModel

from .llm_client import LLMClient
from .models import GenerationConfig, GenerationResult
from .structured import extract_json_object

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


class JudgeUnavailable(RuntimeError):
    """The judge model could not be called (rate limit, outage). Retrying later can succeed."""


class JudgeOutputError(ValueError):
    """The judge answered, but not with a usable score."""


def parse_verdict(text: str) -> Tuple[float, str]:
    try:
        evaluation = extract_json_object(text)
    except ValueError as e:
        raise JudgeOutputError(f"unusable judge output: {e}") from e
    try:
        return float(evaluation["score"]), str(evaluation.get("reasoning", ""))
    except (KeyError, TypeError, ValueError) as e:
        raise JudgeOutputError(f"the judge's JSON has no numeric score: {evaluation!r}") from e


class FaithfulnessEvaluator:
    """
    LLM-as-a-judge: is the answer supported by the context it was given?

    judge() is strict and is what evaluations use: a failed call raises JudgeUnavailable and a
    malformed verdict raises JudgeOutputError, so neither is ever recorded as a score.
    evaluate() is the lenient form chat uses: any failure becomes score 0 with the reason.
    """

    def __init__(self, config: GenerationConfig = None):
        # A private copy fixed at temperature 0.0: the judge must be deterministic, and it
        # must never share (or mutate) the generator's config object.
        self.config = (config or GenerationConfig()).model_copy(update={"temperature": 0.0})
        self.llm_client = LLMClient(self.config)

    @property
    def model(self) -> str:
        return f"{self.config.provider}/{self.config.model_name}"

    def judge(self, result: GenerationResult) -> Tuple[float, str, float]:
        """(score, reasoning, cost of the judge call). Raises JudgeUnavailable or JudgeOutputError."""
        prompt = FAITHFULNESS_PROMPT_TEMPLATE.format(
            context=result.context_window.context_text, question=result.query, answer=result.answer,
        )
        call = self.llm_client.call_llm(prompt, response_schema=EvaluationResponse)
        if call.failed:
            raise JudgeUnavailable(call.text)
        score, reasoning = parse_verdict(call.text)
        logger.info(f"Faithfulness judge -> score {score} | {reasoning}")
        return score, reasoning, call.cost_usd

    def evaluate(self, result: GenerationResult) -> GenerationResult:
        """Sets faithfulness_score and faithfulness_reasoning on the result; never raises."""
        try:
            result.faithfulness_score, result.faithfulness_reasoning, _ = self.judge(result)
        except JudgeUnavailable as e:
            result.faithfulness_score, result.faithfulness_reasoning = 0.0, str(e)
        except JudgeOutputError as e:
            logger.error(f"Faithfulness judge output unusable: {e}")
            result.faithfulness_score, result.faithfulness_reasoning = 0.0, f"Parse error: {e}"
        return result
