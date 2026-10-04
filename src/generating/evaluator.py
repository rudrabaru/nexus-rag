import logging
import math
from typing import Tuple

from pydantic import BaseModel

from src.llm.client import LLMClient
from src.llm.errors import GenerationError
from src.llm.structured import extract_json_object

from .models import GenerationConfig, GenerationResult

logger = logging.getLogger(__name__)

# The only scores the prompt below allows. A judge that answers 0.75 or 7 has not followed the
# instrument, and a score it did not mean must not be averaged into a result.
VALID_SCORES = (0.0, 0.5, 1.0)


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
        score = float(evaluation["score"])
    except (KeyError, TypeError, ValueError) as e:
        raise JudgeOutputError(f"the judge's JSON has no numeric score: {evaluation!r}") from e
    if not math.isfinite(score) or score not in VALID_SCORES:
        raise JudgeOutputError(f"the judge's score {score!r} is not one of {VALID_SCORES}")
    return score, str(evaluation.get("reasoning", ""))


class FaithfulnessEvaluator:
    """
    LLM-as-a-judge: is the answer supported by the context it was given?

    judge() is strict and is what evaluations use: a failed call raises JudgeUnavailable and a
    malformed verdict raises JudgeOutputError, so neither is ever recorded as a score.
    evaluate() is the lenient form chat uses: a failure leaves the score empty (None) with the reason,
    never 0.0, which would read as "the answer was a hallucination".
    """

    def __init__(self, config: GenerationConfig = None):
        # A private copy fixed at temperature 0.0: the judge must be deterministic, and it
        # must never share (or mutate) the generator's config object.
        self.config = (config or GenerationConfig()).model_copy(update={"temperature": 0.0})
        self.llm_client = LLMClient(self.config)

    @property
    def model(self) -> str:
        return self.config.model_string

    def judge(self, result: GenerationResult) -> Tuple[float, str, float]:
        """(score, reasoning, cost of the judge call). Raises JudgeUnavailable or JudgeOutputError."""
        prompt = FAITHFULNESS_PROMPT_TEMPLATE.format(
            context=result.context_window.context_text, question=result.query, answer=result.answer,
        )
        try:
            call = self.llm_client.call_llm(prompt, response_schema=EvaluationResponse)
        except GenerationError as e:
            raise JudgeUnavailable(str(e)) from e
        score, reasoning = parse_verdict(call.text)
        logger.info(f"Faithfulness judge -> score {score} | {reasoning}")
        return score, reasoning, call.cost_usd

    def evaluate(self, result: GenerationResult) -> GenerationResult:
        """Sets faithfulness_score and faithfulness_reasoning on the result; never raises."""
        try:
            result.faithfulness_score, result.faithfulness_reasoning, _ = self.judge(result)
        except JudgeUnavailable as e:
            logger.warning(f"Faithfulness judge unavailable: {e}")
            result.faithfulness_score, result.faithfulness_reasoning = None, "The judge model was unavailable; no score."
        except JudgeOutputError as e:
            logger.error(f"Faithfulness judge output unusable: {e}")
            result.faithfulness_score, result.faithfulness_reasoning = None, "The judge's reply could not be used; no score."
        return result
