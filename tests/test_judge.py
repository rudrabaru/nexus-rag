"""The faithfulness judge runs at temperature 0 on its own copy of the generator's configuration."""
from unittest.mock import MagicMock

from src.generating.evaluator import FaithfulnessEvaluator
from src.generating.models import ContextWindow, GenerationConfig, GenerationResult
from src.llm.client import LLMCall


import pytest


@pytest.fixture
def judge_llm(monkeypatch):
    llm_client = MagicMock()
    llm_client.call_llm.return_value = LLMCall(text='{"score": 1.0, "reasoning": "supported"}')
    monkeypatch.setattr("src.generating.evaluator.LLMClient", lambda config: llm_client)
    return llm_client


def make_result():
    return GenerationResult(query="q", answer="a", context_window=ContextWindow(context_text="c"))


def test_judge_uses_zero_temperature_without_touching_the_generators_config(judge_llm):
    """The judge never mutates the config object it shares with the answer generator."""
    generator_config = GenerationConfig(temperature=0.7)
    evaluator = FaithfulnessEvaluator(config=generator_config)

    seen = []
    judge_llm.call_llm.side_effect = lambda *a, **k: (
        seen.append((generator_config.temperature, evaluator.config.temperature))
        or LLMCall(text='{"score": 1.0, "reasoning": "ok"}')
    )
    result = evaluator.evaluate(make_result())

    assert result.faithfulness_score == 1.0
    assert seen == [(0.7, 0.0)]
    assert generator_config.temperature == 0.7
    assert evaluator.config is not generator_config
