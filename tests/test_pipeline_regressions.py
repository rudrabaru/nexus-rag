from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.generating.evaluator import FaithfulnessEvaluator
from src.generating.models import ContextWindow, GenerationConfig, GenerationResult
from src.ingestion.embedding_worker import EmbeddingUnavailableError, EmbeddingWorker


@pytest.fixture
def judge_llm(monkeypatch):
    llm_client = MagicMock()
    llm_client.call_llm.return_value = ('{"score": 1.0, "reasoning": "supported"}', 0, 0, 0)
    monkeypatch.setattr("src.generating.llm_client.LLMClient", lambda config: llm_client)
    return llm_client


def make_result():
    return GenerationResult(query="q", answer="a", context_window=ContextWindow(context_text="c"))


def test_judge_uses_zero_temperature_without_touching_the_generators_config(judge_llm):
    """Regression: the judge mutated the config object it shared with the answer generator."""
    generator_config = GenerationConfig(temperature=0.7)
    evaluator = FaithfulnessEvaluator(config=generator_config)

    seen = []
    judge_llm.call_llm.side_effect = lambda *a, **k: (
        seen.append((generator_config.temperature, evaluator.config.temperature))
        or ('{"score": 1.0, "reasoning": "ok"}', 0, 0, 0)
    )
    result = evaluator.evaluate(make_result())

    assert result.faithfulness_score == 1.0
    assert seen == [(0.7, 0.0)]
    assert generator_config.temperature == 0.7
    assert evaluator.config is not generator_config


def make_worker(chunk_count=3, embedded_indices=None, embedding_text="e"):
    """embedded_indices=None embeds every chunk; otherwise only those indices succeed."""
    input_chunks = [SimpleNamespace(chunk_id=f"c{i}", chunk_text=embedding_text, token_count=10) for i in range(chunk_count)]
    kept = range(chunk_count) if embedded_indices is None else embedded_indices
    embedded = [SimpleNamespace(chunk_id=f"c{i}", token_count=10) for i in kept]
    failed = sorted(set(range(chunk_count)) - set(kept))

    generator = MagicMock()
    generator.generate_embeddings.return_value = (embedded, failed)
    generator.last_error = None  # matches EmbeddingGenerator's real default
    return EmbeddingWorker(generator), input_chunks


def test_embedding_worker_reports_complete_when_nothing_fails():
    worker, chunks = make_worker()
    outcome = worker.embed(chunks)
    assert outcome.status == "complete"
    assert len(outcome.chunks) == 3
    assert outcome.failed_indices == []
    assert outcome.metadata is None


def test_embedding_worker_reports_partial_success_on_failed_batches():
    worker, chunks = make_worker(embedded_indices=[0, 2])
    outcome = worker.embed(chunks)
    assert outcome.status == "partial_success"
    assert outcome.failed_indices == [1]
    assert outcome.metadata["failed_chunk_indices"] == [1]


def test_embedding_worker_raises_when_nothing_could_be_embedded():
    """A total embedding failure (API outage, dead key) must not be reported as a silent success."""
    worker, chunks = make_worker(embedded_indices=[])
    with pytest.raises(EmbeddingUnavailableError):
        worker.embed(chunks)


def test_embedding_worker_reports_increasing_progress():
    chunks = [SimpleNamespace(chunk_id=f"c{i}", chunk_text="e", token_count=10) for i in range(120)]  # 3 batches of <=50
    generator = MagicMock()
    generator.generate_embeddings.side_effect = lambda batch: (batch, [])
    worker = EmbeddingWorker(generator)

    seen = []
    worker.embed(chunks, on_progress=seen.append)

    assert seen == sorted(seen) and len(seen) == 3
    assert seen[-1] == 99
