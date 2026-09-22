from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.generating.evaluator import FaithfulnessEvaluator
from src.generating.models import ContextWindow, GenerationConfig, GenerationResult
from src.ingestion.embedding_worker import EmbeddingWorker


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


def make_worker(failed_indices=()):
    chunks = [SimpleNamespace(chunk_id=f"c{i}", token_count=10) for i in range(3)]
    generator = MagicMock()
    generator.generate_embeddings.return_value = (chunks, list(failed_indices))
    db = MagicMock()
    db.load_chunks.return_value = len(chunks)
    return EmbeddingWorker(generator, db), chunks


def test_embedding_worker_runs_without_a_registry_or_job():
    worker, chunks = make_worker()
    assert worker.process_batches(chunks, "tenant-1")["job_status"] == "complete"


def test_embedding_worker_runs_with_a_job_id_but_no_registry():
    """Regression: job_status was bound only inside `if registry and job_id`, so this path raised."""
    worker, chunks = make_worker()
    assert worker.process_batches(chunks, "tenant-1", registry=None, job_id="job-1")["job_status"] == "complete"


def test_embedding_worker_reports_partial_success_on_failed_batches():
    worker, chunks = make_worker(failed_indices=[1])
    assert worker.process_batches(chunks, "tenant-1")["job_status"] == "partial_success"
