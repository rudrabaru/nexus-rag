"""Specs and datasets, and the generation stage: judge choice and cache keys."""
import pytest
from pydantic import ValidationError
from src.config import Settings
from src.evaluation.dataset import EvaluationQuery, integrity_problems
from src.evaluation.generation import GenerationStage
from src.evaluation.spec import GenerationSpec
from src.llm.client import LLMCall
from src.retrieving.models import RetrievalResult
from tests.support.eval_integrity import chunk, query


def stage_for(judge="groq/openai/gpt-oss-20b", model="gemini/gemini-3.5-flash"):
    spec = GenerationSpec(
        model=dict(zip(("provider", "model_name"), model.split("/", 1))),
        judge=dict(zip(("provider", "model_name"), judge.split("/", 1))),
    )
    return GenerationStage(spec, "demo", None, Settings(_env_file=None, gemini_api_key="g", groq_api_key="r"))


class Stage:
    """A stage whose model calls are scripted and whose caches are dictionaries."""

    def __init__(self):
        self.stage = stage_for()
        self.cache = {}
        self.stage._get = lambda table, key: self.cache.get((table.name, key))
        self.stage._put = lambda table, **values: self.cache.setdefault((table.name, values["cache_key"]), values)
        self.generator_calls, self.judge_calls = 0, 0
        stage = self

        class Generator:
            def call_llm(self, prompt, **kw):
                stage.generator_calls += 1
                return LLMCall(text="an answer", prompt_tokens=5, completion_tokens=3)

        class Judge:
            def call_llm(self, prompt, **kw):
                stage.judge_calls += 1
                return LLMCall(text='{"score": 1.0, "reasoning": "supported"}', cost_usd=0.001)

        self.stage.generator.llm_client = Generator()
        self.stage.judge.llm_client = Judge()

    def run(self, text="some text", chunk_id="a"):
        return self.stage.run("q", RetrievalResult(query="q", top_k=1, latency_ms=1.0, chunks=[chunk(chunk_id, text)]))


def test_a_generation_experiment_must_name_its_judge():
    with pytest.raises(ValidationError):
        GenerationSpec()
    assert GenerationSpec(judge={"provider": "groq", "model_name": "openai/gpt-oss-20b"}).model is None


def test_a_blank_acceptable_document_or_heading_is_a_dataset_problem():
    blank_doc = EvaluationQuery(query="q", acceptable_documents=["page", " "])
    blank_heading = EvaluationQuery(query="q2", acceptable_documents=["page"], acceptable_headings=[""])
    assert any("blank" in p for p in integrity_problems([blank_doc]))
    assert any("blank" in p for p in integrity_problems([blank_heading]))
    assert integrity_problems([query()]) == []


def test_a_model_may_not_judge_itself():
    with pytest.raises(ValueError, match="favour its own answers"):
        stage_for(judge="gemini/gemini-3.5-flash", model="gemini/gemini-3.5-flash")


def test_an_identical_run_is_answered_and_judged_from_the_caches():
    s = Stage()
    first, second = s.run(), s.run()
    assert (s.generator_calls, s.judge_calls) == (1, 1) and second["generation_cached"] and second["judge_cached"]
    assert first["judge_cost_usd"] == 0.001


def test_changed_chunk_text_under_the_same_chunk_id_is_not_a_cache_hit():
    s = Stage()
    s.run(text="the original text of the chunk")
    s.run(text="the re-ingested, different text")
    assert (s.generator_calls, s.judge_calls) == (2, 2)  # a chunk id can name different text after a re-ingest


def test_changing_the_judge_instrument_or_the_sampling_invalidates_cached_answers_and_verdicts():
    s = Stage()
    s.run()
    s.stage._judge_params = "another-prompt-version;t=0.0;max=4096"
    s.run()
    assert (s.generator_calls, s.judge_calls) == (1, 2)  # the answer is reused, the verdict is not
    s.stage._generation_params = "t=0.9;max=4096"
    s.run()
    assert s.generator_calls == 2


def test_a_run_with_no_context_makes_no_calls_and_says_so():
    s = Stage()
    row = s.stage.run("q", RetrievalResult(query="q", top_k=1, latency_ms=1.0, chunks=[]))
    assert row["empty_context"] is True and (s.generator_calls, s.judge_calls) == (0, 0)
