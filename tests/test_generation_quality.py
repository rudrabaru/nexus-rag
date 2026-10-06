"""Generation edges: what the context builder drops and says, when no model call is made, the judge's contract, and model roles."""
from unittest.mock import MagicMock

import pytest

from src.config import Settings
from src.generating.context_builder import ContextBuilder
from src.generating.evaluator import FaithfulnessEvaluator, JudgeOutputError, JudgeUnavailable, parse_verdict
from src.generating.generator import RAGGenerator
from src.generating.models import ContextWindow, GenerationConfig, GenerationResult
from src.generating.prompt_template import build_prompt
from src.generating.query_rewriter import QueryRewriter
from src.llm.client import LLMCall
from src.llm.errors import GenerationError
from src.llm.roles import role_fields
from src.retrieving.models import RetrievalResult, RetrievedChunk


def retrieved(chunk_id, text="some text", score=0.5, tokens=0, source_url="https://a.example") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, source_document="Doc", source_url=source_url, text=text, similarity_score=score,
        token_count=tokens, metadata={},
    )


def build(chunks, **config):
    return ContextBuilder(GenerationConfig(**config)).build(chunks)


# ── Context building ─────────────────────────────────────────────────────────

def test_every_excluded_chunk_is_recorded_with_its_reason():
    window = build(
        [retrieved("a", "alpha", 0.9, 100), retrieved("dup", "alpha", 0.8, 100), retrieved("low", "beta", 0.05, 100), retrieved("big", "gamma", 0.7, 400)],
        max_context_tokens=300, min_similarity_score=0.1,
    )
    assert [c.chunk_id for c in window.included_chunks] == ["a"]
    assert window.exclusion_reasons == {"dup": "duplicate", "low": "score", "big": "budget"}
    assert {c.chunk_id for c in window.excluded_chunks} == {"dup", "low", "big"}


def test_the_budget_uses_the_chunkers_real_token_count_when_there_is_one():
    window = build([retrieved("a", "word " * 10, tokens=500), retrieved("b", "other " * 10, tokens=500)], max_context_tokens=900)
    assert [c.chunk_id for c in window.included_chunks] == ["a"] and window.total_context_tokens == 500


def test_a_chunk_without_a_url_still_builds_a_citation():
    window = build([retrieved("a", source_url=None)])
    assert "[Source: Doc]" in window.context_text


# ── No model call when there is no context ───────────────────────────────────

def make_generator():
    generator = RAGGenerator(GenerationConfig(max_context_tokens=300, min_similarity_score=0.1))
    generator.llm_client = MagicMock()
    return generator


def result_of(chunks) -> RetrievalResult:
    return RetrievalResult(query="q", top_k=5, latency_ms=1.0, chunks=chunks)


def test_an_empty_retrieval_makes_no_model_call_and_says_nothing_was_retrieved():
    generator = make_generator()
    result = generator.generate("q", result_of([]))
    generator.llm_client.call_llm.assert_not_called()
    assert "no chunks" in result.answer


@pytest.mark.parametrize("chunks, expected", [
    ([retrieved("low", "x", 0.01, 10)], "below the minimum similarity"),
    ([retrieved("big", "x", 0.9, 5000)], "context budget"),
])
def test_the_diagnostic_names_the_real_reason_context_was_lost(chunks, expected):
    generator = make_generator()
    result = generator.generate("q", result_of(chunks))
    generator.llm_client.call_llm.assert_not_called()
    assert expected in result.answer


def test_a_failed_generation_raises_instead_of_returning_text():
    generator = make_generator()
    generator.llm_client.call_llm.side_effect = GenerationError("RateLimitError: slow")
    with pytest.raises(GenerationError):
        generator.generate("q", result_of([retrieved("a", "alpha", 0.9, 10)]))


# ── The prompt ───────────────────────────────────────────────────────────────

def test_the_citation_instruction_follows_cite_sources():
    with_citations = build_prompt("q", "ctx", GenerationConfig(cite_sources=True))
    without = build_prompt("q", "ctx", GenerationConfig(cite_sources=False))
    assert "cite where it came from" in with_citations
    assert "cite" not in without.lower().split("=== context start ===")[0]


# ── The judge ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    '{"score": 1.0, "reasoning": "ok"}', '{"score": 0.5, "reasoning": "partly"}', '{"score": 0, "reasoning": "no"}',
])
def test_a_score_the_prompt_allows_is_accepted(text):
    assert parse_verdict(text)[0] in (0.0, 0.5, 1.0)


@pytest.mark.parametrize("text", [
    '{"score": 7, "reasoning": "x"}', '{"score": -1, "reasoning": "x"}', '{"score": 0.75, "reasoning": "x"}',
    '{"score": NaN, "reasoning": "x"}', '{"reasoning": "no score"}', "not json",
])
def test_a_score_the_prompt_does_not_allow_is_rejected_not_averaged_in(text):
    with pytest.raises(JudgeOutputError):
        parse_verdict(text)


def judge_with(llm_client) -> FaithfulnessEvaluator:
    judge = FaithfulnessEvaluator(GenerationConfig())
    judge.llm_client = llm_client
    return judge


def judged_result() -> GenerationResult:
    return GenerationResult(query="q", answer="a", context_window=ContextWindow(context_text="ctx"))


def test_an_unavailable_judge_leaves_the_chat_score_empty_not_zero():
    client = MagicMock()
    client.call_llm.side_effect = GenerationError("RateLimitError")
    result = judge_with(client).evaluate(judged_result())
    assert result.faithfulness_score is None and "unavailable" in result.faithfulness_reasoning


def test_an_unusable_judge_reply_leaves_the_chat_score_empty_not_zero():
    client = MagicMock()
    client.call_llm.return_value = LLMCall(text='{"score": 9, "reasoning": "x"}')
    result = judge_with(client).evaluate(judged_result())
    assert result.faithfulness_score is None


def test_the_strict_judge_raises_so_an_evaluation_never_records_a_failure_as_a_score():
    client = MagicMock()
    client.call_llm.side_effect = GenerationError("down")
    with pytest.raises(JudgeUnavailable):
        judge_with(client).judge(judged_result())


# ── The rewriter ─────────────────────────────────────────────────────────────

def rewriter_replying(text=None, error=None) -> QueryRewriter:
    rewriter = QueryRewriter(GenerationConfig())
    rewriter.llm_client = MagicMock()
    if error:
        rewriter.llm_client.call_llm.side_effect = error
    else:
        rewriter.llm_client.call_llm.return_value = LLMCall(text=text)
    return rewriter


def test_a_rewrite_in_a_code_fence_is_read():
    assert rewriter_replying('```json\n{"rewritten_query": "how to configure the loop node"}\n```').rewrite("and it?", [{"role": "user", "content": "x"}]) == "how to configure the loop node"


@pytest.mark.parametrize("reply", ['{"rewritten_query": ""}', "no json here", '{"other": 1}'])
def test_an_empty_or_unreadable_rewrite_keeps_the_original_query(reply):
    assert rewriter_replying(reply).generalise("original query") == "original query"


def test_a_failed_rewrite_keeps_the_original_query():
    assert rewriter_replying(error=GenerationError("down")).rewrite("follow up", [{"role": "user", "content": "x"}]) == "follow up"


def test_the_generalising_prompt_names_no_corpus():
    from src.generating.query_rewriter import GENERALISE_PROMPT

    assert "documentation" not in GENERALISE_PROMPT.lower() and "technical" not in GENERALISE_PROMPT.lower()


# ── Model roles ──────────────────────────────────────────────────────────────

def settings(**env) -> Settings:
    return Settings(_env_file=None, gemini_api_key="g", groq_api_key="r", **env)


def test_a_role_resolves_to_its_provider_and_model():
    fields = role_fields(settings(), "chat")
    assert (fields["provider"], fields["model_name"]) == ("gemini", "gemini-3.5-flash")
    assert fields["fallback_config"] == {"provider": "groq", "model_name": "openai/gpt-oss-20b"}


def test_the_judge_has_no_fallback_and_differs_from_the_chat_family_by_default():
    fields = role_fields(settings(), "judge")
    assert "fallback_config" not in fields and fields["provider"] != role_fields(settings(), "chat")["provider"]


def test_a_fallback_without_its_key_is_not_configured():
    assert "fallback_config" not in role_fields(Settings(_env_file=None, gemini_api_key="g"), "chat")


def test_an_empty_fallback_setting_disables_it():
    assert "fallback_config" not in role_fields(settings(llm_chat_fallback=""), "chat")
