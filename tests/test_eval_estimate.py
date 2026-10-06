"""The pre-flight estimate: an experiment that the free tiers cannot finish is found before it starts."""
from src.config import get_settings
from src.evaluation.estimate import estimate
from src.evaluation.spec import ExperimentSpec

QUERIES = [f"How does feature number {i} behave when the input is empty?" for i in range(41)]


def spec_of(trials, generation=None):
    return ExperimentSpec(**{"name": "e", "dataset": "d.json", "tenant_id": "demo", "trials": trials,
                             **({"generation": generation} if generation else {})})


def line(result, label_part):
    return next(l for l in result.lines if label_part in l.label)


FOUR = {"dense": {"strategy": "dense"}, "sparse": {"strategy": "sparse"}, "hybrid": {}, "rerank": {"reranker": "flashrank"}}


def test_a_retrieval_only_experiment_is_embedding_bound_and_fits(monkeypatch):
    result = estimate(spec_of(FOUR), QUERIES, get_settings())
    embeddings = line(result, "voyage embeddings")
    assert embeddings.calls == 41  # embedded once per index, not once per trial
    assert 13 <= embeddings.minutes <= 14  # 41 requests at 3 a minute
    assert result.problems() == []
    assert len(result.lines) == 1  # sparse and local rerankers ask nothing of a provider


def test_a_sparse_only_experiment_embeds_nothing():
    assert estimate(spec_of({"sparse": {"strategy": "sparse"}}), QUERIES, get_settings()).lines == []


def test_geminis_daily_cap_makes_a_generation_run_unfinishable_in_a_day():
    generation = {"model": {"provider": "gemini", "model_name": "gemini-3.5-flash"},
                  "judge": {"provider": "groq", "model_name": "openai/gpt-oss-20b"}}
    result = estimate(spec_of(FOUR, generation), QUERIES, get_settings())
    gemini = line(result, "gemini/")
    assert gemini.calls == 41 * 4 and gemini.days == 9  # 164 calls against 20 a day
    assert any("gemini/gemini-3.5-flash" in p and "9 days" in p for p in result.problems())


def test_groqs_daily_tokens_bound_the_judge():
    generation = {"model": {"provider": "gemini", "model_name": "gemini-3.5-flash"},
                  "judge": {"provider": "groq", "model_name": "openai/gpt-oss-20b"}}
    groq = line(estimate(spec_of({"hybrid": {}}, generation), QUERIES, get_settings()), "groq/")
    assert groq.calls == 41 and groq.tokens > 200_000 and groq.days == 2  # about 270K tokens against 200K a day


def test_one_model_answering_and_judging_shares_one_allowance():
    same = {"model": {"provider": "groq", "model_name": "openai/gpt-oss-20b"}, "judge": {"provider": "groq", "model_name": "openai/gpt-oss-20b"}}
    result = estimate(spec_of({"hybrid": {}}, same), QUERIES, get_settings())
    assert line(result, "groq/").calls == 82


def test_a_voyage_rerank_pool_larger_than_a_minutes_tokens_can_never_be_sent():
    result = estimate(spec_of({"v": {"reranker": "voyage", "rerank_candidates": 20}}), QUERIES, get_settings())
    assert line(result, "voyage rerank").impossible
    assert any("can never be sent" in p for p in result.problems())


def test_a_small_voyage_rerank_pool_is_possible():
    result = estimate(spec_of({"v": {"reranker": "voyage", "rerank_candidates": 8}}), QUERIES, get_settings())
    assert not line(result, "voyage rerank").impossible


def test_cloudflare_embeddings_are_limited_by_the_day_not_the_minute(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "cloudflare")
    get_settings.cache_clear()
    embeddings = line(estimate(spec_of({"dense": {"strategy": "dense"}}), QUERIES, get_settings()), "cloudflare")
    assert embeddings.minutes < 1 and embeddings.days == 1


def test_the_estimate_states_that_it_is_an_upper_bound():
    text = estimate(spec_of(FOUR), QUERIES, get_settings()).render()
    assert "upper bounds" in text and "41 queries x 4 trials" in text


def test_a_trial_with_a_smaller_context_asks_for_fewer_tokens_and_a_trial_may_use_another_model():
    generation = {"model": {"provider": "gemini", "model_name": "gemini-3.5-flash"},
                  "judge": {"provider": "groq", "model_name": "openai/gpt-oss-20b"}}
    wide = estimate(spec_of({"a": {}}, generation), QUERIES, get_settings())
    narrow = estimate(spec_of({"a": {"max_context_tokens": 1000}}, generation), QUERIES, get_settings())
    assert line(narrow, "groq/").tokens < line(wide, "groq/").tokens

    split = estimate(spec_of({"a": {}, "b": {"generation_model": {"provider": "groq", "model_name": "openai/gpt-oss-120b"}}}, generation),
                     QUERIES, get_settings())
    assert line(split, "gemini/").calls == 41 and line(split, "groq/openai/gpt-oss-120b").calls == 41
    assert line(split, "groq/openai/gpt-oss-20b").calls == 82  # the judge sees both trials
