"""Rank fusion and retrieval configuration."""
import pytest
from pydantic import ValidationError
from src.config import get_settings
from src.retrieving.config import RetrievalConfig
from src.retrieving.fusion import fuse, rrf_scores
from src.services.chat_config import chat_retrieval_config
from tests.builders import retrieved_chunk as chunk


def test_unweighted_rrf_matches_ranx():
    from ranx import Run, fuse as ranx_fuse

    dense, sparse = ["a", "b", "c", "d"], ["c", "e", "a"]

    def as_run(ids):
        return Run({"q": {doc: float(len(ids) - i) for i, doc in enumerate(ids)}})

    expected = ranx_fuse(runs=[as_run(dense), as_run(sparse)], method="rrf", params={"k": 60}).to_dict()["q"]
    ours = rrf_scores([(dense, 1.0), (sparse, 1.0)], k=60)

    assert ours.keys() == expected.keys()
    for doc, score in expected.items():
        assert ours[doc] == pytest.approx(score)


def test_weights_scale_each_rankings_contribution():
    scores = rrf_scores([(["a"], 2.0), (["b"], 1.0)], k=60)
    assert scores["a"] == pytest.approx(2 / 61) and scores["b"] == pytest.approx(1 / 61)


def test_a_chunk_found_by_both_rankings_is_fused_not_duplicated():
    """Regression: dense and sparse once used different ids for one chunk, so it filled two slots."""
    fused = fuse([([chunk("shared"), chunk("dense-only")], 1.0), ([chunk("sparse-only"), chunk("shared")], 1.0)], k=60, limit=5)

    ids = [c.chunk_id for c in fused]
    assert sorted(ids) == ["dense-only", "shared", "sparse-only"]
    assert ids[0] == "shared" and fused[0].similarity_score == 1.0  # ranks 1 + 2 beat any single rank; best scaled to 1


def test_fusion_returns_copies_and_a_zero_weight_ranking_contributes_nothing():
    original = chunk("a", score=0.9)
    fused = fuse([([original], 1.0), ([chunk("b")], 0.0)], k=60, limit=5)
    assert [c.chunk_id for c in fused] == ["a"]
    assert original.similarity_score == 0.9


@pytest.mark.parametrize("bad", [
    {"reranker": "flashrank", "top_k": 10, "rerank_candidates": 5},
    {"strategy": "hybrid", "dense_weight": 0, "sparse_weight": 0},
    {"rerank_candidates": 81},
    {"strategy": "keyword"},
    {"unknown_knob": 1},
])
def test_invalid_configurations_are_rejected(bad):
    with pytest.raises(ValidationError):
        RetrievalConfig(**bad)


def test_chat_uses_the_configured_reranker_only_when_the_request_asks(monkeypatch):
    assert chat_retrieval_config(get_settings(), None, 3, use_reranker=False).reranker is None
    config = chat_retrieval_config(get_settings(), None, 3, use_reranker=True)
    assert (config.reranker, config.rerank_candidates, config.strategy) == ("flashrank", 12, "hybrid")

    monkeypatch.setenv("RERANKER", "none")
    get_settings.cache_clear()
    assert chat_retrieval_config(get_settings(), None, 5, use_reranker=True).reranker is None


def test_a_workspaces_choices_override_the_defaults_and_the_request_still_has_the_last_word():
    workspace = {"strategy": "dense", "reranker": "jina", "rerank_candidates": 8, "rrf_k": 30, "top_k": 99, "index_id": "x:y"}

    config = chat_retrieval_config(get_settings(), workspace, top_k=4, use_reranker=True)
    assert (config.strategy, config.reranker, config.rerank_candidates, config.rrf_k) == ("dense", "jina", 8, 30)
    assert config.top_k == 4 and config.index_id is None  # top_k is the caller's and the index is the deployment's

    assert chat_retrieval_config(get_settings(), workspace, top_k=4, use_reranker=False).reranker is None


def test_a_pool_smaller_than_top_k_is_widened_and_one_larger_than_the_limit_is_capped():
    assert chat_retrieval_config(get_settings(), {"rerank_candidates": 2}, 5, True).rerank_candidates == 5
    assert chat_retrieval_config(get_settings(), None, 20, True).rerank_candidates == 80


def test_reranker_none_turns_reranking_off(monkeypatch):
    monkeypatch.setenv("RERANKER", "none")
    get_settings.cache_clear()
    assert get_settings().effective_reranker is None
