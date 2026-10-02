"""The config-driven retrieval pipeline: fusion maths, strategies, reranking and explicit degradation."""
import os

import pytest
from pydantic import ValidationError

from src.api.models.query_models import QueryRequest
from src.config import get_settings
from src.retrieving.config import RetrievalConfig
from src.retrieving.fusion import fuse, rrf_scores
from src.retrieving.models import RetrievalResult, RetrievedChunk
from src.retrieving.pipeline import RetrievalPipeline
from src.retrieving.rerankers import FlashRankReranker, JinaReranker, RerankError
from src.services.query_service import chat_config


def chunk(chunk_id: str, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, source_document=chunk_id, text=f"text of {chunk_id}", similarity_score=score, metadata={})


class FakeRetriever:
    def __init__(self, chunk_ids=(), error=None):
        self.chunk_ids = list(chunk_ids)
        self.error = error
        self.calls = []

    async def retrieve(self, query, top_k=5, tenant_id=None, pipeline_logger=None):
        self.calls.append(top_k)
        if self.error:
            raise self.error
        chunks = [chunk(cid) for cid in self.chunk_ids[:top_k]]
        return RetrievalResult(query=query, top_k=top_k, latency_ms=1.0, embedding_tokens=7, chunks=chunks)


class FakeReranker:
    name = "fake"

    def __init__(self, error=None):
        self.error = error
        self.pool = None

    async def rerank(self, query, candidates, top_k):
        self.pool = [c.chunk_id for c in candidates]
        if self.error:
            raise self.error
        reordered = [c.model_copy(update={"similarity_score": 0.5}) for c in reversed(candidates)][:top_k]
        return RetrievalResult(query=query, top_k=top_k, latency_ms=3.0, rerank_latency_ms=3.0, rerank_cost_usd=0.001, chunks=reordered)


def pipeline(dense=None, sparse=None, reranker_impl=None, **config):
    return RetrievalPipeline(
        config=RetrievalConfig(**config), dense=dense or FakeRetriever(), sparse=sparse or FakeRetriever(), reranker=reranker_impl
    )


# ── Fusion ────────────────────────────────────────────────────────────────────

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


# ── Configuration ─────────────────────────────────────────────────────────────

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
    assert chat_config(QueryRequest(query="q", top_k=3)).reranker is None
    config = chat_config(QueryRequest(query="q", top_k=3, use_reranker=True))
    assert (config.reranker, config.rerank_candidates, config.strategy) == ("flashrank", 12, "hybrid")

    monkeypatch.setenv("RERANKER", "none")
    get_settings.cache_clear()
    assert chat_config(QueryRequest(query="q", use_reranker=True)).reranker is None


def test_legacy_enable_reranker_false_still_turns_reranking_off(monkeypatch):
    monkeypatch.setenv("ENABLE_RERANKER", "false")
    get_settings.cache_clear()
    assert get_settings().effective_reranker is None


# ── Pipeline ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("strategy, dense_calls, sparse_calls", [("dense", 1, 0), ("sparse", 0, 1), ("hybrid", 1, 1)])
async def test_each_strategy_queries_only_its_retrievers(strategy, dense_calls, sparse_calls):
    dense, sparse = FakeRetriever(["d1"]), FakeRetriever(["s1"])
    await pipeline(dense, sparse, strategy=strategy).run("q", "tenant-1")
    assert (len(dense.calls), len(sparse.calls)) == (dense_calls, sparse_calls)


async def test_a_zero_weight_skips_that_retriever_entirely():
    dense, sparse = FakeRetriever(["d1"]), FakeRetriever(["s1"])
    result = await pipeline(dense, sparse, sparse_weight=0).run("q", "tenant-1")
    assert sparse.calls == [] and [c.chunk_id for c in result.chunks] == ["d1"]


async def test_hybrid_serves_the_sparse_ranking_and_says_so_when_the_query_cannot_be_embedded():
    dense, sparse = FakeRetriever(error=RuntimeError("voyage 429")), FakeRetriever(["s1", "s2"])
    result = await pipeline(dense, sparse).run("q", "tenant-1")
    assert [c.chunk_id for c in result.chunks] == ["s1", "s2"]
    assert result.degraded and "voyage 429" in result.degraded[0]


async def test_a_database_failure_is_not_disguised_as_degradation():
    sparse = FakeRetriever(error=ConnectionError("database gone"))
    with pytest.raises(ConnectionError):
        await pipeline(FakeRetriever(["d1"]), sparse).run("q", "tenant-1")


async def test_the_reranker_reorders_the_candidate_pool_down_to_top_k():
    dense = FakeRetriever([f"c{i}" for i in range(30)])
    reranker = FakeReranker()
    result = await pipeline(dense, reranker_impl=reranker, strategy="dense", top_k=3, reranker="flashrank", rerank_candidates=12).run("q", "t")

    assert dense.calls == [12] and len(reranker.pool) == 12  # the pool, not top_k, is retrieved and reranked
    assert [c.chunk_id for c in result.chunks] == ["c11", "c10", "c9"]
    assert [c.chunk_id for c in result.candidates] == reranker.pool
    assert result.rerank_cost_usd == 0.001 and result.embedding_tokens == 7 and result.degraded == []


async def test_a_failed_reranker_keeps_the_first_stage_order_and_records_why():
    dense = FakeRetriever([f"c{i}" for i in range(10)])
    result = await pipeline(
        dense, reranker_impl=FakeReranker(RerankError("jina: quota")), strategy="dense", top_k=3, reranker="jina", rerank_candidates=10
    ).run("q", "t")
    assert [c.chunk_id for c in result.chunks] == ["c0", "c1", "c2"]
    assert "jina: quota" in result.degraded[0]


# ── Rerankers ─────────────────────────────────────────────────────────────────

async def test_flashrank_orders_by_the_model_score_and_leaves_the_candidates_untouched(monkeypatch):
    reranker = FlashRankReranker("any-model", "unused")
    monkeypatch.setattr(reranker, "_order", lambda query, candidates: [(2, 0.9), (0, 0.4), (1, 0.1)])
    candidates = [chunk("a", 0.7), chunk("b", 0.6), chunk("c", 0.5)]

    result = await reranker.rerank("q", candidates, top_k=2)

    assert [(c.chunk_id, c.similarity_score) for c in result.chunks] == [("c", 0.9), ("a", 0.4)]
    assert candidates[2].similarity_score == 0.5 and result.rerank_cost_usd == 0.0


async def test_flashrank_failures_surface_as_rerank_errors(monkeypatch):
    reranker = FlashRankReranker("any-model", "unused")
    monkeypatch.setattr(reranker, "_order", lambda query, candidates: (_ for _ in ()).throw(OSError("model missing")))
    with pytest.raises(RerankError, match="model missing"):
        await reranker.rerank("q", [chunk("a")], top_k=1)


async def test_jina_without_a_key_fails_fast():
    with pytest.raises(RerankError, match="JINA_API_KEY"):
        await JinaReranker("").rerank("q", [chunk("a")], top_k=1)


@pytest.mark.skipif(not os.environ.get("RUN_MODEL_TESTS"), reason="downloads a ~4 MB model; set RUN_MODEL_TESTS=1")
async def test_the_real_flashrank_model_puts_the_answering_passage_first(tmp_path):
    reranker = FlashRankReranker("ms-marco-TinyBERT-L-2-v2", str(tmp_path))
    candidates = [
        chunk("off-topic").model_copy(update={"text": "Invoices are archived for seven years in the finance system."}),
        chunk("answer").model_copy(update={"text": "To allow HTTPS, create a firewall rule that permits tcp:443."}),
    ]
    result = await reranker.rerank("how do I allow https through the firewall", candidates, top_k=2)
    assert result.chunks[0].chunk_id == "answer"
