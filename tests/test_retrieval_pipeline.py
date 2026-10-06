"""The pipeline and its rerankers."""
import os
import pytest
from src.retrieving.config import RetrievalConfig
from src.retrieving.models import RetrievalResult
from src.retrieving.pipeline import RetrievalPipeline
from src.retrieving.rerankers import FlashRankReranker, JinaReranker, RerankError
from tests.builders import retrieved_chunk as chunk


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
