"""
Config-driven retrieval: one RetrievalConfig in, one RetrievalResult out.

RetrievalResources holds the process-wide clients (per-index retrievers with their query
embedding cache, rerankers and their loaded models), built once. build_pipeline composes them
for one configuration; pipelines are cheap, so chat builds one per request and an evaluation
one per trial. Nothing about a query's retrieval behaviour is fixed at startup any more.

Stages: first stage (dense, sparse, or both fused by weighted RRF) -> optional rerank of the
first-stage pool -> top_k. Degradation is explicit: when hybrid cannot embed the query it
serves the sparse ranking, and when a reranker fails the first-stage order is kept; either way
the reason is recorded on the result (RetrievalResult.degraded), never hidden.
"""
import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine

from src.config import Settings
from src.embedding.providers import build_embedder
from src.retrieving.chunk_store import ChunkStore
from src.retrieving.config import RetrievalConfig
from src.retrieving.dense import DenseRetriever
from src.retrieving.fusion import fuse
from src.retrieving.models import RetrievalResult
from src.retrieving.rerankers import Reranker, build_reranker
from src.retrieving.sparse import SparseRetriever

logger = logging.getLogger(__name__)


class RetrievalResources:
    def __init__(self, settings: Settings, sync_engine: Engine, async_engine: AsyncEngine):
        self._settings = settings
        self._sync_engine = sync_engine
        self._async_engine = async_engine
        self.default_index_id = build_embedder(settings).index_id
        self._retrievers: Dict[str, Tuple[DenseRetriever, SparseRetriever]] = {}
        self._rerankers: Dict[str, Reranker] = {}
        self._lock = threading.Lock()

    def retrievers(self, index_id: Optional[str] = None) -> Tuple[DenseRetriever, SparseRetriever]:
        index_id = index_id or self.default_index_id
        with self._lock:
            if index_id not in self._retrievers:
                store = ChunkStore(self._sync_engine, self._async_engine, index_id)
                embedder = build_embedder(self._settings, index_id)
                self._retrievers[index_id] = (DenseRetriever(store, embedder), SparseRetriever(store))
            return self._retrievers[index_id]

    def chunk_store(self, index_id: Optional[str] = None) -> ChunkStore:
        return self.retrievers(index_id)[1].chunk_store

    def reranker(self, name: str) -> Reranker:
        with self._lock:
            if name not in self._rerankers:
                self._rerankers[name] = build_reranker(name, self._settings)
            return self._rerankers[name]


@dataclass
class RetrievalPipeline:
    config: RetrievalConfig
    dense: DenseRetriever
    sparse: SparseRetriever
    reranker: Optional[Reranker] = None

    async def run(self, query: str, tenant_id: Optional[str], pipeline_logger: Any = None) -> RetrievalResult:
        start = time.time()
        result = await self._first_stage(query, tenant_id, pipeline_logger)

        if self.reranker:
            result.candidates = result.chunks
            try:
                reranked = await self.reranker.rerank(query, result.candidates, self.config.top_k)
                result.chunks = reranked.chunks
                result.rerank_latency_ms = reranked.rerank_latency_ms
                result.rerank_cost_usd = reranked.rerank_cost_usd
            except Exception as e:
                logger.warning(f"RERANK | {self.reranker.name} failed, keeping first-stage order: {e}")
                result.degraded.append(f"reranker {self.reranker.name} failed ({e}); first-stage order kept")
                result.chunks = result.candidates[: self.config.top_k]

        result.top_k = self.config.top_k
        result.latency_ms = (time.time() - start) * 1000
        return result

    async def _first_stage(self, query, tenant_id, pipeline_logger) -> RetrievalResult:
        limit = self.config.candidate_count
        depth = self.config.fusion_depth or limit if self.config.strategy == "hybrid" else limit
        dense = lambda: self.dense.retrieve(query, top_k=depth, tenant_id=tenant_id)  # noqa: E731
        sparse = lambda: self.sparse.retrieve(query, top_k=depth, tenant_id=tenant_id, pipeline_logger=pipeline_logger)  # noqa: E731
        if self.config.strategy == "dense":
            return await dense()
        if self.config.strategy == "sparse":
            return await sparse()

        use_dense, use_sparse = self.config.dense_weight > 0, self.config.sparse_weight > 0
        dense_result, sparse_result = await asyncio.gather(
            dense() if use_dense else _nothing(query, depth),
            sparse() if use_sparse else _nothing(query, depth),
            return_exceptions=True,
        )
        if isinstance(sparse_result, BaseException):
            raise sparse_result  # a database failure; dense runs on the same database
        if isinstance(dense_result, BaseException):
            logger.warning(f"HYBRID | query embedding failed, serving the sparse ranking only: {dense_result}")
            sparse_result.degraded.append(f"dense search failed ({dense_result}); sparse ranking only")
            sparse_result.chunks = sparse_result.chunks[:limit]
            return sparse_result

        return RetrievalResult(
            query=query,
            top_k=limit,
            latency_ms=0.0,
            embedding_latency_ms=dense_result.embedding_latency_ms,
            search_latency_ms=max(dense_result.search_latency_ms, sparse_result.search_latency_ms),
            embedding_tokens=dense_result.embedding_tokens,
            query_embedding_tokens=dense_result.query_embedding_tokens,
            embedding_cost_usd=dense_result.embedding_cost_usd,
            chunks=fuse(
                [(dense_result.chunks, self.config.dense_weight), (sparse_result.chunks, self.config.sparse_weight)],
                k=self.config.rrf_k,
                limit=limit,
            ),
        )


async def _nothing(query: str, limit: int) -> RetrievalResult:
    return RetrievalResult(query=query, top_k=limit, latency_ms=0.0, chunks=[])


def build_pipeline(config: RetrievalConfig, resources: RetrievalResources) -> RetrievalPipeline:
    dense, sparse = resources.retrievers(config.index_id)
    reranker = resources.reranker(config.reranker) if config.reranker else None
    return RetrievalPipeline(config=config, dense=dense, sparse=sparse, reranker=reranker)
