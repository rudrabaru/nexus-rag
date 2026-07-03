import time
import logging
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer

from pydantic import BaseModel
from .vector_store import ChromaDBManager
from src.embedding.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class RetrievedChunk(BaseModel):
    chunk_id: str
    source_document: str
    text: str
    similarity_score: float
    metadata: Dict[str, Any]


class RetrievalResult(BaseModel):
    query: str
    top_k: int
    latency_ms: float
    embedding_latency_ms: float = 0.0
    search_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    chunks: List[RetrievedChunk]


class DenseRetriever:
    def __init__(
        self,
        vector_store: ChromaDBManager,
        embedding_config: Optional[EmbeddingConfig] = None,
    ):
        self.vector_store = vector_store
        self.config = embedding_config or EmbeddingConfig()
        logger.info(
            f"Loading embedding model for dense retrieval: {self.config.model_name}"
        )
        self.model = SentenceTransformer(
            self.config.model_name, device=self.config.device
        )

    def retrieve(
        self, query: str, top_k: int = 5, tenant_id: Optional[str] = None
    ) -> RetrievalResult:
        start_time = time.time()

        # 1. Embed query (Add BGE instruction prefix if applicable)
        embed_start = time.time()
        bge_prefix = "Represent this sentence for searching relevant passages: "
        query_text = (
            bge_prefix + query if "bge" in self.config.model_name.lower() else query
        )
        query_embedding = self.model.encode(
            query_text,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=False,
        ).tolist()
        embed_latency = (time.time() - embed_start) * 1000

        # 2. Query ChromaDB
        search_start = time.time()

        where_clause = None
        if tenant_id:
            where_clause = {
                "$or": [
                    {"visibility": {"$eq": "public"}},
                    {"tenant_id": {"$eq": tenant_id}},
                ]
            }
        else:
            where_clause = {"visibility": {"$eq": "public"}}

        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )
        search_latency = (time.time() - search_start) * 1000

        # 3. Format candidates
        candidates = []
        if results["ids"] and len(results["ids"]) > 0:
            ids = results["ids"][0]
            distances = results["distances"][0]
            metadatas = results["metadatas"][0]
            documents = results["documents"][0]

            for i in range(len(ids)):
                metric = getattr(self.vector_store, "distance_metric", "cosine")
                if metric == "l2":
                    similarity = 1.0 / (1.0 + distances[i])
                else:
                    similarity = 1.0 - distances[i] if distances[i] <= 1.0 else 0.0

                chunk = RetrievedChunk(
                    chunk_id=ids[i],
                    source_document=metadatas[i].get("source_document", ""),
                    text=documents[i],
                    similarity_score=similarity,
                    metadata=metadatas[i],
                )
                candidates.append(chunk)

        # Sort by similarity score descending (Chroma should already do this, but just to be sure)
        candidates.sort(key=lambda x: x.similarity_score, reverse=True)

        latency = (time.time() - start_time) * 1000

        return RetrievalResult(
            query=query,
            top_k=top_k,
            latency_ms=latency,
            embedding_latency_ms=embed_latency,
            search_latency_ms=search_latency,
            chunks=candidates,
        )


class OptionalReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
    ):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        logger.info(f"Loading cross-encoder model: {self.model_name}")
        self.reranker = CrossEncoder(self.model_name, device=device)

    def rerank(
        self, query: str, candidates: List[RetrievedChunk], top_k: int
    ) -> RetrievalResult:
        if not candidates:
            return RetrievalResult(query=query, top_k=top_k, latency_ms=0, chunks=[])

        start_time = time.time()

        cross_inp = [[query, chunk.text] for chunk in candidates]
        cross_scores = self.reranker.predict(cross_inp)

        for i in range(len(candidates)):
            candidates[i].similarity_score = float(cross_scores[i])

        candidates.sort(key=lambda x: x.similarity_score, reverse=True)
        reranked_chunks = candidates[:top_k]

        latency = (time.time() - start_time) * 1000

        return RetrievalResult(
            query=query,
            top_k=top_k,
            latency_ms=latency,
            rerank_latency_ms=latency,
            chunks=reranked_chunks,
        )


class HybridRetriever:
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_index_path: str,
    ):
        import pickle

        self.dense_retriever = dense_retriever
        logger.info(f"Loading BM25 index from {bm25_index_path}")
        with open(bm25_index_path, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]
        self.bm25_chunks = bm25_data["chunks"]

    def retrieve(
        self, query: str, top_k: int = 5, tenant_id: Optional[str] = None
    ) -> RetrievalResult:
        start_time = time.time()

        # 1. Dense Retrieval (get more candidates for RRF)
        dense_result = self.dense_retriever.retrieve(
            query, top_k=top_k * 4, tenant_id=tenant_id
        )
        dense_chunks = dense_result.chunks

        # 2. Sparse Retrieval (BM25)
        sparse_start = time.time()
        import re

        query_cleaned = re.sub(r"[^\w\s]", " ", query.lower())
        tokenized_query = query_cleaned.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        import numpy as np

        top_n = min(top_k * 4, len(bm25_scores))
        # Get indices of top_n scores
        top_indices = np.argsort(bm25_scores)[::-1][:top_n]

        sparse_chunks = []
        for idx in top_indices:
            score = bm25_scores[idx]
            if score <= 0:
                continue
            chunk_dict = self.bm25_chunks[idx]

            # Security filter for Multi-Tenancy
            vis = chunk_dict.get("visibility", "public")
            t_id = chunk_dict.get("tenant_id")
            if vis == "private" and t_id != tenant_id:
                continue

            # EmbeddedChunk fields are flattened
            source_doc = chunk_dict.get(
                "source_url", chunk_dict.get("source_document", "")
            )

            sparse_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_dict["chunk_id"],
                    source_document=source_doc,
                    text=chunk_dict["chunk_text"],
                    similarity_score=float(score),
                    metadata=chunk_dict,  # Pass the entire dict as metadata
                )
            )

        sparse_latency = (time.time() - sparse_start) * 1000

        # 3. Reciprocal Rank Fusion (RRF)
        # k=60 is the industry standard constant for RRF. It mitigates the impact of high-ranking outliers
        # and ensures a balanced fusion of both dense and sparse retrieval ranks.
        rrf_k = 60
        scores = {}
        chunk_map = {}

        for rank, chunk in enumerate(dense_chunks):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (
                rrf_k + rank + 1
            )
            chunk_map[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(sparse_chunks):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (
                rrf_k + rank + 1
            )
            if chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk

        # Sort by RRF score descending
        sorted_chunk_ids = sorted(
            scores.keys(), key=lambda cid: scores[cid], reverse=True
        )

        final_chunks = []
        for cid in sorted_chunk_ids[:top_k]:
            c = chunk_map[cid]
            # Replace original similarity score with the RRF score
            c.similarity_score = scores[cid]
            final_chunks.append(c)

        latency = (time.time() - start_time) * 1000

        return RetrievalResult(
            query=query,
            top_k=top_k,
            latency_ms=latency,
            embedding_latency_ms=dense_result.embedding_latency_ms,
            search_latency_ms=dense_result.search_latency_ms + sparse_latency,
            chunks=final_chunks,
        )
