import time
import logging
import os
import httpx
from typing import Optional, Any

from src.retrieving.vector_store import ChromaDBManager
from src.embedding.config import EmbeddingConfig
from src.retrieving.models import RetrievedChunk, RetrievalResult

logger = logging.getLogger(__name__)

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
        self.jina_api_key = os.environ.get("JINA_API_KEY")

    async def retrieve(
        self, query: str, top_k: int = 5, tenant_id: Optional[str] = None, pipeline_logger: Optional[Any] = None
    ) -> RetrievalResult:
        start_time = time.time()

        embed_start = time.time()
        
        if not self.jina_api_key:
            raise ValueError("JINA_API_KEY not set for DenseRetriever.")
            
        import asyncio
        embedding_tokens = 0
        async with httpx.AsyncClient(timeout=10.0) as client:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = await client.post(
                        "https://api.jina.ai/v1/embeddings",
                        headers={"Authorization": f"Bearer {self.jina_api_key}"},
                        json={
                            "model": self.config.model_name,
                            "input": [query],
                            "task": "retrieval.query"
                        }
                    )
                    response.raise_for_status()
                    resp_json = response.json()
                    query_embedding = resp_json["data"][0]["embedding"]
                    embedding_tokens = resp_json.get("usage", {}).get("total_tokens", 0)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    await asyncio.sleep(2 ** attempt)
            
        embed_latency = (time.time() - embed_start) * 1000

        search_start = time.time()

        all_ids = []
        all_dists = []
        all_metas = []
        all_docs = []

        if tenant_id:
            where_filter = {"tenant_id": tenant_id}
        else:
            where_filter = {"tenant_id": "__nonexistent__"}

        collection_size = self.vector_store.collection.count()
        safe_top_k = min(top_k, collection_size)

        if safe_top_k == 0:
            search_latency = (time.time() - search_start) * 1000
            return RetrievalResult(
                query=query,
                top_k=top_k,
                latency_ms=(time.time() - start_time) * 1000,
                embedding_latency_ms=embed_latency,
                search_latency_ms=search_latency,
                chunks=[],
            )

        result = self.vector_store.collection.query(
            query_embeddings=[query_embedding],
            n_results=safe_top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
        if result["ids"] and len(result["ids"]) > 0:
            all_ids = result["ids"][0]
            all_dists = result["distances"][0]
            all_metas = result["metadatas"][0]
            all_docs = result["documents"][0]

        search_latency = (time.time() - search_start) * 1000

        merged = [
            {"id": all_ids[i], "dist": all_dists[i], "meta": all_metas[i], "doc": all_docs[i]}
            for i in range(len(all_ids))
        ]
        merged = merged[:top_k]

        candidates = []
        for item in merged:
            metric = getattr(self.vector_store, "distance_metric", "cosine")
            if metric == "l2":
                similarity = 1.0 / (1.0 + item["dist"])
            else:
                similarity = 1.0 - item["dist"] if item["dist"] <= 1.0 else 0.0

            chunk = RetrievedChunk(
                chunk_id=item["id"],
                source_document=item["meta"].get("source_document", ""),
                source_url=item["meta"].get("source_url"),
                text=item["doc"],
                similarity_score=similarity,
                metadata=item["meta"],
            )
            candidates.append(chunk)

        candidates.sort(key=lambda x: x.similarity_score, reverse=True)

        latency = (time.time() - start_time) * 1000

        if candidates:
            scores_str = ", ".join(f"{c.similarity_score:.4f}" for c in candidates[:5])
            logger.info(
                f"Retrieved {len(candidates)} chunks for tenant={tenant_id}"
                f" | metric={getattr(self.vector_store, 'distance_metric', 'cosine')}"
                f" | top-5 scores: [{scores_str}]"
            )

        return RetrievalResult(
            query=query,
            top_k=top_k,
            latency_ms=latency,
            embedding_latency_ms=embed_latency,
            search_latency_ms=search_latency,
            embedding_tokens=embedding_tokens,
            chunks=candidates,
        )
