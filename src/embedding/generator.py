import logging
import os
import asyncio
import httpx
from typing import List

from .config import EmbeddingConfig
from .models import EmbeddedChunk
from src.chunking.metadata import ChunkMetadata

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings for text chunks using Jina API.
    """

    def __init__(
        self, config: EmbeddingConfig = None
    ):
        self.config = config or EmbeddingConfig()

        self.jina_api_key = os.environ.get("JINA_API_KEY")
        if not self.jina_api_key:
            logger.warning("JINA_API_KEY is not set. Embedding generation will fail.")

        logger.info(f"Loaded JINA API for embeddings: {self.config.model_name}")

        self.stats = {
            "total_chunks_processed": 0,
            "total_failures": 0,
            "total_tokens": 0,
            "start_time": None,
            "end_time": None,
        }
        
        # Optional semaphore injected later if needed
        self.embed_semaphore = None

    async def embed_batch(self, texts: list[str], task_type: str = "retrieval.passage") -> tuple[list[list[float]], list[int]]:
        all_embeddings = []
        failed_indices = []
        batch_size = self.config.batch_size
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                max_retries = self.config.max_retries
                for attempt in range(max_retries):
                    try:
                        if self.embed_semaphore:
                            await self.embed_semaphore.acquire()
                        try:
                            response = await client.post(
                                "https://api.jina.ai/v1/embeddings",
                                headers={"Authorization": f"Bearer {self.jina_api_key}"},
                                json={
                                    "model": self.config.model_name,
                                    "input": batch,
                                    "task": task_type
                                }
                            )
                            response.raise_for_status()
                        finally:
                            if self.embed_semaphore:
                                self.embed_semaphore.release()
                                
                        data = response.json()
                        all_embeddings.extend([item["embedding"] for item in data["data"]])
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            logger.error(f"Batch {i//batch_size} failed after {max_retries} attempts: {e}")
                            failed_indices.extend(range(i, i + len(batch)))
                            all_embeddings.extend([None] * len(batch))
                            break
                        # exponential backoff 1s, 2s, 4s
                        await asyncio.sleep(2 ** attempt)
                        
        return all_embeddings, failed_indices

    def generate_embeddings(self, chunks: List[ChunkMetadata]) -> tuple[List[EmbeddedChunk], list[int]]:
        """
        Takes a list of ChunkMetadata and returns a list of EmbeddedChunk with vectors and failed indices.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            raise RuntimeError("generate_embeddings must be run in a separate thread if an event loop is already running, or use generate_embeddings_async.")
        return asyncio.run(self.generate_embeddings_async(chunks))
        
    async def generate_embeddings_async(self, chunks: List[ChunkMetadata]) -> tuple[List[EmbeddedChunk], list[int]]:
        valid_chunks = [c for c in chunks if c.chunk_text.strip()]
        if len(valid_chunks) < len(chunks):
            logger.warning(
                f"Skipped {len(chunks) - len(valid_chunks)} empty chunks before embedding."
            )

        if not valid_chunks:
            return [], []
            
        if not self.jina_api_key:
            raise ValueError("JINA Client not initialized. Check JINA_API_KEY.")

        texts = []
        for chunk in valid_chunks:
            context = f"Document: {chunk.title}"
            if chunk.heading_path:
                context += f" | Section: {' > '.join(chunk.heading_path)}"
            text = f"{context}\n\n{chunk.chunk_text}"
            texts.append(text)

        embedded_chunks = []
        failed_indices = []
        try:
            logger.debug(f"Encoding {len(texts)} chunks via JINA API...")
            embeddings, failed_indices = await self.embed_batch(texts, task_type="retrieval.passage")
            
            for i, (chunk, emb) in enumerate(zip(valid_chunks, embeddings)):
                if emb is None:
                    continue
                embedded_chunk = EmbeddedChunk(
                    **chunk.model_dump(),
                    embedding=emb,
                    embedding_model=self.config.model_name,
                )
                embedded_chunks.append(embedded_chunk)
                self.stats["total_chunks_processed"] += 1
                self.stats["total_tokens"] += chunk.token_count
            
            if failed_indices:
                self.stats["total_failures"] += len(failed_indices)
        except Exception as e:
            logger.error(f"Failed to encode chunks: {e}")
            self.stats["total_failures"] += len(texts)
            failed_indices = list(range(len(texts)))

        return embedded_chunks, failed_indices
