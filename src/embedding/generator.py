import json
import logging
import time
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer

from .config import EmbeddingConfig
from .models import EmbeddedChunk, EmbeddingReport
from src.chunking.metadata import ChunkMetadata

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings for text chunks using sentence-transformers.
    """

    def __init__(
        self, config: EmbeddingConfig = None, model: SentenceTransformer = None
    ):
        self.config = config or EmbeddingConfig()

        if model:
            logger.info(f"Using shared embedding model: {self.config.model_name}")
            self.model = model
        else:
            logger.info(
                f"Loading embedding model: {self.config.model_name} on {self.config.device}"
            )
            self.model = SentenceTransformer(
                self.config.model_name, device=self.config.device
            )

        # Track statistics
        self.stats = {
            "total_chunks_processed": 0,
            "total_failures": 0,
            "total_tokens": 0,
            "start_time": None,
            "end_time": None,
        }

    def generate_embeddings(self, chunks: List[ChunkMetadata]) -> List[EmbeddedChunk]:
        """
        Takes a list of ChunkMetadata and returns a list of EmbeddedChunk with vectors.
        """
        valid_chunks = [c for c in chunks if c.chunk_text.strip()]
        if len(valid_chunks) < len(chunks):
            logger.warning(
                f"Skipped {len(chunks) - len(valid_chunks)} empty chunks before embedding."
            )

        if not valid_chunks:
            return []

        texts = []
        for chunk in valid_chunks:
            context = f"Document: {chunk.title}"
            if chunk.heading_path:
                context += f" | Section: {' > '.join(chunk.heading_path)}"
            texts.append(f"{context}\n\n{chunk.chunk_text}")

        # BGE models require a prefix for documents to match training distribution
        if "bge" in self.config.model_name.lower():
            prefix = "Represent this passage for retrieval: "
            texts = [prefix + text for text in texts]

        try:
            logger.debug(f"Encoding batch of {len(valid_chunks)} chunks...")
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=False,
            )

            embedded_chunks = []
            for chunk, embedding in zip(valid_chunks, embeddings):
                try:
                    # Convert embedding to list of floats
                    emb_list = embedding.tolist()

                    embedded_chunk = EmbeddedChunk(
                        **chunk.model_dump(),
                        embedding=emb_list,
                        embedding_model=self.config.model_name,
                    )
                    embedded_chunks.append(embedded_chunk)
                    self.stats["total_chunks_processed"] += 1
                    self.stats["total_tokens"] += chunk.token_count
                except Exception as e:
                    logger.error(
                        f"Failed to create EmbeddedChunk for {chunk.chunk_id}: {e}"
                    )
                    self.stats["total_failures"] += 1

            return embedded_chunks

        except Exception as e:
            logger.error(f"Failed to encode batch: {e}")
            self.stats["total_failures"] += len(chunks)
            return []
