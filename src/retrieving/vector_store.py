import json
import logging
from typing import List, Dict, Any
import chromadb
from src.embedding.models import EmbeddedChunk

logger = logging.getLogger(__name__)


class ChromaDBManager:
    """
    Manages interaction with ChromaDB for storing and retrieving embedded chunks.
    """

    def __init__(
        self,
        persist_directory: str = ".chroma_db",
        collection_name: str = "docs_v1",
        distance_metric: str = "cosine",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.distance_metric = distance_metric

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": self.distance_metric}
        )
        logger.info(
            f"Initialized ChromaDB at {persist_directory} with collection '{collection_name}' (space: {self.distance_metric})"
        )

    def _prepare_metadata(self, chunk: EmbeddedChunk) -> Dict[str, Any]:
        """
        Prepares metadata for ChromaDB by flattening lists and ensuring types are supported.
        ChromaDB only supports str, int, float, bool.
        """
        metadata = {
            "chunk_id": chunk.chunk_id,
            "source_document": chunk.source_document,
            "source_url": getattr(chunk, "source_url", ""),
            "title": chunk.title,
            "section_title": chunk.section_title,
            "contains_code": chunk.contains_code,
            "contains_table": chunk.contains_table,
            "content_type": chunk.content_type,
            "chunk_version": chunk.chunk_version,
            "document_version": chunk.document_version,
            "visibility": chunk.visibility or "public",
        }

        if chunk.tenant_id:
            metadata["tenant_id"] = chunk.tenant_id

        # Store heading_path as a JSON array string for lossless round-trip parsing.
        # Consumers must use json.loads() to reconstruct the list.
        metadata["heading_path"] = json.dumps(chunk.heading_path)

        return metadata

    def load_chunks(self, chunks: List[EmbeddedChunk]) -> int:
        """
        Loads a list of EmbeddedChunks into the ChromaDB collection.
        Returns the number of successfully added chunks.
        """
        if not chunks:
            return 0

        ids = []
        embeddings = []
        metadatas = []
        documents = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            embeddings.append(chunk.embedding)
            metadatas.append(self._prepare_metadata(chunk))
            documents.append(chunk.chunk_text)

        max_batch_size = 5000
        total_added = 0

        try:
            for i in range(0, len(ids), max_batch_size):
                self.collection.upsert(
                    ids=ids[i : i + max_batch_size],
                    embeddings=embeddings[i : i + max_batch_size],
                    metadatas=metadatas[i : i + max_batch_size],
                    documents=documents[i : i + max_batch_size],
                )
                total_added += len(ids[i : i + max_batch_size])
            return total_added
        except Exception as e:
            logger.error(f"Failed to load chunks into ChromaDB: {e}")
            return total_added

    def get_collection_size(self) -> int:
        """Returns the number of items in the collection."""
        return self.collection.count()
