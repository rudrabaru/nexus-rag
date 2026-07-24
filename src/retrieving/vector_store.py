import json
import logging
import os
import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from src.embedding.models import EmbeddedChunk

logger = logging.getLogger(__name__)


class QdrantManager:
    """
    Manages interaction with Qdrant Cloud for storing and retrieving embedded chunks.
    """

    def __init__(
        self,
        collection_name: str = None,
        distance_metric: str = "cosine",
        dimension: int = 1024,
    ):
        self.collection_name = collection_name or os.environ.get("QDRANT_COLLECTION_NAME", "nexus_rag_collection")
        self.distance_metric = distance_metric
        self.dimension = dimension

        qdrant_url = os.environ.get("QDRANT_URL")
        qdrant_api_key = os.environ.get("QDRANT_API_KEY")

        if not qdrant_url or not qdrant_api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in environment variables.")

        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=120.0)
        logger.info(f"Initialized Qdrant Cloud Client connected to {qdrant_url}")

        # Ensure collection exists with correct configuration
        try:
            if not self.client.collection_exists(self.collection_name):
                # Map distance metric string to Qdrant enum
                qdrant_distance = rest.Distance.COSINE
                if self.distance_metric.lower() == "l2":
                    qdrant_distance = rest.Distance.EUCLID
                elif self.distance_metric.lower() == "ip":
                    qdrant_distance = rest.Distance.DOT

                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=rest.VectorParams(
                        size=self.dimension, 
                        distance=qdrant_distance
                    )
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="tenant_id",
                    field_schema=rest.PayloadSchemaType.KEYWORD,
                )
                logger.info(
                    f"Created Qdrant collection '{self.collection_name}' with tenant_id index"
                )
            else:
                logger.info(f"Using existing Qdrant collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error checking or creating Qdrant collection: {e}")

    def _prepare_payload(self, chunk: EmbeddedChunk) -> Dict[str, Any]:
        """
        Prepares metadata for Qdrant payload. Qdrant supports complex JSON structures.
        """
        payload = {
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
            "visibility": "private",
            "tenant_id": chunk.tenant_id or "unknown",
            "document_text": chunk.chunk_text, # Store the actual text in the payload
        }

        # Store heading_path as a JSON array string for lossless round-trip parsing.
        payload["heading_path"] = json.dumps(chunk.heading_path)

        return payload

    def load_chunks(self, chunks: List[EmbeddedChunk]) -> int:
        """
        Loads a list of EmbeddedChunks into the Qdrant collection.
        Returns the number of successfully added chunks.
        """
        if not chunks:
            return 0

        points = []
        for chunk in chunks:
            # Qdrant IDs must be UUID or unsigned integer. We generate a deterministic UUID from the chunk_id string.
            point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, chunk.chunk_id))
            points.append(
                rest.PointStruct(
                    id=point_id,
                    vector=chunk.embedding,
                    payload=self._prepare_payload(chunk)
                )
            )

        try:
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch_points = points[i:i+batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch_points
                )
            return len(points)
        except Exception as e:
            logger.error(f"Failed to load chunks into Qdrant: {e}")
            raise

    def get_collection_size(self) -> int:
        """Returns the number of items in the collection."""
        try:
            return self.client.get_collection(self.collection_name).points_count
        except Exception:
            return 0

    def search(self, query_embedding: List[float], top_k: int, tenant_id: str = None) -> List[Dict[str, Any]]:
        """
        Searches the Qdrant collection using strict tenant_id payload filtering.
        Returns a list of dicts with 'id', 'score', 'meta', and 'doc'.
        """
        must_conditions = []
        if tenant_id:
            must_conditions.append(
                rest.FieldCondition(
                    key="tenant_id",
                    match=rest.MatchValue(value=tenant_id)
                )
            )
        else:
            must_conditions.append(
                rest.FieldCondition(
                    key="tenant_id",
                    match=rest.MatchValue(value="__nonexistent__")
                )
            )

        qdrant_filter = rest.Filter(must=must_conditions)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=qdrant_filter,
            limit=top_k
        )

        output = []
        for res in results.points:
            output.append({
                "id": str(res.id),
                "dist": res.score, 
                "meta": res.payload,
                "doc": res.payload.get("document_text", "")
            })
        return output

    def delete_chunks(self, chunk_ids: List[str]) -> bool:
        """
        Deletes chunks from the Qdrant collection by their original chunk_ids.
        """
        if not chunk_ids:
            return True
            
        point_ids = [str(uuid.uuid5(uuid.NAMESPACE_OID, cid)) for cid in chunk_ids]
        
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=rest.PointIdsList(
                    points=point_ids
                )
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete chunks from Qdrant: {e}")
            return False
