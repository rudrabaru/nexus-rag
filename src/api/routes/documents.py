import logging
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import List

from src.retrieving.vector_store import ChromaDBManager
from src.ingestion.asset_store import LocalAssetStore
from src.registry.database import DocumentRegistry
from src.api.startup import app_state

router = APIRouter()
logger = logging.getLogger(__name__)


class DocumentResponse(BaseModel):
    id: str
    url: str
    title: str
    status: str
    created_at: str
    chunks: int
    stats: dict = {}


@router.get("", response_model=List[DocumentResponse])
async def list_documents(request: Request):
    """List all ingested documents in the registry."""
    if "registry" not in app_state:
        raise HTTPException(status_code=500, detail="Registry not initialized")

    registry: DocumentRegistry = app_state["registry"]
    docs = registry.list_documents()

    return [
        DocumentResponse(
            id=d["doc_id"],
            url=d["source"],
            title=d["source"].split("/")[-1] if "/" in d["source"] else d["source"],
            status=d["status"],
            created_at=d["ingested_at"],
            chunks=len(d["chunk_ids"]),
            stats=d.get("stats", {}) or {},
        )
        for d in docs
    ]


@router.delete("/{doc_id}")
async def delete_document(doc_id: str, request: Request):
    """
    Deletes a document from:
    1. Vector store
    2. Asset store
    3. Document Registry
    """
    if "registry" not in app_state:
        raise HTTPException(status_code=500, detail="Registry not initialized")

    registry: DocumentRegistry = app_state["registry"]
    doc = registry.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        # 1. Delete from Vector Store
        if "retriever" not in app_state:
            raise HTTPException(status_code=500, detail="Retriever not initialized")
        db_manager = app_state["retriever"].vector_store
        db_manager.collection.delete(where={"source_url": doc["source"]})

        # 2. Delete Assets
        asset_store = LocalAssetStore()
        asset_store.delete_doc_assets(doc_id)

        # 3. Delete from Registry
        registry.delete_document(doc_id)

        return {
            "status": "success",
            "message": f"Deleted document {doc_id} and all related chunks and assets.",
        }
    except Exception as e:
        logger.error(f"Failed to delete document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
