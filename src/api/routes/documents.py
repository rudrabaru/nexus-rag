import os
import logging
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import List
import asyncio

from typing import Optional
from fastapi import Depends, Security
from src.retrieving.vector_store import ChromaDBManager
from src.ingestion.asset_store import LocalAssetStore
from src.registry.database import DocumentRegistry
from src.api.dependencies import get_registry, get_retriever, get_auth_store
from src.registry.auth_store import AuthStore
from src.api.auth import get_current_tenant_from_admin_or_user, admin_api_key_header, api_key_header

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


class WorkspaceStatsResponse(BaseModel):
    documents_count: int
    total_chunks: int
    total_tokens: int

@router.get("/stats", response_model=WorkspaceStatsResponse)
async def get_workspace_stats(
    request: Request,
    registry: DocumentRegistry = Depends(get_registry),
    tenant_id: Optional[str] = Depends(get_current_tenant_from_admin_or_user),
):
    """Get aggregated workspace stats for the tenant."""
    docs = registry.list_documents(tenant_id=tenant_id)
    total_chunks = 0
    total_tokens = 0
    
    for d in docs:
        total_chunks += len(d.get("chunk_ids", []))
        stats = d.get("stats", {}) or {}
        total_tokens += stats.get("total_tokens", 0)

    return WorkspaceStatsResponse(
        documents_count=len(docs),
        total_chunks=total_chunks,
        total_tokens=total_tokens
    )

@router.get("", response_model=List[DocumentResponse])
async def list_documents(
    request: Request,
    registry: DocumentRegistry = Depends(get_registry),
    tenant_id: Optional[str] = Depends(get_current_tenant_from_admin_or_user),
):
    """List documents for the current tenant."""
    docs = registry.list_documents(tenant_id=tenant_id)

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
async def delete_document(
    doc_id: str,
    request: Request,
    registry: DocumentRegistry = Depends(get_registry),
    retriever = Depends(get_retriever),
    auth_store: AuthStore = Depends(get_auth_store),
    admin_key: str = Security(admin_api_key_header),
    api_key: str = Security(api_key_header),
):
    """
    Deletes a document from:
    1. Vector store
    2. Asset store
    3. Document Registry
    """
    expected_admin_key = os.environ.get("RAG_API_KEY")
    is_admin = admin_key and expected_admin_key and admin_key == expected_admin_key
    
    tenant_id = None
    if not is_admin:
        if api_key:
            tenant_id = auth_store.validate_api_key(api_key)
            if not tenant_id:
                raise HTTPException(status_code=401, detail="Invalid API Key.")
        else:
            raise HTTPException(status_code=401, detail="Authentication required. Please provide a valid API key.")
    doc = registry.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if not is_admin and doc.get("tenant_id") != tenant_id:
        raise HTTPException(status_code=403, detail="You do not own this document.")

    try:
        # Fix Bug 3: Use explicit chunk IDs
        chunk_ids = doc.get("chunk_ids", [])
        if chunk_ids:
            db_manager = getattr(retriever, "vector_store", getattr(getattr(retriever, "dense_retriever", None), "vector_store", None))
            if db_manager:
                await asyncio.to_thread(db_manager.collection.delete, ids=chunk_ids)

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
