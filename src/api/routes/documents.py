import asyncio
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.auth import get_current_tenant_from_admin_or_user
from src.api.dependencies import get_registry
from src.registry.database import DocumentRegistry

router = APIRouter()
logger = logging.getLogger(__name__)


class DocumentResponse(BaseModel):
    id: str
    url: str
    title: str
    status: str
    created_at: str
    chunks: int
    error: Optional[str] = None
    stats: dict = {}


class WorkspaceStatsResponse(BaseModel):
    documents_count: int
    total_chunks: int
    total_tokens: int


@router.get("/stats", response_model=WorkspaceStatsResponse)
async def get_workspace_stats(
    registry: DocumentRegistry = Depends(get_registry),
    tenant_id: Optional[str] = Depends(get_current_tenant_from_admin_or_user),
):
    """Get aggregated workspace stats for the tenant."""
    docs = await asyncio.to_thread(registry.list_documents, tenant_id)
    return WorkspaceStatsResponse(
        documents_count=len(docs),
        total_chunks=sum(d["chunk_count"] for d in docs),
        total_tokens=sum((d.get("stats") or {}).get("total_tokens", 0) for d in docs),
    )


@router.get("", response_model=List[DocumentResponse])
async def list_documents(
    registry: DocumentRegistry = Depends(get_registry),
    tenant_id: Optional[str] = Depends(get_current_tenant_from_admin_or_user),
):
    """List documents for the current tenant."""
    docs = await asyncio.to_thread(registry.list_documents, tenant_id)
    return [
        DocumentResponse(
            id=d["doc_id"],
            url=d["source"],
            title=d["source"].split("/")[-1] if "/" in d["source"] else d["source"],
            status=d["status"],
            created_at=d["ingested_at"],
            chunks=d["chunk_count"],
            error=d.get("error"),
            stats=d.get("stats") or {},
        )
        for d in docs
    ]


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: str,
    registry: DocumentRegistry = Depends(get_registry),
    tenant_id: Optional[str] = Depends(get_current_tenant_from_admin_or_user),
):
    """Deletes a document. Its chunks, vectors and sparse index entries go with it in one transaction."""
    is_admin = tenant_id is None

    doc = await asyncio.to_thread(registry.get_document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if not is_admin and doc.get("tenant_id") != tenant_id:
        raise HTTPException(status_code=403, detail="You do not own this document.")

    await asyncio.to_thread(registry.delete_document, doc_id)
    return {
        "status": "success",
        "message": f"Deleted document {doc_id} and all related chunks.",
    }
