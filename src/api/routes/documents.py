import asyncio
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.dependencies import get_registry
from src.api.models.document_models import DocumentResponse, WorkspaceStatsResponse
from src.api.rate_limit import READ_LIMIT, limiter
from src.api.security import Principal, require_principal
from src.registry.database import DocumentRegistry

router = APIRouter()


def _list_for(registry: DocumentRegistry, principal: Principal) -> list:
    return registry.list_all_documents() if principal.is_admin else registry.list_documents(principal.tenant_id)


@router.get("/stats", response_model=WorkspaceStatsResponse)
@limiter.limit(READ_LIMIT)
async def get_workspace_stats(
    request: Request,
    registry: DocumentRegistry = Depends(get_registry),
    principal: Principal = Depends(require_principal),
):
    """Aggregated workspace stats for the caller's workspace (every workspace for the admin)."""
    docs = await asyncio.to_thread(_list_for, registry, principal)
    return WorkspaceStatsResponse(
        documents_count=len(docs),
        total_chunks=sum(d["chunk_count"] for d in docs),
        total_tokens=sum((d.get("stats") or {}).get("total_tokens", 0) for d in docs),
    )


@router.get("", response_model=List[DocumentResponse])
@limiter.limit(READ_LIMIT)
async def list_documents(
    request: Request,
    registry: DocumentRegistry = Depends(get_registry),
    principal: Principal = Depends(require_principal),
):
    """Documents of the caller's workspace (every workspace for the admin)."""
    docs = await asyncio.to_thread(_list_for, registry, principal)
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
@limiter.limit(READ_LIMIT)
async def delete_document(
    request: Request,
    doc_id: str,
    registry: DocumentRegistry = Depends(get_registry),
    principal: Principal = Depends(require_principal),
):
    """Deletes a document. Its chunks, vectors and sparse index entries go with it in one transaction."""
    doc = await asyncio.to_thread(registry.get_document, doc_id)
    # Another workspace's document is reported as missing, so document ids cannot be probed.
    if not doc or (not principal.is_admin and doc.get("tenant_id") != principal.tenant_id):
        raise HTTPException(status_code=404, detail="Document not found")

    await asyncio.to_thread(registry.delete_document, doc_id)
    return {
        "status": "success",
        "message": f"Deleted document {doc_id} and all related chunks.",
    }
