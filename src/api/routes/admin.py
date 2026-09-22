import asyncio
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.auth import get_admin_tenant
from src.api.dependencies import get_auth_store
from src.api.models.admin_models import IssueKeyRequest, IssueKeyResponse
from src.registry.auth_store import AuthStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/keys", response_model=IssueKeyResponse)
async def issue_api_key(
    body: Optional[IssueKeyRequest] = None,
    _: None = Depends(get_admin_tenant),
    auth_store: AuthStore = Depends(get_auth_store),
):
    """Issues a tenant API key. This replaces the former open /register endpoint."""
    tenant_id = (body.tenant_id if body else None) or str(uuid.uuid4())
    logger.info(f"Admin issued an API key for tenant {tenant_id}")
    return IssueKeyResponse(tenant_id=tenant_id, api_key=auth_store.create_api_key(tenant_id))


@router.post("/rebuild-registry")
async def rebuild_registry(
    request: Request,
    _: None = Depends(get_admin_tenant)
):
    """
    Rebuilds the SQLite registry from Qdrant payloads.
    This restores documents and full-text search indexing on Render Free Tier 
    where SQLite is ephemeral but Qdrant is persistent.
    """
    registry = getattr(request.app.state, "registry", None)
    vector_store = getattr(request.app.state, "vector_store", None)
    
    if not registry or not vector_store:
        raise HTTPException(status_code=500, detail="Database managers not initialized")
        
    try:
        rebuilt_count = await asyncio.to_thread(registry.rebuild_registry_from_qdrant, vector_store)
        return {"status": "success", "message": f"Successfully reconstructed registry from {rebuilt_count} chunks in Qdrant."}
    except Exception as e:
        logger.error(f"Failed to rebuild registry: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to rebuild registry: {str(e)}")
