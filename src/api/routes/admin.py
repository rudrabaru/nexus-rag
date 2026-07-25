import logging
from fastapi import APIRouter, Request, HTTPException, Depends
from src.api.auth import get_current_tenant
import asyncio

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])

@router.post("/rebuild-registry")
async def rebuild_registry(
    request: Request,
    tenant_id: str = Depends(get_current_tenant)
):
    """
    Rebuilds the SQLite registry from Qdrant payloads.
    This restores documents and full-text search indexing on Render Free Tier 
    where SQLite is ephemeral but Qdrant is persistent.
    """
    if not tenant_id:
         raise HTTPException(status_code=401, detail="Authentication required")
         
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
