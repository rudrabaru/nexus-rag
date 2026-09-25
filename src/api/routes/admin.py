import asyncio
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends

from src.api.auth import get_admin_tenant
from src.api.dependencies import get_auth_store
from src.api.models.admin_models import IssueKeyRequest, IssueKeyResponse, RevokeKeysRequest, RevokeKeysResponse
from src.registry.auth_store import AuthStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/keys", response_model=IssueKeyResponse)
async def issue_api_key(
    body: Optional[IssueKeyRequest] = None,
    _: None = Depends(get_admin_tenant),
    auth_store: AuthStore = Depends(get_auth_store),
):
    """Issues a tenant API key. The key is shown once; only its hash is stored."""
    tenant_id = (body.tenant_id if body else None) or str(uuid.uuid4())
    api_key = await asyncio.to_thread(auth_store.create_api_key, tenant_id)
    logger.info(f"Admin issued an API key for tenant {tenant_id}")
    return IssueKeyResponse(tenant_id=tenant_id, api_key=api_key)


@router.post("/keys/revoke", response_model=RevokeKeysResponse)
async def revoke_api_keys(
    body: RevokeKeysRequest,
    _: None = Depends(get_admin_tenant),
    auth_store: AuthStore = Depends(get_auth_store),
):
    """Revokes one key (by its value) or every key of a tenant."""
    if body.api_key:
        revoked = await asyncio.to_thread(auth_store.revoke_api_key, body.api_key)
    else:
        revoked = await asyncio.to_thread(auth_store.revoke_tenant_keys, body.tenant_id)
    logger.info(f"Admin revoked {revoked} API key(s)" + (f" for tenant {body.tenant_id}" if body.tenant_id else ""))
    return RevokeKeysResponse(revoked=revoked)
