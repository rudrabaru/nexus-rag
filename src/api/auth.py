import os
from typing import Optional
from fastapi import Security, HTTPException, status, Depends
from fastapi.security import APIKeyHeader
from fastapi import Request
from src.api.dependencies import get_auth_store
from src.registry.auth_store import AuthStore

def get_real_ip(request: Request) -> str:
    if "x-forwarded-for" in request.headers:
        return request.headers["x-forwarded-for"].split(",")[0].strip()
    if "x-real-ip" in request.headers:
        return request.headers["x-real-ip"].strip()
    if not hasattr(request, "client") or not request.client:
        return "127.0.0.1"
    return request.client.host

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
admin_api_key_header = APIKeyHeader(name="RAG-API-KEY", auto_error=False)

async def get_current_tenant(
    api_key: str = Security(api_key_header),
    auth_store: AuthStore = Depends(get_auth_store)
) -> str:
    if os.getenv("DEMO_MODE", "false").lower() == "true":
        return "demo"

    if not api_key:
        return None

    tenant_id = auth_store.validate_api_key(api_key)
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key. Generate one at POST /register.",
        )
    return tenant_id


async def get_current_tenant_from_admin_or_user(
    admin_key: str = Security(admin_api_key_header),
    api_key: str = Security(api_key_header),
    auth_store: AuthStore = Depends(get_auth_store),
) -> Optional[str]:
    if os.getenv("DEMO_MODE", "false").lower() == "true":
        return "demo"

    expected_admin_key = os.environ.get("RAG_API_KEY")
    if admin_key:
        if expected_admin_key and admin_key == expected_admin_key:
            return None  # Admin sees all
    if api_key:
        tenant_id = auth_store.validate_api_key(api_key)
        if tenant_id:
            return tenant_id
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key.",
        )
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Not authenticated. Provide a valid X-API-Key or RAG-API-KEY.",
    )
