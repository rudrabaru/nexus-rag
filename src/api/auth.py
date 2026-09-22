import secrets
from typing import Optional

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from src.api.dependencies import get_auth_store
from src.config import get_settings
from src.registry.auth_store import AuthStore

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
admin_api_key_header = APIKeyHeader(name="RAG-API-KEY", auto_error=False)


def get_real_ip(request: Request) -> str:
    client_host = request.client.host if request.client else "127.0.0.1"

    if get_settings().trust_proxies:
        if "x-forwarded-for" in request.headers:
            return request.headers["x-forwarded-for"].split(",")[0].strip()
        if "x-real-ip" in request.headers:
            return request.headers["x-real-ip"].strip()

    return client_host


def get_rate_limit_key(request: Request) -> str:
    """
    Authenticated callers are limited per tenant, which cannot be spoofed with a header.
    Anonymous callers fall back to the client IP. The auth dependency runs before the
    rate-limited handler, so request.state.tenant_id is already populated here.
    """
    tenant_id = getattr(request.state, "tenant_id", None)
    return f"tenant:{tenant_id}" if tenant_id else f"ip:{get_real_ip(request)}"


def _is_admin_key(candidate: Optional[str]) -> bool:
    expected = get_settings().effective_admin_key
    return bool(candidate and expected and secrets.compare_digest(candidate, expected))


async def get_admin_tenant(admin_key: str = Security(admin_api_key_header)) -> None:
    """Requires the admin key. Used for administrative routes."""
    if not get_settings().effective_admin_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server is missing admin key configuration.",
        )
    if not _is_admin_key(admin_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin authentication required.",
        )
    return None


async def get_current_tenant(
    request: Request,
    api_key: str = Security(api_key_header),
    auth_store: AuthStore = Depends(get_auth_store),
) -> Optional[str]:
    if not api_key:
        return None

    tenant_id = auth_store.validate_api_key(api_key)
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key. Ask your administrator for a valid key.",
        )
    request.state.tenant_id = tenant_id
    return tenant_id


async def get_current_tenant_from_admin_or_user(
    request: Request,
    admin_key: str = Security(admin_api_key_header),
    api_key: str = Security(api_key_header),
    auth_store: AuthStore = Depends(get_auth_store),
) -> Optional[str]:
    """Returns None for the admin (meaning: all tenants) or the tenant_id for a valid user key."""
    if _is_admin_key(admin_key):
        return None
    if api_key:
        tenant_id = auth_store.validate_api_key(api_key)
        if tenant_id:
            request.state.tenant_id = tenant_id
            return tenant_id
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Not authenticated. Provide a valid X-API-Key or RAG-API-KEY.",
    )
