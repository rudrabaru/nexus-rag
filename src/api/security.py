"""
Who is calling. Every protected route depends on exactly one of these, so a route cannot
forget to authenticate or hand-roll its own check:

- require_tenant: a valid workspace API key (X-API-Key). Returns the tenant id, never None.
- require_admin: the admin key (X-Admin-Key).
- require_principal: either, for routes an admin may also use. The admin is an explicit
  Principal(is_admin=True), not a missing tenant, so "no tenant" can never mean "everyone".

A missing or wrong credential is always HTTP 401.
"""
import asyncio
import secrets
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from src.api.dependencies import get_auth_store
from src.api.rate_limit import auth_failures, client_ip
from src.config import get_settings
from src.stores.api_keys import AuthStore

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
admin_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)


@dataclass(frozen=True)
class Principal:
    is_admin: bool
    tenant_id: Optional[str] = None


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(status_code=401, detail=detail, headers={"WWW-Authenticate": "ApiKey"})


def _is_admin_key(candidate: Optional[str]) -> bool:
    expected = get_settings().admin_api_key.get_secret_value()
    # Compared as bytes: compare_digest rejects non-ASCII str, which a header can carry.
    return bool(candidate and expected and secrets.compare_digest(candidate.encode(), expected.encode()))


async def _tenant_of(request: Request, api_key: Optional[str], auth_store: AuthStore) -> Optional[str]:
    if not api_key:
        return None
    tenant_id = await asyncio.to_thread(auth_store.validate_api_key, api_key)
    if tenant_id:
        request.state.tenant_id = tenant_id
    return tenant_id


async def require_tenant(
    request: Request,
    api_key: Optional[str] = Security(api_key_header),
    auth_store: AuthStore = Depends(get_auth_store),
) -> str:
    ip = client_ip(request)
    auth_failures.check(ip)
    tenant_id = await _tenant_of(request, api_key, auth_store)
    if not tenant_id:
        if api_key:
            auth_failures.record(ip)
        raise _unauthorized("A valid API key is required (X-API-Key header).")
    return tenant_id


async def require_admin(request: Request, admin_key: Optional[str] = Security(admin_key_header)) -> None:
    ip = client_ip(request)
    auth_failures.check(ip)
    if not _is_admin_key(admin_key):
        if admin_key:
            auth_failures.record(ip)
        raise _unauthorized("The admin key is required (X-Admin-Key header).")


async def require_principal(
    request: Request,
    admin_key: Optional[str] = Security(admin_key_header),
    api_key: Optional[str] = Security(api_key_header),
    auth_store: AuthStore = Depends(get_auth_store),
) -> Principal:
    ip = client_ip(request)
    auth_failures.check(ip)
    if _is_admin_key(admin_key):
        return Principal(is_admin=True)
    tenant_id = await _tenant_of(request, api_key, auth_store)
    if tenant_id:
        return Principal(is_admin=False, tenant_id=tenant_id)
    if api_key or admin_key:
        auth_failures.record(ip)
    raise _unauthorized("A valid API key (X-API-Key) or the admin key (X-Admin-Key) is required.")
