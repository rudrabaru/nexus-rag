import asyncio

from fastapi import APIRouter, Depends, Request

from src.api.dependencies import get_documents, get_query_log, get_workspace
from src.api.rate_limit import READ_LIMIT, limiter
from src.api.routes.documents import list_for
from src.api.schemas.workspace import (
    UsageEntry,
    UsageResponse,
    UsageSummary,
    WorkspaceRetrieval,
    WorkspaceSettingsResponse,
    WorkspaceStats,
)
from src.api.security import Principal, require_principal, require_tenant
from src.config import get_settings
from src.services.chat_config import WORKSPACE_FIELDS, environment_defaults
from src.stores.documents import DocumentStore
from src.stores.query_log import QueryLogStore
from src.stores.workspace import WorkspaceSettingsStore

router = APIRouter(prefix="/v1", tags=["workspace"])

RECENT_QUERIES = 100


@router.get("/workspace/stats", response_model=WorkspaceStats, operation_id="workspace_stats")
@limiter.limit(READ_LIMIT)
async def workspace_stats(
    request: Request,
    documents: DocumentStore = Depends(get_documents),
    principal: Principal = Depends(require_principal),
):
    """Documents, chunks and tokens in the caller's workspace (every workspace for the admin)."""
    docs = await asyncio.to_thread(list_for, documents, principal)
    return WorkspaceStats(
        documents_count=len(docs),
        total_chunks=sum(d["chunk_count"] for d in docs),
        total_tokens=sum((d.get("stats") or {}).get("total_tokens", 0) for d in docs),
    )


@router.get("/workspace/settings", response_model=WorkspaceSettingsResponse, operation_id="get_workspace_settings")
@limiter.limit(READ_LIMIT)
async def get_workspace_settings(
    request: Request,
    tenant_id: str = Depends(require_tenant),
    workspace: WorkspaceSettingsStore = Depends(get_workspace),
):
    """The retrieval settings chat runs for this workspace, and whether they are its own or the deployment's defaults."""
    chosen = await asyncio.to_thread(workspace.get_retrieval, tenant_id)
    effective = {**environment_defaults(get_settings()), **{k: v for k, v in (chosen or {}).items() if k in WORKSPACE_FIELDS}}
    return WorkspaceSettingsResponse(source="workspace" if chosen else "default", retrieval=WorkspaceRetrieval(**effective))


@router.put("/workspace/settings", response_model=WorkspaceSettingsResponse, operation_id="put_workspace_settings")
@limiter.limit(READ_LIMIT)
async def put_workspace_settings(
    request: Request,
    body: WorkspaceRetrieval,
    tenant_id: str = Depends(require_tenant),
    workspace: WorkspaceSettingsStore = Depends(get_workspace),
):
    """Sets the retrieval settings this workspace's chat runs (for example the winner of an experiment)."""
    await asyncio.to_thread(workspace.put_retrieval, tenant_id, body.chosen())
    return await get_workspace_settings(request, tenant_id, workspace)


@router.delete("/workspace/settings", response_model=WorkspaceSettingsResponse, operation_id="reset_workspace_settings")
@limiter.limit(READ_LIMIT)
async def reset_workspace_settings(
    request: Request,
    tenant_id: str = Depends(require_tenant),
    workspace: WorkspaceSettingsStore = Depends(get_workspace),
):
    """Returns this workspace to the deployment's defaults."""
    await asyncio.to_thread(workspace.clear_retrieval, tenant_id)
    return await get_workspace_settings(request, tenant_id, workspace)


@router.get("/usage", response_model=UsageResponse, operation_id="usage")
@limiter.limit(READ_LIMIT)
async def usage(
    request: Request,
    tenant_id: str = Depends(require_tenant),
    query_log: QueryLogStore = Depends(get_query_log),
):
    """Cost and latency of this workspace's questions: totals over its retained history, and the most recent ones."""
    summary = await asyncio.to_thread(query_log.summary, tenant_id)
    recent = await asyncio.to_thread(query_log.recent_queries, tenant_id, RECENT_QUERIES)
    return UsageResponse(
        summary=UsageSummary(**summary),
        queries=[UsageEntry(**{**row, "timestamp": str(row["timestamp"])}) for row in recent],
    )
