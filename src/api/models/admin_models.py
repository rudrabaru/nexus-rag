from typing import Optional

from pydantic import BaseModel, Field

from src.registry.auth_store import TENANT_ID_PATTERN


class IssueKeyRequest(BaseModel):
    tenant_id: Optional[str] = Field(
        default=None,
        pattern=TENANT_ID_PATTERN.pattern,
        description="Tenant to issue a key for. A new UUID is generated when omitted.",
    )


class IssueKeyResponse(BaseModel):
    tenant_id: str
    api_key: str
