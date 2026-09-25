from typing import Optional

from pydantic import BaseModel, Field, model_validator

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


class RevokeKeysRequest(BaseModel):
    api_key: Optional[str] = Field(default=None, description="Revoke this key.")
    tenant_id: Optional[str] = Field(
        default=None, pattern=TENANT_ID_PATTERN.pattern, description="Revoke every key of this tenant."
    )

    @model_validator(mode="after")
    def exactly_one_target(self):
        if bool(self.api_key) == bool(self.tenant_id):
            raise ValueError("Provide exactly one of api_key or tenant_id.")
        return self


class RevokeKeysResponse(BaseModel):
    revoked: int
