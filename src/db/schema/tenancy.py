"""Who owns data: tenants (usage counters) and their API keys."""
from sqlalchemy import BigInteger, Column, DateTime, Table, Text

from src.db.schema.base import metadata

# A tenant exists when it holds a key; this table only carries usage counters and is populated
# on first use.
tenants = Table(
    "tenants",
    metadata,
    Column("tenant_id", Text, primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("total_embedding_tokens", BigInteger, nullable=False, default=0),
)

# Only portable column types and no foreign keys, so the key store runs the same SQL against
# SQLite in unit tests.
api_keys = Table(
    "api_keys",
    metadata,
    Column("key_hash", Text, primary_key=True),  # sha256 hex of the full key
    Column("tenant_id", Text, nullable=False, index=True),
    Column("key_prefix", Text),  # first characters, for identifying a key without storing it
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("revoked_at", DateTime(timezone=True)),
)
