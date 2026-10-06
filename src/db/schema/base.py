"""Shared pieces of the schema: the one MetaData, column helpers and the constants every table agrees on."""
from sqlalchemy import MetaData, text
from sqlalchemy.dialects.postgresql import JSONB

# One width for every index: voyage-4 (output_dimension=1024) and bge-m3 both produce it. A model
# with another width needs its own column; src/embedding/providers.py refuses mismatched vectors
# instead of letting the insert fail.
EMBEDDING_DIMENSION = 1024

# 'english' stemming. Known limitation: non-English corpora are tokenized with English rules.
TEXT_SEARCH_CONFIG = "english"

# Python None is stored as SQL NULL, not the JSON value `null`. With the default, a None
# became 'null'::jsonb, and `'null'::jsonb || '{...}'` is array concatenation, so merged job
# metadata turned into [null, {...}] (caught by the integration suite on real Postgres).
Json = JSONB(none_as_null=True)

metadata = MetaData(
    naming_convention={
        "ix": "ix_%(table_name)s_%(column_0_N_name)s",
        "uq": "uq_%(table_name)s_%(column_0_N_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
)


def now():
    return text("now()")
