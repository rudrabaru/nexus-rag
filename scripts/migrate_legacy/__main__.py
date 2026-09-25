"""
One-off migration: the legacy Qdrant collection and SQLite registry -> Postgres.

    python -m scripts.migrate_legacy --dry-run   # read both sources, print the plan, write nothing
    python -m scripts.migrate_legacy             # write in one transaction, then verify

- All-or-nothing: every row is written in a single transaction.
- Idempotent: every write is an upsert or insert-if-absent, so re-running converges.
- Verified: after writing, it checks every chunk arrived, per-document chunk counts, the
  float32 -> float16 (halfvec) precision loss, and that each sampled vector retrieves itself.

Reads QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME and DATABASE_URL (from .env too).
Requires `alembic upgrade head` first.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sqlalchemy import Text, bindparam, cast, func, select, text
from sqlalchemy.dialects.postgresql import insert
from pgvector.sqlalchemy import HALFVEC

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

from scripts.migrate_legacy.plan import build_plan  # noqa: E402
from scripts.migrate_legacy.read_legacy import read_qdrant_points, read_registry  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.embedding.config import EmbeddingConfig  # noqa: E402
from src.registry.engine import get_sync_engine  # noqa: E402
from src.registry.schema import EMBEDDING_DIMENSION, api_keys, chunks, documents, jobs, query_logs, tenants  # noqa: E402
from src.registry.schema_version import assert_schema_current  # noqa: E402
from src.retrieving.chunk_store import vector_literal  # noqa: E402
from src.retrieving.chunk_writes import WRITE_BATCH_SIZE, chunk_upsert_statement  # noqa: E402

VERIFY_SAMPLE_SIZE = 25
# float16 keeps ~3 significant decimal digits; on unit-length 1024-d vectors that bounds the
# cosine between the stored and original vector far above this floor.
MIN_ROUND_TRIP_COSINE = 0.9999
SELF_RETRIEVAL_MAX_DISTANCE = 1e-3


def write(plan, engine) -> None:
    with engine.begin() as conn:
        if plan["tenants"]:
            stmt = insert(tenants)
            conn.execute(
                stmt.on_conflict_do_update(
                    index_elements=[tenants.c.tenant_id],
                    set_={"total_embedding_tokens": func.greatest(tenants.c.total_embedding_tokens, stmt.excluded.total_embedding_tokens)},
                ),
                plan["tenants"],
            )
        if plan["documents"]:
            stmt = insert(documents)
            conn.execute(
                stmt.on_conflict_do_update(
                    index_elements=[documents.c.doc_id],
                    set_={c.name: stmt.excluded[c.name] for c in documents.columns if c.name != "doc_id"},
                ),
                plan["documents"],
            )
        if plan["jobs"]:
            conn.execute(insert(jobs).on_conflict_do_nothing(index_elements=[jobs.c.job_id]), plan["jobs"])
        for start in range(0, len(plan["chunks"]), WRITE_BATCH_SIZE):
            conn.execute(chunk_upsert_statement(), plan["chunks"][start:start + WRITE_BATCH_SIZE])
        if plan["api_keys"]:
            conn.execute(insert(api_keys).on_conflict_do_nothing(index_elements=[api_keys.c.key_hash]), plan["api_keys"])
        if plan["query_logs"]:
            conn.execute(insert(query_logs).on_conflict_do_nothing(index_elements=[query_logs.c.log_id]), plan["query_logs"])
            # Explicit log_ids were inserted, so move the identity past them.
            conn.execute(text("SELECT setval(pg_get_serial_sequence('query_logs', 'log_id'), (SELECT max(log_id) FROM query_logs))"))


def verify(plan, engine) -> dict:
    result = {}
    with engine.connect() as conn:
        stored = set(conn.execute(select(chunks.c.tenant_id, chunks.c.chunk_id)).tuples())
        expected = {(c["tenant_id"], c["chunk_id"]) for c in plan["chunks"]}
        result["chunks_missing_after_write"] = len(expected - stored)

        counts = {doc_id: n for doc_id, n in conn.execute(select(chunks.c.doc_id, func.count()).group_by(chunks.c.doc_id))}
        planned = {}
        for c in plan["chunks"]:
            planned[c["doc_id"]] = planned.get(c["doc_id"], 0) + 1
        result["documents_with_wrong_chunk_count"] = sum(1 for d, n in planned.items() if counts.get(d) != n)

        sample = random.Random(0).sample(plan["chunks"], min(VERIFY_SAMPLE_SIZE, len(plan["chunks"])))
        cosines, self_hits = [], 0
        for c in sample:
            stored_vec = conn.execute(
                select(chunks.c.embedding).where(chunks.c.tenant_id == c["tenant_id"], chunks.c.chunk_id == c["chunk_id"])
            ).scalar_one()
            a, b = np.asarray(c["embedding"], dtype=np.float64), np.asarray(stored_vec, dtype=np.float64)
            cosines.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))

            query_vector = cast(bindparam("v", vector_literal(c["embedding"]), type_=Text), HALFVEC(EMBEDDING_DIMENSION))
            nearest = conn.execute(
                select(chunks.c.embedding.cosine_distance(query_vector).label("d"))
                .where(chunks.c.tenant_id == c["tenant_id"])
                .order_by("d")
                .limit(1)
            ).scalar_one()
            self_hits += nearest <= SELF_RETRIEVAL_MAX_DISTANCE

        result["min_round_trip_cosine"] = round(min(cosines), 8) if cosines else None
        result["self_retrieval"] = f"{self_hits}/{len(sample)}"
        result["pgvector_version"] = conn.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")).scalar()

    result["ok"] = (
        result["chunks_missing_after_write"] == 0
        and result["documents_with_wrong_chunk_count"] == 0
        and (result["min_round_trip_cosine"] or 1.0) >= MIN_ROUND_TRIP_COSINE
        and self_hits == len(sample)
    )
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Migrate the legacy Qdrant collection and SQLite registry to Postgres.")
    parser.add_argument("--registry", type=Path, default=Path("data/registry.db"), help="legacy SQLite registry")
    parser.add_argument("--dry-run", action="store_true", help="read and plan only; write nothing")
    args = parser.parse_args(argv)

    settings = get_settings()
    if not (settings.qdrant_url and settings.qdrant_api_key):
        print("QDRANT_URL and QDRANT_API_KEY are required.")
        return 2

    points = read_qdrant_points(settings.qdrant_url, settings.qdrant_api_key, settings.qdrant_collection_name)
    legacy = read_registry(args.registry)
    plan = build_plan(points, legacy, EmbeddingConfig().model_name, EMBEDDING_DIMENSION)
    print("PLAN\n" + json.dumps(plan["report"], indent=2))
    if args.dry_run:
        return 0

    if not settings.database_url:
        print("DATABASE_URL is required.")
        return 2
    engine = get_sync_engine()
    assert_schema_current(engine)
    write(plan, engine)
    outcome = verify(plan, engine)
    print("VERIFY\n" + json.dumps(outcome, indent=2))
    return 0 if outcome["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
