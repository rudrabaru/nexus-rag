"""
Wipe all corpus data from Neon while keeping the schema and API keys intact.

Deletes:
  documents (cascades to jobs, ingest_sources, fetched_pages, chunks)
  embedding_indexes
  fetch_log, query_logs, pipeline_events

Keeps:
  api_keys, tenants   -- existing keys stay valid
  alembic_version     -- schema revision stays at 0003

Run:
    python -m scripts.reset_corpus [--dry-run]
"""
import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)

from sqlalchemy import func, select, text  # noqa: E402

from src.registry.engine import get_sync_engine  # noqa: E402
from src.registry.schema import (  # noqa: E402
    chunks, documents, embedding_indexes, fetch_log, jobs, pipeline_events, query_logs,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="show counts, do nothing")
    args = parser.parse_args(argv)

    engine = get_sync_engine()
    with engine.connect() as conn:
        counts = {
            "documents":        conn.execute(select(func.count()).select_from(documents)).scalar(),
            "chunks":           conn.execute(select(func.count()).select_from(chunks)).scalar(),
            "jobs":             conn.execute(select(func.count()).select_from(jobs)).scalar(),
            "embedding_indexes":conn.execute(select(func.count()).select_from(embedding_indexes)).scalar(),
            "fetch_log":        conn.execute(select(func.count()).select_from(fetch_log)).scalar(),
            "query_logs":       conn.execute(select(func.count()).select_from(query_logs)).scalar(),
            "pipeline_events":  conn.execute(select(func.count()).select_from(pipeline_events)).scalar(),
        }

    print("Rows to delete:")
    for table, n in counts.items():
        print(f"  {table}: {n:,}")

    if args.dry_run:
        print("Dry run: nothing deleted.")
        return 0

    print("\nDeleting...")
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM documents"))          # cascades to jobs, ingest_sources, fetched_pages, chunks
        conn.execute(text("DELETE FROM embedding_indexes"))  # now safe (chunks gone)
        conn.execute(text("DELETE FROM fetch_log"))
        conn.execute(text("DELETE FROM query_logs"))
        conn.execute(text("DELETE FROM pipeline_events"))

    print("Done. The schema and API keys are untouched.")
    print("Ingest fresh documents with EMBEDDING_PROVIDER=voyage (voyage:voyage-4 index).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
