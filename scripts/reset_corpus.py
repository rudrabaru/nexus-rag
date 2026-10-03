"""
Wipe corpus data from the database named by DATABASE_URL, keeping the schema and API keys.

Deletes:
  documents (cascades to jobs, ingest_sources, fetched_pages, chunks)
  embedding_indexes
  fetch_log, query_logs, pipeline_events
  with --include-experiments: experiments (cascades to trials and runs) and the generation and
  judge caches, whose results refer to chunks that no longer exist

Keeps:
  api_keys, tenants, the schema revision

It names the database host and waits for you to type it before deleting anything.

Run:
    python -m scripts.reset_corpus [--dry-run] [--include-experiments]
"""
import argparse
import sys

from sqlalchemy import func, select, text
from sqlalchemy.engine import make_url

from src.config import get_settings
from src.db.engine import get_sync_engine
from src.db.schema import (
    chunks, documents, embedding_indexes, experiments, fetch_log, jobs, pipeline_events, query_logs,
)
from src.runtime import ConfigurationError, bootstrap

CORPUS_TABLES = {
    "documents": documents, "chunks": chunks, "jobs": jobs, "embedding_indexes": embedding_indexes,
    "fetch_log": fetch_log, "query_logs": query_logs, "pipeline_events": pipeline_events,
}
CORPUS_DELETES = ["documents", "embedding_indexes", "fetch_log", "query_logs", "pipeline_events"]  # documents cascades first
EXPERIMENT_DELETES = ["experiments", "generation_cache", "judge_cache"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="show counts, do nothing")
    parser.add_argument("--include-experiments", action="store_true", help="also delete experiment results and caches")
    args = parser.parse_args(argv)

    try:
        bootstrap("cli")
    except ConfigurationError as e:
        print(e)
        return 2

    host = make_url(get_settings().database_url.get_secret_value()).host or "localhost"
    tables = dict(CORPUS_TABLES)
    if args.include_experiments:
        tables["experiments"] = experiments

    engine = get_sync_engine()
    with engine.connect() as conn:
        counts = {name: conn.execute(select(func.count()).select_from(table)).scalar() for name, table in tables.items()}

    print(f"Database: {host}\nRows to delete:")
    for name, n in counts.items():
        print(f"  {name}: {n:,}")

    if args.dry_run:
        print("Dry run: nothing deleted.")
        return 0

    if input(f"Type the database host ({host}) to confirm: ").strip() != host:
        print("Not confirmed: nothing deleted.")
        return 1

    deletes = CORPUS_DELETES + (EXPERIMENT_DELETES if args.include_experiments else [])
    print("\nDeleting...")
    with engine.begin() as conn:
        for table in deletes:
            conn.execute(text(f"DELETE FROM {table}"))  # names come from the constants above, never from input

    print("Done. The schema and API keys are untouched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
