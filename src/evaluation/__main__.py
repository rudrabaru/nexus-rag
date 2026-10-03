"""
    python -m src.evaluation run SPEC.json                 # create an experiment and run it
    python -m src.evaluation resume EXPERIMENT_ID          # continue a paused or interrupted one
    python -m src.evaluation report EXPERIMENT_ID [--json FILE] [--details] [--fail-on-regression]
    python -m src.evaluation list [--tenant T]

Results live in Postgres (DATABASE_URL); --json exports a report. --fail-on-regression exits 1
when any trial is significantly worse than the baseline: a regression gate for CI.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)

from src.config import get_settings  # noqa: E402
from src.evaluation import store  # noqa: E402
from src.evaluation.dataset import load_dataset  # noqa: E402
from src.evaluation.engine import run_experiment  # noqa: E402
from src.evaluation.ground_truth import missing_chunk_ids  # noqa: E402
from src.evaluation.report import build_report, regressions, render  # noqa: E402
from src.evaluation.spec import ExperimentSpec  # noqa: E402
from src.registry.engine import dispose_engines, get_async_engine, get_sync_engine  # noqa: E402
from src.registry.schema_version import assert_schema_current  # noqa: E402
from src.retrieving.pipeline import RetrievalResources  # noqa: E402


async def _execute(experiment_id: str) -> str:
    settings = get_settings()
    resources = RetrievalResources(settings, get_sync_engine(), get_async_engine())
    try:
        return await run_experiment(get_sync_engine(), resources, settings, experiment_id)
    finally:
        await dispose_engines()  # pooled async connections belong to this event loop


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="create an experiment from a spec file and run it")
    run.add_argument("spec")
    resume = commands.add_parser("resume", help="continue an experiment")
    resume.add_argument("experiment_id")
    report = commands.add_parser("report", help="print an experiment's report")
    report.add_argument("experiment_id")
    for command in (run, resume, report):
        command.add_argument("--json", help="also write the report to this file")
        command.add_argument("--details", action="store_true", help="include per-difficulty breakdowns")
        command.add_argument("--fail-on-regression", action="store_true", help="exit 1 if a trial is significantly worse")
    listing = commands.add_parser("list", help="recent experiments")
    listing.add_argument("--tenant")
    args = parser.parse_args(argv)

    missing = get_settings().missing_required()
    if missing:
        print(f"Missing or invalid configuration: {', '.join(missing)}")
        return 2
    engine = get_sync_engine()
    assert_schema_current(engine)

    if args.command == "list":
        for e in store.list_experiments(engine, args.tenant):
            print(f"{e['experiment_id']}  {e['status']:<9} {e['created_at'][:19]}  {e['tenant_id']:<20} {e['name']}")
        return 0

    if args.command == "run":
        spec = ExperimentSpec(**json.loads(Path(args.spec).read_text(encoding="utf-8")))
        dataset = load_dataset(spec.dataset, spec.relevance)
        if spec.relevance == "chunk":
            missing = missing_chunk_ids(engine, spec.tenant_id, (i for q in dataset.queries for i in q.source_chunk_ids))
            if missing:
                print(f"{len(missing)} source chunks are not in tenant {spec.tenant_id!r} (re-chunked or re-ingested since "
                      f"the test set was made?): {missing[:5]}. Regenerate the test set, or use relevance=document.")
                return 2
        experiment_id = store.create_experiment(engine, spec, dataset)
        print(f"experiment {experiment_id}: {len(spec.trials)} trials x {len(dataset.queries)} queries")
    else:
        experiment_id = args.experiment_id

    if args.command in ("run", "resume"):
        status = asyncio.run(_execute(experiment_id))
        print(f"experiment {experiment_id}: {status}")

    result = build_report(engine, experiment_id)
    print(render(result, details=args.details))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return 1 if args.fail_on_regression and regressions(result) else 0


if __name__ == "__main__":
    sys.exit(main())
