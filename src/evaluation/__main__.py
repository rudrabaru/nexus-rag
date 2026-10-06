"""
    python -m src.evaluation run SPEC.json                 # create an experiment and run it
    python -m src.evaluation resume EXPERIMENT_ID          # continue a paused or interrupted one
    python -m src.evaluation report EXPERIMENT_ID [--json FILE] [--details] [--fail-on-regression]
    python -m src.evaluation list [--tenant T]
    python -m src.evaluation estimate SPEC.json            # what the spec asks of each provider, before running it

Results live in Postgres (DATABASE_URL); --json exports a report. --fail-on-regression is a gate for
CI and exits 1 unless the experiment is evidence: it must be complete, at least --min-valid of every
trial's queries must have produced a valid run (default: the spec's min_valid), every trial must be
comparable with the baseline, and none may be significantly worse on the spec's primary metric. A
broken trial fails the gate; it is never read as "no regression".
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from src.config import get_settings
from src.evaluation import store
from src.evaluation.dataset import resolve_dataset
from src.evaluation.engine import run_experiment
from src.evaluation.estimate import DEFAULT_MAX_DAYS, estimate
from src.evaluation.ground_truth import missing_chunk_ids
from src.evaluation.report import build_report, gate_failures, render
from src.evaluation.spec import ExperimentSpec
from src.llm.config import parse_model
from src.llm.roles import role_model
from src.db.engine import dispose_engines, get_async_engine, get_sync_engine
from src.retrieving.pipeline import RetrievalResources
from src.runtime import ConfigurationError, bootstrap


async def _execute(experiment_id: str) -> str:
    settings = get_settings()
    resources = RetrievalResources(settings, get_sync_engine(), get_async_engine())
    try:
        return await run_experiment(get_sync_engine(), resources, settings, experiment_id)
    finally:
        await dispose_engines()  # pooled async connections belong to this event loop


def missing_keys(spec: ExperimentSpec, settings) -> list:
    """Providers the experiment's models need and have no key for, found before any query runs."""
    if not spec.generation:
        return []
    models = [spec.generation.judge.provider, spec.generation.model.provider if spec.generation.model else role_provider(settings, "chat")]
    return sorted({f"{p.upper()}_API_KEY" for p in models if not settings.has_llm_key(p)})


def role_provider(settings, role: str) -> str:
    return parse_model(role_model(settings, role))[0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="create an experiment from a spec file and run it")
    run.add_argument("spec")
    sizing = commands.add_parser("estimate", help="what a spec will ask of each provider against their limits; runs nothing")
    sizing.add_argument("spec")
    sizing.add_argument("--max-days", type=int, default=DEFAULT_MAX_DAYS)
    run.add_argument("--max-days", type=int, default=DEFAULT_MAX_DAYS, help="refuse a run needing more days of a provider's allowance")
    run.add_argument("--ignore-limits", action="store_true", help="run even when the estimate says a provider's limits will not allow it to finish")
    resume = commands.add_parser("resume", help="continue an experiment")
    resume.add_argument("experiment_id")
    report = commands.add_parser("report", help="print an experiment's report")
    report.add_argument("experiment_id")
    for command in (run, resume, report):
        command.add_argument("--json", help="also write the report to this file")
        command.add_argument("--details", action="store_true", help="include per-difficulty breakdowns")
        command.add_argument("--fail-on-regression", action="store_true", help="exit 1 unless the experiment passes the gate (see above)")
        command.add_argument("--min-valid", type=float, help="share of a trial's queries that must be valid (default: the spec's)")
    listing = commands.add_parser("list", help="recent experiments")
    listing.add_argument("--tenant")
    args = parser.parse_args(argv)

    try:
        bootstrap("cli")
    except ConfigurationError as e:
        print(e)
        return 2
    engine = get_sync_engine()

    if args.command == "list":
        for e in store.list_experiments(engine, args.tenant):
            print(f"{e['experiment_id']}  {e['status']:<9} {e['created_at'][:19]}  {e['tenant_id']:<20} {e['name']}")
        return 0

    if args.command in ("run", "estimate"):
        spec = ExperimentSpec(**json.loads(Path(args.spec).read_text(encoding="utf-8")))
        absent = missing_keys(spec, get_settings())
        if absent:
            print(f"The experiment's models need keys that are not set: {', '.join(absent)}. Nothing was created or run.")
            return 2
        dataset = resolve_dataset(engine, spec.tenant_id, spec.dataset, spec.relevance)
        sizing = estimate(spec, [q.query for q in dataset.queries], get_settings())
        print(sizing.render())
        problems = sizing.problems(args.max_days)
        if args.command == "estimate":
            print("\n" + "\n".join(problems) if problems else "\nWithin the limits.")
            return 1 if problems else 0
        if problems and not args.ignore_limits:
            print("\nRefusing to start; nothing was created:\n  " + "\n  ".join(problems))
            print("Shrink the experiment (fewer trials or queries), use a model with more room, or pass --ignore-limits.")
            return 2
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

    if args.command == "resume":
        absent = missing_keys(ExperimentSpec(**store.get_experiment(engine, experiment_id)["spec"]), get_settings())
        if absent:
            print(f"The experiment's models need keys that are not set: {', '.join(absent)}.")
            return 2

    if args.command in ("run", "resume"):
        status = asyncio.run(_execute(experiment_id))
        print(f"experiment {experiment_id}: {status}")

    result = build_report(engine, experiment_id)
    print(render(result, details=args.details))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    failures = gate_failures(result, args.min_valid)
    if args.fail_on_regression and failures:
        print("\nGate failed:\n  " + "\n  ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
