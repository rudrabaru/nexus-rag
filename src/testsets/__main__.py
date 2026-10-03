"""
    python -m src.testsets generate --tenant T [--count 50] [--provider gemini|groq] [--dry-run] [--show-prompt]
    python -m src.testsets review DRAFT.json
    python -m src.testsets finalize DRAFT.json DATASET.json
    python -m src.testsets verify DATASET.json --tenant T

generate reads the tenant's chunks from Postgres (DATABASE_URL) and writes questions to a draft file;
review and finalize turn the draft into a dataset for `python -m src.evaluation run`. The dataset's
ground truth is chunk ids, so it is only valid for the chunking it was made from: verify checks.
"""
import argparse
import sys
from pathlib import Path

from src.config import get_settings
from src.embedding.providers import build_embedder
from src.evaluation.dataset import load_dataset
from src.evaluation.ground_truth import missing_chunk_ids
from src.generating.llm_client import LLMClient
from src.generating.models import GenerationConfig, default_model_name
from src.registry.engine import get_sync_engine
from src.runtime import ConfigurationError, bootstrap
from src.testsets.drafts import accepted_queries, read_draft, write_dataset, write_draft
from src.testsets.generator import GenerationAborted, generate
from src.testsets.models import PENDING, Draft
from src.testsets.prompt import DIFFICULTY_INSTRUCTIONS, build_prompt
from src.testsets.quality import overlap_by_difficulty, tier_warnings
from src.testsets.review import review, status_counts
from src.testsets.sampling import group_identical, interleave_by_document, load_chunks

# Questions come from different passages, so one call per question already varies; a low
# temperature keeps the question tied to what the passage says. Untuned: a starting point.
GENERATION_TEMPERATURE = 0.3

# Seconds between calls, per provider. Gemini: assumes the 5 requests a minute the free tier has
# shown in practice (Google does not publish it: check AI Studio and pass --min-interval). Groq: its
# free tier allows 8K tokens a minute and a generation prompt is about 1K tokens, so one call every
# ~8 s keeps it under the token limit. Providers without an entry use the Gemini value.
PROVIDER_MIN_INTERVAL_SECONDS = {"gemini": 13.0, "groq": 8.0}


def command_generate(args) -> int:
    settings = get_settings()
    provider = args.provider
    model_name = args.model or default_model_name(provider)
    if not settings.has_llm_key(provider):
        print(f"No API key for {provider}: set {provider.upper()}_API_KEY.")
        return 2
    difficulties = [d.strip() for d in args.difficulties.split(",") if d.strip()]
    unknown = [d for d in difficulties if d not in DIFFICULTY_INSTRUCTIONS]
    if unknown or not difficulties:
        print(f"--difficulties takes a comma-separated list of {sorted(DIFFICULTY_INSTRUCTIONS)}; got {unknown}")
        return 2

    engine = get_sync_engine()
    index_id = args.index or build_embedder(settings).index_id
    source_chunks = load_chunks(engine, args.tenant, index_id)
    groups = group_identical(source_chunks)
    documents = len({c.doc_id for c in source_chunks})
    print(f"tenant {args.tenant!r}, index {index_id}: {len(source_chunks)} chunks in {documents} documents, "
          f"{len(groups)} distinct texts ({len(source_chunks) - len(groups)} chunks repeat another chunk's text)")
    if not groups:
        print("Nothing to generate from: ingest a corpus first.")
        return 2
    ordered = interleave_by_document(groups, args.seed)
    if args.show_prompt:
        print("\n--- the first prompt ---\n" + build_prompt(ordered[0], difficulties[0]) + "--- end ---\n")
    if args.dry_run:
        return 0

    out = args.out or f"evaluation_datasets/{args.tenant}.draft.json"
    meta = {"tenant_id": args.tenant, "index_id": index_id, "model": f"{provider}/{model_name}", "seed": args.seed,
            "difficulties": difficulties, "temperature": GENERATION_TEMPERATURE}
    draft = read_draft(out) if Path(out).exists() else Draft(meta=meta)
    differing = [k for k in meta if draft.meta.get(k) != meta[k]]
    if differing:
        print(f"{out} was made with different {', '.join(differing)}; mixing them would change what the set measures. "
              f"Use another --out, or the same settings to resume.")
        return 2

    client = LLMClient(GenerationConfig(provider=provider, model_name=model_name, temperature=GENERATION_TEMPERATURE,
                                        max_output_tokens=1024))  # no fallback: the model is pinned
    print(f"writing {args.count} questions with {provider}/{model_name} -> {out} (resuming at {len(draft.items)})")
    try:
        generate(draft, ordered, client, args.count, difficulties, lambda d: write_draft(out, d), args.min_interval or PROVIDER_MIN_INTERVAL_SECONDS.get(provider, 13.0))
    except GenerationAborted as e:
        print(f"stopped: {e}\nThe draft is saved; run the same command to resume.")
        return 1

    by_difficulty = overlap_by_difficulty(draft.items)
    print(f"\n{len(draft.items)} questions, {len(draft.abstained)} chunks the model found nothing to ask about")
    for tier, stats in by_difficulty.items():
        print(f"  {tier:<7} n={stats['n']:<4} mean lexical overlap with the source chunk {stats['mean_overlap']:.2f}")
    for warning in tier_warnings(by_difficulty):
        print(f"  WARNING: {warning}")
    print(f"Next: python -m src.testsets review {out}")
    return 0


def command_review(args) -> int:
    draft = read_draft(args.draft)
    review(draft, lambda d: write_draft(args.draft, d))
    counts = status_counts(draft)
    print(f"accepted {counts['accepted']}, rejected {counts['rejected']}, pending {counts[PENDING]}")
    return 0


def command_finalize(args) -> int:
    draft = read_draft(args.draft)
    pending = status_counts(draft)[PENDING]
    queries = accepted_queries(draft)
    write_dataset(args.dataset, queries)
    print(f"{len(queries)} accepted questions -> {args.dataset} ({pending} still pending were left out)")
    print("Label every metric from this set 'synthetic': its questions come from the corpus's own passages.")
    return 0


def command_verify(args) -> int:
    dataset = load_dataset(args.dataset, relevance="chunk")
    engine = get_sync_engine()
    missing = missing_chunk_ids(engine, args.tenant, (i for q in dataset.queries for i in q.source_chunk_ids))
    if missing:
        print(f"{len(missing)} source chunks are not in tenant {args.tenant!r}: {missing[:5]}")
        return 1
    print(f"all source chunks of {len(dataset.queries)} queries are in tenant {args.tenant!r}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    generate_parser = commands.add_parser("generate", help="write questions from a tenant's chunks into a draft")
    generate_parser.add_argument("--tenant", required=True)
    generate_parser.add_argument("--count", type=int, default=50)
    generate_parser.add_argument("--provider", default="gemini")
    generate_parser.add_argument("--model", help="default: the provider's default model")
    generate_parser.add_argument("--index", help="embedding index to read chunks from (default: the configured one)")
    generate_parser.add_argument("--difficulties", default="easy,medium,hard", help="tiers, cycled in order")
    generate_parser.add_argument("--seed", type=int, default=0)
    generate_parser.add_argument("--out", help="draft file (default: evaluation_datasets/TENANT.draft.json)")
    generate_parser.add_argument("--min-interval", type=float, help=f"seconds between LLM calls (default per provider: {PROVIDER_MIN_INTERVAL_SECONDS})")
    generate_parser.add_argument("--dry-run", action="store_true", help="show what would be sampled; call no LLM")
    generate_parser.add_argument("--show-prompt", action="store_true", help="print the first prompt")

    review_parser = commands.add_parser("review", help="accept, edit or reject each pending question")
    review_parser.add_argument("draft")
    finalize_parser = commands.add_parser("finalize", help="write the accepted questions as an evaluation dataset")
    finalize_parser.add_argument("draft")
    finalize_parser.add_argument("dataset")
    verify_parser = commands.add_parser("verify", help="check a dataset's source chunks exist in the index")
    verify_parser.add_argument("dataset")
    verify_parser.add_argument("--tenant", required=True)

    args = parser.parse_args(argv)
    if args.command in ("generate", "verify"):
        try:
            bootstrap("cli")
        except ConfigurationError as e:
            print(e)
            return 2
    return {"generate": command_generate, "review": command_review, "finalize": command_finalize,
            "verify": command_verify}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
