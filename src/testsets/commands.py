"""
What each `python -m src.testsets` command does. They print and return an exit code; the work is done
by the modules they call, so each is testable without a terminal.
"""
from functools import partial

from src.config import get_settings
from src.db.engine import get_sync_engine
from src.embedding.providers import build_embedder
from src.evaluation.dataset import SYNTHETIC
from src.evaluation.ground_truth import missing_chunk_ids
from src.llm.client import LLMClient
from src.generating.models import GenerationConfig
from src.llm.config import parse_model
from src.stores.testsets import PENDING, TestSetError, TestSetStore
from src.testsets.generator import GenerationAborted, generate
from src.testsets.models import Draft
from src.testsets.prompt import DIFFICULTY_INSTRUCTIONS, build_prompt
from src.testsets.quality import overlap_by_difficulty, tier_warnings
from src.testsets.repository import export_dataset, import_dataset, load_draft, save_draft
from src.testsets.review import review, status_counts
from src.testsets.sampling import group_identical, interleave_by_document, load_chunks

# Questions come from different passages, so one call per question already varies; a low
# temperature keeps the question tied to what the passage says. Untuned: a starting point.
GENERATION_TEMPERATURE = 0.3

# Seconds between calls, per provider. Gemini: assumes the 5 requests a minute its free tier has
# shown in practice (Google does not publish it: check AI Studio and pass --min-interval). Groq: its
# free tier allows 8K tokens a minute and a generation prompt is about 1K tokens, so one call every
# ~8 s keeps it under the token limit. Providers without an entry use the Gemini value.
PROVIDER_MIN_INTERVAL_SECONDS = {"gemini": 13.0, "groq": 8.0}
DEFAULT_MIN_INTERVAL_SECONDS = 13.0


def _store() -> TestSetStore:
    return TestSetStore(get_sync_engine())


def _find(args) -> dict:
    head = _store().find(args.tenant, args.set)
    if head is None:
        raise TestSetError(f"Workspace {args.tenant!r} has no test set {args.set!r}. See: python -m src.testsets list --tenant {args.tenant}")
    return head


def command_generate(args) -> int:
    settings = get_settings()
    try:
        provider, model_name = parse_model(args.model or settings.llm_testset)
    except ValueError as e:
        print(e)
        return 2
    if not settings.has_llm_key(provider):
        print(f"No API key for {provider}: set {provider.upper()}_API_KEY.")
        return 2
    difficulties = [d.strip() for d in args.difficulties.split(",") if d.strip()]
    unknown = [d for d in difficulties if d not in DIFFICULTY_INSTRUCTIONS]
    if unknown or not difficulties:
        print(f"--difficulties takes a comma-separated list of {sorted(DIFFICULTY_INSTRUCTIONS)}; got {unknown}")
        return 2

    engine, store = get_sync_engine(), _store()
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

    meta = {"tenant_id": args.tenant, "index_id": index_id, "model": f"{provider}/{model_name}", "seed": args.seed,
            "difficulties": difficulties, "temperature": GENERATION_TEMPERATURE}
    head = store.find(args.tenant, args.set)
    if head is None:
        test_set_id, draft = store.create(args.tenant, args.set, meta), Draft(meta=meta)
    else:
        test_set_id, draft = head["test_set_id"], load_draft(store, head["test_set_id"])
        differing = [k for k in meta if draft.meta.get(k) != meta[k]]
        if differing:
            print(f"Test set {args.set!r} was made with different {', '.join(differing)}; mixing them would change what the "
                  f"set measures. Use another --set, or the same settings to resume.")
            return 2

    client = LLMClient(GenerationConfig(provider=provider, model_name=model_name, temperature=GENERATION_TEMPERATURE,
                                        max_output_tokens=1024))  # no fallback: the model is pinned
    interval = args.min_interval or PROVIDER_MIN_INTERVAL_SECONDS.get(provider, DEFAULT_MIN_INTERVAL_SECONDS)
    print(f"writing {args.count} questions with {provider}/{model_name} into test set {args.set!r} (resuming at {len(draft.items)})")
    try:
        generate(draft, ordered, client, args.count, difficulties, partial(save_draft, store, test_set_id), interval)
    except (GenerationAborted, TestSetError) as e:
        print(f"stopped: {e}\nWhat was generated is saved; run the same command to resume.")
        return 1

    by_difficulty = overlap_by_difficulty(draft.items)
    print(f"\n{len(draft.items)} questions, {len(draft.abstained)} chunks the model found nothing to ask about")
    for tier, stats in by_difficulty.items():
        print(f"  {tier:<7} n={stats['n']:<4} mean lexical overlap with the source chunk {stats['mean_overlap']:.2f}")
    for warning in tier_warnings(by_difficulty):
        print(f"  WARNING: {warning}")
    print(f"Next: python -m src.testsets review --tenant {args.tenant} --set {args.set}")
    return 0


def command_review(args) -> int:
    store, head = _store(), _find(args)
    draft = load_draft(store, head["test_set_id"])
    review(draft, partial(save_draft, store, head["test_set_id"]))
    counts = status_counts(draft)
    print(f"accepted {counts['accepted']}, rejected {counts['rejected']}, pending {counts[PENDING]}")
    return 0


def command_freeze(args) -> int:
    store, head = _store(), _find(args)
    pending = status_counts(load_draft(store, head["test_set_id"]))[PENDING]
    digest = store.freeze(head["test_set_id"])
    print(f"test set {args.set!r} frozen (sha256 {digest[:12]}); {pending} still pending were left out.")
    print(f"Use it in an experiment spec as: \"dataset\": \"testset:{args.set}\"")
    print(f"Label every metric from this set '{SYNTHETIC}': its questions come from the corpus's own passages.")
    return 0


def command_verify(args) -> int:
    head = _find(args)
    questions = _store().accepted_questions(head["test_set_id"])
    missing = missing_chunk_ids(get_sync_engine(), args.tenant, (i for q in questions for i in q["source_chunk_ids"]))
    if missing:
        print(f"{len(missing)} source chunks are not in workspace {args.tenant!r} (re-chunked or re-ingested since "
              f"the set was made?): {missing[:5]}")
        return 1
    print(f"all source chunks of {len(questions)} accepted questions are in workspace {args.tenant!r}")
    return 0


def command_list(args) -> int:
    sets = _store().list_sets(args.tenant)
    if not sets:
        print(f"Workspace {args.tenant!r} has no test sets.")
    for entry in sets:
        counts = ", ".join(f"{n} {status}" for status, n in sorted(entry["questions"].items())) or "no questions"
        print(f"{entry['name']:<24} {entry['status']:<7} {counts}" + (f"  sha256 {entry['content_hash'][:12]}" if entry["content_hash"] else ""))
    return 0


def command_import(args) -> int:
    digest = import_dataset(_store(), args.tenant, args.set, args.file)
    print(f"imported {args.file} as frozen test set {args.set!r} (sha256 {digest[:12]})")
    return 0


def command_export(args) -> int:
    count = export_dataset(_store(), args.tenant, args.set, args.file)
    print(f"{count} accepted questions -> {args.file}")
    return 0
