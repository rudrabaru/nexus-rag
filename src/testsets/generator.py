"""
Generating the draft: one LLM call per chunk group, each producing a question, its reference
answer and the ground truth (every chunk with that text). Runs on the laptop, calls only the
configured LLM provider, and makes no embedding calls (so Voyage's 3 requests a minute is not
touched).

- Resumable: the draft is saved after every question, and chunks already handled (a question
  written, or the model abstained) are skipped on the next run. A failed call is not recorded,
  so it is retried.
- Paced to the provider's free-tier limits (see PROVIDER_MIN_INTERVAL_SECONDS in __main__.py).
- Pinned model, no fallback: a test set's character should not depend on which provider
  happened to be up. A provider that fails several times in a row aborts the run instead.
"""
import time
from typing import Callable, Sequence

from src.evaluation.dataset import SYNTHETIC
from src.llm.client import LLMClient
from src.llm.errors import GenerationError
from src.testsets.models import Draft, DraftItem
from src.testsets.prompt import GeneratedQuestion, build_prompt, parse_generated
from src.testsets.quality import lexical_overlap
from src.testsets.sampling import ChunkGroup

MAX_CONSECUTIVE_FAILURES = 5


class GenerationAborted(RuntimeError):
    """The provider failed too many times in a row; what was generated so far is saved."""


def category_of(group: ChunkGroup) -> str:
    """A structural label (what the chunk holds), never a topic or a source-specific one."""
    source = group.representative
    return "code" if source.contains_code else "table" if source.contains_table else "prose"


def generate(
    draft: Draft,
    groups: Sequence[ChunkGroup],
    client: LLMClient,
    count: int,
    difficulties: Sequence[str],
    save: Callable[[Draft], None],
    min_interval_seconds: float = 1.1,
    progress: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
) -> Draft:
    """Adds questions to the draft until it holds `count` (or the groups run out)."""
    handled = draft.handled_chunk_ids()
    failures, last_call = 0, None
    for group in groups:
        if len(draft.items) >= count:
            break
        if group.chunk_ids[0] in handled:
            continue
        if last_call is not None:
            sleep(max(0.0, min_interval_seconds - (time.monotonic() - last_call)))
        difficulty = difficulties[len(draft.items) % len(difficulties)]
        try:
            reply = client.call_llm(build_prompt(group, difficulty), response_schema=GeneratedQuestion).text
        except GenerationError as e:
            reply = f"[call failed: {e}]"
        last_call = time.monotonic()
        generated = parse_generated(reply)
        if generated is None:
            failures += 1
            progress(f"  {group.chunk_ids[0]}: no usable reply ({reply[:120]!r})")
            if failures >= MAX_CONSECUTIVE_FAILURES:
                raise GenerationAborted(f"{failures} failures in a row; the last: {reply[:200]!r}")
            continue
        failures = 0

        question = generated.question.strip()
        asked = {item.query.lower() for item in draft.items}
        if not generated.answerable or not question or not generated.answer.strip() or question.lower() in asked:
            draft.abstained.append(group.chunk_ids[0])
            progress(f"  {group.chunk_ids[0]}: nothing to ask")
        else:
            source = group.representative
            draft.items.append(DraftItem(
                query=question, reference_answer=generated.answer.strip(), acceptable_documents=group.documents,
                acceptable_headings=group.headings, source_chunk_ids=group.chunk_ids, difficulty=difficulty,
                category=category_of(group), origin=SYNTHETIC, lexical_overlap=lexical_overlap(question, source.text),
                source_text=source.text,
            ))
            progress(f"  [{len(draft.items)}/{count}] {difficulty:<6} {question}")
        save(draft)
    return draft
