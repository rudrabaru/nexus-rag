"""
The review step: a person reads each generated question next to the chunk it came from and
accepts, edits or rejects it. A model-written question can be unanswerable from its chunk,
ambiguous without context, or answered just as well by another passage; only a reader catches
that, and an unreviewed set would put those flaws into every metric.

Each decision is saved at once, so a review can be stopped and resumed. Skipped questions stay
pending. Input and output are injected, so the loop is testable without a terminal.
"""
from collections import Counter
from typing import Callable

from src.stores.testsets import ACCEPTED, PENDING, REJECTED
from src.testsets.models import Draft, DraftItem
from src.testsets.quality import lexical_overlap

SOURCE_PREVIEW_CHARS = 2000
PROMPT = "[a]ccept  [e]dit  [r]eject  [s]kip  [q]uit > "


def card(item: DraftItem, position: int, total: int) -> str:
    text = item.source_text
    if len(text) > SOURCE_PREVIEW_CHARS:
        text = text[:SOURCE_PREVIEW_CHARS] + f"\n... ({len(item.source_text) - SOURCE_PREVIEW_CHARS} more characters)"
    overlap = "-" if item.lexical_overlap is None else f"{item.lexical_overlap:.2f}"
    return "\n".join([
        "=" * 78,
        f"[{position}/{total}]  difficulty {item.difficulty} | {item.category} | overlap with source {overlap}"
        f" | ground truth: {len(item.source_chunk_ids)} chunk(s)",
        f"documents: {', '.join(item.acceptable_documents)}",
        f"headings:  {', '.join(item.acceptable_headings) or '-'}",
        "-" * 78,
        f"QUESTION: {item.query}",
        f"ANSWER:   {item.reference_answer}",
        "-" * 78,
        "SOURCE CHUNK:",
        text,
    ])


def edit(item: DraftItem, ask: Callable[[str], str]) -> None:
    """Replaces the question and/or the answer; an empty reply keeps the current text."""
    item.query = ask(f"question [{item.query}]\n> ").strip() or item.query
    item.reference_answer = ask(f"answer [{item.reference_answer}]\n> ").strip() or item.reference_answer
    item.lexical_overlap = lexical_overlap(item.query, item.source_text)


def status_counts(draft: Draft) -> Counter:
    return Counter(item.review_status for item in draft.items)


def review(
    draft: Draft, save: Callable[[Draft], None],
    ask: Callable[[str], str] = input, show: Callable[[str], None] = print,
) -> Draft:
    pending = [item for item in draft.items if item.review_status == PENDING]
    for position, item in enumerate(pending, 1):
        show(card(item, position, len(pending)))
        while True:
            choice = ask(PROMPT).strip().lower()
            if choice in ("a", "accept"):
                item.review_status = ACCEPTED
            elif choice in ("e", "edit"):
                edit(item, ask)
                item.review_status = ACCEPTED
            elif choice in ("r", "reject"):
                item.review_status = REJECTED
            elif choice in ("s", "skip"):
                break
            elif choice in ("q", "quit"):
                return draft
            else:
                show("Choose a, e, r, s or q.")
                continue
            save(draft)
            break
    return draft
