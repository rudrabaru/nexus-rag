"""
Measuring generated questions. A question written from a chunk tends to borrow that chunk's
words, which makes retrieval look easier than it is for a real user (the 2026-09 prototype
benchmark had this bias). Lexical overlap makes the bias visible: it is the share of a question's
distinct words that also occur in its source chunk, compared across difficulty tiers.

It is a signal, not a filter: no question is dropped for its overlap. Words are alphanumeric
tokens of three or more characters, which sets short function words aside without a stop-word
list (those are language-specific); overlap is for comparing tiers on one corpus, not an absolute.
"""
from collections import defaultdict
from statistics import mean
from typing import Dict, Iterable, List

from src.evaluation.relevance import tokens
from src.testsets.models import DraftItem

MIN_WORD_LENGTH = 3


def words(text: str) -> set:
    return {t for t in tokens(text) if len(t) >= MIN_WORD_LENGTH}


def lexical_overlap(question: str, source_text: str) -> float:
    asked = words(question)
    return len(asked & words(source_text)) / len(asked) if asked else 0.0


def overlap_by_difficulty(items: Iterable[DraftItem]) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[float]] = defaultdict(list)
    for item in items:
        if item.lexical_overlap is not None:
            groups[item.difficulty].append(item.lexical_overlap)
    return {tier: {"n": len(v), "mean_overlap": mean(v)} for tier, v in sorted(groups.items())}


def tier_warnings(by_difficulty: Dict[str, Dict[str, float]]) -> List[str]:
    """Tiers that do not look harder than the one before (by mean overlap, which should fall)."""
    order = [t for t in ("easy", "medium", "hard") if t in by_difficulty]
    return [
        f"{harder!r} questions overlap their source as much as {easier!r} ones "
        f"({by_difficulty[harder]['mean_overlap']:.2f} vs {by_difficulty[easier]['mean_overlap']:.2f}): the tier is not harder"
        for easier, harder in zip(order, order[1:])
        if by_difficulty[harder]["mean_overlap"] >= by_difficulty[easier]["mean_overlap"]
    ]
