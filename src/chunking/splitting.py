"""
Splits a block that is too large to embed, at the nearest boundary that keeps it readable.

A block over the embedding limit used to be cut off with a "[TRUNCATED]" marker, which silently
dropped the rest of a long table or code listing from the index. Splitting keeps every line:

- text: at line boundaries, and inside a single over-long line at word boundaries;
- code: at line boundaries, each piece re-fenced with the original opening fence;
- tables: at row boundaries, each piece repeating the header and separator rows, so a row is never
  read without its column names.
"""
import re
from typing import Callable, List

from .models import Block

SEPARATOR_ROW = re.compile(r"^\|[-| :]+\|$")
FENCE = "```"


def split_oversized(block: Block, limit: int, count: Callable[[str], int]) -> List[Block]:
    """Pieces of `block` that each hold at most `limit` tokens; a block within the limit is returned as it is."""
    if block.token_count <= limit:
        return [block]
    lines = block.text.split("\n")
    prefix, suffix = [], []
    if block.block_type == "code" and lines[0].startswith(FENCE) and lines[-1].strip() == FENCE and len(lines) > 2:
        prefix, suffix, lines = [lines[0]], [FENCE], lines[1:-1]
    elif block.block_type == "table":
        # Everything up to the separator row repeats: the header row, and the heading line when the
        # table sits directly under one. Unless that is most of the budget, which would leave no room for rows.
        separator = next((i for i, line in enumerate(lines) if SEPARATOR_ROW.match(line.strip())), None)
        if separator is not None and count("\n".join(lines[: separator + 1])) <= limit // 2:
            prefix, lines = lines[: separator + 1], lines[separator + 1:]

    frame = "\n".join(prefix + suffix)
    budget = max(limit - count(frame) - 1, 1)
    pieces = [piece for line in lines for piece in _fit_line(line, budget, count)]
    texts = ["\n".join(prefix + body + suffix) for body in _pack(pieces, budget, count)]
    return [Block(text=text, block_type=block.block_type, token_count=count(text)) for text in texts]


def _pack(lines: List[str], budget: int, count: Callable[[str], int]) -> List[List[str]]:
    """Greedy packing; the joined text is measured, not the sum of its lines, because tokens merge across newlines."""
    packed, current = [], []
    for line in lines:
        if current and count("\n".join(current + [line])) > budget:
            packed.append(current)
            current = []
        current.append(line)
    if current:
        packed.append(current)
    return packed


def _fit_line(line: str, budget: int, count: Callable[[str], int]) -> List[str]:
    """A line within the budget as it is; a longer one split at words, and a single huge word by characters."""
    if count(line) <= budget:
        return [line]
    pieces, current = [], []
    for word in line.split(" "):
        if count(word) > budget:
            if current:
                pieces.append(" ".join(current))
                current = []
            step = max(budget // 4, 1)  # a character is at most 4 bytes, so at most 4 tokens
            pieces.extend(word[i:i + step] for i in range(0, len(word), step))
        elif current and count(" ".join(current + [word])) > budget:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces
