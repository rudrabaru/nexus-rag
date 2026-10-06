import re
from typing import List

from .models import Block
from .splitting import split_oversized
from .tokenizer import TokenCounter

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
TABLE_ROW = re.compile(r"^\|.*\|$")
TABLE_SEPARATOR = re.compile(r"^\|[-| :]+\|$")

# A paragraph this long (about 1,000 words) is a wall of text with no blank line to cut at, so it is
# cut at line breaks, then at sentence ends, into pieces of about WALL_PIECE_CHARS. Experiment: set
# from the paragraph-length distribution of one documentation corpus, not tuned on retrieval results;
# the embedding limit, not this value, is what protects the index.
WALL_OF_TEXT_CHARS = 5000
WALL_PIECE_CHARS = 3000
WALL_SENTENCE_FALLBACK_CHARS = 4000


def split_on_sentences(text: str, max_chars: int = WALL_PIECE_CHARS) -> List[str]:
    pieces, current, length = [], [], 0
    for sentence in SENTENCE_END.split(text):
        if current and length + len(sentence) > max_chars:
            pieces.append(" ".join(current))
            current, length = [], 0
        current.append(sentence)
        length += len(sentence) + 1
    if current:
        pieces.append(" ".join(current))
    return pieces


def split_wall_of_text(paragraph: str) -> List[str]:
    pieces, current, length = [], [], 0

    def flush():
        text = "\n".join(current)
        pieces.extend(split_on_sentences(text) if len(text) > WALL_SENTENCE_FALLBACK_CHARS else [text])

    for line in paragraph.split("\n"):
        if current and length + len(line) > WALL_PIECE_CHARS:
            flush()
            current, length = [], 0
        current.append(line)
        length += len(line) + 1
    if current:
        flush()
    return pieces


def classify(paragraph: str) -> str:
    """code, table or text, from the paragraph's own structure."""
    if paragraph.strip().startswith("```"):
        return "code"
    lines = [line.strip() for line in paragraph.split("\n")]
    if any(TABLE_ROW.match(line) for line in lines) and any(TABLE_SEPARATOR.match(line) for line in lines):
        return "table"
    return "text"


def is_table_continuation(paragraph: str) -> bool:
    """Pipe rows with no header of their own: the rest of a table that a blank line cut in two."""
    lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
    rows = [line for line in lines if TABLE_ROW.match(line)]
    return len(rows) >= 2 and len(rows) >= len(lines) * 0.8


def extract_blocks(text: str, counter: TokenCounter, max_block_tokens: int) -> List[Block]:
    """
    Atomic blocks (code, table, paragraph) of a section's text. A block over `max_block_tokens` is
    split rather than cut off, so every line of it reaches the index.
    """
    paragraphs: List[str] = []
    for paragraph in re.split(r"\n\n+", text.strip()):
        if len(paragraph) > WALL_OF_TEXT_CHARS and classify(paragraph) == "text":
            paragraphs.extend(split_wall_of_text(paragraph))
        else:
            paragraphs.append(paragraph)

    blocks: List[Block] = []
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
        block_type = classify(paragraph)
        previous = blocks[-1] if blocks else None
        if block_type == "text" and previous and previous.block_type == "table" and is_table_continuation(paragraph):
            columns = previous.text.strip().split("\n")[0].count("|")
            if paragraph.strip().split("\n")[0].count("|") == columns:
                previous.text += "\n" + paragraph
                previous.token_count = counter.count_tokens(previous.text)
                continue
        blocks.append(Block(text=paragraph, block_type=block_type, token_count=counter.count_tokens(paragraph)))

    return [piece for block in blocks for piece in split_oversized(block, max_block_tokens, counter.count_tokens)]
