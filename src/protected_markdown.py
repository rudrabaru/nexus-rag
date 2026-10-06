"""
Hides fenced code and tables behind placeholders so line- and heading-based parsing cannot cut them.

Both the cleaner (src/processing) and the chunker (src/chunking) split Markdown by headings and blank
lines, and a `#` comment inside a code block or a blank line inside a fence would otherwise look like
structure. One implementation serves both, so they cannot disagree about what a block is.

Placeholders are delimited by NUL, which is stripped from the document first: a document can never
contain one, so a page that happens to print `__TABLE_BLOCK_0__` is left alone. (NUL also cannot be
stored in Postgres text, so stripping it is required anyway.)
"""
import re
from typing import Callable, Dict, Tuple

CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
# Consecutive pipe rows. The match stops before the newline that ends the last row, so the line breaks
# around a table survive and a heading next to it stays on its own line.
TABLE = re.compile(r"^\|[^\n]*\|[ \t]*(?:\n\|[^\n]*\|[ \t]*)*", re.MULTILINE)


def protect(markdown: str) -> Tuple[str, Callable[[str], str]]:
    """Returns (markdown with code and tables replaced by placeholders, a function that restores them)."""
    stored: Dict[str, str] = {}

    def hide(kind: str):
        def replace(match: re.Match) -> str:
            placeholder = f"\x00{kind}{len(stored)}\x00"
            stored[placeholder] = match.group(0)
            return placeholder

        return replace

    safe = markdown.replace("\x00", "")
    safe = CODE_FENCE.sub(hide("CODE"), safe)
    safe = TABLE.sub(hide("TABLE"), safe)

    def restore(text: str) -> str:
        for placeholder, original in stored.items():
            text = text.replace(placeholder, original)
        return text

    return safe, restore
