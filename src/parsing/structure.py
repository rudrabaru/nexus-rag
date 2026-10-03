import re

_BOLD_ONLY_LINE = re.compile(r"^\*\*([^*\n]{3,79})\*\*\s*$")


_HEADING_LINE = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)


def _has_real_headings(markdown: str) -> bool:
    return _HEADING_LINE.search(_CODE_FENCE.sub("", markdown)) is not None


def promote_fake_headings(markdown: str) -> str:
    """
    A document that already has headings is left alone: its bold lines are emphasis inside a
    structure the parser recovered, and promoting them would invent sections the author did not make.

    Promotes a line that is entirely bold, short, does not end like a sentence (. , ;) and is
    followed by a blank line to a `##` heading. A trailing colon is allowed: a standalone bold
    "Label:" line introduces a section. Many DOCX files style section titles as bold paragraphs
    instead of Heading styles; Docling (correctly) reports those as paragraphs.

    Structural evidence only (typography and layout, no keywords). Measured 2026-09-26 on
    "basics of n8n tutorial notes.docx": 0 headings from Docling, 4 after promotion.
    """
    if _has_real_headings(markdown):
        return markdown
    lines = markdown.split("\n")
    result = []
    for i, line in enumerate(lines):
        match = _BOLD_ONLY_LINE.match(line)
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        if match and not match.group(1).rstrip().endswith((".", ",", ";")) and next_line.strip() == "":
            result.append(f"## {match.group(1)}")
        else:
            result.append(line)
    return "\n".join(result)
