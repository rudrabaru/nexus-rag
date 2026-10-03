import re
from typing import List

from src.protected_markdown import protect

from .models import Section

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def parse_sections(content: str) -> List[Section]:
    """
    Splits Markdown into sections by heading. Each section keeps its heading line, carries the full
    heading path from the document root, and holds code and tables intact (they are hidden from the
    heading scan, so a `#` inside a code block never starts a section).
    """
    safe, restore = protect(content)
    sections: List[Section] = []
    heading_stack = []  # (level, title)
    current = Section(title="", level=0, heading_path=[])
    lines: List[str] = []

    def close_section():
        current.text = restore("\n".join(lines))
        if current.text.strip():
            sections.append(current)

    for line in safe.split("\n"):
        match = HEADING.match(line)
        if match:
            close_section()
            level, title = len(match.group(1)), restore(match.group(2).strip())
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            current = Section(title=title, level=level, heading_path=[t for _, t in heading_stack])
            lines = [line]
        else:
            lines.append(line)

    close_section()
    return sections
