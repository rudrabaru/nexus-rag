"""
The question-writing prompt, with difficulty tiers.

The tiers exist to stress retrieval beyond keyword matching (AGENTS.md: hard queries use synonyms,
paraphrase and indirect references). Whether a tier really is harder is measured, not assumed:
quality.py records each question's lexical overlap with its source chunk.
"""
from typing import Optional

from pydantic import BaseModel

from src.llm.structured import extract_json_object
from src.testsets.sampling import ChunkGroup

DIFFICULTY_INSTRUCTIONS = {
    "easy": "Phrase the question naturally. It may use the passage's own terms.",
    "medium": (
        "Paraphrase: express the main idea in different words than the passage uses. "
        "You may keep proper names and identifiers."
    ),
    "hard": (
        "Make the question hard to find by keyword search on purpose. Do not reuse the passage's distinctive "
        "terms: use synonyms, describe the concept indirectly, or describe the situation or goal of someone "
        "who needs this instead of naming the feature. A person who knows the subject must still be able to "
        "tell exactly what is being asked."
    ),
}

PROMPT_TEMPLATE = """You write test questions for a search system that is evaluated on how well it finds the right passage.

SOURCE: {source}
SECTION: {section}

PASSAGE:
{text}

Write ONE question that this passage answers. Rules:
- The question must make sense on its own, as a person would type it into a search box without having seen the passage. Name the subject it is about. Never refer to "the passage", "the text", "this section" or "the document".
- The answer must be stated in the passage. Give it in one to three sentences, using only what the passage says.
- {difficulty}
- If the passage has no self-contained fact worth asking about (for example only headings, links or navigation), set "answerable" to false and leave the other fields empty.

Respond ONLY with a JSON object: {{"answerable": true, "question": "...", "answer": "..."}}
"""


class GeneratedQuestion(BaseModel):
    answerable: bool = True
    question: str = ""
    answer: str = ""


def build_prompt(group: ChunkGroup, difficulty: str) -> str:
    source = group.representative
    return PROMPT_TEMPLATE.format(
        source=source.source,
        section=" > ".join(source.heading_path) or source.section_title or "(none)",
        text=source.text,
        difficulty=DIFFICULTY_INSTRUCTIONS[difficulty],
    )


def parse_generated(text: str) -> Optional[GeneratedQuestion]:
    """The model's question, or None when its reply is not a usable JSON object."""
    try:
        return GeneratedQuestion(**extract_json_object(text))
    except (ValueError, TypeError):
        return None
