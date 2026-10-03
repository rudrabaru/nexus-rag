"""Reading structured output (a JSON object) out of an LLM reply."""
import json
import re


def extract_json_object(text: str) -> dict:
    """
    The JSON object in a reply: the whole text, or the outermost {...} when the model wrapped it
    in prose or a code fence. Raises ValueError when there is none, or it does not parse.
    """
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        found = re.search(r"(\{.*\})", text, re.DOTALL)
        if not found:
            raise ValueError(f"no JSON object in the reply: {text[:200]!r}")
        try:
            parsed = json.loads(found.group(1))
        except json.JSONDecodeError as e:
            raise ValueError(f"unparseable JSON in the reply: {e}") from e
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}: {text[:200]!r}")
    return parsed
