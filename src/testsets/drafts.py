"""Reading and writing drafts and the datasets made from them."""
import json
import os
from pathlib import Path
from typing import List

from src.evaluation.dataset import EvaluationQuery, integrity_problems
from src.testsets.models import ACCEPTED, Draft


def read_draft(path: str) -> Draft:
    return Draft(**json.loads(Path(path).read_text(encoding="utf-8")))


def write_draft(path: str, draft: Draft) -> None:
    """Atomic, so a crash mid-write never costs the reviewer the questions already decided."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = target.with_name(target.name + ".tmp")
    scratch.write_text(draft.model_dump_json(indent=2), encoding="utf-8")
    os.replace(scratch, target)


def accepted_queries(draft: Draft) -> List[EvaluationQuery]:
    return [item.to_query() for item in draft.items if item.review_status == ACCEPTED]


def write_dataset(path: str, queries: List[EvaluationQuery]) -> None:
    """The frozen dataset the evaluation engine loads. Refuses one the engine would reject."""
    problems = integrity_problems(queries, relevance="chunk")
    if problems:
        raise ValueError("The dataset is not usable:\n  " + "\n  ".join(problems))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = [q.model_dump() for q in queries]
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
