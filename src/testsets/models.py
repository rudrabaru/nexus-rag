"""
A draft is the working file between generation and the frozen dataset: every generated question
with the chunk text it came from (what a reviewer needs to judge it) and its review status.
"""
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from src.evaluation.dataset import EvaluationQuery

PENDING, ACCEPTED, REJECTED = "pending", "accepted", "rejected"


class DraftItem(EvaluationQuery):
    source_text: str = ""
    review_status: str = PENDING

    def to_query(self) -> EvaluationQuery:
        return EvaluationQuery(**self.model_dump(exclude={"source_text", "review_status"}))


class Draft(BaseModel):
    meta: Dict[str, Any] = Field(default_factory=dict)  # how it was generated: tenant, index, model, seed, ...
    items: List[DraftItem] = Field(default_factory=list)
    abstained: List[str] = Field(default_factory=list)  # chunks the model found nothing to ask about

    def handled_chunk_ids(self) -> set:
        """Chunks that need no further generation call; a resumed run skips them."""
        return {chunk_id for item in self.items for chunk_id in item.source_chunk_ids} | set(self.abstained)
