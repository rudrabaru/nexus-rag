from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ExperimentSummary(BaseModel):
    experiment_id: str
    name: str
    dataset_name: str
    status: str = Field(description="running | complete | paused | failed")
    created_at: datetime
    finished_at: Optional[datetime] = None


class ExperimentReport(BaseModel):
    """
    What an experiment found: per-trial metrics and, for each trial against the baseline, a verdict
    (better, worse, no significant difference, or insufficient evidence) with its adjusted p-value.
    Metric names depend on the trials' `top_k` (hit_rate@k) and on whether answers were judged, so
    metrics, breakdowns and comparisons are objects keyed by metric name.
    """

    experiment: Dict[str, Any]
    queries: int
    synthetic: bool = Field(description="Every question was written by a model, so absolute scores are optimistic.")
    relevance: str
    baseline: str
    alpha: float
    primary_metric: str
    min_valid: float
    comparison_top_k: Optional[int] = Field(description="The one cutoff every comparison is computed at.")
    trials: List[Dict[str, Any]]
    comparisons: List[Dict[str, Any]]


class TestQuestionView(BaseModel):
    __test__ = False  # not a pytest test class

    position: int
    query: str
    reference_answer: str = ""
    difficulty: str
    category: str
    review_status: str = Field(description="pending | accepted | rejected")
    lexical_overlap: Optional[float] = Field(default=None, description="Share of the question's words that also appear in its source passage.")
    source_chunk_ids: List[str] = Field(default_factory=list)
    source_text: Optional[str] = Field(default=None, description="The passage the question was written from, for review.")


class TestSetSummary(BaseModel):
    __test__ = False

    name: str
    status: str = Field(description="draft | frozen")
    content_hash: Optional[str] = None
    created_at: datetime
    questions: Dict[str, int] = Field(description="How many questions are pending, accepted and rejected.")


class TestSetDetail(BaseModel):
    __test__ = False

    name: str
    status: str
    content_hash: Optional[str] = None
    meta: Dict[str, Any] = Field(description="The model, seed, index and tiers the set was generated with.")
    abstained: int = Field(description="Passages the model declined to write a question for.")
    questions: List[TestQuestionView]
