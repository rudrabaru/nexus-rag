"""
An experiment specification: which queries, which workspace, which configurations.

    {
      "name": "dense vs hybrid vs reranked",
      "dataset": "evaluation_datasets/my_queries.json",
      "tenant_id": "demo",
      "baseline": "dense",
      "relevance": "document",
      "trials": {
        "dense": {"strategy": "dense"},
        "hybrid": {"strategy": "hybrid"},
        "hybrid+flashrank": {"strategy": "hybrid", "reranker": "flashrank", "rerank_candidates": 20}
      },
      "generation": {"judge": {"provider": "groq", "model_name": "openai/gpt-oss-120b"}}
    }

Each trial is a RetrievalConfig (src/retrieving/config.py). Without "generation" the
experiment is retrieval-only and calls no LLM at all.
"""
from typing import Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.stores.api_keys import TENANT_ID_PATTERN
from src.retrieving.config import RetrievalConfig


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    provider: str
    model_name: str


class GenerationSpec(BaseModel):
    """
    Answers are generated and judged with pinned models: an experiment never falls back to
    another model, because a different model mid-experiment changes what is being measured.
    None = the configured chat model (LLM_PROVIDER / LLM_MODEL_NAME). Prefer a judge from a
    different model family than the generator (self-preference bias).
    """

    model_config = ConfigDict(extra="forbid")
    model: Optional[ModelSpec] = None
    judge: Optional[ModelSpec] = None


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    dataset: str
    tenant_id: str = Field(pattern=TENANT_ID_PATTERN.pattern)
    trials: Dict[str, RetrievalConfig] = Field(min_length=1)
    baseline: Optional[str] = None  # the trial every other trial is compared with; default: the first
    # Queries run at once per trial. Latency percentiles are measured under this concurrency;
    # set 1 for latency-faithful numbers. Query embedding is paced to the provider's limits anyway.
    concurrency: int = Field(4, ge=1, le=16)
    # "document": a chunk is relevant when it comes from an acceptable document and heading (survives
    # re-chunking). "chunk": only the query's source_chunk_ids count (strict; needs those ids).
    relevance: Literal["document", "chunk"] = "document"
    alpha: float = Field(0.05, gt=0, lt=1)
    generation: Optional[GenerationSpec] = None

    @model_validator(mode="after")
    def baseline_is_a_trial(self):
        if self.baseline is None:
            self.baseline = next(iter(self.trials))
        elif self.baseline not in self.trials:
            raise ValueError(f"baseline {self.baseline!r} is not one of the trials {sorted(self.trials)}")
        return self
