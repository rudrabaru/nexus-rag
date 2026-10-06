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
    The judge is named in the spec, so what judged an experiment is part of what was asked for and
    cannot change with an environment variable; it must not be the model it judges (self-preference
    bias), and should come from a different model family. `model` None = the chat role (LLM_CHAT).
    """

    model_config = ConfigDict(extra="forbid")
    model: Optional[ModelSpec] = None
    judge: ModelSpec


class TrialSpec(RetrievalConfig):
    """
    One trial: a retrieval configuration plus, when the experiment generates answers, the generation
    knobs a trial may vary. Both are left out to inherit the experiment's generation settings.
    """

    max_context_tokens: Optional[int] = Field(
        None, ge=200, le=30_000,
        description="Tokens of retrieved context the answer prompt may hold; None = the generator's default (5,000).",
    )
    generation_model: Optional[ModelSpec] = Field(None, description="The model that answers in this trial; None = the experiment's.")


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    dataset: str
    tenant_id: str = Field(pattern=TENANT_ID_PATTERN.pattern)
    trials: Dict[str, TrialSpec] = Field(min_length=1)
    baseline: Optional[str] = None  # the trial every other trial is compared with; default: the first
    # Queries run at once per trial. Latency percentiles are measured under this concurrency;
    # set 1 for latency-faithful numbers. Query embedding is paced to the provider's limits anyway.
    concurrency: int = Field(4, ge=1, le=16)
    # "document": a chunk is relevant when it comes from an acceptable document and heading (survives
    # re-chunking). "chunk": only the query's source_chunk_ids count (strict; needs those ids).
    relevance: Literal["document", "chunk"] = "document"
    alpha: float = Field(0.05, gt=0, lt=1)
    # The metric the regression gate looks at. The report tests every metric (one Holm family), but a
    # gate that fires on any of five correlated metrics fires by chance; one is named in advance.
    primary_metric: str = "mrr"
    # The share of a trial's queries that must produce a valid run for the experiment to count. Below it
    # the metrics describe the surviving queries, not the trial, so the gate fails instead of passing on
    # whatever was left. 0.9 tolerates a few transient failures; it is not tuned on a corpus.
    min_valid: float = Field(0.9, gt=0, le=1)
    generation: Optional[GenerationSpec] = None

    @model_validator(mode="after")
    def generation_knobs_need_generation(self):
        for label, trial in self.trials.items():
            if self.generation is None and (trial.max_context_tokens or trial.generation_model):
                raise ValueError(f"trial {label!r} sets a generation knob, but the experiment has no `generation` section")
        return self

    @model_validator(mode="after")
    def baseline_is_a_trial(self):
        if self.baseline is None:
            self.baseline = next(iter(self.trials))
        elif self.baseline not in self.trials:
            raise ValueError(f"baseline {self.baseline!r} is not one of the trials {sorted(self.trials)}")
        return self
