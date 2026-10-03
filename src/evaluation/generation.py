"""
The optional generation stage of an experiment: answer each query from its retrieved context,
then have a judge score the answer's faithfulness to that context.

Both steps are cached in Postgres, keyed by a hash of everything that determines their output:
- an answer by (tenant, model, full prompt). Configurations that retrieve the same context
  build the same prompt, so their answer is generated once (e.g. top_k 5 vs 7 that end up with
  the same chunks after reranking);
- a verdict by (tenant, metric, judge model, question, answer, sorted context chunk ids).
The tenant is part of every key, so a cached answer never crosses workspaces.

Models are pinned: no fallback inside an experiment. A failed generation makes that run invalid
(retried on resume); a judge that cannot be called raises JudgeUnavailable, which pauses the
whole experiment rather than letting a different judge score part of it.
"""
import hashlib
import logging
import threading
from collections import Counter
from typing import Any, Dict, Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine

from src.config import Settings
from src.evaluation.spec import GenerationSpec, ModelSpec
from src.generating.evaluator import FaithfulnessEvaluator, JudgeOutputError
from src.generating.generator import RAGGenerator
from src.generating.models import GenerationConfig, GenerationResult, default_model_name
from src.db.schema import generation_cache, judge_cache
from src.retrieving.models import RetrievalResult

logger = logging.getLogger(__name__)

FAITHFULNESS = "faithfulness"


def cache_key(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _config(model: Optional[ModelSpec], settings: Settings) -> GenerationConfig:
    if model:
        return GenerationConfig(provider=model.provider, model_name=model.model_name)
    provider = settings.llm_provider
    return GenerationConfig(provider=provider, model_name=settings.llm_model_name or default_model_name(provider))


class GenerationStage:
    def __init__(self, spec: GenerationSpec, tenant_id: str, engine: Engine, settings: Settings):
        self.tenant_id = tenant_id
        self.engine = engine
        self.generator = RAGGenerator(_config(spec.model, settings))  # fallback_config is None: pinned
        self.judge = FaithfulnessEvaluator(_config(spec.judge, settings))
        self.model = f"{self.generator.config.provider}/{self.generator.config.model_name}"
        self.counters: Counter = Counter()
        self._lock = threading.Lock()

    def _count(self, name: str) -> None:
        with self._lock:
            self.counters[name] += 1

    def run(self, query: str, retrieval: RetrievalResult) -> Dict[str, Any]:
        """Run columns for the generation stage. Raises JudgeUnavailable."""
        prepared = self.generator.prepare(query, retrieval)
        row: Dict[str, Any] = {"generation_model": self.model}
        if prepared.diagnostic:  # no chunk survived context building: nothing to generate or judge
            return {**row, "answer": prepared.diagnostic, "generation_cost_usd": 0.0, "generation_cached": False}

        answer_key = cache_key(self.tenant_id, self.model, prepared.prompt)
        cached = self._get(generation_cache, answer_key)
        if cached:
            self._count("generation_cache_hits")
            answer = cached["answer"]
            row.update(
                answer=answer, generation_input_tokens=cached["input_tokens"], generation_output_tokens=cached["output_tokens"],
                generation_cost_usd=cached["cost_usd"], generation_cached=True,
            )
        else:
            self._count("generation_calls")
            call = self.generator.llm_client.call_llm(prepared.prompt)
            if call.failed:
                return {**row, "error": f"generation failed: {call.text}"}
            answer = call.text
            self._put(generation_cache, cache_key=answer_key, model=self.model, answer=answer,
                      input_tokens=call.prompt_tokens, output_tokens=call.completion_tokens, cost_usd=call.cost_usd)
            row.update(
                answer=answer, generation_input_tokens=call.prompt_tokens, generation_output_tokens=call.completion_tokens,
                generation_cost_usd=call.cost_usd, generation_cached=False,
            )

        context_ids = ",".join(sorted(c.chunk_id for c in prepared.context_window.included_chunks))
        verdict_key = cache_key(self.tenant_id, FAITHFULNESS, self.judge.model, query, answer, context_ids)
        verdict = self._get(judge_cache, verdict_key)
        if verdict:
            self._count("judge_cache_hits")
            return {**row, "faithfulness": verdict["score"], "faithfulness_reasoning": verdict["reasoning"],
                    "judge_model": self.judge.model, "judge_cached": True}

        self._count("judge_calls")
        try:
            score, reasoning, _ = self.judge.judge(
                GenerationResult(query=query, answer=answer, context_window=prepared.context_window)
            )
        except JudgeOutputError as e:
            return {**row, "judge_model": self.judge.model, "error": f"judge output unusable: {e}"}
        self._put(judge_cache, cache_key=verdict_key, metric=FAITHFULNESS, judge_model=self.judge.model,
                  score=score, reasoning=reasoning)
        return {**row, "faithfulness": score, "faithfulness_reasoning": reasoning,
                "judge_model": self.judge.model, "judge_cached": False}

    def _get(self, table, key: str) -> Optional[Dict[str, Any]]:
        with self.engine.connect() as conn:
            row = conn.execute(select(table).where(table.c.cache_key == key)).mappings().first()
        return dict(row) if row else None

    def _put(self, table, **values) -> None:
        with self.engine.begin() as conn:
            conn.execute(insert(table).values(**values).on_conflict_do_nothing(index_elements=[table.c.cache_key]))
