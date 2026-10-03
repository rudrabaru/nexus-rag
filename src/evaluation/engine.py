"""
Runs an experiment: every trial (retrieval configuration) over every query, one stored run each.

- Resumable. A query whose run is valid is never re-run; a missing, degraded or failed run is.
  Re-running a finished experiment is a no-op, and a paused one picks up where it stopped.
- Bounded concurrency: up to spec.concurrency queries of a trial at once, in one event loop.
  Trials share the retrieval resources, so each query is embedded once per index across all
  trials (the query-embedding cache), while each run still records what its embedding costs.
- A judge that cannot be called pauses the experiment (status "paused") instead of letting
  another judge score the rest; resume it once the provider recovers.
"""
import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy.engine import Engine

from src.config import Settings
from src.evaluation import store
from src.evaluation.dataset import EvaluationQuery
from src.evaluation.generation import GenerationStage
from src.evaluation.relevance import judge
from src.evaluation.spec import ExperimentSpec
from src.generating.evaluator import JudgeUnavailable
from src.retrieving.pipeline import RetrievalPipeline, RetrievalResources, build_pipeline

logger = logging.getLogger(__name__)


async def run_query(
    pipeline: RetrievalPipeline, query: EvaluationQuery, tenant_id: str, stage: Optional[GenerationStage],
    relevance: str = "document",
) -> Dict[str, Any]:
    """The run row for one query. Retrieval failures become an invalid run, not a crash."""
    try:
        result = await pipeline.run(query.query, tenant_id)
    except Exception as e:
        logger.warning(f"EVAL | retrieval failed for {query.query!r}: {e}")
        return {"retrieved": [], "latency_ms": 0.0, "error": f"retrieval failed: {type(e).__name__}: {e}"}

    judgement = judge(result.chunks, query, relevance)
    row: Dict[str, Any] = {
        "rank": judgement.rank,
        "exact_rank": judgement.exact_rank,
        "first_stage_rank": judge(result.candidates, query, relevance).rank if result.candidates else None,
        "retrieved": [
            {"chunk_id": c.chunk_id, "source": c.metadata.get("source_url") or c.source_document,
             "score": c.similarity_score, "match": m}
            for c, m in zip(result.chunks, judgement.matches)
        ],
        "latency_ms": result.latency_ms,
        "embedding_tokens": result.query_embedding_tokens,
        "embedding_cost_usd": pipeline.dense.embedder.cost_usd(result.query_embedding_tokens),
        "rerank_cost_usd": result.rerank_cost_usd,
        "degraded": result.degraded,
        "error": None,
    }
    if stage and not result.degraded:
        row.update(await asyncio.to_thread(stage.run, query.query, result))
    return row


async def run_experiment(
    engine: Engine, resources: RetrievalResources, settings: Settings, experiment_id: str,
    progress: Callable[[str], None] = print,
) -> str:
    """Runs (or resumes) an experiment. Returns its final status: complete or paused."""
    experiment = store.get_experiment(engine, experiment_id)
    if experiment is None:
        raise ValueError(f"No experiment {experiment_id}")
    spec = ExperimentSpec(**experiment["spec"])
    queries: List[EvaluationQuery] = [EvaluationQuery(**q) for q in experiment["queries"]]
    stage = GenerationStage(spec.generation, spec.tenant_id, engine, settings) if spec.generation else None
    await asyncio.to_thread(store.set_status, engine, experiment_id, "running")

    index_changes = []
    for label, config in spec.trials.items():
        pipeline = build_pipeline(config, resources)
        chunk_store = pipeline.dense.chunk_store
        index_count = await asyncio.to_thread(chunk_store.get_collection_size)
        trial = await asyncio.to_thread(
            store.ensure_trial, engine, experiment_id, label, config.model_dump(mode="json"), chunk_store.index_id, index_count
        )
        if trial["index_count"] != index_count:
            index_changes.append(f"{label}: index held {trial['index_count']} chunks at start, {index_count} on resume")

        done = await asyncio.to_thread(store.completed_queries, engine, trial["trial_id"])
        pending = [i for i in range(len(queries)) if i not in done]
        progress(f"trial {label!r}: {len(done)}/{len(queries)} done, running {len(pending)}")
        semaphore = asyncio.Semaphore(spec.concurrency)

        async def one(i: int) -> None:
            async with semaphore:
                row = await run_query(pipeline, queries[i], spec.tenant_id, stage, spec.relevance)
                await asyncio.to_thread(store.save_run, engine, trial["trial_id"], i, row)

        outcomes = await asyncio.gather(*(one(i) for i in pending), return_exceptions=True)
        paused = [o for o in outcomes if isinstance(o, JudgeUnavailable)]
        crashed = [o for o in outcomes if isinstance(o, BaseException) and not isinstance(o, JudgeUnavailable)]
        if crashed:
            await asyncio.to_thread(store.set_status, engine, experiment_id, "failed", _summary(spec, stage, index_changes))
            raise crashed[0]
        if paused:
            progress(f"judge unavailable, experiment paused (resume with: python -m src.evaluation resume {experiment_id}): {paused[0]}")
            await asyncio.to_thread(store.set_status, engine, experiment_id, "paused", _summary(spec, stage, index_changes))
            return "paused"

    await asyncio.to_thread(store.set_status, engine, experiment_id, "complete", _summary(spec, stage, index_changes))
    return "complete"


def _summary(spec: ExperimentSpec, stage: Optional[GenerationStage], index_changes: List[str]) -> dict:
    """Run conditions a report needs to interpret its numbers (cache counters are this session's)."""
    return {
        "concurrency": spec.concurrency,
        "relevance": spec.relevance,
        "generation_model": stage.model if stage else None,
        "judge_model": stage.judge.model if stage else None,
        "cache": dict(stage.counters) if stage else {},
        "index_changes": index_changes,
    }
