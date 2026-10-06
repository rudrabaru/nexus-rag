"""
Runs an experiment: every trial (retrieval configuration) over every query, one stored run each.

- Resumable. A query whose run is valid is never re-run; a missing, degraded or failed run is.
  Re-running a finished experiment is a no-op, and a paused one picks up where it stopped.
- Bounded concurrency: up to spec.concurrency queries of a trial at once, in one event loop.
  Trials share the retrieval resources, so each query is embedded once per index across all
  trials (the query-embedding cache), while each run still records what its embedding costs.
  Latency is recorded without the query embedding, whose cost depends on which trial ran first.
- A judge that cannot be called stops the experiment at the first failure instead of letting every
  remaining query retry it (and instead of letting another judge score the rest). A rate limit or outage
  pauses it (resume once the provider recovers); an unknown model or a bad key fails it, because resuming
  cannot help.
- Whatever goes wrong, the experiment ends as complete, paused or failed, never "running" forever.
"""
import asyncio
import logging
from collections import Counter
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
        "first_stage_rank": judge(result.candidates, query, relevance).rank if result.candidates else None,
        "retrieved": [
            {"chunk_id": c.chunk_id, "source": c.metadata.get("source_url") or c.source_document,
             "score": c.similarity_score, "match": m}
            for c, m in zip(result.chunks, judgement.matches)
        ],
        "latency_ms": result.latency_ms - result.embedding_latency_ms,
        "embedding_latency_ms": result.embedding_latency_ms,
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
    """Runs (or resumes) an experiment. Returns its final status: complete, paused or failed."""
    experiment = store.get_experiment(engine, experiment_id)
    if experiment is None:
        raise ValueError(f"No experiment {experiment_id}")
    spec = ExperimentSpec(**experiment["spec"])
    queries: List[EvaluationQuery] = [EvaluationQuery(**q) for q in experiment["queries"]]
    # One stage per trial: a trial may answer with its own model and context budget. Built up front, so a
    # judge that is the model it judges is refused before any query runs.
    stages: Dict[str, GenerationStage] = {}
    if spec.generation:
        stages = {
            label: GenerationStage(spec.generation, spec.tenant_id, engine, settings, trial.generation_model, trial.max_context_tokens)
            for label, trial in spec.trials.items()
        }
    await asyncio.to_thread(store.set_status, engine, experiment_id, "running")
    index_changes: List[str] = []

    async def finish(status: str, **extra) -> str:
        summary = {**_summary(spec, stages, index_changes), **extra}
        await asyncio.to_thread(store.set_status, engine, experiment_id, status, summary)
        return status

    try:
        for label, config in spec.trials.items():
            outage = await _run_trial(engine, resources, spec, stages.get(label), experiment_id, label, config, queries, index_changes, progress)
            if outage:
                if outage.permanent:
                    progress(f"judge cannot be used, experiment failed: {outage}")
                    return await finish("failed", error=f"judge cannot be used: {outage}")
                progress(f"judge unavailable, experiment paused (resume with: python -m src.evaluation resume {experiment_id}): {outage}")
                return await finish("paused")
        return await finish("complete")
    except BaseException as e:
        await finish("failed", error=f"{type(e).__name__}: {e}")
        raise


async def _run_trial(
    engine, resources, spec: ExperimentSpec, stage, experiment_id: str, label: str, config, queries, index_changes: List[str], progress,
) -> Optional[JudgeUnavailable]:
    """Runs one trial's pending queries. Returns the judge failure that stopped it, if any."""
    pipeline = build_pipeline(config, resources)
    chunk_store = pipeline.dense.chunk_store
    index_count = await asyncio.to_thread(chunk_store.get_collection_size, spec.tenant_id)
    trial = await asyncio.to_thread(
        store.ensure_trial, engine, experiment_id, label, config.model_dump(mode="json"), chunk_store.index_id, index_count
    )
    if trial["index_count"] != index_count:
        index_changes.append(f"{label}: workspace held {trial['index_count']} chunks in the index at start, {index_count} on resume")

    done = await asyncio.to_thread(store.completed_queries, engine, trial["trial_id"])
    pending = [i for i in range(len(queries)) if i not in done]
    progress(f"trial {label!r}: {len(done)}/{len(queries)} done, running {len(pending)}")
    semaphore = asyncio.Semaphore(spec.concurrency)
    stopped = asyncio.Event()
    outages: List[JudgeUnavailable] = []

    async def one(i: int) -> None:
        async with semaphore:
            if stopped.is_set():  # the judge is down: running more queries would only fail the same way
                return
            try:
                row = await run_query(pipeline, queries[i], spec.tenant_id, stage, spec.relevance)
            except JudgeUnavailable as e:
                outages.append(e)
                stopped.set()
                return
            await asyncio.to_thread(store.save_run, engine, trial["trial_id"], i, row)

    await asyncio.gather(*(one(i) for i in pending))
    return next((o for o in outages if o.permanent), outages[0] if outages else None)


def _summary(spec: ExperimentSpec, stages: Dict[str, GenerationStage], index_changes: List[str]) -> dict:
    """Run conditions a report needs to interpret its numbers (cache counters are this session's)."""
    models = {label: stage.model for label, stage in stages.items()}
    counters: Counter = Counter()
    for stage in stages.values():
        counters.update(stage.counters)
    return {
        "concurrency": spec.concurrency,
        "relevance": spec.relevance,
        # One name when every trial answers with the same model, otherwise the model of each trial.
        "generation_model": next(iter(models.values())) if len(set(models.values())) == 1 else (models or None),
        "judge_model": next(iter(stages.values())).judge.model if stages else None,
        "cache": dict(counters),
        "index_changes": index_changes,
    }
