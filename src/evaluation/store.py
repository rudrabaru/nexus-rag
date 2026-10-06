"""
Experiment persistence. Postgres is the system of record for results: every per-query run is a
row, so a report can be recomputed, drilled into, or re-tested at any time.

Synchronous on purpose, like the registry: the engine calls it through asyncio.to_thread.
"""
import uuid
from typing import Any, Dict, List, Optional, Set

from sqlalchemy import and_, desc, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine

from src.evaluation.dataset import Dataset
from src.evaluation.spec import ExperimentSpec
from src.db.rows import row_to_dict, utcnow
from src.db.schema import experiments, runs, trials


def create_experiment(engine: Engine, spec: ExperimentSpec, dataset: Dataset) -> str:
    experiment_id = str(uuid.uuid4())
    with engine.begin() as conn:
        conn.execute(
            experiments.insert().values(
                experiment_id=experiment_id, name=spec.name, tenant_id=spec.tenant_id,
                spec=spec.model_dump(mode="json"), dataset_name=dataset.name, dataset_hash=dataset.content_hash,
                queries=[q.model_dump() for q in dataset.queries], status="running",
            )
        )
    return experiment_id


def get_experiment(engine: Engine, experiment_id: str) -> Optional[Dict[str, Any]]:
    with engine.connect() as conn:
        return row_to_dict(conn.execute(select(experiments).where(experiments.c.experiment_id == experiment_id)).first())


def list_experiments(engine: Engine, tenant_id: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    stmt = select(
        experiments.c.experiment_id, experiments.c.name, experiments.c.tenant_id, experiments.c.dataset_name,
        experiments.c.status, experiments.c.created_at, experiments.c.finished_at,
    ).order_by(desc(experiments.c.created_at)).limit(limit)
    if tenant_id:
        stmt = stmt.where(experiments.c.tenant_id == tenant_id)
    with engine.connect() as conn:
        return [row_to_dict(r) for r in conn.execute(stmt)]


def set_status(engine: Engine, experiment_id: str, status: str, summary: Optional[dict] = None) -> None:
    values: Dict[str, Any] = {"status": status}
    if summary is not None:
        values["summary"] = summary
    if status in ("complete", "failed"):
        values["finished_at"] = utcnow()
    with engine.begin() as conn:
        conn.execute(update(experiments).where(experiments.c.experiment_id == experiment_id).values(**values))


def ensure_trial(engine: Engine, experiment_id: str, label: str, config: dict, index_id: str, index_count: int) -> Dict[str, Any]:
    """The trial row, created on first run; a resumed experiment gets the original row back."""
    stmt = insert(trials).values(
        trial_id=str(uuid.uuid4()), experiment_id=experiment_id, label=label, config=config,
        index_id=index_id, index_count=index_count,
    ).on_conflict_do_nothing(index_elements=[trials.c.experiment_id, trials.c.label])
    with engine.begin() as conn:
        conn.execute(stmt)
        return row_to_dict(conn.execute(
            select(trials).where(and_(trials.c.experiment_id == experiment_id, trials.c.label == label))
        ).first())


def trials_of(engine: Engine, experiment_id: str) -> List[Dict[str, Any]]:
    with engine.connect() as conn:
        rows = conn.execute(select(trials).where(trials.c.experiment_id == experiment_id).order_by(trials.c.created_at))
        return [row_to_dict(r) for r in rows]


def completed_queries(engine: Engine, trial_id: str) -> Set[int]:
    """Queries with a valid run. Degraded or failed runs are not complete: a resume retries them."""
    stmt = select(runs.c.query_index).where(
        runs.c.trial_id == trial_id, runs.c.error.is_(None), runs.c.degraded == [],
    )
    with engine.connect() as conn:
        return set(conn.execute(stmt).scalars())


RUN_DEFAULTS = {"retrieved": [], "latency_ms": 0.0, "embedding_tokens": 0, "embedding_cost_usd": 0.0,
                "rerank_cost_usd": 0.0, "degraded": []}


def save_run(engine: Engine, trial_id: str, query_index: int, row: Dict[str, Any]) -> None:
    """
    Insert or replace: a retried run overwrites the invalid one it retries. Every column is
    written, so nothing from the earlier attempt (a stale answer, rank or error) survives.
    """
    columns = [c.name for c in runs.columns if c.name not in ("trial_id", "query_index", "created_at")]
    values = {"trial_id": trial_id, "query_index": query_index,
              **{name: RUN_DEFAULTS.get(name) for name in columns}, **row}
    stmt = insert(runs).values(**values)
    stmt = stmt.on_conflict_do_update(
        index_elements=[runs.c.trial_id, runs.c.query_index],
        set_={k: stmt.excluded[k] for k in values if k not in ("trial_id", "query_index")},
    )
    with engine.begin() as conn:
        conn.execute(stmt)


def runs_of(engine: Engine, trial_id: str) -> List[Dict[str, Any]]:
    with engine.connect() as conn:
        rows = conn.execute(select(runs).where(runs.c.trial_id == trial_id).order_by(runs.c.query_index))
        return [row_to_dict(r) for r in rows]
