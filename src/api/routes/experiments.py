import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.dependencies import get_engine, get_testsets
from src.api.rate_limit import READ_LIMIT, limiter
from src.api.schemas.experiments import (
    ExperimentReport,
    ExperimentSummary,
    TestQuestionView,
    TestSetDetail,
    TestSetSummary,
)
from src.api.security import require_tenant
from src.evaluation import store
from src.evaluation.report import build_report
from src.stores.testsets import TestSetStore

router = APIRouter(prefix="/v1", tags=["experiments"])

LIST_LIMIT = 50


@router.get("/experiments", response_model=list[ExperimentSummary], operation_id="list_experiments")
@limiter.limit(READ_LIMIT)
async def list_experiments(request: Request, tenant_id: str = Depends(require_tenant), engine=Depends(get_engine)):
    """The workspace's experiments, newest first."""
    rows = await asyncio.to_thread(store.list_experiments, engine, tenant_id, LIST_LIMIT)
    return [ExperimentSummary(**{k: r[k] for k in ExperimentSummary.model_fields}) for r in rows]


@router.get("/experiments/{experiment_id}", response_model=ExperimentReport, operation_id="get_experiment")
@limiter.limit(READ_LIMIT)
async def get_experiment(
    request: Request, experiment_id: str, tenant_id: str = Depends(require_tenant), engine=Depends(get_engine),
):
    """The experiment's report: metrics per trial and each trial's verdict against the baseline."""
    experiment = await asyncio.to_thread(store.get_experiment, engine, experiment_id)
    # Another workspace's experiment is reported as missing, so experiment ids cannot be probed.
    if experiment is None or experiment["tenant_id"] != tenant_id:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return ExperimentReport(**await asyncio.to_thread(build_report, engine, experiment_id))


@router.get("/test-sets", response_model=list[TestSetSummary], operation_id="list_test_sets")
@limiter.limit(READ_LIMIT)
async def list_test_sets(request: Request, tenant_id: str = Depends(require_tenant), testsets: TestSetStore = Depends(get_testsets)):
    """The workspace's test sets with how many questions are pending, accepted and rejected."""
    return [TestSetSummary(**s) for s in await asyncio.to_thread(testsets.list_sets, tenant_id)]


@router.get("/test-sets/{name}", response_model=TestSetDetail, operation_id="get_test_set")
@limiter.limit(READ_LIMIT)
async def get_test_set(
    request: Request, name: str, tenant_id: str = Depends(require_tenant), testsets: TestSetStore = Depends(get_testsets),
):
    """One test set with every question, its review status and the passage it was written from."""
    head = await asyncio.to_thread(testsets.find, tenant_id, name)
    if head is None:
        raise HTTPException(status_code=404, detail="Test set not found")
    loaded = await asyncio.to_thread(testsets.load, head["test_set_id"])
    return TestSetDetail(
        name=head["name"], status=head["status"], content_hash=head.get("content_hash"), meta=loaded["meta"],
        abstained=len(loaded["abstained"]),
        questions=[TestQuestionView(position=i, **{k: q[k] for k in TestQuestionView.model_fields if k != "position"})
                   for i, q in enumerate(loaded["questions"])],
    )
