import json
from typing import Any, AsyncIterator, Dict

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from src.api.capacity import acquire_slot
from src.api.dependencies import get_chat_service
from src.api.rate_limit import QUERY_LIMIT, limiter
from src.api.schemas.chat import ChatRequest, ChatResponse, ChatStreamEvents, RetrievalComparison, Source
from src.api.security import require_tenant
from src.services.chat_service import ChatService

router = APIRouter(prefix="/v1", tags=["chat"])

STREAM_DESCRIPTION = (
    "Server-sent events, one JSON object per `data:` line: `token` (answer text, repeated), `sources`, `done` "
    "(the answer is complete), then optionally `faithfulness` (seconds later, when requested). A failure while "
    "streaming ends the stream with an `error` event; a failure before it starts is an ordinary error response."
)


def _sse(event: Dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"


@router.post("/chat", response_model=ChatResponse, operation_id="chat")
@limiter.limit(QUERY_LIMIT)
async def chat(
    request: Request,
    body: ChatRequest,
    background: BackgroundTasks,
    tenant_id: str = Depends(require_tenant),
    service: ChatService = Depends(get_chat_service),
):
    """Answers a question from the workspace's documents, with the sources it used."""
    async with await acquire_slot(request):
        answer = await service.answer(await service.prepare(tenant_id, body.to_query()))
    if body.evaluate_faithfulness and answer.result is not None:
        background.add_task(service.judge, answer.result, answer.log_id)
    return ChatResponse(
        answer=answer.answer, sources=[Source(**vars(s)) for s in answer.sources], latency_ms=answer.latency_ms,
        latency_breakdown=answer.latency_breakdown,
    )


@router.post(
    "/chat/stream", operation_id="chat_stream", description=STREAM_DESCRIPTION,
    response_class=StreamingResponse, openapi_extra={"x-events": ChatStreamEvents.model_json_schema()},
)
@limiter.limit(QUERY_LIMIT)
async def chat_stream(
    request: Request,
    body: ChatRequest,
    tenant_id: str = Depends(require_tenant),
    service: ChatService = Depends(get_chat_service),
):
    """The same answer as `/v1/chat`, delivered as it is written."""
    slot = await acquire_slot(request)
    try:
        prepared = await service.prepare(tenant_id, body.to_query())  # a retrieval failure is still an HTTP error
    except BaseException:
        slot.release()
        raise

    async def stream() -> AsyncIterator[str]:
        try:
            async for event in service.events(prepared):
                yield _sse(event)
        finally:
            slot.release()

    # The slot belongs to the response: released when the stream ends or closes, or by the
    # response's background task, whichever runs first, so a client that disconnects before the
    # stream starts cannot leak it.
    return StreamingResponse(stream(), media_type="text/event-stream", background=BackgroundTask(slot.release))


@router.post("/retrieval/compare", response_model=RetrievalComparison, operation_id="compare_retrieval")
@limiter.limit(QUERY_LIMIT)
async def compare_retrieval(
    request: Request,
    body: ChatRequest,
    tenant_id: str = Depends(require_tenant),
    service: ChatService = Depends(get_chat_service),
):
    """What the reranker changes for this question: the same retrieval with and without it. No LLM is called."""
    comparison = await service.compare(tenant_id, body.to_query())
    return RetrievalComparison(
        baseline=[Source(**vars(s)) for s in comparison.baseline], reranked=[Source(**vars(s)) for s in comparison.reranked],
        baseline_latency_ms=comparison.baseline_latency_ms, reranked_latency_ms=comparison.reranked_latency_ms,
        reranker=comparison.reranker, degraded=comparison.degraded,
    )
