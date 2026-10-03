"""
The fetch task: turn a URL ingestion into Markdown pages in Postgres, then hand off to ingest.

Runs on the slim fetch worker (the API image, src/jobs/workers.py), never on the heavy
parse worker, and contacts only reader APIs (src/crawling/readers.py) — no process we host
sends a request to the target site. Imports nothing heavier than httpx.

Per job:
1. Re-check the fetch policy (https, public address, domain lists): DNS may have changed
   since the API accepted the URL.
2. A sitemap URL is expanded to at most MAX_SITEMAP_PAGES page URLs; any other URL is one page.
3. Pages are fetched one at a time, paced per domain, within the tenant's daily page quota.
   Pages already stored for this job (a retried run), or already indexed (resume=True),
   are skipped. Every attempt is written to fetch_log (the audit trail).
4. The ingest task is deferred; it reads the pages from fetched_pages.
"""
import asyncio
import logging

import procrastinate
from procrastinate.exceptions import AlreadyEnqueued

from src.config import get_settings
from src.crawling.policy import DomainPacer, check_fetchable
from src.crawling.readers import ReaderError, RobotsBlockedError, read_page
from src.crawling.sitemap import discover_pages, is_sitemap_url
from src.embedding.providers import build_embedder
from src.ingestion.errors import UnprocessableSourceError
from src.ingestion.url_policy import UnsafeUrlError, redact_url
from src.jobs.contract import (
    FETCH_QUEUE,
    FETCH_RECOVERY_TASK_NAME,
    FETCH_TASK,
    FETCH_TASK_NAME,
    INGEST_QUEUE,
    INGEST_TASK,
    MAX_RETRIES,
    RECOVERY_CRON,
    IngestionRequest,
)
from src.jobs.recovery import recover_stalled_jobs
from src.jobs.support import Progress, already_finished, job_tag
from src.registry.database import DocumentRegistry
from src.registry.engine import get_sync_engine
from src.retrieving.chunk_writes import existing_source_urls

logger = logging.getLogger(__name__)

fetch_blueprint = procrastinate.Blueprint()

MAX_SITEMAP_PAGES = 50  # bounds one job's reader calls, fetch quota and memory
MAX_LISTED_URLS_IN_METADATA = 20

_pacers = {}


def _pacer(min_interval: float) -> DomainPacer:
    """One pacer per process, so concurrent jobs on the same domain share its schedule."""
    if min_interval not in _pacers:
        _pacers[min_interval] = DomainPacer(min_interval)
    return _pacers[min_interval]


def _indexed_urls(request: IngestionRequest) -> set:
    if not request.resume:
        return set()
    index_id = build_embedder(get_settings()).index_id
    with get_sync_engine().connect() as conn:
        return existing_source_urls(conn, request.tenant_id, index_id, request.doc_id)


async def _fetch(request: IngestionRequest, app: procrastinate.App) -> None:
    settings = get_settings()
    tag = job_tag(request)
    registry = DocumentRegistry(get_sync_engine())
    progress = Progress(registry, request.job_id)

    async def audit(url, outcome, provider=None, detail=None):
        # Query strings can carry tokens: the audit trail keeps the address, not the secret.
        await asyncio.to_thread(registry.log_fetch, request.tenant_id, request.job_id, redact_url(url), outcome, provider, detail)

    async def report(pct, metadata=None):
        await asyncio.to_thread(progress.set, pct, metadata)

    try:
        await asyncio.to_thread(check_fetchable, request.url, settings.allowed_fetch_domains, settings.denied_fetch_domains)
    except UnsafeUrlError as e:
        await audit(request.url, "denied", detail=str(e))
        raise UnprocessableSourceError(str(e))

    await report(2)
    pacer = _pacer(settings.fetch_min_interval_seconds)

    async def authorize_child_sitemap(child: str) -> None:
        """A child sitemap is read through the reader like any page: policy, pacing and audit apply."""
        try:
            await asyncio.to_thread(check_fetchable, child, settings.allowed_fetch_domains, settings.denied_fetch_domains)
        except UnsafeUrlError as e:
            await audit(child, "denied", detail=str(e))
            raise
        await asyncio.sleep(pacer.reserve(child))
        await audit(child, "sitemap_child")

    if is_sitemap_url(request.url):
        await asyncio.sleep(pacer.reserve(request.url))  # the reader fetches the sitemap from the site too
        try:
            urls = await discover_pages(request.url, MAX_SITEMAP_PAGES, settings.firecrawl_api_key.get_secret_value(), authorize_child_sitemap)
        except RobotsBlockedError as e:
            await audit(request.url, "robots_blocked", "jina", str(e))
            raise UnprocessableSourceError(f"The sitemap is disallowed by the site's robots.txt: {e}")
        await audit(request.url, "sitemap", detail=f"{len(urls)} page urls selected")
        if not urls:
            raise UnprocessableSourceError("The sitemap lists no page URLs (after the ?filter= prefix, if any).")
    else:
        urls = [request.url]

    already_stored = await asyncio.to_thread(registry.fetched_urls, request.job_id)
    already_indexed = await asyncio.to_thread(_indexed_urls, request)
    pending = [u for u in urls if u not in already_stored | already_indexed]
    if not pending and not already_stored:
        logger.info(f"{tag} resume: every page is already indexed; nothing to fetch.")
        await asyncio.to_thread(registry.update_job_status, request.job_id, "complete", 100)
        return
    quota_left = settings.fetch_daily_page_quota - await asyncio.to_thread(registry.pages_fetched_today, request.tenant_id)
    logger.info(f"{tag} FETCH | {len(urls)} urls, {len(urls) - len(pending)} already done, quota left today {quota_left}")

    fetched, robots_blocked, denied, failed, over_quota = [], [], [], [], []
    for position, url in enumerate(pending, start=1):
        if len(fetched) >= quota_left:
            over_quota.append(url)
            await audit(url, "quota_exceeded", detail=f"daily quota of {settings.fetch_daily_page_quota} pages")
            continue
        try:
            await asyncio.to_thread(check_fetchable, url, settings.allowed_fetch_domains, settings.denied_fetch_domains)
            await asyncio.sleep(pacer.reserve(url))
            page = await read_page(url, settings.firecrawl_api_key.get_secret_value())
            if page.final_url and page.final_url != url:  # a redirect must not lead somewhere the policy forbids
                await asyncio.to_thread(
                    check_fetchable, page.final_url, settings.allowed_fetch_domains, settings.denied_fetch_domains
                )
        except UnsafeUrlError as e:
            denied.append(url)
            await audit(url, "denied", detail=str(e))
        except RobotsBlockedError as e:
            robots_blocked.append(url)
            await audit(url, "robots_blocked", "jina", str(e))
        except ReaderError as e:
            failed.append(url)
            await audit(url, "failed", detail=str(e))
        else:
            await asyncio.to_thread(registry.store_fetched_page, request.job_id, url, page.title, page.markdown, page.provider)
            await audit(url, "fetched", page.provider, f"{len(page.markdown)} chars")
            fetched.append(url)
        await report(2 + int(46 * position / len(pending)), _summary(urls, fetched, robots_blocked, denied, failed, over_quota))

    stored = await asyncio.to_thread(registry.fetched_urls, request.job_id)
    if not stored:
        reason = _no_pages_reason(robots_blocked, denied, failed, over_quota)
        if failed:  # a reader outage may pass: let Procrastinate retry the job
            raise ReaderError(reason)
        raise UnprocessableSourceError(reason)

    await report(50, _summary(urls, fetched, robots_blocked, denied, failed, over_quota))
    try:
        await app.configure_task(
            INGEST_TASK, queue=INGEST_QUEUE, lock=request.doc_id, queueing_lock=f"ingest-{request.job_id}"
        ).defer_async(**request.model_dump())
    except AlreadyEnqueued:
        logger.info(f"{tag} ingest already queued by an earlier attempt")
    logger.info(f"{tag} FETCH DONE | stored={len(stored)} robots_blocked={len(robots_blocked)} failed={len(failed)}")


def _summary(urls, fetched, robots_blocked, denied, failed, over_quota) -> dict:
    summary = {
        "total_pages": len(urls), "fetched_pages": len(fetched), "failed_pages": len(failed),
        "robots_blocked_pages": len(robots_blocked), "denied_pages": len(denied), "quota_skipped_pages": len(over_quota),
    }
    if robots_blocked:
        summary["robots_blocked_urls"] = robots_blocked[:MAX_LISTED_URLS_IN_METADATA]
    if failed or robots_blocked or over_quota:
        summary["error_reason"] = _no_pages_reason(robots_blocked, denied, failed, over_quota)
    return summary


def _no_pages_reason(robots_blocked, denied, failed, over_quota) -> str:
    parts = []
    if robots_blocked:
        parts.append(f"{len(robots_blocked)} disallowed by robots.txt")
    if denied:
        parts.append(f"{len(denied)} denied by the fetch policy")
    if failed:
        parts.append(f"{len(failed)} could not be read")
    if over_quota:
        parts.append(f"{len(over_quota)} skipped: daily page quota reached")
    return "No page was fetched: " + ", ".join(parts) if parts else "No page was fetched."


@fetch_blueprint.task(
    name=FETCH_TASK_NAME,
    queue=FETCH_QUEUE,
    retry=procrastinate.RetryStrategy(max_attempts=MAX_RETRIES, exponential_wait=30),
    pass_context=True,
)
async def fetch_source(context: procrastinate.JobContext, **kwargs) -> None:
    request = IngestionRequest(**kwargs)
    registry = DocumentRegistry(get_sync_engine())

    skip_reason = await asyncio.to_thread(already_finished, registry, request.job_id)
    if skip_reason:
        logger.info(f"[job={request.job_id[:8]}] not fetching: {skip_reason}.")
        return

    try:
        await _fetch(request, context.app)
    except UnprocessableSourceError as e:
        logger.error(f"[job={request.job_id[:8]}] nothing to fetch, not retrying: {e}")
        await asyncio.to_thread(registry.fail_job, request.job_id, str(e))
    except Exception as e:
        logger.error(f"[job={request.job_id[:8]}] fetch attempt {context.job.attempts + 1}/{MAX_RETRIES + 1} failed: {e}")
        if context.job.attempts >= MAX_RETRIES:
            await asyncio.to_thread(registry.fail_job, request.job_id, f"Fetching failed after {context.job.attempts + 1} attempts ({type(e).__name__}); details are in the worker log.")
        raise


@fetch_blueprint.periodic(cron=RECOVERY_CRON, periodic_id="recover-stalled-fetches")
@fetch_blueprint.task(name=FETCH_RECOVERY_TASK_NAME, queue=FETCH_QUEUE, lock=FETCH_RECOVERY_TASK_NAME, pass_context=True)
async def recover_stalled_fetches(context: procrastinate.JobContext, timestamp: int) -> None:
    report = await recover_stalled_jobs(context.app.job_manager, DocumentRegistry(get_sync_engine()), FETCH_QUEUE, FETCH_TASK)
    if report.requeued or report.failed:
        logger.warning(f"Stalled-fetch sweep: requeued={report.requeued} failed={report.failed}")
