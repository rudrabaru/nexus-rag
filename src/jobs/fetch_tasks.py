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
from src.crawling.sitemap import MAX_SITEMAP_PAGES, discover_pages, is_sitemap_url
from src.embedding.providers import build_embedder
from src.errors import UnprocessableSourceError
from src.crawling.url_policy import UnsafeUrlError, redact_url
from src.jobs.contract import (
    FETCH_QUEUE,
    FETCH_RECOVERY_TASK_NAME,
    FETCH_TASK,
    FETCH_TASK_NAME,
    INGEST_QUEUE,
    INGEST_TASK,
    MAX_RETRIES,
    IngestionRequest,
)
from src.jobs.fetch_report import content_key, no_pages_reason, summary
from src.jobs.policy import register_recovery, run_with_policy
from src.jobs.support import Progress, job_tag
from src.db.engine import get_sync_engine
from src.retrieving.chunk_writes import existing_source_urls
from src.stores.fetches import FetchStore
from src.stores.jobs import JobStore

logger = logging.getLogger(__name__)

fetch_blueprint = procrastinate.Blueprint()

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
    engine = get_sync_engine()
    fetches, jobs = FetchStore(engine), JobStore(engine)
    progress = Progress(jobs, request.job_id)

    async def audit(url, outcome, provider=None, detail=None):
        # Query strings can carry tokens: the audit trail keeps the address, not the secret.
        await asyncio.to_thread(fetches.log_fetch, request.tenant_id, request.job_id, redact_url(url), outcome, provider, detail)

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

    already_stored = await asyncio.to_thread(fetches.fetched_urls, request.job_id)
    already_indexed = await asyncio.to_thread(_indexed_urls, request)
    pending = [u for u in urls if u not in already_stored | already_indexed]
    if not pending and not already_stored:
        logger.info(f"{tag} resume: every page is already indexed; nothing to fetch.")
        await asyncio.to_thread(jobs.update_job_status, request.job_id, "complete", 100)
        return
    quota_left = settings.fetch_daily_page_quota - await asyncio.to_thread(fetches.pages_fetched_today, request.tenant_id)
    logger.info(f"{tag} FETCH | {len(urls)} urls, {len(urls) - len(pending)} already done, quota left today {quota_left}")

    fetched, robots_blocked, denied, failed, over_quota, duplicates = [], [], [], [], [], []
    seen_content = set()
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
            key = content_key(page.markdown)
            if key in seen_content:  # the same text under another URL: a block page or a mirrored template
                duplicates.append(url)
                await audit(url, "duplicate_content", page.provider, "identical to a page already fetched in this job")
                continue
            seen_content.add(key)
            await asyncio.to_thread(fetches.store_fetched_page, request.job_id, url, page.title, page.markdown, page.provider)
            await audit(url, "fetched", page.provider, f"{len(page.markdown)} chars")
            fetched.append(url)
        await report(2 + int(46 * position / len(pending)), summary(urls, fetched, robots_blocked, denied, failed, over_quota, duplicates))

    stored = await asyncio.to_thread(fetches.fetched_urls, request.job_id)
    if not stored:
        reason = no_pages_reason(robots_blocked, denied, failed, over_quota)
        if failed:  # a reader outage may pass: let Procrastinate retry the job
            raise ReaderError(reason)
        raise UnprocessableSourceError(reason)

    await report(50, summary(urls, fetched, robots_blocked, denied, failed, over_quota, duplicates))
    try:
        await app.configure_task(
            INGEST_TASK, queue=INGEST_QUEUE, lock=request.doc_id, queueing_lock=f"ingest-{request.job_id}"
        ).defer_async(**request.model_dump())
    except AlreadyEnqueued:
        logger.info(f"{tag} ingest already queued by an earlier attempt")
    logger.info(f"{tag} FETCH DONE | stored={len(stored)} robots_blocked={len(robots_blocked)} failed={len(failed)}")


@fetch_blueprint.task(
    name=FETCH_TASK_NAME,
    queue=FETCH_QUEUE,
    retry=procrastinate.RetryStrategy(max_attempts=MAX_RETRIES, exponential_wait=30),
    pass_context=True,
)
async def fetch_source(context: procrastinate.JobContext, **kwargs) -> None:
    request = IngestionRequest(**kwargs)
    await run_with_policy(context, request, lambda: _fetch(request, context.app), "fetching")


register_recovery(fetch_blueprint, FETCH_QUEUE, FETCH_TASK, FETCH_RECOVERY_TASK_NAME)
