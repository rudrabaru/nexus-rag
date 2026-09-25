"""
Bounded concurrent fetching of a sitemap's pages.

Split out of the ingestion task (src/jobs/tasks.py) because it is pure ingestion mechanics —
independent of the job queue — and is easier to reason about and test on its own.
"""
import asyncio
import logging
from typing import Callable, List, Optional, Tuple

from src.crawling.metadata import CrawledDocument, VisualChunkDraft
from src.ingestion.dispatcher import IngestionDispatcher
from src.ingestion.url_policy import validate_public_url

logger = logging.getLogger(__name__)

MAX_SITEMAP_PAGES = 50  # matches the API's own sitemap page cap; keeps one job's memory bounded
FETCH_CONCURRENCY = 4
PAGE_TIMEOUT_SECONDS = 20.0  # stops one hung page from stalling the whole job

ProgressCallback = Callable[[int, dict], None]


async def fetch_sitemap_pages(
    dispatcher: IngestionDispatcher,
    urls: List[str],
    existing_urls: set,
    extract_visuals: bool,
    on_progress: ProgressCallback,
    tag: str,
) -> Tuple[List[CrawledDocument], List[VisualChunkDraft], List[str]]:
    """
    Fetches every sitemap URL concurrently (bounded), skipping ones already indexed (resume).
    Returns (documents, visual_chunks, failed_reasons). on_progress(pct, metadata) is called
    as each page finishes, so a job's progress is visible while a large sitemap is in flight.
    """
    all_docs, all_visual_chunks, failed_reasons = [], [], []
    total = len(urls)
    processed = failed_pages = 0
    sem = asyncio.Semaphore(FETCH_CONCURRENCY)

    def report():
        on_progress(5 + int(45 * processed / total), {"total_pages": total, "indexed_pages": processed - failed_pages, "failed_pages": failed_pages})

    async def fetch_one(url: str) -> Tuple[Optional[object], Optional[str]]:
        nonlocal processed, failed_pages
        if url in existing_urls:
            processed += 1
            report()
            return None, None

        async with sem:
            try:
                await asyncio.to_thread(validate_public_url, url)
                result = await asyncio.wait_for(
                    dispatcher.web_adapter.ingest(url, extract_visuals=extract_visuals), timeout=PAGE_TIMEOUT_SECONDS
                )
                processed += 1
                report()
                return result, None
            except Exception as e:
                logger.warning(f"{tag} Sitemap skipped URL {url}: {e}")
                failed_pages += 1
                processed += 1
                report()
                return None, f"{url}: {str(e)[:60]}"

    results = await asyncio.gather(*(fetch_one(u) for u in urls))
    for result, err in results:
        if result and result.documents:
            all_docs.extend(result.documents)
        if result and result.visual_chunks:
            all_visual_chunks.extend(result.visual_chunks)
        if err:
            failed_reasons.append(err)
    return all_docs, all_visual_chunks, failed_reasons
