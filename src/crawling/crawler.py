"""
Main crawler module using Crawl4AI for deep web crawling.
Orchestrates the crawling process and delegates to specialized submodules.
"""

import asyncio
import time
import logging
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from bs4 import BeautifulSoup

from crawl4ai import AsyncWebCrawler, CrawlResult
from .metadata import (
    CrawledDocument,
    CrawlConfig,
    CrawlMetrics,
    CrawlFailure,
    CrawlManifestEntry,
)
from .filters import URLFilter, ConfigurableURLFilter
from .sitemap import fetch_sitemap_urls
from .parsers import extract_title, extract_links

logger = logging.getLogger(__name__)


class WebCrawler:
    """Main crawler orchestrator using Crawl4AI."""

    def __init__(
        self,
        config: CrawlConfig,
        url_filter: Optional[URLFilter] = None,
    ):
        self.config = config
        self.url_filter = url_filter or URLFilter()

        self.metrics = CrawlMetrics()
        self.crawled_docs: List[CrawledDocument] = []
        self.failures: List[CrawlFailure] = []
        self.manifest: List[CrawlManifestEntry] = []
        self.queue: deque[tuple[str, int]] = deque([(config.start_url, 0)])

    async def crawl(self) -> List[CrawledDocument]:
        """Execute the crawl."""
        start_time = time.time()
        logger.info(f"Starting crawl from {self.config.start_url}")
        logger.info(
            f"Config: max_depth={self.config.max_depth}, max_pages={self.config.max_pages}"
        )

        try:
            from crawl4ai import BrowserConfig, CrawlerRunConfig, CacheMode

            browser_config = BrowserConfig(headless=True, java_script_enabled=True)
            run_config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                wait_until="networkidle",
                page_timeout=30000,
                magic=True,
                simulate_user=True,
            )

            async with AsyncWebCrawler(config=browser_config) as crawler:
                processed_urls = set()

                while self.queue and len(self.crawled_docs) < self.config.max_pages:
                    if self.metrics.total_urls_crawled >= self.config.abort_limit_pages:
                        logger.error(
                            f"ABORT LIMIT REACHED: {self.config.abort_limit_pages} pages. Forcing stop."
                        )
                        break

                    if (
                        self.metrics.total_urls_crawled
                        == self.config.warning_limit_pages
                    ):
                        logger.warning(
                            f"WARNING LIMIT REACHED: {self.config.warning_limit_pages} pages crawled."
                        )
                    elif (
                        self.metrics.total_urls_crawled == self.config.soft_limit_pages
                    ):
                        logger.info(
                            f"SOFT LIMIT REACHED: {self.config.soft_limit_pages} pages crawled."
                        )

                    if (
                        self.metrics.total_urls_crawled > 0
                        and self.metrics.total_urls_attempted % 20 == 0
                    ):
                        self._log_progress(start_time)

                    url, depth = self.queue.popleft()

                    if url in processed_urls:
                        continue
                    processed_urls.add(url)

                    if not self.url_filter.should_crawl(url, reason_log=True):
                        self.metrics.total_urls_filtered += 1
                        continue

                    self.metrics.total_urls_attempted += 1
                    logger.info(f"[{depth}] Crawling: {url}")

                    doc = await self._crawl_page(crawler, run_config, url, depth)

                    if doc:
                        self.crawled_docs.append(doc)
                        self.metrics.total_urls_crawled += 1
                        self.url_filter.mark_crawled(url)

                        if depth < self.config.max_depth:
                            new_urls = extract_links(url, doc.markdown_content)
                            for new_url in new_urls:
                                if new_url not in processed_urls:
                                    self.queue.append((new_url, depth + 1))
                    else:
                        self.metrics.total_urls_failed += 1

                    await self._async_sleep(self.config.delay_seconds)

        except Exception as e:
            logger.error(f"Crawl failed with error: {e}")
            raise

        finally:
            end_time = time.time()
            self.metrics.crawl_duration_seconds = end_time - start_time
            self._finalize_metrics()
            logger.info(str(self.metrics))

        return self.crawled_docs

    async def _crawl_page(
        self, crawler, run_config, url: str, depth: int
    ) -> Optional[CrawledDocument]:
        """Crawl a single page."""
        try:
            result: CrawlResult = await crawler.arun(url, config=run_config)

            if result.status_code >= 400:
                logger.warning(f"Status code {result.status_code} for {url}")
                self.failures.append(
                    CrawlFailure(url=url, error=f"Status code {result.status_code}")
                )
                return None

            title = extract_title(result)
            markdown_content = None

            if hasattr(result, "markdown") and result.markdown:
                markdown_content = result.markdown
            elif hasattr(result, "markdown_content") and result.markdown_content:
                markdown_content = result.markdown_content
            elif hasattr(result, "cleaned_html") and result.cleaned_html:
                markdown_content = result.cleaned_html
            elif hasattr(result, "html") and result.html:
                markdown_content = result.html

            raw_html = None
            if hasattr(result, "html") and result.html:
                raw_html = result.html

            if raw_html:
                soup = BeautifulSoup(raw_html, "html.parser")
                html_tag = soup.find("html")
                if html_tag and html_tag.has_attr("lang"):
                    lang = html_tag["lang"].lower()
                    if not lang.startswith("en"):
                        logger.info(f"Skipping {url} (Language: {lang})")
                        self.failures.append(
                            CrawlFailure(
                                url=url, error=f"Skipped non-English language: {lang}"
                            )
                        )
                        return None

            if not markdown_content:
                logger.warning(f"No markdown content extracted from {url}")
                logger.debug(f"Result attributes: {dir(result)}")
                self.failures.append(
                    CrawlFailure(url=url, error="No markdown content extracted")
                )
                return None

            word_count = len(markdown_content.split())

            links = extract_links(url, markdown_content, getattr(result, "links", None))

            doc = CrawledDocument(
                url=url,
                title=title,
                markdown_content=markdown_content,
                crawl_depth=depth,
                crawled_at=datetime.utcnow(),
                outgoing_links=links,
                word_count=word_count,
                status_code=200,
                raw_html=raw_html,
            )

            self.manifest.append(CrawlManifestEntry(url=url, status="success"))

            logger.info(f"[OK] Crawled {url} ({word_count} words)")
            return doc

        except Exception as e:
            logger.error(f"Failed to crawl {url}: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            self.failures.append(CrawlFailure(url=url, error=str(e)))
            return None

    def _log_progress(self, start_time: float):
        """Log the current progress of the crawl."""
        crawled = self.metrics.total_urls_crawled
        if crawled == 0:
            return

        elapsed = time.time() - start_time
        avg_time = elapsed / crawled
        queue_size = len(self.queue)
        eta_seconds = queue_size * avg_time

        eta_str = (
            f"{eta_seconds / 60:.1f}m"
            if eta_seconds < 3600
            else f"{eta_seconds / 3600:.1f}h"
        )

        logger.info(
            f"PROGRESS: Crawled: {crawled} | "
            f"Remaining (Queue): {queue_size} | "
            f"Avg Time: {avg_time:.2f}s/page | "
            f"ETA: {eta_str}"
        )

    def _finalize_metrics(self):
        """Calculate final metrics."""
        if self.crawled_docs:
            total_words = sum(doc.word_count for doc in self.crawled_docs)
            self.metrics.total_words = total_words
            self.metrics.average_page_words = total_words / len(self.crawled_docs)

    @staticmethod
    async def _async_sleep(seconds: float):
        """Async sleep for rate limiting."""
        await asyncio.sleep(seconds)


async def run_crawler(
    start_url: str = None,
    sitemap_url: str = None,
    allowed_domains: List[str] = None,
    required_keywords: List[str] = None,
    exclude_patterns: List[str] = None,
    max_depth: int = 3,
    max_pages: int = 100,
    soft_limit_pages: int = 1000,
    warning_limit_pages: int = 5000,
    abort_limit_pages: int = 10000,
    output_dir: str = "./raw_docs",
    html_dir: str = "./raw_html",
    metrics_dir: Optional[str] = None,
) -> List[CrawledDocument]:
    """
    Run the crawler with generic configuration.
    """
    if not start_url and not sitemap_url:
        raise ValueError("Must provide either start_url or sitemap_url")

    config = CrawlConfig(
        start_url=start_url or sitemap_url,
        max_depth=max_depth,
        max_pages=max_pages,
        soft_limit_pages=soft_limit_pages,
        warning_limit_pages=warning_limit_pages,
        abort_limit_pages=abort_limit_pages,
    )

    if exclude_patterns is None:
        exclude_patterns = [
            r".*\.pdf$",
            r".*\.jpg$",
            r".*\.png$",
            r".*\.gif$",
            r".*\.(zip|gz|tar)$",
        ]

    url_filter = ConfigurableURLFilter(
        allowed_domains=allowed_domains,
        exclude_patterns=exclude_patterns,
        required_path_keywords=required_keywords,
        min_path_length=0,
        allow_pagination=False,
    )

    crawler = WebCrawler(config, url_filter, output_dir=output_dir, html_dir=html_dir)

    if sitemap_url:
        logger.info(f"Fetching sitemap from {sitemap_url}")
        sitemap_urls = fetch_sitemap_urls(sitemap_url)
        logger.info(f"Found {len(sitemap_urls)} URLs in sitemap")

        valid_urls = [u for u in sitemap_urls if url_filter.should_crawl(u)]
        logger.info(f"Filtered to {len(valid_urls)} valid URLs")

        crawler.queue.clear()
        for u in valid_urls:
            crawler.queue.append((u, 0))

        if max_depth > 0:
            logger.warning(
                "Using sitemap with max_depth > 0. Crawler will follow outgoing links from sitemap URLs."
            )
        else:
            logger.info(
                "Using sitemap with max_depth = 0. Crawler will only process sitemap URLs."
            )

    docs = await crawler.crawl()

    return docs
