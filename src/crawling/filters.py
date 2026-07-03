"""
URL filtering logic to control which URLs are crawled.
Prevents crawling off-topic pages, duplicates, and unwanted content.
"""

import re
from typing import List, Optional, Set
from urllib.parse import urlparse, parse_qs


class URLFilter:
    """Filters URLs based on domain, patterns, and duplicates."""

    def __init__(
        self,
        allowed_domains: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        allow_pagination: bool = False,
    ):
        """
        Args:
            allowed_domains: List of allowed domain patterns (e.g., ["docs.cloud.google.com"])
                           If empty, all domains are allowed.
            exclude_patterns: List of regex patterns to exclude (e.g., [r".*\\.pdf$"])
            allow_pagination: If False, exclude URLs with pagination params (?page=, ?offset=, etc.)
        """
        self.allowed_domains = allowed_domains or []
        self.exclude_patterns = [
            re.compile(pattern) for pattern in (exclude_patterns or [])
        ]
        self.allow_pagination = allow_pagination
        self.seen_urls: Set[str] = set()
        self.filtered_log = []

    def normalize_url(self, url: str) -> str:
        """Normalize URL for comparison (remove fragments, standardize scheme)."""
        # Remove fragment (#)
        url = url.split("#")[0]
        # Ensure lowercase domain
        parsed = urlparse(url)

        return f"{parsed.scheme}://{parsed.netloc.lower()}{parsed.path}{'?' + parsed.query if parsed.query else ''}"

    def is_domain_allowed(self, url: str) -> bool:
        """Check if URL domain is in allowed list."""
        if not self.allowed_domains:
            return True  # No restrictions

        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        for allowed in self.allowed_domains:
            if domain == allowed.lower() or domain.endswith("." + allowed.lower()):
                return True

        return False

    def is_pattern_excluded(self, url: str) -> bool:
        """Check if URL matches any exclusion patterns."""
        for pattern in self.exclude_patterns:
            if pattern.match(url):
                return True
        return False

    def has_pagination_params(self, url: str) -> bool:
        """Check if URL has pagination parameters."""
        pagination_keys = {"page", "offset", "start", "page_num", "pagenum"}
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        return bool(pagination_keys & set(params.keys()))

    def should_crawl(self, url: str, reason_log: bool = False) -> bool:
        """
        Determine if URL should be crawled.

        Args:
            url: URL to check
            reason_log: If True, log the reason for filtering

        Returns:
            True if URL should be crawled, False otherwise
        """
        normalized = self.normalize_url(url)

        # Check if already crawled
        if normalized in self.seen_urls:
            if reason_log:
                self._log_filter(url, "Duplicate: already crawled")
            return False

        # Check domain allowlist
        if not self.is_domain_allowed(url):
            if reason_log:
                self._log_filter(url, "Domain not allowed")
            return False

        # Check exclusion patterns
        if self.is_pattern_excluded(url):
            if reason_log:
                self._log_filter(url, "Matches exclusion pattern")
            return False

        # Check pagination
        if not self.allow_pagination and self.has_pagination_params(url):
            if reason_log:
                self._log_filter(url, "Pagination URL excluded")
            return False

        return True

    def mark_crawled(self, url: str):
        """Mark URL as crawled."""
        normalized = self.normalize_url(url)
        self.seen_urls.add(normalized)

    def _log_filter(self, url: str, reason: str):
        """Log a filtered URL."""
        self.filtered_log.append({"url": url, "reason": reason})

    def get_filter_report(self) -> dict:
        """Get summary of filtered URLs."""
        report = {"total_filtered": len(self.filtered_log), "by_reason": {}}

        for entry in self.filtered_log:
            reason = entry["reason"]
            report["by_reason"][reason] = report["by_reason"].get(reason, 0) + 1

        return report


class ConfigurableURLFilter(URLFilter):
    """Generic filter that allows requiring specific path keywords and minimum path length."""

    def __init__(
        self,
        allowed_domains: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        allow_pagination: bool = False,
        required_path_keywords: Optional[List[str]] = None,
        min_path_length: int = 0,
    ):
        super().__init__(
            allowed_domains=allowed_domains,
            exclude_patterns=exclude_patterns,
            allow_pagination=allow_pagination,
        )
        self.required_path_keywords = [
            kw.lower() for kw in (required_path_keywords or [])
        ]
        self.min_path_length = min_path_length

    def should_crawl(self, url: str, reason_log: bool = False) -> bool:
        """Applies generic filtering logic including keywords and path length."""
        # First, apply parent class filters
        if not super().should_crawl(url, reason_log=reason_log):
            return False

        parsed = urlparse(url)
        path_lower = parsed.path.lower()

        # Check required path keywords
        if self.required_path_keywords:
            has_keyword = any(kw in path_lower for kw in self.required_path_keywords)
            if not has_keyword:
                if reason_log:
                    self._log_filter(url, "Missing required path keywords")
                return False

        # Exclude short paths (like pure domain root navigation)
        if len(parsed.path) < self.min_path_length:
            if reason_log:
                self._log_filter(url, "Path too short (likely top-level navigation)")
            return False

        return True
