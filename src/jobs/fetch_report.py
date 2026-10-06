"""What a fetch run tells its job: a page's identity, the per-job counters and the reason nothing was fetched."""
import hashlib

MAX_LISTED_URLS_IN_METADATA = 20


def content_key(markdown: str) -> str:
    """Identifies a page by its text alone, ignoring whitespace differences."""
    return hashlib.sha256(" ".join(markdown.split()).encode("utf-8")).hexdigest()


def summary(urls, fetched, robots_blocked, denied, failed, over_quota, duplicates) -> dict:
    summary = {
        "total_pages": len(urls), "fetched_pages": len(fetched), "failed_pages": len(failed),
        "robots_blocked_pages": len(robots_blocked), "denied_pages": len(denied), "quota_skipped_pages": len(over_quota),
        "duplicate_pages": len(duplicates),
    }
    if robots_blocked:
        summary["robots_blocked_urls"] = robots_blocked[:MAX_LISTED_URLS_IN_METADATA]
    if failed or robots_blocked or over_quota:
        summary["error_reason"] = no_pages_reason(robots_blocked, denied, failed, over_quota)
    return summary


def no_pages_reason(robots_blocked, denied, failed, over_quota) -> str:
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
