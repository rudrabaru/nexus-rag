import re
import os
import urllib.request
import tempfile
import logging

logger = logging.getLogger(__name__)

def promote_fake_headings(markdown: str) -> str:
    """
    Detect lines that are entirely bold (**text**) and appear to function as
    section headers. Promote them to ## markdown headings.
    """
    lines = markdown.split("\n")
    result = []
    _BOLD_ONLY = re.compile(r"^\*\*([^*\n]{3,79})\*\*\s*$")

    for i, line in enumerate(lines):
        match = _BOLD_ONLY.match(line)
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        is_heading_like = (
            match
            and not line.rstrip().endswith((".", ",", ";", ":"))
            and next_line.strip() == ""
        )
        if is_heading_like:
            logger.debug(f"Promoting fake heading: {line!r}")
            result.append(f"## {match.group(1)}")
        else:
            result.append(line)
    return "\n".join(result)

def strip_tracked_change_artifacts(markdown: str) -> str:
    """
    Remove artifacts left by DOCX tracked changes extraction.

    python-docx sometimes emits both the deleted and accepted text as a run
    pair with no visual separator, creating repeated adjacent phrases or
    zero-width character sequences.
    """
    # Strip zero-width and soft-hyphen characters
    markdown = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", markdown)

    # Strip adjacent duplicate phrases (3+ word sequences repeated within 50 chars)
    # This targets patterns like: "the process the process" or "is running is running"
    markdown = re.sub(r"\b(\w+(?:\s+\w+){2,})\s+\1\b", r"\1", markdown)

    return markdown

def download_file(url: str) -> str:
    """Downloads a file from a URL to a temporary local path."""
    _temp_file = tempfile.NamedTemporaryFile(delete=False)
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    
    max_size = 50 * 1024 * 1024 # 50 MB limit
    downloaded_size = 0
    chunk_size = 65536 # 64 KB chunks

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                downloaded_size += len(chunk)
                if downloaded_size > max_size:
                    raise ValueError(f"File exceeds 50MB limit: {url}")
                _temp_file.write(chunk)
                
        _temp_file.flush()
        return _temp_file.name
    except Exception as e:
        _temp_file.close()
        try:
            os.unlink(_temp_file.name)
        except OSError:
            pass
        raise e
