"""The document shape every source becomes before processing: Markdown plus provenance."""
from typing import Optional

from pydantic import BaseModel


class CrawledDocument(BaseModel):
    url: str  # the page URL, or upload://<doc_id>/<filename> for an uploaded file
    title: Optional[str] = None
    markdown_content: str
