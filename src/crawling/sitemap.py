import urllib.request
import xml.etree.ElementTree as ET
import logging
from typing import List

logger = logging.getLogger(__name__)


def fetch_sitemap_urls(sitemap_url: str) -> List[str]:
    """
    Fetch and parse an XML sitemap to extract all URL locations.
    Handles standard sitemap.xml formats including sitemap index files.
    """
    urls = []
    try:
        req = urllib.request.Request(sitemap_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            xml_data = response.read()

        root = ET.fromstring(xml_data)
        for elem in root.iter():
            # Strip XML namespace prefix to handle any sitemap schema version generically
            tag = elem.tag.split("}", 1)[-1]
            if tag == "loc" and elem.text and elem.text.strip():
                urls.append(elem.text.strip())
    except Exception as e:
        logger.error(f"Failed to fetch or parse sitemap {sitemap_url}: {e}")

    return list(set(urls))
