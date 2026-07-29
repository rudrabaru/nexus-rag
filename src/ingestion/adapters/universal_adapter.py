import logging
from pathlib import Path
from datetime import datetime, timezone
from src.crawling.metadata import CrawledDocument, AdapterResult
from src.ingestion.base import IngestionAdapter
from markitdown import MarkItDown
import os
import asyncio

from src.ingestion.adapters.helpers import promote_fake_headings, strip_tracked_change_artifacts, download_file_async

logger = logging.getLogger(__name__)

class UniversalAdapter(IngestionAdapter):
    """
    Universal adapter using MarkItDown to handle PDFs, PPTX, DOCX, XLSX, etc.
    Converts directly to pristine Markdown.

    For scanned/image-only PDFs, automatically falls back to page-by-page
    Gemini Vision extraction when text content is sparse (<50 chars).
    This fallback is independent of the extract_visuals flag — that flag
    controls visual chunk extraction, not text recovery from scanned pages.
    """
    
    def __init__(self):
        self.md_standard = MarkItDown()
        self.md_vision = None
        
        provider = os.environ.get("LLM_PROVIDER", "gemini")
        if provider == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY")
            if api_key:
                from openai import OpenAI
                llm_client = OpenAI(
                    api_key=api_key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                )
                llm_model = os.environ.get("LLM_MODEL_NAME", "gemini-2.5-flash")
                self.md_vision = MarkItDown(llm_client=llm_client, llm_model=llm_model)
                logger.info("UniversalAdapter: Vision extraction enabled (Gemini)")
        elif provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                from openai import OpenAI
                llm_client = OpenAI(api_key=api_key)
                llm_model = os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini")
                self.md_vision = MarkItDown(llm_client=llm_client, llm_model=llm_model)
                logger.info("UniversalAdapter: Vision extraction enabled (OpenAI)")
        
    async def ingest(self, source: str, extract_visuals: bool = False, **kwargs) -> AdapterResult:
        logger.info(f"UniversalAdapter processing: {source}")
        
        local_path = source
        _temp_file = None

        if source.startswith("http://") or source.startswith("https://"):
            try:
                import socket
                import ipaddress
                from urllib.parse import urlparse
                
                parsed = urlparse(source)
                hostname = parsed.hostname
                
                if not hostname:
                    raise ValueError("Invalid URL format.")
                    
                ip = socket.gethostbyname(hostname)
                parsed_ip = ipaddress.ip_address(ip)
                
                if parsed_ip.is_private or parsed_ip.is_loopback or parsed_ip.is_link_local:
                    raise ValueError(f"SSRF blocked: Host resolves to internal IP {ip}")
                    
                local_path = await download_file_async(source)
                _temp_file = local_path
            except Exception as e:
                logger.error(f"Failed to download {source}: {e}")
                raise Exception(f"Download failed: {str(e)}")

        try:
            if local_path.lower().endswith(".txt"):
                with open(local_path, encoding="utf-8", errors="replace") as f:
                    markdown_content = f.read()
                logger.info(f"ADAPTER | TXT read | chars={len(markdown_content)}")
            else:
                md_instance = self.md_vision if (extract_visuals and self.md_vision) else self.md_standard
                mode = "vision" if md_instance is self.md_vision else "standard"
                logger.info(f"ADAPTER | MarkItDown mode={mode!r} extract_visuals={extract_visuals} source={source!r}")
                if extract_visuals and not self.md_vision:
                    logger.warning("ADAPTER | extract_visuals=True but no LLM client configured. Using standard extraction.")

                try:
                    result = await asyncio.to_thread(md_instance.convert, local_path)
                    markdown_content = result.text_content or ""
                    logger.info(f"ADAPTER | MarkItDown done | chars={len(markdown_content)}")
                except Exception as convert_err:
                    if source.lower().endswith(".pdf"):
                        logger.warning(
                            f"ADAPTER | MarkItDown conversion FAILED for {source!r}: {convert_err!r}. "
                            "Will attempt vision fallback if this is a PDF."
                        )
                        markdown_content = ""
                    else:
                        raise ValueError(f"MarkItDown failed to parse document: {str(convert_err)}") from convert_err
            
            if source.lower().endswith(".docx"):
                markdown_content = strip_tracked_change_artifacts(markdown_content)
                markdown_content = promote_fake_headings(markdown_content)
                logger.info(f"ADAPTER | DOCX post-processing done | chars={len(markdown_content)}")
                
            if source.lower().endswith(".pdf"):
                lines = [line for line in markdown_content.splitlines() if line.strip()]
                if not lines or (sum(len(line) for line in lines) / len(lines)) < 5:
                    logger.warning(
                        f"ADAPTER | Suspicious text for {source!r} "
                        "(empty or avg line length < 5). Retrying with PyMuPDF text extraction."
                    )
                    try:
                        import fitz
                        pdf = fitz.open(local_path)
                        markdown_content = "\n\n".join(page.get_text() for page in pdf)
                        logger.info(f"ADAPTER | PyMuPDF text fallback done | chars={len(markdown_content)}")
                    except Exception as e:
                        logger.error(f"ADAPTER | PyMuPDF text fallback FAILED: {e!r}")
            
            title = Path(source).name
            word_count = len(markdown_content.split())
            
            if len(markdown_content) < 50 and local_path.lower().endswith(".pdf"):
                logger.info(
                    f"ADAPTER | Sparse content ({len(markdown_content)} chars) for {source!r}. "
                    "Attempting page-by-page Gemini Vision extraction..."
                )
                try:
                    import fitz
                    from src.ingestion.visual_processor import VisualProcessor
                    logger.info("ADAPTER | VISION | Initialising VisualProcessor...")
                    processor = VisualProcessor()
                    logger.info(f"ADAPTER | VISION | fitz.open({local_path!r})")
                    pdf = fitz.open(local_path)
                    page_count = len(pdf)
                    logger.info(f"ADAPTER | VISION | PDF has {page_count} pages")
                    page_texts = []
                    for page_num, page in enumerate(pdf):
                        mat = fitz.Matrix(150 / 72, 150 / 72)
                        pix = page.get_pixmap(matrix=mat)
                        image_bytes = pix.tobytes("png")
                        img_kb = len(image_bytes) // 1024
                        logger.info(f"ADAPTER | VISION | Page {page_num+1}/{page_count}: rendered {img_kb} KB PNG in-memory")
                        
                        page_md = await asyncio.to_thread(processor.describe_page, image_bytes)
                        md_chars = len(page_md.strip())
                        logger.info(f"ADAPTER | VISION | Page {page_num+1}/{page_count}: describe_page returned {md_chars} chars")
                        if page_md.strip():
                            page_texts.append(f"## Page {page_num + 1}\n\n{page_md}")
                            
                        if page_num < page_count - 1:
                            await asyncio.sleep(2)
                            
                    if page_texts:
                        markdown_content = "\n\n---\n\n".join(page_texts)
                        word_count = len(markdown_content.split())
                        logger.info(f"ADAPTER | VISION | Extraction complete: {len(page_texts)} pages, {len(markdown_content)} chars total")
                    else:
                        logger.error(f"ADAPTER | VISION | All {page_count} pages returned empty content from describe_page")
                except Exception as e:
                    import traceback as _tb
                    logger.error(
                        f"ADAPTER | VISION | Vision fallback FAILED with {type(e).__name__}: {e!r}\n"
                        f"{_tb.format_exc()}"
                    )
            
            crawled_doc = CrawledDocument(
                url=f"file:///{source}",
                title=title,
                markdown_content=markdown_content,
                crawl_depth=0,
                crawled_at=datetime.now(timezone.utc),
                word_count=word_count,
                status_code=200,
            )
            
            return AdapterResult(documents=[crawled_doc], visual_chunks=[])
            
        except Exception as e:
            logger.error(f"UniversalAdapter failed to read {source}: {e}")
            raise Exception(f"File processing failed: {str(e)}")
        finally:
            if _temp_file:
                try:
                    os.unlink(_temp_file)
                except Exception as cleanup_error:
                    logger.warning(
                        f"Failed to clean up temp file {_temp_file}: {cleanup_error}"
                    )
