import fitz  # PyMuPDF
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from src.crawling.metadata import CrawledDocument, AdapterResult, VisualChunkDraft
from src.ingestion.base import IngestionAdapter

logger = logging.getLogger(__name__)


class PDFAdapter(IngestionAdapter):
    """
    Adapter for reading local PDF files and extracting text, tables, and images.
    """

    async def ingest(
        self, source: str, extract_visuals: bool = False, **kwargs
    ) -> AdapterResult:
        logger.info(f"PDFAdapter reading: {source}, extract_visuals={extract_visuals}")

        local_path = source
        _temp_file = None

        if source.startswith("http://") or source.startswith("https://"):
            import tempfile
            import urllib.request
            import os

            try:
                _temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
                headers = {"User-Agent": "Mozilla/5.0"}
                req = urllib.request.Request(source, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    _temp_file.write(resp.read())
                _temp_file.flush()
                local_path = _temp_file.name
            except Exception as e:
                logger.error(f"Failed to download PDF {source}: {e}")
                if _temp_file:
                    os.unlink(_temp_file.name)
                return AdapterResult(documents=[], visual_chunks=[])

        try:
            try:
                doc = fitz.open(local_path)
            except fitz.PasswordError:
                logger.error(f"Failed to read PDF {source}: Password protected.")
                return AdapterResult(documents=[], visual_chunks=[])

            text_blocks = []
            visual_chunks = []

            # Generate deterministic doc_id from source path
            doc_id = hashlib.md5(source.encode()).hexdigest()

            processor = None
            asset_store = None
            if extract_visuals:
                try:
                    from src.ingestion.visual_processor import VisualProcessor
                    from src.ingestion.asset_store import LocalAssetStore

                    processor = VisualProcessor()
                    asset_store = LocalAssetStore()
                except Exception as e:
                    logger.error(f"Failed to load visual extraction dependencies: {e}")
                    extract_visuals = False

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Prefix with page number as a comment to maintain structure without polluting headings
                text_blocks.append(f"<!-- Page {page_num + 1} -->\n\n")

                page_described = False
                if extract_visuals and processor:
                    try:
                        # Rasterize full page
                        pix = page.get_pixmap(dpi=150)
                        if pix.n - pix.alpha > 3:
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        page_bytes = pix.tobytes("png")

                        desc = processor.describe_page(page_bytes)
                        if desc:
                            text_blocks.append(f"{desc}\n\n")
                            page_described = True
                    except Exception as e:
                        logger.error(
                            f"Failed full-page VLM extraction for page {page_num+1}: {e}"
                        )

                if not page_described:
                    # 1. Tables (if supported in PyMuPDF 1.23+)
                    if hasattr(page, "find_tables"):
                        try:
                            tables = page.find_tables()
                            for tab in tables:
                                df = tab.to_pandas()
                                markdown_table = df.to_markdown(index=False)
                                text_blocks.append(markdown_table + "\n\n")
                        except Exception as e:
                            logger.warning(
                                f"Table extraction failed on page {page_num+1}: {e}"
                            )

                    # 2. Text blocks — sort by vertical then horizontal position for reading order
                    blocks = page.get_text("blocks")
                    blocks.sort(key=lambda b: (round(b[1] / 20) * 20, b[0]))
                    for b in blocks:
                        if b[6] == 0:  # 0 = text block
                            text = b[4].strip()
                            if not text:
                                continue
                            # Only treat as heading if it's short, single-line, and doesn't end with punctuation
                            is_heading = (
                                len(text) < 80
                                and "\n" not in text
                                and not text[-1] in ".,;:?!"
                            )
                            if is_heading:
                                text_blocks.append(f"### {text}\n\n")
                            else:
                                text_blocks.append(f"{text}\n\n")

                # 3. Images (still extract individual images for asset store / visual chunks)
                if extract_visuals and processor and asset_store:
                    images = page.get_images()
                    for img_index, img_info in enumerate(images):
                        xref = img_info[0]
                        try:
                            pix = fitz.Pixmap(doc, xref)
                            if pix.n - pix.alpha > 3:  # convert CMYK etc to RGB
                                pix = fitz.Pixmap(fitz.csRGB, pix)

                            img_data = pix.tobytes("png")

                            asset_ref = asset_store.save(
                                doc_id, img_data, f"img_p{page_num+1}_{img_index}.png"
                            )
                            desc = processor.describe_image(img_data)

                            visual_chunks.append(
                                VisualChunkDraft(
                                    text=desc,
                                    asset_ref=asset_ref,
                                    asset_type="photo",  # Could try to infer type, but generic is fine for now
                                    page_number=page_num + 1,
                                )
                            )

                            # Add a reference in the text
                            text_blocks.append(
                                f"\n\n*[Visual asset on Page {page_num + 1}]*\n\n"
                            )
                        except Exception as e:
                            logger.error(
                                f"Failed to process image on page {page_num+1}: {e}"
                            )

            markdown_content = "".join(text_blocks)
            word_count = len(markdown_content.split())

            title = Path(source).name

            crawled_doc = CrawledDocument(
                url=f"file:///{source}",
                title=title,
                markdown_content=markdown_content,
                crawl_depth=0,
                crawled_at=datetime.utcnow(),
                word_count=word_count,
                status_code=200,
            )

            return AdapterResult(documents=[crawled_doc], visual_chunks=visual_chunks)

        except Exception as e:
            logger.error(f"Failed to read PDF {source}: {e}")
            return AdapterResult(documents=[], visual_chunks=[])
        finally:
            if _temp_file:
                import os

                try:
                    os.unlink(_temp_file.name)
                except Exception as cleanup_error:
                    logger.warning(
                        f"Failed to clean up temp file {_temp_file.name}: {cleanup_error}"
                    )
