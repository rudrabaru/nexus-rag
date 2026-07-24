import logging
from PIL import Image
import io

from src.generating.models import GenerationConfig
from src.generating.llm_client import LLMClient

logger = logging.getLogger(__name__)


class VisualProcessor:
    """
    Sends extracted images to Gemini Vision and returns structured text descriptions.
    Zero-cost within Gemini free tier.
    """

    PAGE_PROMPT = """
    You are a precise document extraction engine. Given a page image from a document, extract ALL content into clean Markdown. Rules:
    - Text paragraphs: copy as-is.
    - Headings: use # / ## / ### based on visual hierarchy.
    - Tables: always convert to pipe-format Markdown tables (| col | col |).
    - Code blocks: wrap in triple backticks with language hint.
    - Flowcharts / Diagrams: describe step-by-step in a numbered list, preserving arrows and decision branches.
    - Charts / Graphs: describe axes, labels, data series, and key values.
    - Images / Photos: write a concise descriptive caption.
    Output ONLY the Markdown. No commentary.
    """

    def __init__(self):
        import os
        model_name = os.environ.get("LLM_MODEL_NAME", "gemini-2.0-flash")
        config = GenerationConfig(
            provider="gemini", model_name=model_name, temperature=0.0
        )
        self.llm = LLMClient(config)

    def describe_page(self, image_bytes: bytes) -> str:
        """
        Sends an entire page image to Gemini Vision to extract as pristine Markdown.
        Limits tokens to 4096 to prevent oversized chunks.
        """
        try:
            from google.genai import types

            img = Image.open(io.BytesIO(image_bytes))
            kb = len(image_bytes) // 1024
            logger.info(
                f"VISION | describe_page | model={self.llm.config.model_name} "
                f"image={img.size[0]}x{img.size[1]}px {kb}KB"
            )

            import time
            import re
            
            max_retries = 4
            base_delay = 5.0
            
            for attempt in range(max_retries):
                try:
                    response = self.llm._provider_instance.client.models.generate_content(
                        model=self.llm.config.model_name,
                        contents=[self.PAGE_PROMPT, img],
                        config=types.GenerateContentConfig(
                            temperature=0.0,
                            max_output_tokens=4096,
                        ),
                    )
                    result_chars = len(response.text) if response.text else 0
                    logger.info(f"VISION | describe_page | OK → {result_chars} chars (attempt {attempt+1})")
                    return response.text or ""
                    
                except Exception as e:
                    err_msg = str(e)
                    is_rate_limit = "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg
                    
                    if attempt < max_retries - 1 and (is_rate_limit or "503" in err_msg):
                        # Try to parse the exact retry delay from the error message
                        # e.g., "Please retry in 8.175s."
                        match = re.search(r"Please retry in ([\d\.]+)s", err_msg)
                        if match:
                            delay = float(match.group(1)) + 1.0  # add 1s buffer
                        else:
                            delay = base_delay * (2 ** attempt)
                            
                        logger.warning(f"VISION | describe_page | API limit hit ({type(e).__name__}). Retrying in {delay:.1f}s... (attempt {attempt+1}/{max_retries})")
                        time.sleep(delay)
                    else:
                        logger.error(f"VISION | describe_page | FAILED after {attempt+1} attempts. {type(e).__name__}: {e!r}")
                        return ""
                        
            return ""

        except Exception as e:
            logger.error(f"VISION | describe_page | FATAL {type(e).__name__}: {e!r}")
            return ""

