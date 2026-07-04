import logging
from typing import Optional
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

    VISION_PROMPT = """
    Analyze this image extracted from a document. Produce a detailed text 
    description covering:
    1. Visual type (chart, diagram, flowchart, table, screenshot, photo)
    2. Main subject or title if visible
    3. For charts: axes labels, data series, key numerical values, trends
    4. For flowcharts/diagrams: step-by-step flow including decision points
    5. For tables: transcribe content in markdown pipe table format
    6. All visible labels, legends, annotations, and numerical values
    Be exhaustive. Do not omit numbers or labels.
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
        config = GenerationConfig(
            provider="gemini", model_name="gemini-2.0-flash-lite", temperature=0.0
        )
        self.llm = LLMClient(config)

    def describe_image(
        self, image_bytes: bytes, asset_type_hint: Optional[str] = None
    ) -> str:
        """
        Sends image_bytes to Gemini Vision (multimodal content).
        Returns text description.
        """
        try:
            from google.genai import types

            # Use PIL to load and verify image
            img = Image.open(io.BytesIO(image_bytes))

            # Convert to PIL Image for genai client
            # The genai SDK accepts PIL Image objects directly in `contents`

            response = self.llm._llm_client.models.generate_content(
                model=self.llm.config.model_name,
                contents=[self.VISION_PROMPT, img],
                config=types.GenerateContentConfig(
                    temperature=self.llm.config.temperature,
                ),
            )
            return response.text

        except Exception as e:
            logger.error(f"Failed to extract visual description: {e}")
            return f"[Failed to extract visual description: {e}]"

    def describe_page(self, image_bytes: bytes) -> str:
        """
        Sends an entire page image to Gemini Vision to extract as pristine Markdown.
        Limits tokens to 4096 to prevent oversized chunks.
        """
        try:
            from google.genai import types

            img = Image.open(io.BytesIO(image_bytes))

            response = self.llm._llm_client.models.generate_content(
                model=self.llm.config.model_name,
                contents=[self.PAGE_PROMPT, img],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=4096,
                ),
            )
            return response.text

        except Exception as e:
            logger.error(f"Failed to extract page content: {e}")
            return ""
