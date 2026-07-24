import os
import logging
import asyncio
from typing import Tuple, AsyncGenerator
from src.generating.providers.base import BaseProvider

logger = logging.getLogger(__name__)

class GoogleProvider(BaseProvider):
    def _init_client(self):
        try:
            from google import genai
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise EnvironmentError("GEMINI_API_KEY environment variable not set.")
            client = genai.Client(api_key=api_key)
            logger.info(f"Gemini client initialized: {self.config.model_name}")
            return client
        except ImportError:
            raise ImportError("google-genai is not installed. Run: pip install google-genai")

    def call(self, prompt: str, response_schema=None) -> Tuple[str, str, int, int]:
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
        )
        if response_schema:
            config.response_mime_type = "application/json"
            config.response_schema = response_schema

        response = self.client.models.generate_content(
            model=self.config.model_name,
            contents=prompt,
            config=config,
        )
        raw_text = response.text
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
        return raw_text, raw_text, prompt_tokens, completion_tokens

    async def call_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        from google.genai import types

        response = await asyncio.to_thread(
            self.client.models.generate_content_stream,
            model=self.config.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
            ),
        )
        
        def get_next(iterator):
            try:
                return next(iterator)
            except StopIteration:
                return None
                
        while True:
            chunk = await asyncio.to_thread(get_next, response)
            if chunk is None:
                break
            yield chunk.text
