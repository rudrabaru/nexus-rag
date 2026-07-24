import os
import logging
import asyncio
from typing import Tuple, AsyncGenerator
from src.generating.providers.base import BaseProvider
from src.generating.models import GenerationConfig

logger = logging.getLogger(__name__)

class GroqProvider(BaseProvider):
    def _init_client(self):
        try:
            from openai import OpenAI
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise EnvironmentError("GROQ_API_KEY environment variable not set.")
            client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
            logger.info(f"Groq client initialized: {self.config.model_name}")
            return client
        except ImportError:
            raise ImportError("openai is not installed. Run: pip install openai")

    def call(self, prompt: str, response_schema=None) -> Tuple[str, str, int, int]:
        kwargs = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
        }
        if response_schema:
            kwargs["response_format"] = {"type": "json_object"}
            
        response = self.client.chat.completions.create(**kwargs)
        raw_text = response.choices[0].message.content
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        return raw_text, raw_text, prompt_tokens, completion_tokens

    async def call_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.config.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
            stream=True,
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
            content = chunk.choices[0].delta.content
            if content:
                yield content
