import logging
import time
import random
from src.generating.models import GenerationConfig
from src.generating.providers.google_provider import GoogleProvider
from src.generating.providers.groq_provider import GroqProvider

logger = logging.getLogger(__name__)

class LLMClient:
    """
    LLM Client factory and wrapper.
    Handles initialization, calling, streaming, and fallback for 'gemini' and 'groq'.
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        self._provider_instance = self._init_provider(config)
        self._fallback_client = None
        if getattr(self.config, "fallback_config", None):
            fallback_cfg = GenerationConfig(**self.config.fallback_config)
            self._fallback_client = LLMClient(fallback_cfg)

    def _init_provider(self, config: GenerationConfig):
        provider = config.provider.lower()
        if provider == "gemini":
            return GoogleProvider(config)
        elif provider == "groq":
            return GroqProvider(config)
        else:
            raise ValueError(f"Unknown provider: '{provider}'. Supported: 'gemini', 'groq'.")

    def call_llm(
        self, prompt: str, is_fallback: bool = False, response_schema=None, max_retries: int = 3
    ) -> tuple[str, str, int, int]:
        """Call the LLM and return (answer, raw_response, prompt_tokens, completion_tokens)."""
        for attempt in range(max_retries + 1):
            try:
                return self._provider_instance.call(prompt, response_schema)
            except Exception as e:
                error_str = str(e)
                is_rate_limit = any(
                    err in error_str
                    for err in [
                        "429", "503", "ResourceExhausted", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "Too Many Requests",
                    ]
                )
                
                if is_rate_limit:
                    if attempt < max_retries:
                        sleep_time = (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"Rate limit hit ({e}). Retrying in {sleep_time:.2f}s (Attempt {attempt + 1}/{max_retries})...")
                        # NOTE: This time.sleep() runs in a thread pool via asyncio.to_thread() in generator.py.
                        # It will not hard-block the async event loop, but it will consume a worker thread slot 
                        # for the duration of the sleep. On constrained environments (e.g. Render Free Tier),
                        # this may reduce concurrent throughput during rate limits.
                        time.sleep(sleep_time)
                        continue
                    elif not is_fallback and self._fallback_client:
                        logger.warning(
                            f"Primary LLM failed after retries ({e}). Falling back to {self.config.fallback_config.get('provider')}..."
                        )
                        return self._fallback_client.call_llm(prompt, is_fallback=True, response_schema=response_schema)
                
                error_msg = f"[Generation failed: {type(e).__name__}: {e}]"
                logger.error(error_msg)
                return error_msg, error_msg, 0, 0

    async def call_llm_stream(self, prompt: str, is_fallback: bool = False, max_retries: int = 3):
        import asyncio
        for attempt in range(max_retries + 1):
            yielded_any = False
            try:
                async for chunk in self._provider_instance.call_stream(prompt):
                    yield chunk
                    yielded_any = True
                return
            except Exception as e:
                error_str = str(e)
                is_rate_limit = any(
                    err in error_str
                    for err in [
                        "429", "503", "ResourceExhausted", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "Too Many Requests",
                    ]
                )
                
                if is_rate_limit and not yielded_any:
                    if attempt < max_retries:
                        sleep_time = (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"Rate limit hit in stream ({e}). Retrying in {sleep_time:.2f}s (Attempt {attempt + 1}/{max_retries})...")
                        await asyncio.sleep(sleep_time)
                        continue
                    elif not is_fallback and self._fallback_client:
                        logger.warning(
                            f"Primary LLM stream failed after retries ({e}). Falling back to {self.config.fallback_config.get('provider')}..."
                        )
                        async for c in self._fallback_client.call_llm_stream(prompt, is_fallback=True):
                            yield c
                        return
                
                error_msg = f"[Generation failed: {type(e).__name__}: {e}]"
                logger.error(error_msg)
                yield error_msg
                return
