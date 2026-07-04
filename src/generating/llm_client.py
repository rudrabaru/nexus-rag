import logging
import os
from src.generating.models import GenerationConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """
    LLM Client factory and wrapper.
    Handles initialization, calling, streaming, and fallback for 'gemini' and 'groq'.
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        self._llm_client = self._init_llm_client()
        self._fallback_client = None

    def _init_llm_client(self):
        """Initialize and return the LLM SDK client for the configured provider."""
        provider = self.config.provider.lower()

        if provider == "gemini":
            try:
                from google import genai

                api_key = os.environ.get("GEMINI_API_KEY")
                if not api_key:
                    raise EnvironmentError(
                        "GEMINI_API_KEY environment variable not set. "
                        "Set it before running the generation phase."
                    )
                client = genai.Client(api_key=api_key)
                logger.info(f"Gemini client initialized: {self.config.model_name}")
                return client
            except ImportError:
                raise ImportError(
                    "google-genai is not installed. Run: pip install google-genai"
                )

        elif provider == "groq":
            try:
                from openai import OpenAI

                api_key = os.environ.get("GROQ_API_KEY")
                if not api_key:
                    raise EnvironmentError("GROQ_API_KEY environment variable not set.")
                client = OpenAI(
                    api_key=api_key, base_url="https://api.groq.com/openai/v1"
                )
                logger.info(f"Groq client initialized: {self.config.model_name}")
                return client
            except ImportError:
                raise ImportError("openai is not installed. Run: pip install openai")

        else:
            raise ValueError(
                f"Unknown provider: '{provider}'. Supported: 'gemini', 'groq'."
            )

    def call_llm(
        self, prompt: str, is_fallback: bool = False
    ) -> tuple[str, str, int, int]:
        """Call the LLM and return (answer, raw_response, prompt_tokens, completion_tokens)."""
        provider = self.config.provider.lower()

        try:
            if provider == "gemini":
                from google.genai import types

                response = self._llm_client.models.generate_content(
                    model=self.config.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.config.temperature,
                        max_output_tokens=self.config.max_output_tokens,
                    ),
                )
                raw_text = response.text
                usage = getattr(response, "usage_metadata", None)
                prompt_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
                completion_tokens = (
                    getattr(usage, "candidates_token_count", 0) if usage else 0
                )
                return raw_text, raw_text, prompt_tokens, completion_tokens

            elif provider == "groq":
                response = self._llm_client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_output_tokens,
                )
                raw_text = response.choices[0].message.content
                usage = response.usage
                prompt_tokens = usage.prompt_tokens if usage else 0
                completion_tokens = usage.completion_tokens if usage else 0
                return raw_text, raw_text, prompt_tokens, completion_tokens

        except Exception as e:
            error_str = str(e)
            is_rate_limit = any(
                err in error_str
                for err in [
                    "429",
                    "503",
                    "ResourceExhausted",
                    "RESOURCE_EXHAUSTED",
                    "UNAVAILABLE",
                ]
            )
            if (
                not is_fallback
                and is_rate_limit
                and getattr(self.config, "fallback_config", None)
            ):
                logger.warning(
                    f"Primary LLM failed ({e}). Falling back to {self.config.fallback_config.get('provider')}..."
                )
                original_config = self.config
                original_client = self._llm_client
                try:
                    self.config = GenerationConfig(**self.config.fallback_config)
                    if not self._fallback_client:
                        self._fallback_client = self._init_llm_client()
                    self._llm_client = self._fallback_client
                    return self.call_llm(prompt, is_fallback=True)
                finally:
                    self.config = original_config
                    self._llm_client = original_client

            error_msg = f"[Generation failed: {type(e).__name__}: {e}]"
            logger.error(error_msg)
            return error_msg, error_msg, 0, 0

    def call_llm_stream(self, prompt: str, is_fallback: bool = False):
        provider = self.config.provider.lower()
        try:
            if provider == "gemini":
                from google.genai import types

                response = self._llm_client.models.generate_content_stream(
                    model=self.config.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.config.temperature,
                        max_output_tokens=self.config.max_output_tokens,
                    ),
                )
                for chunk in response:
                    yield chunk.text
            elif provider == "groq":
                response = self._llm_client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_output_tokens,
                    stream=True,
                )
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
        except Exception as e:
            error_str = str(e)
            is_rate_limit = any(
                err in error_str
                for err in [
                    "429",
                    "503",
                    "ResourceExhausted",
                    "RESOURCE_EXHAUSTED",
                    "UNAVAILABLE",
                ]
            )
            if (
                not is_fallback
                and is_rate_limit
                and getattr(self.config, "fallback_config", None)
            ):
                logger.warning(
                    f"Primary LLM stream failed ({e}). Falling back to {self.config.fallback_config.get('provider')}..."
                )
                original_config = self.config
                original_client = self._llm_client
                try:
                    self.config = GenerationConfig(**self.config.fallback_config)
                    if not self._fallback_client:
                        self._fallback_client = self._init_llm_client()
                    self._llm_client = self._fallback_client
                    yield from self.call_llm_stream(prompt, is_fallback=True)
                    return
                finally:
                    self.config = original_config
                    self._llm_client = original_client

            error_msg = f"[Generation failed: {type(e).__name__}: {e}]"
            logger.error(error_msg)
            yield error_msg
