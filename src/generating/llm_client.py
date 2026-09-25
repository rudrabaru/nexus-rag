"""
LLM access via LiteLLM.

Uses litellm as a unified SDK over provider APIs (auth, request formatting, response
parsing, cost lookup) but implements retry and fallback ourselves rather than through
litellm.Router. As of 2026-09-22 several Router-level mid-stream-fallback bugs are open
(github.com/BerriAI/litellm issues #28216, #40404), so streaming never uses Router
machinery here; both call_llm and call_llm_stream call litellm.completion/acompletion
directly. See docs/phases/phase9_production_tradeoffs.md for the full rationale.

Error handling is deliberately more precise than a single "is this a rate limit"
check: a dead/misconfigured model (NotFoundError, BadRequestError, ...) is retried
zero times locally before falling back, because retrying an identical request against
a model that does not exist cannot succeed. A transient error (RateLimitError,
InternalServerError, ServiceUnavailableError, APIConnectionError) is retried with backoff
first. A Timeout (every call is bounded by GenerationConfig.request_timeout_seconds) falls
back immediately: the call's time budget is already spent, and retrying a deployment that
just hung multiplies the wait. An empty response body on a 200 (observed live: Gemini 3.5's
"thinking" mode can consume the entire token budget on internal reasoning and return
no visible text, surfacing as finish_reason="length" with content=None) is treated the
same as a transient error rather than silently passed on as an answer.
"""
import logging
import time
import random
import asyncio
from typing import Optional

import litellm

from src.generating.models import GenerationConfig

logger = logging.getLogger(__name__)

# litellm raises its own exception hierarchy (all inheriting from the matching openai.*
# exception), regardless of which provider actually served the request.
_TRANSIENT_ERRORS = (
    litellm.RateLimitError,
    litellm.InternalServerError,
    litellm.ServiceUnavailableError,
    litellm.APIConnectionError,
)
# A timeout has already spent the call's whole time budget. Retrying the same deployment
# would multiply the user's wait (3 retries x request_timeout_seconds) against a provider
# that just hung, so a timeout goes straight to the fallback.
_FAIL_OVER_NOW_ERRORS = (litellm.Timeout,)
_NO_LOCAL_RETRY_ERRORS = (
    litellm.NotFoundError,
    litellm.BadRequestError,
    litellm.AuthenticationError,
    litellm.PermissionDeniedError,
    litellm.ContentPolicyViolationError,
)


class EmptyResponseError(Exception):
    """Raised when a provider returns HTTP 200 with no usable message content."""


def _model_string(config: GenerationConfig) -> str:
    return f"{config.provider}/{config.model_name}"


def _safe_cost(model: str, prompt: str, completion: str) -> float:
    """
    Best-effort cost lookup. Never raises: a model litellm has not priced yet (observed
    live for some newly-listed Groq models under certain lookup paths) must not break
    generation. Pass prompt/completion text explicitly rather than a full response
    object — passing a completion_response through this path mis-resolved the model's
    provider prefix for at least one real multi-segment model id during testing
    (groq/openai/gpt-oss-20b) and raised "model not mapped" even though the same model
    string resolves correctly when passed directly.
    """
    try:
        return litellm.completion_cost(model=model, prompt=prompt, completion=completion or "")
    except Exception as e:
        logger.debug(f"Cost lookup unavailable for {model}: {e}")
        return 0.0


class LLMClient:
    """Calls one configured model, with retry-then-fallback to config.fallback_config."""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_cost_usd = 0.0
        # Which model actually answered — differs from self.config after a fallback.
        self.last_served_provider = config.provider
        self.last_served_model = config.model_name
        self._fallback_client: Optional["LLMClient"] = None
        if getattr(self.config, "fallback_config", None):
            self._fallback_client = LLMClient(GenerationConfig(**self.config.fallback_config))

    def call_llm(
        self, prompt: str, is_fallback: bool = False, response_schema=None, max_retries: int = 3
    ) -> tuple[str, str, int, int]:
        """Returns (answer, raw_response, prompt_tokens, completion_tokens)."""
        model = _model_string(self.config)
        kwargs = dict(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
            timeout=self.config.request_timeout_seconds,
            num_retries=0,  # we own retry/backoff below
        )
        if response_schema:
            kwargs["response_format"] = response_schema

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = litellm.completion(**kwargs)
                content = response.choices[0].message.content
                if not content:
                    raise EmptyResponseError(
                        f"empty content, finish_reason={response.choices[0].finish_reason}"
                    )
                usage = response.usage
                self.last_prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                self.last_completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                self.last_cost_usd = _safe_cost(model, prompt, content)
                self.last_served_provider = self.config.provider
                self.last_served_model = self.config.model_name
                return content, content, self.last_prompt_tokens, self.last_completion_tokens
            except _NO_LOCAL_RETRY_ERRORS as e:
                last_error = e
                logger.warning(f"{model} rejected the request ({type(e).__name__}); not retrying locally: {e}")
                break
            except _FAIL_OVER_NOW_ERRORS as e:
                last_error = e
                logger.warning(f"{model} timed out after {self.config.request_timeout_seconds:.0f}s; not retrying locally.")
                break
            except (*_TRANSIENT_ERRORS, EmptyResponseError) as e:
                last_error = e
                if attempt < max_retries:
                    sleep_time = (2**attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"{model} transient error ({type(e).__name__}: {e}). "
                        f"Retrying in {sleep_time:.2f}s (attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(sleep_time)
                    continue
                break
            except Exception as e:
                last_error = e
                logger.error(f"{model} unexpected error, not retrying: {type(e).__name__}: {e}")
                break

        if not is_fallback and self._fallback_client:
            logger.warning(
                f"{model} failed ({type(last_error).__name__}: {last_error}). "
                f"Falling back to {_model_string(self._fallback_client.config)}..."
            )
            result = self._fallback_client.call_llm(prompt, is_fallback=True, response_schema=response_schema)
            self.last_served_provider = self._fallback_client.last_served_provider
            self.last_served_model = self._fallback_client.last_served_model
            self.last_cost_usd = self._fallback_client.last_cost_usd
            return result

        error_msg = f"[Generation failed: {type(last_error).__name__}: {last_error}]"
        logger.error(error_msg)
        return error_msg, error_msg, 0, 0

    async def call_llm_stream(self, prompt: str, is_fallback: bool = False, max_retries: int = 3):
        """
        Yields answer text chunks. Falls back only before the first token is yielded —
        a client mid-stream cannot be handed a second, unrelated answer.
        """
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_cost_usd = 0.0
        model = _model_string(self.config)
        full_answer = ""

        for attempt in range(max_retries + 1):
            yielded_any = False
            last_error = None
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_output_tokens,
                    stream=True,
                    stream_options={"include_usage": True},
                    timeout=self.config.request_timeout_seconds,
                    num_retries=0,
                )
                async for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content
                    if delta:
                        full_answer += delta
                        yielded_any = True
                        yield delta
                    usage = getattr(chunk, "usage", None)
                    if usage:
                        self.last_prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                        self.last_completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                self.last_cost_usd = _safe_cost(model, prompt, full_answer)
                self.last_served_provider = self.config.provider
                self.last_served_model = self.config.model_name
                return
            except Exception as e:
                last_error = e
                is_transient = isinstance(e, (*_TRANSIENT_ERRORS, EmptyResponseError))
                fail_over_now = isinstance(e, _FAIL_OVER_NOW_ERRORS)

                if (is_transient or fail_over_now) and not yielded_any:
                    if is_transient and attempt < max_retries:
                        sleep_time = (2**attempt) + random.uniform(0, 1)
                        logger.warning(
                            f"{model} transient error in stream ({type(e).__name__}: {e}). "
                            f"Retrying in {sleep_time:.2f}s (attempt {attempt + 1}/{max_retries})..."
                        )
                        await asyncio.sleep(sleep_time)
                        continue
                    if not is_fallback and self._fallback_client:
                        logger.warning(
                            f"{model} stream failed before any token ({type(e).__name__}: {e}). "
                            f"Falling back to {_model_string(self._fallback_client.config)}..."
                        )
                        async for c in self._fallback_client.call_llm_stream(prompt, is_fallback=True):
                            yield c
                        self.last_prompt_tokens = self._fallback_client.last_prompt_tokens
                        self.last_completion_tokens = self._fallback_client.last_completion_tokens
                        self.last_cost_usd = self._fallback_client.last_cost_usd
                        self.last_served_provider = self._fallback_client.last_served_provider
                        self.last_served_model = self._fallback_client.last_served_model
                        return

                error_msg = f"[Generation failed: {type(e).__name__}: {e}]"
                logger.error(error_msg)
                yield error_msg
                return

        # Unreachable when max_retries >= 0, kept only for static analysis clarity.
        error_msg = f"[Generation failed: {type(last_error).__name__}: {last_error}]"
        logger.error(error_msg)
        yield error_msg
