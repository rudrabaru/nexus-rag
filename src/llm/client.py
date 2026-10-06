"""
LLM access via LiteLLM.

Uses litellm as a unified SDK over provider APIs (auth, request formatting, response parsing, cost
lookup) but implements retry and fallback ourselves rather than through litellm.Router. As of
2026-09-22 several Router-level mid-stream-fallback bugs are open (github.com/BerriAI/litellm issues
#28216, #40404), so streaming never uses Router machinery here; both call_llm and call_llm_stream
call litellm.completion/acompletion directly. See docs/phases/phase9_production_tradeoffs.md.

Both paths classify a failure the same way (src/llm/errors.py): a transient error is retried with
backoff, anything else goes straight to the fallback model, and when there is no fallback (or it
fails too) the call raises GenerationError. A call never returns failure text as if it were an answer.

A client holds no per-call state. One client serves every concurrent request, so each call reports
its usage, cost and serving model in its own LLMCall.
"""
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

import litellm

from src.llm.config import LLMConfig
from src.llm.errors import EmptyResponseError, GenerationError, is_permanent, is_retryable
from src.retry import backoff_seconds

logger = logging.getLogger(__name__)

MAX_RETRIES = 3  # attempts after the first: 1 s, 2 s, 4 s of backoff before giving up on a model


@dataclass
class LLMCall:
    """What one call produced, which model actually answered (after a fallback, not the configured one) and its cost."""

    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cost_known: bool = True  # False when litellm has no price for the model: cost_usd is then 0, not free
    provider: str = ""
    model: str = ""


def _cost(model: str, prompt: str, completion: str) -> Optional[float]:
    """
    Best-effort cost lookup; None when litellm cannot price the model. It never raises: a model litellm
    has not priced yet must not break generation. Prompt and completion text are passed explicitly
    rather than a response object, which mis-resolved the provider prefix of multi-segment model ids
    (groq/openai/gpt-oss-20b) in testing.
    """
    try:
        return litellm.completion_cost(model=model, prompt=prompt, completion=completion or "")
    except Exception as e:
        logger.debug(f"Cost lookup unavailable for {model}: {e}")
        return None


def _record_cost(call: LLMCall, model: str, prompt: str) -> None:
    cost = _cost(model, prompt, call.text)
    call.cost_known = cost is not None
    call.cost_usd = cost or 0.0


class LLMClient:
    """Calls one configured model, with retry-then-fallback to config.fallback_config."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._fallback_client: Optional["LLMClient"] = None
        if self.config.fallback_config:
            self._fallback_client = LLMClient(LLMConfig(**self.config.fallback_config))

    def _served_here(self, call: LLMCall) -> LLMCall:
        call.provider, call.model = self.config.provider, self.config.model_name
        return call

    def _request(self, prompt: str, **extra) -> dict:
        return dict(
            model=self.config.model_string,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
            timeout=self.config.request_timeout_seconds,
            num_retries=0,  # we own retry/backoff
            **extra,
        )

    def _fallback_from(self, error: Exception, is_fallback: bool) -> Optional["LLMClient"]:
        """The fallback client, when this model is the primary and has one; logs why it is being used."""
        if is_fallback or not self._fallback_client:
            return None
        logger.warning(
            f"{self.config.model_string} failed ({type(error).__name__}: {error}). "
            f"Falling back to {self._fallback_client.config.model_string}..."
        )
        return self._fallback_client

    def call_llm(self, prompt: str, is_fallback: bool = False, response_schema=None, max_retries: int = MAX_RETRIES) -> LLMCall:
        """The model's answer. Raises GenerationError once retries and the fallback are spent."""
        model = self.config.model_string
        extra = {"response_format": response_schema} if response_schema else {}
        last_error: Exception = GenerationError("no attempt was made")
        for attempt in range(max_retries + 1):
            try:
                response = litellm.completion(**self._request(prompt, **extra))
                content = response.choices[0].message.content
                if not content:
                    raise EmptyResponseError(f"empty content, finish_reason={response.choices[0].finish_reason}")
                call = self._served_here(LLMCall(
                    text=content,
                    prompt_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
                ))
                _record_cost(call, model, prompt)
                return call
            except Exception as e:
                last_error = e
                if not is_retryable(e) or attempt == max_retries:
                    logger.warning(f"{model} not retried further ({type(e).__name__}): {e}")
                    break
                delay = backoff_seconds(attempt, jitter=True)
                logger.warning(f"{model} transient error ({type(e).__name__}: {e}). Retrying in {delay:.2f}s ({attempt + 1}/{max_retries})...")
                time.sleep(delay)

        fallback = self._fallback_from(last_error, is_fallback)
        if fallback:
            return fallback.call_llm(prompt, is_fallback=True, response_schema=response_schema)
        logger.error(f"{model} failed: {type(last_error).__name__}: {last_error}")
        raise GenerationError(f"{type(last_error).__name__}: {last_error}", is_permanent(last_error)) from last_error

    async def call_llm_stream(self, prompt: str, call: LLMCall, is_fallback: bool = False, max_retries: int = MAX_RETRIES):
        """
        Yields answer text chunks and fills `call`, which the caller owns. Falls back only before the
        first token is yielded: a client mid-stream cannot be handed a second, unrelated answer, so a
        failure after the first token raises GenerationError.
        """
        model = self.config.model_string
        last_error: Exception = GenerationError("no attempt was made")
        for attempt in range(max_retries + 1):
            yielded_any = False
            try:
                response = await litellm.acompletion(
                    **self._request(prompt, stream=True, stream_options={"include_usage": True})
                )
                self._served_here(call)
                async for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content
                    if delta:
                        call.text += delta
                        yielded_any = True
                        yield delta
                    usage = getattr(chunk, "usage", None)
                    if usage:
                        call.prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                        call.completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                if not yielded_any:
                    raise EmptyResponseError("the stream ended without any text")
                _record_cost(call, model, prompt)
                return
            except Exception as e:
                last_error = e
                if yielded_any:
                    break
                if not is_retryable(e) or attempt == max_retries:
                    logger.warning(f"{model} stream not retried further ({type(e).__name__}): {e}")
                    break
                delay = backoff_seconds(attempt, jitter=True)
                logger.warning(f"{model} transient error in stream ({type(e).__name__}: {e}). Retrying in {delay:.2f}s ({attempt + 1}/{max_retries})...")
                await asyncio.sleep(delay)

        fallback = None if call.text else self._fallback_from(last_error, is_fallback)
        if fallback:
            async for piece in fallback.call_llm_stream(prompt, call, is_fallback=True):
                yield piece
            return
        self._served_here(call)
        logger.error(f"{model} stream failed: {type(last_error).__name__}: {last_error}")
        raise GenerationError(f"{type(last_error).__name__}: {last_error}", is_permanent(last_error)) from last_error
