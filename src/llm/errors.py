"""How a failed model call is classified, once, for the streaming and the non-streaming path."""
import re

import litellm

# litellm raises its own exception hierarchy (all inheriting from the matching openai.*
# exception), whichever provider actually served the request.
TRANSIENT_ERRORS = (
    litellm.RateLimitError,
    litellm.InternalServerError,
    litellm.ServiceUnavailableError,
    litellm.APIConnectionError,
)


class EmptyResponseError(Exception):
    """HTTP 200 with no usable message content (a reasoning model can spend its whole budget thinking)."""


# Errors that no later attempt can fix: the model does not exist, or the key is wrong or not allowed
# to use it. A timeout or a rate limit is not here: the same call can succeed once the provider recovers.
PERMANENT_ERRORS = (litellm.NotFoundError, litellm.AuthenticationError, litellm.PermissionDeniedError)


# A rate limit that will not lift within a request's lifetime. Measured 2026-10-06: Gemini's free tier
# allows 20 requests a day per model (quota id GenerateRequestsPerDayPerProject...), and once spent it
# answers 429 with a retry delay of about 11 hours. Retrying for seconds cannot help, and a run that
# keeps retrying only burns time, so such an error goes straight to the fallback and ends long runs.
QUOTA_EXHAUSTED_WAIT_SECONDS = 600  # a limit that resets within a few minutes is an ordinary 429
_DAILY_QUOTA = re.compile(r"PerDay|per day|requests_per_day|tokens per day|\bTPD\b", re.IGNORECASE)
_RETRY_DELAY = re.compile(r'retryDelay\\*"?\s*:\s*\\*"?(\d+(?:\.\d+)?)s', re.IGNORECASE)


def is_quota_exhausted(error: Exception) -> bool:
    """Whether a rate limit is a spent daily quota (or any wait longer than QUOTA_EXHAUSTED_WAIT_SECONDS)."""
    if not isinstance(error, litellm.RateLimitError):
        return False
    text = str(error)
    if _DAILY_QUOTA.search(text):
        return True
    delay = _RETRY_DELAY.search(text)
    return bool(delay) and float(delay.group(1)) >= QUOTA_EXHAUSTED_WAIT_SECONDS


def is_permanent(error: Exception) -> bool:
    """Whether retrying the same model later cannot help soon: a dead model, a bad key, or a spent daily quota."""
    return isinstance(error, PERMANENT_ERRORS) or is_quota_exhausted(error)


class GenerationError(RuntimeError):
    """
    The model could not produce an answer after retries and any fallback. The message is for logs, not
    for callers. `permanent` is True when retrying later cannot help, so a long run should stop and say so
    instead of waiting for the provider to recover.
    """

    def __init__(self, message: str, permanent: bool = False):
        super().__init__(message)
        self.permanent = permanent


def is_retryable(error: Exception) -> bool:
    """
    Whether trying the same model again can succeed. A transient provider error or an empty body
    can. Everything else cannot, and goes straight to the fallback:
    - a dead or misconfigured model (not found, bad request, bad key): an identical request fails the same way;
    - a timeout: the call's whole time budget is spent, and retrying a deployment that just hung
      would multiply the user's wait (retries x request_timeout_seconds).
    """
    if is_quota_exhausted(error):
        return False
    return isinstance(error, (*TRANSIENT_ERRORS, EmptyResponseError))
