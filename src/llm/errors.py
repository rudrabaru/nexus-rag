"""How a failed model call is classified, once, for the streaming and the non-streaming path."""
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


class GenerationError(RuntimeError):
    """The model could not produce an answer after retries and any fallback. The message is for logs, not for callers."""


def is_retryable(error: Exception) -> bool:
    """
    Whether trying the same model again can succeed. A transient provider error or an empty body
    can. Everything else cannot, and goes straight to the fallback:
    - a dead or misconfigured model (not found, bad request, bad key): an identical request fails the same way;
    - a timeout: the call's whole time budget is spent, and retrying a deployment that just hung
      would multiply the user's wait (retries x request_timeout_seconds).
    """
    return isinstance(error, (*TRANSIENT_ERRORS, EmptyResponseError))
