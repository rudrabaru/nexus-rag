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


# Errors that no later attempt can fix: the model does not exist, or the key is wrong or not allowed
# to use it. A timeout or a rate limit is not here: the same call can succeed once the provider recovers.
PERMANENT_ERRORS = (litellm.NotFoundError, litellm.AuthenticationError, litellm.PermissionDeniedError)


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
    return isinstance(error, (*TRANSIENT_ERRORS, EmptyResponseError))
