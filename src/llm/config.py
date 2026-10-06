from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class LLMConfig(BaseModel):
    """What one model call needs: which model, how it samples, and what to do when it fails."""

    model_config = ConfigDict(validate_assignment=True)

    provider: str = Field("gemini", description="LiteLLM provider prefix: 'gemini', 'groq' or 'openai'")
    model_name: str = Field("gemini-3.5-flash", description="Model identifier passed to the provider API")
    max_output_tokens: int = Field(4096, description="Maximum tokens the model may generate in its response")
    temperature: float = Field(
        0.1,
        description=(
            "Sampling temperature. Low values (0.0-0.2) keep the model close to the retrieved "
            "context; judges run at 0.0."
        ),
    )
    request_timeout_seconds: float = Field(
        60.0,
        description=(
            "Upper bound on one call. litellm's default is 6000 s, so without this a provider that "
            "hangs holds a query slot for up to 100 minutes and the fallback never fires. 60 s covers a "
            "full answer (4096 tokens at the ~100 tokens/s of flash-class models is ~40 s) with headroom."
        ),
    )
    fallback_config: Optional[dict] = Field(None, description="LLMConfig fields of the model to use when this one fails")

    @property
    def model_string(self) -> str:
        return f"{self.provider}/{self.model_name}"


def parse_model(spec: str) -> Tuple[str, str]:
    """'groq/openai/gpt-oss-20b' -> ('groq', 'openai/gpt-oss-20b'): the provider is everything before the first slash."""
    provider, _, model = spec.strip().partition("/")
    if not provider or not model:
        raise ValueError(f"Expected 'provider/model' (for example gemini/gemini-3.5-flash), got {spec!r}.")
    return provider.lower(), model
