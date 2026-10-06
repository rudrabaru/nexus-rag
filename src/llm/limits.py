"""
What each provider's free tier lets a project do, in one place, so an experiment can be sized
before it is started instead of failing hours in.

These are measured or documented values on a date, not guarantees: providers change them without
notice, and Gemini publishes none (its numbers are read from the error a spent quota returns).
Check the provider's console before depending on one. A limit left out (None) is unknown or absent.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Limits:
    requests_per_minute: Optional[int] = None
    tokens_per_minute: Optional[int] = None
    requests_per_day: Optional[int] = None
    tokens_per_day: Optional[int] = None
    source: str = ""


# Answer and judge models, by provider. Gemini's daily cap is per model.
LLM_LIMITS = {
    "groq": Limits(30, 8_000, 1_000, 200_000, "Groq's rate-limit page for gpt-oss models, read 2026-10-02"),
    "gemini": Limits(requests_per_day=20, source="measured 2026-10-06: quota GenerateRequestsPerDayPerProjectPerModel-FreeTier = 20"),
}

# Cloudflare Workers AI: 10,000 neurons a day free, bge-m3 at 1,075 neurons per million input tokens.
CLOUDFLARE_TOKENS_PER_DAY = int(10_000 / 1_075 * 1_000_000)


def llm_limits(provider: str) -> Limits:
    return LLM_LIMITS.get(provider, Limits(source="no limit known"))
