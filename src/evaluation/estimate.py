"""
A pre-flight estimate of what an experiment will ask of each provider, against the limits that
bind it, before anything is created or run.

With free tiers the binding limit is rarely money: Gemini allows 20 requests a day, Groq 200K
tokens a day, Voyage 10K tokens a minute. An experiment that cannot finish within them would
otherwise fail partway through, hours in. Every figure here is an upper bound or a labelled
estimate:
- generation and judge calls assume no cache hits (a configuration that retrieves the same context
  as another is generated once, so real use is lower);
- the days figure assumes a full fresh daily allowance, because a provider's remaining quota cannot be read;
- token sizes per call are experiments (below), not measurements of this corpus.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src.config import Settings
from src.evaluation.spec import ExperimentSpec
from src.generating.models import GenerationConfig
from src.llm.config import parse_model
from src.llm.limits import CLOUDFLARE_TOKENS_PER_DAY, Limits, llm_limits
from src.llm.roles import role_model
from src.tokens import estimate_tokens

# Experiment values, not tuned: the prompt around the context (instructions, citation markers), a typical
# answer, and a judge prompt (context + answer + rubric). A retrieved chunk averages about 600 tokens.
PROMPT_OVERHEAD_TOKENS = 500
ANSWER_TOKENS = 400
JUDGE_OVERHEAD_TOKENS = 600
JUDGE_REPLY_TOKENS = 150
CHUNK_TOKENS = 600
DEFAULT_MAX_DAYS = 2


@dataclass
class Line:
    """One provider's share of the experiment."""

    label: str
    calls: int
    tokens: int
    limits: Limits
    largest_call_tokens: int = 0

    @property
    def minutes(self) -> float:
        by_rate = [self.calls / v for v in (self.limits.requests_per_minute,) if v]
        by_tokens = [self.tokens / v for v in (self.limits.tokens_per_minute,) if v]
        return max(by_rate + by_tokens, default=0.0)

    @property
    def days(self) -> int:
        by_day = [self.calls / v for v in (self.limits.requests_per_day,) if v]
        by_day += [self.tokens / v for v in (self.limits.tokens_per_day,) if v]
        return max(1, math.ceil(max(by_day, default=0.0)))

    @property
    def impossible(self) -> bool:
        """One call larger than a whole minute's allowance can never be admitted."""
        cap = self.limits.tokens_per_minute
        return bool(cap) and self.largest_call_tokens > cap


@dataclass
class Estimate:
    queries: int
    trials: int
    lines: List[Line] = field(default_factory=list)

    def problems(self, max_days: int = DEFAULT_MAX_DAYS) -> List[str]:
        found = []
        for line in self.lines:
            if line.impossible:
                found.append(f"{line.label}: one request (~{line.largest_call_tokens:,} tokens) is larger than the "
                             f"{line.limits.tokens_per_minute:,} tokens a minute allowed, so it can never be sent")
            elif line.days > max_days:
                found.append(f"{line.label}: needs about {line.days} days of its daily allowance "
                             f"({line.calls:,} calls, ~{line.tokens:,} tokens)")
        return found

    def render(self) -> str:
        out = [f"Estimate for {self.queries} queries x {self.trials} trials (upper bounds; no cache hits assumed, "
               "a full daily allowance assumed):"]
        for line in self.lines:
            when = f"{line.minutes:.0f} min" if line.minutes < 120 else f"{line.minutes / 60:.1f} h"
            out.append(f"  {line.label:<34} {line.calls:>6,} calls  ~{line.tokens:>10,} tokens  "
                       f">= {when} at the rate limit, {line.days} day(s) of allowance  [{line.limits.source or 'no limit known'}]")
        return "\n".join(out)


def _embedding_limits(provider: str, settings: Settings) -> Limits:
    if provider == "voyage":
        return Limits(settings.voyage_rpm, settings.voyage_tpm, source="measured card-free Voyage limit")
    if provider == "cloudflare":
        return Limits(3_000, tokens_per_day=CLOUDFLARE_TOKENS_PER_DAY, source="Cloudflare free plan, documented")
    return Limits(source="local or unknown: no limit")


def _add(demand: Dict[Tuple[str, Limits], List[int]], label: str, limits: Limits, calls: int, tokens: int, largest: int) -> None:
    total = demand.setdefault((label, limits), [0, 0, 0])
    total[0] += calls
    total[1] += tokens
    total[2] = max(total[2], largest)


def estimate(spec: ExperimentSpec, queries: List[str], settings: Settings) -> Estimate:
    n = len(queries)
    demand: Dict[Tuple[str, Limits], List[int]] = {}

    # Each query is embedded once per index, whatever the number of trials that use it.
    query_tokens = sum(estimate_tokens(q) for q in queries)
    indexes = {t.index_id.partition(":")[0] if t.index_id else settings.embedding_provider.lower()
               for t in spec.trials.values() if t.strategy != "sparse"}
    for provider in sorted(indexes):
        _add(demand, f"{provider} embeddings (queries)", _embedding_limits(provider, settings), n, query_tokens, query_tokens // max(n, 1))

    for label, trial in spec.trials.items():
        if trial.reranker == "voyage":
            per_call = trial.rerank_candidates * (CHUNK_TOKENS + query_tokens // max(n, 1))
            _add(demand, "voyage rerank", _embedding_limits("voyage", settings), n, n * per_call, per_call)

    if spec.generation:
        context = GenerationConfig().max_context_tokens
        model = spec.generation.model
        answerer = (model.provider, model.model_name) if model else parse_model(role_model(settings, "chat"))
        judge = (spec.generation.judge.provider, spec.generation.judge.model_name)
        calls = n * len(spec.trials)
        answer_call = context + PROMPT_OVERHEAD_TOKENS + ANSWER_TOKENS
        judge_call = context + ANSWER_TOKENS + JUDGE_OVERHEAD_TOKENS + JUDGE_REPLY_TOKENS
        # One model answering and judging shares one daily allowance, so both land on the same line.
        for (provider, name), per_call in ((answerer, answer_call), (judge, judge_call)):
            _add(demand, f"{provider}/{name}", llm_limits(provider), calls, calls * per_call, per_call)

    lines = [Line(label, c, t, limits, big) for (label, limits), (c, t, big) in demand.items()]
    return Estimate(queries=n, trials=len(spec.trials), lines=lines)
