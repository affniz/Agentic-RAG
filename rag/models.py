"""Model prices and per-model request options, in one place.

Prices are list prices in USD per million tokens, used for cost estimates only. They change:
check https://platform.claude.com/docs/en/about-claude/pricing and update PRICES_CHECKED.
"""

from __future__ import annotations

from typing import Any, Iterable

PRICES_CHECKED = "2026-11-03"

# (input, output, cache read, cache write) per million tokens.
PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-sonnet-5-5": (2.00, 10.00, 0.10, 2.50),
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
}

# Models whose price goes up when one request's prompt is longer than a threshold:
# {model: (threshold in prompt tokens, prices above it)}. The prompt counts uncached input,
# cache reads and cache writes.
LONG_PROMPT_PRICES: dict[str, tuple[int, tuple[float, float, float, float]]] = {
    "claude-haiku-5-5": (100_000, (0.50, 2.50, 0.05, 0.625)),
}

# If a safety classifier declines, the API reruns the request on a fallback model. Only some
# models have a server-side fallback: Claude Haiku 5.5 has none, so it gets no `fallbacks` parameter.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = frozenset({"claude-sonnet-5-5", "claude-opus-5-5"})


def call_cost(model: str, input_tokens: int, output_tokens: int, cache_read: int = 0,
              cache_write: int = 0) -> float | None:
    """Cost of one API call in USD, or None for a model without a known price."""
    p = PRICES.get(model)
    if p is None:
        return None
    long = LONG_PROMPT_PRICES.get(model)
    if long and input_tokens + cache_read + cache_write > long[0]:
        p = long[1]
    return (input_tokens * p[0] + output_tokens * p[1] + cache_read * p[2] + cache_write * p[3]) / 1_000_000


def request_options(model: str, betas: Iterable[str] = ()) -> dict[str, Any]:
    """Extra keyword arguments for `client.beta.messages.*`: the given betas, plus the refusal
    fallback where the model supports one. Empty when there is nothing to add."""
    out = list(betas)
    opts: dict[str, Any] = {}
    if model in FALLBACK_MODELS:
        out.append(FALLBACK_BETA)
        opts["fallbacks"] = "default"
    if out:
        opts["betas"] = out
    return opts
