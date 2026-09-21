"""Public, configurable price table for backtest cost accounting.

These are ILLUSTRATIVE per-1k-token prices for the tier classes jev-route
routes between, close enough to public list prices to make the backtest
number meaningful. They are configuration, not law: deployments should set
their actual contract prices in the policy's `pricing:` block, which
overrides these.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: USD per 1k tokens, per tier. "frontier" is the always-expensive baseline
#: the backtest compares against.
DEFAULT_PRICES: dict[str, float] = {
    "local": 0.0,        # self-hosted: electricity, not invoices
    "cheap": 0.0003,
    "strong": 0.003,
    "frontier": 0.015,
}

#: Estimated output tokens per request by category (input tokens come from the
#: trace itself; outputs are modelled, not measured).
DEFAULT_OUTPUT_TOKENS = 400


def prices_from_policy(policy_raw: Mapping[str, Any] | None) -> dict[str, float]:
    out = dict(DEFAULT_PRICES)
    if policy_raw:
        for k, v in (policy_raw.get("pricing", {}) or {}).items():
            out[str(k)] = float(v)
    return out


def estimate_cost_usd(tier: str, input_tokens: int, output_tokens: int, prices: Mapping[str, float]) -> float:
    price = prices.get(tier, prices.get("strong", 0.003))
    return (input_tokens + output_tokens) / 1000.0 * price
