"""Cost and latency simulation over measured routing decisions.

This module answers two questions a sceptical engineer will ask within thirty
seconds of reading the README, and it answers them with arithmetic rather than
adjectives:

**1. "What does the routing actually save?"** Four strategies are priced over the
same prompts: ``always_strong``, ``always_cheap``, ``jev_route`` (the measured
decisions), and ``gate_only`` (the local hard gate plus always-cheap -- no model
classification at all). The last one is the important one. A regex gate is free,
runs in microseconds, and catches structured identifiers with perfect precision;
if it were enough, this project would be a wrapper around ``re.compile``. It is
not enough, and the numbers here show exactly how it is not enough: it never
misroutes a prompt it can *see*, and it is blind to every prompt whose
sensitivity is a matter of meaning rather than shape.

**2. "What does the routing cost in latency?"** Percentiles of the measured Jev
decision latency, expressed against a representative completion latency so the
tax can be compared with the thing it is taxing.

Honesty constraints baked into the code, because a cost model is exactly where
plausible-looking fiction gets published:

* Every price in :data:`PRICE_TABLE` is **illustrative**. These are not quotes,
  not contract prices, and not scraped from a provider page at run time. They are
  order-of-magnitude numbers chosen so the *ratios* between strategies are
  meaningful. The ratio between a frontier model and a flash model is the actual
  claim; the dollar total is a consequence of an assumption and is labelled as
  such wherever it is printed.
* Token counts are a **character-count proxy**. We do not call the completion
  models in this evaluation -- the dataset is prompts without reference
  completions -- so there is no real ``prompt_tokens`` to read. ``CHARS_PER_TOKEN``
  is stated at the top of the file and every cost is reported per million
  *estimated* tokens.
* The output side is a **single global constant**. Because it is identical for
  every strategy, it cannot change the ranking between them; it only shifts the
  absolute totals. Both the input-only and the input+assumed-output totals are
  emitted so a reader who dislikes the constant can use the other one.
* The **Jev decision API cost is not included**. It is not published in this
  repo, and inventing it would be exactly the sin the bullets above exist to
  prevent. The number of decision calls is reported instead, so anyone with a
  price sheet can add the term themselves.

Stdlib only.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# --------------------------------------------------------------------------- #
# ASSUMPTIONS. Change these and every number below moves; the report prints them.
# --------------------------------------------------------------------------- #
#: ILLUSTRATIVE ONLY. Per-million-token prices, USD. Not a quote, not live data.
#: The ratio strong:cheap is the load-bearing part of the model (~10x on input,
#: which is the shape of every frontier-vs-flash price pair in the market); the
#: absolute values are placeholders chosen to be in the right order of magnitude.
#: ``qwen38`` is self-hosted, so its marginal API price is 0.0 by construction --
#: the real cost is GPU-hours you already own, which this model does not attempt
#: to price.
PRICE_TABLE: dict[str, dict[str, float]] = {
    "qwen3.8-max": {"input": 1.60, "output": 6.40},
    "qwen3.8-flash": {"input": 0.15, "output": 1.50},
    "qwen38": {"input": 0.0, "output": 0.0},
}

#: Tier -> concrete model, matching ``tiers:`` in policies/default.yaml. Kept here
#: as a default because the cost model must be runnable from a results file alone;
#: ``run_eval.py`` passes the mapping it actually loaded so the two cannot drift.
MODEL_FOR_TIER: dict[str, str] = {
    "local": "qwen38",
    "cheap": "qwen3.8-flash",
    "strong": "qwen3.8-max",
}

#: Character-to-token proxy. 4.0 is the conventional English approximation and is
#: wrong for code (denser) and for Hungarian/German (also denser). It is wrong by
#: the same factor for every strategy, so it cancels in comparisons.
CHARS_PER_TOKEN = 4.0

#: Assumed completion length, identical for every strategy. Deliberately a
#: constant rather than a function of the prompt: modelling output length from
#: prompt length would be a second invented relationship stacked on the first.
ASSUMED_OUTPUT_TOKENS = 256.0

#: A representative end-to-end completion latency in milliseconds, used only to
#: express the routing tax as a percentage of something a reader can picture.
#: ILLUSTRATIVE: real completion latency spans ~400 ms for a short flash call to
#: 30 s+ for a long frontier generation.
REPRESENTATIVE_COMPLETION_MS = 3000.0


# --------------------------------------------------------------------------- #
# Row shape
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CostRow:
    """One evaluated prompt, reduced to what a cost model is allowed to use.

    Deliberately text-free: length and tier. That keeps the cost simulation
    runnable from the persisted JSONL without re-reading prompts, and it makes
    the model's inputs auditable at a glance.
    """

    id: str
    chars: int
    #: Tier the strategy under test assigned.
    tier: str
    #: Ground truth from the dataset, so the same row can be scored and priced.
    expected_tier: str = ""
    #: Words, carried for a second token estimate to sanity-check the char proxy.
    words: int = 0


@dataclass(frozen=True)
class StrategyCost:
    """Priced outcome of one strategy over the whole set."""

    strategy: str
    n: int
    tier_counts: dict[str, int]
    tier_share: dict[str, float]
    input_tokens: float
    #: Cost with the assumed constant output length.
    total_cost: float
    #: Cost of the prompt alone. Preferred by anyone who distrusts
    #: ``ASSUMED_OUTPUT_TOKENS``; it preserves every ranking.
    input_only_cost: float
    output_tokens: float
    cost_per_request: float
    input_only_cost_per_request: float
    #: Number of completion calls that would hit a paid cloud model at all.
    cloud_calls: int
    #: Correctness of the strategy's tier assignment, reported next to its cost
    #: because a cheap strategy that is wrong is not cheap.
    tier_accuracy: float
    unsafe_errors: int
    assumptions: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "n": self.n,
            "tier_counts": dict(self.tier_counts),
            "tier_share": {k: round(v, 6) for k, v in self.tier_share.items()},
            "input_tokens": round(self.input_tokens, 3),
            "output_tokens": round(self.output_tokens, 3),
            "total_cost_usd": round(self.total_cost, 6),
            "input_only_cost_usd": round(self.input_only_cost, 6),
            "cost_per_request_usd": round(self.cost_per_request, 9),
            "input_only_cost_per_request_usd": round(self.input_only_cost_per_request, 9),
            "cloud_calls": self.cloud_calls,
            "tier_accuracy": round(self.tier_accuracy, 6),
            "unsafe_errors": self.unsafe_errors,
            "assumptions": dict(self.assumptions),
        }


# --------------------------------------------------------------------------- #
# Token and price arithmetic
# --------------------------------------------------------------------------- #
def estimate_input_tokens(chars: int, *, chars_per_token: float = CHARS_PER_TOKEN) -> float:
    """Character-count proxy for prompt tokens. See the module docstring."""
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    return max(0, int(chars)) / chars_per_token


def price_for(model: str, *, price_table: Mapping[str, Mapping[str, float]] = PRICE_TABLE) -> tuple[float, float]:
    """``(input, output)`` per-million-token prices for a model.

    An unknown model raises rather than defaulting to zero. Silently pricing a
    model at $0 would make any typo in the tier map look like a spectacular
    saving, which is the worst possible failure mode for a cost report.
    """
    entry = price_table.get(model)
    if entry is None:
        raise KeyError(
            f"no price for model {model!r}; known models: {sorted(price_table)}. "
            "Add it to PRICE_TABLE rather than letting it default to free."
        )
    return float(entry["input"]), float(entry["output"])


def call_cost(
    model: str,
    input_tokens: float,
    output_tokens: float,
    *,
    price_table: Mapping[str, Mapping[str, float]] = PRICE_TABLE,
) -> float:
    """USD cost of one completion call."""
    price_in, price_out = price_for(model, price_table=price_table)
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000.0


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #
#: A strategy maps one row to a tier. Keeping them as plain callables means the
#: negative control below can compare any two of them without special-casing.
Strategy = Callable[[CostRow], str]


def strategy_always(tier: str) -> Strategy:
    """Constant strategy: every request to ``tier``. The trivial bounds."""

    def _pick(_row: CostRow) -> str:
        return tier

    return _pick


def strategy_measured(row: CostRow) -> str:
    """The router's own decision, as recorded.

    ``jev_route`` and ``gate_only`` both use this: the difference between them is
    not in the strategy function but in *which run's* tiers the rows carry. A
    gate-only row's tier comes from a run where the gate was the only classifier.
    """
    return row.tier


STRATEGIES: dict[str, Strategy] = {
    "always_strong": strategy_always("strong"),
    "always_cheap": strategy_always("cheap"),
    "jev_route": strategy_measured,
    "gate_only": strategy_measured,
}


# --------------------------------------------------------------------------- #
# Error classification -- shared vocabulary with run_eval.py
# --------------------------------------------------------------------------- #
#: Ascending data-egress risk. ``local`` is the most restrictive tier: the data
#: never leaves the building. Anything to the right of the expected tier means
#: data moved further out than the label justified.
TIER_RISK_ORDER: tuple[str, ...] = ("local", "cheap", "strong")


def tier_risk(tier: str) -> int:
    """Position in :data:`TIER_RISK_ORDER`; unknown tiers sort to the riskiest end."""
    try:
        return TIER_RISK_ORDER.index(tier)
    except ValueError:
        return len(TIER_RISK_ORDER)


def classify_error(expected: str, predicted: str) -> str:
    """Bucket one tier assignment.

    The buckets are NOT interchangeable and must never be averaged into a single
    error rate. ``unsafe`` and ``expensive`` differ by orders of magnitude in
    consequence: one leaks regulated data to a third party, the other wastes a
    GPU you already own.

    ``unsafe``
        Labelled ``local``, routed to ``cheap``/``strong``. Data left the
        building. For a data-residency router this is the only number that
        matters, and it is reported as a headline of its own.
    ``expensive``
        Labelled ``cheap``/``strong``, routed to ``local``. Capability wasted on
        a self-hosted model. Costs quality and GPU time; leaks nothing.
    ``overspend``
        Labelled ``cheap``, routed to ``strong``. Pays frontier prices for flash
        work. No data risk, so it is not ``unsafe``; not a downgrade to local, so
        it is not ``expensive`` either. Reported separately rather than folded in.
    ``underpowered``
        Labelled ``strong``, routed to ``cheap``. A quality risk, not a data
        risk. Kept apart from ``unsafe`` on purpose: conflating "wrong answer"
        with "leaked data" is how privacy routers end up optimising the wrong one.
    """
    if expected == predicted:
        return "correct"
    if expected == "local":
        # Anything other than local, for a row whose data must stay in.
        return "unsafe"
    if predicted == "local":
        return "expensive"
    # Neither end is local, so this is a lateral capability mismatch.
    return "overspend" if tier_risk(predicted) > tier_risk(expected) else "underpowered"


ERROR_KINDS: tuple[str, ...] = ("correct", "unsafe", "expensive", "overspend", "underpowered")


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def _assumptions() -> dict[str, Any]:
    """The assumption block printed next to every cost figure."""
    return {
        "price_table_usd_per_million_tokens": {k: dict(v) for k, v in PRICE_TABLE.items()},
        "prices_are_illustrative": True,
        "chars_per_token": CHARS_PER_TOKEN,
        "token_counts_are_char_proxy": True,
        "assumed_output_tokens": ASSUMED_OUTPUT_TOKENS,
        "jev_decision_api_cost_included": False,
        "representative_completion_ms": REPRESENTATIVE_COMPLETION_MS,
    }


def simulate(
    strategy: str,
    rows: Sequence[CostRow],
    *,
    tier_for: Strategy | None = None,
    model_for_tier: Mapping[str, str] = MODEL_FOR_TIER,
    price_table: Mapping[str, Mapping[str, float]] = PRICE_TABLE,
    chars_per_token: float = CHARS_PER_TOKEN,
    assumed_output_tokens: float = ASSUMED_OUTPUT_TOKENS,
) -> StrategyCost:
    """Price one strategy over a fixed set of rows.

    The strategy callable decides the tier; everything else is arithmetic. Cost
    is summed per request because per-request pricing is what a bill looks like,
    and the mean is reported alongside the total so a 223-row total is never
    mistaken for a production-scale one.
    """
    pick: Strategy = tier_for if tier_for is not None else STRATEGIES[strategy]
    tier_counts: dict[str, int] = {}
    input_tokens = 0.0
    output_tokens = 0.0
    total_cost = 0.0
    input_only_cost = 0.0
    cloud_calls = 0
    correct = 0
    unsafe = 0

    for row in rows:
        tier = pick(row)
        model = model_for_tier.get(tier)
        if model is None:
            raise KeyError(
                f"strategy {strategy!r} produced tier {tier!r}, which is not in the tier->model "
                f"map {dict(model_for_tier)}"
            )
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        in_tok = estimate_input_tokens(row.chars, chars_per_token=chars_per_token)
        input_tokens += in_tok
        output_tokens += assumed_output_tokens
        total_cost += call_cost(model, in_tok, assumed_output_tokens, price_table=price_table)
        input_only_cost += call_cost(model, in_tok, 0.0, price_table=price_table)
        if price_for(model, price_table=price_table) != (0.0, 0.0):
            cloud_calls += 1
        if row.expected_tier:
            kind = classify_error(row.expected_tier, tier)
            correct += kind == "correct"
            unsafe += kind == "unsafe"

    n = len(rows)
    scored = sum(1 for r in rows if r.expected_tier)
    assumptions = _assumptions()
    assumptions["chars_per_token"] = chars_per_token
    assumptions["assumed_output_tokens"] = assumed_output_tokens

    return StrategyCost(
        strategy=strategy,
        n=n,
        tier_counts=tier_counts,
        tier_share={k: (v / n if n else 0.0) for k, v in sorted(tier_counts.items())},
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_cost=total_cost,
        input_only_cost=input_only_cost,
        cost_per_request=(total_cost / n) if n else 0.0,
        input_only_cost_per_request=(input_only_cost / n) if n else 0.0,
        cloud_calls=cloud_calls,
        tier_accuracy=(correct / scored) if scored else 0.0,
        unsafe_errors=unsafe,
        assumptions=assumptions,
    )


def compare_strategies(costs: Mapping[str, StrategyCost], *, baseline: str) -> dict[str, dict[str, Any]]:
    """Each strategy against a named baseline, as ratios and percentage deltas.

    Ratios are reported instead of only differences because a $0 local tier makes
    absolute deltas look like the whole story when the story is really that the
    frontier model is an order of magnitude more expensive per input token.
    """
    base = costs[baseline]
    out: dict[str, dict[str, Any]] = {}
    for name, cost in costs.items():
        delta = cost.total_cost - base.total_cost
        out[name] = {
            "total_cost_usd": round(cost.total_cost, 6),
            "vs_baseline_usd": round(delta, 6),
            "vs_baseline_ratio": round(cost.total_cost / base.total_cost, 6) if base.total_cost else None,
            "vs_baseline_pct": round(100.0 * delta / base.total_cost, 3) if base.total_cost else None,
            "tier_accuracy": round(cost.tier_accuracy, 6),
            "unsafe_errors": cost.unsafe_errors,
        }
    return out


def negative_control(
    rows_a: Sequence[CostRow],
    rows_b: Sequence[CostRow],
    *,
    name_a: str,
    name_b: str,
) -> dict[str, Any]:
    """Where two strategies disagree about *correctness*, not about cost.

    This is the answer to "why not just regex?". It reports the prompts each
    strategy gets right that the other gets wrong, in both directions, so the
    comparison cannot be tilted by only counting one side. The example lists
    carry row ids so every claim is traceable into the JSONL.
    """
    if len(rows_a) != len(rows_b):
        raise ValueError(f"row sets must be aligned, got {len(rows_a)} and {len(rows_b)}")
    both_right = both_wrong = 0
    a_only: list[dict[str, str]] = []
    b_only: list[dict[str, str]] = []
    for ra, rb in zip(rows_a, rows_b, strict=True):
        if ra.id != rb.id:
            raise ValueError(f"row sets are not aligned at {ra.id!r} vs {rb.id!r}")
        exp = ra.expected_tier
        ka = classify_error(exp, ra.tier)
        kb = classify_error(exp, rb.tier)
        if ka == "correct" and kb == "correct":
            both_right += 1
        elif ka == "correct":
            a_only.append({"id": ra.id, "expected": exp, name_a: ra.tier, name_b: rb.tier, "b_error": kb})
        elif kb == "correct":
            b_only.append({"id": ra.id, "expected": exp, name_a: ra.tier, name_b: rb.tier, "a_error": ka})
        else:
            both_wrong += 1
    return {
        "a": name_a,
        "b": name_b,
        "n": len(rows_a),
        "both_correct": both_right,
        "both_wrong": both_wrong,
        name_a + "_only_correct": len(a_only),
        name_b + "_only_correct": len(b_only),
        name_a + "_only_examples": a_only,
        name_b + "_only_examples": b_only,
    }


# --------------------------------------------------------------------------- #
# Latency
# --------------------------------------------------------------------------- #
def percentile(values: Sequence[float], pct: float) -> float | None:
    """Linear-interpolated percentile (the numpy default method).

    Implemented here rather than imported so the eval harness stays stdlib-only
    and so the interpolation method is visible instead of being whatever version
    of numpy happens to be installed. Returns ``None`` for an empty input rather
    than raising: an all-gate-blocked run legitimately has no backend latency,
    and "no measurement" must not be printed as "0 ms".
    """
    if not values:
        return None
    if not 0.0 <= pct <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {pct}")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def latency_summary(values: Sequence[float], *, percentiles: Sequence[float] = (50.0, 95.0, 99.0)) -> dict[str, Any]:
    """Count, mean, and the requested percentiles of a latency sample, in ms."""
    vals = [float(v) for v in values]
    out: dict[str, Any] = {
        "n": len(vals),
        "mean_ms": round(sum(vals) / len(vals), 3) if vals else None,
        "min_ms": round(min(vals), 3) if vals else None,
        "max_ms": round(max(vals), 3) if vals else None,
    }
    for p in percentiles:
        value = percentile(vals, p)
        out[f"p{int(p)}_ms"] = None if value is None else round(value, 3)
    return out


def overhead_as_pct(added_ms: float | None, completion_ms: float = REPRESENTATIVE_COMPLETION_MS) -> float | None:
    """Routing overhead as a percentage of a representative completion latency.

    The denominator is an assumption, not a measurement, so the result is only
    meaningful next to the value of ``completion_ms`` that produced it. The
    report prints both.
    """
    if added_ms is None or completion_ms <= 0:
        return None
    return round(100.0 * added_ms / completion_ms, 3)


__all__ = [
    "ASSUMED_OUTPUT_TOKENS",
    "CHARS_PER_TOKEN",
    "ERROR_KINDS",
    "MODEL_FOR_TIER",
    "PRICE_TABLE",
    "REPRESENTATIVE_COMPLETION_MS",
    "STRATEGIES",
    "TIER_RISK_ORDER",
    "CostRow",
    "StrategyCost",
    "call_cost",
    "classify_error",
    "compare_strategies",
    "estimate_input_tokens",
    "latency_summary",
    "negative_control",
    "overhead_as_pct",
    "percentile",
    "price_for",
    "simulate",
    "strategy_always",
    "strategy_measured",
]
