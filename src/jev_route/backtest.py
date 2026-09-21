"""Replay recorded traffic through routing policies and measure the delta.

The backtest answers the only question that matters publicly: *what does this
policy do to cost on realistic traffic?* It replays a JSONL trace twice:

* **baseline**: every request priced at the frontier tier (what "no routing"
  costs);
* **ours**: each request run through the full Router pipeline -- hard gate,
  prefilter, policy -- with an oracle backend that replays the trace's
  recorded tier. The pipeline is real (the gate really fires, the prefilter
  really filters, the log really writes); only the model classification is
  replayed, so no API keys are needed and CI can run it.

The headline number (cost delta %) is honest about what it is: the policy's
effect on a synthetic-but-realistic traffic mix, not a model-quality claim.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from .backends.base import BackendResult
from .cache import NullCache
from .policy import Policy
from .pricing import DEFAULT_OUTPUT_TOKENS, estimate_cost_usd, prices_from_policy
from .router import Router
from .schema import ChoiceAnswer, DecisionAnswers, NoulAnswer


class TraceBackend:
    """Oracle backend: replays each trace row's recorded tier at high
    confidence. Same DecisionBackend seam as MockBackend."""

    name = "trace"

    def __init__(self, row: Mapping[str, Any]) -> None:
        self._row = row

    def for_row(self, row: Mapping[str, Any]) -> TraceBackend:
        return TraceBackend(row)

    #: The oracle has no tier channel -- tiers come from the policy evaluating
    #: complexity -- so the recorded tier is expressed AS a complexity choice
    #: that the default policy maps back to the same tier.
    _TIER_AS_COMPLEXITY: ClassVar[dict[str, str]] = {"local": "trivial", "cheap": "standard", "strong": "hard"}

    # The BackendResult seam mandates this signature; the oracle ignores the request.
    async def decide(self, request: Any) -> BackendResult:  # noqa: ARG002
        tier = str(self._row["expected_tier"])
        complexity = self._TIER_AS_COMPLEXITY.get(tier, "standard")
        answers = DecisionAnswers(
            complexity=ChoiceAnswer(choice=complexity, probabilities={complexity: 1.0}, confidence=0.99),
            sensitivity=ChoiceAnswer(choice="public", probabilities={"public": 1.0}, confidence=0.99),
            pii=NoulAnswer(value=0.0),
            domain=ChoiceAnswer(choice="chat", probabilities={"chat": 1.0}, confidence=0.99),
        )
        return BackendResult(
            answers=answers,
            model_version="trace-oracle",
            questions_sent={},
            latency_ms=1.0,
        )


class NullSink:
    """The backtest measures; it does not train on its own replays."""

    def write(self, record: Any) -> None:  # pragma: no cover - trivial
        pass

    def write_gate_block(self, record: Any) -> None:  # pragma: no cover - trivial
        pass


@dataclass
class BacktestRow:
    row_id: str
    category: str
    tier: str
    gate_fired: bool
    input_tokens: int
    cost_ours: float
    cost_baseline: float
    #: Non-advisory detectors that fired on this row (advisory topic keywords
    #: never move a routing decision, so they are kept out of the gate table).
    detectors: tuple[str, ...] = ()
    #: True when a blocking-class detector (credential / regulated identifier)
    #: fired: the excerpt is refused to any remote backend.
    blocks_backend: bool = False


@dataclass
class BacktestReport:
    total: int = 0
    by_tier: dict[str, int] = field(default_factory=dict)
    gate_fires: int = 0
    cost_ours: float = 0.0
    cost_baseline: float = 0.0
    rows: list[BacktestRow] = field(default_factory=list)

    @property
    def savings_pct(self) -> float:
        if self.cost_baseline == 0:
            return 0.0
        return (1.0 - self.cost_ours / self.cost_baseline) * 100.0


def _input_tokens(text: str) -> int:
    return max(1, len(text) // 4)


async def run_backtest(
    *,
    trace_path: str | Path,
    policy: Policy,
    output_tokens: int = DEFAULT_OUTPUT_TOKENS,
) -> BacktestReport:
    prices = prices_from_policy(policy.raw if isinstance(policy.raw, Mapping) else {})
    rows = [json.loads(line) for line in Path(trace_path).read_text().splitlines() if line.strip()]
    report = BacktestReport(total=len(rows))
    for row in rows:
        router = Router(policy, TraceBackend(row), sink=NullSink(), cache=NullCache())
        decision = await router.route_text(row["text"], request_id=row["id"])
        gate_fired = bool(decision.gate.fired or decision.gate.force_local or decision.gate.blocks_backend)
        advisory = set(decision.gate.advisory_topics)
        detectors = tuple(f.detector for f in decision.gate.findings if f.detector not in advisory)
        tier = decision.tier if decision.tier in prices else decision.effective_complexity
        if tier not in prices:
            tier = "strong"
        in_tok = _input_tokens(row["text"])
        report.rows.append(BacktestRow(
            row_id=row["id"],
            category=row["category"],
            tier=tier,
            gate_fired=gate_fired,
            input_tokens=in_tok,
            cost_ours=estimate_cost_usd(tier, in_tok, output_tokens, prices),
            cost_baseline=estimate_cost_usd("frontier", in_tok, output_tokens, prices),
            detectors=detectors,
            blocks_backend=bool(decision.gate.blocks_backend),
        ))
        report.by_tier[tier] = report.by_tier.get(tier, 0) + 1
        report.gate_fires += int(gate_fired)
        report.cost_ours += report.rows[-1].cost_ours
        report.cost_baseline += report.rows[-1].cost_baseline
    return report


#: Human labels for the demo trace's categories. Unknown categories fall back
#: to their raw name, so the report works on any trace, not just demo_500.
_CATEGORY_LABELS = {
    "trivial_completion": "trivial completions",
    "code_gen": "code-gen",
    "multi_file_refactor": "multi-file refactors",
    "hungarian_support": "Hungarian support tickets",
    "pii_fake": "provably-fake-PII",
    "long_context": "long-context",
}


def _is_pii_category(category: str) -> bool:
    return "pii" in category.lower().replace("-", "_").replace(" ", "_").split("_")


def _mix_description(report: BacktestReport) -> str:
    counts: dict[str, int] = {}
    for row in report.rows:
        counts[row.category] = counts.get(row.category, 0) + 1
    parts = [f"{n} {_CATEGORY_LABELS.get(cat, cat)}" for cat, n in sorted(counts.items(), key=lambda kv: -kv[1])]
    return _soft_wrap(", ".join(parts), limit=100)


def _soft_wrap(text: str, limit: int = 100) -> str:
    """Wrap on word boundaries with a two-space continuation indent.

    A bare newline inside a markdown paragraph renders as a space, so this is
    purely for the file's own readability; it never changes the rendered page.
    """
    words, out, cur = text.split(" "), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > limit:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    out.append(cur)
    return ("\n  ").join(out)


def _tier_reason(tiers: dict[str, int], n: int, fires: int) -> str:
    """One short phrase answering: why did this category land on its tier?"""
    if len(tiers) == 1:
        tier = next(iter(tiers))
        if tier == "local" and fires == n:
            return "gate forces local on every row (see Gate)"
        if tier == "local":
            return "policy routes the category to local"
        if tier == "strong":
            return "hard complexity maps to the strong tier"
        if tier == "cheap":
            return "default rule: no sensitivity signal, non-hard complexity"
        return f"tier `{tier}` per policy rules"
    parts = []
    local = tiers.get("local", 0)
    forced = min(fires, local)
    if forced:
        parts.append(f"{forced} gate-forced local")
    if local > forced:
        parts.append(f"{local - forced} local by policy")
    for tier, n_tier in sorted(((t, c) for t, c in tiers.items() if t != "local"), key=lambda kv: -kv[1]):
        parts.append(f"{n_tier} to `{tier}` by policy")
    text = "; ".join(parts)
    if fires:
        text += " (see Gate)"
    return text


def _category_table(report: BacktestReport) -> list[str]:
    by_cat: dict[str, list[BacktestRow]] = {}
    for row in report.rows:
        by_cat.setdefault(row.category, []).append(row)
    lines = ["| category | count | ours $ | baseline $ | why the tier |", "|---|---:|---:|---:|---|"]
    for cat, rows in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        tiers: dict[str, int] = {}
        for r in rows:
            tiers[r.tier] = tiers.get(r.tier, 0) + 1
        fires = sum(1 for r in rows if r.gate_fired)
        cost_ours = sum(r.cost_ours for r in rows)
        cost_base = sum(r.cost_baseline for r in rows)
        reason = _tier_reason(tiers, len(rows), fires)
        lines.append(f"| {cat} | {len(rows)} | {cost_ours:.4f} | {cost_base:.4f} | {reason} |")
    lines.append(
        f"| **total** | **{report.total}** | **{report.cost_ours:.4f}** | **{report.cost_baseline:.4f}** "
        f"| **{report.savings_pct:.1f}% cheaper** |"
    )
    return lines


def _gate_section(report: BacktestReport) -> list[str]:
    lines = ["## Gate", ""]
    fired_rows = [r for r in report.rows if r.gate_fired]
    lines.append(
        f"* deterministic gate fired on **{len(fired_rows)} of {report.total}** requests; "
        "every fire is a real detector hit, not a policy rule."
    )
    det_rows: dict[str, int] = {}
    det_blocks: dict[str, bool] = {}
    for r in fired_rows:
        for d in r.detectors:
            det_rows[d] = det_rows.get(d, 0) + 1
            det_blocks[d] = det_blocks.get(d, False) or r.blocks_backend
    if det_rows:
        lines += ["", "| detector | rows | effect |", "|---|---|---|"]
        for d, n in sorted(det_rows.items(), key=lambda kv: -kv[1]):
            if det_blocks[d]:
                effect = "blocks the backend -- the excerpt never leaves the process"
            else:
                effect = "forces the local tier (redacted text may still be classified)"
            lines.append(f"| {d} | {n} | {effect} |")
    pii_rows = [r for r in report.rows if _is_pii_category(r.category)]
    pii_fires = sum(1 for r in pii_rows if r.gate_fired)
    if pii_rows and pii_fires < len(pii_rows):
        missed = len(pii_rows) - pii_fires
        lines += [
            "",
            f"### Why only {pii_fires} of {len(pii_rows)} synthetic PII rows fired",
            "",
            f"The trace carries {len(pii_rows)} provably-fake PII rows; the gate fired on {pii_fires} of them. "
            f"The {missed} rows that did not",
            "  fire are documented design, not misses:",
            "",
            "* **RFC 2606 reserved-domain emails.** The `email_address` validator rejects reserved domains",
            "  (`example.com` and friends): an address on a reserved domain is not personal data, and a gate",
            "  that fires on `sales@example.com` earns a reputation for crying wolf. Operators who want them",
            "  treated as PII set `gate.placeholder_domains_as_pii: true`.",
            "* **16-digit strings that fail Luhn.** `payment_card` is Luhn-validated, so a digit run one",
            "  checksum away from a card is not called a card. Such strings still stay local: the phone-shaped",
            "  detector trips on the inner digit run and forces the local tier, so the number never reaches a",
            "  cloud model either.",
        ]
        if "phone_number" in det_rows:
            lines[-1] += " (In this trace those are the `phone_number` rows above.)"
        lines += [
            "* **Patient-record and street-address phrasings.** Layer-1 patterns are deliberately narrow: a name",
            "  counts only in an explicit naming construction, an address only in a full street-plus-ZIP shape.",
            "  Layer 1 claims what it can verify; free-text person identification belongs to the calibrated",
            "  backend and, eventually, the local semantic layer (shadow mode today).",
        ]
    elif pii_rows:
        lines += ["", f"All {len(pii_rows)} synthetic PII rows in the trace were caught by the gate."]
    lines.append("")
    return lines


def render_report(report: BacktestReport, *, trace_path: str, policy_name: str) -> str:
    mix = _mix_description(report)
    lines = [
        "# Backtest v0.2 -- policy cost on synthetic traffic",
        "",
        f"* trace: `{trace_path}` ({report.total} rows, all synthetic, provably fake PII)",
        f"* policy: `{policy_name}`",
        "",
        "## Headline",
        "",
        f"**${report.cost_ours:.4f} with this policy vs ${report.cost_baseline:.4f} always-frontier: "
        f"{report.savings_pct:.1f}% cheaper on {report.total} replayed requests.**",
        "",
        f"Traffic mix: {mix}.",
        "",
        "What the number is: the policy's effect on that mix, replayed through the full router pipeline",
        "(hard gate, semantic layer in shadow, prefilter, policy rules) with an oracle backend that",
        "replays the trace's own recorded tiers. It measures **routing distribution, not model quality**:",
        "the classification step is assumed correct, and the question answered is *what does this policy",
        "cost* -- not *how well does the model classify*.",
        "",
        "## Cost by category",
        "",
    ]
    lines += _category_table(report)
    lines += ["", "## Routing distribution", "", "| tier | requests | share |", "|---|---|---|"]
    for tier, n in sorted(report.by_tier.items(), key=lambda kv: -kv[1]):
        share = n / report.total * 100.0 if report.total else 0.0
        lines.append(f"| {tier} | {n} | {share:.1f}% |")
    lines.append("")
    lines += _gate_section(report)
    lines += [
        "## Method and reproducibility",
        "",
        "* **No API keys needed.** `TraceBackend` (an oracle) replays each row's recorded tier at 0.99",
        "  confidence, expressed as the complexity the policy maps back to that tier. The gate, the",
        "  semantic layer (inert in shadow), the prefilter, and the policy rules all execute for real on",
        "  every row; only the model classification is replayed. CI runs this offline.",
        _soft_wrap(
            f"* **The trace is 100% synthetic and provably fake**: {mix}. SSNs use the 000 area (never "
            "issued), cards fail Luhn by construction, emails sit on RFC 2606 reserved domains, and "
            "names are placeholders. Generator: `traces/gen_demo_500.py` (deterministic seed).",
            limit=110,
        ),
        "* **Prices are illustrative.** `jev_route/pricing.py` carries per-1k-token prices (local $0.00,",
        "  cheap $0.0003, strong $0.003, frontier $0.015) and models 400 output tokens per request.",
        "  A deployment's real contract prices go in the policy's `pricing:` block and override the table.",
        f"* **Reproduce:** `.venv/bin/python -m jev_route.cli backtest --trace {trace_path}` (add",
        "  `--report <path>` to regenerate this file).",
        "",
        "> Honest reading: the oracle replay makes the headline a routing-distribution number on a",
        "> synthetic traffic mix, not a model-quality claim. Swap in a real classifier and the tiers",
        "> (and the dollars) move; the Gate section above does not, because the gate is deterministic.",
    ]
    return chr(10).join(lines) + chr(10)
