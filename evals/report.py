"""Render ``evals/results/REPORT.md`` from ``summary.json``.

Kept separate from :mod:`run_eval` for one reason: the report is the artefact the
README links to and the thing a sceptic reads first, so it gets edited more often
than the measurement code. Mixing a markdown renderer into the harness means every
wording change risks breaking a run that costs API calls. Here, the renderer is a
pure function of the summary dict -- it can be iterated on against a persisted
``summary.json`` with ``run_eval.py --reuse``, forever, at zero cost.

Two rules this renderer follows, because a report is where honest numbers most
easily become dishonest ones:

* **Every table names its denominator.** ``n=186`` next to an accuracy of 0.71 is
  a measurement; ``0.71`` on its own is an advertisement.
* **Assumptions are printed next to the numbers they produced.** The cost section
  repeats the price table, the character-per-token proxy, and the assumed output
  length immediately above the dollar figures, not in a footnote three screens
  away.
"""

from __future__ import annotations

import re
import shlex
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import calibration
import cost_model

#: Tier order used for every table and matrix. Fixed, so two reports are diffable.
TIER_LABELS: tuple[str, ...] = ("local", "cheap", "strong")
COMPLEXITY_LABELS: tuple[str, ...] = ("trivial", "standard", "hard", "frontier")
SENSITIVITY_LABELS: tuple[str, ...] = ("public", "internal", "confidential", "regulated")
DOMAIN_LABELS: tuple[str, ...] = ("code", "writing", "analysis", "chat", "data-extraction")


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def pct(value: Any, digits: int = 1) -> str:
    """Percentage or an explicit ``n/a``. Never an empty cell, never ``0.0%``."""
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.{digits}f}%"


def num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def signed(value: Any, digits: int = 1) -> str:
    """A percentage-point delta with an explicit sign, so direction is unmissable."""
    if value is None:
        return "n/a"
    return f"{float(value) * 100.0:+.{digits}f} pp"


def usd(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"${float(value):,.{digits}f}"


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]], *, aligns: Sequence[str] | None = None) -> str:
    """A markdown table. ``aligns`` is a per-column ``---``/``---:`` spec."""
    aligns = list(aligns) if aligns else ["---"] * len(headers)
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "| " + " | ".join(aligns) + " |"]
    for row in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in row) + " |")
    return "\n".join(out)


def confusion_md(matrix: Mapping[str, Any], labels: Sequence[str]) -> str:
    """Expected-down / predicted-across confusion matrix as markdown."""
    rows = []
    for gold in labels:
        row = matrix.get(gold) or {}
        cells = [int(row.get(pred, 0) or 0) for pred in labels]
        total = sum(cells)
        hit = cells[labels.index(gold)] if gold in labels else 0
        rows.append([f"**{gold}**", *cells, total, pct(hit / total) if total else "n/a"])
    rows.append(["**total**", *[
        sum(int((matrix.get(g) or {}).get(p, 0) or 0) for g in labels) for p in labels
    ], sum(int((matrix.get(g) or {}).get(p, 0) or 0) for g in labels for p in labels), ""])
    return table(
        ["expected \\ predicted", *labels, "n", "recall"],
        rows,
        aligns=["---", *(["---:"] * len(labels)), "---:", "---:"],
    )


def per_class_md(per_class: Mapping[str, Any], labels: Sequence[str]) -> str:
    rows = []
    for label in labels:
        entry = per_class.get(label) or {}
        rows.append([
            label,
            entry.get("support", 0),
            pct(entry.get("accuracy")),
            num(entry.get("precision")),
            num(entry.get("recall")),
            num(entry.get("f1")),
        ])
    macro = per_class.get("_macro") or {}
    rows.append([
        f"**macro (n={macro.get('n', 0)})**", "", pct(macro.get("accuracy")), "", "",
        f"**{num(macro.get('macro_f1'), 4)}**",
    ])
    return table(
        ["class", "support", "accuracy", "precision", "recall", "F1"],
        rows,
        aligns=["---", "---:", "---:", "---:", "---:", "---:"],
    )


# --------------------------------------------------------------------------- #
# Accessors. A partial summary renders "n/a"; it never raises.
# --------------------------------------------------------------------------- #
def _dig(data: Any, *path: str, default: Any = None) -> Any:
    """Read a nested key path, falling back to ``default`` instead of raising.

    WHY: this renderer is pointed at persisted ``summary.json`` files of every
    vintage, and ``run_eval.py --reuse`` re-analyses saved rows without the
    run-time counters (``backend_stats``, ``cache``) a live pass records. A
    missing subtree has to cost one ``n/a`` cell, not the whole report.
    """
    current = data
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return default if current is None else current


def _na(value: Any) -> Any:
    """``n/a`` for anything that would otherwise print as an empty cell.

    ``0`` and ``False`` pass through untouched: "no rows errored" is a result and
    must not be flattened into "we did not measure it".
    """
    if value is None:
        return "n/a"
    if isinstance(value, (str, Mapping, Sequence)) and len(value) == 0:
        return "n/a"
    return value


def _yesno(value: Any) -> str:
    """A boolean as a word, because ``True`` in a provenance table reads as noise."""
    if value is None:
        return "n/a"
    return "yes" if value else "no"


def _pct_units(value: Any, digits: int = 1) -> str:
    """Format a percentage that ``cost_model`` already scaled to percent units.

    WHY a second formatter next to :func:`pct`: ``pct`` takes a fraction
    (``0.643`` -> ``64.3%``), but ``vs_baseline_pct`` and
    ``overhead_pct_of_completion`` arrive pre-multiplied (``-64.316`` means
    ``-64.3%``). Feeding those to ``pct`` would print ``-6431.6%``, and a
    ``value / 100`` at a dozen call sites would hide the unit question instead of
    answering it. Two conventions, two names.
    """
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}%"


def _ms(value: Any, digits: int = 1) -> str:
    """Milliseconds to a fixed precision. Latency is wall-clock, not a ratio."""
    if value is None:
        return "n/a"
    return f"{float(value):,.{digits}f}"


def _tier_count(counts: Any, tier: str) -> str:
    """One tier's row count, where an absent key in a present table means zero.

    WHY not :func:`_na`: ``tier_counts`` only carries the tiers a strategy
    actually used, so ``always_strong`` has no ``local`` key at all. Printing
    ``n/a`` there would claim a missing measurement where the summary is
    positively reporting a count of zero.
    """
    block = _mapping(counts)
    if block is None:
        return "n/a"
    return str(block.get(tier, 0))


def _signed_usd(value: Any, digits: int = 4) -> str:
    """A signed dollar delta. :func:`usd` prints ``$-0.2972``, which reads as a typo."""
    if value is None:
        return "n/a"
    amount = float(value)
    if amount == 0:
        # The baseline row's delta is exactly zero; "+$0.0000" invites the reader
        # to look for the rounding that produced it.
        return usd(0.0, digits)
    return f"{'-' if amount < 0 else '+'}{usd(abs(amount), digits)}"


def _ratio(value: Any, digits: int = 2) -> str:
    """A multiplicative factor. Used for "how many times worse", never for cost."""
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}x"


def _is_number(value: Any) -> bool:
    """True for a real number.

    ``bool`` is excluded on purpose: it is an ``int`` subclass, so ``True - False``
    would happily render as a delta of ``+1`` and read as a measurement.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _delta_int(value: Any) -> str:
    """A signed count delta, so direction survives a table of bare integers."""
    if value is None:
        return "n/a"
    number = int(value)
    # "+0" reads like a rounding artefact; a flat zero is just zero.
    return f"{number:+d}" if number else "0"


def _delta_pair(enabled: Mapping[str, Any], disabled: Mapping[str, Any], *path: str) -> str:
    """``enabled - disabled`` for one field, computed from the two arms.

    WHY computed rather than read from ``enabled_minus_disabled``: the summary only
    persists deltas for accuracy, UNSAFE and expensive. A delta row that is half
    ``n/a`` next to two fully populated rows invites the reader to guess, and
    subtracting two counts the summary does report invents nothing. A field absent
    from either arm still renders ``n/a`` -- absence is never zero.
    """
    left = _dig(enabled, *path, default=None)
    right = _dig(disabled, *path, default=None)
    if not (_is_number(left) and _is_number(right)):
        return "n/a"
    return _delta_int(left - right)


def _delta_mix(enabled: Mapping[str, Any], disabled: Mapping[str, Any]) -> str:
    """The tier-distribution delta column, one signed count per tier, in risk order."""
    left = _block(enabled, "tier_distribution")
    right = _block(disabled, "tier_distribution")
    cells = []
    for tier in cost_model.TIER_RISK_ORDER:
        before, after = left.get(tier), right.get(tier)
        cells.append(
            _delta_int(before - after) if _is_number(before) and _is_number(after) else "n/a"
        )
    return " / ".join(cells)


def _share(part: Any, whole: Any, digits: int = 1) -> str:
    """``part / whole`` as a percentage, or ``n/a`` when the denominator is absent."""
    if part is None or not whole:
        return "n/a"
    return pct(float(part) / float(whole), digits)


def _backends(summary: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    """``(name, analysis)`` pairs, in the order the harness wrote them.

    WHY not sorted: ``run_eval.py`` writes the real backend first when both ran.
    Re-sorting would put the offline mock above the measurement it exists to be
    compared against.
    """
    block = summary.get("backends")
    if not isinstance(block, Mapping):
        return []
    return [(str(name), data) for name, data in block.items() if isinstance(data, Mapping)]


def _primary(backends: Sequence[tuple[str, Mapping[str, Any]]]) -> str | None:
    """The backend the headline quotes: the real API when it was part of the run."""
    names = [name for name, _ in backends]
    if "jev" in names:
        return "jev"
    return names[0] if names else None


def _labels_for(block: Mapping[str, Any] | None, fallback: Sequence[str]) -> list[str]:
    """The class ladder as the run recorded it, falling back to the module constant."""
    recorded = _dig(block, "labels", default=None) if block else None
    if isinstance(recorded, Sequence) and not isinstance(recorded, str) and recorded:
        return [str(x) for x in recorded]
    return list(fallback)


def _sensitivity_ladder(analysis: Mapping[str, Any]) -> list[str]:
    return _labels_for(_dig(analysis, "components", "sensitivity", default=None), SENSITIVITY_LABELS)


def _iter_rows(value: Any) -> list[Mapping[str, Any]]:
    """A list of row mappings from a field that may be absent, ``None``, or scalar."""
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _mapping(value: Any) -> Mapping[str, Any] | None:
    """``value`` when it is a mapping, else ``None``. Keeps `isinstance` noise out of the tables."""
    return value if isinstance(value, Mapping) else None


def _block(data: Any, *path: str) -> Mapping[str, Any]:
    """A nested mapping, or an empty one when the path is absent or holds a scalar.

    WHY: a hand-edited or truncated ``summary.json`` can legitimately put a
    string or a number where a subtree belongs. Every section reads dozens of
    nested fields, so the type check happens once here instead of at each read.
    """
    return _mapping(_dig(data, *path, default=None)) or {}


# --------------------------------------------------------------------------- #
# Prose that is not a number: vocabulary, definitions, and what each field means
# --------------------------------------------------------------------------- #
#: Printed where UNSAFE first appears. A report that quotes "UNSAFE 11" without
#: this paragraph is quoting a number the reader will silently convert into
#: "11 mistakes", which is the exact confusion the metric exists to prevent.
UNSAFE_DEFINITION: str = "\n".join([
    "> **UNSAFE** means: a prompt whose dataset label requires the air-gapped `local` tier, but",
    "> which the router sent to a cloud tier (`cheap` or `strong`). The text left the building.",
    "> It is a count of **privacy violations**, not of accuracy errors, and it is the single most",
    "> important number in this file. It is deliberately *not* the complement of tier accuracy:",
    "> a router that never leaves `local` scores zero UNSAFE and is useless, and a router can be",
    "> mostly accurate and still leak. Read the two numbers separately, always.",
])

#: Meanings for the error vocabulary. The names and their order come from
#: ``cost_model.ERROR_KINDS`` so the table cannot drift from the classifier; only
#: the one-line glosses live here, because a docstring is not a table cell.
ERROR_KIND_MEANING: dict[str, str] = {
    "correct": "routed to the tier the label asked for",
    "unsafe": "labelled `local`, routed to a cloud tier: the data left the building",
    "expensive": "labelled `cheap`/`strong`, routed to `local`: capability wasted, nothing leaked",
    "overspend": "labelled `cheap`, routed to `strong`: frontier prices for flash work",
    "underpowered": "labelled `strong`, routed to `cheap`: a quality risk, not a data risk",
}

#: Glosses for the ``on_uncertain`` knobs, so the policy table says what a
#: threshold does rather than only restating its value.
UNCERTAINTY_MEANING: dict[str, str] = {
    "sensitivity_confidence_below": "below this reported sensitivity confidence, escalate (`null` disables)",
    "sensitivity_bump_levels": "how many rungs up the sensitivity ladder the bump moves",
    "complexity_confidence_below": "below this reported complexity confidence, escalate (`null` disables)",
    "complexity_bump_levels": "how many rungs up the complexity ladder the bump moves",
    "pii_uncertain_threshold": "PII scores in `[t, 1-t]` count as uncertain",
    "pii_uncertain_counts_as_present": "treat an uncertain PII signal as present, i.e. fail closed",
}

#: Glosses for the cost assumptions, printed immediately above the dollars they
#: produced. ``cost_model`` calls these illustrative in its own docstring; the
#: report has to say it in the same screenful as the total.
ASSUMPTION_MEANING: dict[str, str] = {
    "price_table_usd_per_million_tokens": "ILLUSTRATIVE USD per million tokens. Not a quote, not scraped, not live",
    "prices_are_illustrative": "the ratios between strategies are the claim; the totals are a consequence",
    "chars_per_token": "token counts are a character-count proxy: no completion model was called",
    "token_counts_are_char_proxy": "the same proxy for every strategy, so it cancels in comparisons",
    "assumed_output_tokens": "constant assumed completion length, identical for every strategy",
    "jev_decision_api_cost_included": "the decision API's own price is NOT inside any total below",
    "representative_completion_ms": "ASSUMED end-to-end completion latency; every overhead % inherits it",
}

#: Matches the confidence out of a recorded escalation string such as
#: ``"sensitivity internal->confidential (confidence 0.46 < 0.8)"``. Parsing is
#: defensive on purpose: the string is presentation-layer text, so a format
#: change must cost one "n/a" cell and never a traceback.
_ESCALATION_CONFIDENCE = re.compile(r"confidence\s+([0-9]*\.?[0-9]+)")


# --------------------------------------------------------------------------- #
# Front matter: what this file is, the headline, and where the numbers came from
# --------------------------------------------------------------------------- #
def _title(summary: Mapping[str, Any]) -> str:
    """The masthead: identity of the run, before any result is quoted."""
    meta = _block(summary, "meta")
    backends = _backends(summary)
    names = ", ".join(f"`{name}`" for name, _ in backends) or "n/a"
    argv = _dig(meta, "argv", default=None)
    command = "python evals/run_eval.py"
    if isinstance(argv, Sequence) and not isinstance(argv, str) and argv:
        command = f"{command} {shlex.join(str(a) for a in argv)}"
    return "\n".join([
        "# jev-route evaluation report",
        "",
        # The timestamp is the *run's*, taken from the summary, not the render's:
        # a pure function of the summary renders byte-identically forever, which
        # is what makes two reports diffable.
        f"Run completed {_na(meta.get('generated_at'))} (UTC). Rendered from `summary.json` by "
        "`evals/report.py`; the render adds no timestamp of its own, so re-rendering the same "
        "summary is byte-identical.",
        "",
        table(
            ["field", "value"],
            [
                ["backends evaluated", names],
                ["dataset", f"`{_na(_dig(meta, 'dataset', 'path'))}`"],
                [
                    "dataset sha256_16 / rows",
                    f"`{_na(_dig(meta, 'dataset', 'sha256_16'))}` / {_na(_dig(meta, 'dataset', 'n_rows'))}",
                ],
                ["policy", f"`{_na(_dig(meta, 'policy', 'path'))}`"],
                [
                    "policy sha256_16 / version",
                    f"`{_na(_dig(meta, 'policy', 'sha256_16'))}` / {_na(_dig(meta, 'policy', 'version'))}",
                ],
                ["git commit", f"`{_na(meta.get('git_commit'))}`"],
                ["`TYPESAFE_API_KEY` present at run time", _yesno(meta.get("typesafe_api_key_present"))],
                ["re-analysed from saved rows (`--reuse`)", _yesno(meta.get("reuse"))],
                ["calibration bins", _na(meta.get("calibration_bins"))],
                ["concurrency", _na(meta.get("concurrency"))],
            ],
        ),
        "",
        "Command:",
        "",
        "```console",
        command,
        "```",
        "",
        "Every number in this file is read out of `summary.json` at render time; nothing is "
        "hardcoded, so re-running the evaluation regenerates a truthful report. **`n/a` means "
        "the summary did not carry that field. It never means zero.**",
    ])


def _headline_numbers(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """The fields a reader opens this file for, dug out of one backend block.

    Same six quantities ``run_eval._print_headline`` writes to stderr, so the
    console line at the end of a run and this table cannot drift apart.
    """
    tier = _block(analysis, "tier")
    latency = _block(analysis, "latency", "backend_latency_ms")
    return {
        "accuracy": tier.get("accuracy"),
        "n_scored": tier.get("n_scored"),
        "n_rows": tier.get("n_rows"),
        "unsafe": tier.get("unsafe_errors"),
        "unsafe_rate": tier.get("unsafe_rate"),
        "sensitivity_ece": _dig(analysis, "calibration", "sensitivity", "ece"),
        "cost_vs_strong_pct": _dig(analysis, "cost", "vs_always_strong", "jev_route", "vs_baseline_pct"),
        "p50_ms": latency.get("p50_ms"),
        "p95_ms": latency.get("p95_ms"),
        "p99_ms": latency.get("p99_ms"),
        "error": _dig(tier, "excluded", "error"),
        "degraded": _dig(tier, "excluded", "degraded"),
        "missing_tier": _dig(tier, "excluded", "missing_tier"),
    }


def _excluded_cell(h: Mapping[str, Any]) -> str:
    """``error/degraded/missing_tier`` in one cell, with the order stated once."""
    return f"{_na(h.get('error'))} / {_na(h.get('degraded'))} / {_na(h.get('missing_tier'))}"


def _headline(summary: Mapping[str, Any]) -> str:
    """The three numbers that matter, then the full headline table."""
    backends = _backends(summary)
    if not backends:
        return "\n".join(["## Headline", "", UNSAFE_DEFINITION, "", "n/a: this summary carries no `backends` block."])

    numbers = {name: _headline_numbers(analysis) for name, analysis in backends}
    primary = _primary(backends) or backends[0][0]
    lead = numbers[primary]
    lines = [
        "## Headline",
        "",
        UNSAFE_DEFINITION,
        "",
        f"**The three numbers that decide whether this router is worth running** (backend `{primary}`):",
        "",
        f"1. **{_na(lead['unsafe'])} UNSAFE rows** out of {_na(lead['n_scored'])} scored "
        f"({pct(lead['unsafe_rate'], 2)} of the run). Privacy violations, as defined above. "
        "This is the number to fix first.",
        f"2. **{pct(lead['accuracy'])} tier accuracy** over {_na(lead['n_scored'])} scored prompts, with "
        f"{_excluded_cell(lead)} rows excluded (errored / degraded / missing tier). A tier accuracy of "
        f"{pct(lead['accuracy'])} is a measurement of a work in progress, not a product claim: "
        f"{_na(lead['unsafe'])} prompts that should never have left the building did.",
        f"3. **{_pct_units(lead['cost_vs_strong_pct'])} cost versus `always_strong`** on the same prompts. "
        "Assumption-derived: the price table, the character-per-token proxy and the assumed output "
        "length are all estimates, printed in full in the cost section. The *ratio* between "
        "strategies is the load-bearing part; the dollar total is not.",
        "",
        f"Supporting numbers for the same run: sensitivity ECE {num(lead['sensitivity_ece'], 4)}, "
        f"decision latency p50/p95/p99 = {_ms(lead['p50_ms'])} / {_ms(lead['p95_ms'])} / "
        f"{_ms(lead['p99_ms'])} ms.",
        "",
        table(
            [
                "backend",
                "tier accuracy (n scored)",
                "UNSAFE",
                "unsafe rate",
                "sensitivity ECE",
                "cost vs `always_strong`",
                "decision latency p50/p95/p99 (ms)",
                "excluded err/degr/miss",
            ],
            [
                [
                    f"`{name}`",
                    f"{pct(h['accuracy'])} (n={_na(h['n_scored'])})",
                    _na(h["unsafe"]),
                    pct(h["unsafe_rate"], 2),
                    num(h["sensitivity_ece"], 4),
                    _pct_units(h["cost_vs_strong_pct"]),
                    f"{_ms(h['p50_ms'])} / {_ms(h['p95_ms'])} / {_ms(h['p99_ms'])}",
                    _excluded_cell(h),
                ]
                for name, h in numbers.items()
            ],
            aligns=["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
        ),
        "",
        "Two naming traps, because both cost the reader a wrong inference:",
        "",
        "* The cost strategy called `jev_route` means \"the measured routing decisions of the backend in "
        "this row\". It is not specific to the real Jev API; the offline mock's decisions are priced "
        "under the same strategy name.",
        "* `unsafe rate` is UNSAFE divided by *scored* rows, not by dataset rows. The denominator is "
        "printed next to every accuracy and rate in this file.",
        "",
        "Before quoting the accuracy column, read "
        "[The tradeoff: uncertainty escalation on vs off](#the-tradeoff-uncertainty-escalation-on-vs-off). "
        "It shows the accuracy number is what the router *pays* for privacy, not a score it maximises.",
    ]
    return "\n".join(lines)


def _provenance(summary: Mapping[str, Any]) -> str:
    """Everything needed to tell a real run from a mock-only or re-analysed one."""
    meta = _block(summary, "meta")
    backends = _backends(summary)
    lines = [
        "## What was measured",
        "",
        "Provenance first, so no result below has to be taken on trust.",
        "",
        table(
            ["field", "value"],
            [
                ["generated_at (UTC)", _na(meta.get("generated_at"))],
                ["git commit", f"`{_na(meta.get('git_commit'))}`"],
                ["dataset", f"`{_na(_dig(meta, 'dataset', 'path'))}`"],
                ["dataset sha256_16", f"`{_na(_dig(meta, 'dataset', 'sha256_16'))}`"],
                ["dataset rows", _na(_dig(meta, "dataset", "n_rows"))],
                ["policy", f"`{_na(_dig(meta, 'policy', 'path'))}`"],
                ["policy sha256_16", f"`{_na(_dig(meta, 'policy', 'sha256_16'))}`"],
                ["policy version", _na(_dig(meta, "policy", "version"))],
                ["policy failure mode", f"`{_na(_dig(meta, 'policy', 'failure_mode'))}`"],
                ["gate `on_force_local`", f"`{_na(_dig(meta, 'policy', 'gate_on_force_local'))}`"],
                ["`TYPESAFE_API_KEY` present", _yesno(meta.get("typesafe_api_key_present"))],
                ["`--reuse` (re-analysed saved rows)", _yesno(meta.get("reuse"))],
                ["calibration bins", _na(meta.get("calibration_bins"))],
                ["concurrency", _na(meta.get("concurrency"))],
            ],
        ),
    ]
    if meta.get("typesafe_api_key_present"):
        lines += [
            "",
            "A real API key was available to the harness, so a `jev` block below can only have come "
            "from the live TypeSafe API. Confirm it against `model_versions_observed`: that field is "
            "read off the responses, not off the configuration.",
        ]
    else:
        lines += [
            "",
            "No API key was present, so this is an **offline run**: only the deterministic `MockBackend` "
            "could have answered. Nothing here measures the real Jev model.",
        ]
    if meta.get("reuse"):
        lines += [
            "",
            "`--reuse` re-analyses the persisted per-row JSONL instead of calling any backend. The "
            "routing, calibration and cost numbers are therefore the same measurement, but two things "
            "are not carried over and show as `n/a`: `backend_stats` (API call and memo counters) and "
            "the `cache` block. Dataset and policy hashes are recomputed at analysis time, so on a "
            "reused run they describe the files *as they are now*, which need not be the files that "
            "produced the rows.",
        ]

    lines += [
        "",
        "### Per-backend run facts",
        "",
        table(
            ["field", *[f"`{name}`" for name, _ in backends]],
            [
                [
                    label,
                    *[_na(getter(analysis)) for _, analysis in backends],
                ]
                for label, getter in _PROVENANCE_FIELDS
            ],
            aligns=["---", *(["---"] * len(backends))],
        ),
    ]
    if not backends:
        lines += ["", "n/a: this summary carries no `backends` block."]
    else:
        lines += ["", "Raw per-row evidence (one JSONL per policy configuration):", ""]
        for name, analysis in backends:
            paths = _block(analysis, "run_info", "paths")
            if not isinstance(paths, Mapping) or not paths:
                lines += [f"* `{name}`: n/a"]
                continue
            lines += [f"* `{name}`:"]
            lines += [f"  * `{config}`: `{path}`" for config, path in paths.items()]
    return "\n".join(lines)


def _model_versions_cell(analysis: Mapping[str, Any]) -> str:
    """The model versions read off the responses, which is the only real proof of
    which backend answered."""
    versions = _dig(analysis, "model_versions_observed", default=[]) or []
    if not isinstance(versions, Sequence) or not versions:
        return "n/a"
    return ", ".join(f"`{version}`" for version in versions)


#: (label, accessor) pairs for the per-backend provenance table. Accessors are
#: lambdas over one analysis block so the table stays a data structure.
_PROVENANCE_FIELDS: tuple[tuple[str, Any], ...] = (
    ("model versions observed", _model_versions_cell),
    ("run started", lambda a: a.get("run_started")),
    ("run finished", lambda a: a.get("run_finished")),
    ("rows in default configuration", lambda a: a.get("n_rows")),
    ("backend API calls", lambda a: _dig(a, "backend_stats", "api_calls")),
    ("logical router requests", lambda a: _dig(a, "backend_stats", "logical_requests")),
    ("memo hits (replayed answers)", lambda a: _dig(a, "backend_stats", "memo_hits")),
    ("retry attempts", lambda a: _dig(a, "backend_stats", "retry_attempts_total")),
    ("requests needing a retry", lambda a: _dig(a, "backend_stats", "requests_needing_retry")),
    ("final model version", lambda a: _dig(a, "backend_stats", "model_version_final")),
    ("degraded at end of run", lambda a: _dig(a, "backend_stats", "degraded_final")),
    ("reused saved rows", lambda a: _yesno(_dig(a, "run_info", "reused"))),
    ("questions sha256", lambda a: _dig(a, "run_info", "questions_sha256")),
    ("concurrency", lambda a: _dig(a, "run_info", "concurrency")),
)


def _policy_under_test(summary: Mapping[str, Any]) -> str:
    """The policy the numbers belong to. A routing result without it is unfileable."""
    policy = _block(summary, "meta", "policy")
    tiers = policy.get("tiers")
    tier_rows: list[list[Any]] = []
    if isinstance(tiers, Mapping):
        tier_rows = [
            [f"`{tier}`", ", ".join(f"`{m}`" for m in models) if isinstance(models, Sequence) else _na(models)]
            for tier, models in tiers.items()
        ]
    uncertain = policy.get("on_uncertain")
    uncertain_rows: list[list[Any]] = []
    if isinstance(uncertain, Mapping):
        uncertain_rows = [
            [f"`{knob}`", _na(value), UNCERTAINTY_MEANING.get(str(knob), "")] for knob, value in uncertain.items()
        ]
    lines = [
        "## The policy under test",
        "",
        f"Version {_na(policy.get('version'))}, sha256_16 `{_na(policy.get('sha256_16'))}`, "
        f"failure mode `{_na(policy.get('failure_mode'))}`, gate `on_force_local` = "
        f"`{_na(policy.get('gate_on_force_local'))}`.",
        "",
        "### Tiers",
        "",
        table(["tier", "models"], tier_rows or [["n/a", "n/a"]]),
        "",
        "Tier order by data-egress risk: "
        + " < ".join(f"`{t}`" for t in cost_model.TIER_RISK_ORDER)
        + ". `local` is the air-gapped tier; anything to its right moves text out of the building.",
        "",
        "### `on_uncertain` (the escalation policy)",
        "",
        table(["knob", "value", "what it does"], uncertain_rows or [["n/a", "n/a", "n/a"]]),
        "",
        "These knobs are the whole subject of the A/B below: they are what turns \"the model is not "
        "sure\" into \"route one rung more carefully\", and they cost tier accuracy to buy privacy.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The two sections that carry the argument: the A/B, and the unsafe rows
# --------------------------------------------------------------------------- #
def _arms_label(provenance: Any) -> str:
    """``default vs no_escalation`` from the provenance field, whatever its shape."""
    if isinstance(provenance, Sequence) and not isinstance(provenance, str):
        return " vs ".join(str(p) for p in provenance)
    return str(_na(provenance))


def _ab_arm(arm: Any) -> Mapping[str, Any]:
    """One A/B arm as a mapping. ``disabled`` is legitimately ``None`` when the
    ablation configuration was not run."""
    return arm if isinstance(arm, Mapping) else {}


def _ab_row(label: str, arm: Mapping[str, Any]) -> list[Any]:
    """One configuration row of the A/B table, with its denominator attached."""
    kinds = _block(arm, "error_kinds")
    dist = _block(arm, "tier_distribution")
    mix = " / ".join(str(_na(dist.get(tier))) for tier in cost_model.TIER_RISK_ORDER)
    return [
        label,
        _na(arm.get("n_scored")),
        pct(arm.get("accuracy")),
        _na(arm.get("unsafe_errors")),
        _na(kinds.get("expensive")),
        _na(kinds.get("overspend")),
        _na(kinds.get("underpowered")),
        _na(arm.get("escalated_rows")),
        mix,
    ]


def _ab_verdict(acc_on: Any, acc_off: Any, un_on: Any, un_off: Any) -> str:
    """One honest sentence about which arm wins on accuracy and which on privacy.

    WHY the sentence is built from comparisons instead of written once: the
    tradeoff direction is a *finding*, not a constant. On a different dataset or
    a different policy the ablation could move both numbers the same way, and a
    canned "escalation costs accuracy to buy privacy" would then be a lie printed
    next to a table that contradicts it.
    """
    parts: list[str] = []
    if acc_on is not None and acc_off is not None:
        if acc_off > acc_on:
            parts.append(
                f"switching escalation **off** *raises* tier accuracy from {pct(acc_on)} to {pct(acc_off)}"
            )
        elif acc_off < acc_on:
            parts.append(
                f"switching escalation **off** *lowers* tier accuracy from {pct(acc_on)} to {pct(acc_off)}"
            )
        else:
            parts.append(f"switching escalation **off** leaves tier accuracy unchanged at {pct(acc_on)}")
    if un_on is not None and un_off is not None:
        ratio = f" ({_ratio(un_off / un_on)} as many)" if un_on else ""
        if un_off > un_on:
            parts.append(f"and *raises* UNSAFE privacy violations from {un_on} to {un_off}{ratio}")
        elif un_off < un_on:
            parts.append(f"and *lowers* UNSAFE privacy violations from {un_on} to {un_off}{ratio}")
        else:
            parts.append(f"and leaves UNSAFE unchanged at {un_on}")
    if not parts:
        return "n/a: this summary does not carry both A/B arms."
    sentence = " ".join(parts)
    sentence = sentence[0].upper() + sentence[1:] + "."
    if (
        acc_on is not None
        and acc_off is not None
        and un_on is not None
        and un_off is not None
        and acc_off > acc_on
        and un_off > un_on
    ):
        sentence += (
            " **That is the argument for calibrated routing in one line.** An argmax classifier that "
            "ignores its own uncertainty is the *more accurate* tier predictor and the *less safe* "
            "router. Ranking configurations by tier accuracy alone would pick the one that leaks more. "
            "The escalation policy is deliberately buying privacy with accuracy, and the price of both "
            "sides is on the row above."
        )
    return sentence


def _escalation_tradeoff(summary: Mapping[str, Any]) -> str:
    """The A/B: what `on_uncertain` costs and what it buys, per backend."""
    lines = [
        "## The tradeoff: uncertainty escalation on vs off",
        "",
        "Both arms replay **the same recorded backend answers**; only the policy differs "
        "(`on_uncertain` enabled vs disabled, gate and rules identical). That makes the comparison a "
        "counterfactual over one sample rather than two draws from a nondeterministic API. The "
        "configuration each arm came from is printed per backend under `provenance.escalation_ab`.",
        "",
    ]
    backends = _backends(summary)
    if not backends:
        lines += ["n/a: this summary carries no `backends` block."]
        return "\n".join(lines)
    for name, analysis in backends:
        ab = _dig(analysis, "escalation_ab", default=None)
        lines += [f"### Backend `{name}`", ""]
        if not isinstance(ab, Mapping) or not ab:
            lines += ["n/a: no `escalation_ab` block in this summary.", ""]
            continue
        enabled, disabled = _ab_arm(ab.get("enabled")), _ab_arm(ab.get("disabled"))
        provenance = _dig(analysis, "provenance", "escalation_ab", default=None)
        lines += [
            f"Arms: `{_arms_label(provenance)}`.",
            "",
            table(
                [
                    "configuration",
                    "n scored",
                    "tier accuracy",
                    "UNSAFE",
                    "expensive",
                    "overspend",
                    "underpowered",
                    "escalated rows",
                    "local / cheap / strong",
                ],
                [
                    _ab_row("`on_uncertain` **enabled** (shipped policy)", enabled),
                    _ab_row("`on_uncertain` **disabled** (ablation)", disabled),
                    [
                        "**delta (enabled - disabled)**",
                        _delta_pair(enabled, disabled, "n_scored"),
                        signed(_dig(ab, "enabled_minus_disabled", "accuracy_delta")),
                        _delta_int(_dig(ab, "enabled_minus_disabled", "unsafe_delta")),
                        _delta_int(_dig(ab, "enabled_minus_disabled", "expensive_delta")),
                        _delta_pair(enabled, disabled, "error_kinds", "overspend"),
                        _delta_pair(enabled, disabled, "error_kinds", "underpowered"),
                        _delta_pair(enabled, disabled, "escalated_rows"),
                        _delta_mix(enabled, disabled),
                    ],
                ],
                aligns=["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
            ),
            "",
            _ab_verdict(
                enabled.get("accuracy"),
                disabled.get("accuracy"),
                enabled.get("unsafe_errors"),
                disabled.get("unsafe_errors"),
            ),
            "",
        ]
        bumped, scored = enabled.get("escalated_rows"), enabled.get("n_scored")
        if bumped is not None and scored:
            lines += [
                f"The shipped policy bumped {_na(bumped)} of {_na(scored)} scored rows "
                f"({_share(bumped, scored)}) after at least one head came back below its confidence "
                "floor. Those bumps are the mechanism: they move a row up the sensitivity or "
                "complexity ladder, and the rules then route it one tier more carefully.",
                "",
            ]
        if not disabled:
            lines += [
                "The `disabled` arm is absent, so this run measured the shipped policy only. "
                "Nothing here supports a claim about what escalation costs or buys.",
                "",
            ]
    return "\n".join(lines)


def _unsafe_breakdown(rows: Sequence[Mapping[str, Any]], ladder: Sequence[str]) -> dict[str, Any]:
    """Counts over the unsafe rows, derived only from fields the summary carries.

    ``bumped`` means an escalation string for the sensitivity head was recorded,
    i.e. the router *knew* it was unsure about sensitivity and moved one rung.
    ``confident`` means no such string was recorded: the reported confidence was
    at or above the policy floor and the call was simply wrong.
    """
    bumped: list[Mapping[str, Any]] = []
    confident: list[Mapping[str, Any]] = []
    confidences: list[float] = []
    labels: Counter[str] = Counter()
    predicted: Counter[str] = Counter()
    below: Counter[int] = Counter()
    rules: Counter[str] = Counter()
    for row in rows:
        sens_bumps = [
            str(e) for e in _iter_rows_none(row.get("escalated")) if str(e).startswith("sensitivity")
        ]
        if sens_bumps:
            bumped.append(row)
            match = _ESCALATION_CONFIDENCE.search(sens_bumps[0])
            if match:
                confidences.append(float(match.group(1)))
        else:
            confident.append(row)
        labels[str(_dig(row, "labels", "sensitivity", default="n/a"))] += 1
        predicted[str(row.get("predicted", "n/a"))] += 1
        rules[str(row.get("rule_id", "n/a"))] += 1
        gold = str(_dig(row, "labels", "sensitivity", default=""))
        routed = str(_dig(row, "effective", "sensitivity", default=""))
        if gold in ladder and routed in ladder:
            below[ladder.index(gold) - ladder.index(routed)] += 1
    return {
        "bumped": bumped,
        "confident": confident,
        "confidences": sorted(confidences),
        "labels": labels,
        "predicted": predicted,
        "levels_below": below,
        "rules": rules,
    }


def _iter_rows_none(value: Any) -> list[Any]:
    """A list from a field that may be ``None``, a scalar, or already a list."""
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, str):
        return list(value)
    return [value]


def _unsafe_error_analysis(summary: Mapping[str, Any]) -> str:
    """Every unsafe row, and the split between "unsure and bumped" and "sure and wrong"."""
    on_uncertain = _block(summary, "meta", "policy", "on_uncertain")
    floor = on_uncertain.get("sensitivity_confidence_below")
    bump_levels = on_uncertain.get("sensitivity_bump_levels")
    lines = [
        "## Error analysis: every UNSAFE row",
        "",
        "Listed row by row, because a privacy count that cannot be traced to a prompt id is an "
        "assertion rather than a measurement. Every id below appears in the per-row JSONL named in the "
        "provenance section.",
        "",
        f"The policy's sensitivity floor is `{_na(floor)}`; a call below it is bumped "
        f"{_rungs(bump_levels)}. "
        "A row is **bumped** when the backend reported sensitivity confidence below that floor and the "
        "policy moved it up the ladder; it is **confidently wrong** when no sensitivity bump was "
        "recorded, meaning the backend's reported confidence cleared the floor and the call was "
        "nevertheless not the label.",
        "",
    ]
    backends = _backends(summary)
    if not backends:
        lines += ["n/a: this summary carries no `backends` block."]
        return "\n".join(lines)
    for name, analysis in backends:
        tier = _block(analysis, "tier")
        ladder = _sensitivity_ladder(analysis)
        wrong = _iter_rows(tier.get("wrong_rows"))
        unsafe_rows = [row for row in wrong if row.get("kind") == "unsafe"]
        lines += [f"### Backend `{name}`", ""]
        if not wrong:
            lines += [
                "n/a: this summary carries no `tier.wrong_rows` list, so the unsafe rows cannot be "
                f"itemised here. The counter itself reads {_na(tier.get('unsafe_errors'))}.",
                "",
            ]
            continue
        counted = tier.get("unsafe_errors")
        if counted is not None and int(counted) != len(unsafe_rows):
            lines += [
                f"**Inconsistency in the summary:** `tier.unsafe_errors` says {_na(counted)} but "
                f"`tier.wrong_rows` carries {len(unsafe_rows)} rows of kind `unsafe`. The table below "
                "shows what the file actually contains; the discrepancy is reported, not smoothed over.",
                "",
            ]
        if not unsafe_rows:
            lines += [
                "No row of kind `unsafe` in `tier.wrong_rows`: nothing labelled `local` was routed to "
                f"a cloud tier in this run (n scored = {_na(tier.get('n_scored'))}).",
                "",
            ]
            continue
        breakdown = _unsafe_breakdown(unsafe_rows, ladder)
        lines += [
            table(
                [
                    "row id",
                    "label sensitivity",
                    "routed sensitivity",
                    "rungs below label",
                    "routed to",
                    "rule that fired",
                    "sensitivity bump recorded",
                    "difficulty",
                ],
                [
                    [
                        f"`{_na(row.get('id'))}`",
                        _na(_dig(row, "labels", "sensitivity")),
                        _na(_dig(row, "effective", "sensitivity")),
                        _rungs_below(row, ladder),
                        f"`{_na(row.get('predicted'))}`",
                        f"`{_na(row.get('rule_id'))}`",
                        _bump_cell(row),
                        _na(row.get("difficulty")),
                    ]
                    for row in unsafe_rows
                ],
                aligns=["---", "---", "---", "---:", "---", "---", "---", "---"],
            ),
            "",
            f"Of the {len(unsafe_rows)} UNSAFE rows in this run:",
            "",
            table(
                ["breakdown", "rows"],
                [
                    [
                        f"bumped: sensitivity confidence below the floor, moved {_rungs(bump_levels)}, "
                        "still not far enough",
                        len(breakdown["bumped"]),
                    ],
                    [
                        "confidently wrong: no sensitivity bump recorded, so the reported confidence "
                        "cleared the floor",
                        len(breakdown["confident"]),
                    ],
                    ["routed to `cheap`", breakdown["predicted"].get("cheap", 0)],
                    ["routed to `strong`", breakdown["predicted"].get("strong", 0)],
                    *[
                        [f"label was `{label}`", count]
                        for label, count in breakdown["labels"].most_common()
                    ],
                    *[
                        [f"{_rungs(rungs)} below the label", count]
                        for rungs, count in sorted(breakdown["levels_below"].items())
                    ],
                    *[
                        [f"rule that fired: `{rule}`", count]
                        for rule, count in breakdown["rules"].most_common()
                    ],
                ],
                aligns=["---", "---:"],
            ),
            "",
        ]
        confidences = breakdown["confidences"]
        if confidences:
            listed = ", ".join(f"{c:.2f}" for c in confidences[:_MAX_LISTED_CONFIDENCES])
            if len(confidences) > _MAX_LISTED_CONFIDENCES:
                listed += (
                    f" ... {len(confidences)} values in total, spanning {min(confidences):.2f} to "
                    f"{max(confidences):.2f}"
                )
            lines += [
                f"The recorded sensitivity confidences on the bumped rows are {listed} "
                f"(floor `{_na(floor)}`, bump `{_na(bump_levels)}`). A bump of {_rungs(bump_levels)} "
                "cannot recover a call that sits further down the ladder than that, which is what "
                "the \"rungs below label\" column above shows.",
                "",
            ]
        top_rule = breakdown["rules"].most_common(1)
        if top_rule:
            rule, rule_count = top_rule[0]
            gloss = (
                ", i.e. a complexity rule sent the request to a cloud tier while the sensitivity "
                "signal that should have forced `local` never arrived"
                if str(rule).startswith("complexity")
                else ""
            )
            lines += [
                f"The rule that fired most often on these rows was `{rule}` "
                f"({rule_count} of {len(unsafe_rows)}){gloss}. Escalation cannot fix a head that is "
                "confidently wrong; it can only widen the band in which a head is allowed to say it "
                "does not know.",
                "",
            ]
        local_fn = _dig(analysis, "tier", "per_class", "local", "fn")
        if local_fn is not None and counted is not None and int(local_fn) == int(counted):
            lines += [
                f"Consistency check: `tier.per_class.local.fn` ({_na(local_fn)}) equals "
                f"`tier.unsafe_errors` ({_na(counted)}). By construction every `local` recall miss is "
                "a privacy violation, so the two must agree; if they ever diverge, one of them is "
                "being computed on a different row set.",
                "",
            ]
        lines += [
            "**What this file cannot tell you.** `summary.json` records a confidence only where a bump "
            "fired. For the confidently-wrong rows the backend's reported sensitivity confidence is "
            "*not* in the summary, so no range is quoted for them here. It is in the per-row JSONL "
            "under `raw_answers.sensitivity.confidence`, and reading it out is a one-line `jq`.",
            "",
        ]
    return "\n".join(lines)


#: How many recorded confidences are listed inline before the list becomes a
#: range. A four-value list is evidence; a thirty-six-value list is wallpaper.
_MAX_LISTED_CONFIDENCES = 12

#: How many row ids are quoted from the negative control's example lists. Every
#: one of them is in `summary.json`, so a sample is a pointer, not a selection.
_MAX_LISTED_EXAMPLES = 5


def _rungs(value: Any) -> str:
    """``1 rung`` / ``2 rungs``. Prose that has to agree with a summary field."""
    if value is None:
        return "an unknown number of rungs"
    try:
        count = int(value)
    except (TypeError, ValueError):
        return f"{value} rungs"
    return f"{count} rung" if count == 1 else f"{count} rungs"


def _rungs_below(row: Mapping[str, Any], ladder: Sequence[str]) -> str:
    """How far the routed sensitivity sits below the label, on the recorded ladder."""
    gold = str(_dig(row, "labels", "sensitivity", default=""))
    routed = str(_dig(row, "effective", "sensitivity", default=""))
    if gold not in ladder or routed not in ladder:
        return "n/a"
    return str(ladder.index(gold) - ladder.index(routed))


def _bump_cell(row: Mapping[str, Any]) -> str:
    """The recorded sensitivity escalations verbatim, or an explicit "none"."""
    bumps = [str(e) for e in _iter_rows_none(row.get("escalated")) if str(e).startswith("sensitivity")]
    return "; ".join(f"`{b}`" for b in bumps) if bumps else "none recorded"


# --------------------------------------------------------------------------- #
# Per-backend detail
# --------------------------------------------------------------------------- #
def _backend_detail(name: str, analysis: Mapping[str, Any], calibration_reports: Mapping[str, Any] | None) -> str:
    """Everything about one backend: routing, heads, calibration, cost, latency, cache."""
    lines = [f"## Backend detail: `{name}`", ""]
    lines += _routing_block(analysis)
    lines += _components_block(analysis)
    lines += _calibration_section(name, analysis, calibration_reports)
    lines += _cost_section(analysis)
    lines += _latency_section(analysis)
    lines += _cache_section(analysis)
    return "\n".join(lines)


def _provenance_line(analysis: Mapping[str, Any], *keys: str) -> str:
    """``Configuration: X`` for the named provenance entries, so no table is orphaned."""
    provenance = _block(analysis, "provenance")
    parts = [
        f"`{key}` measured on configuration `{_arms_label(provenance.get(key))}`"
        for key in keys
        if key in provenance
    ]
    if not parts:
        return ""
    return "Provenance: " + "; ".join(parts) + "."


def _routing_block(analysis: Mapping[str, Any]) -> list[str]:
    """Tier accuracy, the confusion matrix, per-class metrics, and where errors land."""
    tier = _block(analysis, "tier")
    excluded = _block(tier, "excluded")
    ladder = list(cost_model.TIER_RISK_ORDER)
    kinds = _block(tier, "error_kinds")
    n_scored = tier.get("n_scored")
    return [
        "### Routing decisions",
        "",
        _provenance_line(analysis, "tier_accuracy"),
        "",
        table(
            ["measure", "value"],
            [
                ["rows in the default configuration", _na(tier.get("n_rows"))],
                ["rows scored", _na(n_scored)],
                ["tier accuracy", pct(tier.get("accuracy"))],
                ["UNSAFE (privacy violations)", _na(tier.get("unsafe_errors"))],
                ["unsafe rate (of scored rows)", pct(tier.get("unsafe_rate"), 2)],
                ["expensive errors", _na(tier.get("expensive_errors"))],
                ["classified by the backend", _na(tier.get("classified_rows"))],
                ["escalated by `on_uncertain`", _na(tier.get("escalated_rows"))],
                ["gate forced `local`", _na(tier.get("gate_forced_rows"))],
                ["gate blocked the backend call", _na(tier.get("gate_blocked_rows"))],
                ["excluded: errored", _na(excluded.get("error"))],
                ["excluded: degraded", _na(excluded.get("degraded"))],
                ["excluded: missing tier", _na(excluded.get("missing_tier"))],
            ],
        ),
        "",
        "Excluded rows are counted out loud and kept out of the denominator. They are *not* scored as "
        "wrong, and they are *not* scored as right: a degraded backend fails closed to `local`, which "
        "on this dataset would count as a correct routing decision for a large share of rows and "
        "inflate accuracy by accident.",
        "",
        "#### Tier confusion (expected down, predicted across)",
        "",
        confusion_md(_block(tier, "confusion"), ladder),
        "",
        "Read the upper-right cell as the privacy number: rows labelled `local` that were predicted "
        "`strong`. Every off-diagonal cell to the right of the diagonal in the `local` row is an "
        "UNSAFE row.",
        "",
        "#### Per-tier precision, recall, F1",
        "",
        per_class_md(_block(tier, "per_class"), ladder),
        "",
        "#### Where the errors land",
        "",
        table(
            ["kind", "rows", "share of scored", "what it means"],
            [
                [
                    f"`{kind}`",
                    _na(kinds.get(kind)),
                    _share(kinds.get(kind), n_scored, 2),
                    ERROR_KIND_MEANING.get(kind, ""),
                ]
                for kind in cost_model.ERROR_KINDS
            ],
            aligns=["---", "---:", "---:", "---"],
        ),
        "",
        "These buckets are not interchangeable and must never be averaged into one error rate. "
        "`unsafe` and `expensive` differ by orders of magnitude in consequence: one leaks regulated "
        "data to a third party, the other wastes a GPU you already own.",
        "",
        "#### By expected tier",
        "",
        _per_expected_table(_block(tier, "per_expected_tier")),
        "",
        "#### By labelled difficulty",
        "",
        _by_difficulty_table(_block(tier, "by_difficulty")),
        "",
    ]


def _per_expected_table(per_expected: Mapping[str, Any]) -> str:
    if not isinstance(per_expected, Mapping) or not per_expected:
        return "n/a"
    rows = []
    for tier, entry in per_expected.items():
        entry = entry if isinstance(entry, Mapping) else {}
        rows.append([
            f"`{tier}`",
            _na(entry.get("n")),
            _na(entry.get("correct")),
            pct(entry.get("accuracy")),
            _na(entry.get("unsafe")),
            _na(entry.get("expensive")),
            _na(entry.get("overspend")),
            _na(entry.get("underpowered")),
        ])
    return table(
        ["expected tier", "n", "correct", "accuracy", "unsafe", "expensive", "overspend", "underpowered"],
        rows,
        aligns=["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
    )


def _by_difficulty_table(by_difficulty: Mapping[str, Any]) -> str:
    """Accuracy on rows the dataset labels `clear` versus `ambiguous`.

    Printed because the two subsets are not the same measurement: on an ambiguous
    row a human labeller could have picked a different tier, so a *lower*
    accuracy there is partly a property of the labels.
    """
    if not isinstance(by_difficulty, Mapping) or not by_difficulty:
        return "n/a"
    rows = []
    for difficulty, entry in by_difficulty.items():
        entry = entry if isinstance(entry, Mapping) else {}
        rows.append([
            f"`{difficulty}`",
            _na(entry.get("n")),
            pct(entry.get("accuracy")),
            _na(entry.get("unsafe")),
        ])
    return (
        table(["difficulty", "n", "accuracy", "UNSAFE"], rows, aligns=["---", "---:", "---:", "---:"])
        + _difficulty_inversion(by_difficulty)
    )


def _difficulty_inversion(by_difficulty: Mapping[str, Any]) -> str:
    """A note when `ambiguous` rows outscore `clear` ones, which otherwise reads as a paradox.

    WHY derived rather than always printed: on most runs the ordering is the
    intuitive one, and a standing caveat about an inversion that did not happen
    would be noise next to a table that contradicts it.
    """
    clear = _mapping(by_difficulty.get("clear")) or {}
    ambiguous = _mapping(by_difficulty.get("ambiguous")) or {}
    if clear.get("accuracy") is None or ambiguous.get("accuracy") is None:
        return ""
    if float(ambiguous["accuracy"]) <= float(clear["accuracy"]):
        return ""
    return (
        "\n\nThe inversion is worth naming: rows the dataset labels `ambiguous` score *higher* "
        "than rows labelled `clear`. `difficulty` records whether a reasonable expert could "
        "disagree with the label, not how hard the row is for this policy, so the two need not "
        "line up. Here they do not."
    )


def _components_block(analysis: Mapping[str, Any]) -> list[str]:
    """Per-head label accuracy: what the backend actually judged, before policy."""
    components = _block(analysis, "components")
    lines = [
        "### Component heads",
        "",
        _provenance_line(analysis, "component_labels"),
        "",
        "These are the backend's **raw** judgements, scored against the dataset's label for that head: "
        "pre-gate-merge, pre-escalation. Scoring the effective label here would credit the gate and the "
        "policy for the model's accuracy and debit them for its mistakes, and the resulting number "
        "would describe neither.",
        "",
        "`n_unclassified` counts rows the backend never answered (the gate blocked the call). That is a "
        "real selection effect: blocked rows are the ones containing checksum-valid identifiers, so the "
        "denominators below are not the whole dataset.",
        "",
    ]
    choice_heads = [
        (head, _labels_for(_mapping(components.get(head)), ladder))
        for head, ladder in (
            ("complexity", COMPLEXITY_LABELS),
            ("sensitivity", SENSITIVITY_LABELS),
            ("domain", DOMAIN_LABELS),
        )
    ]
    for head, ladder in choice_heads:
        block = components.get(head)
        lines += [f"#### `{head}`", ""]
        if not isinstance(block, Mapping) or not block:
            lines += [f"n/a: this summary carries no `components.{head}` block.", ""]
            continue
        lines += [
            table(
                ["measure", "value"],
                [
                    ["macro F1", num(block.get("macro_f1"), 4)],
                    ["accuracy", pct(block.get("accuracy"))],
                    ["rows scored", _na(block.get("n_scored"))],
                    ["rows never answered", _na(block.get("n_unclassified"))],
                    ["rows where the recorded choice was not the argmax", _na(block.get("argmax_mismatch_rows"))],
                ],
            ),
            "",
            "Confusion (expected down, predicted across):",
            "",
            confusion_md(_block(block, "confusion"), ladder),
            "",
            "Per class:",
            "",
            per_class_md(_block(block, "per_class"), ladder),
            "",
        ]
    lines += _pii_block(components.get("pii"))
    return lines


def _pii_block(block: Any) -> list[str]:
    """PII gets two views because they answer two different questions."""
    lines = ["#### `pii`", ""]
    if not isinstance(block, Mapping) or not block:
        return [*lines, "n/a: this summary carries no `components.pii` block.", ""]
    lines += [
        f"Decision threshold: `{_na(block.get('threshold'))}`. "
        f"Rows never answered: {_na(block.get('n_unclassified'))}.",
        "",
    ]
    for view, caption in (
        ("raw", "`raw`: the backend's PII score thresholded. Can the model tell PII from non-PII?"),
        (
            "effective",
            "`effective`: after the gate floor and the uncertain-band rule. "
            "This is the view that decides data egress.",
        ),
    ):
        metrics = block.get(view)
        lines += [caption, ""]
        if not isinstance(metrics, Mapping) or not metrics:
            lines += ["n/a", ""]
            continue
        lines += [
            table(
                ["measure", "value"],
                [
                    ["n", _na(metrics.get("n"))],
                    ["true positives", _na(metrics.get("tp"))],
                    ["true negatives", _na(metrics.get("tn"))],
                    ["false positives", _na(metrics.get("fp"))],
                    ["false negatives (missed PII)", _na(metrics.get("fn"))],
                    ["accuracy", pct(metrics.get("accuracy"))],
                    ["precision", num(metrics.get("precision"), 4)],
                    ["recall", num(metrics.get("recall"), 4)],
                    ["F1", num(metrics.get("f1"), 4)],
                    ["missed PII", _na(metrics.get("missed_pii"))],
                    ["false PII", _na(metrics.get("false_pii"))],
                ],
                aligns=["---", "---:"],
            ),
            "",
        ]
    note = block.get("note")
    if note:
        lines += [f"> {note}", ""]
    return lines


def _report_from_dict(data: Any) -> Any:
    """A :class:`calibration.CalibrationReport` rebuilt from its serialised form."""
    if not isinstance(data, Mapping) or not data:
        return None
    try:
        return calibration.CalibrationReport.from_dict(data)
    except (TypeError, ValueError, KeyError):
        # A hand-edited or truncated bin list must cost one missing table, not the
        # whole report.
        return None


def _report_object(
    calibration_reports: Mapping[str, Any] | None, backend: str, head: str, data: Any
) -> Any:
    """The calibration report for one head, preferring the persisted dict.

    WHY the dict wins over the live object ``run_eval.py`` hands us:
    ``CalibrationReport.to_dict`` rounds to six decimals, so the live object and
    the summary can disagree in the *fourth* decimal of a printed MCE when the
    value lands on a rounding tie (0.50875 prints as 0.5088 from the dict and as
    0.5087 from the live float). Preferring the dict makes the committed
    ``REPORT.md`` byte-identical to what anyone gets by re-rendering
    ``summary.json`` alone, which is the property that makes the file checkable.
    The live object stays as the fallback for a summary that lost its bins.
    """
    rebuilt = _report_from_dict(data)
    if rebuilt is not None:
        return rebuilt
    live = _dig(calibration_reports, backend, head, default=None)
    if isinstance(live, calibration.CalibrationReport):
        return live
    return None


def _calibration_section(
    name: str, analysis: Mapping[str, Any], calibration_reports: Mapping[str, Any] | None
) -> list[str]:
    """ECE and reliability per head. Calibration is the input to `on_uncertain`."""
    blocks = _mapping(_dig(analysis, "calibration", default=None)) or {}
    preferred = ("complexity", "sensitivity", "domain", "pii")
    heads = [head for head in preferred if head in blocks]
    heads += [head for head in blocks if head not in preferred]
    lines = [
        "### Calibration",
        "",
        _provenance_line(analysis, "calibration"),
        "",
        "Confidence is what `on_uncertain` reads, so a miscalibrated head is not a cosmetic problem: "
        "an over-confident head never triggers the bump and quietly routes sensitive text to the "
        "cloud, and an under-confident head triggers it constantly and quietly makes the router "
        "expensive. ECE is the aggregate; the reliability tables below show which side of the "
        "diagonal each confidence band sits on.",
        "",
        table(
            [
                "head",
                "n",
                "accuracy",
                "ECE",
                "ECE (top-probability)",
                "MCE",
                "Brier",
                "Brier skill vs class prior",
                "populated bins",
                "confidence source",
                "mean reported confidence",
            ],
            [
                [
                    f"`{head}`",
                    _na(_dig(blocks, head, "n")),
                    pct(_dig(blocks, head, "accuracy")),
                    num(_dig(blocks, head, "ece"), 4),
                    num(_dig(blocks, head, "ece_top_probability"), 4),
                    num(_dig(blocks, head, "mce"), 4),
                    num(_dig(blocks, head, "brier"), 4),
                    num(_dig(blocks, head, "brier_skill"), 4),
                    f"{_na(_dig(blocks, head, 'populated_bins'))} / {_na(_dig(blocks, head, 'n_bins'))}",
                    _na(_dig(blocks, head, "confidence_source")),
                    num(_dig(blocks, head, "mean_reported_confidence"), 4),
                ]
                for head in heads
            ] or [["n/a", *[""] * 10]],
            aligns=["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---", "---:"],
        ),
        "",
        "A Brier skill below zero means the head's probabilities are worse than always predicting the "
        "class prior. The `populated bins` column matters as much as the ECE: with a few hundred rows "
        "and a coarse reported confidence, an ECE can rest on a handful of non-empty bins, and quoting "
        "it without that count overstates the precision of the measurement.",
        "",
    ]
    if not heads:
        lines += ["n/a: this summary carries no `calibration` block.", ""]
        return lines
    for head in heads:
        data = blocks.get(head)
        report = _report_object(calibration_reports, name, head, data)
        lines += [f"#### `{head}` reliability", ""]
        notes = _dig(data, "notes", default=None) if isinstance(data, Mapping) else None
        if isinstance(notes, Sequence) and not isinstance(notes, str):
            lines += [f"* {note}" for note in notes]
            lines += [""] if notes else []
        if report is None:
            lines += ["n/a: no calibration report for this head in the summary.", ""]
            continue
        lines += [calibration.render_reliability_table(report), ""]
        if head == "pii" and getattr(report, "event_bins", ()):
            lines += [
                "Event view (predicted `p(yes)` against the observed yes-rate), which is the one that "
                "matters for a binary gate:",
                "",
                calibration.render_reliability_table(report, event=True),
                "",
            ]
        if getattr(report, "coverage", None):
            lines += [calibration.render_coverage_table(report), ""]
    return lines


def _assumptions_block(assumptions: Mapping[str, Any]) -> list[str]:
    """Print the assumptions above the dollars they produced, never in a footnote."""
    lines = [
        "#### Assumptions behind every dollar figure below",
        "",
        "**These are estimates, not measurements.** No completion model was called during this "
        "evaluation, so there is no real token count to read; the prices are illustrative placeholders "
        "chosen to be in the right order of magnitude. The *ratio* between strategies is the claim. "
        "The absolute total is a consequence of the numbers in this table and should be recomputed "
        "against your own price sheet.",
        "",
    ]
    if not isinstance(assumptions, Mapping) or not assumptions:
        return [*lines, "n/a: this summary carries no `cost.assumptions` block.", ""]
    price_table = assumptions.get("price_table_usd_per_million_tokens")
    rows = []
    for key, value in assumptions.items():
        if key == "price_table_usd_per_million_tokens":
            rows.append([f"`{key}`", "see the table below", ASSUMPTION_MEANING.get(str(key), "")])
        elif isinstance(value, bool):
            rows.append([f"`{key}`", _yesno(value), ASSUMPTION_MEANING.get(str(key), "")])
        else:
            rows.append([f"`{key}`", _na(value), ASSUMPTION_MEANING.get(str(key), "")])
    lines += [table(["assumption", "value", "what it means"], rows), ""]
    if isinstance(price_table, Mapping) and price_table:
        lines += [
            table(
                ["model", "USD per million input tokens", "USD per million output tokens"],
                [
                    [
                        f"`{model}`",
                        usd(_dig(prices, "input"), 2),
                        usd(_dig(prices, "output"), 2),
                    ]
                    for model, prices in price_table.items()
                ],
                aligns=["---", "---:", "---:"],
            ),
            "",
            "A `$0.00` row is a self-hosted model: its marginal API price is zero by construction, and "
            "the real cost is GPU-hours already owned, which this model does not attempt to price. That "
            "is why the savings percentage against `always_strong` looks large.",
            "",
        ]
    return lines


def _cost_section(analysis: Mapping[str, Any]) -> list[str]:
    """Four priced strategies over the same prompts, plus the regex-only control."""
    cost = _block(analysis, "cost")
    strategies = _block(cost, "strategies")
    assumptions = _block(cost, "assumptions")
    lines = [
        "### Cost",
        "",
        _provenance_line(analysis, "cost"),
        "",
        f"Rows priced: {_na(cost.get('n_scored'))}. Tier to model: "
        + ", ".join(
            f"`{tier}` -> `{model}`"
            for tier, model in (_mapping(cost.get("model_for_tier")) or {}).items()
        )
        + ".",
        "",
    ]
    lines += _assumptions_block(assumptions)
    if not isinstance(strategies, Mapping) or not strategies:
        return [*lines, "n/a: this summary carries no `cost.strategies` block.", ""]
    order = [name for name in cost_model.STRATEGIES if name in strategies]
    order += [name for name in strategies if name not in order]
    lines += [
        "#### Strategy outcomes",
        "",
        table(
            [
                "strategy",
                "n",
                *[f"`{tier}`" for tier in cost_model.TIER_RISK_ORDER],
                "cloud calls",
                "tier accuracy",
                "UNSAFE",
            ],
            [
                [
                    f"`{name}`",
                    _na(_dig(strategies, name, "n")),
                    *[
                        _tier_count(_dig(strategies, name, "tier_counts", default=None), tier)
                        for tier in cost_model.TIER_RISK_ORDER
                    ],
                    _na(_dig(strategies, name, "cloud_calls")),
                    pct(_dig(strategies, name, "tier_accuracy")),
                    _na(_dig(strategies, name, "unsafe_errors")),
                ]
                for name in order
            ],
            aligns=["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
        ),
        "",
        "`gate_only` is the local hard gate plus always-cheap: no model classification anywhere. "
        "`always_strong` and `always_cheap` are the trivial bounds. Accuracy and UNSAFE are printed "
        "next to the tier mix because a cheap strategy that is wrong is not cheap.",
        "",
        "#### Strategy prices (assumption-derived)",
        "",
        table(
            [
                "strategy",
                "estimated input tokens",
                "assumed output tokens",
                "total cost",
                "input-only cost",
                "cost per request",
                "input-only per request",
            ],
            [
                [
                    f"`{name}`",
                    num(_dig(strategies, name, "input_tokens"), 1),
                    num(_dig(strategies, name, "output_tokens"), 1),
                    usd(_dig(strategies, name, "total_cost_usd")),
                    usd(_dig(strategies, name, "input_only_cost_usd")),
                    usd(_dig(strategies, name, "cost_per_request_usd"), 6),
                    usd(_dig(strategies, name, "input_only_cost_per_request_usd"), 6),
                ]
                for name in order
            ],
            aligns=["---", "---:", "---:", "---:", "---:", "---:", "---:"],
        ),
        "",
        "Both totals are printed because the assumed output length is the weakest assumption here. "
        "`input-only` removes it entirely and preserves every ranking; anyone who distrusts the "
        "constant can use that column instead.",
        "",
    ]
    for baseline_key, baseline_name in (("vs_always_strong", "always_strong"), ("vs_gate_only", "gate_only")):
        comparison = _block(cost, baseline_key)
        if not isinstance(comparison, Mapping) or not comparison:
            lines += [f"n/a: no `cost.{baseline_key}` block.", ""]
            continue
        lines += [
            f"#### Versus `{baseline_name}`",
            "",
            table(
                ["strategy", "total cost", "delta vs baseline", "ratio", "delta %", "tier accuracy", "UNSAFE"],
                [
                    [
                        f"`{name}`",
                        usd(_dig(comparison, name, "total_cost_usd")),
                        _signed_usd(_dig(comparison, name, "vs_baseline_usd")),
                        _ratio(_dig(comparison, name, "vs_baseline_ratio")),
                        _pct_units(_dig(comparison, name, "vs_baseline_pct")),
                        pct(_dig(comparison, name, "tier_accuracy")),
                        _na(_dig(comparison, name, "unsafe_errors")),
                    ]
                    for name in order
                    if name in comparison
                ],
                aligns=["---", "---:", "---:", "---:", "---:", "---:", "---:"],
            ),
            "",
        ]
    lines += [
        "A negative `delta %` means cheaper than the baseline. Read it together with the UNSAFE column: "
        "`always_cheap` and `gate_only` are both cheaper than routing, and both leak more. Cost saved "
        "per privacy violation bought is not a trade this project is willing to make silently.",
        "",
    ]
    lines += _negative_control_block(_dig(cost, "negative_control", default=None))
    return lines


def _negative_control_block(control: Any) -> list[str]:
    """Where the router and the regex gate disagree about *correctness*, both ways."""
    lines = ["#### Negative control: why not just the regex gate?", ""]
    nc = _mapping(control)
    if not nc:
        return [*lines, "n/a: this summary carries no `cost.negative_control` block.", ""]
    name_a = str(nc.get("a", "a"))
    name_b = str(nc.get("b", "b"))
    lines += [
        "Counted in both directions, so the comparison cannot be tilted by reporting only the side "
        "that flatters the router.",
        "",
        table(
            ["comparison", "rows"],
            [
                ["rows compared", _na(nc.get("n"))],
                [f"both `{name_a}` and `{name_b}` correct", _na(nc.get("both_correct"))],
                ["both wrong", _na(nc.get("both_wrong"))],
                [f"`{name_a}` correct, `{name_b}` wrong", _na(nc.get(f"{name_a}_only_correct"))],
                [f"`{name_b}` correct, `{name_a}` wrong", _na(nc.get(f"{name_b}_only_correct"))],
            ],
            aligns=["---", "---:"],
        ),
        "",
    ]
    for winner, loser in ((name_a, name_b), (name_b, name_a)):
        examples = _iter_rows(nc.get(f"{winner}_only_examples"))
        error_key = "b_error" if winner == name_a else "a_error"
        only = nc.get(f"{winner}_only_correct")
        lines += [f"Rows where `{winner}` was right and `{loser}` was wrong:", ""]
        if not examples:
            lines += ["n/a: no example rows recorded.", ""]
            continue
        if only is not None and len(examples) != int(only):
            lines += [
                f"The summary lists {len(examples)} example rows against a count of {_na(only)}, so the "
                "list is a subset and the breakdown below covers only the listed rows.",
                "",
            ]
        counts: Counter[str] = Counter(str(ex.get(error_key, "n/a")) for ex in examples)
        lines += [
            table(
                [f"`{loser}` error kind", "rows"],
                [[f"`{kind}`", count] for kind, count in counts.most_common()],
                aligns=["---", "---:"],
            ),
            "",
        ]
        dominant = counts.most_common(1)
        ids = ", ".join(f"`{ex.get('id')}`" for ex in examples[:_MAX_LISTED_EXAMPLES])
        lines += [
            f"`{loser}`'s misses here are dominated by `{dominant[0][0]}` ({dominant[0][1]} of "
            f"{len(examples)} listed rows). Example ids ({len(examples)} rows in total): {ids}.",
            "",
        ]
    return lines


def _latency_section(analysis: Mapping[str, Any]) -> list[str]:
    """Measured routing overhead, and the assumption its percentage is divided by."""
    latency = _block(analysis, "latency")
    if not isinstance(latency, Mapping) or not latency:
        return ["### Latency", "", "n/a: this summary carries no `latency` block.", ""]
    completion = latency.get("representative_completion_ms")
    overhead = _mapping(latency.get("overhead_pct_of_completion")) or {}
    backend = _mapping(latency.get("backend_latency_ms")) or {}
    measures = [
        ("backend decision call only", "backend_latency_ms"),
        ("router total (gate, redaction, features, merge, escalation, policy)", "router_total_latency_ms"),
        ("gate-skipped rows (no backend call at all)", "gate_skipped_latency_ms"),
    ]
    lines = [
        "### Latency",
        "",
        _provenance_line(analysis, "latency"),
        "",
        f"Real backend calls measured: {_na(latency.get('n_real_backend_calls'))}. "
        f"Gate-skipped rows: {_na(latency.get('n_gate_skipped'))}.",
        "",
        table(
            ["measure", "n", "mean ms", "min ms", "p50 ms", "p95 ms", "p99 ms", "max ms"],
            [
                [
                    label,
                    _na(_dig(latency, key, "n")),
                    _ms(_dig(latency, key, "mean_ms")),
                    _ms(_dig(latency, key, "min_ms")),
                    _ms(_dig(latency, key, "p50_ms")),
                    _ms(_dig(latency, key, "p95_ms")),
                    _ms(_dig(latency, key, "p99_ms")),
                    _ms(_dig(latency, key, "max_ms")),
                ]
                for label, key in measures
            ],
            aligns=["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
        ),
        "",
        "Gate skips are kept out of the backend-latency sample on purpose: their near-zero cost is a "
        "design feature, and averaging it into the decision call would understate the tax on "
        "everything else.",
        "",
        f"**Assumption, not a measurement:** `representative_completion_ms` = {_ms(completion, 1)} ms. "
        "Every percentage in the next table is that measured overhead divided by this invented "
        "denominator. Real completion latency runs from a few hundred milliseconds for a short flash "
        "call to tens of seconds for a long frontier generation, so treat the column as an order of "
        "magnitude.",
        "",
        table(
            ["percentile of the backend decision call", "added ms", "% of the assumed completion"],
            [
                [label, _ms(backend.get(ms_key)), _pct_units(overhead.get(pct_key), 2)]
                for label, ms_key, pct_key in (
                    ("p50", "p50_ms", "p50"),
                    ("p95", "p95_ms", "p95"),
                    ("mean", "mean_ms", "mean"),
                )
            ],
            aligns=["---", "---:", "---:"],
        ),
        "",
    ]
    mean_backend = _dig(latency, "backend_latency_ms", "mean_ms")
    if mean_backend is not None and float(mean_backend) == 0.0 and latency.get("n_real_backend_calls"):
        lines += [
            "Every value in the backend-latency row is 0.0 ms because this backend is in-process and "
            "performs no I/O: the row measures nothing and must not be compared with a real backend's. "
            "The router-total row is still a real measurement of everything the router does around the "
            "call.",
            "",
        ]
    note = latency.get("note")
    if note:
        lines += [f"> {note}", ""]
    return lines


def _cache_section(analysis: Mapping[str, Any]) -> list[str]:
    """The two-pass cache replay, when the run performed one."""
    cache = _dig(analysis, "cache", default=None)
    lines = ["### Cache", ""]
    block = _mapping(cache)
    if not block:
        reused = _dig(analysis, "run_info", "reused")
        reason = (
            "the `--reuse` path re-analyses saved rows and does not replay the cache"
            if reused
            else "the run recorded no cache block"
        )
        return [*lines, f"n/a: {reason}.", ""]
    passes = _iter_rows(block.get("passes"))
    if passes:
        lines += [
            table(
                [
                    "pass",
                    "backend",
                    "enabled",
                    "hits",
                    "misses",
                    "hit rate",
                    "evictions",
                    "entries",
                    "max entries",
                    "ttl s",
                    "p50 ms",
                    "p95 ms",
                ],
                [
                    [
                        _na(entry.get("pass")),
                        _na(_dig(entry, "cache_stats", "backend")),
                        _yesno(_dig(entry, "cache_stats", "enabled")),
                        _na(_dig(entry, "cache_stats", "hits")),
                        _na(_dig(entry, "cache_stats", "misses")),
                        pct(_dig(entry, "cache_stats", "hit_rate")),
                        _na(_dig(entry, "cache_stats", "evictions")),
                        _na(_dig(entry, "cache_stats", "size")),
                        _na(_dig(entry, "cache_stats", "max_entries")),
                        num(_dig(entry, "cache_stats", "ttl_seconds"), 1),
                        _ms(_dig(entry, "latency", "p50_ms"), 3),
                        _ms(_dig(entry, "latency", "p95_ms"), 3),
                    ]
                    for entry in passes
                ],
                aligns=["---:", "---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
            ),
            "",
            f"Warm pass: {_na(block.get('warm_pass_hits'))} hits, {_na(block.get('warm_pass_misses'))} "
            f"misses, hit rate {pct(block.get('warm_pass_hit_rate'))}.",
            "",
        ]
    else:
        lines += ["n/a: no cache passes recorded.", ""]
    note = block.get("note")
    if note:
        lines += [f"> {note}", ""]
    return lines


# --------------------------------------------------------------------------- #
# Cross-backend comparison, reproduction, and the limits of what was measured
# --------------------------------------------------------------------------- #
def _mock_vs_jev(summary: Mapping[str, Any]) -> str:
    """What paying for a calibrated decision model buys, as a difference."""
    gap = _mapping(summary.get("mock_vs_jev")) or {}
    lines = [
        "## What the cloud bootstrap buys: real backend vs offline mock",
        "",
        "The offline `MockBackend` is a deterministic keyword-and-shape reader with no model behind it. "
        "It is the floor, not a competitor: if the real backend cannot beat it, the dataset is "
        "measuring keyword matching and the whole exercise is measuring nothing.",
        "",
    ]
    if not gap:
        return "\n".join([*lines, "n/a: this summary carries no `mock_vs_jev` block.", ""])
    if not gap.get("available"):
        reason = gap.get("reason")
        return "\n".join([*lines, f"n/a: {_na(reason)}. Only one backend was part of this run.", ""])
    heads = _mapping(gap.get("heads")) or {}
    rows: list[list[Any]] = [
        ["tier accuracy", pct(gap.get("tier_accuracy_jev")), pct(gap.get("tier_accuracy_mock")),
         signed(gap.get("tier_accuracy_gap"))],
        ["UNSAFE (privacy violations)", _na(gap.get("unsafe_errors_jev")), _na(gap.get("unsafe_errors_mock")),
         _delta_int(gap.get("unsafe_errors_gap"))],
        ["PII F1 (raw noul)", num(gap.get("pii_f1_jev"), 4), num(gap.get("pii_f1_mock"), 4),
         signed(_diff(gap.get("pii_f1_jev"), gap.get("pii_f1_mock")))],
    ]
    for head in ("complexity", "sensitivity", "domain"):
        block = _mapping(heads.get(head)) or {}
        rows += [
            [f"`{head}` macro F1", num(block.get("macro_f1_jev"), 4), num(block.get("macro_f1_mock"), 4),
             signed(block.get("macro_f1_gap"))],
            [f"`{head}` accuracy", pct(block.get("accuracy_jev")), pct(block.get("accuracy_mock")),
             signed(_diff(block.get("accuracy_jev"), block.get("accuracy_mock")))],
            [f"`{head}` ECE", num(block.get("ece_jev"), 4), num(block.get("ece_mock"), 4),
             num(_diff(block.get("ece_jev"), block.get("ece_mock")), 4)],
        ]
    lines += [
        table(["metric", "real backend (`jev`)", "offline mock (`mock`)", "gap (jev - mock)"], rows,
              aligns=["---", "---:", "---:", "---:"]),
        "",
        "The gap column is `jev` minus `mock`. For accuracy and F1 a positive gap is the point of the "
        "project. For UNSAFE and for ECE, lower is better, so a *negative* gap is the good direction. "
        "Units: accuracy and F1 gaps are percentage points; ECE gaps are plain differences on the 0-1 "
        "error scale; the UNSAFE gap is a row count.",
        "",
    ]
    lines += _ece_caveat(heads)
    return "\n".join(lines)


def _diff(a: Any, b: Any) -> float | None:
    """``a - b`` when both are numbers, else ``None`` (rendered `n/a`)."""
    if a is None or b is None:
        return None
    try:
        return round(float(a) - float(b), 6)
    except (TypeError, ValueError):
        return None


def _ece_caveat(heads: Mapping[str, Any]) -> list[str]:
    """Name every head where the mock calibrates better than the real backend.

    WHY this is derived rather than asserted once: a lower ECE alongside a lower
    accuracy is not a win, it is a head that is honest about being useless. The
    report has to say so where it happens, and it has to stop saying so if a
    future run does not produce it.
    """
    flips = []
    for head in ("complexity", "sensitivity", "domain"):
        block = _mapping(heads.get(head)) or {}
        ece_jev, ece_mock = block.get("ece_jev"), block.get("ece_mock")
        acc_jev, acc_mock = block.get("accuracy_jev"), block.get("accuracy_mock")
        if None in (ece_jev, ece_mock, acc_jev, acc_mock):
            continue
        if float(ece_mock) < float(ece_jev):
            flips.append((head, ece_jev, ece_mock, acc_jev, acc_mock))
    if not flips:
        return ["The real backend has the lower ECE on every choice head in this run.", ""]
    lines = ["One result here is easy to misread:", ""]
    for head, ece_jev, ece_mock, acc_jev, acc_mock in flips:
        lines += [
            f"* On `{head}` the offline mock has the **lower** ECE ({num(ece_mock, 4)} vs "
            f"{num(ece_jev, 4)}) while its accuracy is also lower ({pct(acc_mock)} vs {pct(acc_jev)}). "
            "That is not a better model; it is a head that is accurately unsure. Calibration measures "
            "whether a stated confidence matches an observed rate, and a reader that says \"I am 40% "
            "sure\" and is right 40% of the time is perfectly calibrated and perfectly useless. This "
            "is why every ECE in this report is printed next to accuracy and Brier skill.",
        ]
    return [*lines, ""]


def _reproduce(summary: Mapping[str, Any]) -> str:
    """How to regenerate this exact file, and what would change if you did."""
    meta = _block(summary, "meta")
    argv = _dig(meta, "argv", default=None)
    command = "python evals/run_eval.py"
    if isinstance(argv, Sequence) and not isinstance(argv, str) and argv:
        command = f"{command} {shlex.join(str(a) for a in argv)}"
    lines = [
        "## Reproducing this report",
        "",
        "```console",
        "# the run that produced these numbers",
        command,
        "",
        "# re-analyse the persisted per-row JSONL and re-render, at zero API cost",
        _reuse_command(summary),
        "```",
        "",
        "`--reuse` is the idempotence switch: it reads the JSONL already in the results directory "
        "instead of calling any backend, so the report can be re-rendered forever without spending "
        "anything. It needs no API key. Two consequences are visible above: `backend_stats` and the "
        "`cache` block are properties of a live run and show as `n/a` on a reused one.",
        "",
        "Rows are reproduced from the same recorded answers, not re-sampled: the A/B arms are an exact "
        "counterfactual over one pass. The question set fingerprint is per backend in the provenance "
        "table (`questions_sha256`).",
        "",
        "The evaluation itself is deterministic given the same recorded answers, but the real backend is "
        "not: a fresh run against the live API re-samples every judgement, so the numbers will move. "
        "The dataset sha256_16 and the policy sha256_16 in the provenance table are what make two runs "
        "comparable.",
        "",
    ]
    return "\n".join(lines)


def _reuse_command(summary: Mapping[str, Any]) -> str:
    """The ``--reuse`` line for the backends this summary actually contains.

    WHY derived: ``--reuse`` refuses to run when any configuration's JSONL is
    missing, so printing ``--backend both`` after a mock-only run would hand the
    reader a command that fails.
    """
    names = {name for name, _ in _backends(summary)}
    parts = ["python evals/run_eval.py"]
    if {"jev", "mock"} <= names:
        parts.append("--backend both")
    elif names:
        parts.append(f"--backend {sorted(names)[0]}")
    parts.append("--reuse")
    return " ".join(parts)


def _limits(summary: Mapping[str, Any]) -> str:
    """What this file does not measure. Every bullet is a real hole, not modesty."""
    backends = dict(_backends(summary))
    analysis = backends.get(_primary(list(backends.items())) or "", {})
    assumptions = _mapping(_dig(analysis, "cost", "assumptions", default=None)) or {}
    lines = [
        "## What this evaluation does not measure",
        "",
        "* **No completions were generated.** The dataset is prompts without reference completions, so "
        "no completion model was called and no real token count exists. Every cost is prompt-side plus "
        f"a constant assumed output of {_na(assumptions.get('assumed_output_tokens'))} tokens. "
        "Answer quality per tier is therefore *assumed* by the `expected_tier` label, not measured.",
        f"* **The decision API's own price is excluded** "
        f"(`jev_decision_api_cost_included` = {_yesno(assumptions.get('jev_decision_api_cost_included'))}). "
        "It is not published, and inventing it would be exactly the failure the assumption tables "
        "exist to prevent. The call count is reported instead, so anyone with a price sheet can add "
        "the term.",
        f"* **Token counts are a character-count proxy** at "
        f"{_na(assumptions.get('chars_per_token'))} characters per token. That is the conventional "
        "English approximation and is wrong for code and for Hungarian or German text, both of which "
        "this dataset contains. It is wrong by a similar factor for every strategy, so it cancels in "
        "comparisons and not in totals.",
        f"* **Prices are illustrative** "
        f"(`prices_are_illustrative` = {_yesno(assumptions.get('prices_are_illustrative'))}). They are "
        "not a quote and not live data.",
        f"* **`representative_completion_ms` = {_ms(_dig(analysis, 'latency', 'representative_completion_ms'), 1)} ms "
        "is an assumption.** Every overhead percentage inherits it.",
        "* **`expected_tier` is an author-assigned label**, one per prompt. A different labeller would "
        "move tier accuracy, and the ambiguous subset in the by-difficulty table is where that "
        "judgement is weakest. Accuracy here measures agreement with a documented labelling policy, "
        "not agreement with the world.",
        "* **Denominators differ between sections by design.** Tier accuracy, cost and latency come "
        "from the shipped policy configuration; component labels and calibration come from the "
        "configuration that lets the backend answer the most rows. Each section names its own "
        "configuration and its own `n`.",
        "* **Excluded rows are counted, not scored.** Errored and degraded rows sit outside the "
        "accuracy denominator and are printed in the headline table. A run that quietly scored fewer "
        "rows than it fetched is how benchmark numbers become fiction.",
        "* **UNSAFE is a dataset-labelled notion of \"must not leave the building\".** It is derived "
        "from the `expected_tier` label, not from a legal review of your jurisdiction, your contracts "
        "or your residency obligations.",
        "* **This is a single sample of a nondeterministic backend.** No confidence interval is "
        "reported, because the harness records one pass. Differences smaller than a few rows between "
        "two live runs should be treated as noise.",
        "",
        "---",
        "",
        "Rendered by `evals/report.py` from `summary.json`. Every figure in this file is read from that "
        "summary at render time; the renderer holds no results of its own.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def render(summary: Mapping[str, Any], *, calibration_reports: Mapping[str, Any] | None = None) -> str:
    """Render the whole ``REPORT.md`` as a markdown string.

    ``summary`` is the dict ``run_eval.main`` built (or one loaded back from
    ``summary.json``). ``calibration_reports`` optionally carries the live
    :class:`calibration.CalibrationReport` objects per backend and head; when it
    is absent the same tables are rebuilt from the serialised bins, so the report
    is renderable from the persisted file alone.

    The contract is deliberately weak on input and strict on output: **any**
    subset of the summary renders. A missing subtree costs one ``n/a`` and never
    an exception, because the renderer runs at the end of a long, possibly
    expensive harness pass, and losing the report to a KeyError in a cosmetic
    table would throw away the measurement with it.
    """
    data: Mapping[str, Any] = summary if isinstance(summary, Mapping) else {}
    sections = [
        _title(data),
        _headline(data),
        _provenance(data),
        _policy_under_test(data),
        _escalation_tradeoff(data),
        _unsafe_error_analysis(data),
    ]
    for name, analysis in _backends(data):
        sections.append(_backend_detail(name, analysis, calibration_reports))
    sections += [_mock_vs_jev(data), _reproduce(data), _limits(data)]
    body = "\n\n".join(section for section in sections if section)
    # Section builders end with a blank line for readability in the source, and
    # the join adds another; collapsing the run keeps the markdown tidy without
    # making every builder responsible for its own trailing whitespace.
    return re.sub(r"\n{3,}", "\n\n", body).strip() + "\n"


__all__ = [
    "COMPLEXITY_LABELS",
    "DOMAIN_LABELS",
    "SENSITIVITY_LABELS",
    "TIER_LABELS",
    "confusion_md",
    "num",
    "pct",
    "per_class_md",
    "render",
    "signed",
    "table",
    "usd",
]
