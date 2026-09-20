"""Tests for ``evals/report.py``, the renderer behind ``evals/results/REPORT.md``.

Lives in ``tests/`` rather than next to the module so that the documented
``pytest`` run collects it: a renderer test nobody runs is not a test.

Two failure modes matter here, and each gets its own test:

* **A renderer that raises loses the measurement with it.** It runs at the end of
  a long harness pass, against summaries of every vintage, including ones where a
  whole subtree is missing (``--reuse`` carries no ``backend_stats`` and no
  ``cache``). Every hostile shape below must still produce a report.
* **A renderer that carries its own numbers stops being a report.** The suite
  asserts that a summary with no results in it produces a file with no digits in
  it, so a future "helpful" constant cannot slip into the prose unnoticed.

The real ``evals/results/summary.json`` is deliberately never loaded: a 267 KB
fixture would make these tests slow, unreadable, and silently dependent on
whatever the last run happened to produce. Every input here is hand-built and
small enough to audit by eye.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pytest

EVALS_DIR = Path(__file__).resolve().parent.parent / "evals"
if str(EVALS_DIR) not in sys.path:
    # The eval harness is a directory of sibling scripts, not an importable
    # package: `evals/report.py` imports `evals/calibration.py` as a top-level
    # module, so the directory itself has to be on sys.path. Same shim
    # `evals/run_eval.py` uses when it runs as a script.
    sys.path.insert(0, str(EVALS_DIR))

import calibration  # noqa: E402  (the shim above has to run first)
import report  # noqa: E402

#: The section headings a complete report must contain. Asserted as a list rather
#: than spot-checked, because a section that silently stops rendering is exactly
#: the kind of regression a reader would not notice.
EXPECTED_HEADINGS = (
    "# jev-route evaluation report",
    "## Headline",
    "## What was measured",
    "## The policy under test",
    "## The tradeoff: uncertainty escalation on vs off",
    "## Error analysis: every UNSAFE row",
    "### Routing decisions",
    "### Component heads",
    "### Calibration",
    "### Cost",
    "### Latency",
    "### Cache",
    "## What the cloud bootstrap buys",
    "## Reproducing this report",
    "## What this evaluation does not measure",
)

#: The definition that must appear wherever UNSAFE first appears.
UNSAFE_NEEDLES = ("**UNSAFE**", "air-gapped `local` tier", "privacy violations")


# --------------------------------------------------------------------------- #
# Synthetic fixtures. Small, explicit, and free of any real result.
# --------------------------------------------------------------------------- #
def _calibration_block(head: str, *, ece: float = 0.25) -> dict[str, Any]:
    return {
        "head": head,
        "n": 4,
        "accuracy": 0.5,
        "ece": ece,
        "ece_top_probability": ece,
        "mce": 0.5,
        "brier": 0.6,
        "brier_skill": 0.1,
        "baseline_brier": 0.66,
        "n_bins": 2,
        "populated_bins": 2,
        "confidence_source": "reported",
        "mean_reported_confidence": 0.75,
        "bins": [
            {"lo": 0.0, "hi": 0.5, "count": 2, "mean_confidence": 0.25, "accuracy": 0.5, "gap": 0.25},
            {"lo": 0.5, "hi": 1.0, "count": 2, "mean_confidence": 0.75, "accuracy": 0.5, "gap": -0.25},
        ],
        "event_bins": [],
        "event_ece": 0.0,
        "coverage": [
            {"threshold": 0.0, "n": 4, "coverage": 1.0, "accuracy": 0.5},
            {"threshold": 0.5, "n": 2, "coverage": 0.5, "accuracy": 1.0},
        ],
        "notes": [],
    }


def _strategy(name: str, *, tier_counts: dict[str, int], total: float, unsafe: int) -> dict[str, Any]:
    return {
        "strategy": name,
        "n": 4,
        "tier_counts": tier_counts,
        "tier_share": {},
        "input_tokens": 100.0,
        "output_tokens": 1024.0,
        "total_cost_usd": total,
        "input_only_cost_usd": total / 10,
        "cost_per_request_usd": total / 4,
        "input_only_cost_per_request_usd": total / 40,
        "cloud_calls": sum(tier_counts.values()),
        "tier_accuracy": 0.5,
        "unsafe_errors": unsafe,
    }


def _backend(name: str = "mock", *, zero_latency: bool = False) -> dict[str, Any]:
    """One complete-but-tiny backend block, shaped like ``analyse_backend``'s output."""
    measured_ms = 0.0 if zero_latency else 12.5
    return {
        "backend": name,
        "run_started": "2026-01-01T00:00:00+00:00",
        "run_finished": "2026-01-01T00:00:01+00:00",
        "n_rows": 4,
        "model_versions_observed": [f"{name}-1.0.0"],
        "backend_stats": {"api_calls": 4, "logical_requests": 8, "memo_hits": 4, "degraded_final": 0},
        "provenance": {
            "tier_accuracy": "default",
            "cost": "default",
            "latency": "default",
            "escalation_ab": ["default", "no_escalation"],
            "component_labels": "still_classify",
            "calibration": "still_classify",
        },
        "tier": {
            "n_rows": 4,
            "n_scored": 4,
            "excluded": {"error": 0, "degraded": 0, "missing_tier": 0},
            "accuracy": 0.5,
            "confusion": {
                "local": {"local": 1, "cheap": 0, "strong": 1},
                "cheap": {"local": 0, "cheap": 1, "strong": 0},
                "strong": {"local": 1, "cheap": 0, "strong": 0},
            },
            "per_class": {
                "local": {"support": 2, "precision": 0.5, "recall": 0.5, "f1": 0.5, "accuracy": 0.5},
                "cheap": {"support": 1, "precision": 1.0, "recall": 1.0, "f1": 1.0, "accuracy": 1.0},
                "strong": {"support": 1, "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0},
                "_macro": {"macro_f1": 0.5, "accuracy": 0.5, "n": 4},
            },
            "error_kinds": {"correct": 2, "unsafe": 1, "expensive": 1, "overspend": 0, "underpowered": 0},
            "unsafe_rate": 0.25,
            "unsafe_errors": 1,
            "expensive_errors": 1,
            "per_expected_tier": {"local": {"n": 2, "correct": 1, "accuracy": 0.5, "unsafe": 1}},
            "by_difficulty": {"clear": {"n": 4, "accuracy": 0.5, "unsafe": 1}},
            "tier_distribution": {"local": 1, "cheap": 1, "strong": 2},
            "escalated_rows": 2,
            "gate_forced_rows": 1,
            "gate_blocked_rows": 0,
            "classified_rows": 4,
            "wrong_rows": [
                {
                    "id": "row-unsafe-1",
                    "expected": "local",
                    "predicted": "strong",
                    "kind": "unsafe",
                    "rule_id": "complexity.hard",
                    "labels": {"sensitivity": "confidential", "complexity": "hard"},
                    "effective": {"sensitivity": "internal", "complexity": "hard"},
                    "escalated": ["sensitivity public->internal (confidence 0.46 < 0.8)"],
                    "difficulty": "clear",
                },
                {
                    "id": "row-expensive-1",
                    "expected": "strong",
                    "predicted": "local",
                    "kind": "expensive",
                    "rule_id": "data.sensitive",
                    "labels": {"sensitivity": "internal"},
                    "effective": {"sensitivity": "confidential"},
                    "escalated": [],
                    "difficulty": "clear",
                },
            ],
        },
        "escalation_ab": {
            "enabled": {
                "accuracy": 0.5,
                "unsafe_errors": 1,
                "expensive_errors": 1,
                "error_kinds": {"correct": 2, "unsafe": 1, "expensive": 1, "overspend": 0, "underpowered": 0},
                "escalated_rows": 2,
                "tier_distribution": {"local": 1, "cheap": 1, "strong": 2},
                "n_scored": 4,
            },
            "disabled": {
                "accuracy": 0.75,
                "unsafe_errors": 2,
                "expensive_errors": 0,
                "error_kinds": {"correct": 3, "unsafe": 2, "expensive": 0, "overspend": 0, "underpowered": 0},
                "escalated_rows": 0,
                "tier_distribution": {"local": 0, "cheap": 2, "strong": 2},
                "n_scored": 4,
            },
            "enabled_minus_disabled": {"accuracy_delta": -0.25, "unsafe_delta": -1, "expensive_delta": 1},
        },
        "components": {
            "complexity": {
                "head": "complexity",
                "labels": ["trivial", "standard", "hard", "frontier"],
                "n_scored": 4,
                "n_unclassified": 0,
                "argmax_mismatch_rows": 0,
                "confusion": {"hard": {"hard": 2, "standard": 2}},
                "per_class": {
                    "hard": {"support": 2, "precision": 1.0, "recall": 1.0, "f1": 1.0, "accuracy": 1.0},
                    "_macro": {"macro_f1": 0.5, "accuracy": 0.5, "n": 4},
                },
                "macro_f1": 0.5,
                "accuracy": 0.5,
            },
            "sensitivity": {
                "head": "sensitivity",
                "labels": ["public", "internal", "confidential", "regulated"],
                "n_scored": 4,
                "n_unclassified": 0,
                "argmax_mismatch_rows": 0,
                "confusion": {"confidential": {"internal": 1}, "internal": {"internal": 1}},
                "per_class": {"_macro": {"macro_f1": 0.5, "accuracy": 0.5, "n": 4}},
                "macro_f1": 0.5,
                "accuracy": 0.5,
            },
            "pii": {
                "threshold": 0.5,
                "raw": {"n": 4, "tp": 1, "tn": 3, "fp": 0, "fn": 0, "accuracy": 1.0, "precision": 1.0,
                        "recall": 1.0, "f1": 1.0, "missed_pii": 0, "false_pii": 0},
                "effective": {"n": 4, "tp": 1, "tn": 3, "fp": 0, "fn": 0, "accuracy": 1.0, "precision": 1.0,
                              "recall": 1.0, "f1": 1.0, "missed_pii": 0, "false_pii": 0},
                "n_unclassified": 0,
                "note": "raw is the model, effective is the router.",
            },
        },
        "calibration": {head: _calibration_block(head) for head in ("complexity", "sensitivity", "domain", "pii")},
        "cost": {
            "n_scored": 4,
            "model_for_tier": {"local": "m0", "cheap": "m1", "strong": "m2"},
            "strategies": {
                "always_strong": _strategy("always_strong", tier_counts={"strong": 4}, total=1.0, unsafe=2),
                "always_cheap": _strategy("always_cheap", tier_counts={"cheap": 4}, total=0.25, unsafe=2),
                "jev_route": _strategy(
                    "jev_route", tier_counts={"local": 1, "cheap": 1, "strong": 2}, total=0.5, unsafe=1
                ),
                "gate_only": _strategy("gate_only", tier_counts={"local": 1, "cheap": 3}, total=0.2, unsafe=2),
            },
            "vs_always_strong": {
                name: {
                    "total_cost_usd": cost,
                    "vs_baseline_usd": cost - 1.0,
                    "vs_baseline_ratio": cost,
                    "vs_baseline_pct": -64.316 if name == "jev_route" else 0.0,
                    "tier_accuracy": 0.5,
                    "unsafe_errors": unsafe,
                }
                for name, cost, unsafe in (
                    ("always_strong", 1.0, 2), ("always_cheap", 0.25, 2), ("jev_route", 0.5, 1),
                    ("gate_only", 0.2, 2),
                )
            },
            "vs_gate_only": {},
            "negative_control": {
                "a": "jev_route",
                "b": "gate_only",
                "n": 4,
                "both_correct": 1,
                "both_wrong": 1,
                "jev_route_only_correct": 1,
                "gate_only_only_correct": 1,
                "jev_route_only_examples": [
                    {"id": "row-a", "expected": "local", "jev_route": "local", "gate_only": "cheap",
                     "b_error": "unsafe"}
                ],
                "gate_only_only_examples": [
                    {"id": "row-b", "expected": "cheap", "jev_route": "strong", "gate_only": "cheap",
                     "a_error": "overspend"}
                ],
            },
            "assumptions": {
                "price_table_usd_per_million_tokens": {"m2": {"input": 2.0, "output": 8.0},
                                                       "m1": {"input": 0.2, "output": 1.0},
                                                       "m0": {"input": 0.0, "output": 0.0}},
                "prices_are_illustrative": True,
                "chars_per_token": 4.0,
                "token_counts_are_char_proxy": True,
                "assumed_output_tokens": 256.0,
                "jev_decision_api_cost_included": False,
                "representative_completion_ms": 3000.0,
            },
        },
        "latency": {
            "backend": name,
            "n_real_backend_calls": 4,
            "n_gate_skipped": 0,
            "backend_latency_ms": {"n": 4, "mean_ms": measured_ms, "min_ms": measured_ms,
                                   "max_ms": measured_ms, "p50_ms": measured_ms, "p95_ms": measured_ms,
                                   "p99_ms": measured_ms},
            "router_total_latency_ms": {"n": 4, "mean_ms": 1.0, "min_ms": 0.5, "max_ms": 2.0,
                                        "p50_ms": 1.0, "p95_ms": 2.0, "p99_ms": 2.0},
            "gate_skipped_latency_ms": {"n": 0, "mean_ms": None, "min_ms": None, "max_ms": None,
                                        "p50_ms": None, "p95_ms": None},
            "representative_completion_ms": 3000.0,
            "overhead_pct_of_completion": {"p50": 0.417, "p95": 0.417, "mean": 0.417},
            "note": "backend_latency is the decision call only.",
        },
        "cache": {
            "passes": [{"pass": 1, "cache_stats": {"backend": "memory", "enabled": True, "hits": 0,
                                                   "misses": 4, "hit_rate": 0.0, "evictions": 0, "size": 4,
                                                   "max_entries": 64, "ttl_seconds": 60.0},
                        "latency": {"n": 4, "p50_ms": 0.2, "p95_ms": 0.4}}],
            "warm_pass_hit_rate": 0.5,
            "warm_pass_hits": 2,
            "warm_pass_misses": 2,
            "note": "replay only.",
        },
        "run_info": {"reused": False, "paths": {"default": "/tmp/x.jsonl"}, "concurrency": 1,
                     "questions_sha256": None},
    }


def _summary(**overrides: Any) -> dict[str, Any]:
    """A complete two-backend summary, with any top-level key overridable."""
    base: dict[str, Any] = {
        "meta": {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "dataset": {"path": "evals/data/labeled_prompts.jsonl", "sha256_16": "aaaabbbbccccdddd",
                        "n_rows": 4},
            "policy": {"path": "policies/default.yaml", "sha256_16": "1111222233334444", "version": 1,
                       "tiers": {"local": ["m0"], "cheap": ["m1"], "strong": ["m2"]},
                       "on_uncertain": {"sensitivity_confidence_below": 0.8, "sensitivity_bump_levels": 1,
                                        "complexity_confidence_below": 0.7, "complexity_bump_levels": 1},
                       "gate_on_force_local": "skip_backend", "failure_mode": "fail_closed"},
            "git_commit": "abc1234",
            "argv": ["--backend", "both"],
            "typesafe_api_key_present": True,
            "calibration_bins": 2,
            "concurrency": 4,
            "reuse": False,
        },
        "backends": {"jev": _backend("jev"), "mock": _backend("mock", zero_latency=True)},
        "mock_vs_jev": {
            "available": True,
            "tier_accuracy_jev": 0.5,
            "tier_accuracy_mock": 0.25,
            "tier_accuracy_gap": 0.25,
            "unsafe_errors_jev": 1,
            "unsafe_errors_mock": 2,
            "unsafe_errors_gap": -1,
            "pii_f1_jev": 1.0,
            "pii_f1_mock": 0.0,
            "heads": {"complexity": {"macro_f1_jev": 0.5, "macro_f1_mock": 0.25, "macro_f1_gap": 0.25,
                                     "accuracy_jev": 0.5, "accuracy_mock": 0.25, "ece_jev": 0.25,
                                     "ece_mock": 0.5},
                      "sensitivity": {"macro_f1_jev": 0.5, "macro_f1_mock": 0.25, "macro_f1_gap": 0.25,
                                      "accuracy_jev": 0.5, "accuracy_mock": 0.25, "ece_jev": 0.25,
                                      "ece_mock": 0.1},
                      "domain": {"macro_f1_jev": 0.5, "macro_f1_mock": 0.25, "macro_f1_gap": 0.25,
                                 "accuracy_jev": 0.5, "accuracy_mock": 0.25, "ece_jev": 0.25,
                                 "ece_mock": 0.5}},
        },
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# The contract: a complete summary renders, and every section is there
# --------------------------------------------------------------------------- #
def test_minimal_summary_renders_every_section() -> None:
    out = report.render(_summary())
    assert isinstance(out, str)
    assert out.endswith("\n")
    for heading in EXPECTED_HEADINGS:
        assert heading in out, f"missing section: {heading}"
    for needle in UNSAFE_NEEDLES:
        assert needle in out, f"UNSAFE is not defined where it first appears: {needle!r}"


def test_both_backends_get_their_own_detail_section() -> None:
    out = report.render(_summary())
    assert "## Backend detail: `jev`" in out
    assert "## Backend detail: `mock`" in out
    # The real backend is quoted in the lead-in, not the mock.
    assert "(backend `jev`)" in out


def test_render_is_a_pure_function_of_the_summary() -> None:
    summary = _summary()
    assert report.render(summary) == report.render(summary)


# --------------------------------------------------------------------------- #
# The contract: nothing raises, and missing means "n/a", never zero
# --------------------------------------------------------------------------- #
HOSTILE_SUMMARIES = {
    "empty": {},
    "meta only": {"meta": {"generated_at": "now"}},
    "no backends": {"meta": {}, "backends": {}},
    "hollow backend": {"meta": {}, "backends": {"jev": {}}},
    "None subtrees": {"meta": None, "backends": {"jev": None}, "mock_vs_jev": None},
    "scalars where mappings belong": {"meta": 5, "backends": "nope", "mock_vs_jev": []},
    "backend block is a scalar": {"backends": {"jev": 7, "mock": {"tier": "x"}}},
    "wrong_rows is a string": {"backends": {"jev": {"tier": {"wrong_rows": "oops", "unsafe_errors": 3}}}},
    "calibration bins are malformed": {
        "backends": {"jev": {"calibration": {"sensitivity": {"bins": [{"nope": 1}], "coverage": [{}]}}}}
    },
    "cost subtrees are malformed": {
        "backends": {"jev": {"cost": {"strategies": {"x": "y"}, "assumptions": [],
                                      "negative_control": {"a": "p", "b": "q"}}}}
    },
    "latency subtrees are malformed": {
        "backends": {"jev": {"latency": {"backend_latency_ms": 3, "overhead_pct_of_completion": "z"}}}
    },
    "cache passes is a string": {"backends": {"jev": {"cache": {"passes": "no"}}}},
    "single backend, no ablation": {
        "mock_vs_jev": {"available": False, "reason": "only one backend was run"},
        "backends": {"mock": {"tier": {"accuracy": None, "unsafe_errors": None}}},
    },
    "not a mapping at all": None,
}


@pytest.mark.parametrize("label", sorted(HOSTILE_SUMMARIES))
def test_partial_summaries_render_instead_of_raising(label: str) -> None:
    out = report.render(HOSTILE_SUMMARIES[label])
    assert isinstance(out, str) and out.startswith("# jev-route evaluation report")
    for heading in ("## Headline", "## What this evaluation does not measure"):
        assert heading in out


def test_missing_fields_read_as_na_and_zeros_read_as_zeros() -> None:
    summary = _summary()
    # A live-run field the reuse path cannot supply.
    del summary["backends"]["jev"]["backend_stats"]
    summary["backends"]["jev"]["run_info"] = {"reused": True, "paths": {}}
    out = report.render(summary)
    assert "| backend API calls | n/a |" in out
    assert "--reuse" in out
    # Zero is a measurement, not an absence: the excluded-row counters stay 0.
    assert "| excluded: errored | 0 |" in out


def test_reuse_run_says_what_it_cannot_show() -> None:
    """`--reuse` carries no cache block; the report must say why instead of showing a hole."""
    summary = _summary()
    summary["meta"]["reuse"] = True
    summary["backends"]["mock"]["cache"] = {}
    summary["backends"]["mock"]["run_info"] = {"reused": True, "paths": {}}
    out = report.render(summary)
    assert "re-analyses the persisted per-row JSONL" in out
    assert "n/a: the `--reuse` path re-analyses saved rows and does not replay the cache." in out


def test_absent_tier_count_is_zero_not_na() -> None:
    """``always_strong`` has no ``local`` key at all; that is a count of zero."""
    summary = _summary()
    out = report.render(summary)
    assert "| `always_strong` | 4 | 0 | 0 | 4 | 4 | 50.0% | 2 |" in out


# --------------------------------------------------------------------------- #
# The contract: the renderer owns no results of its own
# --------------------------------------------------------------------------- #
def test_an_empty_summary_produces_no_numbers() -> None:
    """Nothing in the prose may carry a result.

    The only digits allowed are inside the literal field name ``sha256_16``. If
    this test starts failing, a number was written into the renderer instead of
    read from the summary -- which is how a report turns into an advertisement
    the first time the evaluation is re-run.
    """
    out = report.render({})
    # `sha256` / `sha256_16` are field names, not results.
    stripped = re.sub(r"sha256(?:_16)?", "", out)
    assert not re.search(r"[0-9]", stripped), f"hardcoded number in prose: {stripped!r}"
    assert "n/a" in out


def test_every_headline_number_comes_from_the_summary() -> None:
    summary = _summary()
    out = report.render(summary)
    assert "| `jev` | 50.0% (n=4) | 1 | 25.00% | 0.2500 | -64.3% | 12.5 / 12.5 / 12.5 | 0 / 0 / 0 |" in out
    assert "**1 UNSAFE rows** out of 4 scored (25.00% of the run)" in out


# --------------------------------------------------------------------------- #
# Units, direction words, and the two renderings of calibration
# --------------------------------------------------------------------------- #
def test_percent_unit_fields_are_not_rescaled() -> None:
    """``vs_baseline_pct`` arrives pre-multiplied; ``pct()`` would inflate it 100x."""
    out = report.render(_summary())
    assert "-64.3%" in out
    assert "-6431" not in out
    assert "| p50 | 12.5 | 0.42% |" in out


def test_signed_dollar_deltas_put_the_sign_before_the_currency() -> None:
    out = report.render(_summary())
    assert "-$0.5000" in out
    assert "$-0.5000" not in out


def test_ab_verdict_follows_the_direction_of_the_numbers() -> None:
    """The tradeoff sentence is derived, so it must vanish when the shape does."""
    tradeoff = report.render(_summary())
    assert "*raises* tier accuracy from 50.0% to 75.0%" in tradeoff
    # The whole delta row is derived: the summary only persists three of these
    # deltas, so the rest are the difference of two counts it does report.
    delta_row = (
        "| **delta (enabled - disabled)** | 0 | -25.0 pp | -1 | +1 | 0 | 0 | +2 | +1 / -1 / 0 |"
    )
    assert delta_row in tradeoff
    assert "*raises* UNSAFE privacy violations from 1 to 2" in tradeoff
    assert "**That is the argument for calibrated routing in one line.**" in tradeoff

    # Same direction on both axes, on every backend: no tradeoff, so no claim.
    summary = _summary()
    for data in summary["backends"].values():
        data["escalation_ab"]["disabled"]["accuracy"] = 0.25
        data["escalation_ab"]["disabled"]["unsafe_errors"] = 0
    honest = report.render(summary)
    assert "*lowers* tier accuracy from 50.0% to 25.0%" in honest
    assert "*lowers* UNSAFE privacy violations from 1 to 0" in honest
    assert "**That is the argument for calibrated routing in one line.**" not in honest


def test_unsafe_rows_are_itemised_by_id_and_split_by_cause() -> None:
    out = report.render(_summary())
    assert "`row-unsafe-1`" in out
    assert "`row-expensive-1`" not in out.split("## Error analysis")[1].split("## Backend detail")[0]
    assert "Of the 1 UNSAFE rows in this run:" in out
    assert "sensitivity public->internal (confidence 0.46 < 0.8)" in out
    confident_row = (
        "| confidently wrong: no sensitivity bump recorded, so the reported confidence cleared "
        "the floor | 0 |"
    )
    assert confident_row in out
    assert "raw_answers.sensitivity.confidence" in out


def test_live_calibration_reports_cannot_override_the_summary() -> None:
    """The committed report has to be re-renderable from ``summary.json`` alone.

    ``CalibrationReport.to_dict`` rounds to six decimals, so a live object can
    disagree with the persisted dict in the last printed digit. The dict wins:
    "re-render the summary and diff the file" is only a valid check if the
    renderer reads nothing else.
    """
    summary = _summary()
    hostile = {
        name: {
            head: calibration.CalibrationReport.from_dict({**block, "ece": 0.9999, "mce": 0.9999})
            for head, block in data["calibration"].items()
        }
        for name, data in summary["backends"].items()
    }
    assert report.render(summary, calibration_reports=hostile) == report.render(summary)
    assert "0.9999" not in report.render(summary, calibration_reports=hostile)


def test_live_reports_are_the_fallback_when_the_persisted_bins_are_unusable() -> None:
    summary = _summary()
    live = {
        name: {head: calibration.CalibrationReport.from_dict(block) for head, block in data["calibration"].items()}
        for name, data in summary["backends"].items()
    }
    for data in summary["backends"].values():
        data["calibration"] = {head: {"bins": [{"nope": 1}]} for head in data["calibration"]}
    out = report.render(summary, calibration_reports=live)
    assert "ECE=0.2500" in out


def test_calibration_renders_from_dicts_and_from_live_reports() -> None:
    """The live objects and the persisted dicts must produce the same tables."""
    summary = _summary()
    from_dicts = report.render(summary)
    live = {
        name: {head: calibration.CalibrationReport.from_dict(block) for head, block in data["calibration"].items()}
        for name, data in summary["backends"].items()
    }
    from_objects = report.render(summary, calibration_reports=live)
    assert from_dicts == from_objects
    assert "ECE=0.2500" in from_dicts
    assert "| bin | n | predicted | observed | gap |" in from_dicts
    assert "| confidence >= | n | coverage | accuracy | delta vs all |" in from_dicts


def test_zero_backend_latency_is_flagged_as_measuring_nothing() -> None:
    out = report.render(_summary())
    assert "performs no I/O" in out


def test_mock_comparison_says_which_direction_is_better() -> None:
    out = report.render(_summary())
    assert "| tier accuracy | 50.0% | 25.0% | +25.0 pp |" in out
    assert "| UNSAFE (privacy violations) | 1 | 2 | -1 |" in out
    # `sensitivity` is the head where the mock's ECE is lower in this fixture.
    assert "On `sensitivity` the offline mock has the **lower** ECE" in out


def test_single_backend_run_reports_the_gap_as_unavailable() -> None:
    summary = _summary()
    summary["backends"] = {"mock": summary["backends"]["mock"]}
    summary["mock_vs_jev"] = {"available": False, "reason": "only one backend was run"}
    out = report.render(summary)
    assert "n/a: only one backend was run." in out
    # The reuse command must not promise a backend whose rows are absent.
    assert "python evals/run_eval.py --backend mock --reuse" in out


def test_assumptions_are_labelled_next_to_the_dollars() -> None:
    out = report.render(_summary())
    assert "**These are estimates, not measurements.**" in out
    assert "**Assumption, not a measurement:** `representative_completion_ms`" in out
    assert "| `prices_are_illustrative` | yes |" in out
    assert "| `jev_decision_api_cost_included` | no |" in out
