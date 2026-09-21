"""The backtest harness: oracle replay, cost accounting, CI-safe (no keys)."""

from __future__ import annotations

from pathlib import Path

from jev_route.backtest import render_report, run_backtest
from jev_route.policy import Policy


def _trace(tmp_path: Path) -> Path:
    rows = [
        {"id": "t1", "category": "trivial_completion", "text": "Say hello in three words.",
         "expected_tier": "local", "pii_expected": False},
        {"id": "t2", "category": "code_gen", "text": "Write a Python function that reverses a linked list.",
         "expected_tier": "cheap", "pii_expected": False},
        {"id": "t3", "category": "pii_fake", "text": "My SSN is 000-12-3456, check my tax status.",
         "expected_tier": "local", "pii_expected": True},
    ]
    import json

    p = tmp_path / "trace.jsonl"
    p.write_text(chr(10).join(json.dumps(r) for r in rows))
    return p


async def test_backtest_replays_and_saves(tmp_path: Path, default_policy: Policy) -> None:
    report = await run_backtest(trace_path=_trace(tmp_path), policy=default_policy)
    assert report.total == 3
    assert report.cost_baseline > report.cost_ours
    assert 0.0 < report.savings_pct < 100.0
    # the synthetic SSN row must be caught by the gate and kept local
    t3 = next(r for r in report.rows if r.row_id == "t3")
    assert t3.tier == "local"


def test_render_report_contains_the_headline(tmp_path: Path, default_policy: Policy) -> None:
    import asyncio

    report = asyncio.run(run_backtest(trace_path=_trace(tmp_path), policy=default_policy))
    md = render_report(report, trace_path="trace.jsonl", policy_name="default")
    assert "cheaper" in md
    assert "gate" in md.lower()
