"""The graduate --stage enforce live gate: every unmet criterion refuses by name."""

from __future__ import annotations

import json
import time
from pathlib import Path

from jev_route.distill.cli import _injection_eval_check, _live_window_checks
from jev_route.shadow_metrics import ShadowMetrics


def _low_agreement_status(tmp_path: Path):
    m = ShadowMetrics(tmp_path / "w.jsonl")
    now = time.time()
    for i in range(600):
        agree = i % 5 == 0  # 20% agreement -- far below the 0.95 bar
        m.record(decision_fired=True, shadow_score=0.9 if agree else 0.1, threshold=0.5,
                 ts=now - (8.0 * 86400.0 * (1 - i / 599)))
    return m.window_status()


def test_low_agreement_window_refuses_naming_agreement(tmp_path: Path) -> None:
    eval_json = tmp_path / "eval.json"
    eval_json.write_text(json.dumps({"leaks": 0, "false_positives": 0}))
    status = _low_agreement_status(tmp_path)
    checks, _payload = _live_window_checks(status, eval_json)
    by_name = {c.name: c for c in checks}
    assert by_name["agreement_rate"].ok is False
    assert "0.95" in by_name["agreement_rate"].requirement  # the bar is named
    assert by_name["window_complete"].ok is True       # volume/days are fine
    assert by_name["injection_eval"].ok is True


def test_incomplete_window_refuses_naming_it(tmp_path: Path) -> None:
    m = ShadowMetrics(tmp_path / "w.jsonl")
    m.record(decision_fired=True, shadow_score=0.9, threshold=0.5, ts=time.time())
    eval_json = tmp_path / "eval.json"
    eval_json.write_text(json.dumps({"leaks": 0, "false_positives": 0}))
    checks, _ = _live_window_checks(m.window_status(), eval_json)
    by_name = {c.name: c for c in checks}
    assert by_name["window_complete"].ok is False
    assert "decisions" in by_name["window_complete"].measured


def test_injection_eval_check_missing_good_and_leaky(tmp_path: Path) -> None:
    ok, measured, _p = _injection_eval_check(tmp_path / "nope.json")
    assert ok is False and "missing file" in measured

    good = tmp_path / "good.json"
    good.write_text(json.dumps({"leaks": 0, "false_positives": 0}))
    ok, measured, _p = _injection_eval_check(good)
    assert ok is True

    leaky = tmp_path / "leaky.json"
    leaky.write_text(json.dumps({"leaks": 3, "false_positives": 1}))
    ok, measured, _p = _injection_eval_check(leaky)
    assert ok is False
