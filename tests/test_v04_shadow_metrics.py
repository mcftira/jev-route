"""Live shadow metrics + the graduate gate track: the window decides
promotion, drift demotes, and every refusal names its criterion."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from jev_route.shadow_metrics import (
    BASELINE_SIZE,
    REQUIRED_DAYS,
    REQUIRED_DECISIONS,
    ShadowMetrics,
    maybe_demote,
)


def _feed(m: ShadowMetrics, n: int, *, agree: bool, days_span: float = 8.0) -> None:
    now = time.time()
    for i in range(n):
        ts = now - (days_span * 86400.0 * (1 - i / max(1, n - 1)))
        fired = True
        score = 0.9 if agree else 0.1
        m.record(decision_fired=fired, shadow_score=score, threshold=0.5, ts=ts)


def test_window_completes_on_volume_and_days(tmp_path: Path) -> None:
    m = ShadowMetrics(tmp_path / "w.jsonl")
    assert not m.window_status().window_complete
    _feed(m, REQUIRED_DECISIONS, agree=True, days_span=8.0)
    status = m.window_status()
    assert status.window_complete
    assert status.n_decisions == REQUIRED_DECISIONS
    assert status.days_covered >= REQUIRED_DAYS


def test_window_needs_both_volume_and_days(tmp_path: Path) -> None:
    m = ShadowMetrics(tmp_path / "w.jsonl")
    _feed(m, REQUIRED_DECISIONS, agree=True, days_span=1.0)  # enough decisions, not enough days
    assert not m.window_status().window_complete


def test_agreement_rate_and_ece(tmp_path: Path) -> None:
    m = ShadowMetrics(tmp_path / "w.jsonl")
    for i in range(100):
        agree = i % 2 == 0
        m.record(decision_fired=True, shadow_score=0.9 if agree else 0.1, threshold=0.5, ts=time.time() - i)
    status = m.window_status()
    assert status.agreement_rate == pytest.approx(0.5)
    assert status.ece is not None and status.ece >= 0.0


def test_low_agreement_is_measured_not_hidden(tmp_path: Path) -> None:
    m = ShadowMetrics(tmp_path / "w.jsonl")
    _feed(m, 50, agree=False)
    status = m.window_status()
    assert status.agreement_rate == pytest.approx(0.0)


def test_drift_alarm_and_demote_event(tmp_path: Path) -> None:
    m = ShadowMetrics(tmp_path / "w.jsonl")
    now = time.time()
    # baseline: first BASELINE_SIZE records, perfect agreement
    for i in range(BASELINE_SIZE):
        m.record(decision_fired=True, shadow_score=0.95, threshold=0.5, ts=now - 100 + i)
    # current: near-total disagreement -> drift > DRIFT_FACTOR x baseline(0)
    for i in range(50):
        m.record(decision_fired=True, shadow_score=0.05, threshold=0.5, ts=now + i)
    status = m.window_status()
    assert status.drift_alarm
    assert maybe_demote(m)
    events = [json.loads(line) for line in m.events_path.read_text().splitlines() if line.strip()]
    assert events[-1]["kind"] == "jev_route.demotion"
    # every call during an open alarm records an event (the log is the point;
    # the host polls the file, the layer stops enforcing in-process)
    assert maybe_demote(m) is True


def test_no_drift_no_demote(tmp_path: Path) -> None:
    m = ShadowMetrics(tmp_path / "w.jsonl")
    _feed(m, BASELINE_SIZE + 20, agree=True)
    assert not m.window_status().drift_alarm
    assert maybe_demote(m) is False
    assert not m.events_path.exists()


def test_record_is_metadata_only(tmp_path: Path) -> None:
    m = ShadowMetrics(tmp_path / "w.jsonl")
    m.record(decision_fired=True, shadow_score=0.5, threshold=0.5, ts=1.0)
    line = (tmp_path / "w.jsonl").read_text().strip()
    rec = json.loads(line)
    assert set(rec) <= {"decision_fired", "shadow_score", "threshold", "ts"}
