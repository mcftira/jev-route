"""v0.5 Phase 0: pinned eval model + the alias-refusal criterion."""

from __future__ import annotations

import json
import time
from pathlib import Path

from jev_route.distill.cli import _live_window_checks
from jev_route.shadow_metrics import ShadowMetrics


def _complete_status(tmp_path: Path):
    m = ShadowMetrics(tmp_path / "w.jsonl")
    now = time.time()
    for i in range(600):
        m.record(decision_fired=True, shadow_score=0.98, threshold=0.5,
                 ts=now - (8.0 * 86400.0 * (1 - i / 599)))
    return m.window_status()


def _eval_file(tmp_path: Path) -> Path:
    p = tmp_path / "eval.json"
    p.write_text(json.dumps({"leaks": 0, "false_positives": 0}))
    return p


def test_alias_model_refuses_promotion_by_name(tmp_path: Path) -> None:
    checks, _ = _live_window_checks(
        _complete_status(tmp_path), _eval_file(tmp_path), policy_backend_model="jev-latest"
    )
    by_name = {c.name: c for c in checks}
    assert by_name["model_pinned"].ok is False
    assert by_name["model_pinned"].measured == "jev-latest"
    # everything else still passes -- the only blocker is the alias
    assert all(c.ok for n, c in by_name.items() if n != "model_pinned")


def test_versioned_model_passes_the_pinned_bar(tmp_path: Path) -> None:
    checks, _ = _live_window_checks(
        _complete_status(tmp_path), _eval_file(tmp_path), policy_backend_model="jev-1.13.0"
    )
    by_name = {c.name: c for c in checks}
    assert by_name["model_pinned"].ok is True
    assert all(c.ok for c in checks)


def test_pinned_eval_model_constant() -> None:
    from jev_route.backends.jev import PINNED_EVAL_MODEL

    assert PINNED_EVAL_MODEL == "jev-1.13.0"
    assert not PINNED_EVAL_MODEL.endswith("-latest")
