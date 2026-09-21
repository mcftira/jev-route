"""The injection eval: generator + runner are deterministic and keyless."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

INJ = Path(__file__).resolve().parents[1] / "evals" / "injection"


def test_cases_file_exists_and_is_balanced() -> None:
    cases = [json.loads(line) for line in (INJ / "cases.jsonl").read_text().splitlines() if line.strip()]
    assert len(cases) >= 200
    cats = {c["category"] for c in cases}
    assert {"injection_wrapper", "authority_framing", "encoding_trick", "benign_control"} == cats
    assert all(c["expect"] in ("blocked", "allowed") for c in cases)


def test_runner_produces_results_with_zero_cat12_leaks() -> None:
    proc = subprocess.run(
        [sys.executable, str(INJ / "run.py")],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-500:]
    assert (INJ / "RESULTS.md").exists()
    md = (INJ / "RESULTS.md").read_text()
    assert "injection_wrapper" in md
    assert "0 leaks" in md or "**0 leaks**" in md
