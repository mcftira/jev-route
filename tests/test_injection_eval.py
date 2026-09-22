"""The injection eval: generator + runner are deterministic and keyless.

The new ``--semantic-laya`` mode is tested two ways: without the flag the
output must stay byte-identical to the deterministic-only report (rebuilt
here from first principles -- the cases file plus the gate, no runner code),
and with the flag the loader is monkeypatched to a mock ENFORCE-mode
semantic layer so the suite never needs torch, a checkpoint, or a fitted
artifact (the artifact path may not exist in CI -- the flag is simply
unused there).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from jev_route.gate import HardGate

INJ = Path(__file__).resolve().parents[1] / "evals" / "injection"
_CATEGORIES = ("injection_wrapper", "authority_framing", "encoding_trick", "benign_control")
_RUN_MODULE: Any = None


def _load_cases() -> list[dict]:
    return [json.loads(line) for line in (INJ / "cases.jsonl").read_text().splitlines() if line.strip()]


def _det_blocked(case: dict, gate: HardGate) -> bool:
    verdict = gate.scan(case["text"])
    return bool(verdict.blocks_backend or verdict.force_local)


def _run_module() -> Any:
    """Import evals/injection/run.py as a module (it is a script, not a package)."""
    global _RUN_MODULE
    if _RUN_MODULE is None:
        spec = importlib.util.spec_from_file_location("injection_run", INJ / "run.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _RUN_MODULE = module
    return _RUN_MODULE


def _expected_deterministic_md(cases: list[dict]) -> str:
    """Rebuild the deterministic-only report from first principles.

    The counts come straight from the cases file and the gate; every other
    line is the format contract. If this disagrees with the runner's output,
    either the runner drifted from the committed report format or the gate
    changed under it -- both are findings, not test bugs.
    """
    gate = HardGate()
    blocked_by_cat: dict[str, int] = {}
    total_by_cat: dict[str, int] = {}
    cat12_leaks = 0
    cat3_leak_ids: list[str] = []
    n_fps = 0
    for case in cases:
        cat = case["category"]
        total_by_cat[cat] = total_by_cat.get(cat, 0) + 1
        blocked = _det_blocked(case, gate)
        blocked_by_cat[cat] = blocked_by_cat.get(cat, 0) + int(blocked)
        if case["expect"] == "blocked" and not blocked:
            if cat in ("injection_wrapper", "authority_framing"):
                cat12_leaks += 1
            elif cat == "encoding_trick":
                cat3_leak_ids.append(case["id"])
        elif case["expect"] == "allowed" and blocked:
            n_fps += 1
    lines = [
        "# Injection eval -- sensitivity gate (v0.2)",
        "",
        "300 synthetic cases (`evals/injection/cases.jsonl`, all provably fake PII:",
        "000-area SSNs, Luhn-invalid cards, RFC 2606 emails, placeholder names).",
        "The threat model: inputs that try to convince the router that sensitive",
        "data is safe for cloud models.",
        "",
        "## Results by category",
        "",
        "| category | cases | blocked | allowed (leaked) |",
        "|---|---|---|---|",
    ]
    for cat in sorted(total_by_cat):
        total = total_by_cat[cat]
        blocked = blocked_by_cat.get(cat, 0)
        lines.append(f"| {cat} | {total} | {blocked} | {total - blocked} |")
    lines += [
        "",
        "## Acceptance (v0.2)",
        "",
        f"* categories 1-2 (injection wrappers, authority framing): **{cat12_leaks} leaks** --"
        f" {'PASS' if not cat12_leaks else 'FAIL'} (bar: 0).",
        f"* category 3 (encoding tricks): **{len(cat3_leak_ids)} leaks** -- since v0.3 the gate",
        "  normalizes encodings (base64/spaced/leet) and rescans deterministically, so this",
        "  class is closed without a model; see RESULTS_v0.3.md.",
        f"* category 4 (benign controls): **{n_fps} false positives** -- these are the",
        "  provably-fake cases the gate must NOT block.",
        "",
        "## Category 3 findings (the honest part)",
        "",
        "The deterministic layer is regex-class. These encodings defeat it BY DESIGN,",
        "and every leak here is a published requirement for the semantic gate:",
        "",
    ]
    if cat3_leak_ids:
        lines += [f"* `{leak_id}` -- not detected by any deterministic detector" for leak_id in cat3_leak_ids]
    else:
        lines.append("* (none -- the deterministic layer caught everything this round)")
    lines += [
        "",
        "That list is the roadmap: the semantic layer (`gate.semantic`, shadow today)",
        "exists precisely because a regex cannot read base64 or typo'd Hungarian",
        "identifiers. RESULTS.md does not claim perfection; it shows the boundary",
        "and where the distilled layer takes over.",
        "",
        "## Method and data hygiene",
        "",
        "* Runner: `evals/injection/run.py` (stdlib + the repo gate; no API keys).",
        "* Committed results record METADATA only (category, blocked, detector ids).",
        "  Case content and model payloads never appear in committed output.",
        "* Regenerate with `python evals/injection/gen_cases.py` (deterministic).",
    ]
    return chr(10).join(lines) + chr(10)


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


def test_flag_absent_output_is_byte_identical(tmp_path: Path) -> None:
    """Without --semantic-laya the report and stdout are exactly the deterministic-only ones."""
    out = tmp_path / "RESULTS.md"
    proc = subprocess.run(
        [sys.executable, str(INJ / "run.py"), "--out", str(out)],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-500:]
    cases = _load_cases()
    expected = _expected_deterministic_md(cases)
    assert out.read_text() == expected
    gate = HardGate()
    cat12 = sum(
        1 for c in cases
        if c["category"] in ("injection_wrapper", "authority_framing") and not _det_blocked(c, gate)
    )
    total_leaks = sum(1 for c in cases if c["expect"] == "blocked" and not _det_blocked(c, gate))
    n_fps = sum(1 for c in cases if c["expect"] == "allowed" and _det_blocked(c, gate))
    assert proc.stdout == (
        f"wrote {out}; cat1-2 leaks: {cat12}; total leaks: {total_leaks}; false positives: {n_fps}\n"
    )
    # No semantic-enforce view may leak into the deterministic-only report.
    assert "semantic-enforce" not in out.read_text()


class _AlwaysFireScorer:
    """A layer-2 scorer that fires on everything: the maximum-strictness probe.

    Every case -- including the gate-allowed benign controls -- scores above
    the enforce threshold, so the semantic column must count them as blocked.
    """

    name = "mock-semantic"
    model_version = "mock-semantic-1"

    def score(self, text: str, *, features: Any = None) -> float:
        del text, features
        return 0.99


def _mock_enforce_layer(run: Any, cases: list[dict]) -> Any:
    """The ENFORCE-mode semantic layer the monkeypatched loader stands in for."""
    n_pos = sum(1 for c in cases if c["expect"] == "blocked")
    n_neg = len(cases) - n_pos
    return run.SemanticLayer(
        run.SemanticGatePolicy(
            mode="enforce",
            threshold=run.SEMANTIC_ENFORCE_THRESHOLD,
            level=run.SEMANTIC_ENFORCE_LEVEL,
            enforce_requires=run.eval_enforce_criteria(n_pos, n_neg),
        ),
        scorer=_AlwaysFireScorer(),
        metrics={
            "n_positives": n_pos,
            "n_negatives": n_neg,
            "recall": 1.0,
            "false_positive_rate": 0.0,
            "disagreement_rate": 0.0,
            "semantic_miss_rate": 0.0,
            "shadow_n_examples": 0,
            "threshold": run.SEMANTIC_ENFORCE_THRESHOLD,
            "model_version": _AlwaysFireScorer.model_version,
        },
    )


def test_flag_present_mock_scorer_counts_gate_allowed_cases_as_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the flag, a case the gate allows but the scorer blocks is blocked in the
    semantic-enforce column, and both views are reported in the acceptance section."""
    run = _run_module()
    cases = _load_cases()
    layer = _mock_enforce_layer(run, cases)
    monkeypatch.setattr(run, "load_semantic_laya", lambda artifact, cases_path: layer)
    out = tmp_path / "RESULTS.md"
    monkeypatch.setattr(
        sys, "argv", ["run.py", "--out", str(out), "--semantic-laya", str(tmp_path / "never-opened")]
    )
    assert run.main() == 0

    gate = HardGate()
    md = out.read_text()
    assert "blocked (semantic-enforce)" in md
    rows = {line.split(" | ")[0].lstrip("| "): line for line in md.splitlines() if line.startswith("| ")}
    for cat in _CATEGORIES:
        total = sum(1 for c in cases if c["category"] == cat)
        det_blocked = sum(1 for c in cases if c["category"] == cat and _det_blocked(c, gate))
        # The mock fires on everything, so the union column is fully blocked.
        expected_row = f"| {cat} | {total} | {det_blocked} | {total - det_blocked} | {total} | 0 |"
        assert rows[cat] == expected_row

    n_benign = sum(1 for c in cases if c["expect"] == "allowed")
    assert f"semantic-enforce: **{n_benign} false positives** -- layer 2 alarmed on fake PII." in md
    assert "  - deterministic:" in md and "  - semantic-enforce:" in md

    # Case level: a benign case the deterministic gate allowed is blocked in the
    # semantic view (a semantic-enforce false positive) and not a deterministic one.
    results = run.evaluate(INJ / "cases.jsonl", semantic=layer)
    benign = next(c for c in cases if c["expect"] == "allowed" and not _det_blocked(c, gate))
    sem_fp_ids = {entry["id"] for entry in results["sem_false_positives"]}
    det_fp_ids = {entry["id"] for entry in results["false_positives"]}
    assert benign["id"] in sem_fp_ids
    assert benign["id"] not in det_fp_ids
    counts = results["per_cat"]["benign_control"]
    assert counts["blocked"] == 0
    assert counts["sem_blocked"] == n_benign


def test_flag_present_missing_artifact_fails_clean(tmp_path: Path) -> None:
    """The artifact path may not exist (e.g. in CI): using the flag there fails loudly,
    before any report is written, instead of silently degrading to the deterministic view."""
    out = tmp_path / "RESULTS.md"
    proc = subprocess.run(
        [sys.executable, str(INJ / "run.py"), "--out", str(out),
         "--semantic-laya", str(tmp_path / "no-such-artifact")],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode != 0
    assert "refused to build" in proc.stdout + proc.stderr
    assert not out.exists()
