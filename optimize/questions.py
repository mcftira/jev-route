"""Question compilation: GEPA optimizes the wording of our routing questions
against measured routing outcomes on the synthetic trace.

Nothing here runs in CI: compilation is a manual, keyed, local step. Only its
artifact (policy/compiled_questions.yaml, provenance-headed) and the adoption
gate's numbers are committed. The eval case file is the held-out test and is
never in any GEPA dataset.
"""

from __future__ import annotations

import asyncio
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from jev_route.backends.jev import JevBackend, build_questions
from jev_route.cache import NullCache
from jev_route.logging_sink import NullSink
from jev_route.policy import Policy
from jev_route.pricing import DEFAULT_OUTPUT_TOKENS, estimate_cost_usd, prices_from_policy
from jev_route.router import Router

#: The seed candidate: the current hand-written routing + sensitivity question
#: texts, serialized as the artifact GEPA evolves. The schema never changes;
#: only wording may move.
def seed_candidate() -> dict[str, Any]:
    qs = build_questions(include_domain=False)
    return {
        "questions_yaml": yaml.safe_dump(
            {k: {"instructions": v["instructions"], "criteria": v["criteria"]} for k, v in qs.items()},
            sort_keys=True, allow_unicode=True,
        )
    }


def load_stratified(trace_path: str | Path, *, n_train: int, n_val: int, seed: int = 20260922) -> tuple[list[dict], list[dict]]:
    """Stratified disjoint train/val splits over the synthetic trace."""
    rows = [json.loads(line) for line in Path(trace_path).read_text().splitlines() if line.strip()]
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = {}
    for row in rows:
        by_cat.setdefault(row["category"], []).append(row)
    for cat in by_cat:
        rng.shuffle(by_cat[cat])
    total = sum(len(v) for v in by_cat.values())
    train: list[dict] = []
    val: list[dict] = []
    for cat, cat_rows in by_cat.items():
        n_cat_train = max(1, round(len(cat_rows) / total * n_train))
        n_cat_val = max(1, round(len(cat_rows) / total * n_val))
        train.extend(cat_rows[:n_cat_train])
        val.extend(cat_rows[n_cat_train : n_cat_train + n_cat_val])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


#: category -> the tier the DEFAULT policy intends for it (cheap by default,
#: strong for hard/frontier). pii_fake is gate-owned and excluded from scoring.
_POLICY_EXPECTED: dict[str, str] = {
    "trivial_completion": "cheap",
    "code_gen": "cheap",
    "hungarian_support": "cheap",
    "multi_file_refactor": "strong",
    "long_context": "strong",
}


class QuestionEvaluator:
    """Run the router with CANDIDATE questions; score accuracy vs expected tier,
    cost-adjusted. Feedback text per misroute is the gradient."""

    def __init__(self, *, policy: Policy, api_key: str | None = None) -> None:
        self.policy = policy
        self.api_key = api_key
        self._prices = prices_from_policy(policy.raw if isinstance(policy.raw, dict) else {})

    def _router_for(self, candidate: dict[str, Any]) -> Router | None:
        blob = candidate.get("questions_yaml", "") if isinstance(candidate, dict) else str(candidate)
        try:
            overrides = yaml.safe_load(blob) or {}
        except yaml.YAMLError:
            return None
        backend = JevBackend(api_key=self.api_key, question_overrides=overrides, include_domain=False)
        return Router(self.policy, backend, sink=NullSink(), cache=NullCache())

    def evaluate_one(self, candidate: dict[str, Any], example: dict[str, Any]) -> dict[str, Any]:
        """One trace row through the router with CANDIDATE questions (GEPA's
        per-example contract)."""
        router = self._router_for(candidate)
        expected = _POLICY_EXPECTED.get(str(example.get("expected", "")), "cheap")
        if router is None:
            return {"score": 0.0, "expected": expected, "got": "parse-error", "rule_id": "-", "miss": True}
        try:
            decision = asyncio.run(router.route_text(example["input"], request_id=example["id"]))
        except Exception as exc:
            return {"score": 0.0, "expected": expected, "got": type(exc).__name__, "rule_id": "-", "miss": True}
        got = decision.tier if decision.tier in self._prices else "strong"
        in_tok = max(1, len(example["input"]) // 4)
        ours = estimate_cost_usd(got, in_tok, DEFAULT_OUTPUT_TOKENS, self._prices)
        base = estimate_cost_usd("frontier", in_tok, DEFAULT_OUTPUT_TOKENS, self._prices)
        savings = 0.0 if base == 0 else max(0.0, 1.0 - ours / base)
        correct = got == expected
        score = (1.0 if correct else 0.0) + 0.1 * savings
        return {
            "score": score,
            "expected": expected,
            "got": got,
            "rule_id": decision.rule_id,
            "miss": not correct,
        }

    def evaluate(self, candidate: dict[str, Any], batch: list[dict]) -> dict[str, Any]:
        router = self._router_for(candidate)
        if router is None:
            return {"score": 0.0, "feedback": "candidate YAML did not parse; produce valid question YAML"}
        correct = 0
        ours_cost = 0.0
        base_cost = 0.0
        misses: list[str] = []
        for row in batch:
            try:
                decision = asyncio.run(router.route_text(row["text"], request_id=row["id"]))
            except Exception as exc:  # a candidate that breaks the call scores low, never crashes
                misses.append(f"{row['id']}: decision failed ({type(exc).__name__})")
                continue
            # The metric is policy intent, not the trace's label table: the
            # category's complexity maps to a tier through THIS policy's rules
            # (default policy: cheap default, hard/frontier -> strong; local
            # only via the deterministic gate, which no question influences --
            # so pii_fake rows are excluded from question scoring entirely).
            expected = _POLICY_EXPECTED.get(row["category"], "cheap")
            if row["category"] == "pii_fake":
                continue
            got = decision.tier if decision.tier in self._prices else "strong"
            in_tok = max(1, len(row["text"]) // 4)
            ours_cost += estimate_cost_usd(got, in_tok, DEFAULT_OUTPUT_TOKENS, self._prices)
            base_cost += estimate_cost_usd("frontier", in_tok, DEFAULT_OUTPUT_TOKENS, self._prices)
            if got == expected:
                correct += 1
            else:
                misses.append(
                    f"{row['id']}: policy intends {expected}, candidate questions produced {got} "
                    f"(rule {decision.rule_id})"
                )
        n = max(1, len(batch))
        accuracy = correct / n
        savings = 0.0 if base_cost == 0 else max(0.0, 1.0 - ours_cost / base_cost)
        score = accuracy + 0.25 * savings  # accuracy dominates; cost is the tiebreaker
        nl = chr(10)
        feedback = (
            f"accuracy {correct}/{n} = {accuracy:.2f}, cost savings vs frontier {savings:.2f}." + nl
            + ("misroutes:" + nl + nl.join(misses[:12]) if misses else "no misroutes this batch.")
        )
        return {"score": score, "feedback": feedback, "accuracy": accuracy, "savings": savings, "misses": misses}


def provenance_header(*, baseline_score: float, compiled_score: float, seed: int, max_metric_calls: int) -> str:
    import subprocess

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = "unknown"
    import gepa as _g

    nl = chr(10)
    return (
        "# compiled_questions.yaml -- GEPA-optimized routing question wording" + nl
        + f"# gepa: {getattr(_g, '__version__', 'unknown')} | seed: {seed} | max_metric_calls: {max_metric_calls}" + nl
        + f"# baseline score: {baseline_score:.4f} | compiled score: {compiled_score:.4f}" + nl
        + f"# generated: {datetime.now(timezone.utc).isoformat()} | git: {sha}" + nl
        + "# schema: identical to the hand-written defaults (wording only)" + nl
    )



def write_compiled(out_path: str | Path, compiled_blob: str, header: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + compiled_blob)
