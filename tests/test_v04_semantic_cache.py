"""v0.4: the IntentCache semantic pass (batched noul) and the backtest hit-rate split.

Rules pinned here (the work order):

* On an exact/Jaccard miss, with the policy ``intent_cache.semantic: true``
  AND a seam wired, the cache asks ONE batched noul over the top-K (K <= 5)
  most-similar cached summaries -- "same intent and same constraints?" -- via
  an injectable ``noul_fn(prompt_state) -> float``. The cache itself makes no
  model call.
* A noul answer at or above 0.7 replays the most-similar candidate and the
  lookup result records ``hit_kind == "semantic"`` (vs ``"exact"`` / ``"near"``;
  a miss is ``None`` = no result object at all).
* Gate-fired requests are never cached and never candidates -- even when the
  semantic pass is on and maximally permissive.
* A semantic hit whose token overlap with the replayed summary is below the
  floor is replayed but counted in ``semantic_suspicious`` and logged: a
  false replay must be visible in the report, not hidden inside the hit rate.
* The backtest splits exact / near-dup / semantic counts when an intent cache
  is attached; with none attached, the run and the report are unchanged.

Offline and deterministic, like the rest of the suite: the noul is a
programmable double, the backtest runs the oracle TraceBackend, no network.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from jev_route.backtest import render_report, run_backtest
from jev_route.intent_cache import (
    DEFAULT_SEMANTIC_THRESHOLD,
    IntentCache,
    build_intent_cache,
    normalize_text,
    semantic_match_from_noul,
)
from jev_route.policy import Policy
from jev_route.schema import (
    COMPLEXITY_LEVELS,
    DOMAINS,
    SENSITIVITY_LEVELS,
    ChoiceAnswer,
    DecisionAnswers,
    GateVerdict,
    NoulAnswer,
    RoutingDecision,
)

# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #

def make_decision(
    tier: str = "cheap",
    model: str = "qwen3.8-flash",
    *,
    gate_fired: bool = False,
) -> RoutingDecision:
    """A RoutingDecision with the right shape and nothing more than the test asserts."""
    answers = DecisionAnswers(
        complexity=ChoiceAnswer(
            choice="standard",
            probabilities=dict.fromkeys(COMPLEXITY_LEVELS, 0.25),
            confidence=0.8,
        ),
        sensitivity=ChoiceAnswer(
            choice="public",
            probabilities={level: (1.0 if level == "public" else 0.0) for level in SENSITIVITY_LEVELS},
            confidence=1.0,
        ),
        pii=NoulAnswer(value=0.1),
        domain=ChoiceAnswer(choice="chat", probabilities=dict.fromkeys(DOMAINS, 0.2), confidence=0.7),
    )
    return RoutingDecision(
        tier=tier,
        model=model,
        rule_id="fixture",
        reason="fixture decision",
        answers=answers,
        gate=GateVerdict(fired=gate_fired),
        backend="mock",
        backend_model_version="mock-1.0.0",
        effective_sensitivity="public",
        effective_complexity="standard",
        uncertain=False,
    )


#: Ten tokens; the reference ask every fixture is measured against.
CLEAN_A = "how do i optimize the p99 latency of the checkout service"
#: Same normalized text, different case / whitespace / a fullwidth p: an exact hit.
EXACT_VARIANT = "  How  Do I  Optimize the ｐ99 Latency  of the CHECKOUT service "  # noqa: RUF001
#: Nine of the ten tokens: token-set Jaccard exactly 0.9 -> near, not semantic.
NEAR_B = "how do i optimize the p99 latency of the checkout"
#: Seven of the ten tokens (0.54): below the Jaccard threshold, above the
#: suspicious floor -> a clean semantic hit when the noul says yes.
SEMANTIC_C = "help me optimize the latency of the checkout service p99 now"
#: One shared token ("the"): overlap ~0.05, below the floor -> a suspicious
#: (false) semantic hit when the noul says yes.
SUSPICIOUS_D = "completely different ask about banana bread and the weather"
#: A gate-fired variant of the reference ask (a reserved-domain SSN).
FIRED_E = CLEAN_A + " my ssn is 000-12-3456"
#: A trace row the deterministic gate really fires on.
GATE_ROW = "My SSN is 000-12-3456, check my tax status."


def write_trace(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    p = tmp_path / "trace.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# The semantic pass: matching
# --------------------------------------------------------------------------- #

def test_semantic_hit_replays_and_records_hit_kind() -> None:
    calls: list[dict[str, Any]] = []

    def noul(prompt_state: dict[str, Any]) -> float:
        calls.append(prompt_state)
        return 0.9

    cache = IntentCache(noul_fn=noul)
    decision = make_decision(tier="strong")
    cache.store(CLEAN_A, decision)

    hit = cache.lookup(SEMANTIC_C)
    assert hit is not None
    assert hit.hit_kind == "semantic"
    assert hit.matched == "semantic"
    assert hit.decision is decision  # replayed, not re-routed
    assert cache.stats()["semantic_hits"] == 1
    assert cache.stats()["hits"] == 1 and cache.stats()["misses"] == 0

    # ONE batched call, carrying the question, the incoming text and the
    # candidate summary with its token overlap.
    assert len(calls) == 1
    state = calls[0]
    assert state["question"] == "same intent and same constraints?"
    assert state["incoming"] == normalize_text(SEMANTIC_C)
    assert len(state["candidates"]) == 1
    assert state["candidates"][0]["summary"] == normalize_text(CLEAN_A)
    assert state["candidates"][0]["token_overlap"] == pytest.approx(0.538462, abs=1e-5)


def test_hit_kind_for_exact_near_and_miss() -> None:
    cache = IntentCache()
    cache.store(CLEAN_A, make_decision())
    assert cache.lookup(EXACT_VARIANT).hit_kind == "exact"
    assert cache.lookup(NEAR_B).hit_kind == "near"
    assert cache.lookup(SUSPICIOUS_D) is None  # a miss is None: no result object


def test_noul_below_threshold_does_not_replay() -> None:
    cache = IntentCache(noul_fn=lambda ps: DEFAULT_SEMANTIC_THRESHOLD - 0.01)
    decision = make_decision(tier="strong")
    cache.store(CLEAN_A, decision)
    assert cache.lookup(SEMANTIC_C) is None
    stats = cache.stats()
    assert stats["semantic_hits"] == 0 and stats["hits"] == 0 and stats["misses"] == 1
    # The threshold is inclusive: exactly 0.7 replays.
    boundary = IntentCache(noul_fn=lambda ps: DEFAULT_SEMANTIC_THRESHOLD)
    boundary.store(CLEAN_A, decision)
    hit = boundary.lookup(SEMANTIC_C)
    assert hit is not None and hit.hit_kind == "semantic"


def test_batched_question_carries_the_top_k_most_similar() -> None:
    calls: list[dict[str, Any]] = []

    def noul(prompt_state: dict[str, Any]) -> float:
        calls.append(prompt_state)
        return 0.95

    # Seven live entries, ranked by token overlap with the query:
    # s6 0.8 > s1 4/6 > s2 3/7 > s5 0.4 > s3 2/7 > s4 1/7 > s7 0.
    cache = IntentCache(noul_fn=noul)
    query = "alpha beta gamma delta epsilon"
    entries = {
        "s1": "alpha beta gamma delta zeta",
        "s2": "alpha beta gamma eta theta",
        "s3": "alpha beta iota kappa",
        "s4": "alpha lambda mu",
        "s5": "alpha epsilon",
        "s6": "beta gamma delta epsilon",
        "s7": "nu xi omicron",
    }
    for key, text in entries.items():
        cache.store(text, make_decision(model=f"model-{key}"))

    hit = cache.lookup(query)
    assert hit is not None
    assert hit.hit_kind == "semantic"
    assert hit.decision.model == "model-s6"  # the most similar candidate replays

    # ONE call; the batch is capped at 5 and ordered most-similar first.
    assert len(calls) == 1
    summaries = [c["summary"] for c in calls[0]["candidates"]]
    assert summaries == [entries["s6"], entries["s1"], entries["s2"], entries["s5"], entries["s3"]]
    overlaps = [c["token_overlap"] for c in calls[0]["candidates"]]
    assert overlaps == pytest.approx([0.8, 4 / 6, 3 / 7, 0.4, 2 / 7], abs=1e-5)

    # The cap holds even when the caller asks for more.
    calls2: list[dict[str, Any]] = []
    cache2 = IntentCache(semantic_match=semantic_match_from_noul(lambda ps: calls2.append(ps) or 0.95, top_k=99))
    for text in entries.values():
        cache2.store(text, make_decision())
    cache2.lookup(query)
    assert len(calls2[0]["candidates"]) <= 5


def test_policy_flag_and_seam_are_both_required() -> None:
    seam = semantic_match_from_noul(lambda ps: 1.0)  # maximally permissive
    # Built from a policy WITHOUT the flag: the wired seam is not consulted.
    cache = build_intent_cache({"intent_cache": {"enabled": True}})
    assert cache.semantic is False
    cache.semantic_match = seam
    cache.store(CLEAN_A, make_decision())
    assert cache.lookup(SEMANTIC_C) is None
    assert cache.stats()["semantic_hits"] == 0

    # The same policy WITH semantic: true: the seam runs and the hit is semantic.
    flagged = build_intent_cache({"intent_cache": {"enabled": True, "semantic": True}})
    assert flagged.semantic is True
    flagged.semantic_match = seam
    flagged.store(CLEAN_A, make_decision())
    hit = flagged.lookup(SEMANTIC_C)
    assert hit is not None and hit.hit_kind == "semantic"

    # A typo'd flag is a config error, not a silent off-switch.
    with pytest.raises(ValueError, match=r"intent_cache\.semantic"):
        build_intent_cache({"intent_cache": {"enabled": True, "semantic": "yes"}})


def test_noul_exception_is_a_miss() -> None:
    def noul(ps: dict[str, Any]) -> float:
        raise RuntimeError("model down")

    cache = IntentCache(noul_fn=noul)
    cache.store(CLEAN_A, make_decision())
    assert cache.lookup(SEMANTIC_C) is None
    stats = cache.stats()
    assert stats["semantic_hits"] == 0 and stats["misses"] == 1


def test_semantic_pass_never_runs_on_an_empty_cache() -> None:
    calls: list[dict[str, Any]] = []
    cache = IntentCache(noul_fn=lambda ps: calls.append(ps) or 1.0)
    assert cache.lookup("anything at all") is None
    assert calls == []  # no candidates, no call


# --------------------------------------------------------------------------- #
# The gate contract, with the semantic pass on
# --------------------------------------------------------------------------- #

def test_gate_fired_never_cached_even_with_semantic_on() -> None:
    cache = IntentCache(noul_fn=lambda ps: 1.0)  # maximally permissive
    cache.store(FIRED_E, make_decision(tier="strong"), gate_fired=True)
    stats = cache.stats()
    assert stats["size"] == 0  # a fired decision is never stored
    assert stats["gate_skips"] == 1


def test_gate_fired_never_candidate_even_with_semantic_on() -> None:
    cache = IntentCache(noul_fn=lambda ps: 1.0)  # says "same intent" to everything
    cache.store(CLEAN_A, make_decision(tier="strong"))

    # A fired variant of the same ask: with the gate verdict, the cache must
    # not serve it -- even though the permissive noul would call it a match.
    hit = cache.lookup(FIRED_E, gate_fired=True)
    assert hit is None
    stats = cache.stats()
    assert stats["gate_skips"] == 1
    assert stats["semantic_hits"] == 0 and stats["hits"] == 0 and stats["misses"] == 0

    # Sanity: without the gate verdict the same text IS a semantic candidate,
    # so the verdict above is what blocked it.
    assert cache.lookup(FIRED_E) is not None


# --------------------------------------------------------------------------- #
# False replays must be visible
# --------------------------------------------------------------------------- #

def test_false_semantic_hit_is_counted_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    def noul(ps: dict[str, Any]) -> float:
        return 1.0 if ps["incoming"] == normalize_text(SUSPICIOUS_D) else 0.0

    cache = IntentCache(noul_fn=noul)
    cache.store(CLEAN_A, make_decision(tier="strong"))

    with caplog.at_level(logging.WARNING, logger="jev_route.intent_cache"):
        hit = cache.lookup(SUSPICIOUS_D)
    # Replayed (the model's yes is the decision) ...
    assert hit is not None and hit.hit_kind == "semantic"
    assert cache.stats()["semantic_hits"] == 1
    # ... but flagged, and logged.
    assert cache.stats()["semantic_suspicious"] == 1
    assert any("suspicious semantic hit" in r.message for r in caplog.records)


def test_clean_semantic_hit_is_not_suspicious() -> None:
    cache = IntentCache(noul_fn=lambda ps: 0.9)
    cache.store(CLEAN_A, make_decision())
    hit = cache.lookup(SEMANTIC_C)
    assert hit is not None
    assert cache.stats()["semantic_suspicious"] == 0


# --------------------------------------------------------------------------- #
# Backtest: the hit-rate section
# --------------------------------------------------------------------------- #

def _trace_rows() -> list[dict[str, Any]]:
    return [
        {"id": "t1", "category": "code_gen", "text": CLEAN_A, "expected_tier": "cheap", "pii_expected": False},
        {"id": "t2", "category": "code_gen", "text": EXACT_VARIANT, "expected_tier": "cheap", "pii_expected": False},
        {"id": "t3", "category": "code_gen", "text": NEAR_B, "expected_tier": "cheap", "pii_expected": False},
        {"id": "t4", "category": "code_gen", "text": SEMANTIC_C, "expected_tier": "cheap", "pii_expected": False},
        {"id": "t5", "category": "pii_fake", "text": GATE_ROW, "expected_tier": "local", "pii_expected": True},
    ]


async def test_backtest_splits_exact_near_semantic(tmp_path: Path, default_policy: Policy) -> None:
    trace = write_trace(tmp_path, _trace_rows())
    cache = IntentCache(
        noul_fn=lambda ps: 0.9 if ps["incoming"] == normalize_text(SEMANTIC_C) else 0.0
    )
    report = await run_backtest(trace_path=trace, policy=default_policy, intent_cache=cache)

    stats = report.intent_cache_stats
    assert stats is not None
    # One stored entry (t1); t2 exact, t3 near, t4 semantic; t5 gate-fired:
    # not served, not stored.
    assert stats["exact_hits"] == 1
    assert stats["near_hits"] == 1
    assert stats["semantic_hits"] == 1
    assert stats["semantic_suspicious"] == 0
    assert stats["gate_skips"] == 1
    assert stats["size"] == 1

    # Hits replay the cached decision: same tier as the first row.
    t1 = next(r for r in report.rows if r.row_id == "t1")
    for rid in ("t2", "t3", "t4"):
        row = next(r for r in report.rows if r.row_id == rid)
        assert row.tier == t1.tier
    t5 = next(r for r in report.rows if r.row_id == "t5")
    assert t5.gate_fired and t5.tier == "local"

    md = render_report(report, trace_path="trace.jsonl", policy_name="default")
    assert "## Intent cache (hit rate)" in md
    # The three kinds are split out, with their counts.
    assert "| exact (normalized) | 1 |" in md
    assert "| near-dup (token Jaccard) | 1 |" in md
    assert "| semantic (batched noul) | 1 |" in md
    # Cache-served rows are excluded from outcome verification, said so.
    assert "served from the intent cache are excluded" in md


async def test_false_semantic_hit_is_visible_in_the_report(tmp_path: Path, default_policy: Policy) -> None:
    rows = [
        {"id": "t1", "category": "code_gen", "text": CLEAN_A, "expected_tier": "cheap", "pii_expected": False},
        {"id": "t2", "category": "code_gen", "text": SUSPICIOUS_D, "expected_tier": "cheap", "pii_expected": False},
    ]
    trace = write_trace(tmp_path, rows)
    cache = IntentCache(
        noul_fn=lambda ps: 1.0 if ps["incoming"] == normalize_text(SUSPICIOUS_D) else 0.0
    )
    report = await run_backtest(trace_path=trace, policy=default_policy, intent_cache=cache)
    md = render_report(report, trace_path="trace.jsonl", policy_name="default")
    assert "## Intent cache (hit rate)" in md
    assert "semantic_suspicious: 1" in md
    assert report.intent_cache_stats["semantic_suspicious"] == 1


async def test_backtest_without_cache_is_unchanged(tmp_path: Path, default_policy: Policy) -> None:
    trace = write_trace(tmp_path, _trace_rows())
    report = await run_backtest(trace_path=trace, policy=default_policy)
    assert report.intent_cache_stats is None
    md = render_report(report, trace_path="trace.jsonl", policy_name="default")
    assert "Intent cache" not in md
    assert "semantic_suspicious" not in md
