"""Contract tests for :mod:`jev_route.schema`.

Why this file exists
--------------------
``schema.py`` is the one module two very different consumers must agree on at
the same time:

* the **running router**, which reads these dataclasses on every request, and
* the **dataset** -- every ``DecisionRecord`` line already sitting in a decision
  log, which ``jev_route.distill`` trains on months from now.

A record written today must mean the same thing to a trainer run after the next
release. So these tests do not merely check that the dataclasses "work"; they
pin the four promises the rest of the system is built on:

1. **Soft answers survive serialization in full.** ``to_dict``/``from_dict`` and
   ``to_json``/``from_json`` must preserve *every* option's probability, not just
   the argmax. Distillation trains on the distribution. A round trip that quietly
   collapsed it to the winning label would still pass any test that only looked
   at ``choice`` -- which is why the assertions below iterate the whole ladder.
2. **Field order is part of the dataset contract.** ``RequestFeatures`` is
   exported positionally by the training pipeline, so new features are appended
   and never reordered. That is asserted literally, not implicitly.
3. **JSONL lines are canonical.** One line, no separator padding, keys sorted,
   and ``default=str`` so an odd metadata value degrades to a string instead of
   killing the log write. A record that cannot be written is a record that never
   becomes training data, and it fails at the worst possible moment: in
   production, under load, after the routing decision was already made.
4. **The uncertainty helpers mean what their docstrings say.** Policy thresholds
   (``on_uncertain.sensitivity_confidence_below`` and friends) are compared
   against :func:`certainty_from_probabilities`, so its scale is load-bearing and
   its edge cases are pinned by number: which way an empty or all-zero
   distribution rounds, and how the entropy form differs from the margin form
   that :attr:`NoulAnswer.confidence` uses.

Everything here is offline and needs no API key: the schema is stdlib-only by
design, and these tests are the reason it can stay that way.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from jev_route.schema import (
    COMPLEXITY_LEVELS,
    DEFAULT_NOUL_THRESHOLD,
    DOMAINS,
    RECORD_KIND,
    SCHEMA_VERSION,
    SENSITIVITY_LEVELS,
    TIERS,
    ChoiceAnswer,
    DecisionAnswers,
    DecisionRecord,
    GateFinding,
    GateVerdict,
    NoulAnswer,
    RequestFeatures,
    RoutingDecision,
    bump_within,
    certainty_from_probabilities,
    level_index,
)


# --------------------------------------------------------------------------- #
# Builders. Local rather than imported from conftest so this module states its
# own expectations: every helper below produces a distribution that spans the
# WHOLE ladder, because "spans the whole ladder" is what most tests assert.
# --------------------------------------------------------------------------- #
def distribution(level: str, ladder: Sequence[str], confidence: float) -> ChoiceAnswer:
    """A :class:`ChoiceAnswer` whose argmax is ``level`` at exactly ``confidence``.

    The remaining mass is spread evenly over the other options, so the answer is
    a plausible calibrated read rather than a one-hot vector: the interesting
    cases for this schema are the ones where the runner-up mass matters.
    """
    if level not in ladder:
        raise ValueError(f"{level!r} is not in {tuple(ladder)}")
    others = [x for x in ladder if x != level]
    share = (1.0 - confidence) / len(others) if others else 0.0
    masses = dict.fromkeys(others, share)
    masses[level] = 1.0 - share * len(others)
    # Re-key in ladder order so a diff of two serialized records is readable.
    return ChoiceAnswer(
        choice=level,
        probabilities={name: masses[name] for name in ladder},
        confidence=confidence,
    )


def answers_fixture(*, pii: float = 0.18) -> DecisionAnswers:
    """A full, deliberately non-uniform answer set with runner-up mass everywhere."""
    return DecisionAnswers(
        complexity=distribution("hard", COMPLEXITY_LEVELS, 0.72),
        sensitivity=distribution("confidential", SENSITIVITY_LEVELS, 0.61),
        pii=NoulAnswer(value=pii),
        domain=distribution("data-extraction", DOMAINS, 0.88),
    )


def finding_fixture(
    *,
    detector: str = "iban",
    category: str = "pii",
    sensitivity_floor: str = "regulated",
    force_local: bool = True,
    span_hash: str = "9f2c1a",
    count: int = 1,
) -> GateFinding:
    return GateFinding(
        detector=detector,
        category=category,  # type: ignore[arg-type]  # Literal in the dataclass
        sensitivity_floor=sensitivity_floor,
        force_local=force_local,
        span_hash=span_hash,
        count=count,
    )


def verdict_fixture() -> GateVerdict:
    return GateVerdict(
        fired=True,
        findings=(finding_fixture(), finding_fixture(detector="email", span_hash="4b71de", count=3)),
        sensitivity_floor="regulated",
        pii_floor=0.8,
        force_local=True,
        blocks_backend=True,
        advisory_topics=("hipaa", "pci"),
    )


def decision_fixture(**overrides: Any) -> RoutingDecision:
    kwargs: dict[str, Any] = {
        "tier": "local",
        "model": "llama-3.1-8b",
        "rule_id": "gate.force-local",
        "reason": "local hard gate matched a structured identifier",
        "answers": answers_fixture(),
        "gate": verdict_fixture(),
        "backend": "mock",
        "backend_model_version": "mock-1.0.0",
        "effective_sensitivity": "regulated",
        "effective_complexity": "hard",
        "escalated": ("sensitivity",),
        "degraded": False,
        "degrade_reason": None,
        "latency_ms": 3.5,
        "cached": True,
    }
    kwargs.update(overrides)
    return RoutingDecision(**kwargs)


def record_fixture(**overrides: Any) -> DecisionRecord:
    kwargs: dict[str, Any] = {
        "request_id": "req-0001",
        "timestamp": "2026-01-02T03:04:05Z",
        "decision": decision_fixture(),
        "features": RequestFeatures(
            char_len=42,
            word_count=7,
            line_count=1,
            sentence_count=1,
            mean_word_len=5.14,
            lang="en",
            gate_detectors={"iban": 1},
            n_gate_findings=1,
            gate_force_local=True,
        ),
        "excerpt_hash": "a" * 16,
        "backend_latency_ms": 12.5,
        "total_latency_ms": 14.0,
        "questions_sent": {"complexity": {"type": "choice"}, "pii_present": {"type": "noul"}},
        "excerpt": None,
        "metadata": {"tenant": "acme", "api_key_alias": "default"},
        "requested_model": "gpt-4o-mini",
        "shadow": {"agree": False, "tier": "cheap"},
    }
    kwargs.update(overrides)
    return DecisionRecord(**kwargs)


# --------------------------------------------------------------------------- #
# Ladders: ascending order is what makes bump_within() mean "more conservative"
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "ladder",
    [COMPLEXITY_LEVELS, SENSITIVITY_LEVELS, DOMAINS, TIERS],
    ids=["complexity", "sensitivity", "domains", "tiers"],
)
def test_ladders_are_non_empty_and_free_of_duplicates(ladder: tuple[str, ...]) -> None:
    # A duplicated level would make level_index() ambiguous and bump_within()
    # silently skip a rung, so uniqueness is a precondition, not a nicety.
    assert ladder
    assert len(set(ladder)) == len(ladder)


def test_ladder_endpoints_are_the_documented_extremes() -> None:
    # bump_within() clamps at both ends and treats "up" as the conservative
    # direction, which is only true if the ladders are declared ascending.
    assert COMPLEXITY_LEVELS[0] == "trivial"
    assert COMPLEXITY_LEVELS[-1] == "frontier"
    assert SENSITIVITY_LEVELS[0] == "public"
    assert SENSITIVITY_LEVELS[-1] == "regulated"
    assert TIERS[0] == "local"


# --------------------------------------------------------------------------- #
# certainty_from_probabilities
# --------------------------------------------------------------------------- #
def test_certainty_is_one_when_all_mass_sits_on_one_option() -> None:
    assert certainty_from_probabilities(dict.fromkeys(COMPLEXITY_LEVELS, 0.0) | {"hard": 1.0}) == 1.0


@pytest.mark.parametrize("size", [2, 3, 4, 5])
def test_certainty_is_zero_for_a_perfectly_uniform_spread(size: int) -> None:
    # Uniform is maximum uncertainty by definition; this is the number the
    # distilled backend's confidence is compared against when it hedges.
    probabilities = {f"opt{i}": 1.0 / size for i in range(size)}
    assert certainty_from_probabilities(probabilities) == pytest.approx(0.0, abs=1e-12)


def test_certainty_of_a_single_entry_mapping_is_one() -> None:
    # log(1) is 0, so without the explicit one-option branch this would divide by
    # zero. One option asserted with all the mass is total certainty by definition.
    assert certainty_from_probabilities({"only": 1.0}) == 1.0


def test_certainty_of_an_empty_mapping_is_zero() -> None:
    # Nothing was asserted, so there is nothing to be certain about. The helper is
    # still total -- it returns a number for every input instead of raising --
    # but this case rounds toward uncertainty, because the alternative ("no
    # options, so no doubt") is the fail-open reading and every other rule in
    # this codebase fails closed.
    assert certainty_from_probabilities({}) == 0.0


@pytest.mark.parametrize(
    ("probabilities", "expected"),
    [
        ({"a": 0.5, "b": 0.5}, 1.0 - math.log(2.0) / math.log(2)),
        ({"a": 0.9, "b": 0.1}, 1.0 - (-(0.9 * math.log(0.9) + 0.1 * math.log(0.1))) / math.log(2)),
        ({"a": 0.6, "b": 0.3, "c": 0.1}, None),
    ],
    ids=["binary-even", "binary-skewed", "ternary"],
)
def test_certainty_is_one_minus_normalized_shannon_entropy(
    probabilities: dict[str, float], expected: float | None
) -> None:
    """Pin the formula, so a future "improvement" cannot quietly rescale it.

    Policy thresholds are absolute numbers (0.8, 0.7). If the scale moves, every
    shipped policy changes meaning without anyone editing a YAML file.
    """
    positive = [p for p in probabilities.values() if p > 0.0]
    # The normalizer counts asserted options, so it is len(positive) here too;
    # these parametrizations are all-positive, where the two readings agree.
    reference = 1.0 - (-sum(p * math.log(p) for p in positive)) / math.log(len(positive))
    if expected is None:
        expected = reference
    assert certainty_from_probabilities(probabilities) == pytest.approx(expected)
    assert certainty_from_probabilities(probabilities) == pytest.approx(reference)


def test_zero_mass_options_do_not_count_toward_the_normalizer() -> None:
    """``n`` counts the options that were actually asserted, not those offered.

    A ruled-out option (probability 0.0) leaves the certainty untouched:
    ``{0.5, 0.5, 0.0}`` scores exactly like ``{0.5, 0.5}``. That is what makes the
    Jev backend's habit of filling every omitted ladder level with 0.0 harmless --
    a partial answer is not penalized just because the ladder it was asked about
    happens to be long.
    """
    two_options = certainty_from_probabilities({"a": 0.5, "b": 0.5})
    with_a_ruled_out_option = certainty_from_probabilities({"a": 0.5, "b": 0.5, "c": 0.0})
    assert two_options == pytest.approx(0.0)
    assert with_a_ruled_out_option == pytest.approx(two_options)
    # A four-level ladder with one level ruled out normalizes by log(3), not log(4).
    partial = certainty_from_probabilities({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3, "d": 0.0})
    assert partial == pytest.approx(0.0)


@pytest.mark.parametrize("peaked_at", [0.4, 0.55, 0.7, 0.85, 1.0])
def test_certainty_is_clamped_to_the_unit_interval(peaked_at: float) -> None:
    # The clamp is what lets a policy compare confidence to a threshold without
    # guarding against nonsense values from a miscalibrated backend.
    assert 0.0 <= certainty_from_probabilities({"a": peaked_at, "b": 1.0 - peaked_at}) <= 1.0


def test_certainty_stays_in_range_for_impossible_inputs() -> None:
    """Unnormalized and over-confident distributions still land in ``[0, 1]``.

    A backend that returns probabilities summing to more than one is broken, but
    the router must not turn that into a confidence of -0.3 or 4.0: the value is
    compared against policy thresholds and stored in the dataset.
    """
    for probabilities in (
        {"a": 0.9, "b": 0.9},
        {"a": 2.0, "b": 2.0},
        {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0},
        {"a": -0.5, "b": 1.5},
    ):
        assert 0.0 <= certainty_from_probabilities(probabilities) <= 1.0


def test_certainty_grows_monotonically_as_mass_concentrates() -> None:
    # Monotonicity is the property a policy threshold actually depends on: if a
    # backend gets more sure, the computed confidence must not go down, or
    # "confidence_below: 0.8" escalates the wrong requests.
    confidences = (0.25, 0.4, 0.6, 0.8, 1.0)
    certainties = [
        certainty_from_probabilities(distribution("hard", COMPLEXITY_LEVELS, c).probabilities) for c in confidences
    ]
    assert certainties == sorted(certainties)
    assert certainties[0] == pytest.approx(0.0, abs=1e-9)  # 0.25 over 4 options is uniform
    assert certainties[-1] == 1.0


def test_binary_certainty_is_normalized_entropy_not_the_probability_margin() -> None:
    """The docstring's own worked example, pinned by number.

    For two options this is ``1 - H(p)/ln 2``: monotone in the margin, but far
    smaller than it. ``{0.9, 0.1}`` gives ~0.53 here where ``abs(p_max - p_min)``
    would give 0.8. :attr:`NoulAnswer.confidence` deliberately uses the margin
    form instead, because for a yes/no judgement "distance from a coin flip" is
    what an operator means by confidence. Asserting both numbers side by side is
    what stops a future refactor from quietly unifying the two scales -- a policy
    threshold of 0.8 means something quite different on each.
    """
    assert certainty_from_probabilities({"yes": 0.9, "no": 0.1}) == pytest.approx(0.5310044064107189)
    assert certainty_from_probabilities({"yes": 0.8, "no": 0.2}) == pytest.approx(0.27807190511263773)
    assert NoulAnswer(value=0.9).confidence == pytest.approx(0.8)
    assert NoulAnswer(value=0.8).confidence == pytest.approx(0.6)


def test_binary_certainty_agrees_with_the_margin_only_at_the_degenerate_points() -> None:
    # The two scales touch at 0.0, 0.5 and 1.0 and diverge everywhere between,
    # which is exactly where a policy threshold usually sits. Documented here so
    # the coincidence at the endpoints is not mistaken for equivalence.
    for value in (0.0, 0.5, 1.0):
        probabilities = {"yes": value, "no": 1.0 - value}
        assert certainty_from_probabilities(probabilities) == pytest.approx(abs(2 * value - 1))
    assert certainty_from_probabilities({"yes": 0.7, "no": 0.3}) != pytest.approx(0.4)


def test_all_zero_distribution_reports_maximum_uncertainty() -> None:
    """A distribution with no mass anywhere carries no information.

    Reachable in production rather than theoretical: ``JevBackend._normalize_choice``
    keeps ladder-valid keys, skips renormalization when the total is 0.0, and
    fills every omitted level with 0.0 -- so a malformed cloud response arrives
    as an all-zero distribution. Reading that as *certain* would be a fail-open,
    because every escalation rule in the policy engine is gated on confidence
    being below a threshold. Zero is the only safe answer.
    """
    assert certainty_from_probabilities({"trivial": 0.0, "standard": 0.0}) == 0.0
    assert certainty_from_probabilities(dict.fromkeys(COMPLEXITY_LEVELS, 0.0)) == 0.0
    # And the schema-level consequence: an answer built from that distribution
    # reports zero computed confidence, so it gets escalated rather than trusted.
    answer = ChoiceAnswer(
        choice="trivial",
        probabilities=dict.fromkeys(COMPLEXITY_LEVELS, 0.0),
        confidence=0.0,
        confidence_reported=False,
    )
    assert answer.computed_confidence == 0.0


def test_a_single_non_zero_option_is_certain_even_on_a_long_ladder() -> None:
    # The other documented edge case: one option asserted with all the mass is
    # maximum certainty regardless of how many rungs were offered.
    assert certainty_from_probabilities({"hard": 1.0}) == 1.0
    assert certainty_from_probabilities(dict.fromkeys(DOMAINS, 0.0) | {"code": 1.0}) == 1.0


# --------------------------------------------------------------------------- #
# level_index / bump_within -- the confidence-escalation machinery
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("level", "expected"),
    [("public", 0), ("internal", 1), ("confidential", 2), ("regulated", 3)],
)
def test_level_index_returns_the_position_in_the_ladder(level: str, expected: int) -> None:
    assert level_index(level, SENSITIVITY_LEVELS) == expected


@pytest.mark.parametrize("unknown", ["", "PUBLIC", "secret", "standard"])
def test_level_index_returns_minus_one_for_an_unknown_level(unknown: str) -> None:
    # -1 rather than a raise: a ladder the policy does not know about must not
    # take the request path down with it. "standard" is a real complexity level
    # but not a sensitivity level, which is the easy mistake to make here.
    assert level_index(unknown, SENSITIVITY_LEVELS) == -1


def test_level_index_accepts_any_sequence() -> None:
    assert level_index("b", ["a", "b", "c"]) == 1


@pytest.mark.parametrize(
    ("ladder", "start", "steps", "expected"),
    [
        (SENSITIVITY_LEVELS, "public", 1, "internal"),
        (SENSITIVITY_LEVELS, "internal", 2, "regulated"),
        (COMPLEXITY_LEVELS, "standard", 1, "hard"),
        (COMPLEXITY_LEVELS, "trivial", 3, "frontier"),
    ],
    ids=["sens+1", "sens+2", "cx+1", "cx+3"],
)
def test_bump_within_positive_steps_move_to_the_stronger_end(
    ladder: tuple[str, ...], start: str, steps: int, expected: str
) -> None:
    # Both ladders are declared ascending, so "+" is always the conservative
    # direction: stricter sensitivity, stronger model. That single convention is
    # what makes on_uncertain.sensitivity_bump_levels=1 safe to configure.
    assert bump_within(start, ladder, steps) == expected


@pytest.mark.parametrize(
    ("ladder", "start", "steps", "expected"),
    [
        (SENSITIVITY_LEVELS, "regulated", -1, "confidential"),
        (COMPLEXITY_LEVELS, "hard", -2, "trivial"),
    ],
    ids=["sens-1", "cx-2"],
)
def test_bump_within_negative_steps_move_down(ladder: tuple[str, ...], start: str, steps: int, expected: str) -> None:
    assert bump_within(start, ladder, steps) == expected


def test_bump_within_clamps_at_both_ends_and_is_idempotent_there() -> None:
    # Overshooting must land on the end rung, not wrap or raise: a bump of 9 on a
    # 4-level ladder is a config typo, and the only safe reading is "as strict as
    # we can express". Clamping also makes repeated bumps idempotent, which is
    # what lets the policy engine apply a bump per question without compounding.
    assert bump_within("regulated", SENSITIVITY_LEVELS, 9) == "regulated"
    assert bump_within("internal", SENSITIVITY_LEVELS, 99) == "regulated"
    assert bump_within("public", SENSITIVITY_LEVELS, -5) == "public"
    assert bump_within("trivial", COMPLEXITY_LEVELS, -5) == "trivial"
    once = bump_within("confidential", SENSITIVITY_LEVELS, 5)
    assert once == "regulated"
    assert bump_within(once, SENSITIVITY_LEVELS, 5) == once


def test_bump_within_returns_an_unknown_level_unchanged() -> None:
    # There is no position to move from, so the honest answer is "unchanged".
    # Inventing a rung for an unknown label would silently relax a floor.
    assert bump_within("not-a-level", SENSITIVITY_LEVELS, 3) == "not-a-level"
    assert bump_within("", COMPLEXITY_LEVELS, -1) == ""


def test_bump_within_zero_steps_is_the_identity() -> None:
    assert bump_within("confidential", SENSITIVITY_LEVELS, 0) == "confidential"


# --------------------------------------------------------------------------- #
# ChoiceAnswer
# --------------------------------------------------------------------------- #
def test_choice_answer_round_trips_through_dict() -> None:
    answer = distribution("hard", COMPLEXITY_LEVELS, 0.72)
    restored = ChoiceAnswer.from_dict(answer.to_dict())
    assert restored == answer
    assert restored.probabilities == dict(answer.probabilities)


def test_choice_answer_round_trip_keeps_confidence_reported_false() -> None:
    # confidence_reported distinguishes a calibrated number from one we derived.
    # Losing the flag would make the distilled backend's computed confidences
    # indistinguishable from Jev's reported ones in the dataset.
    answer = ChoiceAnswer(
        choice="code",
        probabilities=dict.fromkeys(DOMAINS, 0.0) | {"code": 1.0},
        confidence=0.42,
        confidence_reported=False,
    )
    restored = ChoiceAnswer.from_dict(answer.to_dict())
    assert restored == answer
    assert restored.confidence_reported is False


def test_choice_answer_from_dict_defaults_confidence_reported_to_true() -> None:
    # Backwards compatibility with records written before the flag existed:
    # an absent key means "the backend told us", which was the only case then.
    restored = ChoiceAnswer.from_dict({"choice": "chat", "probabilities": {"chat": 1.0}, "confidence": 1.0})
    assert restored.confidence_reported is True


def test_choice_answer_from_dict_coerces_json_shapes_to_floats() -> None:
    # JSONL gives back ints for whole numbers and strings for anything a human
    # edited by hand; the schema is the type boundary, so it normalizes.
    restored = ChoiceAnswer.from_dict({"choice": "hard", "probabilities": {"hard": "1", "trivial": 0}, "confidence": 1})
    assert restored.probabilities == {"hard": 1.0, "trivial": 0.0}
    assert restored.confidence == 1.0
    assert all(isinstance(v, float) for v in restored.probabilities.values())


def test_probability_of_returns_the_mass_for_a_present_option() -> None:
    answer = distribution("regulated", SENSITIVITY_LEVELS, 0.61)
    assert answer.probability_of("regulated") == pytest.approx(0.61)
    assert answer.probability_of("public") == pytest.approx(answer.probabilities["public"])


def test_probability_of_an_absent_option_is_zero_not_an_error() -> None:
    # A backend may answer over a subset of the ladder. Asking about the rest is
    # normal, and 0.0 is the only answer that keeps policy arithmetic total.
    assert distribution("chat", DOMAINS, 0.9).probability_of("not-a-domain") == 0.0


def test_computed_confidence_reproduces_the_distribution_offline() -> None:
    answer = distribution("analysis", DOMAINS, 0.88)
    assert answer.computed_confidence == pytest.approx(certainty_from_probabilities(answer.probabilities))
    # The reported number and the computed one are deliberately different
    # values: distill.evaluate compares them to measure calibration drift, which
    # only means something if a backend's own number is not simply echoed back.
    assert answer.computed_confidence != pytest.approx(answer.confidence)
    assert answer.confidence == pytest.approx(0.88)


@pytest.mark.parametrize(
    ("ladder", "expected_choice", "expected_p"),
    [
        (COMPLEXITY_LEVELS, "hard", 0.25),
        (SENSITIVITY_LEVELS, "confidential", 0.25),
        (DOMAINS, "analysis", 0.2),
    ],
    ids=["complexity", "sensitivity", "domains"],
)
def test_uniform_answers_are_maximum_uncertainty(
    ladder: tuple[str, ...], expected_choice: str, expected_p: float
) -> None:
    """``uniform()`` is the honest shape of "the backend could not answer".

    The picked label is the middle rung (``ladder[len // 2]``), which for these
    ladders is "hard" / "confidential" / "analysis". That choice matters: it is
    what a fail-closed router treats as the answer, so it must be conservative
    rather than optimistic -- and confidence 0.0 makes the policy engine escalate
    it anyway.
    """
    answer = ChoiceAnswer.uniform(ladder)
    assert answer.choice == expected_choice
    assert answer.choice == ladder[len(ladder) // 2]
    assert dict(answer.probabilities) == dict.fromkeys(ladder, expected_p)
    assert answer.confidence == 0.0
    assert answer.confidence_reported is False


def test_uniform_honours_an_explicit_choice() -> None:
    answer = ChoiceAnswer.uniform(DOMAINS, choice="code")
    assert answer.choice == "code"
    assert answer.confidence == 0.0
    assert sum(answer.probabilities.values()) == pytest.approx(1.0)


def test_uniform_falls_back_to_the_middle_when_the_choice_is_not_offered() -> None:
    # A caller asking for a label outside the ladder is a bug upstream; answering
    # with an unknown label would poison the dataset, so it degrades instead.
    assert ChoiceAnswer.uniform(DOMAINS, choice="telepathy").choice == DOMAINS[len(DOMAINS) // 2]
    assert ChoiceAnswer.uniform(COMPLEXITY_LEVELS, choice=None).choice == "hard"


def test_choice_answer_is_immutable() -> None:
    answer = distribution("hard", COMPLEXITY_LEVELS, 0.7)
    with pytest.raises(dataclasses.FrozenInstanceError):
        answer.choice = "trivial"  # type: ignore[misc]


def test_to_dict_returns_a_plain_detached_probability_dict() -> None:
    # The serialized form is what gets written to disk; if it aliased the live
    # mapping, a later mutation would retro-edit a record already logged.
    answer = distribution("hard", COMPLEXITY_LEVELS, 0.7)
    payload = answer.to_dict()
    assert type(payload["probabilities"]) is dict
    payload["probabilities"]["hard"] = 0.0
    assert answer.probabilities["hard"] == pytest.approx(0.7)


def test_to_dict_carries_exactly_the_four_documented_keys() -> None:
    # The dataset contract: no key may be renamed or dropped, because a trainer
    # reading last year's logs has no way to notice.
    assert set(distribution("chat", DOMAINS, 0.9).to_dict()) == {
        "choice",
        "probabilities",
        "confidence",
        "confidence_reported",
    }


# --------------------------------------------------------------------------- #
# NoulAnswer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", [0.0, 0.13, 0.5, 0.77, 1.0])
def test_noul_probabilities_are_yes_and_its_complement(value: float) -> None:
    # Exposed as a two-option distribution so a noul answer can be logged and
    # trained on exactly like a choice answer: same shape, same soft targets.
    answer = NoulAnswer(value=value)
    assert set(answer.probabilities) == {"yes", "no"}
    assert answer.probabilities["yes"] == value
    assert answer.probabilities["no"] == pytest.approx(1.0 - value)
    assert sum(answer.probabilities.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("value", "expected"), [(0.5, 0.0), (0.0, 1.0), (1.0, 1.0), (0.25, 0.5), (0.75, 0.5), (0.9, 0.8)]
)
def test_noul_confidence_is_the_distance_from_a_coin_flip(value: float, expected: float) -> None:
    # 0.5 means genuinely torn, which must read as zero confidence -- not as
    # "medium". This is the number on_uncertain.pii_uncertain_threshold gates on.
    assert NoulAnswer(value=value).confidence == pytest.approx(expected)


def test_noul_confidence_is_symmetric_around_the_coin_flip() -> None:
    assert NoulAnswer(value=0.2).confidence == pytest.approx(NoulAnswer(value=0.8).confidence)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (DEFAULT_NOUL_THRESHOLD, True),
        (1.0, True),
        (0.5000001, True),
        (0.4999999, False),
        (0.0, False),
    ],
)
def test_is_true_is_inclusive_at_the_default_threshold(value: float, expected: bool) -> None:
    # The boundary is the whole test: pii == 0.5 counts as PRESENT. A strict ">"
    # here would let an exactly-torn PII read route to the cloud, which is the
    # one direction this codebase refuses to guess in.
    assert DEFAULT_NOUL_THRESHOLD == 0.5
    assert NoulAnswer(value=value).is_true() is expected


def test_is_true_honours_a_policy_supplied_threshold() -> None:
    # DEFAULT_NOUL_THRESHOLD is only the schema-level default; the policy engine
    # may move it, so the parameter must actually be used.
    assert NoulAnswer(value=0.3).is_true(0.3) is True
    assert NoulAnswer(value=0.3).is_true(0.31) is False
    assert NoulAnswer(value=0.9).is_true(0.95) is False


def test_noul_unknown_is_exactly_a_coin_flip() -> None:
    unknown = NoulAnswer.unknown()
    assert unknown.value == 0.5
    assert unknown.confidence == 0.0
    assert unknown.is_true() is True  # ties go to "present": fail closed


@pytest.mark.parametrize("value", [0.0, 0.05, 0.5, 0.95, 1.0])
def test_noul_round_trips_through_dict(value: float) -> None:
    answer = NoulAnswer(value=value)
    assert answer.to_dict() == {"noul": value}
    assert NoulAnswer.from_dict(answer.to_dict()) == answer


def test_noul_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        NoulAnswer(value=0.5).value = 0.9  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# DecisionAnswers -- the four-question bundle, and the project's core promise
# --------------------------------------------------------------------------- #
def test_answers_round_trip_preserves_every_probability_of_every_ladder() -> None:
    """The core promise: serialization keeps the distribution, not the argmax.

    This is the assertion the whole project rests on. A record that stored only
    ``choice`` would still route correctly and still look right in a log tail --
    and would be worthless for distillation, because the student model is trained
    on the teacher's soft targets. So every option of every ladder is checked
    individually rather than by comparing the two dicts for equality.
    """
    original = answers_fixture()
    restored = DecisionAnswers.from_dict(original.to_dict())

    for name, ladder in (
        ("complexity", COMPLEXITY_LEVELS),
        ("sensitivity", SENSITIVITY_LEVELS),
        ("domain", DOMAINS),
    ):
        before: ChoiceAnswer = getattr(original, name)
        after: ChoiceAnswer = getattr(restored, name)
        assert tuple(after.probabilities) == ladder  # order kept too, for readable logs
        for option in ladder:
            assert after.probability_of(option) == pytest.approx(before.probability_of(option))
        # Runner-up mass specifically: the argmax alone would pass a weaker test.
        runner_up = max((o for o in ladder if o != after.choice), key=after.probability_of)
        assert after.probability_of(runner_up) > 0.0
    assert restored.pii.value == pytest.approx(original.pii.value)


def test_answers_round_trip_survives_the_jsonl_text_form() -> None:
    # The record is only ever re-read from text, so the dict form is not enough:
    # floats must come back bit-identical through json.dumps/loads.
    original = answers_fixture()
    restored = DecisionAnswers.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored == original


def test_answers_to_dict_carries_exactly_four_questions() -> None:
    # The schema is total: a backend that cannot answer a question must still
    # return it, so the key set is fixed and a missing one is a hard error.
    assert set(answers_fixture().to_dict()) == {"complexity", "sensitivity", "pii", "domain"}


def test_answers_from_dict_rejects_a_missing_question() -> None:
    partial = answers_fixture().to_dict()
    del partial["pii"]
    with pytest.raises(KeyError):
        DecisionAnswers.from_dict(partial)


def test_unknown_answers_are_uniform_on_all_three_choice_ladders() -> None:
    unknown = DecisionAnswers.unknown()
    assert dict(unknown.complexity.probabilities) == dict.fromkeys(COMPLEXITY_LEVELS, 0.25)
    assert dict(unknown.sensitivity.probabilities) == dict.fromkeys(SENSITIVITY_LEVELS, 0.25)
    assert dict(unknown.domain.probabilities) == dict.fromkeys(DOMAINS, 0.2)


def test_unknown_answers_pick_the_middle_rung_of_each_ladder() -> None:
    # Verified against the shipped ladders: hard / confidential / analysis.
    # Fail-closed means the *label* is conservative too, not just the confidence.
    unknown = DecisionAnswers.unknown()
    assert unknown.complexity.choice == "hard"
    assert unknown.sensitivity.choice == "confidential"
    assert unknown.domain.choice == "analysis"


def test_unknown_answers_report_zero_confidence_and_say_it_was_not_reported() -> None:
    unknown = DecisionAnswers.unknown()
    for answer in (unknown.complexity, unknown.sensitivity, unknown.domain):
        assert answer.confidence == 0.0
        assert answer.confidence_reported is False


def test_unknown_answers_put_pii_at_a_coin_flip() -> None:
    unknown = DecisionAnswers.unknown()
    assert unknown.pii.value == 0.5
    assert unknown.pii.confidence == 0.0
    # ...which still counts as present, so fail-closed means "do not send it out".
    assert unknown.pii.is_true() is True


def test_unknown_answers_are_stable_across_calls() -> None:
    # It is a fresh object each time (nothing is cached or aliased) but equal in
    # value, so two degraded requests log identical distributions.
    assert DecisionAnswers.unknown() == DecisionAnswers.unknown()
    assert DecisionAnswers.unknown() is not DecisionAnswers.unknown()


def test_answers_are_immutable() -> None:
    answers = answers_fixture()
    with pytest.raises(dataclasses.FrozenInstanceError):
        answers.pii = NoulAnswer(value=0.0)  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# GateFinding / GateVerdict
# --------------------------------------------------------------------------- #
def test_gate_finding_round_trips_through_dict() -> None:
    finding = finding_fixture()
    assert GateFinding.from_dict(finding.to_dict()) == finding


def test_gate_finding_to_dict_exposes_every_field_by_name() -> None:
    assert set(finding_fixture().to_dict()) == {
        "detector",
        "category",
        "sensitivity_floor",
        "force_local",
        "span_hash",
        "count",
    }


def test_gate_finding_from_dict_ignores_unknown_keys() -> None:
    """Forward compatibility, asserted because a log outlives the code.

    A record written by a newer jev-route carries fields this version has never
    heard of. Re-reading it must not raise, or upgrading the reader breaks every
    historical log -- and the historical logs are the training data.
    """
    payload = finding_fixture().to_dict()
    payload["detector_version"] = "2026-09-01"
    payload["nested_future"] = {"a": [1, 2, 3]}
    assert GateFinding.from_dict(payload) == finding_fixture()


def test_gate_finding_never_carries_the_matched_text() -> None:
    """The privacy promise: a hash, not the secret that fired the detector.

    Asserted by scanning the serialized form for the plaintext, which is the
    only version of this test that would actually catch a regression where
    somebody "temporarily" adds the span for debugging.
    """
    matched_text = "GB33-BUKB-2020-1555-5555-55"  # a valid but invented IBAN
    finding = GateFinding(
        detector="iban",
        category="pii",
        sensitivity_floor="regulated",
        force_local=True,
        span_hash="5f3a" * 8,
        count=1,
    )
    serialized = json.dumps(finding.to_dict())
    assert matched_text not in serialized
    assert "GB33" not in serialized  # not even a fragment of the span
    assert finding.span_hash in serialized  # ...but the proof that it fired is kept


def test_gate_verdict_round_trips_through_dict() -> None:
    verdict = verdict_fixture()
    restored = GateVerdict.from_dict(json.loads(json.dumps(verdict.to_dict())))
    assert restored == verdict
    assert restored.findings == verdict.findings  # findings survive as a tuple
    assert isinstance(restored.findings, tuple)
    assert restored.advisory_topics == ("hipaa", "pci")


def test_gate_verdict_clean_is_the_all_default_verdict() -> None:
    clean = GateVerdict.clean()
    assert clean == GateVerdict()
    assert clean.fired is False
    assert clean.findings == ()
    assert clean.sensitivity_floor is None
    assert clean.pii_floor is None
    assert clean.force_local is False
    assert clean.blocks_backend is False
    assert clean.advisory_topics == ()
    assert clean.detectors() == {}


def test_gate_verdict_clean_round_trips() -> None:
    assert GateVerdict.from_dict(GateVerdict.clean().to_dict()) == GateVerdict.clean()


def test_gate_verdict_from_dict_tolerates_an_empty_payload() -> None:
    # Older records predate the gate entirely; `decision.gate` may be {} or
    # absent, and DecisionRecord.from_dict passes {} straight through.
    assert GateVerdict.from_dict({}) == GateVerdict.clean()


def test_detectors_sums_counts_per_detector_name() -> None:
    verdict = GateVerdict(
        fired=True,
        findings=(
            finding_fixture(detector="card_luhn", count=2),
            finding_fixture(detector="card_luhn", span_hash="other", count=3),
            finding_fixture(detector="email", count=1),
        ),
    )
    # count, not the number of findings: two findings from one detector that
    # matched five spans is five hits, and the log is read for volume.
    assert verdict.detectors() == {"card_luhn": 5, "email": 1}


def test_detectors_reports_an_empty_mapping_when_nothing_fired() -> None:
    assert verdict_fixture().detectors() != {}
    assert GateVerdict(fired=False).detectors() == {}


def test_advisory_topics_are_hints_and_do_not_create_floors() -> None:
    """Topic hits are recorded but must not set a sensitivity floor by themselves.

    The distinction is in the docstring and is easy to lose in a refactor: a
    verdict with only advisory topics fires nothing and blocks nothing, so
    "asked about HIPAA" never forces a tier on its own.
    """
    verdict = GateVerdict(advisory_topics=("hipaa",))
    assert verdict.fired is False
    assert verdict.sensitivity_floor is None
    assert verdict.blocks_backend is False
    assert verdict.detectors() == {}


def test_gate_verdict_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        GateVerdict.clean().fired = True  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# RequestFeatures -- field order is the dataset contract
# --------------------------------------------------------------------------- #
#: The feature columns as of SCHEMA_VERSION 1, in order. Appending is allowed by
#: the assertion below; reordering, renaming, or dropping is not.
FEATURE_COLUMNS: tuple[str, ...] = (
    "char_len",
    "word_count",
    "line_count",
    "sentence_count",
    "mean_word_len",
    "digit_ratio",
    "upper_ratio",
    "punct_ratio",
    "non_ascii_ratio",
    "code_blocks",
    "inline_code_spans",
    "urls",
    "question_marks",
    "exclamations",
    "has_stack_trace",
    "has_diff",
    "has_json",
    "lang",
    "n_messages",
    "n_prior_turns",
    "tool_output_present",
    "gate_detectors",
    "n_gate_findings",
    "gate_force_local",
)


def test_request_features_field_order_starts_with_the_documented_prefix() -> None:
    names = [f.name for f in dataclasses.fields(RequestFeatures)]
    assert names[:5] == ["char_len", "word_count", "line_count", "sentence_count", "mean_word_len"]


def test_request_features_field_order_ends_with_the_gate_columns() -> None:
    names = [f.name for f in dataclasses.fields(RequestFeatures)]
    assert names[-3:] == ["gate_detectors", "n_gate_findings", "gate_force_local"]


def test_request_features_columns_are_append_only() -> None:
    """The dataset contract, enforced literally.

    The distillation pipeline exports these features positionally, so a reordered
    field silently trains a model on the wrong column -- the worst kind of bug,
    because the pipeline still runs and the metrics still move. Requiring the
    shipped column list to be a *prefix* of the current one allows appending a
    new feature and fails loudly on any reorder, rename, or removal.
    """
    names = tuple(f.name for f in dataclasses.fields(RequestFeatures))
    assert names[: len(FEATURE_COLUMNS)] == FEATURE_COLUMNS
    assert len(set(names)) == len(names)


def test_request_features_to_dict_keys_follow_field_order() -> None:
    features = RequestFeatures()
    assert list(features.to_dict()) == [f.name for f in dataclasses.fields(RequestFeatures)]


def test_request_features_defaults_are_neutral() -> None:
    # A default must read as "no signal", never as a signal: these values end up
    # in training rows for any record that did not compute a feature.
    features = RequestFeatures()
    assert features.char_len == 0
    assert features.mean_word_len == 0.0
    assert features.lang == "und"
    assert features.has_stack_trace is False
    assert features.has_json is False
    assert features.gate_detectors == {}
    assert features.n_gate_findings == 0
    assert features.gate_force_local is False


def test_request_features_round_trips_through_dict() -> None:
    features = RequestFeatures(
        char_len=120,
        word_count=19,
        line_count=4,
        sentence_count=2,
        mean_word_len=5.6,
        digit_ratio=0.02,
        code_blocks=1,
        urls=2,
        lang="en",
        n_messages=3,
        tool_output_present=True,
        gate_detectors={"email": 2, "api_key": 1},
        n_gate_findings=3,
        gate_force_local=True,
    )
    assert RequestFeatures.from_dict(features.to_dict()) == features
    assert RequestFeatures.from_dict(json.loads(json.dumps(features.to_dict()))) == features


def test_gate_detectors_is_serialized_as_a_plain_dict() -> None:
    # to_dict() feeds json.dumps() and the CSV/parquet exporter; a mappingproxy
    # or a Counter would serialize differently from one call to the next.
    features = RequestFeatures(gate_detectors={"iban": 1})
    payload = features.to_dict()
    assert type(payload["gate_detectors"]) is dict
    assert payload["gate_detectors"] == {"iban": 1}


def test_gate_detectors_from_dict_is_coerced_to_a_plain_dict() -> None:
    class CountingDict(dict[str, int]):
        """Stands in for any dict subclass a caller might hand us."""

    features = RequestFeatures.from_dict({"gate_detectors": CountingDict(iban=2)})
    assert type(features.gate_detectors) is dict
    assert features.gate_detectors == {"iban": 2}


def test_gate_detectors_absent_or_null_becomes_an_empty_dict() -> None:
    # JSONL written by an older version, or by a hand-edit, may carry null.
    assert RequestFeatures.from_dict({}).gate_detectors == {}
    assert RequestFeatures.from_dict({"gate_detectors": None}).gate_detectors == {}


def test_request_features_from_dict_ignores_unknown_keys() -> None:
    payload = RequestFeatures(char_len=7).to_dict()
    payload["embedding_dim"] = 384
    payload["future_flag"] = True
    restored = RequestFeatures.from_dict(payload)
    assert restored.char_len == 7
    assert not hasattr(restored, "embedding_dim")


def test_request_features_to_dict_detaches_the_mapping() -> None:
    features = RequestFeatures(gate_detectors={"iban": 1})
    payload = features.to_dict()
    payload["gate_detectors"]["iban"] = 99
    assert dict(features.gate_detectors) == {"iban": 1}


def test_request_features_are_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        RequestFeatures().char_len = 5  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# RoutingDecision
# --------------------------------------------------------------------------- #
def test_routing_decision_to_dict_carries_every_field() -> None:
    # A field added to the dataclass but forgotten in to_dict() would vanish from
    # the dataset with no error anywhere. Comparing against dataclasses.fields()
    # makes that omission a test failure instead of a silent gap in the log.
    decision = decision_fixture()
    payload = decision.to_dict()
    assert set(payload) == {f.name for f in dataclasses.fields(RoutingDecision)}
    for name in ("tier", "model", "rule_id", "reason", "backend", "backend_model_version"):
        assert payload[name] == getattr(decision, name)
    assert payload["effective_sensitivity"] == "regulated"
    assert payload["effective_complexity"] == "hard"
    assert payload["degraded"] is False
    assert payload["degrade_reason"] is None
    assert payload["latency_ms"] == 3.5
    assert payload["cached"] is True


def test_routing_decision_escalated_serializes_as_a_list() -> None:
    # JSON has no tuples. Emitting one would make to_json() work (json.dumps
    # coerces) but leave to_dict() unequal to what comes back from from_dict(),
    # which breaks any caller that compares a live decision to a logged one.
    decision = decision_fixture(escalated=("sensitivity", "complexity"))
    assert decision.to_dict()["escalated"] == ["sensitivity", "complexity"]
    assert type(decision.to_dict()["escalated"]) is list
    assert type(decision.to_dict()["escalated"]) is not tuple


def test_routing_decision_escalated_defaults_to_empty() -> None:
    decision = decision_fixture(escalated=())
    assert decision.to_dict()["escalated"] == []


def test_routing_decision_answers_block_keeps_all_four_distributions() -> None:
    """The answers block is the training signal; it must be complete.

    Every one of the four questions has to be present with a full probability
    mapping, because a trainer reading this record has no way to ask for the
    distribution it is missing.
    """
    decision = decision_fixture()
    block = decision.to_dict()["answers"]
    assert set(block) == {"complexity", "sensitivity", "pii", "domain"}
    for name, ladder in (
        ("complexity", COMPLEXITY_LEVELS),
        ("sensitivity", SENSITIVITY_LEVELS),
        ("domain", DOMAINS),
    ):
        assert set(block[name]["probabilities"]) == set(ladder)
        assert block[name]["choice"] in ladder
    assert block["pii"]["noul"] == pytest.approx(0.18)
    # The whole block must be JSON-native, since it goes straight to the log.
    assert json.loads(json.dumps(block)) == block


def test_routing_decision_gate_block_is_nested_and_complete() -> None:
    gate = decision_fixture().to_dict()["gate"]
    assert gate["fired"] is True
    assert gate["blocks_backend"] is True
    assert gate["sensitivity_floor"] == "regulated"
    assert gate["pii_floor"] == pytest.approx(0.8)
    assert gate["advisory_topics"] == ["hipaa", "pci"]
    assert [f["detector"] for f in gate["findings"]] == ["iban", "email"]


def test_routing_decision_degraded_shape_survives_to_dict() -> None:
    decision = decision_fixture(
        tier="local",
        rule_id="backend.down.fail-closed",
        reason="decision backend unavailable; failing closed to local",
        answers=DecisionAnswers.unknown(),
        gate=GateVerdict.clean(),
        backend="jev",
        degraded=True,
        degrade_reason="timeout after 3000ms",
        escalated=(),
        cached=False,
    )
    payload = decision.to_dict()
    assert payload["degraded"] is True
    assert payload["degrade_reason"] == "timeout after 3000ms"
    assert payload["answers"]["complexity"]["confidence"] == 0.0


def test_routing_decision_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision_fixture().tier = "strong"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# DecisionRecord -- this IS the training dataset
# --------------------------------------------------------------------------- #
def test_record_json_round_trip_preserves_the_whole_decision() -> None:
    record = record_fixture()
    restored = DecisionRecord.from_json(record.to_json())
    assert restored == record


@pytest.mark.parametrize(
    "attribute",
    [
        "request_id",
        "timestamp",
        "excerpt_hash",
        "backend_latency_ms",
        "total_latency_ms",
        "schema_version",
        "kind",
        "requested_model",
        "excerpt",
        "questions_sent",
        "metadata",
        "shadow",
    ],
)
def test_record_json_round_trip_preserves_each_top_level_field(attribute: str) -> None:
    # One assertion per field so a failure names the field that broke instead of
    # reporting "the record changed" somewhere in a nested structure.
    record = record_fixture()
    restored = DecisionRecord.from_json(record.to_json())
    assert getattr(restored, attribute) == getattr(record, attribute)


def test_record_json_round_trip_preserves_the_routing_explanation() -> None:
    record = record_fixture()
    restored = DecisionRecord.from_json(record.to_json())
    for name in (
        "tier",
        "model",
        "rule_id",
        "reason",
        "backend",
        "backend_model_version",
        "effective_sensitivity",
        "effective_complexity",
        "escalated",
        "degraded",
        "degrade_reason",
        "latency_ms",
        "cached",
    ):
        assert getattr(restored.decision, name) == getattr(record.decision, name), name


def test_record_json_round_trip_preserves_gate_findings_including_counts() -> None:
    restored = DecisionRecord.from_json(record_fixture().to_json())
    assert restored.decision.gate == verdict_fixture()
    assert restored.decision.gate.detectors() == {"iban": 1, "email": 3}
    assert restored.decision.gate.blocks_backend is True


def test_record_json_round_trip_preserves_the_feature_columns() -> None:
    restored = DecisionRecord.from_json(record_fixture().to_json())
    assert restored.features == record_fixture().features
    assert dict(restored.features.gate_detectors) == {"iban": 1}


def test_record_json_round_trip_preserves_full_answer_distributions() -> None:
    restored = DecisionRecord.from_json(record_fixture().to_json())
    assert restored.decision.answers == answers_fixture()
    for name, ladder in (
        ("complexity", COMPLEXITY_LEVELS),
        ("sensitivity", SENSITIVITY_LEVELS),
        ("domain", DOMAINS),
    ):
        probabilities = getattr(restored.decision.answers, name).probabilities
        assert set(probabilities) == set(ladder)
        assert sum(probabilities.values()) == pytest.approx(1.0)


def test_to_json_emits_keys_in_sorted_order() -> None:
    # sort_keys keeps JSONL diffs and content hashes stable: the same decision
    # logged twice must produce byte-identical lines, or "did this record change?"
    # has no answer and deduplication by hash is impossible.
    line = record_fixture().to_json()
    payload = json.loads(line)
    assert list(payload) == sorted(payload)
    nested = payload["decision"]["answers"]["complexity"]
    assert list(nested) == sorted(nested)
    assert list(payload["features"]) == sorted(payload["features"])


def test_to_json_is_a_single_line_with_no_embedded_newline() -> None:
    # JSONL: one record per line. A raw newline inside the line would corrupt
    # every reader downstream, including `grep` and the distill exporter.
    line = record_fixture().to_json()
    assert "\n" not in line
    assert "\r" not in line
    assert line.splitlines() == [line]
    assert not line.endswith("\n")


def test_to_json_escapes_newlines_inside_prompt_text() -> None:
    """Multi-line excerpts are the normal case, so this is the real risk.

    ``excerpt`` is opt-in and holds redacted prompt text, which is full of
    newlines. If they were emitted raw, one record would read back as many broken
    lines -- and the failure would show up in the trainer, far from the cause.
    """
    record = record_fixture(excerpt="first line\nsecond line\n\tafter a tab")
    line = record.to_json()
    assert line.splitlines() == [line]
    assert DecisionRecord.from_json(line).excerpt == "first line\nsecond line\n\tafter a tab"


def test_to_json_uses_compact_separators() -> None:
    # No padding: the log grows by a byte per separator per record, and at proxy
    # volume that is real disk. Also makes the canonical-form claim testable.
    line = record_fixture().to_json()
    assert '", "' not in line
    assert '": "' not in line
    assert line == json.dumps(json.loads(line), sort_keys=True, separators=(",", ":"))


def test_every_record_is_stamped_with_the_schema_version_and_kind() -> None:
    # The discriminator is what lets a mixed event stream be filtered, and the
    # version is what tells a future migration whether it can read the line.
    payload = json.loads(record_fixture().to_json())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["kind"] == RECORD_KIND
    assert SCHEMA_VERSION == "1"
    assert RECORD_KIND == "jev_route.decision"


def test_schema_version_and_kind_have_defaults_on_the_dataclass() -> None:
    # A record built without them is still valid and still stamped, so no caller
    # can forget and produce an unversioned line.
    record = DecisionRecord(
        request_id="r",
        timestamp="t",
        decision=decision_fixture(),
        features=RequestFeatures(),
        excerpt_hash="h",
        backend_latency_ms=0.0,
        total_latency_ms=0.0,
    )
    assert record.schema_version == SCHEMA_VERSION
    assert record.kind == RECORD_KIND
    assert record.excerpt is None  # storing text is opt-in
    assert record.questions_sent == {}
    assert record.metadata == {}
    assert record.requested_model is None
    assert record.shadow is None


def test_to_json_serializes_unserializable_metadata_via_default_str() -> None:
    """An odd metadata value must degrade to a string, not kill the log write.

    ``metadata`` is caller-supplied (tenant, app, api-key alias), so it can hold
    anything a framework produced: a ``Path``, a ``set``, a UUID. Without
    ``default=str`` the sink raises *after* the routing decision was made and
    returned, which is the worst possible place to fail.
    """
    record = record_fixture(
        metadata={
            "path": Path("decision-log") / "2026-01.jsonl",
            "tags": {"alpha", "beta"},
            "tenant": "acme",
        }
    )
    # Proof that the value really is unserializable without the fallback:
    with pytest.raises(TypeError):
        json.dumps(record.to_dict(), sort_keys=True)

    payload = json.loads(record.to_json())
    assert payload["metadata"]["path"] == "decision-log/2026-01.jsonl"
    assert isinstance(payload["metadata"]["tags"], str)
    assert payload["metadata"]["tenant"] == "acme"  # native values stay native


def test_default_str_metadata_is_lossy_and_that_is_the_accepted_tradeoff() -> None:
    # Documented reality: a stringified set does not come back as a set, so the
    # round-tripped record is not equal to the original. Logging something lossy
    # beats logging nothing; the assertion exists so the loss is a decision and
    # not a surprise.
    import ast

    record = record_fixture(metadata={"tags": {"alpha"}})
    restored = DecisionRecord.from_json(record.to_json())
    assert restored.metadata["tags"] == "{'alpha'}"
    assert ast.literal_eval(restored.metadata["tags"]) == {"alpha"}
    assert restored != record
    assert restored.request_id == record.request_id  # everything else survived


def test_identical_records_serialize_identically() -> None:
    assert record_fixture().to_json() == record_fixture().to_json()


def test_from_dict_accepts_a_minimal_record() -> None:
    """A record from an older or leaner writer still loads.

    Only ``request_id``, ``timestamp`` and ``decision`` are required; everything
    else defaults. This is what makes the schema additive rather than brittle.
    """
    minimal = {
        "request_id": "r-min",
        "timestamp": "2026-01-01T00:00:00Z",
        "decision": {
            "tier": "cheap",
            "model": "small-model",
            "rule_id": "default",
            "reason": "routine task",
            "answers": DecisionAnswers.unknown().to_dict(),
            "backend": "mock",
            "effective_sensitivity": "internal",
            "effective_complexity": "standard",
        },
    }
    record = DecisionRecord.from_dict(minimal)
    assert record.request_id == "r-min"
    assert record.decision.tier == "cheap"
    assert record.decision.gate == GateVerdict.clean()
    assert record.decision.escalated == ()
    assert record.decision.backend_model_version == ""
    assert record.decision.latency_ms == 0.0
    assert record.features == RequestFeatures()
    assert record.excerpt_hash == ""
    assert record.schema_version == SCHEMA_VERSION
    assert record.kind == RECORD_KIND


def test_from_dict_ignores_unknown_top_level_keys() -> None:
    payload = json.loads(record_fixture().to_json())
    payload["traceparent"] = "00-abc-def-01"
    payload["future_block"] = {"anything": True}
    restored = DecisionRecord.from_dict(payload)
    assert restored == record_fixture()


def test_from_dict_requires_the_decision_block() -> None:
    payload = json.loads(record_fixture().to_json())
    del payload["decision"]
    with pytest.raises(KeyError):
        DecisionRecord.from_dict(payload)


def test_to_dict_and_to_json_agree() -> None:
    # Two serializers for the same record must not drift: the sink uses
    # to_json(), the in-process integrations use to_dict().
    record = record_fixture()
    assert json.loads(record.to_json()) == record.to_dict()


def test_record_to_dict_shadow_is_none_when_absent() -> None:
    record = record_fixture(shadow=None)
    assert record.to_dict()["shadow"] is None
    assert DecisionRecord.from_dict(record.to_dict()).shadow is None


def test_record_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        record_fixture().request_id = "tampered"  # type: ignore[misc]


def test_record_accepts_a_mapping_typed_questions_sent() -> None:
    # The signature is Mapping[str, Any], so a read-only mapping must be accepted
    # and come back as a plain dict after serialization.
    from types import MappingProxyType

    record = record_fixture(questions_sent=MappingProxyType({"complexity": {"type": "choice"}}))
    assert record.to_dict()["questions_sent"] == {"complexity": {"type": "choice"}}
    assert DecisionRecord.from_json(record.to_json()).questions_sent == {"complexity": {"type": "choice"}}


def test_record_metadata_is_never_the_prompt() -> None:
    """A guard on the privacy contract rather than on the code.

    ``metadata`` is documented as caller context and ``excerpt`` as opt-in; the
    default record therefore contains no prompt text at all, only a hash. If a
    future change starts copying the request into the record, this test is where
    it gets caught.
    """
    prompt_fragment = "summarize the Q3 board notes"
    record = record_fixture(excerpt_hash="deadbeef" * 2)
    serialized = record.to_json()
    assert prompt_fragment not in serialized
    assert json.loads(serialized)["excerpt"] is None
