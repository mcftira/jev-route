"""Tests for :mod:`jev_route.router` -- the chain, in order.

The router is where every other module's promise becomes observable, so these tests
are written against the eight numbered steps in its docstring: excerpt, gate, redact,
decide, merge, escalate, evaluate, log. Each class below owns one link, plus a few
classes for the properties that only exist *between* links.

The three assertions that matter most, and why:

* **A gate-blocked request sends nothing anywhere.** ``backend == "gate"``,
  ``questions_sent == {}``, and no excerpt in the log even when the operator turned
  ``excerpt_mode: redacted`` on. Redaction is not the boundary; the gate is.
* **A topic mention is not a floor.** A HIPAA *question* with no data in it must be
  classified normally, not forced local by a keyword.
* **A config knob cannot become a data-egress path.** ``fail_open`` is honoured for
  quality risk, but never when the gate found confidential or regulated data.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from jev_route.backends.mock import MockBackend
from jev_route.cache import InMemoryTTLCache, NullCache
from jev_route.gate import DEFAULT_DETECTORS, Detector, HardGate
from jev_route.logging_sink import JsonlSink, iter_records
from jev_route.policy import Policy
from jev_route.router import Router, _merge_gate
from jev_route.schema import (
    COMPLEXITY_LEVELS,
    DOMAINS,
    SENSITIVITY_LEVELS,
    ChoiceAnswer,
    DecisionAnswers,
    DecisionRecord,
    GateVerdict,
    NoulAnswer,
)

from .conftest import FakeBackend, RecordingSink, make_policy_doc

CARD = "4111 1111 1111 1111"
EMAIL = "dana.kovacs@northside-health.org"
TOPIC_HIPAA = "Explain how HIPAA actually works and who it applies to."


def answers(
    *,
    complexity: str = "standard",
    complexity_confidence: float = 0.95,
    sensitivity: str = "public",
    sensitivity_confidence: float = 0.95,
    domain: str = "chat",
    pii: float = 0.02,
) -> DecisionAnswers:
    """A hand-built, fully-specified answer set for merge/escalation arithmetic."""

    def peaked(level: str, ladder: tuple[str, ...], confidence: float) -> ChoiceAnswer:
        share = round((1.0 - confidence) / (len(ladder) - 1), 6)
        probabilities = {x: share for x in ladder if x != level}
        probabilities[level] = round(1.0 - share * (len(ladder) - 1), 6)
        return ChoiceAnswer(level, {k: probabilities[k] for k in ladder}, confidence, True)

    return DecisionAnswers(
        complexity=peaked(complexity, COMPLEXITY_LEVELS, complexity_confidence),
        sensitivity=peaked(sensitivity, SENSITIVITY_LEVELS, sensitivity_confidence),
        pii=NoulAnswer(pii),
        domain=peaked(domain, DOMAINS, 0.9),
    )


@pytest.fixture
def router_policy(policy_doc: dict[str, Any]) -> Policy:
    return Policy.from_dict(policy_doc)


# --------------------------------------------------------------------------- #
# 1-3. The gate runs first, and a blocked request goes nowhere
# --------------------------------------------------------------------------- #
class TestGateBlocksTheCloud:
    async def test_card_number_routes_local_without_calling_a_backend(
        self, router_factory: Any, sink: RecordingSink
    ) -> None:
        backend = FakeBackend()
        router = router_factory(backend)
        decision = await router.route_text(f"Please charge the card {CARD} for invoice 88.")

        assert decision.tier == "local"
        assert decision.rule_id == "gate.force-local"
        assert decision.backend == "gate", "no backend answered this; the gate did"
        assert backend.calls == 0, "nothing may be sent to a decision backend"
        assert decision.gate.blocks_backend is True
        assert decision.effective_sensitivity == "regulated"
        assert decision.answers.sensitivity.probability_of("regulated") == 1.0

    async def test_gate_blocked_record_sends_no_questions(self, router_factory: Any, sink: RecordingSink) -> None:
        router = router_factory(FakeBackend())
        await router.route_text(f"card {CARD}")
        assert sink.last.questions_sent == {}

    async def test_gate_blocked_record_has_no_text_even_in_redacted_mode(
        self, policy_doc: dict[str, Any], sink: RecordingSink
    ) -> None:
        """The same judgement that kept the excerpt out of a cloud API keeps it out of the log."""
        policy_doc["logging"]["excerpt_mode"] = "redacted"
        router = Router(Policy.from_dict(policy_doc), FakeBackend(), sink=sink, cache=NullCache())
        await router.route_text(f"Please charge the card {CARD} for invoice 88.")

        record = sink.last
        assert record.excerpt is None
        assert CARD not in record.to_json()
        assert CARD.replace(" ", "") not in record.to_json()
        # ...but the hash and the features are still there: the row is still trainable.
        assert record.excerpt_hash
        assert record.features.gate_detectors == {"payment_card": 1}
        assert record.features.gate_force_local is True

    async def test_force_local_counts_as_blocked_for_logging_purposes(
        self, policy_doc: dict[str, Any], sink: RecordingSink
    ) -> None:
        """``skip_backend`` is the logging boundary, not ``blocks_backend``.

        A personal identifier the gate refused to classify is treated exactly like a
        card number it refused to send: no text in the log under any excerpt_mode.
        """
        policy_doc["logging"]["excerpt_mode"] = "redacted"
        router = Router(Policy.from_dict(policy_doc), FakeBackend(), sink=sink, cache=NullCache())
        await router.route_text(f"Please forward this to {EMAIL} today.")
        assert sink.last.excerpt is None
        assert EMAIL not in sink.last.to_json()

    async def test_redacted_mode_stores_redacted_text_when_a_backend_was_called(
        self, policy_doc: dict[str, Any], sink: RecordingSink
    ) -> None:
        policy_doc["logging"]["excerpt_mode"] = "redacted"
        policy_doc["gate"]["on_force_local"] = "still_classify"
        router = Router(Policy.from_dict(policy_doc), FakeBackend(), sink=sink, cache=NullCache())
        await router.route_text(f"Please forward this to {EMAIL} today.")

        assert sink.last.excerpt is not None, "text-based distillation is the reason this mode exists"
        assert EMAIL not in sink.last.excerpt, "and it is the REDACTED excerpt that gets stored"
        assert "[email_address]" in sink.last.excerpt
        assert EMAIL not in sink.last.to_json()

    async def test_redacted_mode_stores_plain_text_for_a_clean_request(
        self, policy_doc: dict[str, Any], sink: RecordingSink
    ) -> None:
        policy_doc["logging"]["excerpt_mode"] = "redacted"
        router = Router(Policy.from_dict(policy_doc), FakeBackend(), sink=sink, cache=NullCache())
        await router.route_text("Summarize the roadmap for the whole team.")
        assert sink.last.excerpt == "Summarize the roadmap for the whole team."

    async def test_hash_mode_never_stores_text(self, router_factory: Any, sink: RecordingSink) -> None:
        router = router_factory(FakeBackend(), excerpt_mode="hash")
        await router.route_text("Summarize the roadmap for the whole team.")
        assert sink.last.excerpt is None
        assert sink.last.excerpt_hash

    async def test_none_mode_never_stores_text(self, router_factory: Any, sink: RecordingSink) -> None:
        router = router_factory(FakeBackend(), excerpt_mode="none")
        await router.route_text("Summarize the roadmap for the whole team.")
        assert sink.last.excerpt is None

    async def test_gate_blocked_answers_are_honest_uncertainty(self, router_factory: Any, sink: RecordingSink) -> None:
        """Nobody classified it, so complexity and domain must stay uniform."""
        router = router_factory(FakeBackend())
        decision = await router.route_text(f"card {CARD}")
        for level in COMPLEXITY_LEVELS:
            assert decision.answers.complexity.probability_of(level) == pytest.approx(0.25)
        for level in DOMAINS:
            assert decision.answers.domain.probability_of(level) == pytest.approx(0.2)
        # Sensitivity is the exception: a checksum-validated identifier is not a guess.
        assert decision.answers.sensitivity.confidence == 1.0

    async def test_gate_blocked_decision_is_not_marked_degraded(self, router_factory: Any, sink: RecordingSink) -> None:
        """A gate block is a success, not an outage: fail-closed must not swallow it."""
        decision = await router_factory(FakeBackend(degraded=True)).route_text(f"card {CARD}")
        assert decision.degraded is False
        assert decision.rule_id == "gate.force-local"

    async def test_force_local_skips_the_backend_by_default(self, router_factory: Any, sink: RecordingSink) -> None:
        backend = FakeBackend()
        decision = await router_factory(backend).route_text(f"Send it to {EMAIL} please.")
        assert backend.calls == 0
        assert decision.backend == "gate"
        assert decision.tier == "local"

    async def test_still_classify_calls_the_backend_with_redacted_text_only(
        self, policy_doc: dict[str, Any], sink: RecordingSink
    ) -> None:
        """``still_classify`` buys complexity labels for sensitive prompts; it must not buy egress."""
        policy_doc["gate"]["on_force_local"] = "still_classify"
        backend = FakeBackend()
        router = Router(Policy.from_dict(policy_doc), backend, sink=sink, cache=NullCache())
        decision = await router.route_text(f"Send the report to {EMAIL} before Friday.")

        assert backend.calls == 1
        sent = backend.requests[0]
        assert EMAIL not in sent.redacted_excerpt
        assert "[email_address]" in sent.redacted_excerpt
        assert decision.backend == "fake"
        assert decision.gate.force_local is True

    async def test_blocking_detectors_are_never_classified_even_with_still_classify(
        self, policy_doc: dict[str, Any], sink: RecordingSink
    ) -> None:
        policy_doc["gate"]["on_force_local"] = "still_classify"
        backend = FakeBackend()
        router = Router(Policy.from_dict(policy_doc), backend, sink=sink, cache=NullCache())
        decision = await router.route_text(f"card {CARD}")
        assert backend.calls == 0, "blocks_backend outranks on_force_local"
        assert decision.backend == "gate"

    async def test_the_gate_scans_the_raw_excerpt_not_the_redacted_one(
        self, policy_doc: dict[str, Any], sink: RecordingSink
    ) -> None:
        """Ordering is a privacy property: gate, then redact. Never the reverse."""
        seen: dict[str, Any] = {}

        class Watching(HardGate):
            def redaction_map(self, text: str) -> Any:
                seen["redacted_input"] = text
                return super().redaction_map(text)

            def scan(self, text: str) -> Any:
                seen["scanned_input"] = text
                return super().scan(text)

        router = Router(Policy.from_dict(policy_doc), FakeBackend(), gate=Watching(), sink=sink, cache=NullCache())
        await router.route_text(f"Send it to {EMAIL} now.")
        assert seen["scanned_input"] == seen["redacted_input"]
        assert EMAIL in seen["scanned_input"]

    async def test_gate_runs_on_every_request_including_trivial_ones(
        self, router_factory: Any, sink: RecordingSink
    ) -> None:
        router = router_factory(FakeBackend())
        for text in ("hi", "thanks!", "ok"):
            decision = await router.route_text(text)
            assert isinstance(decision.gate, GateVerdict)


# --------------------------------------------------------------------------- #
# Advisory topics
# --------------------------------------------------------------------------- #
class TestAdvisoryTopicsAreHintsNotFloors:
    @pytest.mark.parametrize(
        "text", [TOPIC_HIPAA, "What is a credit card CVV?", "What does PCI-DSS require of merchants?"]
    )
    async def test_topic_question_is_classified_normally(
        self, router_factory: Any, sink: RecordingSink, text: str
    ) -> None:
        """A backend that reads the question as public gets the cheap tier. Full stop."""
        backend = FakeBackend(sensitivity="public", sensitivity_confidence=0.95, pii=0.02)
        router = router_factory(backend)
        decision = await router.route_text(text)

        assert decision.gate.advisory_topics, "the topic was seen"
        assert decision.gate.sensitivity_floor is None, "and it set no floor"
        assert decision.gate.force_local is False
        assert decision.gate.blocks_backend is False
        assert backend.calls == 1
        assert decision.effective_sensitivity == "public"
        assert decision.tier == "cheap"
        assert decision.escalated == ()

    async def test_topic_hints_are_forwarded_to_the_backend(self, router_factory: Any, sink: RecordingSink) -> None:
        backend = FakeBackend(sensitivity="public", sensitivity_confidence=0.95)
        await router_factory(backend).route_text(TOPIC_HIPAA)
        assert backend.requests[0].advisory_topics == ("kw_health_regulation",)

    async def test_topic_hint_appears_in_the_backend_state(self, router_factory: Any, sink: RecordingSink) -> None:
        backend = FakeBackend(sensitivity="public", sensitivity_confidence=0.95)
        await router_factory(backend).route_text(TOPIC_HIPAA)
        state = backend.requests[0].state()
        assert state["local_gate_topic_hints"] == ["kw_health_regulation"]

    async def test_mock_backend_is_not_forced_local_by_the_gate(self, router_factory: Any, sink: RecordingSink) -> None:
        """The mock may still choose local -- that is its judgement, not the gate's.

        MockBackend deliberately over-weights regulated vocabulary (it is a coarse
        keyword scorer standing in for a model), so this test asserts the *provenance*
        rather than the tier: no floor, no block, no gate escalation, and a backend
        that was actually consulted. The tier-level version of this promise lives in
        :meth:`test_topic_question_is_classified_normally` above.
        """
        decision = await router_factory(MockBackend()).route_text(TOPIC_HIPAA)
        assert decision.backend == "mock"
        assert decision.gate.sensitivity_floor is None
        assert decision.gate.force_local is False
        assert not any("local gate floor" in note for note in decision.escalated)

    async def test_advisory_topic_is_recorded_for_training(self, router_factory: Any, sink: RecordingSink) -> None:
        await router_factory(FakeBackend(sensitivity="public", sensitivity_confidence=0.95)).route_text(TOPIC_HIPAA)
        assert sink.last.decision.gate.to_dict()["advisory_topics"] == ["kw_health_regulation"]
        assert [f.detector for f in sink.last.decision.gate.findings] == ["kw_health_regulation"]


# --------------------------------------------------------------------------- #
# 5. Merging the gate floor over the backend answer
# --------------------------------------------------------------------------- #
class TestGateFloorMerging:
    def test_merge_takes_the_max_never_the_min(self) -> None:
        (complexity, sensitivity, pii), notes = _merge_gate(
            answers(sensitivity="public", pii=0.05),
            GateVerdict(fired=True, sensitivity_floor="regulated", pii_floor=1.0),
        )
        assert sensitivity == "regulated"
        assert complexity == "standard", "the gate says nothing about complexity"
        assert pii == 1.0
        assert any("local gate floor" in note for note in notes)

    def test_merge_leaves_a_stricter_backend_answer_alone(self) -> None:
        (_, sensitivity, _), notes = _merge_gate(
            answers(sensitivity="regulated"),
            GateVerdict(fired=True, sensitivity_floor="confidential", pii_floor=None),
        )
        assert sensitivity == "regulated"
        assert notes == []

    def test_merge_with_a_clean_verdict_changes_nothing(self) -> None:
        (complexity, sensitivity, pii), notes = _merge_gate(
            answers(sensitivity="internal", pii=0.4), GateVerdict.clean()
        )
        assert (complexity, sensitivity, pii) == ("standard", "internal", 0.4)
        assert notes == []

    def test_merge_does_not_lower_the_pii(self) -> None:
        (_, _, pii), notes = _merge_gate(answers(pii=0.9), GateVerdict(fired=True, pii_floor=0.2))
        assert pii == 0.9
        assert notes == []

    @pytest.mark.parametrize("backend_says", ["public", "internal", "confidential"])
    def test_a_regulated_floor_always_wins(self, backend_says: str) -> None:
        (_, sensitivity, _), _ = _merge_gate(
            answers(sensitivity=backend_says),
            GateVerdict(fired=True, sensitivity_floor="regulated", pii_floor=1.0),
        )
        assert sensitivity == "regulated"

    async def test_full_chain_floor_merge_over_a_public_backend_answer(
        self, policy_doc: dict[str, Any], sink: RecordingSink
    ) -> None:
        """End-to-end: the backend says public, the gate says regulated, regulated wins.

        A floor-setting detector that neither blocks nor forces local does not exist in
        DEFAULT_DETECTORS (every hard detector does one or the other), so the gate here
        carries one custom detector. That is the point: the merge is a property of the
        router, not of the shipped rule set.
        """
        floor_only = Detector(
            name="acme_export_controlled",
            category="regulated",
            pattern=re.compile(r"\bZZTOP\b", re.IGNORECASE),
            sensitivity_floor="regulated",
            pii=False,  # isolate the sensitivity floor from the pii floor
        )
        policy_doc["gate"]["on_force_local"] = "still_classify"
        backend = FakeBackend(sensitivity="public", sensitivity_confidence=0.99, pii=0.01)
        router = Router(
            Policy.from_dict(policy_doc), backend, gate=HardGate(detectors=(floor_only,)), sink=sink, cache=NullCache()
        )

        decision = await router.route_text("The ZZTOP protocol specification is a routine read.")
        assert backend.calls == 1, "the backend really did answer 'public'"
        assert decision.effective_sensitivity == "regulated"
        assert decision.tier == "local"
        assert decision.rule_id == "data.sensitive"
        assert any("local gate floor: regulated" in note for note in decision.escalated)

    async def test_gate_floor_is_recorded_on_the_decision(self, router_factory: Any, sink: RecordingSink) -> None:
        decision = await router_factory(FakeBackend()).route_text(f"card {CARD}")
        assert decision.gate.to_dict()["sensitivity_floor"] == "regulated"
        assert decision.effective_sensitivity == "regulated"


# --------------------------------------------------------------------------- #
# 6. Uncertainty escalation
# --------------------------------------------------------------------------- #
class TestUncertaintyEscalation:
    async def test_low_sensitivity_confidence_escalates_one_level(
        self, policy_doc: dict[str, Any], sink: RecordingSink
    ) -> None:
        """ "40% sure it is internal" must be treated as "confidential".

        This measures the one-level GRADIENT, so the separate too-uncertain-to-egress
        floor is switched off -- at confidence 0.4 that floor would otherwise take
        over and escalate straight to the top of the ladder, hiding the behaviour
        under test. ``test_very_low_sensitivity_confidence_forces_the_safest_tier``
        covers the floor itself.
        """
        backend = FakeBackend(sensitivity="internal", sensitivity_confidence=0.4)
        router = Router(Policy.from_dict(_gradient_only(policy_doc)), backend, sink=sink, cache=NullCache())
        decision = await router.route_text("An ordinary internal status update about the roadmap.")
        assert decision.effective_sensitivity == "confidential"
        assert any(note.startswith("sensitivity internal->confidential") for note in decision.escalated)
        assert any("0.40 < 0.8" in note for note in decision.escalated)
        assert decision.tier == "local"

    async def test_high_sensitivity_confidence_does_not_escalate(
        self, router_factory: Any, sink: RecordingSink
    ) -> None:
        backend = FakeBackend(sensitivity="internal", sensitivity_confidence=0.95)
        decision = await router_factory(backend).route_text("An ordinary internal status update about the roadmap.")
        assert decision.effective_sensitivity == "internal"
        assert decision.escalated == ()
        assert decision.tier == "cheap"

    async def test_low_complexity_confidence_escalates(self, router_factory: Any, sink: RecordingSink) -> None:
        backend = FakeBackend(
            complexity="standard", complexity_confidence=0.2, sensitivity="public", sensitivity_confidence=0.95
        )
        decision = await router_factory(backend).route_text("Do the thing with the data.")
        assert decision.effective_complexity == "hard"
        assert any(note.startswith("complexity standard->hard") for note in decision.escalated)
        assert decision.tier == "strong"

    async def test_complexity_escalation_respects_its_own_threshold(
        self, router_factory: Any, sink: RecordingSink
    ) -> None:
        backend = FakeBackend(
            complexity="standard", complexity_confidence=0.9, sensitivity="public", sensitivity_confidence=0.95
        )
        decision = await router_factory(backend).route_text("Do the thing with the data.")
        assert decision.effective_complexity == "standard"
        assert decision.escalated == ()

    async def test_escalation_clamps_at_the_top_of_the_ladder(self, router_factory: Any, sink: RecordingSink) -> None:
        backend = FakeBackend(
            sensitivity="regulated", sensitivity_confidence=0.1, complexity="frontier", complexity_confidence=0.1
        )
        decision = await router_factory(backend).route_text("Something at the very top of both ladders.")
        assert decision.effective_sensitivity == "regulated"
        assert decision.effective_complexity == "frontier"
        # A bump that cannot move is not reported as an escalation.
        assert not any(note.startswith("sensitivity regulated->") for note in decision.escalated)
        assert not any(note.startswith("complexity frontier->") for note in decision.escalated)

    async def test_no_escalation_is_reported_when_the_gate_blocked_the_call(
        self, router_factory: Any, sink: RecordingSink
    ) -> None:
        """Regression: nobody classified it, so nothing may claim a confidence bump.

        A gate-blocked decision carries uniform complexity/domain by construction.
        Applying the confidence floor to those would log "complexity hard->frontier"
        for a request no model ever saw -- a fabricated training label.
        """
        decision = await router_factory(FakeBackend()).route_text(f"Please charge the card {CARD}.")
        assert decision.backend == "gate"
        assert decision.escalated == ()
        assert not any("confidence" in note for note in decision.escalated)
        assert decision.effective_complexity == decision.answers.complexity.choice

    async def test_no_confidence_escalation_when_the_backend_is_degraded(
        self, policy_doc: dict[str, Any], sink: RecordingSink
    ) -> None:
        """Same reasoning: a degraded answer is already maximum uncertainty."""
        router = Router(Policy.from_dict(policy_doc), FakeBackend(degraded=True), sink=sink, cache=NullCache())
        decision = await router.route_text("An ordinary request about nothing sensitive.")
        assert decision.degraded is True
        assert not any("confidence" in note for note in decision.escalated)

    async def test_escalation_can_be_turned_off_per_knob(self, policy_doc: dict[str, Any], sink: RecordingSink) -> None:
        policy_doc["on_uncertain"]["sensitivity_confidence_below"] = None
        policy_doc["on_uncertain"]["complexity_confidence_below"] = None
        policy_doc["on_uncertain"]["force_local_confidence_below"] = None
        router = Router(
            Policy.from_dict(policy_doc),
            FakeBackend(sensitivity="internal", sensitivity_confidence=0.05),
            sink=sink,
            cache=NullCache(),
        )
        decision = await router.route_text("An ordinary internal status update.")
        assert decision.effective_sensitivity == "internal"
        assert decision.escalated == ()

    async def test_bump_levels_are_honoured(self, policy_doc: dict[str, Any], sink: RecordingSink) -> None:
        policy_doc["on_uncertain"]["sensitivity_bump_levels"] = 2
        router = Router(
            Policy.from_dict(_gradient_only(policy_doc)),
            FakeBackend(sensitivity="public", sensitivity_confidence=0.1),
            sink=sink,
            cache=NullCache(),
        )
        decision = await router.route_text("Something the backend is very unsure about.")
        assert decision.effective_sensitivity == "confidential"

    async def test_effective_answers_are_what_the_policy_reasoned_about(
        self, policy_doc: dict[str, Any], sink: RecordingSink
    ) -> None:
        """The log shows the effective choice; the raw distribution is preserved beside it."""
        backend = FakeBackend(sensitivity="internal", sensitivity_confidence=0.4)
        router = Router(Policy.from_dict(_gradient_only(policy_doc)), backend, sink=sink, cache=NullCache())
        decision = await router.route_text("An ordinary internal status update.")
        assert decision.answers.sensitivity.choice == "confidential"
        assert decision.answers.sensitivity.probability_of("internal") == pytest.approx(0.4)
        assert decision.answers.sensitivity.confidence == pytest.approx(0.4)


def _gradient_only(doc: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``doc`` with the too-uncertain-to-egress floor disabled.

    Used by tests that measure the one-level confidence bump on its own. Without
    it, any fixture confidence below ``force_local_confidence_below`` escalates
    straight to the top of the ladder and the gradient is never observable.
    """
    return {**doc, "on_uncertain": {**doc["on_uncertain"], "force_local_confidence_below": None}}


class TestTooUncertainToEgress:
    """The floor below which an uncertain sensitivity answer keeps data local.

    Motivated by measurement, not taste: on the shipped 223-prompt eval set, four
    prompts labelled confidential/regulated reached a cloud tier. All four had
    sensitivity confidence between 0.20 and 0.73 and all four were bumped exactly
    one level, which was not enough. A near-coin-flip judgement about whether data
    is sensitive is not a licence to send it to a third party.
    """

    async def test_very_low_sensitivity_confidence_forces_the_safest_tier(
        self, policy_doc: dict[str, Any], sink: RecordingSink
    ) -> None:
        backend = FakeBackend(sensitivity="public", sensitivity_confidence=0.2)
        router = Router(Policy.from_dict(policy_doc), backend, sink=sink, cache=NullCache())
        decision = await router.route_text("Some request the backend is barely sure about.")
        assert decision.effective_sensitivity == SENSITIVITY_LEVELS[-1]
        assert decision.tier == "local"
        assert any("too uncertain to leave the infrastructure" in note for note in decision.escalated)

    async def test_the_bump_still_applies_between_the_two_thresholds(
        self, policy_doc: dict[str, Any], sink: RecordingSink
    ) -> None:
        """0.5 <= confidence < 0.8 is the gradient band: one level, not air-gapped."""
        backend = FakeBackend(sensitivity="public", sensitivity_confidence=0.6)
        router = Router(Policy.from_dict(policy_doc), backend, sink=sink, cache=NullCache())
        decision = await router.route_text("Some request the backend is mildly unsure about.")
        assert decision.effective_sensitivity == "internal"
        assert decision.tier == "cheap"
        assert not any("too uncertain" in note for note in decision.escalated)

    async def test_disabling_the_floor_restores_the_gradient(
        self, policy_doc: dict[str, Any], sink: RecordingSink
    ) -> None:
        backend = FakeBackend(sensitivity="public", sensitivity_confidence=0.2)
        router = Router(Policy.from_dict(_gradient_only(policy_doc)), backend, sink=sink, cache=NullCache())
        decision = await router.route_text("Some request the backend is barely sure about.")
        assert decision.effective_sensitivity == "internal"
        assert decision.tier == "cheap"

    async def test_a_confident_answer_is_untouched(self, policy_doc: dict[str, Any], sink: RecordingSink) -> None:
        backend = FakeBackend(sensitivity="public", sensitivity_confidence=0.97)
        router = Router(Policy.from_dict(policy_doc), backend, sink=sink, cache=NullCache())
        decision = await router.route_text("Explain how HIPAA works.")
        assert decision.effective_sensitivity == "public"
        assert decision.escalated == ()

    async def test_no_floor_is_claimed_when_nobody_classified(
        self, policy_doc: dict[str, Any], sink: RecordingSink
    ) -> None:
        """A gate-blocked request never reached a backend, so it has no confidence to floor."""
        router = Router(Policy.from_dict(policy_doc), FakeBackend(), sink=sink, cache=NullCache())
        decision = await router.route_text(f"Please charge the card {CARD}.")
        assert decision.backend == "gate"
        assert not any("too uncertain" in note for note in decision.escalated)

    def test_the_shipped_default_keeps_the_floor_below_the_bump(self, default_policy: Policy) -> None:
        """Misconfiguration guard: equal thresholds would air-gap all uncertain traffic."""
        unc = default_policy.uncertainty
        assert unc.force_local_confidence_below is not None
        assert unc.sensitivity_confidence_below is not None
        assert unc.force_local_confidence_below < unc.sensitivity_confidence_below


# --------------------------------------------------------------------------- #
# PII thresholds
# --------------------------------------------------------------------------- #
class TestPiiBands:
    async def test_above_threshold_is_present(self, router_factory: Any, sink: RecordingSink) -> None:
        decision = await router_factory(
            FakeBackend(sensitivity="public", sensitivity_confidence=0.95, pii=0.9)
        ).route_text("A prompt with a person in it.")
        assert decision.tier == "local"
        assert decision.rule_id == "data.sensitive"

    async def test_uncertain_band_counts_as_present(self, router_factory: Any, sink: RecordingSink) -> None:
        """0.4 is below the 0.5 threshold but inside the 0.35 "coin flip" band."""
        decision = await router_factory(
            FakeBackend(sensitivity="public", sensitivity_confidence=0.95, pii=0.4)
        ).route_text("A prompt that might name a person.")
        assert decision.tier == "local"
        assert any("pii treated as present at p=0.40" in note for note in decision.escalated)

    async def test_below_the_uncertain_band_is_absent(self, router_factory: Any, sink: RecordingSink) -> None:
        decision = await router_factory(
            FakeBackend(sensitivity="public", sensitivity_confidence=0.95, pii=0.2)
        ).route_text("A prompt with nobody in it.")
        assert decision.tier == "cheap"
        assert decision.escalated == ()

    @pytest.mark.parametrize(
        ("value", "expected_local"), [(0.34, False), (0.35, True), (0.49, True), (0.5, True), (0.51, True)]
    )
    async def test_band_boundaries(
        self, router_factory: Any, sink: RecordingSink, value: float, expected_local: bool
    ) -> None:
        decision = await router_factory(
            FakeBackend(sensitivity="public", sensitivity_confidence=0.95, pii=value)
        ).route_text("Boundary probe.")
        assert (decision.tier == "local") is expected_local, value

    async def test_gate_pii_floor_overrides_a_confident_no(
        self, policy_doc: dict[str, Any], sink: RecordingSink
    ) -> None:
        """A deterministic hit is not a probability: the backend cannot answer 0.0 out of it.

        ``still_classify`` so a backend actually answers; with the default
        ``skip_backend`` the gate supplies pii=1.0 itself and there is nothing to merge.
        """
        policy_doc["gate"]["on_force_local"] = "still_classify"
        backend = FakeBackend(sensitivity="public", sensitivity_confidence=0.99, pii=0.0)
        router = Router(Policy.from_dict(policy_doc), backend, sink=sink, cache=NullCache())
        decision = await router.route_text(f"Mail {EMAIL} now.")

        assert backend.calls == 1
        assert backend.requests[0] is not None
        assert decision.answers.pii.value == 1.0
        assert any("pii 0.00->1.00 (local gate matched an identifier)" in note for note in decision.escalated)
        assert decision.tier == "local"

    async def test_gate_supplies_pii_when_it_blocks_the_call(self, router_factory: Any, sink: RecordingSink) -> None:
        decision = await router_factory(FakeBackend(pii=0.0)).route_text(f"Mail {EMAIL} now.")
        assert decision.backend == "gate"
        assert decision.answers.pii.value == 1.0

    async def test_uncertain_band_can_be_disabled(self, policy_doc: dict[str, Any], sink: RecordingSink) -> None:
        policy_doc["on_uncertain"]["pii_uncertain_counts_as_present"] = False
        router = Router(
            Policy.from_dict(policy_doc),
            FakeBackend(sensitivity="public", sensitivity_confidence=0.95, pii=0.4),
            sink=sink,
            cache=NullCache(),
        )
        decision = await router.route_text("A prompt that might name a person.")
        assert decision.tier == "cheap"

    async def test_pii_threshold_is_configurable(self, policy_doc: dict[str, Any], sink: RecordingSink) -> None:
        policy_doc["pii_threshold"] = 0.9
        policy_doc["on_uncertain"]["pii_uncertain_counts_as_present"] = False
        router = Router(
            Policy.from_dict(policy_doc),
            FakeBackend(sensitivity="public", sensitivity_confidence=0.95, pii=0.6),
            sink=sink,
            cache=NullCache(),
        )
        assert (await router.route_text("probe")).tier == "cheap"


# --------------------------------------------------------------------------- #
# 7b. Backend outage: fail closed, fail open, and the gate's veto
# --------------------------------------------------------------------------- #
class TestBackendOutage:
    async def test_fail_closed_routes_to_the_safest_tier(self, router_factory: Any, sink: RecordingSink) -> None:
        backend = FakeBackend(degraded=True, degrade_reason="connection refused")
        decision = await router_factory(backend).route_text("An ordinary request.")
        assert decision.tier == "local"
        assert decision.rule_id == "backend.down"
        assert decision.degraded is True
        assert "connection refused" in decision.degrade_reason
        assert "fail_closed" in decision.reason

    async def test_fail_open_routes_to_the_capable_tier(self, policy_doc: dict[str, Any], sink: RecordingSink) -> None:
        policy_doc["on_backend_down"]["mode"] = "fail_open"
        router = Router(Policy.from_dict(policy_doc), FakeBackend(degraded=True), sink=sink, cache=NullCache())
        decision = await router.route_text("An ordinary request.")
        assert decision.tier == "strong"
        assert decision.rule_id == "backend.down"
        assert decision.degraded is True

    @pytest.mark.parametrize(
        "text",
        [
            f"Send the report to {EMAIL} today.",  # confidential floor, force_local
            f"Please charge the card {CARD}.",  # regulated floor, blocks_backend
        ],
    )
    async def test_fail_open_cannot_override_the_gate(
        self, policy_doc: dict[str, Any], sink: RecordingSink, text: str
    ) -> None:
        """A config knob is not allowed to become a data-egress path.

        ``still_classify`` is required to reach this code path at all: with the default
        ``skip_backend`` a hard finding never calls a backend, so there is no outage to
        fail open from. That is the belt; this test is the braces.
        """
        policy_doc["on_backend_down"]["mode"] = "fail_open"
        policy_doc["gate"]["on_force_local"] = "still_classify"
        router = Router(Policy.from_dict(policy_doc), FakeBackend(degraded=True), sink=sink, cache=NullCache())
        decision = await router.route_text(text)

        assert decision.degraded is True
        assert decision.tier == "local", "fail_closed_tier wins over fail_open_tier"
        assert decision.rule_id == "backend.down"
        assert "local gate floor took precedence over fail_open" in decision.reason

    async def test_fail_open_cannot_override_the_semantic_layer(
        self, policy_doc: dict[str, Any], sink: RecordingSink
    ) -> None:
        """The same egress check for layer 2, where there is no detector to help.

        During an outage with ``fail_open``, a request whose only sensitivity
        signal is an enforced layer-2 assertion is still a data-egress risk, and
        ``fail_open`` must not be the path that sends it to the cloud tier. The
        floor that outranks ``fail_open`` is the combined floor of both layers.
        """
        from jev_route.gate_semantic import SemanticGatePolicy, SemanticLayer
        from jev_route.schema import RequestFeatures

        class _Scorer:
            name = "stub"
            model_version = "stub-1"

            def score(self, text: str, *, features: RequestFeatures | None = None) -> float:
                del text, features
                return 0.99

        layer = SemanticLayer(
            SemanticGatePolicy(mode="enforce", level="confidential"),
            scorer=_Scorer(),
            metrics={
                "measured_at": "2026-01-01T00:00:00+00:00",
                "model_version": "stub-1",
                "scorer": "stub",
                "threshold": 0.5,
                "n_examples": 1500,
                "n_positives": 400,
                "n_negatives": 1100,
                "n_layer1_fired": 120,
                "recall": 0.9975,
                "false_positive_rate": 0.008,
                "disagreement_rate": 0.02,
                "semantic_only_rate": 0.018,
                "semantic_miss_rate": 0.008,
                "layer1_recall": 0.3,
                "shadow_n_examples": 4200,
                "shadow_model_version": "stub-1",
            },
        )
        policy_doc["on_backend_down"]["mode"] = "fail_open"
        # As in the layer-1 case: still_classify is what lets the outage happen at
        # all. With the default skip_backend an enforced layer-2 firing never
        # calls a backend, so there is nothing to fail open from.
        policy_doc["gate"]["on_force_local"] = "still_classify"
        router = Router(
            Policy.from_dict(policy_doc), FakeBackend(degraded=True), sink=sink, cache=NullCache(), semantic=layer
        )
        decision = await router.route_text("summarise the notes from our disciplinary hearing last week")
        assert decision.degraded is True
        assert decision.tier == "local", "the enforced layer-2 floor wins over fail_open_tier"
        assert decision.rule_id == "backend.down"
        assert "took precedence over fail_open" in decision.reason

    async def test_fail_closed_is_unaffected_by_the_gate(self, policy_doc: dict[str, Any], sink: RecordingSink) -> None:
        """fail_closed already lands on the safest tier, so the gate has nothing to add."""
        policy_doc["gate"]["on_force_local"] = "still_classify"
        router = Router(Policy.from_dict(policy_doc), FakeBackend(degraded=True), sink=sink, cache=NullCache())
        decision = await router.route_text(f"Send it to {EMAIL}.")
        assert decision.tier == "local"
        assert decision.rule_id == "backend.down"
        assert "took precedence" not in decision.reason

    async def test_gate_block_outranks_an_outage(self, router_factory: Any, sink: RecordingSink) -> None:
        """No backend was called, so this is not an outage: the gate rule answers."""
        decision = await router_factory(FakeBackend(degraded=True)).route_text(f"Send it to {EMAIL}.")
        assert decision.rule_id == "gate.force-local"
        assert decision.degraded is False
        assert decision.tier == "local"

    async def test_degraded_answers_are_maximum_uncertainty(self, router_factory: Any, sink: RecordingSink) -> None:
        decision = await router_factory(FakeBackend(degraded=True)).route_text("An ordinary request.")
        for level in SENSITIVITY_LEVELS:
            assert decision.answers.sensitivity.probability_of(level) == pytest.approx(0.25)
        assert decision.answers.pii.value == 0.5

    async def test_a_backend_that_raises_is_not_silently_swallowed(
        self, router_factory: Any, sink: RecordingSink
    ) -> None:
        """``decide`` may only degrade for *outages*; a contract violation must surface.

        Failing closed on a programming error would hide the bug behind a routing
        decision that looks fine, so the router deliberately lets it propagate.
        """
        backend = FakeBackend(exc=RuntimeError("backend broke its contract"))
        with pytest.raises(RuntimeError, match="broke its contract"):
            await router_factory(backend).route_text("hello")

    async def test_outage_still_writes_a_training_row(self, router_factory: Any, sink: RecordingSink) -> None:
        await router_factory(FakeBackend(degraded=True)).route_text("An ordinary request.")
        record = sink.last
        assert record.decision.degraded is True
        assert record.decision.rule_id == "backend.down"
        assert record.questions_sent, "the questions we tried to ask are part of the row"


# --------------------------------------------------------------------------- #
# 4. The cache: judgements, not decisions
# --------------------------------------------------------------------------- #
class TestCache:
    async def test_second_identical_request_is_served_from_cache(
        self, router_factory: Any, sink: RecordingSink, memory_cache: InMemoryTTLCache
    ) -> None:
        backend = FakeBackend()
        router = router_factory(backend, cache=memory_cache)
        first = await router.route_text("Summarize the quarterly roadmap for the team.")
        second = await router.route_text("Summarize the quarterly roadmap for the team.")

        assert backend.calls == 1
        assert first.cached is False
        assert second.cached is True
        assert first.backend == "fake"
        assert second.backend == "fake:cached"
        assert memory_cache.stats()["hits"] == 1
        assert memory_cache.stats()["misses"] == 1

    async def test_a_policy_change_applies_to_cached_traffic(
        self, policy_doc: dict[str, Any], sink: RecordingSink
    ) -> None:
        """Why the cache stores judgements and not decisions.

        If the final tier were cached, tightening a rule would appear to work while
        warm traffic kept taking the old path -- the exact failure an operator cannot
        see. Here the answers come from the cache and the NEW policy still decides.
        """
        text = "Summarize the quarterly roadmap for the team."
        cache = InMemoryTTLCache(ttl_seconds=60.0)
        backend = FakeBackend(
            sensitivity="public", sensitivity_confidence=0.95, complexity="standard", complexity_confidence=0.95
        )

        first_router = Router(Policy.from_dict(policy_doc), backend, cache=cache, sink=sink)
        before = await first_router.route_text(text)
        assert before.tier == "cheap"

        tightened = copy_of(policy_doc)
        tightened["rules"] = [
            {
                "id": "everything-local",
                "if": "sensitivity == 'public'",
                "then": {"tier": "local"},
                "reason": "tightened mid-flight",
            },
            {"id": "default", "then": {"tier": "cheap"}},
        ]
        second_router = Router(Policy.from_dict(tightened), backend, cache=cache, sink=sink)
        after = await second_router.route_text(text)

        assert backend.calls == 1, "the judgement was reused..."
        assert after.cached is True
        assert after.tier == "local", "...and the new policy still applied to it"
        assert after.rule_id == "everything-local"

    async def test_degraded_results_are_never_cached(
        self, router_factory: Any, sink: RecordingSink, memory_cache: InMemoryTTLCache
    ) -> None:
        """A one-second blip must not pin fail-closed for the whole TTL."""
        backend = FakeBackend(degraded=True)
        router = router_factory(backend, cache=memory_cache)
        await router.route_text("An ordinary request.")
        await router.route_text("An ordinary request.")
        assert backend.calls == 2
        assert memory_cache.stats()["size"] == 0

    async def test_recovery_after_an_outage_is_immediate(
        self, router_factory: Any, sink: RecordingSink, memory_cache: InMemoryTTLCache
    ) -> None:
        """The flip side of not caching degraded answers: the first good call wins."""
        backend = FakeBackend(degraded=True)
        router = router_factory(backend, cache=memory_cache)
        assert (await router.route_text("probe")).tier == "local"
        backend.degraded = False
        backend.sensitivity = "public"
        decision = await router.route_text("probe")
        assert decision.degraded is False
        assert decision.cached is False
        assert decision.tier == "cheap"

    async def test_different_text_is_a_different_key(
        self, router_factory: Any, sink: RecordingSink, memory_cache: InMemoryTTLCache
    ) -> None:
        backend = FakeBackend()
        router = router_factory(backend, cache=memory_cache)
        await router.route_text("Summarize the roadmap.")
        await router.route_text("Summarize the roadmap!")
        assert backend.calls == 2

    async def test_gate_blocked_requests_do_not_touch_the_cache(
        self, router_factory: Any, sink: RecordingSink, memory_cache: InMemoryTTLCache
    ) -> None:
        backend = FakeBackend()
        router = router_factory(backend, cache=memory_cache)
        await router.route_text(f"card {CARD}")
        await router.route_text(f"card {CARD}")
        assert backend.calls == 0
        assert memory_cache.stats()["size"] == 0

    async def test_different_advisory_topics_are_different_keys(
        self, router_factory: Any, sink: RecordingSink, memory_cache: InMemoryTTLCache
    ) -> None:
        """Same excerpt, different gate hints: the backend saw different state."""
        backend = FakeBackend(sensitivity="public", sensitivity_confidence=0.95)
        router = router_factory(backend, cache=memory_cache)
        await router.route_text("Explain the reporting rules.")
        await router.route_text("Explain the HIPAA reporting rules.")
        assert backend.calls == 2

    async def test_null_cache_always_calls_the_backend(self, router_factory: Any, sink: RecordingSink) -> None:
        backend = FakeBackend()
        router = router_factory(backend, cache=NullCache())
        for _ in range(3):
            await router.route_text("Summarize the roadmap.")
        assert backend.calls == 3

    async def test_cache_is_built_from_policy_when_not_injected(self, policy_doc: dict[str, Any]) -> None:
        policy_doc["cache"] = {"enabled": True, "kind": "memory", "ttl_seconds": 5, "max_entries": 3}
        router = Router(Policy.from_dict(policy_doc), FakeBackend(), sink=RecordingSink())
        assert isinstance(router.cache, InMemoryTTLCache)
        assert router.cache.ttl_seconds == 5.0
        assert router.cache.max_entries == 3


def copy_of(doc: dict[str, Any]) -> dict[str, Any]:
    """Deep copy, because ``with_overrides``-style mutation of a fixture leaks between tests."""
    import copy

    return copy.deepcopy(doc)


# --------------------------------------------------------------------------- #
# route_messages vs route_text
# --------------------------------------------------------------------------- #
class TestMessagesAndText:
    TEXT = "Explain the difference between a mutex and a semaphore, with two examples."

    async def test_single_user_message_matches_route_text(self, router_factory: Any, sink: RecordingSink) -> None:
        from_messages = await router_factory(MockBackend(), cache=NullCache()).route_messages(
            [{"role": "user", "content": self.TEXT}]
        )
        from_text = await router_factory(MockBackend(), cache=NullCache()).route_text(self.TEXT)

        assert from_messages.tier == from_text.tier
        assert from_messages.model == from_text.model
        assert from_messages.rule_id == from_text.rule_id
        assert from_messages.effective_sensitivity == from_text.effective_sensitivity
        assert from_messages.effective_complexity == from_text.effective_complexity
        assert from_messages.answers.to_dict() == from_text.answers.to_dict()
        assert from_messages.gate == from_text.gate

    async def test_message_features_are_recorded(self, router_factory: Any, sink: RecordingSink) -> None:
        messages = [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "first question about routing"},
            {"role": "assistant", "content": "an answer"},
            {"role": "tool", "content": "tool output " * 50},
            {"role": "user", "content": self.TEXT},
        ]
        await router_factory(MockBackend()).route_messages(messages)
        features = sink.last.features
        assert features.n_messages == 5
        assert features.n_prior_turns == 1
        assert features.tool_output_present is True

    async def test_gate_runs_over_the_whole_excerpted_conversation(
        self, router_factory: Any, sink: RecordingSink
    ) -> None:
        """A secret in an *earlier* turn must not slip past because the newest one is clean."""
        messages = [{"role": "user", "content": f"my card is {CARD}"}, {"role": "user", "content": "anyway, say hi"}]
        backend = FakeBackend()
        decision = await router_factory(backend).route_messages(messages)
        assert decision.gate.blocks_backend is True
        assert backend.calls == 0
        assert decision.tier == "local"

    async def test_empty_message_list(self, router_factory: Any, sink: RecordingSink) -> None:
        decision = await router_factory(MockBackend()).route_messages([])
        assert decision.gate.fired is False
        assert sink.count == 1

    async def test_none_message_list(self, router_factory: Any, sink: RecordingSink) -> None:
        decision = await router_factory(MockBackend()).route_messages(None)
        assert isinstance(decision.tier, str)

    async def test_multipart_content_is_excerpted_and_redacted(
        self, policy_doc: dict[str, Any], sink: RecordingSink
    ) -> None:
        policy_doc["gate"]["on_force_local"] = "still_classify"
        backend = FakeBackend(sensitivity="public", sensitivity_confidence=0.95)
        router = Router(Policy.from_dict(policy_doc), backend, sink=sink, cache=NullCache())
        await router.route_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"mail {EMAIL}"},
                        {"type": "image_url", "image_url": {"url": "x"}},
                    ],
                }
            ]
        )
        assert backend.calls == 1
        assert EMAIL not in backend.requests[0].redacted_excerpt
        assert "[email_address]" in backend.requests[0].redacted_excerpt
        assert "[image_url]" in backend.requests[0].redacted_excerpt

    async def test_requested_model_and_id_are_carried_through(self, router_factory: Any, sink: RecordingSink) -> None:
        decision = await router_factory(MockBackend()).route_text(
            self.TEXT, requested_model="gpt-anything", request_id="req-42"
        )
        assert decision is not None
        assert sink.last.request_id == "req-42"
        assert sink.last.requested_model == "gpt-anything"

    async def test_request_ids_are_unique_when_not_supplied(self, router_factory: Any, sink: RecordingSink) -> None:
        router = router_factory(MockBackend())
        for _ in range(5):
            await router.route_text(self.TEXT)
        assert len({r.request_id for r in sink.records}) == 5

    def test_sync_wrapper_matches_the_async_path(self, router_factory: Any, sink: RecordingSink) -> None:
        router = router_factory(MockBackend())
        decision = router.route_text_sync(self.TEXT)
        assert decision.tier in {"cheap", "strong", "local"}
        assert sink.count == 1


# --------------------------------------------------------------------------- #
# 8. Logging: the training dataset
# --------------------------------------------------------------------------- #
class TestLogging:
    async def test_exactly_one_record_per_decision(self, router_factory: Any, sink: RecordingSink) -> None:
        router = router_factory(MockBackend())
        for text in ("hello", f"card {CARD}", "prove the spectral theorem", f"mail {EMAIL}"):
            await router.route_text(text)
        assert sink.count == 4

    async def test_record_round_trips_through_from_json(self, router_factory: Any, sink: RecordingSink) -> None:
        router = router_factory(MockBackend())
        await router.route_text(f"Please charge {CARD} and mail {EMAIL}.")
        line = sink.last.to_json()
        restored = DecisionRecord.from_json(line)

        assert restored.request_id == sink.last.request_id
        assert restored.decision.tier == sink.last.decision.tier
        assert restored.decision.rule_id == sink.last.decision.rule_id
        assert restored.decision.gate.to_dict() == sink.last.decision.gate.to_dict()
        assert restored.features.to_dict() == sink.last.features.to_dict()
        assert json.loads(line) == json.loads(restored.to_json())

    async def test_full_distributions_are_logged_not_just_the_argmax(
        self, router_factory: Any, sink: RecordingSink
    ) -> None:
        """The single most important property of the log: soft targets survive."""
        await router_factory(MockBackend()).route_text("Summarize the incident report from last night.")
        payload = json.loads(sink.last.to_json())["decision"]["answers"]

        assert sorted(payload["complexity"]["probabilities"]) == sorted(COMPLEXITY_LEVELS)
        assert sorted(payload["sensitivity"]["probabilities"]) == sorted(SENSITIVITY_LEVELS)
        assert sorted(payload["domain"]["probabilities"]) == sorted(DOMAINS)
        assert "noul" in payload["pii"]
        for question in ("complexity", "sensitivity", "domain"):
            assert sum(payload[question]["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
            assert 0.0 < max(payload[question]["probabilities"].values()) < 1.0, (
                "non-degenerate, i.e. real soft targets"
            )

    async def test_gate_blocked_record_still_carries_soft_targets(
        self, router_factory: Any, sink: RecordingSink
    ) -> None:
        await router_factory(MockBackend()).route_text(f"card {CARD}")
        payload = json.loads(sink.last.to_json())["decision"]["answers"]
        assert sorted(payload["complexity"]["probabilities"]) == sorted(COMPLEXITY_LEVELS)
        assert payload["sensitivity"]["probabilities"]["regulated"] == 1.0

    async def test_record_has_no_prompt_text_in_hash_mode(self, router_factory: Any, sink: RecordingSink) -> None:
        text = f"Summarize the note from {EMAIL} about the outage."
        await router_factory(MockBackend(), excerpt_mode="hash").route_text(text)
        line = sink.last.to_json()
        assert EMAIL not in line
        assert "outage" not in line

    async def test_metadata_is_filtered_to_scalars(self, router_factory: Any, sink: RecordingSink) -> None:
        await router_factory(MockBackend()).route_text(
            "hello there",
            metadata={
                "tenant": "acme",
                "retries": 2,
                "flag": True,
                "nothing": None,
                "nested": {"a": 1},
                "items": [1, 2],
            },
        )
        assert sink.last.metadata == {"tenant": "acme", "retries": 2, "flag": True, "nothing": None}

    async def test_metadata_cannot_smuggle_the_prompt(self, router_factory: Any, sink: RecordingSink) -> None:
        """Callers routinely stuff request bodies into metadata; the log must not inherit them."""
        await router_factory(MockBackend()).route_text(
            "a clean prompt", metadata={"prompt": "SECRET RAW TEXT", "tenant": "acme"}
        )
        assert "SECRET RAW TEXT" not in sink.last.to_json()
        assert sink.last.metadata == {"tenant": "acme"}

    @pytest.mark.parametrize("unsafe_key", ["messages", "prompt", "content", "input", "raw", "body"])
    async def test_unsafe_metadata_keys_never_reach_the_log(
        self, router_factory: Any, sink: RecordingSink, unsafe_key: str
    ) -> None:
        """Same promise as the backend path, asserted for the key set the code already knows."""
        await router_factory(MockBackend()).route_text("a clean prompt", metadata={unsafe_key: "SECRET RAW TEXT"})
        assert "SECRET RAW TEXT" not in sink.last.to_json()

    async def test_metadata_scalars_are_preserved(self, router_factory: Any, sink: RecordingSink) -> None:
        await router_factory(MockBackend()).route_text("a clean prompt", metadata={"tenant": "acme", "retries": 2})
        assert sink.last.metadata == {"tenant": "acme", "retries": 2}

    async def test_hash_salt_changes_the_stored_hash(self, router_factory: Any, sink: RecordingSink) -> None:
        text = "Summarize the roadmap for the team."
        await router_factory(MockBackend(), hash_salt="deployment-a").route_text(text)
        salted = sink.last.excerpt_hash
        await router_factory(MockBackend(), hash_salt="deployment-b").route_text(text)
        assert sink.last.excerpt_hash != salted
        assert len(sink.last.excerpt_hash) == 16

    async def test_same_salt_same_hash(self, router_factory: Any, sink: RecordingSink) -> None:
        text = "Summarize the roadmap for the team."
        await router_factory(MockBackend(), hash_salt="s").route_text(text)
        first = sink.last.excerpt_hash
        await router_factory(MockBackend(), hash_salt="s").route_text(text)
        assert sink.last.excerpt_hash == first

    async def test_latency_is_measured(self, router_factory: Any, sink: RecordingSink) -> None:
        decision = await router_factory(FakeBackend(latency_ms=5.0)).route_text("hello there")
        assert decision.latency_ms >= 0.0
        assert sink.last.backend_latency_ms == pytest.approx(5.0)
        assert sink.last.total_latency_ms >= sink.last.backend_latency_ms

    async def test_records_reach_a_real_jsonl_file(self, policy_doc: dict[str, Any], tmp_path: Path) -> None:
        """The sink a production deployment uses, exercised end to end."""
        log_path = tmp_path / "logs" / "decisions.jsonl"
        policy_doc["logging"]["path"] = str(log_path)
        router = Router(Policy.from_dict(policy_doc), MockBackend(), cache=NullCache())
        await router.route_text("Summarize the roadmap for the team.")
        await router.route_text(f"card {CARD}")
        await router.aclose()

        written = list(iter_records(log_path))
        assert len(written) == 2
        assert written[0].decision.tier == "cheap"
        assert written[1].decision.tier == "local"
        assert written[1].decision.backend == "gate"
        assert written[1].excerpt is None
        assert all(r.schema_version for r in written)

    async def test_aclose_closes_the_sink(self, policy_doc: dict[str, Any]) -> None:
        sink = RecordingSink()
        router = Router(Policy.from_dict(policy_doc), MockBackend(), sink=sink, cache=NullCache())
        await router.aclose()
        assert sink.closed == 1

    async def test_aclose_closes_the_backend(self, policy_doc: dict[str, Any]) -> None:
        closed: list[str] = []

        class Tracking(FakeBackend):
            async def aclose(self) -> None:
                closed.append("backend")

        router = Router(Policy.from_dict(policy_doc), Tracking(), sink=RecordingSink(), cache=NullCache())
        await router.aclose()
        assert closed == ["backend"]

    async def test_stats_reports_cache_and_log(self, router_factory: Any, sink: RecordingSink) -> None:
        stats = router_factory(MockBackend(), cache=NullCache()).stats()
        assert stats["cache"]["backend"] == "null"
        assert stats["log"]["backend"] == "recording"


# --------------------------------------------------------------------------- #
# Shadow mode: the cutover path
# --------------------------------------------------------------------------- #
class TestShadowMode:
    async def test_shadow_disagreement_is_recorded_without_changing_the_decision(
        self, policy_doc: dict[str, Any], sink: RecordingSink
    ) -> None:
        policy_doc["shadow"] = {"enabled": True, "sample_rate": 1.0}
        primary = FakeBackend(
            sensitivity="public",
            sensitivity_confidence=0.95,
            complexity="trivial",
            complexity_confidence=0.95,
            domain="chat",
        )
        shadow = FakeBackend(name="shadow", sensitivity="regulated", complexity="frontier", domain="code", pii=0.9)
        router = Router(Policy.from_dict(policy_doc), primary, sink=sink, cache=NullCache(), shadow_backend=shadow)

        decision = await router.route_text("Summarize the roadmap for the team.")
        assert decision.tier == "cheap", "the shadow never serves"
        assert decision.backend == "fake"
        report = sink.last.shadow
        assert report is not None
        assert report["backend"] == "shadow"
        assert report["agrees"] is False
        assert report["disagreements"]["complexity"] == {"primary": "trivial", "shadow": "frontier"}
        assert "pii" in report["disagreements"]

    async def test_shadow_disabled_by_default(self, router_factory: Any, sink: RecordingSink) -> None:
        shadow = FakeBackend(name="shadow")
        await router_factory(FakeBackend(), shadow_backend=shadow).route_text("hello there")
        assert shadow.calls == 0
        assert sink.last.shadow is None

    async def test_a_failing_shadow_is_telemetry_not_an_error(
        self, policy_doc: dict[str, Any], sink: RecordingSink
    ) -> None:
        policy_doc["shadow"] = {"enabled": True, "sample_rate": 1.0}
        shadow = FakeBackend(name="shadow", exc=RuntimeError("shadow died"))
        router = Router(
            Policy.from_dict(policy_doc),
            FakeBackend(sensitivity="public", sensitivity_confidence=0.95),
            sink=sink,
            cache=NullCache(),
            shadow_backend=shadow,
        )
        decision = await router.route_text("Summarize the roadmap.")
        assert decision.tier == "cheap"
        assert "error" in (sink.last.shadow or {})

    async def test_gate_blocked_requests_are_not_shadowed(
        self, policy_doc: dict[str, Any], sink: RecordingSink
    ) -> None:
        """If the primary is not allowed to see it, neither is the shadow."""
        policy_doc["shadow"] = {"enabled": True, "sample_rate": 1.0}
        shadow = FakeBackend(name="shadow")
        router = Router(
            Policy.from_dict(policy_doc), FakeBackend(), sink=sink, cache=NullCache(), shadow_backend=shadow
        )
        await router.route_text(f"card {CARD}")
        assert shadow.calls == 0

    async def test_zero_sample_rate_disables_shadowing(self, policy_doc: dict[str, Any], sink: RecordingSink) -> None:
        policy_doc["shadow"] = {"enabled": True, "sample_rate": 0.0}
        shadow = FakeBackend(name="shadow")
        router = Router(
            Policy.from_dict(policy_doc), FakeBackend(), sink=sink, cache=NullCache(), shadow_backend=shadow
        )
        await router.route_text("Summarize the roadmap.")
        assert shadow.calls == 0


# --------------------------------------------------------------------------- #
# Construction and configuration
# --------------------------------------------------------------------------- #
class TestConstruction:
    def test_the_gate_is_always_a_real_gate(self, router_policy: Policy) -> None:
        """There is no configuration in which unscanned text is routed."""
        router = Router(router_policy, MockBackend(), gate=None, sink=RecordingSink())
        assert isinstance(router.gate, HardGate)
        assert router.gate.detectors
        assert {d.name for d in router.gate.detectors} == {d.name for d in DEFAULT_DETECTORS}

    def test_gate_config_is_honoured(self, policy_doc: dict[str, Any]) -> None:
        policy_doc["gate"] = {
            "on_force_local": "still_classify",
            "disabled_detectors": ["phone_number"],
            "placeholder_domains_as_pii": True,
        }
        router = Router(Policy.from_dict(policy_doc), MockBackend(), sink=RecordingSink())
        assert "phone_number" not in {d.name for d in router.gate.detectors}
        assert router._on_force_local == "still_classify"

    def test_placeholder_domains_as_pii_reaches_the_gate(self, policy_doc: dict[str, Any]) -> None:
        policy_doc["gate"]["placeholder_domains_as_pii"] = True
        router = Router(Policy.from_dict(policy_doc), MockBackend(), sink=RecordingSink())
        assert router.gate.scan("mail user@example.com").force_local is True

    def test_bad_excerpt_mode_is_rejected(self, router_policy: Policy) -> None:
        with pytest.raises(ValueError, match="excerpt_mode must be hash"):
            Router(router_policy, MockBackend(), excerpt_mode="everything")

    def test_bad_on_force_local_is_rejected(self, policy_doc: dict[str, Any]) -> None:
        policy_doc["gate"]["on_force_local"] = "maybe"
        with pytest.raises(ValueError, match="on_force_local must be skip_backend"):
            Router(Policy.from_dict(policy_doc), MockBackend())

    def test_from_policy_file_builds_the_named_backend(self, tmp_path: Path) -> None:
        """The one-call entry point the CLI and integrations use."""
        doc = make_policy_doc(log_path=tmp_path / "decisions.jsonl", backend={"name": "mock"})
        path = tmp_path / "policy.yaml"
        path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")

        router = Router.from_policy_file(path)
        assert isinstance(router.backend, MockBackend)
        assert isinstance(router.gate, HardGate)
        assert isinstance(router.sink, JsonlSink)
        assert router.excerpt_mode == "hash"

    async def test_from_policy_file_routes_with_no_api_key(self, tmp_path: Path) -> None:
        doc = make_policy_doc(log_path=tmp_path / "decisions.jsonl")
        path = tmp_path / "policy.yaml"
        path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        router = Router.from_policy_file(path)
        decision = await router.route_text(f"Please charge {CARD}.")
        assert decision.tier == "local"
        await router.aclose()

    def test_injected_backend_wins_over_the_policy(self, tmp_path: Path) -> None:
        doc = make_policy_doc(log_path=tmp_path / "d.jsonl", backend={"name": "jev"})
        path = tmp_path / "policy.yaml"
        path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        router = Router.from_policy_file(path, backend=MockBackend())
        assert isinstance(router.backend, MockBackend)

    def test_excerpt_bounds_are_configurable(self, router_policy: Policy) -> None:
        router = Router(
            router_policy, MockBackend(), sink=RecordingSink(), max_excerpt_chars=100, include_prior_turns=0
        )
        assert router.max_excerpt_chars == 100
        assert router.include_prior_turns == 0

    async def test_max_excerpt_chars_bounds_what_leaves_the_process(
        self, router_policy: Policy, sink: RecordingSink
    ) -> None:
        backend = FakeBackend(sensitivity="public", sensitivity_confidence=0.95)
        router = Router(router_policy, backend, sink=sink, cache=NullCache(), max_excerpt_chars=40)
        await router.route_text("x" * 5000)
        assert len(backend.requests[0].redacted_excerpt) <= 40


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #
class TestConcurrency:
    async def test_parallel_requests_each_log_exactly_once(self, router_factory: Any, sink: RecordingSink) -> None:
        router = router_factory(MockBackend(), cache=NullCache())
        await asyncio.gather(*(router.route_text(f"prompt number {i} about model routing") for i in range(20)))
        assert sink.count == 20
        assert len({r.request_id for r in sink.records}) == 20

    async def test_parallel_requests_with_a_shared_cache_do_not_double_serve(
        self, router_factory: Any, sink: RecordingSink, memory_cache: InMemoryTTLCache
    ) -> None:
        backend = FakeBackend()
        router = router_factory(backend, cache=memory_cache)
        await asyncio.gather(*(router.route_text("the same prompt for every caller") for _ in range(10)))
        assert sink.count == 10
        assert backend.calls <= 10
        assert memory_cache.stats()["size"] == 1

    async def test_a_mix_of_blocked_and_classified_requests(self, router_factory: Any, sink: RecordingSink) -> None:
        router = router_factory(MockBackend(), cache=NullCache())
        texts = [f"card {CARD}", "hello there", f"mail {EMAIL}", "prove the spectral theorem", TOPIC_HIPAA]
        decisions = await asyncio.gather(*(router.route_text(t) for t in texts))
        assert decisions[0].backend == "gate"
        assert decisions[1].backend == "mock"
        assert sink.count == len(texts)
