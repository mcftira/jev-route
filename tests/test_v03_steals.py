"""v0.3 router behaviors: the anti-downgrade uncertain-fallback hold and the
quota tandem.

Two "almost a mistake" shapes, and what v0.3 does about each:

* **Confident downgrade** -- the dominant error class in the live decision
  log: the router is confident about a weak tier, not unsure about a strong
  one. So when confidence falls below ``routing.min_confidence`` the default
  is ``hold_middle``: keep the policy's (already bumped) choice, mark the
  decision ``uncertain`` for the human review queue, and do not divert to a
  fallback that is cheaper and weaker by default.

* **Quota exhaustion** -- the decision backend's provider ran out of capacity
  (HTTP 429 / "quota"), it is not down. The affected tier is marked
  quota-exhausted in the prefilter config for the rest of the process, and
  the in-flight request is served from the tier's ``tandem`` deployment
  (configured per tier in the policy) instead of the tier's first model.
"""

from __future__ import annotations

from typing import Any

import pytest

from jev_route.cache import NullCache
from jev_route.policy import Policy, PolicyError
from jev_route.router import Router

from .conftest import FakeBackend, make_policy_doc


def make_router(policy_doc: dict, backend: Any, sink: Any) -> Router:
    """Router with one programmable backend, no cache, and the test sink."""
    return Router(Policy.from_dict(policy_doc), backend, sink=sink, cache=NullCache())


def tandem_policy_doc(policy_doc: dict) -> dict:
    """The default test doc with a mapping-form local tier that has a tandem.

    Two models in the local tier on purpose: the flip must pick the tandem,
    not the first model, and round-robin must not be involved at all.
    """
    doc = dict(policy_doc)
    doc["tiers"] = {
        "local": {"models": ["local-model", "local-second"], "tandem": "local-tandem"},
        "cheap": ["cheap-model"],
        "strong": ["strong-model"],
    }
    return doc


# --------------------------------------------------------------------------- #
# 1. Anti-downgrade hold: routing.uncertain_fallback
# --------------------------------------------------------------------------- #
class TestAntiDowngradeHold:
    async def test_default_is_hold_middle_keeps_tier_and_marks_uncertain(self, policy_doc: dict, sink: Any) -> None:
        """No uncertain_fallback configured: keep the policy's choice, mark only."""
        doc = dict(policy_doc)
        doc["routing"] = {"min_confidence": 0.99}
        router = make_router(doc, FakeBackend(complexity="standard", complexity_confidence=0.75), sink)
        decision = await router.route_text("hold the line on this one", request_id="hold-1")

        assert decision.uncertain is True
        assert decision.tier == "cheap", "hold_middle keeps the tier the policy chose"
        assert decision.model == "cheap-model", "hold_middle keeps the model too"
        assert any("hold_middle" in e and "marked uncertain" in e for e in decision.escalated)

    async def test_explicit_hold_middle_behaves_like_the_default(self, policy_doc: dict, sink: Any) -> None:
        doc = dict(policy_doc)
        doc["routing"] = {"min_confidence": 0.99, "uncertain_fallback": "hold_middle"}
        router = make_router(doc, FakeBackend(complexity_confidence=0.75), sink)
        decision = await router.route_text("explicit hold", request_id="hold-2")
        assert decision.uncertain is True
        assert decision.tier == "cheap"
        assert decision.model == "cheap-model"

    async def test_cheapest_local_diverts_to_the_local_tiers_first_model(self, policy_doc: dict, sink: Any) -> None:
        doc = tandem_policy_doc(policy_doc)
        doc["routing"] = {"min_confidence": 0.99, "uncertain_fallback": "cheapest_local"}
        router = make_router(doc, FakeBackend(complexity_confidence=0.75), sink)
        decision = await router.route_text("cheapest local please", request_id="cheap-1")

        assert decision.uncertain is True
        assert decision.tier == "local"
        assert decision.model == "local-model", "the local tier's FIRST model -- not the tandem"

    async def test_frontier_diverts_to_the_strongest_tier(self, policy_doc: dict, sink: Any) -> None:
        doc = tandem_policy_doc(policy_doc)
        doc["tier_order"] = ["local", "cheap", "strong"]
        doc["routing"] = {"min_confidence": 0.99, "uncertain_fallback": "frontier"}
        router = make_router(doc, FakeBackend(complexity_confidence=0.75), sink)
        decision = await router.route_text("frontier please", request_id="front-1")

        assert decision.uncertain is True
        assert decision.tier == "strong", "the strongest tier is the last one in tier_order"
        assert decision.model == "strong-model"

    async def test_literal_model_name_keeps_the_v02_behavior(self, policy_doc: dict, sink: Any) -> None:
        doc = dict(policy_doc)
        doc["routing"] = {"min_confidence": 0.99, "uncertain_fallback": "review-queue-model"}
        router = make_router(doc, FakeBackend(complexity_confidence=0.75), sink)
        decision = await router.route_text("literal fallback", request_id="lit-1")

        assert decision.uncertain is True
        assert decision.tier == "uncertain"
        assert decision.model == "review-queue-model"

    async def test_confident_decision_is_neither_marked_nor_moved(self, policy_doc: dict, sink: Any) -> None:
        """Above the floor the enum is inert, whatever it says."""
        doc = dict(policy_doc)
        doc["routing"] = {"min_confidence": 0.99, "uncertain_fallback": "frontier"}
        router = make_router(doc, FakeBackend(complexity_confidence=0.995), sink)
        decision = await router.route_text("confident and calm", request_id="ok-1")
        assert decision.uncertain is False
        assert decision.tier == "cheap"
        assert decision.model == "cheap-model"


# --------------------------------------------------------------------------- #
# 2. Quota tandem: 429/quota degraded backend
# --------------------------------------------------------------------------- #
class TestQuotaTandem:
    async def test_429_routes_to_tandem_and_marks_quota_exhausted(self, policy_doc: dict, sink: Any) -> None:
        """The work-order case: a 429-flavored degraded backend.

        First request: the fail-closed tier, served by the tier's tandem
        instead of its first model, with an escalation entry logging the flip.
        Subsequent request: the tier is gone from the prefilter's candidates,
        which makes strong the single survivor (cheap is denied) and skips the
        backend entirely -- the mark persisted for the rest of the process.
        """
        doc = tandem_policy_doc(policy_doc)
        doc["prefilter"] = {"deny_tiers": ["cheap"]}
        backend = FakeBackend(
            degraded=True,
            degrade_reason="http 429: rate limited, quota exceeded for this key",
        )
        router = make_router(doc, backend, sink)
        first = await router.route_text("quota probe one", request_id="q1")

        assert first.degraded is True
        assert first.tier == "local", "fail_closed tier, unchanged"
        assert first.model == "local-tandem", "the flip: tandem, not the tier's first model"
        assert any("quota" in e and "local-tandem" in e for e in first.escalated)
        assert router.policy.raw["prefilter"]["quota_exhausted"] == ["local"]

        second = await router.route_text("quota probe two", request_id="q2")
        assert second.backend == "prefilter", "local exhausted, cheap denied: strong is the only candidate"
        assert second.tier == "strong"
        assert backend.calls == 1, "the second request never spends a backend call"

    async def test_quota_word_alone_triggers_the_same_path(self, policy_doc: dict, sink: Any) -> None:
        doc = tandem_policy_doc(policy_doc)
        backend = FakeBackend(degraded=True, degrade_reason="provider said: Quota exceeded for monthly budget")
        router = make_router(doc, backend, sink)
        first = await router.route_text("quota word probe", request_id="qw1")
        assert first.model == "local-tandem"
        assert router.policy.raw["prefilter"]["quota_exhausted"] == ["local"]

    async def test_repeat_429_marks_once_and_keeps_serving_the_tandem(self, policy_doc: dict, sink: Any) -> None:
        doc = tandem_policy_doc(policy_doc)
        backend = FakeBackend(degraded=True, degrade_reason="http 429: quota")
        router = make_router(doc, backend, sink)
        await router.route_text("first 429", request_id="r1")
        await router.route_text("second 429", request_id="r2")

        assert router.policy.raw["prefilter"]["quota_exhausted"] == ["local"], "idempotent mark"
        assert sink.records[0].decision.model == "local-tandem"
        assert sink.records[1].decision.model == "local-tandem"

    async def test_timeout_degradation_does_not_mark_or_flip(self, policy_doc: dict, sink: Any) -> None:
        """A provider that is DOWN, not drained, takes the plain degraded path."""
        doc = tandem_policy_doc(policy_doc)
        backend = FakeBackend(degraded=True, degrade_reason="timeout after 30.0s")
        router = make_router(doc, backend, sink)
        first = await router.route_text("timeout probe", request_id="t1")

        assert first.tier == "local"
        assert first.model == "local-model", "no flip: a timeout is not a quota error"
        assert "quota_exhausted" not in (router.policy.raw.get("prefilter") or {})

    async def test_429_without_a_tandem_marks_but_keeps_the_first_model(self, policy_doc: dict, sink: Any) -> None:
        doc = dict(policy_doc)  # bare list tiers: no tandem anywhere
        backend = FakeBackend(degraded=True, degrade_reason="quota exhausted: monthly limit reached")
        router = make_router(doc, backend, sink)
        first = await router.route_text("bare tier quota", request_id="b1")

        assert first.tier == "local"
        assert first.model == "local-model"
        assert router.policy.raw["prefilter"]["quota_exhausted"] == ["local"]
        assert any("no tandem" in e for e in first.escalated)


# --------------------------------------------------------------------------- #
# 3. Tier config: the mapping form {models: [...], tandem: <model>}
# --------------------------------------------------------------------------- #
class TestTierMappingForm:
    def test_bare_forms_parse_with_no_tandem(self, policy_doc: dict) -> None:
        policy = Policy.from_dict(policy_doc)
        assert policy.tandem == {}
        assert policy.pick_quota_model("local") == "local-model"

    def test_mapping_form_parses_models_and_tandem(self) -> None:
        doc = make_policy_doc(
            tiers={
                "local": {"models": ["l1"], "tandem": "l2"},
                "cheap": ["c1"],
                "strong": {"models": ["s1", "s2"]},
            }
        )
        policy = Policy.from_dict(doc)
        assert policy.tiers["local"] == ("l1",)
        assert policy.tiers["strong"] == ("s1", "s2")
        assert policy.tandem == {"local": "l2"}
        assert policy.pick_quota_model("local") == "l2"
        assert policy.pick_quota_model("strong") == "s1"

    def test_mapping_form_without_models_key_fails_at_load(self) -> None:
        doc = make_policy_doc(tiers={"local": {"tandem": "l2"}, "cheap": ["c1"], "strong": ["s1"]})
        with pytest.raises(PolicyError, match="models"):
            Policy.from_dict(doc)

    def test_mapping_form_with_blank_tandem_fails_at_load(self) -> None:
        doc = make_policy_doc(tiers={"local": {"models": ["l1"], "tandem": "   "}, "cheap": ["c1"], "strong": ["s1"]})
        with pytest.raises(PolicyError, match="tandem"):
            Policy.from_dict(doc)
