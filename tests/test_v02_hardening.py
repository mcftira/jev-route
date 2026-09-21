"""v0.2 router behaviors: opt-in bypass, prefilter free routing, uncertainty marking."""

from __future__ import annotations

from typing import Any

from jev_route.backends.mock import MockBackend
from jev_route.cache import NullCache
from jev_route.policy import Policy
from jev_route.router import Router


def make_router(policy_doc: dict, sink: Any, **extra: Any) -> Router:
    doc = dict(policy_doc)
    doc.update(extra.pop("extra_policy", {}))
    return Router(Policy.from_dict(doc), MockBackend(), sink=sink, cache=NullCache(), **extra)


async def test_pinned_model_still_logs_by_default(policy_doc: dict, sink: Any) -> None:
    """The existing contract: requested_model is log metadata unless the
    policy opts into bypass_on_pinned."""
    decision = await make_router(policy_doc, sink).route_text("hello", requested_model="gpt-x", request_id="r1")
    assert sink.last.request_id == "r1"
    assert decision.tier != "bypass"


async def test_bypass_when_policy_opts_in(policy_doc: dict, sink: Any) -> None:
    doc = dict(policy_doc)
    doc["routing"] = {"bypass_on_pinned": True}
    decision = await make_router(doc, sink).route_text("hello", requested_model="gpt-pinned")
    assert decision.tier == "bypass"
    assert decision.model == "gpt-pinned"
    assert sink.records == []  # bypass writes no training record


async def test_routing_mode_off_bypasses_everything(policy_doc: dict, sink: Any) -> None:
    doc = dict(policy_doc)
    doc["routing"] = {"mode": "off"}
    decision = await make_router(doc, sink).route_text("hi")
    assert decision.tier == "bypass"
    assert sink.records == []


async def test_single_candidate_skips_backend_call(policy_doc: dict, sink: Any) -> None:
    """A policy that denies all but one tier must never spend a model call,
    and the decision must still be logged (backend='prefilter')."""
    doc = dict(policy_doc)
    doc["prefilter"] = {"deny_tiers": ["cheap", "strong"]}
    decision = await make_router(doc, sink).route_text("refactor this small function")
    assert decision.backend == "prefilter"
    assert decision.tier == "local"
    assert sink.last is not None


async def test_zero_candidates_fails_closed_to_local(policy_doc: dict, sink: Any) -> None:
    doc = dict(policy_doc)
    doc["prefilter"] = {"deny_tiers": ["local", "cheap", "strong"]}
    decision = await make_router(doc, sink).route_text("hello world")
    assert decision.backend == "prefilter"
    assert decision.tier == "local"
    assert "failed closed" in decision.reason


async def test_uncertainty_marks_and_diverts(policy_doc: dict, sink: Any) -> None:
    """MockBackend's confidence is fixed; a floor above it must mark
    `uncertain: true` and divert to the uncertain fallback."""
    doc = dict(policy_doc)
    doc["routing"] = {"min_confidence": 0.99, "uncertain_fallback": "local-mini"}
    decision = await make_router(doc, sink).route_text("uncertainty probe", request_id="u1")
    assert decision.uncertain is True
    assert decision.model == "local-mini"
    assert sink.last.request_id == "u1"


async def test_no_marking_below_the_floor(policy_doc: dict, sink: Any) -> None:
    doc = dict(policy_doc)
    doc["routing"] = {"min_confidence": 0.01}
    decision = await make_router(doc, sink).route_text("hi")
    assert decision.uncertain is False
