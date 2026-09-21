"""The deterministic tier pre-filter: pure functions, no model calls."""

from __future__ import annotations

from dataclasses import dataclass

from jev_route.prefilter import estimate_tokens, filter_tiers

TIERS = ("local", "cheap", "strong")


@dataclass
class FakeVerdict:
    blocks_backend: bool = False
    force_local: bool = False


@dataclass
class FakeAssessment:
    force_local: bool = False


def test_everything_survives_a_plain_request():
    res = filter_tiers(tiers=TIERS, excerpt="hello world", features={}, verdict=FakeVerdict())
    assert res.surviving == TIERS
    assert res.single_candidate is False
    assert res.removed == ()


def test_gate_blocks_cloud_tiers():
    res = filter_tiers(tiers=TIERS, excerpt="my ssn is 000-12-3456", features={},
                       verdict=FakeVerdict(blocks_backend=True))
    assert res.surviving == ("local",)
    assert res.single_candidate is True
    reasons = [r for _, r in res.removed]
    assert all("sensitivity" in r for r in reasons)


def test_force_local_removes_cloud():
    res = filter_tiers(tiers=TIERS, excerpt="x", features={}, verdict=FakeVerdict(force_local=True))
    assert res.surviving == ("local",)


def test_semantic_layer_also_removes_cloud():
    res = filter_tiers(tiers=TIERS, excerpt="x", features={}, verdict=FakeVerdict(),
                       assessment=FakeAssessment(force_local=True))
    assert res.surviving == ("local",)


def test_context_window_rules_out_small_tiers():
    big = "x" * 4 * 10_000  # ~10k tokens
    res = filter_tiers(tiers=TIERS, excerpt=big, features={}, verdict=FakeVerdict())
    assert "local" not in res.surviving
    assert "cheap" in res.surviving


def test_capability_mismatch():
    res = filter_tiers(tiers=TIERS, excerpt="x", features={"required_capabilities": {"tools"}},
                       verdict=FakeVerdict())
    assert res.surviving == ("strong",)
    assert res.single_candidate is True


def test_quota_exhaustion():
    res = filter_tiers(tiers=TIERS, excerpt="x", features={}, verdict=FakeVerdict(),
                       config={"quota_exhausted": ["cheap"]})
    assert res.surviving == ("local", "strong")


def test_deny_and_allow_lists():
    res = filter_tiers(tiers=TIERS, excerpt="x", features={}, verdict=FakeVerdict(),
                       config={"deny_tiers": ["strong"]})
    assert res.surviving == ("local", "cheap")
    res2 = filter_tiers(tiers=TIERS, excerpt="x", features={}, verdict=FakeVerdict(),
                        config={"allow_tiers": ["strong"]})
    assert res2.surviving == ("strong",)


def test_zero_survivors_fail_closed_to_local():
    res = filter_tiers(tiers=TIERS, excerpt="x", features={}, verdict=FakeVerdict(),
                       config={"deny_tiers": ["local", "cheap", "strong"]})
    assert res.surviving == ("local",)
    assert res.single_candidate is True
    assert res.fallback_tier == "local"


def test_estimate_tokens():
    assert estimate_tokens("") == 1
    assert estimate_tokens("x" * 400) == 101
