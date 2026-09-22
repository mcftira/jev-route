"""v0.3: the intent cache (near-duplicate shortcut in front of the router) and the
thin LiteLLM classifier adapter (text in, one four-key dict out).

Two rules, both inherited from the rest of the suite:

**Offline and deterministic.** Everything runs on the MockBackend with the
decision log pointed at ``tmp_path``; no network, no API key, no wall-clock
sleeps (the cache\'s clock is injected).

**The gate contract is the security property.** A request the local sensitivity
gate fired on is never cached and never served from the cache. These tests pin
that from both directions -- ``store`` must not keep it, and ``lookup`` must
not return it -- because a decision cache that replays answers for
credential-bearing text is a retention path for exactly the content the gate
refuses to hold.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from jev_route import JsonlSink, MockBackend, Policy, Router
from jev_route.backends.base import DecisionRequest
from jev_route.integrations.litellm_classifier import JevRouteClassifierAdapter
from jev_route.intent_cache import (
    CachedAnswer,
    IntentCache,
    build_intent_cache,
    jaccard,
    normalize_text,
    token_set,
)
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
#: The same prompts the integration suite pins against the MockBackend, so a
#: change to the mock cannot silently retune these fixtures in both places.
BENIGN = "Say hello and thank me for the recipe."
HARD = (
    "Design the architecture for a distributed consensus layer, derive the proof that it "
    "is safe under partition, then optimize the p99 latency and explain the trade-offs of "
    "sharding versus replication. Debug the intermittent race condition in the migration."
)
SENSITIVE = (
    "Patient record for John Smith, SSN 123-45-6789, card 4111 1111 1111 1111, "
    "diagnosis and prescription attached. Summarize the clinical notes."
)

EXPECTED_KEYS = {"tier", "model", "gate_fired", "uncertain"}


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scrub JEV_ROUTE_* env vars so construction-time reads see a clean shell."""
    for name in [k for k in os.environ if k.startswith("JEV_ROUTE_")]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    yield


def make_decision(
    tier: str = "cheap",
    model: str = "qwen3.8-flash",
    *,
    gate_fired: bool = False,
    uncertain: bool = False,
    sensitivity: str = "public",
) -> RoutingDecision:
    """A RoutingDecision with the right shape and nothing more than the test asserts."""
    answers = DecisionAnswers(
        complexity=ChoiceAnswer(
            choice="standard",
            probabilities=dict.fromkeys(COMPLEXITY_LEVELS, 0.25),
            confidence=0.8,
        ),
        sensitivity=ChoiceAnswer(
            choice=sensitivity,
            probabilities={level: (1.0 if level == sensitivity else 0.0) for level in SENSITIVITY_LEVELS},
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
        effective_sensitivity=sensitivity,
        effective_complexity="standard",
        uncertain=uncertain,
    )


class FakeClock:
    """A controllable monotonic clock: tests advance time instead of sleeping."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class CountingMockBackend(MockBackend):
    """MockBackend that counts decide() calls, so cache-before-routing is observable."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def decide(self, request: DecisionRequest) -> Any:
        self.calls += 1
        return await super().decide(request)


class FailingBackend:
    """A backend that raises: exercises the adapter\'s never-raise contract."""

    name = "failing"
    model_version = "fail-0.0.1"

    async def decide(self, request: DecisionRequest) -> Any:
        raise RuntimeError("backend exploded")

    async def aclose(self) -> None:
        return None


POLICY_TEMPLATE = """
version: 1
backend:
  name: {backend_name}
tiers:
  local: [qwen38]
  cheap: [qwen3.8-flash]
  strong: [qwen3.8-max]
tier_order: [cheap, strong]
rules:
  - id: gate.force-local
    if: gate_force_local
    then:
      tier: local
    reason: local hard gate matched a structured identifier or credential
  - id: data.sensitive
    if: sensitivity in ["confidential", "regulated"] or pii_present
    then:
      tier: local
    reason: sensitive or personal data must not leave the infrastructure
  - id: complexity.hard
    if: complexity in ["hard", "frontier"]
    then:
      tier: strong
    reason: task needs a strong model
  - id: default
    then:
      tier: cheap
    reason: routine task, no sensitivity signal
on_backend_down:
  mode: fail_closed
  fail_closed_tier: local
  fail_open_tier: strong
logging:
  enabled: true
  path: {log_path}
  excerpt_mode: hash
"""


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "decisions.jsonl"


@pytest.fixture
def router(tmp_path: Path, log_path: Path) -> Router:
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(POLICY_TEMPLATE.format(backend_name="mock", log_path=str(log_path)), encoding="utf-8")
    return Router(Policy.from_file(policy_path), MockBackend(), sink=JsonlSink(log_path))


@pytest.fixture
def adapter(router: Router) -> JevRouteClassifierAdapter:
    return JevRouteClassifierAdapter(router)


def assert_dict_shape(out: Any) -> None:
    """The adapter contract: exactly four keys, with these types, nothing else."""
    assert isinstance(out, dict)
    assert set(out) == EXPECTED_KEYS
    assert isinstance(out["gate_fired"], bool)
    assert isinstance(out["uncertain"], bool)
    assert out["tier"] is None or isinstance(out["tier"], str)
    assert out["model"] is None or isinstance(out["model"], str)


# --------------------------------------------------------------------------- #
# IntentCache: matching
# --------------------------------------------------------------------------- #
def test_lookup_misses_on_empty_cache() -> None:
    assert IntentCache().lookup("hello world") is None
    assert IntentCache().lookup("") is None
    assert IntentCache().lookup("   \t\n  ") is None


def test_lookup_hits_exact_after_normalization() -> None:
    cache = IntentCache()
    decision = make_decision()
    # Case, internal whitespace and a fullwidth character differ from the query;
    # NFKC + casefold + whitespace collapse must fold them onto the same key.
    cache.store("  How  Do I  Optimize the ｐ99 Latency ", decision)  # noqa: RUF001
    hit = cache.lookup("how do i optimize the p99 latency")
    assert hit is not None
    assert hit.matched == "exact"
    assert hit.similarity == 1.0
    assert hit.decision is decision
    stats = cache.stats()
    assert stats["exact_hits"] == 1 and stats["hits"] == 1 and stats["misses"] == 0


def test_punctuation_changes_the_match_path_not_the_answer() -> None:
    """A stray "?" is not the same normalized text, but it is the same intent:
    the Jaccard pass catches it at full similarity."""
    cache = IntentCache()
    decision = make_decision()
    cache.store("how do i optimize the p99 latency", decision)
    hit = cache.lookup("how do i optimize the p99 latency?")
    assert hit is not None
    assert hit.matched == "near"
    assert hit.similarity == 1.0
    assert hit.decision is decision


def test_lookup_hits_near_duplicate_at_jaccard_threshold() -> None:
    cache = IntentCache()
    decision = make_decision(tier="strong", model="qwen3.8-max")
    stored = "how do i optimize the p99 latency of the checkout service right now"
    cache.store(stored, decision)
    # Eleven of the stored twelve tokens, one word dropped: 11/12 = 0.917 >= 0.9.
    hit = cache.lookup("how do i optimize the p99 latency of the checkout service now")
    assert hit is not None
    assert hit.matched == "near"
    assert hit.decision is decision
    assert hit.similarity == pytest.approx(11 / 12, abs=1e-6)
    assert cache.stats()["near_hits"] == 1


def test_lookup_misses_below_jaccard_threshold() -> None:
    cache = IntentCache()
    stored = "how do i optimize the p99 latency of the checkout service right now"
    cache.store(stored, make_decision())
    # Shares 9 of 13 tokens = 0.69 < 0.9: a different ask.
    assert cache.lookup("how do i debug the race condition in the payments service tonight") is None
    assert cache.lookup("what is the best recipe for chocolate cake") is None
    stats = cache.stats()
    assert stats["misses"] == 2 and stats["hits"] == 0


def test_jaccard_is_symmetric_and_zero_on_empty() -> None:
    a, b = token_set("alpha beta gamma"), token_set("beta gamma delta")
    assert jaccard(a, b) == pytest.approx(2 / 4)
    assert jaccard(b, a) == jaccard(a, b)
    assert jaccard(frozenset(), a) == 0.0
    assert jaccard(a, frozenset()) == 0.0
    assert normalize_text("  A  b\n\tC ") == "a b c"


# --------------------------------------------------------------------------- #
# IntentCache: TTL and bounds
# --------------------------------------------------------------------------- #
def test_entries_expire_after_the_ttl() -> None:
    clock = FakeClock()
    cache = IntentCache(ttl_s=10.0, clock=clock)
    decision = make_decision()
    cache.store("weather today", decision)
    clock.advance(9.999)
    assert cache.lookup("weather today") is not None
    clock.advance(0.001)
    assert cache.lookup("weather today") is None
    assert cache.stats()["expired"] == 1


def test_zero_ttl_disables_storage() -> None:
    cache = IntentCache(ttl_s=0.0)
    cache.store("anything", make_decision())
    assert cache.stats()["size"] == 0
    assert cache.lookup("anything") is None


def test_lru_eviction_respects_max_entries() -> None:
    cache = IntentCache(max_entries=2)
    d1, d2, d3 = make_decision(model="a"), make_decision(model="b"), make_decision(model="c")
    cache.store("one one one", d1)
    cache.store("two two two", d2)
    cache.store("three three three", d3)
    assert cache.stats()["size"] == 2
    assert cache.stats()["evictions"] == 1
    assert cache.lookup("one one one") is None  # oldest evicted
    hit = cache.lookup("two two two")
    assert hit is not None and hit.decision is d2  # a lookup refreshes LRU order
    cache.store("four four four", make_decision(model="d"))
    hit2 = cache.lookup("two two two")
    assert hit2 is not None and hit2.decision is d2
    assert cache.lookup("three three three") is None


def test_restore_same_text_replaces_the_entry() -> None:
    cache = IntentCache()
    d1, d2 = make_decision(model="a"), make_decision(model="b")
    cache.store("same text", d1)
    cache.store("same text", d2)
    assert cache.stats()["size"] == 1
    assert cache.lookup("same text").decision is d2  # type: ignore[union-attr]


def test_store_rejects_non_decisions() -> None:
    cache = IntentCache()
    with pytest.raises(TypeError):
        cache.store("text", "not a decision")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# IntentCache: the gate contract
# --------------------------------------------------------------------------- #
def test_gate_fired_requests_are_never_stored() -> None:
    cache = IntentCache()
    cache.store(SENSITIVE, make_decision(gate_fired=True), gate_fired=True)
    assert cache.stats()["size"] == 0
    assert cache.lookup(SENSITIVE) is None
    assert cache.stats()["gate_skips"] == 1


def test_gate_fired_lookups_are_never_served() -> None:
    cache = IntentCache()
    decision = make_decision()
    cache.store("some ordinary prompt", decision)
    # Even though an entry exists, a gate-fired request gets nothing from the cache.
    assert cache.lookup("some ordinary prompt", gate_fired=True) is None
    assert cache.stats()["gate_skips"] == 1
    # ...and the ordinary path still works for requests the gate did not fire on.
    assert cache.lookup("some ordinary prompt") is not None


def test_cached_decision_carries_its_own_gate_verdict() -> None:
    """A hit can only be a gate-clean entry, so the cached verdict is authoritative.

    The adapter consults the cache BEFORE the gate has run for the new request;
    this is safe because the gate is deterministic on the text and gate-fired
    decisions are never stored, so the stored decision\'s verdict is exactly
    what a fresh gate pass would produce.
    """
    cache = IntentCache()
    clean = make_decision()
    cache.store("clean text", clean)
    hit = cache.lookup("clean text")
    assert hit is not None
    assert hit.decision.gate.fired is False


# --------------------------------------------------------------------------- #
# IntentCache: the semantic seam
# --------------------------------------------------------------------------- #
def test_semantic_match_seam_is_closed_by_default() -> None:
    cache = IntentCache()
    assert cache.semantic_match is None
    assert cache.stats()["semantic_match"] is False


def test_semantic_match_hook_is_consulted_only_on_miss() -> None:
    calls: list[str] = []
    cache = IntentCache()
    decision = make_decision(tier="strong")
    cache.store("alpha beta gamma delta", decision)

    def hook(text: str, candidates: list[CachedAnswer]) -> CachedAnswer | None:
        calls.append(text)
        return candidates[0] if candidates else None

    cache.semantic_match = hook
    # Exact hit: the hook must not even be asked.
    assert cache.lookup("alpha beta gamma delta") is not None
    assert calls == []
    # Miss: the hook is consulted with the live entries and its pick is stamped.
    hit = cache.lookup("entirely different words in here")
    assert hit is not None
    assert hit.matched == "semantic"
    assert hit.decision is decision
    assert calls == ["entirely different words in here"]
    assert cache.stats()["semantic_hits"] == 1


def test_semantic_match_hook_that_raises_is_a_miss() -> None:
    cache = IntentCache()
    cache.store("alpha beta gamma delta", make_decision())

    def hook(text: str, candidates: list[CachedAnswer]) -> CachedAnswer | None:
        raise RuntimeError("model call failed")

    cache.semantic_match = hook
    assert cache.lookup("entirely different words in here") is None


def test_semantic_match_hook_returning_garbage_is_a_miss() -> None:
    cache = IntentCache()
    cache.store("alpha beta gamma delta", make_decision())
    cache.semantic_match = lambda text, candidates: "not a cached answer"
    assert cache.lookup("entirely different words in here") is None


# --------------------------------------------------------------------------- #
# build_intent_cache: the policy config key
# --------------------------------------------------------------------------- #
def test_build_intent_cache_from_policy_raw() -> None:
    raw = {"intent_cache": {"enabled": True, "max_entries": 64, "ttl_s": 60}}
    cache = build_intent_cache(raw)
    assert isinstance(cache, IntentCache)
    assert cache.max_entries == 64
    assert cache.ttl_s == 60.0


def test_build_intent_cache_defaults_and_opt_out() -> None:
    cache = build_intent_cache({"intent_cache": {"enabled": True}})
    assert cache is not None
    assert cache.max_entries == 256 and cache.ttl_s == 3600.0
    assert build_intent_cache({}) is None  # key absent: disabled, the caller pays nothing
    assert build_intent_cache({"intent_cache": {"enabled": False}}) is None
    assert build_intent_cache(None) is None


def test_build_intent_cache_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        build_intent_cache({"intent_cache": {"max_entries": -5}})
    with pytest.raises(ValueError):
        build_intent_cache({"intent_cache": {"max_entries": "lots"}})
    with pytest.raises(ValueError):
        build_intent_cache({"intent_cache": {"ttl_s": -1}})
    with pytest.raises(ValueError):
        build_intent_cache({"intent_cache": "yes"})


# --------------------------------------------------------------------------- #
# The thin adapter: dict shape on the MockBackend
# --------------------------------------------------------------------------- #
def test_classify_returns_the_four_key_dict(adapter: JevRouteClassifierAdapter) -> None:
    out = adapter.classify(BENIGN)
    assert_dict_shape(out)


def test_classify_benign_routes_to_cheap(adapter: JevRouteClassifierAdapter) -> None:
    out = adapter.classify(BENIGN)
    assert out == {"tier": "cheap", "model": "qwen3.8-flash", "gate_fired": False, "uncertain": False}


def test_classify_hard_routes_to_strong(adapter: JevRouteClassifierAdapter) -> None:
    out = adapter.classify(HARD)
    assert_dict_shape(out)
    assert out["tier"] == "strong"
    assert out["gate_fired"] is False
    assert out["uncertain"] is False


def test_classify_sensitive_routes_local_with_gate_fired(adapter: JevRouteClassifierAdapter) -> None:
    out = adapter.classify(SENSITIVE)
    assert_dict_shape(out)
    assert out["tier"] == "local"
    assert out["gate_fired"] is True


def test_classify_is_deterministic_on_the_mock_backend(adapter: JevRouteClassifierAdapter) -> None:
    assert adapter.classify(BENIGN) == adapter.classify(BENIGN)
    assert adapter.classify(HARD) == adapter.classify(HARD)
    assert adapter.classify(SENSITIVE) == adapter.classify(SENSITIVE)


# --------------------------------------------------------------------------- #
# The thin adapter: construction contracts
# --------------------------------------------------------------------------- #
def test_default_construction_uses_mock_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No arguments: policy from the env var, and MockBackend even when the policy
    asks for the cloud without a credential -- a distribution entry point must
    work with no key and no network."""
    log = tmp_path / "decisions.jsonl"
    policy_path = tmp_path / "policy.yaml"
    # backend.name: jev with no API key: the adapter must still route, on the mock.
    policy_path.write_text(POLICY_TEMPLATE.format(backend_name="jev", log_path=str(log)), encoding="utf-8")
    monkeypatch.setenv("JEV_ROUTE_POLICY", str(policy_path))
    monkeypatch.setenv("JEV_ROUTE_DECISION_LOG", str(log))

    adapter = JevRouteClassifierAdapter()
    assert adapter.router.backend.name == "mock"
    out = adapter.classify(BENIGN)
    assert_dict_shape(out)
    assert out["tier"] == "cheap"


def test_default_construction_survives_a_missing_policy_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A named policy file that does not exist degrades to the built-in fallback:
    construction never raises, because this is a startup path elsewhere."""
    monkeypatch.setenv("JEV_ROUTE_POLICY", str(tmp_path / "does-not-exist.yaml"))
    monkeypatch.setenv("JEV_ROUTE_DECISION_LOG", str(tmp_path / "decisions.jsonl"))
    adapter = JevRouteClassifierAdapter()
    assert adapter.router.backend.name == "mock"
    assert_dict_shape(adapter.classify(BENIGN))


def test_explicit_backend_is_configurable(router: Router) -> None:
    backend = CountingMockBackend()
    adapter = JevRouteClassifierAdapter(backend=backend, policy_path=router.policy.source)
    assert adapter.router.backend is backend
    out = adapter.classify(BENIGN)
    assert_dict_shape(out)
    assert backend.calls == 1


def test_explicit_router_outranks_backend(router: Router) -> None:
    other = CountingMockBackend()
    adapter = JevRouteClassifierAdapter(router, backend=other)
    assert adapter.router is router
    adapter.classify(BENIGN)
    assert other.calls == 0  # the injected backend must never be used


def test_classify_declines_without_raising_when_the_backend_fails(tmp_path: Path) -> None:
    log = tmp_path / "decisions.jsonl"
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(POLICY_TEMPLATE.format(backend_name="mock", log_path=str(log)), encoding="utf-8")
    router = Router(Policy.from_file(policy_path), FailingBackend(), sink=JsonlSink(log))
    adapter = JevRouteClassifierAdapter(router)
    out = adapter.classify(BENIGN)
    assert out == {"tier": None, "model": None, "gate_fired": False, "uncertain": True}


# --------------------------------------------------------------------------- #
# The thin adapter: intent cache wiring (checked BEFORE routing)
# --------------------------------------------------------------------------- #
def test_intent_cache_is_checked_before_routing(tmp_path: Path) -> None:
    log = tmp_path / "decisions.jsonl"
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(POLICY_TEMPLATE.format(backend_name="mock", log_path=str(log)), encoding="utf-8")
    backend = CountingMockBackend()
    router = Router(Policy.from_file(policy_path), backend, sink=JsonlSink(log))
    adapter = JevRouteClassifierAdapter(router, intent_cache=IntentCache())

    text = "Summarize this customer support ticket about a broken printer"
    first = adapter.classify(text)
    assert_dict_shape(first)
    assert backend.calls == 1
    second = adapter.classify(text)
    assert second == first
    assert backend.calls == 1  # the near-duplicate was answered without the backend
    stats = adapter.intent_cache.stats()
    assert stats["hits"] == 1 and stats["size"] == 1


def test_intent_cache_never_stores_gate_fired_decisions(tmp_path: Path) -> None:
    log = tmp_path / "decisions.jsonl"
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(POLICY_TEMPLATE.format(backend_name="mock", log_path=str(log)), encoding="utf-8")
    router = Router(Policy.from_file(policy_path), MockBackend(), sink=JsonlSink(log))
    adapter = JevRouteClassifierAdapter(router, intent_cache=IntentCache())

    assert adapter.classify(SENSITIVE)["gate_fired"] is True
    assert adapter.intent_cache.stats()["size"] == 0  # the gate-fired decision was not kept
    assert adapter.intent_cache.stats()["gate_skips"] == 1
    # A re-ask of the sensitive text goes through the router again: it is never
    # served from the cache, so it is never "sticky" in the privacy sense.
    assert adapter.classify(SENSITIVE)["gate_fired"] is True
    assert adapter.intent_cache.stats()["size"] == 0

    adapter.classify(BENIGN)
    assert adapter.intent_cache.stats()["size"] == 1  # the clean one is kept


def test_intent_cache_from_policy_raw(tmp_path: Path) -> None:
    """The ``intent_cache:`` policy section builds the cache the adapter is told to use."""
    policy_path = tmp_path / "policy.yaml"
    log = str(tmp_path / "decisions.jsonl")
    policy_path.write_text(POLICY_TEMPLATE.format(backend_name="mock", log_path=log), encoding="utf-8")
    policy = Policy.from_file(policy_path)
    policy_raw_with_cache = {**policy.raw, "intent_cache": {"enabled": True, "max_entries": 16, "ttl_s": 300}}
    cache = build_intent_cache(policy_raw_with_cache)
    assert cache is not None and cache.max_entries == 16 and cache.ttl_s == 300.0
    assert build_intent_cache(policy.raw) is None  # the shipped template has no intent_cache section
