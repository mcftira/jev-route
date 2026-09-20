"""Tests for the three LiteLLM integrations.

Two rules shape this file:

**Real LiteLLM types.** ``RoutingContext`` is imported from
``litellm.types.router`` and the plugin/classifier instances are asserted
against LiteLLM's own ``runtime_checkable`` protocols. A test that invented a
duck-typed context would keep passing after LiteLLM renamed a field, and the
failure would surface in a proxy at 3am instead of in CI.

**No network, no API key.** Every router here is built from a policy in
``tmp_path`` on the deterministic MockBackend, with the decision log pointed at
``tmp_path`` too. That is the same contract the core tests hold, and it is what
makes "the whole system runs end to end offline" a claim this repo can prove
rather than assert.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.router import ClassifierPlugin, RoutingContext, RoutingPlugin

from jev_route import JsonlSink, MockBackend, Policy, Router, iter_records
from jev_route.integrations import _shared
from jev_route.integrations.litellm_classifier import JevRouteClassifier
from jev_route.integrations.litellm_hook import JevRoutePreCallHook
from jev_route.integrations.litellm_plugin import JevRouteRoutingPlugin

#: These are framework-integration tests: they import litellm types on purpose.
pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
#: Three tiers, three model names in the ``litellm_params.model`` namespace,
#: because that -- not the operator-facing ``model_name`` -- is what
#: ``RoutingContext.candidate_models`` holds. See litellm_plugin's docstring.
TIER_MODELS = {
    "local": ("openai/local-llama",),
    "cheap": ("openai/cheap-flash",),
    "strong": ("openai/strong-max",),
}
ALL_CANDIDATES = [TIER_MODELS["local"][0], TIER_MODELS["cheap"][0], TIER_MODELS["strong"][0]]

POLICY_TEMPLATE = """
version: 1
backend:
  name: mock
gate:
  on_force_local: skip_backend
tiers:
  local: [openai/local-llama]
  cheap: [openai/cheap-flash]
  strong: [openai/strong-max]
tier_order: [cheap, strong]
pii_threshold: 0.5
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
on_uncertain:
  sensitivity_confidence_below: 0.8
  sensitivity_bump_levels: 1
  complexity_confidence_below: 0.7
  complexity_bump_levels: 1
  pii_uncertain_threshold: 0.35
  pii_uncertain_counts_as_present: true
on_backend_down:
  mode: fail_closed
  fail_closed_tier: local
  fail_open_tier: strong
cache:
  enabled: false
logging:
  enabled: true
  path: {log_path}
  excerpt_mode: hash
"""

#: Prompts chosen against the MockBackend so each one exercises a different rule.
#: The tier each produces is asserted in test_prompt_fixtures_still_route_as_expected,
#: so a change to the mock shows up as a named failure there rather than as a
#: mysterious one several tests away.
BENIGN_MESSAGES = [{"role": "user", "content": "Say hello and thank me for the recipe."}]
HARD_MESSAGES = [
    {
        "role": "user",
        "content": (
            "Design the architecture for a distributed consensus layer, derive the proof that it "
            "is safe under partition, then optimize the p99 latency and explain the trade-offs of "
            "sharding versus replication. Debug the intermittent race condition in the migration."
        ),
    }
]
SENSITIVE_MESSAGES = [
    {
        "role": "user",
        "content": (
            "Patient record for John Smith, SSN 123-45-6789, card 4111 1111 1111 1111, "
            "diagnosis and prescription attached. Summarize the clinical notes."
        ),
    }
]


@pytest.fixture(autouse=True)
def isolated_singleton(monkeypatch: pytest.MonkeyPatch):
    """Keep shared state out of the way of explicitly built objects.

    Two pieces of state can leak between tests here, and both have bitten:

    * the process-wide router, which the integrations fall back to when
      constructed without one, and which would carry one test's policy into the
      next;
    * ``JEV_ROUTE_*`` environment variables. These are read at construction time,
      so a developer who booted the example proxy in the same shell (or a CI job
      that exports them) would silently change what these tests assert. Every
      var is removed per test; tests that need one set it with ``monkeypatch``.
    """
    _shared.reset_router()
    for name in [k for k in os.environ if k.startswith("JEV_ROUTE_")]:
        monkeypatch.delenv(name, raising=False)
    yield
    _shared.reset_router()


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "decisions.jsonl"


@pytest.fixture
def router(tmp_path: Path, log_path: Path) -> Router:
    """A router on the deterministic mock backend, logging into tmp_path."""
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(POLICY_TEMPLATE.format(log_path=str(log_path)), encoding="utf-8")
    policy = Policy.from_file(policy_path)
    return Router(policy, MockBackend(), sink=JsonlSink(log_path))


@pytest.fixture
def plugin(router: Router) -> JevRouteRoutingPlugin:
    return JevRouteRoutingPlugin(router, tier_models=TIER_MODELS)


@pytest.fixture
def classifier(router: Router) -> JevRouteClassifier:
    return JevRouteClassifier(router, tiers=tuple(TIER_MODELS))


@pytest.fixture
def hook(router: Router) -> JevRoutePreCallHook:
    return JevRoutePreCallHook(router, managed_models=("auto",))


def make_context(
    messages: Sequence[Mapping[str, Any]] | None = None,
    *,
    candidates: Sequence[str] = ALL_CANDIDATES,
    metadata: Mapping[str, Any] | None = None,
) -> RoutingContext:
    """A real ``litellm.types.router.RoutingContext``, shaped the way the Router builds it."""
    raw = [dict(m) for m in (messages or BENIGN_MESSAGES)]
    return RoutingContext(
        raw_messages=raw,
        structured_messages=[dict(m) for m in raw],
        candidate_models=list(candidates),
        metadata=dict(metadata or {}),
    )


def records(log_path: Path) -> list[Any]:
    return list(iter_records(log_path))


# --------------------------------------------------------------------------- #
# Sanity: the fixtures themselves
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_prompt_fixtures_still_route_as_expected(router: Router) -> None:
    """Pin the fixture prompts to the tiers they are supposed to produce.

    Without this, a change to the MockBackend's markers would silently turn the
    narrowing tests into "the plugin narrows to whichever tier this prompt happens
    to land in", which still passes and no longer tests anything.
    """
    assert (await router.route_messages(BENIGN_MESSAGES)).tier == "cheap"
    assert (await router.route_messages(HARD_MESSAGES)).tier == "strong"
    sensitive = await router.route_messages(SENSITIVE_MESSAGES)
    assert sensitive.tier == "local"
    assert sensitive.gate.force_local, "the local hard gate must fire on this prompt"


# --------------------------------------------------------------------------- #
# RoutingPlugin
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_plugin_satisfies_litellm_protocol(plugin: JevRouteRoutingPlugin) -> None:
    """Litellm's proxy checks this at startup; a mismatch is a boot failure."""
    assert isinstance(plugin, RoutingPlugin)
    assert inspect.iscoroutinefunction(plugin.run)


@pytest.mark.asyncio
async def test_plugin_narrows_candidates_to_the_chosen_tier(plugin: JevRouteRoutingPlugin, router: Router) -> None:
    context = make_context(HARD_MESSAGES)
    expected = await router.route_messages([dict(m) for m in HARD_MESSAGES])

    result = await plugin.run(context)

    assert result is context, "a plugin mutates and returns the same context object"
    assert result.candidate_models == list(TIER_MODELS[expected.tier])
    signals = result.signals["jev_route"]
    assert signals["tier"] == expected.tier
    assert signals["rule_id"] == expected.rule_id
    assert signals["candidates_after"] == list(TIER_MODELS[expected.tier])
    assert "fallback_applied" not in signals


@pytest.mark.asyncio
async def test_plugin_signals_are_compact_json_and_carry_no_distributions(
    plugin: JevRouteRoutingPlugin,
) -> None:
    """Signals land in LiteLLM's request metadata and therefore in every
    downstream observability system. The full soft distributions belong in the
    JSONL decision log only -- copying them here would put the training set in
    somebody's Langfuse."""
    context = make_context(SENSITIVE_MESSAGES)
    result = await plugin.run(context)
    signals = result.signals["jev_route"]

    encoded = json.dumps(signals)  # must not raise: everything is JSON-serializable
    assert len(encoded) < 2000, f"signals are not compact: {len(encoded)} bytes"
    assert "probabilities" not in encoded
    assert "excerpt" not in encoded
    # ...but the key probabilities are there, because "why did this route" has to
    # be answerable from the spend log alone.
    assert 0.0 <= signals["pii"] <= 1.0
    assert 0.0 <= signals["sensitivity_confidence"] <= 1.0
    assert signals["gate_force_local"] is True


@pytest.mark.asyncio
async def test_plugin_never_returns_an_empty_candidate_list(plugin: JevRouteRoutingPlugin) -> None:
    """An empty candidate list is how a routing decision becomes a 500.

    LiteLLM raises ``No deployments left after routing-plugin filtering`` rather
    than falling back, so a tier whose models are absent from the pool must
    degrade to the safest *available* candidate.
    """
    # The chosen tier's models are not in the pool at all: only two unrelated
    # deployments survived health checks.
    pool = ["openai/other-a", "openai/other-b"]
    context = make_context(SENSITIVE_MESSAGES, candidates=pool)

    result = await plugin.run(context)

    assert result.candidate_models, "the plugin must never narrow to nothing"
    assert set(result.candidate_models) <= set(pool)
    fallback = result.signals["jev_route"]["fallback_applied"]
    assert fallback["requested_tier"] == "local"
    assert fallback["reason"]


@pytest.mark.asyncio
async def test_plugin_falls_back_to_the_safest_available_tier(plugin: JevRouteRoutingPlugin) -> None:
    """When the chosen tier is unavailable, fall back along the policy's own
    safety ranking -- not to an arbitrary candidate."""
    # `local` is missing from the pool, so the safest available is `cheap`.
    pool = [TIER_MODELS["cheap"][0], TIER_MODELS["strong"][0]]
    context = make_context(SENSITIVE_MESSAGES, candidates=pool)

    result = await plugin.run(context)

    assert result.candidate_models == [TIER_MODELS["cheap"][0]]
    assert result.signals["jev_route"]["fallback_applied"]["chosen_tier"] == "cheap"


@pytest.mark.asyncio
async def test_plugin_leaves_candidates_alone_when_nothing_maps(plugin: JevRouteRoutingPlugin) -> None:
    """A tier mapping that matches no candidate is a config bug, not a reason to
    strand the request."""
    unmatched = JevRouteRoutingPlugin(plugin.router, tier_models={"local": ("openai/nonexistent",)})
    context = make_context(BENIGN_MESSAGES, candidates=["openai/other-a", "openai/other-b"])

    result = await unmatched.run(context)

    assert result.candidate_models == ["openai/other-a", "openai/other-b"]
    assert result.signals["jev_route"]["fallback_applied"]["chosen_tier"] is None


@pytest.mark.asyncio
async def test_plugin_derives_tier_models_from_the_policy_when_unconfigured(router: Router) -> None:
    """The policy's own ``tiers:`` is the default mapping, which is right when the
    policy lists provider-qualified model strings."""
    bare = JevRouteRoutingPlugin(router)
    assert bare.tier_models() == {tier: tuple(models) for tier, models in router.policy.tiers.items()}

    context = make_context(BENIGN_MESSAGES)
    result = await bare.run(context)
    assert result.candidate_models == [TIER_MODELS["cheap"][0]]


@pytest.mark.asyncio
async def test_plugin_fails_open_on_an_internal_error(plugin: JevRouteRoutingPlugin) -> None:
    """A bug in the integration must cost a routing decision, never a request."""
    context = make_context(HARD_MESSAGES)
    before = list(context.candidate_models)

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("integration bug")

    plugin._select = explode  # type: ignore[method-assign]
    result = await plugin.run(context)

    assert result is context
    assert result.candidate_models == before
    assert "integration bug" in result.signals["jev_route"]["error"]
    assert result.signals["jev_route"]["routed"] is False


class ExplodingRouter:
    """A router whose decision path raises, standing in for a corrupted backend."""

    def __init__(self) -> None:
        self.calls = 0

    async def route_messages(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        raise ValueError("backend exploded")

    async def route_text(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        raise ValueError("backend exploded")


@pytest.mark.asyncio
async def test_plugin_fails_open_when_the_router_raises(plugin: JevRouteRoutingPlugin) -> None:
    broken = JevRouteRoutingPlugin(ExplodingRouter(), tier_models=TIER_MODELS)  # type: ignore[arg-type]
    context = make_context(HARD_MESSAGES)
    before = list(context.candidate_models)

    result = await broken.run(context)

    assert result.candidate_models == before
    assert result.signals["jev_route"]["routed"] is False
    assert "backend exploded" in result.signals["jev_route"]["error"]


@pytest.mark.asyncio
async def test_plugin_skips_models_it_does_not_manage(router: Router) -> None:
    """An opt-in list that is ignored is worse than no opt-in list."""
    scoped = JevRouteRoutingPlugin(router, tier_models=TIER_MODELS, managed_models=("auto",))
    context = make_context(HARD_MESSAGES, metadata={"model_group": "pinned-model"})
    before = list(context.candidate_models)

    result = await scoped.run(context)

    assert result.candidate_models == before
    assert "jev_route" not in result.signals


@pytest.mark.asyncio
async def test_plugin_skips_a_payload_with_nothing_to_judge(plugin: JevRouteRoutingPlugin) -> None:
    context = make_context([], candidates=ALL_CANDIDATES)
    context.structured_messages = []
    context.raw_messages = []

    result = await plugin.run(context)

    assert result.candidate_models == ALL_CANDIDATES


@pytest.mark.asyncio
async def test_plugin_decline_mode_defaults_to_leaving_the_pool(
    plugin: JevRouteRoutingPlugin,
) -> None:
    """The default is fail-open at the integration layer: no decision, no rewrite."""
    assert plugin.decline_mode == "leave"
    broken = JevRouteRoutingPlugin(ExplodingRouter(), tier_models=TIER_MODELS)  # type: ignore[arg-type]
    context = make_context(HARD_MESSAGES)
    before = list(context.candidate_models)

    result = await broken.run(context)

    assert result.candidate_models == before


@pytest.mark.asyncio
async def test_plugin_decline_mode_safest_narrows_without_a_decision() -> None:
    """For deployments where egress, not availability, is the risk worth taking.

    A group holding both local and cloud deployments would otherwise let LiteLLM
    load-balance an unjudged request across all of them.
    """
    guarded = JevRouteRoutingPlugin(ExplodingRouter(), tier_models=TIER_MODELS, decline_mode="safest")  # type: ignore[arg-type]
    context = make_context(HARD_MESSAGES)

    result = await guarded.run(context)

    assert result.candidate_models == [TIER_MODELS["local"][0]]
    signals = result.signals["jev_route"]
    assert signals["routed"] is False
    assert signals["declined_to_tier"] == "local"
    assert signals["declined_to"] == [TIER_MODELS["local"][0]]


@pytest.mark.asyncio
async def test_plugin_decline_mode_safest_still_never_empties_the_pool() -> None:
    guarded = JevRouteRoutingPlugin(ExplodingRouter(), tier_models=TIER_MODELS, decline_mode="safest")  # type: ignore[arg-type]
    pool = ["openai/other-a", "openai/other-b"]
    context = make_context(HARD_MESSAGES, candidates=pool)

    result = await guarded.run(context)

    assert result.candidate_models == pool


def test_plugin_rejects_an_unknown_decline_mode(router: Router) -> None:
    with pytest.raises(ValueError, match="decline_mode"):
        JevRouteRoutingPlugin(router, tier_models=TIER_MODELS, decline_mode="yolo")


@pytest.mark.asyncio
async def test_plugin_records_a_decision_even_when_it_cannot_narrow(
    plugin: JevRouteRoutingPlugin, log_path: Path
) -> None:
    """The decision log is the product. A request we saw is a row we owe."""
    context = make_context(HARD_MESSAGES, candidates=["openai/other-a"])
    await plugin.run(context)
    assert len(records(log_path)) == 1


# --------------------------------------------------------------------------- #
# ClassifierPlugin
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_classifier_satisfies_litellm_protocol(classifier: JevRouteClassifier) -> None:
    """``resolve_classifier_plugin`` rejects a sync ``classify`` at proxy startup."""
    assert isinstance(classifier, ClassifierPlugin)
    assert inspect.iscoroutinefunction(classifier.classify)


@pytest.mark.asyncio
async def test_classifier_returns_a_tier_name_not_a_model(classifier: JevRouteClassifier, router: Router) -> None:
    context = make_context(HARD_MESSAGES)
    expected = await router.route_messages([dict(m) for m in HARD_MESSAGES])

    tier = await classifier.classify(context)

    assert tier == expected.tier
    assert tier in TIER_MODELS
    assert tier not in ALL_CANDIDATES, "a classifier returns a tier name, never a model"


@pytest.mark.asyncio
async def test_classifier_returns_local_for_sensitive_traffic(classifier: JevRouteClassifier) -> None:
    assert await classifier.classify(make_context(SENSITIVE_MESSAGES)) == "local"


@pytest.mark.asyncio
async def test_classifier_declines_an_unknown_tier(router: Router, caplog: Any) -> None:
    """A tier the proxy never defined must decline, not error.

    The policy's tiers and the proxy's ``tier_definitions`` are two files; when
    they drift, ``None`` hands the request to LiteLLM's ``fallback_tier``.
    """
    narrow = JevRouteClassifier(router, tiers=("local",))
    context = make_context(BENIGN_MESSAGES)  # the policy sends this to `cheap`

    assert await narrow.classify(context) is None
    assert any("not in the configured tier set" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_classifier_declines_when_the_router_raises(caplog: Any) -> None:
    broken = JevRouteClassifier(ExplodingRouter(), tiers=tuple(TIER_MODELS))  # type: ignore[arg-type]
    assert await broken.classify(make_context(BENIGN_MESSAGES)) is None


@pytest.mark.asyncio
async def test_classifier_never_mutates_candidate_models(classifier: JevRouteClassifier) -> None:
    """On the classifier surface ``candidate_models`` is an informational snapshot
    of every tier's pool: the returned tier decides the pool, so narrowing it
    would be a no-op at best and a lie at worst."""
    context = make_context(HARD_MESSAGES)
    before = list(context.candidate_models)
    await classifier.classify(context)
    assert context.candidate_models == before


@pytest.mark.asyncio
async def test_classifier_on_decision_callback_runs_and_cannot_break_routing(
    router: Router,
) -> None:
    seen: list[tuple[str, str | None]] = []

    def observe(decision: Any, tier: str | None) -> None:
        seen.append((decision.tier, tier))
        raise RuntimeError("operator callback bug")

    observed = JevRouteClassifier(router, tiers=tuple(TIER_MODELS), on_decision=observe)
    tier = await observed.classify(make_context(BENIGN_MESSAGES))

    assert tier == "cheap", "a broken observer must not change the answer"
    assert seen == [("cheap", "cheap")]


@pytest.mark.asyncio
async def test_classifier_on_decision_callback_may_be_async(router: Router) -> None:
    seen: list[str] = []

    async def observe(decision: Any, tier: str | None) -> None:
        seen.append(str(tier))

    observed = JevRouteClassifier(router, tiers=tuple(TIER_MODELS), on_decision=observe)
    await observed.classify(make_context(HARD_MESSAGES))
    await asyncio.sleep(0)  # the callback is scheduled, not awaited: let it run
    assert seen == ["strong"]


@pytest.mark.asyncio
async def test_classifier_from_env_boots_without_a_policy_or_a_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The proxy resolves this dotted path at startup; it must not raise there."""
    monkeypatch.setenv(_shared.POLICY_ENV_VAR, str(tmp_path / "does-not-exist.yaml"))
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    instance = JevRouteClassifier.from_env()

    assert isinstance(instance, ClassifierPlugin)
    tier = await instance.classify(make_context(SENSITIVE_MESSAGES))
    assert tier == "local", "the built-in fallback policy still routes sensitive data local"


# --------------------------------------------------------------------------- #
# Pre-call hook
# --------------------------------------------------------------------------- #
def test_hook_is_a_litellm_custom_logger(hook: JevRoutePreCallHook) -> None:
    assert isinstance(hook, CustomLogger)
    # ProxyLogging only dispatches the hook when it is defined on the leaf class.
    assert "async_pre_call_hook" in vars(type(hook))
    assert type(hook).async_pre_call_hook is not CustomLogger.async_pre_call_hook
    # A hook that rewrites for routing must not be replayed per batch record.
    assert hook.enforces_request_content is False


@pytest.mark.asyncio
async def test_hook_rewrites_a_managed_model(hook: JevRoutePreCallHook, router: Router) -> None:
    data = {
        "model": "auto",
        "messages": [dict(m) for m in SENSITIVE_MESSAGES],
        "metadata": {"litellm_call_id": "call_abc123", "user_api_key_alias": "prod"},
    }
    expected = await router.route_messages([dict(m) for m in SENSITIVE_MESSAGES])

    result = await hook.async_pre_call_hook(None, None, data, "acompletion")

    assert result is data
    assert data["model"] == expected.model == TIER_MODELS["local"][0]
    stamped = data["metadata"]["jev_route"]
    assert stamped["tier"] == "local"
    assert stamped["requested_model"] == "auto"
    assert stamped["request_id"] == "call_abc123"


@pytest.mark.asyncio
async def test_hook_leaves_an_unmanaged_model_untouched(hook: JevRoutePreCallHook) -> None:
    data = {"model": "gpt-4o-pinned", "messages": [dict(m) for m in SENSITIVE_MESSAGES]}
    result = await hook.async_pre_call_hook(None, None, data, "acompletion")
    assert result is None
    assert data["model"] == "gpt-4o-pinned"
    assert "jev_route" not in data.get("metadata", {})


@pytest.mark.asyncio
async def test_hook_rewrites_every_model_by_default(router: Router) -> None:
    """``JEV_ROUTE_MANAGED_MODELS`` defaults to ``*``: an operator who did not
    scope the hook gets the hook, and the docs tell them to scope it."""
    everything = JevRoutePreCallHook(router)
    assert everything.managed_models == frozenset({"*"})
    data = {"model": "whatever-was-requested", "messages": [dict(m) for m in BENIGN_MESSAGES]}
    await everything.async_pre_call_hook(None, None, data, "acompletion")
    assert data["model"] == TIER_MODELS["cheap"][0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call_type", ["embedding", "aembedding", "rerank", "image_generation", "transcription", "create_file"]
)
async def test_hook_ignores_non_completion_call_types(hook: JevRoutePreCallHook, call_type: str) -> None:
    data = {"model": "auto", "input": "some text to embed", "messages": [dict(m) for m in BENIGN_MESSAGES]}
    assert await hook.async_pre_call_hook(None, None, data, call_type) is None
    assert data["model"] == "auto"


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", sorted(_shared.ROUTABLE_CALL_TYPES))
async def test_hook_accepts_every_routable_call_type(hook: JevRoutePreCallHook, call_type: str) -> None:
    data = {"model": "auto", "messages": [dict(m) for m in SENSITIVE_MESSAGES]}
    await hook.async_pre_call_hook(None, None, data, call_type)
    assert data["model"] == TIER_MODELS["local"][0]


@pytest.mark.asyncio
async def test_hook_routes_text_completions_from_prompt(hook: JevRoutePreCallHook, log_path: Path) -> None:
    """/v1/completions carries no message list, and the logged features must not
    claim a conversation that never existed."""
    data = {"model": "auto", "prompt": "Say hello and thank me for the recipe."}
    await hook.async_pre_call_hook(None, None, data, "text_completion")
    assert data["model"] == TIER_MODELS["cheap"][0]

    record = records(log_path)[0]
    assert record.features.n_messages == 1
    assert record.requested_model == "auto"


@pytest.mark.asyncio
async def test_hook_routes_anthropic_and_responses_payloads(hook: JevRoutePreCallHook) -> None:
    anthropic = {
        "model": "auto",
        "system": [{"type": "text", "text": "You are a clinical assistant."}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": SENSITIVE_MESSAGES[0]["content"]}]}],
    }
    await hook.async_pre_call_hook(None, None, anthropic, "anthropic_messages")
    assert anthropic["model"] == TIER_MODELS["local"][0]

    responses = {"model": "auto", "input": SENSITIVE_MESSAGES[0]["content"], "instructions": "be careful"}
    await hook.async_pre_call_hook(None, None, responses, "aresponses")
    assert responses["model"] == TIER_MODELS["local"][0]


@pytest.mark.asyncio
async def test_hook_passes_a_request_through_when_there_is_nothing_to_judge(
    hook: JevRoutePreCallHook,
) -> None:
    data = {"model": "auto", "metadata": {}}
    assert await hook.async_pre_call_hook(None, None, data, "acompletion") is None
    assert data["model"] == "auto"


@pytest.mark.asyncio
async def test_hook_fails_open_on_an_internal_error(router: Router) -> None:
    """Same contract as the other two surfaces: no decision, no rewrite, no 500."""
    broken = JevRoutePreCallHook(ExplodingRouter(), managed_models=("auto",))  # type: ignore[arg-type]
    data = {"model": "auto", "messages": [dict(m) for m in SENSITIVE_MESSAGES]}

    result = await broken.async_pre_call_hook(None, None, data, "acompletion")

    assert result is data
    assert data["model"] == "auto"
    assert data["metadata"]["jev_route"]["routed"] is False


@pytest.mark.asyncio
async def test_hook_never_raises_on_a_malformed_payload(hook: JevRoutePreCallHook) -> None:
    for data in ({"model": "auto", "messages": "not-a-list"}, {"model": "auto", "messages": [None, 3, {}]}):
        result = await hook.async_pre_call_hook(None, None, data, "acompletion")
        assert result in (None, data)


@pytest.mark.asyncio
async def test_hook_success_event_writes_no_second_record(hook: JevRoutePreCallHook, log_path: Path) -> None:
    """Dataset integrity: one decision, one row. Observing the outcome must not
    append a second record kind to a versioned schema."""
    data = {"model": "auto", "messages": [dict(m) for m in BENIGN_MESSAGES]}
    await hook.async_pre_call_hook(None, None, data, "acompletion")
    assert len(records(log_path)) == 1

    await hook.async_log_success_event(
        {"model": TIER_MODELS["cheap"][0], "metadata": data["metadata"]}, object(), None, None
    )
    await hook.async_log_failure_event({"model": "auto", "exception": ValueError("x")}, None, None, None)

    assert len(records(log_path)) == 1, "the success/failure hooks must not write to the dataset"
    assert data["metadata"]["jev_route"]["request_id"] == records(log_path)[0].request_id


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"messages": [{"role": "user", "content": "hi"}]}, [{"role": "user", "content": "hi"}]),
        (
            {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "a"}]},
                    {"role": "model", "content": "b"},
                ]
            },
            [
                {"role": "user", "content": [{"type": "text", "text": "a"}]},
                {"role": "assistant", "content": "b"},
            ],
        ),
        (
            {"system": "be brief", "messages": [{"role": "user", "content": "hi"}]},
            [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}],
        ),
        (
            {"system": [{"type": "text", "text": "be brief"}], "messages": [{"role": "user", "content": "hi"}]},
            [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}],
        ),
        (
            {"input": "do it", "instructions": "be nice"},
            [{"role": "system", "content": "be nice"}, {"role": "user", "content": "do it"}],
        ),
        (
            {
                "input": [
                    {"type": "message", "role": "user", "content": "a"},
                    {"type": "message", "role": "assistant", "content": "b"},
                ]
            },
            [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
        ),
        ({"input": [{"role": "user", "text": "a"}]}, [{"role": "user", "content": "a"}]),
        ({"prompt": "once upon a time"}, [{"role": "user", "content": "once upon a time"}]),
        ({"prompt": ["tok", "en", 123]}, [{"role": "user", "content": "tok en"}]),
        (
            {"contents": [{"role": "user", "parts": [{"text": "gem"}]}, {"role": "model", "parts": [{"text": "ok"}]}]},
            [{"role": "user", "content": "gem"}, {"role": "assistant", "content": "ok"}],
        ),
        ({}, []),
        ({"messages": []}, []),
        ({"messages": None, "input": None}, []),
    ],
)
def test_extract_messages_handles_every_wire_shape(payload: dict, expected: list) -> None:
    assert _shared.extract_messages(payload) == expected


def test_extract_messages_tolerates_garbage() -> None:
    assert _shared.extract_messages(None) == []
    assert _shared.extract_messages({"messages": [None, 3, {}, {"role": "user"}]}) == [{"role": "user", "content": ""}]


def test_extract_prompt_text_only_fires_for_a_bare_prompt() -> None:
    assert _shared.extract_prompt_text({"prompt": "hi"}) == "hi"
    assert _shared.extract_prompt_text({"prompt": "hi", "messages": [{"role": "user", "content": "x"}]}) is None
    assert _shared.extract_prompt_text({"messages": [{"role": "user", "content": "x"}]}) is None


def test_extract_metadata_strips_secrets_and_payload() -> None:
    """Metadata reaches both the training dataset and a cloud backend."""
    leaked = _shared.extract_metadata(
        {
            "user": "end-user-7",
            "metadata": {
                "user_api_key": "sk-live-SUPERSECRET",
                "user_api_key_hash": "deadbeef",
                "headers": {"Authorization": "Bearer sk-live-SUPERSECRET"},
                "messages": [{"role": "user", "content": "the prompt text"}],
                "prompt": "the prompt text",
                "raw_request": {"body": "the prompt text"},
                "user_api_key_alias": "prod-key",
                "user_api_key_team_alias": "team-a",
                "litellm_call_id": "call_1",
                "tags": ["prod", "eu"],
                "nested": {"deep": "value"},
                # An allowlisted key carrying far more than an identity should:
                # truncated, not trusted.
                "session_id": "x" * 5000,
            },
        }
    )
    encoded = json.dumps(leaked)
    assert "SUPERSECRET" not in encoded
    assert "the prompt text" not in encoded
    assert "deadbeef" not in encoded
    assert leaked["user_api_key_alias"] == "prod-key"
    assert leaked["end_user"] == "end-user-7"
    assert leaked["tags"] == ["prod", "eu"]
    assert len(leaked["session_id"]) == _shared.MAX_METADATA_VALUE_CHARS
    assert "nested" not in leaked
    assert "long" not in leaked


def test_filter_metadata_accepts_litellms_caller_tags_spelling() -> None:
    """A router/plugin context spells tags `caller_tags`; a proxy body spells them `tags`."""
    assert _shared.filter_metadata({"caller_tags": ("prod", "eu")})["tags"] == ["prod", "eu"]
    assert _shared.filter_metadata({"tags": ["prod"], "caller_tags": ["ignored"]})["tags"] == ["prod"]
    assert "tags" not in _shared.filter_metadata({"caller_tags": [{"not": "a scalar"}]})


#: The real key set LiteLLM 1.101 puts in a classifier plugin's context metadata,
#: captured from a running proxy. A regression test against the actual payload,
#: not against an imagined one.
LITELLM_101_METADATA_KEYS = (
    "user_api_key",
    "user_api_key_hash",
    "user_api_key_auth",
    "user_api_key_auth_metadata",
    "user_api_key_metadata",
    "user_api_key_team_metadata",
    "user_api_key_user_email",
    "headers",
    "requester_ip_address",
    "user_agent",
    "endpoint",
    "litellm_parent_otel_span",
    "requester_metadata",
    "queue_time_seconds",
    "attempted_retries",
    "attempted_fallbacks",
    "global_max_parallel_requests",
    "max_retries",
    "litellm_api_version",
    "litellm_received_at",
    "user_api_key_spend",
    "user_api_key_max_budget",
    "user_api_key_budget_reset_at",
    "inherited_tags",
    "agent_id",
    "model_group_size",
)


def test_the_real_litellm_metadata_blob_leaks_nothing() -> None:
    """Feed the keys LiteLLM actually sends, with values that must never be logged."""
    raw = {key: f"SECRET-{key}" for key in LITELLM_101_METADATA_KEYS}
    raw["headers"] = {"Authorization": "Bearer SECRET-token", "cookie": "SECRET-cookie"}
    raw.update(
        {
            "model_group": "auto",
            "model_group_alias": "auto-alias",
            "original_model_group": "auto",
            "user_api_key_alias": "prod-key",
            "user_api_key_team_id": "team_1",
            "caller_tags": ["prod"],
        }
    )

    kept = _shared.filter_metadata(raw)

    assert "SECRET" not in json.dumps(kept)
    assert kept == {
        "model_group": "auto",
        "model_group_alias": "auto-alias",
        "original_model_group": "auto",
        "user_api_key_alias": "prod-key",
        "user_api_key_team_id": "team_1",
        "tags": ["prod"],
    }


def test_request_id_finds_a_nested_litellm_call_id() -> None:
    assert _shared.request_id_for(None, {"requester_metadata": {"litellm_call_id": "c9"}}) == "c9"


def test_request_id_prefers_litellms_own_ids() -> None:
    assert _shared.request_id_for({"metadata": {"litellm_call_id": "c1", "request_id": "r1"}}) == "c1"
    assert _shared.request_id_for({"metadata": {"request_id": "r1"}}) == "r1"
    generated = _shared.request_id_for({})
    assert isinstance(generated, str) and len(generated) == 32


def test_decision_log_env_var_overrides_the_policy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log_path: Path
) -> None:
    """The deployment knows where storage is mounted; the policy file does not.

    ``JEV_ROUTE_DECISION_LOG`` moves the log and nothing else -- an environment
    variable must not be able to switch on prompt-text retention.
    """
    target = tmp_path / "var" / "log" / "jev-route" / "decisions.jsonl"
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        POLICY_TEMPLATE.format(log_path=str(log_path)).replace("name: mock", "name: mock"),
        encoding="utf-8",
    )
    monkeypatch.setenv(_shared.DECISION_LOG_ENV_VAR, str(target))

    built = _shared.build_router(policy_path, strict=True)

    assert Path(str(built.policy.logging["path"])) == target
    assert built.policy.logging["excerpt_mode"] == "hash", "only the path may be overridden"
    assert built.sink.path == target  # type: ignore[attr-defined]


def test_decision_log_env_var_is_ignored_when_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log_path: Path
) -> None:
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(POLICY_TEMPLATE.format(log_path=str(log_path)), encoding="utf-8")
    monkeypatch.setenv(_shared.DECISION_LOG_ENV_VAR, "   ")
    built = _shared.build_router(policy_path, strict=True)
    assert Path(str(built.policy.logging["path"])) == log_path


def test_build_router_degrades_to_mock_when_the_backend_needs_a_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No TYPESAFE_API_KEY must still produce a working, fail-closed router."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        POLICY_TEMPLATE.format(log_path=str(tmp_path / "d.jsonl")).replace(
            "backend:\n  name: mock", "backend:\n  name: jev"
        ),
        encoding="utf-8",
    )
    built = _shared.build_router(policy_path)

    assert built.backend.name == "mock"
    assert built.policy.failure.fail_closed_tier == "local", "the operator's rules survive the downgrade"


@pytest.mark.asyncio
async def test_decline_reason_reaches_the_signals(plugin: JevRouteRoutingPlugin) -> None:
    """The reason for not routing belongs in the spend log, not only in stderr."""
    broken = JevRouteRoutingPlugin(ExplodingRouter(), tier_models=TIER_MODELS)  # type: ignore[arg-type]
    context = make_context(HARD_MESSAGES)
    result = await broken.run(context)
    assert "backend exploded" in result.signals["jev_route"]["error"]

    declined = JevRoutePreCallHook(ExplodingRouter(), managed_models=("auto",))  # type: ignore[arg-type]
    data = {"model": "auto", "messages": [dict(m) for m in HARD_MESSAGES]}
    await declined.async_pre_call_hook(None, None, data, "acompletion")
    assert data["metadata"]["jev_route"]["routed"] is False
    assert "backend exploded" in data["metadata"]["jev_route"]["reason"]


# --------------------------------------------------------------------------- #
# Caller identity: what a proxy actually hands us, and what a rule can see
# --------------------------------------------------------------------------- #
def router_with_extra_rule(tmp_path: Path, log_path: Path, rule_yaml: str) -> Router:
    """:data:`POLICY_TEMPLATE` with one rule inserted above ``default``."""
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(POLICY_TEMPLATE.replace("  - id: default", rule_yaml + "  - id: default"), encoding="utf-8")
    return Router(Policy.from_file(policy_path), MockBackend(), sink=JsonlSink(log_path))


@pytest.mark.asyncio
async def test_caller_tags_reach_the_policy_engine(tmp_path: Path, log_path: Path) -> None:
    """A tenant rule can route on the caller's tags, and the record stays scalar.

    Verified against a live proxy (litellm 1.101): a client's ``metadata.tags``
    arrives in ``data["metadata"]["tags"]`` at pre-call time (``caller_tags`` is
    the router/plugin spelling and is empty there), so the allowlist accepts both
    and emits one name.

    Two behaviours are pinned here because they look like a bug from either side
    alone:

    * the list-valued ``tags`` **is** used for routing;
    * the list-valued ``tags`` is **not** in the persisted record, because
      ``Router`` keeps only scalar metadata (``router._is_scalar``) so every
      JSONL column stays flat. Scalars such as ``session_id`` do survive.
    """
    router = router_with_extra_rule(
        tmp_path,
        log_path,
        """  - id: tenant.phi
    if: '"tags" in metadata and "phi-workload" in metadata["tags"]'
    then:
      tier: local
    reason: this tenant's workload must not leave the infrastructure
""",
    )
    hook = JevRoutePreCallHook(router, managed_models=("auto",))
    data = {
        "model": "auto",
        "messages": [dict(m) for m in BENIGN_MESSAGES],
        "metadata": {
            "tags": ["prod", "phi-workload"],
            "session_id": "sess-42",
            "user_api_key": "sk-must-not-be-logged",
        },
    }
    await hook.async_pre_call_hook(None, None, data, "acompletion")

    assert data["model"] == TIER_MODELS["local"][0]
    row = json.loads(records(log_path)[-1].to_json())
    assert row["decision"]["rule_id"] == "tenant.phi"
    assert row["metadata"] == {"session_id": "sess-42"}


@pytest.mark.asyncio
async def test_hook_fails_open_when_a_rule_subscripts_absent_metadata(tmp_path: Path, log_path: Path) -> None:
    """A rule that assumes ``metadata["tags"]`` exists breaks requests without tags.

    ``Policy`` raises :class:`~jev_route.policy.PolicyError` for a subscript on an
    absent key -- the expression language allows no calls and no conditional
    expressions, so there is no ``metadata.get(...)`` form. The safe spelling is
    ``'"tags" in metadata and ...'``, which short-circuits (see the test above).

    What matters at this layer is the blast radius: the request is **not** failed
    and **not** mis-tiered. jev-route declines, LiteLLM routes the model the caller
    asked for, and the reason is stamped where an operator will see it.
    """
    router = router_with_extra_rule(
        tmp_path,
        log_path,
        """  - id: tenant.phi
    if: '"phi-workload" in metadata["tags"]'
    then:
      tier: local
    reason: this tenant's workload must not leave the infrastructure
""",
    )
    hook = JevRoutePreCallHook(router, managed_models=("auto",))
    data = {"model": "auto", "messages": [dict(m) for m in BENIGN_MESSAGES], "metadata": {"session_id": "s"}}
    await hook.async_pre_call_hook(None, None, data, "acompletion")

    assert data["model"] == "auto", "no decision means no rewrite"
    stamped = data["metadata"]["jev_route"]
    assert stamped["routed"] is False
    assert "tags" in stamped["reason"]


# --------------------------------------------------------------------------- #
# The thing the whole project is for
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_decisions_through_the_integration_keep_full_distributions(
    plugin: JevRouteRoutingPlugin, log_path: Path
) -> None:
    """Routing through LiteLLM must produce the same dataset as routing directly.

    Soft distributions are the reason to bootstrap on a calibrated model at all;
    an integration that logged only the argmax would quietly destroy the thing the
    distillation step needs.
    """
    await plugin.run(make_context(HARD_MESSAGES, metadata={"litellm_call_id": "call_xyz"}))
    await plugin.run(make_context(SENSITIVE_MESSAGES, metadata={"litellm_call_id": "call_pii"}))

    rows = records(log_path)
    assert [r.request_id for r in rows] == ["call_xyz", "call_pii"]
    for row in rows:
        raw = json.loads(row.to_json())
        answers = raw["decision"]["answers"]
        assert set(answers) == {"complexity", "sensitivity", "pii", "domain"}
        for question in ("complexity", "sensitivity", "domain"):
            probabilities = answers[question]["probabilities"]
            assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-6)
            assert len(probabilities) >= 4, "the whole distribution, not the argmax"
        assert "noul" in answers["pii"]
        assert raw["excerpt"] is None, "excerpt_mode: hash keeps prompt text out of the log"
        assert raw["kind"] == "jev_route.decision"
        assert raw["schema_version"] == "1"
