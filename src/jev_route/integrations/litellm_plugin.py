"""LiteLLM routing plugin: jev-route as a stage in ``Router(plugins=[...])``.

This is the *SDK-facing* integration. LiteLLM 1.101 added a routing-plugin
pipeline to :class:`litellm.router.Router`: before a deployment is selected, the
router builds a :class:`litellm.types.router.RoutingContext` and runs it through
every plugin in order. Each plugin may mutate the context; the next one sees the
previous one's changes. Then the router keeps only the healthy deployments whose
``litellm_params.model`` survived in ``context.candidate_models``.

That contract is a good fit for jev-route and a bad fit for a naive
implementation of it, so three facts about LiteLLM drive the design below:

**1. ``candidate_models`` holds ``litellm_params.model`` strings, not the
``model_name`` an operator gave the deployment.** ``Router._run_routing_plugins``
seeds the list from ``resolved_litellm_models(model)``, and
``_filter_by_routing_plugin_candidates`` matches deployments on
``litellm_params.model``. So the tier -> models mapping this plugin is built with
must be in the *provider* namespace (``openai/qwen3.8``), not the alias
namespace (``qwen38``). Get that wrong and every decision falls through to the
safest-candidate path, which looks like "the plugin works but always picks
local" rather than like a config error.

**2. An empty candidate list is a 500, not a fallback.** The router raises
``ValueError("No deployments left after routing-plugin filtering")`` when a
plugin narrows to nothing, and it does so *deliberately* -- LiteLLM treats it as
a policy decision that must not be bypassed. jev-route has no use for that
escape hatch: a tier whose models are not among the candidates means the
operator's policy and their ``model_list`` disagree, which is a configuration
bug, and the right response to a configuration bug in a *safety* router is the
safest available candidate plus a loud signal -- not a dropped request. So this
plugin never returns an empty list.

**3. Narrowing to a same-length list is silently ignored.** The router only
stores the narrowed list when ``len(context.candidate_models) < len(original)``.
That is safe for a *filtering* plugin (a subset of the same length is the same
set, and the filter is set-based) but it is a trap for a plugin that tries to
reorder or replace candidates. Worth knowing before you extend this class.

Alternative integration points, and why this one is not the only door:
``litellm_classifier.JevRouteClassifier`` plugs the same decision into LiteLLM's
native complexity router, and ``litellm_hook.JevRoutePreCallHook`` rewrites the
model on the proxy with no router coupling at all.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..router import Router
from ..schema import RoutingDecision
from . import _shared
from ._shared import LOGGER

#: Where this plugin writes its summary inside ``context.signals``.
SIGNALS_KEY = _shared.SIGNALS_KEY

#: What the plugin does when jev-route produced no decision (timeout, internal
#: error, backend bug).
#:
#: ``leave``
#:     Return the context untouched and let LiteLLM pick from the whole pool.
#:     This is the default and the right one for most deployments: an
#:     integration-layer failure should cost a routing decision, not a request.
#:     The decision layer already fails closed on everything it can see -- a
#:     backend outage still produces a decision, for the safest tier -- so
#:     ``leave`` only applies when jev-route could not form an opinion at all.
#: ``safest``
#:     Narrow to the safest tier's candidates anyway. Choose this when the risk
#:     you care about is data egress rather than availability, because with
#:     ``leave`` a plugin that fails while a model group holds both local and
#:     cloud deployments lets LiteLLM load-balance an unjudged request across all
#:     of them. It still never returns an empty list: if the safest tier has no
#:     deployment in the pool, the pool is left alone and the decline is recorded.
DECLINE_MODES = ("leave", "safest")

RouterFactory = Callable[[], Router]


class JevRouteRoutingPlugin:
    """Narrows LiteLLM's candidate deployments to the tier jev-route chose.

    Args:
        router: an explicit :class:`~jev_route.router.Router`, or a zero-arg
            factory returning one. ``None`` uses the process-wide router from
            :func:`~._shared.get_router`, which is what a proxy deployment wants
            (one cache, one circuit breaker, one log handle).
        tier_models: tier name -> the ``litellm_params.model`` strings that serve
            it. ``None`` derives the mapping from the policy's own ``tiers:``,
            which is correct when the policy lists provider-qualified model
            strings and wrong when it lists LiteLLM aliases -- see fact 1 above.
        managed_models: the model groups this plugin is allowed to steer.
            Defaults to ``JEV_ROUTE_MANAGED_MODELS`` (``*`` = all). Best effort:
            :class:`RoutingContext` carries no ``model`` field, so the group is
            read from ``context.metadata["model_group"]`` when LiteLLM put it
            there and the check is skipped when it did not.
        timeout_ms: per-request decision budget. ``None`` reads
            ``JEV_ROUTE_TIMEOUT_MS``, defaulting to 3000.
        signals_key: where to write the summary in ``context.signals``.
    """

    def __init__(
        self,
        router: Router | RouterFactory | None = None,
        *,
        tier_models: Mapping[str, Sequence[str]] | None = None,
        managed_models: Sequence[str] | str | None = None,
        timeout_ms: float | None = None,
        decline_mode: str = "leave",
        signals_key: str = SIGNALS_KEY,
    ) -> None:
        self._router = router
        self._explicit_tier_models = (
            {str(tier): tuple(str(m) for m in models) for tier, models in tier_models.items()} if tier_models else None
        )
        self._managed = _shared.parse_managed_models(managed_models)
        self._timeout_ms = timeout_ms
        if decline_mode not in DECLINE_MODES:
            raise ValueError(f"decline_mode must be one of {sorted(DECLINE_MODES)}, got {decline_mode!r}")
        self.decline_mode = decline_mode
        self.signals_key = signals_key

    # -- construction ----------------------------------------------------- #
    @classmethod
    def from_env(cls, **kwargs: Any) -> JevRouteRoutingPlugin:
        """Build from the environment. Never raises: a plugin that cannot
        configure itself still has to let the request through."""
        return cls(router=None, **kwargs)

    # -- internals -------------------------------------------------------- #
    @property
    def router(self) -> Router:
        """The router, resolved lazily so importing this module has no side effects."""
        if self._router is None:
            self._router = _shared.get_router()
        elif isinstance(self._router, Callable) and not isinstance(self._router, Router):
            self._router = self._router()
        return self._router

    def tier_models(self) -> dict[str, tuple[str, ...]]:
        """Tier -> provider model strings. The explicit mapping wins over the policy's.

        Returns ``{}`` rather than raising when the policy is unreachable, because
        this is also called on the *decline* path -- see :meth:`_safety_ranking`.
        """
        if self._explicit_tier_models:
            return dict(self._explicit_tier_models)
        try:
            return {str(tier): tuple(models) for tier, models in self.router.policy.tiers.items()}
        except Exception as exc:  # broad except, deliberately: a broken router must not break the fallback too
            LOGGER.debug("jev-route: tier mapping unavailable (%s); nothing to narrow to.", exc)
            return {}

    def _safety_ranking(self) -> tuple[str, ...]:
        """Tiers ordered safest-first, used when the chosen tier is unavailable.

        "Safest" is not this plugin's opinion: it is the tier the operator named
        as ``on_backend_down.fail_closed_tier`` -- the place their own policy says
        data may always go. The rest follow the policy's ascending ``tier_order``,
        then any tier the order does not mention, and finally the operator's own
        ``tier_models`` declaration order.

        That last source matters more than it looks. This ranking is consulted on
        the decline path, which by definition runs when something already went
        wrong -- possibly the router itself. A fallback that depends on the object
        that just failed is not a fallback, so the policy read is wrapped and the
        declared mapping order stands in for it.
        """
        ranked: list[str] = []
        for tier in (*self._policy_ranking(), *self.tier_models()):
            if tier not in ranked:
                ranked.append(tier)
        return tuple(ranked)

    def _policy_ranking(self) -> tuple[str, ...]:
        """The policy's own safety order, or ``()`` when the policy is unreachable."""
        try:
            policy = self.router.policy
            fail_closed = policy.failure.fail_closed_tier
            ranked = [fail_closed] if fail_closed in policy.tiers else []
            ranked += [tier for tier in policy.tier_order if tier not in ranked]
            ranked += [tier for tier in policy.tiers if tier not in ranked]
            return tuple(ranked)
        except Exception as exc:  # broad except, deliberately: documented above
            LOGGER.debug("jev-route: policy safety ranking unavailable (%s).", exc)
            return ()

    def _select(self, decision: RoutingDecision, candidates: Sequence[str]) -> tuple[list[str], dict[str, Any] | None]:
        """Pick the candidates to keep. Never returns an empty list."""
        pool = list(candidates)
        if not pool:
            # Nothing to narrow. Returning [] here would be identical to the
            # input, but returning early keeps the "we never produce an empty
            # list" property obvious rather than incidental.
            return [], None

        tier_map = self.tier_models()
        # A rule may pin a concrete model (`then: {model: ...}`) with no tier at
        # all; honour the pin before the tier, since it is the more specific ask.
        pinned = [decision.model] if decision.model in pool else []
        tier_models = [m for m in tier_map.get(decision.tier, ()) if m in pool]

        if tier_models:
            # Keep the whole tier, not just the policy's round-robin pick: LiteLLM
            # still gets to apply cooldowns, health and load balancing inside it.
            return [m for m in pool if m in set(tier_models)], None
        if pinned:
            return pinned, None

        kept, tier = self._safest_candidates(pool, tier_map)
        if tier is not None:
            return kept, {
                "reason": "chosen tier has no deployments among the candidates",
                "requested_tier": decision.tier,
                "requested_model": decision.model,
                "chosen_tier": tier,
            }
        return pool, {
            "reason": "no tier mapping matched any candidate; candidates left unnarrowed",
            "requested_tier": decision.tier,
            "requested_model": decision.model,
            "chosen_tier": None,
        }

    def _safest_candidates(
        self, pool: Sequence[str], tier_map: Mapping[str, Sequence[str]]
    ) -> tuple[list[str], str | None]:
        """The candidates belonging to the safest tier present in ``pool``.

        Walks :meth:`_safety_ranking` and returns the first tier that has at
        least one live candidate, so the answer is the operator's own ordering
        rather than this plugin's idea of what is safe. ``(pool, None)`` when no
        mapped tier is available at all -- leaving the pool alone is better than
        inventing a preference.
        """
        for tier in self._safety_ranking():
            wanted = set(tier_map.get(tier, ()))
            group = [m for m in pool if m in wanted]
            if group:
                return group, tier
        return list(pool), None

    # -- the plugin interface --------------------------------------------- #
    async def run(self, context: Any) -> Any:
        """LiteLLM ``RoutingPlugin.run``. Returns the context, narrowed or untouched.

        Never raises. A routing plugin sits inside the caller's request path, and
        the only failure mode worse than "routed somewhere LiteLLM chose" is
        "the request 500'd because a telemetry-adjacent policy hook had a bug".
        The router already fails closed on the *decision*; this layer fails open
        on the *integration*. Every decline is logged and recorded in
        ``context.signals`` so it is visible in the spend log rather than silent.
        """
        try:
            return await self._run(context)
        except Exception as exc:  # broad except, deliberately: documented fail-open at the integration layer
            LOGGER.warning(
                "jev-route: routing plugin failed (%s: %s); leaving the request unrouted.",
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            self._record_error(context, exc)
            return context

    async def _run(self, context: Any) -> Any:
        requested_model = self._requested_model(context)
        if requested_model is not None and not _shared.is_managed(requested_model, self._managed):
            # Not ours to steer. No decision is made and nothing is logged: a
            # decision record for a request we did not route would be a lie in
            # the training set.
            return context

        messages = list(getattr(context, "structured_messages", None) or []) or list(
            getattr(context, "raw_messages", None) or []
        )
        candidates = list(getattr(context, "candidate_models", None) or [])
        if not messages:
            # Nothing to judge. This is not the "backend is down" case the router
            # fails closed on -- there is no request content to be conservative
            # about, and narrowing an unjudgeable call to the safest tier would
            # quietly move traffic the operator never asked us to move.
            LOGGER.debug("jev-route: no messages in the routing context; leaving candidates alone.")
            return context

        raw_metadata = getattr(context, "metadata", None) or {}
        _shared.log_metadata_keys("routing context", raw_metadata)
        metadata = _shared.filter_metadata(raw_metadata)
        request_id = _shared.request_id_for(None, metadata)
        decision, reason = await _shared.decide_with_reason(
            self.router,
            messages=messages,
            metadata=metadata,
            requested_model=requested_model,
            request_id=request_id,
            timeout_ms=self._timeout_ms if self._timeout_ms is not None else _shared.timeout_ms(),
        )
        if decision is None:
            self._record_error(context, None, note=reason or "no decision")
            if self.decline_mode == "safest" and candidates:
                kept, tier = self._safest_candidates(candidates, self.tier_models())
                if kept and tier is not None and len(kept) < len(candidates):
                    context.candidate_models = kept
                    self._amend_signals(context, {"declined_to_tier": tier, "declined_to": kept})
            return context

        kept, fallback = self._select(decision, candidates)
        signals = _shared.decision_signals(decision, request_id=request_id)
        signals["requested_model"] = requested_model
        if candidates:
            signals["candidates_before"] = len(candidates)
            signals["candidates_after"] = kept
        if fallback:
            signals["fallback_applied"] = fallback
        self._write_signals(context, signals)

        if kept and candidates and len(kept) < len(candidates):
            # Assign a new list rather than mutating in place: RoutingContext is a
            # pydantic model, and a fresh list makes the mutation obvious in a
            # debugger and immune to a caller holding the old reference.
            context.candidate_models = kept
        return context

    @staticmethod
    def _requested_model(context: Any) -> str | None:
        """The model group this request asked for, when LiteLLM recorded one.

        ``RoutingContext`` has no ``model`` field -- the pipeline is handed the
        resolved provider models instead -- so the group is read from metadata
        where the proxy puts it. ``None`` means "unknown", which this plugin
        treats as "managed", because refusing to route every SDK-side request
        for want of a metadata key would be worse than the opt-in it protects.
        """
        metadata = getattr(context, "metadata", None)
        if not isinstance(metadata, Mapping):
            return None
        for key in ("model_group", "model"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    def _write_signals(self, context: Any, signals: dict[str, Any]) -> None:
        container = getattr(context, "signals", None)
        if not isinstance(container, dict):
            return
        container[self.signals_key] = signals

    def _amend_signals(self, context: Any, extra: Mapping[str, Any]) -> None:
        """Merge keys into the summary already written for this request."""
        container = getattr(context, "signals", None)
        if not isinstance(container, dict):
            return
        current = container.get(self.signals_key)
        container[self.signals_key] = {**(current if isinstance(current, dict) else {}), **dict(extra)}

    def _record_error(self, context: Any, exc: Exception | None, *, note: str | None = None) -> None:
        """Leave a breadcrumb in ``signals`` even when there is no decision."""
        self._write_signals(
            context,
            {
                "tier": None,
                "model": None,
                "routed": False,
                "error": note or (f"{type(exc).__name__}: {exc}" if exc is not None else "unknown"),
            },
        )


#: Module-level instance, so ``Router(plugins=[...])`` and LiteLLM's dotted-path
#: resolution (``get_instance_fn``: last segment is an attribute holding an
#: instance) both find something usable with no configuration.
routing_plugin = JevRouteRoutingPlugin.from_env()

__all__ = ["DECLINE_MODES", "SIGNALS_KEY", "JevRouteRoutingPlugin", "routing_plugin"]
