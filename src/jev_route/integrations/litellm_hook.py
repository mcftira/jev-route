"""LiteLLM proxy pre-call hook: jev-route with zero router coupling.

``litellm.integrations.custom_logger.CustomLogger.async_pre_call_hook`` runs on
the proxy before a request is dispatched, receives the parsed body as a plain
dict, and may return a modified dict. Rewriting ``data["model"]`` there changes
which model group LiteLLM routes to -- which is all a router needs to do, and
the reason this is the least invasive of the three integrations:

* no ``Router(plugins=[...])``, no ``auto_router/complexity_router`` entry, no
  tier machinery to configure;
* it works on any LiteLLM proxy version that has ``async_pre_call_hook``, which
  is a long-stable documented surface, rather than on the 1.101-era routing
  plugin pipeline;
* the decision still lands in jev-route's own JSONL log, and a compact summary
  lands in ``data["metadata"]["jev_route"]`` so it appears in LiteLLM's spend
  logs next to the cost of the request it caused.

Wire it up with::

    litellm_settings:
      callbacks: jev_route.integrations.litellm_hook.proxy_handler_instance

Two LiteLLM behaviours shape this file, and both are easy to get wrong:

**The hook must be defined on the leaf class.** ``ProxyLogging.pre_call_hook``
walks its callbacks and only calls ``async_pre_call_hook`` when
``"async_pre_call_hook" in vars(_callback.__class__)`` -- an override inherited
from an intermediate base class is invisible to it. So this class defines the
method directly and must not grow a mixin layer between it and ``CustomLogger``.

**``enforces_request_content`` must stay ``False``.** LiteLLM's own docstring for
that flag says a hook that *judges* content sets it True, while a hook that
*rewrites the payload for routing* stays False, because with True the hook is
replayed for every record of a batch upload and a per-record model rewrite would
be attributed to the wrong request. jev-route rewrites for routing. False.

The alternative surface, and why it is not used here: ``CustomLogger`` also
declares ``async_pre_routing_hook(model, request_kwargs, ...) ->
PreRoutingHookResponse | None``, which can return a modified model *and*
messages plus a ``routing_decision`` for LiteLLM's standard logging. It is the
better fit when you also want to rewrite the prompt, and it is dispatched only
for callbacks registered as pre-routing strategies, so implementing both would
not double-log in practice. This class still implements only the pre-call hook:
one decision per request, one record per decision, from a surface every proxy
version in use today actually calls. If you port it, keep exactly one of the two
active -- two live hooks means two decisions and two rows in the training set
for the same request, and a dataset that silently counts some requests twice is
worse than no dataset.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

from ..router import Router
from . import _shared
from ._shared import LOGGER, ROUTABLE_CALL_TYPES, SIGNALS_KEY

RouterFactory = Callable[[], Router]


def _extract_response_text(response_obj: Any) -> str:
    """Best-effort response text for the verifier; empty string when the shape
    is unrecognised (the verifier treats empty as a low-confidence signal,
    never as a crash)."""
    try:
        choices = getattr(response_obj, "choices", None) or []
        if choices:
            msg = getattr(choices[0], "message", None)
            if msg is not None and getattr(msg, "content", None):
                return str(msg.content)
        text = getattr(response_obj, "text", None)
        return str(text) if text else ""
    except Exception:
        return ""


class JevRoutePreCallHook(CustomLogger):
    """Rewrites ``data["model"]`` on the proxy to the model jev-route chose.

    Args:
        router: an explicit :class:`~jev_route.router.Router` or a zero-arg
            factory. ``None`` uses the process-wide router from
            :func:`._shared.get_router`, so the hook shares one decision cache,
            one circuit breaker and one log handle with every other integration
            in the process.
        managed_models: the model names this hook may rewrite. ``"*"`` (the
            default, from ``JEV_ROUTE_MANAGED_MODELS``) means every model;
            a comma-separated list means exactly those. Opt-in matters more here
            than anywhere else in jev-route, because a pre-call hook sees *every*
            request the proxy receives -- including ``/v1/embeddings``, direct
            calls to a model an operator deliberately pinned, and health checks.
        timeout_ms: per-request decision budget (``JEV_ROUTE_TIMEOUT_MS``,
            default 3000). LiteLLM gives pre-call hooks no timeout of their own.
        metadata_key: the key under ``data["metadata"]`` that carries the summary.
    """

    #: See the module docstring: a hook that rewrites for routing stays False, or
    #: LiteLLM replays it per record of a batch payload.
    enforces_request_content = False

    def __init__(
        self,
        router: Router | RouterFactory | None = None,
        *,
        managed_models: Sequence[str] | str | None = None,
        timeout_ms: float | None = None,
        metadata_key: str = SIGNALS_KEY,
    ) -> None:
        # CustomLogger.__init__ takes message-logging flags we do not care about;
        # passing nothing keeps LiteLLM's defaults, which is what an operator
        # registering this alongside other callbacks expects.
        super().__init__()
        self._router = router
        self._managed = _shared.parse_managed_models(managed_models)
        self._timeout_ms = timeout_ms
        self.metadata_key = metadata_key

    # -- construction ----------------------------------------------------- #
    @classmethod
    def from_env(cls, **kwargs: Any) -> JevRoutePreCallHook:
        """Build from the environment. Never raises: the proxy has to boot."""
        managed = kwargs.pop("managed_models", None)
        if managed is None:
            managed = os.environ.get(_shared.MANAGED_MODELS_ENV_VAR)
        instance = cls(managed_models=managed, **kwargs)
        LOGGER.info(
            "jev-route: pre-call hook ready; managing model%s %s.",
            "" if len(instance.managed_models) == 1 else "s",
            "*" if "*" in instance.managed_models else ", ".join(sorted(instance.managed_models)),
        )
        return instance

    _outcome_verifier: Any = None

    @property
    def managed_models(self) -> frozenset[str]:
        """The model names this hook is allowed to rewrite."""
        return self._managed

    @property
    def router(self) -> Router:
        """The router, resolved lazily so importing this module stays cheap."""
        if self._router is None:
            self._router = _shared.get_router()
        elif isinstance(self._router, Callable) and not isinstance(self._router, Router):
            self._router = self._router()
        return self._router

    # -- the hook --------------------------------------------------------- #
    async def async_pre_call_hook(
        self,
        # The first two are LiteLLM's signature, not ours: identity comes out of
        # data["metadata"] (see _shared.extract_metadata) and jev-route keeps its
        # own decision cache, so neither argument is read.
        user_api_key_dict: Any,  # noqa: ARG002
        cache: Any,  # noqa: ARG002
        data: dict,
        call_type: str,
    ) -> Exception | str | dict | None:
        """Route one proxy request.

        ``user_api_key_dict`` is a ``litellm.proxy.auth.auth_utils.UserAPIKeyAuth``
        and ``cache`` a ``litellm.caching.DualCache``; both are typed loosely here
        so this module does not depend on proxy internals that move between
        releases. Identity comes out of ``data["metadata"]`` instead, which is
        where LiteLLM already put the caller's key alias and team.

        Returns the (mutated) ``data`` when something changed and ``None``
        otherwise. LiteLLM treats a returned dict as the new body, a returned
        string as a rejection shown to the caller, and a returned ``Exception`` as
        a raised error. jev-route never rejects: a routing layer that can refuse
        a request is a guardrail, and jev-route is not one -- the local hard gate
        inside the router is what stops a request going somewhere it must not, and
        it does that by choosing the tier, not by blocking the caller.
        """
        try:
            return await self._pre_call(data, call_type)
        except Exception as exc:  # broad except, deliberately: documented fail-open at the integration layer
            LOGGER.warning(
                "jev-route: pre-call hook failed (%s: %s); passing the request through unrouted.",
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            return None

    async def _pre_call(self, data: dict, call_type: str) -> dict | None:
        if not isinstance(data, dict):
            return None
        if call_type not in ROUTABLE_CALL_TYPES:
            # Embeddings, reranks, images, audio, files, batches: no prompt to
            # judge and no chat tier behind them. Logging a "decision" for these
            # would put rows in the training set that no distilled model should
            # ever learn from.
            LOGGER.debug("jev-route: ignoring call_type %r.", call_type)
            return None

        requested_model = data.get("model")
        if not _shared.is_managed(requested_model, self._managed):
            LOGGER.debug("jev-route: model %r is not managed; leaving it alone.", requested_model)
            return None

        # Key *names* only, never values: this is the fastest way to answer "why
        # is my tenant rule not firing", which is usually "the proxy had not
        # stamped user_api_key_alias into data['metadata'] yet at pre-call time".
        LOGGER.debug("jev-route: pre-call payload keys: %s", sorted(str(k) for k in data))
        _shared.log_metadata_keys("pre-call payload", data.get("metadata"))

        metadata = _shared.extract_metadata(data)
        request_id = _shared.request_id_for(data, metadata)
        prompt = _shared.extract_prompt_text(data, call_type)
        messages = None if prompt is not None else _shared.extract_messages(data, call_type)
        if not prompt and not messages:
            LOGGER.debug("jev-route: no routable content in the payload; leaving the model alone.")
            return None

        decision, reason = await _shared.decide_with_reason(
            self.router,
            messages=messages,
            prompt=prompt,
            metadata=metadata,
            requested_model=requested_model if isinstance(requested_model, str) else None,
            request_id=request_id,
            timeout_ms=self._timeout_ms if self._timeout_ms is not None else _shared.timeout_ms(),
        )
        if decision is None:
            # No decision -> no rewrite. LiteLLM routes the model the caller
            # asked for, which the operator should have mapped to their safest
            # deployment (the example configs do exactly that).
            self._stamp(
                data,
                {
                    "tier": None,
                    "model": None,
                    "routed": False,
                    # The reason travels with the request into the spend log, so
                    # "why did jev-route not route this" is answerable from the
                    # file an operator is already looking at.
                    "reason": reason or "no decision",
                    "request_id": request_id,
                },
            )
            return data

        signals = _shared.decision_signals(decision, request_id=request_id)
        signals["requested_model"] = requested_model if isinstance(requested_model, str) else None
        self._stamp(data, signals)

        chosen = decision.model
        if chosen and chosen != requested_model:
            data["model"] = chosen
            LOGGER.debug(
                "jev-route: %s -> %s (tier=%s rule=%s)", requested_model, chosen, decision.tier, decision.rule_id
            )
        return data

    def _stamp(self, data: dict, signals: dict[str, Any]) -> None:
        """Put the summary where LiteLLM's spend log will carry it.

        ``data["metadata"]`` is copied into the standard logging payload, so this
        is the cheapest way to answer "why did this request cost what it cost"
        from the logs an operator already has. The full record -- soft
        distributions included -- is in jev-route's JSONL log, and
        ``request_id`` is the join key between the two.
        """
        metadata = data.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {} if metadata is None else {"_replaced": type(metadata).__name__}
            data["metadata"] = metadata
        metadata[self.metadata_key] = signals

    # -- success/failure observation -------------------------------------- #
    async def async_log_success_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: Any,  # noqa: ARG002 -- protocol, not ours to trim
        end_time: Any,  # noqa: ARG002
    ) -> None:
        """Debug-log the observed outcome. Deliberately writes no second record.

        It is tempting to append the upstream latency and the model that actually
        answered to the decision log, and that temptation should be refused: the
        decision log is a *dataset* with a versioned schema
        (:data:`jev_route.schema.SCHEMA_VERSION`, one row per decision), not an
        event stream. A second record kind would either duplicate decisions --
        so every naive ``pandas.read_json`` double-counts them -- or split one
        event across two rows that a reader has to know to join. Both make
        distillation silently wrong, and "silently" is the problem.

        Outcome attribution is a *distillation-time* concern, and it already has
        a key: ``request_id`` appears in the decision record and in
        ``data["metadata"]["jev_route"]``, which LiteLLM carries into its spend
        log. Join the two files when you want "did the chosen tier produce a
        good answer", and keep the training set one clean row per decision.
        """
        try:
            signals = ((kwargs or {}).get("litellm_params", {}) or {}).get("metadata", {})
            if not isinstance(signals, dict):
                signals = (kwargs or {}).get("metadata", {}) or {}
            summary = signals.get(self.metadata_key) if isinstance(signals, dict) else None
            model = (kwargs or {}).get("model")
            if isinstance(summary, dict):
                LOGGER.debug(
                    "jev-route: request_id=%s routed=%s -> served model=%s (observed latency is in "
                    "LiteLLM's spend log, keyed by the same request_id)",
                    summary.get("request_id"),
                    summary.get("model"),
                    model,
                )
                await self._verify_outcome(kwargs, response_obj, summary)
            else:
                LOGGER.debug("jev-route: success event for model=%s with no jev-route metadata.", model)
        except Exception as exc:  # broad except, deliberately: observation must never break a response
            LOGGER.debug("jev-route: async_log_success_event failed (%s); ignoring.", exc)

    async def _verify_outcome(self, kwargs: dict, response_obj: Any, summary: dict) -> None:  # noqa: ARG002 -- LiteLLM callback signature
        """Outcome verification (v0.3): one noul on the configured backend,
        appended as a `jev_route.outcome` record linked by request_id.

        Standing rules (from the work order, non-negotiable): never verify
        gate-blocked content; a failed verification logs completed_p=None and
        never touches the response; default-off in the policy."""
        try:
            cfg = {}
            raw = getattr(self.router.policy, "raw", None)
            if isinstance(raw, dict):
                cfg = raw.get("outcome_verification", {}) or {}
            if not cfg.get("enabled", False):
                return
            from ..outcome import OutcomeVerifier

            if self._outcome_verifier is None:
                self._outcome_verifier = OutcomeVerifier(
                    self.router.backend, self.router.sink,
                    sample_rate=float(cfg.get("sample_rate", 1.0)),
                )
            response_text = _extract_response_text(response_obj)
            await self._outcome_verifier.maybe_verify(
                request_id=str(summary.get("request_id") or ""),
                request_summary=str(summary.get("excerpt") or "")[:500],
                response_excerpt=response_text,
                model_served=str(summary.get("model") or ""),
                gate_blocked=bool(summary.get("gate_blocked")),
            )
        except Exception as exc:  # observation must never break a response
            LOGGER.debug("jev-route: outcome verification failed (%s); ignoring.", exc)

    async def async_log_failure_event(
        self,
        kwargs: dict,
        response_obj: Any,  # noqa: ARG002 -- signature fixed by LiteLLM's CustomLogger
        start_time: Any,  # noqa: ARG002 -- protocol, not ours to trim
        end_time: Any,  # noqa: ARG002
    ) -> None:
        """Mirror of :meth:`async_log_success_event`, for the same reason and with the same restraint."""
        try:
            model = (kwargs or {}).get("model")
            error = (kwargs or {}).get("exception")
            LOGGER.debug("jev-route: failure event for model=%s (%s); no record written.", model, error)
        except Exception as exc:  # broad except, deliberately
            LOGGER.debug("jev-route: async_log_failure_event failed (%s); ignoring.", exc)


#: Module-level instance for ``litellm_settings.callbacks``. LiteLLM resolves the
#: dotted path with ``get_instance_fn``, whose last segment must be an attribute
#: holding an instance -- the same convention as ``custom_callback.
#: proxy_handler_instance`` in LiteLLM's own docs.
proxy_handler_instance = JevRoutePreCallHook.from_env()

__all__ = ["JevRoutePreCallHook", "proxy_handler_instance"]
