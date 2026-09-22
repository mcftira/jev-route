"""The router: one request in, one routing decision out.

The chain, in order, and the reason the order is not negotiable:

1. **Excerpt** the request down to a bounded slice of text.
2. **Gate** that raw slice with the local deterministic detectors. This runs
   before any model, in-process, with no network, and its verdict can only make
   the outcome stricter.
2b. **Score** the same slice with the semantic gate layer, which is also local
   and also can only make the outcome stricter. Shadow by default: it measures
   and logs until it has earned enforce. See :mod:`jev_route.gate_semantic`.
3. **Redact**, then decide whether a backend may be called at all. If the gate
   blocked the call, no text leaves the process -- not even redacted.
4. **Decide** via the configured :class:`~jev_route.backends.base.DecisionBackend`,
   with the cache in front of it.
5. **Merge** the gate's floors over the backend's answers, taking the max.
6. **Escalate** where confidence fell below the policy floor.
7. **Evaluate** the policy rules and resolve a tier and a concrete model.
8. **Log** the whole thing, soft distributions included.

Steps 5 and 6 are what make this a calibrated router rather than a classifier with
extra steps. Step 8 is what makes it a router you eventually own.

Step 8 writes two streams, not one. Every request produces a
:class:`~jev_route.schema.DecisionRecord`. A request the gate refused also
produces a :class:`~jev_route.schema.GateBlockRecord` -- detector ids, feature
vector, excerpt hash, and no text -- on a separate sink, because refusal
telemetry is how an operator improves the rules and must never be mistaken for
training data.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backends.base import BackendResult, DecisionBackend, DecisionRequest
from .cache import DecisionCache, build_cache
from .gate import HardGate
from .gate_semantic import SemanticAssessment, SemanticLayer, resolve_floor
from .logging_sink import DecisionSink, build_blocked_sink, build_sink
from .policy import Policy
from .prompts import (
    MAX_EXCERPT_CHARS,
    compute_features,
    excerpt_from_text,
    excerpt_messages,
    hash_text,
    redact,
    safe_caller_metadata,
)
from .schema import (
    SENSITIVITY_LEVELS,
    ChoiceAnswer,
    DecisionAnswers,
    DecisionRecord,
    GateBlockRecord,
    GateVerdict,
    NoulAnswer,
    RequestFeatures,
    RoutingDecision,
    level_index,
)

#: Sensitivity levels at which the gate's floor outranks a ``fail_open`` config.
_GATE_OVERRIDES_FAIL_OPEN = ("confidential", "regulated")


_BYPASS_MODEL_NAMES = frozenset({"jev-route/auto", "auto", ""})

#: ``routing.uncertain_fallback`` values with router-level meaning (v0.3).
#: ``hold_middle`` is the default; anything outside this tuple is a literal
#: model name and keeps the v0.2 behaviour (divert to that exact model, tier
#: recorded as ``"uncertain"``).
UNCERTAIN_FALLBACKS: tuple[str, ...] = ("hold_middle", "cheapest_local", "frontier")

#: ``degrade_reason`` substrings that mean "the provider is out of capacity"
#: rather than "the provider is down". Both spellings are the same condition:
#: HTTP 429 and "quota exceeded".
_QUOTA_REASON_MARKERS: tuple[str, ...] = ("429", "quota")


def _is_quota_error(reason: str | None) -> bool:
    """Whether a ``degrade_reason`` is a provider quota error, not an outage.

    A quota error means the tier is DRAINED: its primary model should stop
    receiving traffic for the rest of the process and its tandem deployment
    stands in. A timeout or a 5xx means the provider is DOWN, and marking a
    tier quota-exhausted on a blip would air-gap it for no reason, so only
    the two quota spellings qualify.
    """
    if not reason:
        return False
    lowered = reason.lower()
    return any(marker in lowered for marker in _QUOTA_REASON_MARKERS)


def _is_bypass(requested_model: str | None, policy: Policy) -> bool:
    """Pinned-model or routing-off pass-through (Jevonian's bypass rule).

    ``requested_model`` in this codebase is ALSO log metadata, so pinning is
    strictly opt-in: it only bypasses when the policy says
    ``routing.bypass_on_pinned: true``. ``routing.mode: "off"`` always
    bypasses (that knob means exactly one thing)."""
    routing_cfg = policy.raw.get("routing", {}) if isinstance(policy.raw, Mapping) else {}
    if str(routing_cfg.get("mode", "on")) == "off":
        return True
    if not routing_cfg.get("bypass_on_pinned", False):
        return False
    return bool(requested_model) and requested_model not in _BYPASS_MODEL_NAMES


class Router:
    """Routes requests to model tiers using calibrated decisions.

    Construct once per process and reuse. The router owns its backend, cache, and
    sink; call :meth:`aclose` on shutdown.
    """

    def __init__(
        self,
        policy: Policy,
        backend: DecisionBackend,
        *,
        gate: HardGate | None = None,
        cache: DecisionCache | None = None,
        sink: DecisionSink | None = None,
        blocked_sink: DecisionSink | None = None,
        semantic: SemanticLayer | None = None,
        shadow_backend: DecisionBackend | None = None,
        excerpt_mode: str | None = None,
        hash_salt: str | None = None,
        max_excerpt_chars: int = MAX_EXCERPT_CHARS,
        include_prior_turns: int = 2,
    ) -> None:
        self.policy = policy
        self.backend = backend
        # The gate always exists. There is no configuration in which it is None,
        # because there is no configuration in which unscanned text is routed.
        self.gate = gate if gate is not None else _gate_from_policy(policy)
        self.cache = cache if cache is not None else build_cache(policy.cache)
        self.sink = sink if sink is not None else build_sink(policy.logging)
        #: Refusal telemetry, on its own stream. Built here rather than lazily so
        #: a misconfigured path fails at startup and not on the first blocked
        #: request, which is the one moment you least want a logging surprise.
        self.blocked_sink = (
            blocked_sink
            if blocked_sink is not None
            else build_blocked_sink(policy.blocked_metadata, logging_config=policy.logging)
        )
        #: Layer 2. Constructing it is where ``mode: enforce`` is checked against
        #: the artifact's measured metrics, so an unearned promotion refuses to
        #: start instead of starting in a mode nobody asked for.
        self.semantic = semantic if semantic is not None else SemanticLayer(policy.semantic_gate)
        self.shadow_backend = shadow_backend
        self._shadow_enabled = bool((policy.shadow or {}).get("enabled")) and shadow_backend is not None
        self._shadow_sample_rate = float((policy.shadow or {}).get("sample_rate", 1.0))

        log_cfg = policy.logging or {}
        mode = excerpt_mode or str(log_cfg.get("excerpt_mode", "hash")).lower()
        if mode not in ("hash", "redacted", "none"):
            raise ValueError(f"logging.excerpt_mode must be hash|redacted|none, got {mode!r}")
        self.excerpt_mode = mode
        self.hash_salt = hash_salt if hash_salt is not None else str(log_cfg.get("hash_salt", ""))
        self.max_excerpt_chars = max_excerpt_chars
        self.include_prior_turns = include_prior_turns
        self._on_force_local = str((policy.gate or {}).get("on_force_local", "skip_backend")).lower()
        if self._on_force_local not in ("skip_backend", "still_classify"):
            raise ValueError(f"gate.on_force_local must be skip_backend|still_classify, got {self._on_force_local!r}")

    # -- construction ----------------------------------------------------- #
    @classmethod
    def from_policy_file(cls, path: str | Path, *, backend: DecisionBackend | None = None) -> Router:
        """Load a policy file and build the backend it names.

        This is the one-call entry point the CLI and the LiteLLM integrations use.
        """
        from .backends import build_backend

        policy = Policy.from_file(path)
        return cls(policy, backend or build_backend(policy), gate=_gate_from_policy(policy))

    # -- public API ------------------------------------------------------- #
    async def route_messages(
        self,
        messages: Sequence[Mapping[str, Any]] | None,
        *,
        metadata: Mapping[str, Any] | None = None,
        requested_model: str | None = None,
        request_id: str | None = None,
    ) -> RoutingDecision:
        """Route a chat-completions message list."""
        raw = excerpt_messages(
            messages,
            max_chars=self.max_excerpt_chars,
            include_prior_turns=self.include_prior_turns,
        )
        return await self._route(
            raw_excerpt=raw,
            messages=messages,
            metadata=metadata or {},
            requested_model=requested_model,
            request_id=request_id,
        )

    async def route_text(
        self,
        text: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        requested_model: str | None = None,
        request_id: str | None = None,
    ) -> RoutingDecision:
        """Route a bare string (text completions, CLI, evals)."""
        return await self._route(
            raw_excerpt=excerpt_from_text(text, max_chars=self.max_excerpt_chars),
            messages=None,
            metadata=metadata or {},
            requested_model=requested_model,
            request_id=request_id,
        )

    def route_text_sync(self, text: str, **kwargs: Any) -> RoutingDecision:
        """Blocking convenience wrapper for scripts and sync integrations."""
        return _run_coroutine(self.route_text(text, **kwargs))

    # -- the chain ---------------------------------------------------- #
    async def _route(
        self,
        *,
        raw_excerpt: str,
        messages: Sequence[Mapping[str, Any]] | None,
        metadata: Mapping[str, Any],
        requested_model: str | None,
        request_id: str | None,
    ) -> RoutingDecision:
        started = time.perf_counter()
        request_id = request_id or uuid.uuid4().hex

        # 1. Bypass: an explicitly pinned model, or the policy turning routing
        #    off, passes through untouched -- no gate, no backend, no log
        #    entry (the point of pinning is that the caller already decided).
        if _is_bypass(requested_model, self.policy):
            return RoutingDecision(
                tier="bypass",
                model=requested_model or "direct",
                rule_id="bypass",
                reason="pinned model; routing bypassed",
                answers=DecisionAnswers(
                    complexity=ChoiceAnswer(choice="standard", probabilities={}, confidence=1.0),
                    sensitivity=ChoiceAnswer(choice="public", probabilities={}, confidence=1.0),
                    pii=NoulAnswer(value=0.0),
                    domain=ChoiceAnswer(choice="chat", probabilities={}, confidence=1.0),
                ),
                gate=GateVerdict(),
                backend="bypass",
                backend_model_version="",
                effective_sensitivity="public",
                effective_complexity="standard",
                latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
            )

        # 2. Local hard gate, on the RAW excerpt. Always. Never bypassed.
        verdict = self.gate.scan(raw_excerpt)

        # 3. Redact for anything that may leave the process.
        redacted, _n_redacted = redact(raw_excerpt, self.gate)
        features = compute_features(
            raw_excerpt,
            messages=messages,
            gate_detectors=verdict.detectors(),
            n_gate_findings=len(verdict.findings),
            gate_force_local=verdict.force_local,
        )

        # 2b. The semantic layer, on the same RAW excerpt and still local. It runs
        #     after feature computation because a features-mode scorer reads them,
        #     and before the backend decision because in enforce mode it can make
        #     the request skip the backend entirely. Shadow mode returns an
        #     assessment whose `sensitivity_floor` and `force_local` are both inert,
        #     so this line cannot change a routing decision until it is promoted.
        assessment = self.semantic.assess(raw_excerpt, features=features, verdict=verdict)

        # Layer 2 forcing local is treated exactly like layer 1 forcing local for
        # the purposes of egress: the same `on_force_local` knob decides whether a
        # request that is staying local anyway still pays for a complexity read.
        # 2c. Deterministic tier pre-filter (see jev_route.prefilter): rule out
        #     every tier that could never serve this request BEFORE spending a
        #     backend call. Gate blocking already implied cloud removal; this
        #     generalises it (context, capabilities, quota, allow/deny). One
        #     surviving tier means routing is free: skip the backend and log
        #     with backend="prefilter" so the distill pipeline still sees it.
        from .prefilter import filter_tiers

        prefilter_cfg = self.policy.raw.get("prefilter", {}) if isinstance(self.policy.raw, Mapping) else {}
        pf = filter_tiers(
            tiers=tuple(self.policy.tiers.keys()),
            excerpt=raw_excerpt,
            features=features,
            verdict=verdict,
            assessment=assessment,
            config=prefilter_cfg,
        )
        # The gate's own force-local/blocking paths name themselves
        # (gate.force-local); prefilter only owns the case the gate left alone.
        prefilter_single = pf.single_candidate and not (
            verdict.blocks_backend or verdict.force_local or assessment.force_local
        )
        skip_backend = verdict.blocks_backend or prefilter_single or (
            (verdict.force_local or assessment.force_local) and self._on_force_local == "skip_backend"
        )
        excerpt_hash = hash_text(raw_excerpt, salt=self.hash_salt)

        backend_result: BackendResult
        backend_name: str
        shadow_report: dict[str, Any] | None = None
        #: True only when a backend actually classified this request. Gate-blocked
        #: and skipped requests have uniform answers by construction, so applying
        #: confidence-floor escalation to them would report a "hard -> frontier"
        #: bump that nobody ever judged. Honest provenance matters more in a
        #: training log than a tidy-looking one.
        classified = False

        if prefilter_single:
            # Free routing: one viable tier. The labels are deterministic, so
            # the log entry is a high-confidence training target -- and the
            # model call is never made.
            backend_result = _gate_only_result(verdict)
            backend_name = "prefilter"
        elif skip_backend:
            # Nothing is sent anywhere. The gate itself supplies the labels, which
            # are deterministic and therefore high-confidence training targets.
            backend_result = _gate_only_result(verdict)
            backend_name = "gate"
        else:
            request = DecisionRequest(
                redacted_excerpt=redacted,
                features=features,
                advisory_topics=verdict.advisory_topics,
                metadata=metadata,
                request_id=request_id,
            )
            cache_key = _cache_key(request, self.backend.name, self.policy.version)
            cached = self.cache.get(cache_key)
            if cached is not None:
                backend_result = replace(cached, latency_ms=0.0)
                backend_name = f"{self.backend.name}:cached"
            else:
                backend_result = await self.backend.decide(request)
                backend_name = self.backend.name
                self.cache.put(cache_key, backend_result)
            classified = True

            if self._shadow_enabled and _sample_hit(self._shadow_sample_rate, excerpt_hash):
                shadow_report = await self._run_shadow(request, backend_result)

        # 5. Merge the gate floor over the answer. Max, never min: the gate can
        #    only make this stricter, and no backend can talk its way out. Both
        #    gate layers go in at the same point and the same way.
        merged, escalations = _merge_gate(backend_result.answers, verdict, semantic=assessment)

        # 6. Escalate on low confidence.
        effective_complexity, effective_sensitivity, effective_pii = merged
        uncertainty = self.policy.uncertainty

        if (
            uncertainty.complexity_confidence_below is not None
            and backend_result.answers.complexity.confidence < uncertainty.complexity_confidence_below
            and classified
            and not backend_result.degraded
        ):
            before = effective_complexity
            effective_complexity = self.policy.escalate_complexity(
                effective_complexity, uncertainty.complexity_bump_levels
            )
            if effective_complexity != before:
                escalations.append(
                    f"complexity {before}->{effective_complexity} "
                    f"(confidence {backend_result.answers.complexity.confidence:.2f} "
                    f"< {uncertainty.complexity_confidence_below})"
                )

        if (
            uncertainty.sensitivity_confidence_below is not None
            and backend_result.answers.sensitivity.confidence < uncertainty.sensitivity_confidence_below
            and classified
            and not backend_result.degraded
        ):
            before = effective_sensitivity
            effective_sensitivity = self.policy.escalate_sensitivity(
                effective_sensitivity, uncertainty.sensitivity_bump_levels
            )
            if effective_sensitivity != before:
                escalations.append(
                    f"sensitivity {before}->{effective_sensitivity} "
                    f"(confidence {backend_result.answers.sensitivity.confidence:.2f} "
                    f"< {uncertainty.sensitivity_confidence_below})"
                )

        # Too uncertain to egress. Distinct from the bump above and from the gate:
        # the gate is a deterministic local finding, the bump is a one-level
        # gradient, and this is "the backend has no idea, so do not send it out".
        # Escalating to the top of the ladder rather than setting a separate flag
        # means the ordinary `data.sensitive` rule does the routing, so there is
        # exactly one code path that decides what stays local.
        force_local_floor = uncertainty.force_local_confidence_below
        if (
            force_local_floor is not None
            and classified
            and not backend_result.degraded
            and backend_result.answers.sensitivity.confidence < force_local_floor
        ):
            before = effective_sensitivity
            effective_sensitivity = self.policy.escalate_sensitivity(before, len(SENSITIVITY_LEVELS))
            if effective_sensitivity != before:
                escalations.append(
                    f"sensitivity {before}->{effective_sensitivity} "
                    f"(confidence {backend_result.answers.sensitivity.confidence:.2f} "
                    f"< {force_local_floor}: too uncertain to leave the infrastructure)"
                )

        pii_present = effective_pii >= self.policy.pii_threshold
        if (
            uncertainty.pii_uncertain_counts_as_present
            and not pii_present
            and effective_pii >= uncertainty.pii_uncertain_threshold
        ):
            pii_present = True
            escalations.append(
                f"pii treated as present at p={effective_pii:.2f} "
                f"(uncertain band >= {uncertainty.pii_uncertain_threshold})"
            )

        # The answers recorded on the decision are the *effective* ones, so a
        # reader of the log sees what the policy actually reasoned about. The raw
        # backend distribution is preserved alongside, unchanged.
        effective_answers = DecisionAnswers(
            complexity=replace(backend_result.answers.complexity, choice=effective_complexity),
            sensitivity=replace(backend_result.answers.sensitivity, choice=effective_sensitivity),
            pii=NoulAnswer(value=effective_pii),
            domain=backend_result.answers.domain,
        )

        namespace = _namespace(
            complexity=effective_complexity,
            sensitivity=effective_sensitivity,
            answers=backend_result.answers,
            pii=effective_pii,
            pii_present=pii_present,
            verdict=verdict,
            features=features,
            semantic=assessment,
            degraded=backend_result.degraded,
            metadata=metadata,
            requested_model=requested_model,
        )

        # 7. Resolve tier and model.
        if backend_result.degraded:
            tier, rule_id, reason = self._failure_tier(
                verdict, backend_result.degrade_reason, semantic=assessment
            )
            model = self.policy.pick_model(tier)
            if _is_quota_error(backend_result.degrade_reason):
                # v0.3 quota tandem: the provider ran out of capacity for this
                # tier; it is not down. Mark the tier quota-exhausted in the
                # prefilter config for the rest of the process (the router
                # re-reads policy.raw on every request, and the Policy is
                # process-lifetime, so the mark lives exactly as long as this
                # router) and serve THIS request from the tier's tandem
                # deployment instead of the tier's first model. A timeout or a
                # 5xx falls through here untouched: the plain degraded path.
                self._mark_tier_quota_exhausted(tier)
                tandem_model = self.policy.pick_quota_model(tier)
                if tandem_model != model:
                    escalations.append(
                        f"quota tandem flip: tier {tier} reported a provider quota error "
                        f"({backend_result.degrade_reason}); {tier} marked quota-exhausted for "
                        f"the rest of this process, serving tandem {tandem_model} instead of {model}"
                    )
                    model = tandem_model
                else:
                    escalations.append(
                        f"quota tandem: tier {tier} reported a provider quota error "
                        f"({backend_result.degrade_reason}); {tier} marked quota-exhausted for "
                        f"the rest of this process (no tandem configured for {tier})"
                    )
        else:
            rule, tier, model = self.policy.evaluate(namespace)
            rule_id, reason = rule.rule_id, rule.reason
        if backend_name == "prefilter":
            tier = pf.surviving[0]
            model = self.policy.pick_model(tier)
            rule_id = "prefilter.single_candidate"
            reason = "single viable tier after deterministic prefilter; backend call skipped"
            if not pf.surviving or pf.surviving[0] == pf.fallback_tier:
                reason = "no viable tier after deterministic prefilter; failed closed to local fallback"

        total_ms = (time.perf_counter() - started) * 1000.0
        decision = RoutingDecision(
            tier=tier,
            model=model,
            rule_id=rule_id,
            reason=reason,
            answers=effective_answers,
            gate=verdict,
            backend=backend_name,
            backend_model_version=backend_result.model_version,
            effective_sensitivity=effective_sensitivity,
            effective_complexity=effective_complexity,
            escalated=tuple(escalations),
            degraded=backend_result.degraded,
            degrade_reason=backend_result.degrade_reason,
            latency_ms=round(total_ms, 3),
            cached=backend_name.endswith(":cached"),
        )

        # 7b. Uncertainty marking (Jevonian's ledger rule): a shaky route is
        #     never silently accepted. Below the policy's min_confidence the
        #     decision is marked `uncertain: true`, and where it ends up is
        #     `routing.uncertain_fallback`, a v0.3 enum:
        #
        #     hold_middle   (default) keep the tier the policy chose; mark only
        #     cheapest_local divert to the local tier's first model
        #     frontier       divert to the strongest tier's first model
        #     <model-name>   divert to that exact model (the v0.2 behaviour)
        #
        # WHY the default is hold: live decision-log data shows confident-
        # downgrade is the dominant error class -- the router is confidently
        # sure of a WEAK tier, not unsure about a strong one. The confidence
        # bumps above already pushed the answer stricter, so on the remaining
        # uncertainty the safe move is to keep that choice and flag it for a
        # human, not to divert to a fallback that is cheaper and weaker by
        # default. The log doubles as a human-review queue either way.
        routing_cfg = self.policy.raw.get("routing", {}) if isinstance(self.policy.raw, Mapping) else {}
        min_conf = routing_cfg.get("min_confidence")
        if (
            min_conf is not None
            and classified
            and not backend_result.degraded
            and not decision.uncertain
            and backend_result.answers.complexity.confidence < float(min_conf)
        ):
            tier, model, note = self._resolve_uncertain_fallback(routing_cfg, decision)
            escalations.append(
                f"confidence {backend_result.answers.complexity.confidence:.2f} "
                f"< min_confidence {float(min_conf):.2f}: marked uncertain ({note})"
            )
            decision = replace(
                decision,
                tier=tier,
                model=model,
                escalated=tuple(escalations),
                uncertain=True,
            )

        # 8. Log. This is the training dataset being written.
        self._log(
            decision=decision,
            request_id=request_id,
            features=features,
            excerpt_hash=excerpt_hash,
            redacted_excerpt=redacted,
            gate_blocked=skip_backend,
            questions_sent=backend_result.questions_sent,
            backend_latency_ms=backend_result.latency_ms,
            total_latency_ms=total_ms,
            metadata=metadata,
            requested_model=requested_model,
            shadow=shadow_report,
            semantic=assessment,
            blocked_detectors=(
                # "Blocked" means the gate refused to let this request reach a
                # decision backend, so nothing left the process. Advisory-only
                # hits do not qualify: they never set force_local or blocks_backend.
                tuple(dict.fromkeys(f.detector for f in verdict.findings)) if skip_backend else ()
            ),
        )
        return decision

    def _failure_tier(
        self, verdict: GateVerdict, reason: str | None, *, semantic: SemanticAssessment | None = None
    ) -> tuple[str, str, str]:
        """Pick a tier when the backend could not answer.

        ``fail_open`` is honoured for quality risk, but never at the expense of the
        gate: if either local layer found sensitive data, the safest tier wins
        regardless of how the operator configured outages. A config knob is not
        allowed to become a data-egress path.

        The floor that outranks ``fail_open`` is the *combined* floor of both
        layers, the same ``resolve_floor`` maximum the normal path applies. A
        request whose only sensitivity signal came from layer 2 in enforce mode
        is the same egress risk during an outage as one layer 1 judged, and
        ``fail_open`` must not be the path that sends it out.
        """
        mode = self.policy.failure.mode
        tier = self.policy.failure.fail_open_tier if mode == "fail_open" else self.policy.failure.fail_closed_tier
        note = f"decision backend unavailable ({reason or 'unknown'}); {mode} -> tier {tier}"
        combined_floor = resolve_floor(
            verdict.sensitivity_floor, semantic.sensitivity_floor if semantic is not None else None
        )
        gate_index = level_index(combined_floor or "", SENSITIVITY_LEVELS)
        overrides = gate_index >= level_index(_GATE_OVERRIDES_FAIL_OPEN[0], SENSITIVITY_LEVELS)
        if (verdict.force_local or overrides) and mode == "fail_open":
            tier = self.policy.failure.fail_closed_tier
            note += f"; local gate floor took precedence over fail_open -> tier {tier}"
        return tier, "backend.down", note

    def _resolve_uncertain_fallback(
        self, routing_cfg: Mapping[str, Any], decision: RoutingDecision
    ) -> tuple[str, str, str]:
        """Resolve ``routing.uncertain_fallback`` for a request below min_confidence.

        Returns ``(tier, model, note)``; the note is a short human-readable
        outcome that goes into the decision's escalations, so the review queue
        can see what happened to the shaky route.
        """
        raw_value = routing_cfg.get("uncertain_fallback")
        value = str(raw_value).strip() if raw_value is not None else "hold_middle"
        if value in ("", "hold_middle"):
            return (
                decision.tier,
                decision.model,
                f"hold_middle: kept the policy's choice ({decision.tier}/{decision.model})",
            )
        if value == "cheapest_local":
            # The tier named "local" is the self-hosted tier by project
            # convention; a policy without one falls back to the cheapest tier
            # in tier_order (index 0 is the least capable by definition of the
            # ordering). First model, deliberately: the divert target must be
            # deterministic, a review-queue destination, not a round-robin.
            tier = "local" if "local" in self.policy.tiers else self.policy.tier_order[0]
            model = self.policy.models_for_tier(tier)[0]
            return tier, model, f"cheapest_local: diverted to {tier}/{model}"
        if value == "frontier":
            tier = self.policy.tier_order[-1]
            model = self.policy.models_for_tier(tier)[0]
            return tier, model, f"frontier: diverted to {tier}/{model}"
        # Anything else is a literal model name: the v0.2 behaviour, unchanged.
        return "uncertain", value, f"uncertain_fallback: diverted to {value}"

    def _mark_tier_quota_exhausted(self, tier: str) -> None:
        """Record a tier as quota-exhausted in the prefilter config.

        The mark lives in ``policy.raw["prefilter"]["quota_exhausted"]`` --
        the same key :func:`~jev_route.prefilter.filter_tiers` reads on every
        request -- so it persists for the rest of the process (a Policy is
        built once and owned by the router) and nowhere else. Idempotent: a
        repeat 429 must not duplicate the entry.
        """
        prefilter = self.policy.raw.get("prefilter")
        if not isinstance(prefilter, dict):
            prefilter = {}
            self.policy.raw["prefilter"] = prefilter
        marked = prefilter.get("quota_exhausted")
        if not isinstance(marked, list):
            marked = []
            prefilter["quota_exhausted"] = marked
        if tier not in marked:
            marked.append(tier)

    async def _run_shadow(self, request: DecisionRequest, primary: BackendResult) -> dict[str, Any]:
        """Run the shadow backend without letting it affect or delay the decision."""
        assert self.shadow_backend is not None
        try:
            shadow = await self.shadow_backend.decide(request)
        except Exception as exc:  # a shadow failure is telemetry, not an error
            return {"backend": self.shadow_backend.name, "error": f"{type(exc).__name__}: {exc}"}

        disagreements = {}
        for field_name in ("complexity", "sensitivity", "domain"):
            a = getattr(primary.answers, field_name).choice
            b = getattr(shadow.answers, field_name).choice
            if a != b:
                disagreements[field_name] = {"primary": a, "shadow": b}
        if abs(primary.answers.pii.value - shadow.answers.pii.value) >= 0.25:
            disagreements["pii"] = {
                "primary": round(primary.answers.pii.value, 4),
                "shadow": round(shadow.answers.pii.value, 4),
            }
        return {
            "backend": self.shadow_backend.name,
            "model_version": shadow.model_version,
            "answers": shadow.answers.to_dict(),
            "latency_ms": shadow.latency_ms,
            "degraded": shadow.degraded,
            "disagreements": disagreements,
            "agrees": not disagreements,
        }

    def _log(
        self,
        *,
        decision: RoutingDecision,
        request_id: str,
        features: RequestFeatures,
        excerpt_hash: str,
        redacted_excerpt: str,
        gate_blocked: bool,
        questions_sent: Mapping[str, Any],
        backend_latency_ms: float,
        total_latency_ms: float,
        metadata: Mapping[str, Any],
        requested_model: str | None,
        shadow: Mapping[str, Any] | None,
        semantic: SemanticAssessment | None = None,
        blocked_detectors: Sequence[str] = (),
    ) -> None:
        excerpt: str | None = None
        # Gate-blocked requests are never written as text, whatever the operator
        # configured. The same judgement that kept the excerpt out of a cloud API
        # keeps it out of the log.
        if self.excerpt_mode == "redacted" and not gate_blocked:
            excerpt = redacted_excerpt

        record = DecisionRecord(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            decision=decision,
            features=features,
            excerpt_hash=excerpt_hash,
            backend_latency_ms=round(backend_latency_ms, 3),
            total_latency_ms=round(total_latency_ms, 3),
            questions_sent=dict(questions_sent or {}),
            excerpt=excerpt,
            metadata=_loggable_metadata(metadata),
            requested_model=requested_model,
            shadow=dict(shadow) if shadow else None,
            semantic=semantic.to_dict() if semantic is not None and semantic.ran else None,
        )
        self.sink.write(record)

        if blocked_detectors:
            # Same request_id and the same timestamp as the decision record, so the
            # two streams can be joined without either of them carrying text.
            self.blocked_sink.write(
                GateBlockRecord(
                    request_id=record.request_id,
                    timestamp=record.timestamp,
                    detectors=tuple(dict.fromkeys(blocked_detectors)),
                    features=features,
                    excerpt_hash=record.excerpt_hash,
                )
            )

    async def aclose(self) -> None:
        for closable in (self.backend, self.shadow_backend):
            if closable is not None and hasattr(closable, "aclose"):
                # Shutdown must not raise: a backend that fails to close cleanly is
                # telemetry, not a reason to break the caller's teardown.
                with contextlib.suppress(Exception):
                    await closable.aclose()
        self.sink.close()
        # Closed separately and second: the two streams are separate files, and a
        # refusal record that never reaches disk is a rule-improvement signal lost.
        if self.blocked_sink is not self.sink:
            self.blocked_sink.close()

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {"cache": self.cache.stats(), "semantic_gate": self.semantic.stats()}
        if hasattr(self.sink, "stats"):
            out["log"] = self.sink.stats()
        if hasattr(self.blocked_sink, "stats"):
            out["blocked_log"] = self.blocked_sink.stats()
        return out


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _gate_from_policy(policy: Policy) -> HardGate:
    cfg = policy.gate or {}
    return HardGate(
        disabled_detectors=tuple(cfg.get("disabled_detectors") or ()),
        placeholder_domains_as_pii=bool(cfg.get("placeholder_domains_as_pii", False)),
    )


def _gate_only_result(verdict: GateVerdict) -> BackendResult:
    """Answers for a request the gate refused to send anywhere.

    The sensitivity answer is the gate's floor at full confidence, because a
    checksum-validated identifier is not a guess. Everything else is honest
    uncertainty: nobody classified it.
    """
    from .schema import COMPLEXITY_LEVELS, DOMAINS

    floor = verdict.sensitivity_floor or "internal"
    sensitivity = ChoiceAnswer(
        choice=floor,
        probabilities={level: (1.0 if level == floor else 0.0) for level in SENSITIVITY_LEVELS},
        confidence=1.0,
        confidence_reported=False,
    )
    return BackendResult(
        answers=DecisionAnswers(
            complexity=ChoiceAnswer.uniform(COMPLEXITY_LEVELS),
            sensitivity=sensitivity,
            pii=NoulAnswer(value=verdict.pii_floor if verdict.pii_floor is not None else 0.5),
            domain=ChoiceAnswer.uniform(DOMAINS),
        ),
        model_version="gate-1.0.0",
        questions_sent={},
        latency_ms=0.0,
        degraded=False,
    )


def _merge_gate(
    answers: DecisionAnswers, verdict: GateVerdict, *, semantic: SemanticAssessment | None = None
) -> tuple[tuple[str, str, float], list[str]]:
    """Apply both gate layers' floors to a backend answer. Returns effective values + notes.

    The two floors are combined with :func:`~jev_route.gate_semantic.resolve_floor`
    before either is applied, and that function is a maximum. So the invariant
    "layer 2 may make routing stricter, never looser" holds by construction: a
    semantic layer that answers ``public`` contributes ``None``, and a
    deterministic ``regulated`` survives whatever layer 2 said.

    ``semantic`` is keyword-only and optional because ``jev_route.distill.evaluate``
    replays this function over historical records, which have no layer-2
    assessment and must keep producing the answers they produced at the time.
    """
    escalations: list[str] = []
    sensitivity = answers.sensitivity.choice
    semantic_floor = semantic.sensitivity_floor if semantic is not None else None
    combined_floor = resolve_floor(verdict.sensitivity_floor, semantic_floor)
    if combined_floor:
        before = sensitivity
        sensitivity = _max_level(sensitivity, combined_floor)
        if sensitivity != before:
            # Name the layer that raised it. "The gate said so" is not an
            # explanation an operator can act on once there are two gates.
            origin = f"local gate floor: {verdict.sensitivity_floor}"
            semantic_outranks = (
                semantic is not None
                and semantic_floor is not None
                and level_index(semantic_floor, SENSITIVITY_LEVELS)
                > level_index(verdict.sensitivity_floor or "", SENSITIVITY_LEVELS)
            )
            if semantic_outranks and semantic is not None:
                origin = f"semantic gate layer 2 asserted {semantic_floor} at p={semantic.score:.2f}"
            escalations.append(f"sensitivity {before}->{sensitivity} ({origin})")
    pii = answers.pii.value
    if verdict.pii_floor is not None and verdict.pii_floor > pii:
        escalations.append(f"pii {pii:.2f}->{verdict.pii_floor:.2f} (local gate matched an identifier)")
        pii = verdict.pii_floor
    return (answers.complexity.choice, sensitivity, pii), escalations


def _max_level(a: str, b: str) -> str:
    return a if level_index(a, SENSITIVITY_LEVELS) >= level_index(b, SENSITIVITY_LEVELS) else b


def _namespace(**kwargs: Any) -> dict[str, Any]:
    answers: DecisionAnswers = kwargs["answers"]
    features: RequestFeatures = kwargs["features"]
    verdict: GateVerdict = kwargs["verdict"]
    #: Read with ``.get`` so a caller replaying historical decisions -- which have
    #: no layer-2 assessment -- does not have to invent one.
    semantic: SemanticAssessment | None = kwargs.get("semantic")
    # Rules see the *actionable* projection of layer 2, not the raw assessment:
    # zeroed unless it is enforcing. Shadow mode's contract is that it cannot
    # change a routing decision, and a rule reading `semantic_fired` would break
    # that contract without anybody deciding to. `semantic_mode` is always the
    # truth, so a policy can tell the two cases apart; the full assessment, score
    # included, is on the logged record either way.
    acts = semantic is not None and semantic.enforced
    return {
        "complexity": kwargs["complexity"],
        "sensitivity": kwargs["sensitivity"],
        "domain": answers.domain.choice,
        "complexity_confidence": answers.complexity.confidence,
        "sensitivity_confidence": answers.sensitivity.confidence,
        "domain_confidence": answers.domain.confidence,
        "pii": kwargs["pii"],
        "pii_present": kwargs["pii_present"],
        "gate_force_local": verdict.force_local,
        "gate_blocks_backend": verdict.blocks_backend,
        "gate_sensitivity_floor": verdict.sensitivity_floor,
        "gate_detectors": {f.detector for f in verdict.findings},
        "advisory_topics": set(verdict.advisory_topics),
        "semantic_fired": bool(acts and semantic.fired),
        "semantic_score": float(semantic.score) if acts else 0.0,
        "semantic_force_local": bool(acts),
        "semantic_mode": semantic.mode if semantic is not None else "off",
        "degraded": kwargs["degraded"],
        "cached": False,
        "tier": "",
        "model": "",
        "char_len": features.char_len,
        "word_count": features.word_count,
        "line_count": features.line_count,
        "code_blocks": features.code_blocks,
        "question_marks": features.question_marks,
        "has_stack_trace": features.has_stack_trace,
        "lang": features.lang,
        "n_prior_turns": features.n_prior_turns,
        "requested_model": kwargs["requested_model"] or "",
        "metadata": dict(kwargs["metadata"] or {}),
    }


def _cache_key(request: DecisionRequest, backend_name: str, policy_version: int) -> str:
    payload = "\x00".join(
        [
            f"v{policy_version}",
            backend_name,
            request.redacted_excerpt,
            ",".join(request.advisory_topics),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _loggable_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Caller metadata that is safe to write into the decision log.

    Two independent filters, and both are necessary:

    * by KEY, via :data:`~jev_route.prompts.UNSAFE_METADATA_KEYS` -- the same list
      the cloud path applies. Without it, ``metadata={"prompt": "..."}`` writes
      the raw prompt into the training dataset, which silently defeats
      ``excerpt_mode: hash``. ``DecisionRecord`` documents metadata as "never the
      prompt"; this is what makes that true.
    * by TYPE, keeping scalars only -- the log is JSONL read back for training, so
      a nested dict or list in one record is a schema surprise for every consumer.
    """
    return {k: v for k, v in safe_caller_metadata(metadata).items() if _is_scalar(v)}


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _sample_hit(rate: float, key: str) -> bool:
    """Deterministic sampling from the excerpt hash, so replays are reproducible."""
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < rate


def _run_coroutine(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


__all__ = ["Router"]
