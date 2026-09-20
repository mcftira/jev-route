"""The jev-route decision schema.

This module is the contract every other part of the system depends on, and it is
also the contract of the *dataset* jev-route produces: every routing decision is
persisted as a :class:`DecisionRecord`, and those records are what the
``jev_route.distill`` pipeline trains on later.

Two rules follow from that, and they are the reason this file is boring and
explicit:

1. **Records are schema-versioned.** ``SCHEMA_VERSION`` is stamped on every
   record. A field may be added; it may not be silently repurposed, renamed, or
   changed in type. If the shape must change, bump ``SCHEMA_VERSION`` and add a
   migration in ``jev_route.distill.export``.
2. **Soft answers are preserved in full.** The whole point of routing on a
   calibrated model is the probability distribution behind the argmax. Storing
   only the winning label would throw away the signal that makes distillation
   work, so :class:`ChoiceAnswer` always carries every option's probability.

Everything here is stdlib-only and immutable. No framework types leak into the
core, which is what lets the same record flow through the LiteLLM integrations,
the CLI, the JSONL sink, and the trainer.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

#: Bump on any breaking change to :class:`DecisionRecord`. Additive changes that
#: keep old records readable do not require a bump; renaming, retyping, or
#: removing a field does.
SCHEMA_VERSION = "1"

#: Discriminator written into every record so a JSONL file can be filtered even
#: when it shares a stream with other event kinds.
RECORD_KIND = "jev_route.decision"

#: Discriminator for the gate's refusal stream. A different kind *and* a different
#: file, because the two streams have different jobs and different lifetimes:
#: decision records are the training dataset, refusal records are how an operator
#: works out which detector to write next. See :class:`GateBlockRecord`.
GATE_BLOCK_RECORD_KIND = "jev_route.gate_block"

COMPLEXITY_LEVELS: tuple[str, ...] = ("trivial", "standard", "hard", "frontier")
SENSITIVITY_LEVELS: tuple[str, ...] = ("public", "internal", "confidential", "regulated")
DOMAINS: tuple[str, ...] = ("code", "writing", "analysis", "chat", "data-extraction")
TIERS: tuple[str, ...] = ("local", "cheap", "strong")

#: A noul answer at or above this probability counts as "the thing is present".
#: The policy engine can override it; this is only the schema-level default.
DEFAULT_NOUL_THRESHOLD = 0.5

#: Which decision backends exist. ``jev`` is the cloud bootstrap, ``mock`` runs
#: with no key at all, ``distilled`` is a local model trained from your own log,
#: and ``laya`` is the pretrained local edition (fully air-gapped).
BackendName = Literal["jev", "mock", "distilled", "laya", "shadow"]


# --------------------------------------------------------------------------- #
# Uncertainty helpers
# --------------------------------------------------------------------------- #
def certainty_from_probabilities(probabilities: Mapping[str, float]) -> float:
    """Normalized-entropy certainty of a distribution, in ``[0, 1]``.

    ``1.0`` means all mass on one option; ``0.0`` means a perfectly uniform
    spread. Used for backends that do not report their own confidence (the
    distilled backend), so that policy thresholds mean the same thing regardless
    of which backend answered.

    Two edge cases matter and are easy to get backwards:

    * **A single non-zero option is certain**, and returns ``1.0``.
    * **An all-zero distribution carries no information** and returns ``0.0``,
      not ``1.0``. It shows up when a backend returns probabilities that failed
      to parse, and treating it as maximum certainty would send the policy
      engine's confidence floors the exact wrong signal.

    For two options this is ``1 - H(p)/ln 2``, which is monotone in the margin
    but *not* equal to it: ``{0.9, 0.1}`` gives ~0.53 here and 0.8 as a margin.
    :attr:`NoulAnswer.confidence` deliberately uses the margin form ``abs(2p-1)``
    instead, because for a binary judgement "distance from a coin flip" is the
    quantity an operator means by confidence. The two are not interchangeable and
    should not be compared against the same threshold.
    """
    probs = [float(p) for p in probabilities.values() if p > 0.0]
    n = len(probs)
    if n == 0:
        # Nothing was asserted. Maximum uncertainty, not maximum certainty.
        return 0.0
    if n == 1:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probs)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(n)))


def level_index(level: str, ladder: Sequence[str]) -> int:
    """Position of ``level`` in an ordered ``ladder``; ``-1`` when unknown."""
    try:
        return ladder.index(level)
    except ValueError:
        return -1


def bump_within(level: str, ladder: Sequence[str], steps: int) -> str:
    """Move ``level`` ``steps`` positions up the ``ladder``, clamped to the ends.

    "Up" means stricter for sensitivity and stronger for complexity; both ladders
    are declared in ascending order, so a positive ``steps`` is always the
    conservative direction.
    """
    idx = level_index(level, ladder)
    if idx < 0:
        return level
    return ladder[max(0, min(len(ladder) - 1, idx + steps))]


# --------------------------------------------------------------------------- #
# Soft answers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChoiceAnswer:
    """One-of-N answer with the full distribution and a confidence.

    ``confidence`` is whatever the backend reported when it reports one (Jev
    does), otherwise :func:`certainty_from_probabilities` of ``probabilities``.
    Keeping both is deliberate: the reported confidence is the calibrated
    number, the computed one is reproducible offline, and
    ``jev_route.distill.evaluate`` compares them.
    """

    choice: str
    probabilities: Mapping[str, float]
    confidence: float
    #: True when the backend supplied ``confidence`` itself rather than us
    #: deriving it from the distribution. Affects how much to trust it.
    confidence_reported: bool = True

    @property
    def computed_confidence(self) -> float:
        return certainty_from_probabilities(self.probabilities)

    def probability_of(self, option: str) -> float:
        return float(self.probabilities.get(option, 0.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "choice": self.choice,
            "probabilities": dict(self.probabilities),
            "confidence": self.confidence,
            "confidence_reported": self.confidence_reported,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ChoiceAnswer:
        return cls(
            choice=str(data["choice"]),
            probabilities={str(k): float(v) for k, v in data["probabilities"].items()},
            confidence=float(data["confidence"]),
            confidence_reported=bool(data.get("confidence_reported", True)),
        )

    @classmethod
    def uniform(cls, ladder: Sequence[str], choice: str | None = None) -> ChoiceAnswer:
        """A maximum-uncertainty answer: the honest shape of "backend unavailable"."""
        p = 1.0 / len(ladder)
        picked = choice if choice in ladder else ladder[len(ladder) // 2]
        return cls(
            choice=str(picked),
            probabilities=dict.fromkeys(ladder, p),
            confidence=0.0,
            confidence_reported=False,
        )


@dataclass(frozen=True)
class NoulAnswer:
    """A yes/no answer as the probability of *yes*.

    There is no separate confidence for a noul: the probability *is* the
    calibration. ``0.5`` means genuinely torn, which is not "medium intensity"
    and must be treated as uncertainty by the policy engine.
    """

    value: float

    @property
    def probabilities(self) -> dict[str, float]:
        return {"yes": self.value, "no": 1.0 - self.value}

    @property
    def confidence(self) -> float:
        """Distance from a coin flip, in ``[0, 1]``."""
        return abs(2.0 * self.value - 1.0)

    def is_true(self, threshold: float = DEFAULT_NOUL_THRESHOLD) -> bool:
        return self.value >= threshold

    def to_dict(self) -> dict[str, Any]:
        return {"noul": self.value}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NoulAnswer:
        return cls(value=float(data["noul"]))

    @classmethod
    def unknown(cls) -> NoulAnswer:
        return cls(value=0.5)


@dataclass(frozen=True)
class DecisionAnswers:
    """The four typed answers that make up one routing decision.

    Same shape for every backend. A backend that cannot answer a question must
    still return one, at maximum uncertainty, rather than omitting it -- the
    policy engine relies on the schema being total.
    """

    complexity: ChoiceAnswer
    sensitivity: ChoiceAnswer
    pii: NoulAnswer
    domain: ChoiceAnswer

    def to_dict(self) -> dict[str, Any]:
        return {
            "complexity": self.complexity.to_dict(),
            "sensitivity": self.sensitivity.to_dict(),
            "pii": self.pii.to_dict(),
            "domain": self.domain.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DecisionAnswers:
        return cls(
            complexity=ChoiceAnswer.from_dict(data["complexity"]),
            sensitivity=ChoiceAnswer.from_dict(data["sensitivity"]),
            pii=NoulAnswer.from_dict(data["pii"]),
            domain=ChoiceAnswer.from_dict(data["domain"]),
        )

    @classmethod
    def unknown(cls) -> DecisionAnswers:
        """Every answer at maximum uncertainty: the fail-closed shape."""
        return cls(
            complexity=ChoiceAnswer.uniform(COMPLEXITY_LEVELS),
            sensitivity=ChoiceAnswer.uniform(SENSITIVITY_LEVELS),
            pii=NoulAnswer.unknown(),
            domain=ChoiceAnswer.uniform(DOMAINS),
        )


# --------------------------------------------------------------------------- #
# Local hard gate
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GateFinding:
    """One deterministic hit from the local PII/sensitivity gate.

    The matched text itself is *never* stored -- only a hash of it, so a logged
    record can prove a detector fired without retaining the secret that fired it.
    """

    detector: str
    category: Literal["pii", "secret", "regulated", "keyword"]
    #: Lowest sensitivity level this finding is allowed to resolve to.
    sensitivity_floor: str
    force_local: bool
    span_hash: str
    count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GateFinding:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(frozen=True)
class GateVerdict:
    """Result of running the local gate over a redacted excerpt.

    The gate is a *floor*, never a ceiling: it can only make a decision stricter.
    When :attr:`blocks_backend` is true the router must not send the excerpt to a
    cloud backend at all -- that is what makes "the hard gate runs before Jev"
    mean something for privacy rather than being decoration.
    """

    fired: bool = False
    findings: tuple[GateFinding, ...] = ()
    sensitivity_floor: str | None = None
    pii_floor: float | None = None
    force_local: bool = False
    blocks_backend: bool = False
    #: Topic-level keyword hits. These are *hints*, not floors: they are passed to
    #: the decision backend as part of its state and recorded for training, but
    #: they never force a tier on their own. Only :attr:`findings` whose detector
    #: is non-advisory contribute to the floors above.
    advisory_topics: tuple[str, ...] = ()

    def detectors(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.findings:
            out[f.detector] = out.get(f.detector, 0) + f.count
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "fired": self.fired,
            "findings": [f.to_dict() for f in self.findings],
            "sensitivity_floor": self.sensitivity_floor,
            "pii_floor": self.pii_floor,
            "force_local": self.force_local,
            "blocks_backend": self.blocks_backend,
            "advisory_topics": list(self.advisory_topics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GateVerdict:
        return cls(
            fired=bool(data.get("fired", False)),
            findings=tuple(GateFinding.from_dict(f) for f in data.get("findings", ())),
            sensitivity_floor=data.get("sensitivity_floor"),
            pii_floor=data.get("pii_floor"),
            force_local=bool(data.get("force_local", False)),
            blocks_backend=bool(data.get("blocks_backend", False)),
            advisory_topics=tuple(data.get("advisory_topics", ())),
        )

    @classmethod
    def clean(cls) -> GateVerdict:
        return cls()


@dataclass(frozen=True)
class GateBlockRecord:
    """What the gate refused, recorded so the rules can be improved.

    A gate that blocks a request and says nothing else is a gate an operator
    cannot tune: the only way to find out what it caught is to ask the user to
    paste the prompt again, which is exactly what the gate just decided must not
    happen. This record is the alternative. It answers "which detectors fire, on
    what shape of request, how often" -- enough to spot a detector that never
    fires, one that fires on everything, and a class of prompt the rule set does
    not cover yet.

    **This metadata feeds rule improvement, not model training.** The distinction
    is the whole reason the record exists in this shape. New regex and NER
    patterns are written by a person reading aggregate counts; they are not fitted
    to blocked text. Real blocked content must never become training data, for the
    gate or for anything else -- see the bootstrap paradox in
    :mod:`jev_route.gate_semantic`. So the payload is closed on purpose:
    ``request_id``, ``timestamp``, the detector ids that fired, the deterministic
    feature vector, and the excerpt hash. No excerpt text under any configuration,
    no raw secret, and no matched span -- the same identify-without-retaining rule
    :class:`GateFinding` applies to a span, applied here to the whole request.

    ``schema_version`` and ``kind`` are carried alongside because every line in a
    jev-route JSONL stream has to be identifiable and versionable; they are
    envelope, not payload.
    """

    request_id: str
    timestamp: str
    #: Every detector id that fired, hard and advisory, in gate order. Advisory
    #: ids are included because "which topic keywords are firing" is part of
    #: tuning a rule set; the counts live in ``features.gate_detectors``.
    detectors: tuple[str, ...]
    features: RequestFeatures
    excerpt_hash: str
    schema_version: str = SCHEMA_VERSION
    kind: str = GATE_BLOCK_RECORD_KIND

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "detectors": list(self.detectors),
            "excerpt_hash": self.excerpt_hash,
            "features": self.features.to_dict(),
        }

    def to_json(self) -> str:
        """One line of JSONL, canonical form: sorted keys, compact separators."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GateBlockRecord:
        return cls(
            request_id=str(data["request_id"]),
            timestamp=str(data.get("timestamp", "")),
            detectors=tuple(str(d) for d in (data.get("detectors") or ())),
            features=RequestFeatures.from_dict(data.get("features") or {}),
            excerpt_hash=str(data.get("excerpt_hash", "")),
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            kind=str(data.get("kind", GATE_BLOCK_RECORD_KIND)),
        )

    @classmethod
    def from_json(cls, line: str) -> GateBlockRecord:
        return cls.from_dict(json.loads(line))


# --------------------------------------------------------------------------- #
# Request features (the privacy-preserving training signal)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RequestFeatures:
    """Cheap deterministic features of the request, computed locally.

    These exist so a distilled model can be trained *without* retaining prompt
    text. They are also useful in their own right for debugging why a prompt
    routed the way it did. Field order is part of the dataset contract: append
    new features at the end, never reorder.
    """

    char_len: int = 0
    word_count: int = 0
    line_count: int = 0
    sentence_count: int = 0
    mean_word_len: float = 0.0
    digit_ratio: float = 0.0
    upper_ratio: float = 0.0
    punct_ratio: float = 0.0
    non_ascii_ratio: float = 0.0
    code_blocks: int = 0
    inline_code_spans: int = 0
    urls: int = 0
    question_marks: int = 0
    exclamations: int = 0
    has_stack_trace: bool = False
    has_diff: bool = False
    has_json: bool = False
    lang: str = "und"
    n_messages: int = 0
    n_prior_turns: int = 0
    tool_output_present: bool = False
    gate_detectors: Mapping[str, int] = field(default_factory=dict)
    n_gate_findings: int = 0
    gate_force_local: bool = False

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["gate_detectors"] = dict(self.gate_detectors)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RequestFeatures:
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {k: v for k, v in data.items() if k in known}
        kwargs["gate_detectors"] = dict(kwargs.get("gate_detectors") or {})
        return cls(**kwargs)


# --------------------------------------------------------------------------- #
# The routing decision
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RoutingDecision:
    """What the router decided, and why.

    ``tier`` is the policy-level answer (``local`` / ``cheap`` / ``strong``);
    ``model`` is the concrete deployment name handed to LiteLLM. They are kept
    apart because the tier is what the policy reasons about and the model is what
    the provider config owns.
    """

    tier: str
    model: str
    rule_id: str
    reason: str
    answers: DecisionAnswers
    gate: GateVerdict
    backend: str
    backend_model_version: str
    #: Effective sensitivity after applying the gate floor and confidence bumps.
    effective_sensitivity: str
    #: Effective complexity after confidence bumps.
    effective_complexity: str
    #: Set when a confidence floor forced a stricter answer than the argmax.
    escalated: tuple[str, ...] = ()
    #: Set when the backend failed and fail-closed kicked in.
    degraded: bool = False
    degrade_reason: str | None = None
    latency_ms: float = 0.0
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "model": self.model,
            "rule_id": self.rule_id,
            "reason": self.reason,
            "answers": self.answers.to_dict(),
            "gate": self.gate.to_dict(),
            "backend": self.backend,
            "backend_model_version": self.backend_model_version,
            "effective_sensitivity": self.effective_sensitivity,
            "effective_complexity": self.effective_complexity,
            "escalated": list(self.escalated),
            "degraded": self.degraded,
            "degrade_reason": self.degrade_reason,
            "latency_ms": self.latency_ms,
            "cached": self.cached,
        }


# --------------------------------------------------------------------------- #
# The persisted record -- this IS the training dataset
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DecisionRecord:
    """One routing decision, as persisted to the decision log.

    Privacy contract: by default a record contains the *hash* of the excerpt and
    the deterministic features, never the excerpt itself. Storing text is opt-in
    (``logging.excerpt_mode: redacted`` in policy config) and only ever stores the
    redacted form. See ``docs/privacy.md`` for the tradeoff, and
    ``jev_route.distill.export`` for what each mode can train.
    """

    request_id: str
    timestamp: str
    decision: RoutingDecision
    features: RequestFeatures
    excerpt_hash: str
    backend_latency_ms: float
    total_latency_ms: float
    #: Questions actually sent to the backend, verbatim. Empty when the gate
    #: blocked the call -- which is itself important training signal.
    questions_sent: Mapping[str, Any] = field(default_factory=dict)
    #: Redacted excerpt, present only when explicitly enabled.
    excerpt: str | None = None
    schema_version: str = SCHEMA_VERSION
    kind: str = RECORD_KIND
    #: Caller-supplied context (tenant, app, api-key alias). Never the prompt.
    metadata: Mapping[str, Any] = field(default_factory=dict)
    #: LiteLLM's originally requested model, when routed through a proxy.
    requested_model: str | None = None
    #: Shadow-mode disagreement, when a shadow backend ran alongside.
    shadow: Mapping[str, Any] | None = None
    #: The semantic gate layer's assessment, when layer 2 ran. Same idea as
    #: ``shadow`` and the same privacy rule: scores, levels, and detector ids,
    #: never text. ``None`` when ``gate.semantic.mode: off``.
    semantic: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "requested_model": self.requested_model,
            "excerpt_hash": self.excerpt_hash,
            "excerpt": self.excerpt,
            "features": self.features.to_dict(),
            "questions_sent": dict(self.questions_sent),
            "decision": self.decision.to_dict(),
            "backend_latency_ms": self.backend_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "metadata": dict(self.metadata),
            "shadow": dict(self.shadow) if self.shadow else None,
            "semantic": dict(self.semantic) if self.semantic else None,
        }

    def to_json(self) -> str:
        """One line of JSONL. ``sort_keys`` keeps diffs and hashes stable."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DecisionRecord:
        decision_raw = data["decision"]
        decision = RoutingDecision(
            tier=decision_raw["tier"],
            model=decision_raw["model"],
            rule_id=decision_raw["rule_id"],
            reason=decision_raw["reason"],
            answers=DecisionAnswers.from_dict(decision_raw["answers"]),
            gate=GateVerdict.from_dict(decision_raw.get("gate", {})),
            backend=decision_raw["backend"],
            backend_model_version=decision_raw.get("backend_model_version", ""),
            effective_sensitivity=decision_raw["effective_sensitivity"],
            effective_complexity=decision_raw["effective_complexity"],
            escalated=tuple(decision_raw.get("escalated", ())),
            degraded=bool(decision_raw.get("degraded", False)),
            degrade_reason=decision_raw.get("degrade_reason"),
            latency_ms=float(decision_raw.get("latency_ms", 0.0)),
            cached=bool(decision_raw.get("cached", False)),
        )
        return cls(
            request_id=str(data["request_id"]),
            timestamp=str(data["timestamp"]),
            decision=decision,
            features=RequestFeatures.from_dict(data.get("features", {})),
            excerpt_hash=str(data.get("excerpt_hash", "")),
            backend_latency_ms=float(data.get("backend_latency_ms", 0.0)),
            total_latency_ms=float(data.get("total_latency_ms", 0.0)),
            questions_sent=dict(data.get("questions_sent") or {}),
            excerpt=data.get("excerpt"),
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            kind=str(data.get("kind", RECORD_KIND)),
            metadata=dict(data.get("metadata") or {}),
            requested_model=data.get("requested_model"),
            shadow=dict(data["shadow"]) if data.get("shadow") else None,
            semantic=dict(data["semantic"]) if data.get("semantic") else None,
        )

    @classmethod
    def from_json(cls, line: str) -> DecisionRecord:
        return cls.from_dict(json.loads(line))


__all__ = [
    "COMPLEXITY_LEVELS",
    "DEFAULT_NOUL_THRESHOLD",
    "DOMAINS",
    "GATE_BLOCK_RECORD_KIND",
    "RECORD_KIND",
    "SCHEMA_VERSION",
    "SENSITIVITY_LEVELS",
    "TIERS",
    "BackendName",
    "ChoiceAnswer",
    "DecisionAnswers",
    "DecisionRecord",
    "GateBlockRecord",
    "GateFinding",
    "GateVerdict",
    "NoulAnswer",
    "RequestFeatures",
    "RoutingDecision",
    "bump_within",
    "certainty_from_probabilities",
    "level_index",
]
