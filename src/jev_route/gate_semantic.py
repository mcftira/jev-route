"""Layer 2 of the gate: a local, distilled semantic sensitivity model.

The hard gate (:mod:`jev_route.gate`) is a checksum and a regex. It is
permanent, auditable, and it is the compliance floor -- but it can only see
structure. An HR disciplinary narrative, a clinical note written as prose, a
legal-privilege discussion between two named parties: none of those contains a
single detectable identifier, and all of them are exactly the traffic that should
not leave the building. This module is the layer that reads for *context*.

The two layers, and the order they run in::

    (1) deterministic floor        regex + checksum. never removed, never a model.
            | passed?
            v
    (2) semantic layer (this one)  local model. shadow by default, enforce only
            |                      once measured criteria are met.
            v
    (3) router                     JevBackend now, DistilledBackend after graduation.

Four rules govern this module, and each one is enforced in code rather than in a
review checklist:

**Layer 2 may only make routing stricter.** It can raise the sensitivity floor
and force the local tier. It cannot lower either. A semantic layer that answers
"public" for a prompt containing a Luhn-valid card number produces a floor of
``None``, and :func:`resolve_floor` takes the maximum of the two layers, so the
deterministic "regulated" survives. That is not a convention the router happens
to follow; it is the only arithmetic available.

**Shadow is the default, and shadow never blocks.** In shadow mode the layer
runs, scores, and records what it would have done -- and the assessment is
deliberately invisible to the policy engine (see :meth:`SemanticLayer.assess`).
A shadow that could change routing would stop being a measurement.

**Promotion is measured, not configured.** ``mode: enforce`` is necessary and
not sufficient. Enforce also requires an artifact whose *held-out* metrics meet
:class:`EnforceCriteria`; :func:`check_promotion` compares them and
:class:`SemanticLayer.from_config` refuses to build the layer -- with the
measured numbers in the exception -- when they are not met. It does not quietly
fall back to shadow, because a gate that looks enforced and is not is worse than
one that was never turned on.

**The bootstrap paradox.** The gate cannot ask the cloud whether something is
safe to send to the cloud -- the model call *is* the thing being guarded. So the
deterministic floor never consults a model, and this layer is trained only on
synthetic/public positives plus real production **negatives**: prompts that were
allowed through. Real blocked content is never a training example for the gate
that blocks it. That rule is checked at load time: an artifact whose provenance
says it was trained on blocked content is refused outright
(:meth:`SemanticArtifact.load`), and one that does not declare its provenance is
refused too.

Everything here is stdlib-only on purpose. The router runs inline on every
request, in containers that have no numpy, no torch and no scikit-learn, and a
gate that only works when the training extra is installed is a gate that is off
in production. So the scorer is a sparse logistic regression over a stored
vocabulary -- small enough to evaluate in pure Python, and readable with
``json.load`` so a security reviewer can see every token the model keys on
without running anything. Heavier students can be substituted by implementing
:class:`SemanticScorer`; the artifact format, the promotion criteria and the
router integration do not change. The v0.3 Laya sensitivity head is exactly
such a substitute: its artifact payload carries ``scorer.kind: "laya"`` plus
the checkpoint directory (see :mod:`jev_route.backends.laya_scorer`), and the
loader builds the matching scorer -- the layer above it does not know, or
care, which one answered.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from .schema import SENSITIVITY_LEVELS, GateVerdict, RequestFeatures, level_index

if TYPE_CHECKING:
    # Import-for-typing only. The two policy/gate modules are stdlib-light, but
    # the gate must not depend on the policy engine at runtime and the policy
    # engine already depends on this module, so the edge is drawn here and
    # nowhere else. ``LayaScorer`` joins the union for the same reason: its
    # module is stdlib-only at import time, but it lives under ``backends``,
    # whose package init already imports the policy engine -- a module-level
    # edge from the gate there would be a cycle.
    from .backends.laya_scorer import LayaScorer
    from .gate import HardGate
    from .policy import Policy

#: Discriminator for the on-disk artifact, so a directory of JSON is identifiable
#: and a distilled-model artifact is not mistaken for a gate artifact.
SEMANTIC_ARTIFACT_KIND = "jev_route.gate.semantic"
SEMANTIC_ARTIFACT_VERSION = "1"
SUPPORTED_SEMANTIC_ARTIFACT_VERSIONS: frozenset[str] = frozenset({"1"})
SEMANTIC_ARTIFACT_FILE = "semantic.json"

SEMANTIC_MODES: tuple[str, ...] = ("off", "shadow", "enforce")
#: Shadow, not off: the layer should be measuring from the moment it is
#: configured, and an absent artifact makes shadow inert anyway.
DEFAULT_SEMANTIC_MODE = "shadow"
DEFAULT_SEMANTIC_THRESHOLD = 0.5
DEFAULT_SEMANTIC_LEVEL = "confidential"

#: Scorer kinds the artifact format can carry (``scorer.kind``). ``"lexicon"`` is
#: the default and may be omitted from the payload; ``"laya"`` carries a
#: fine-tuned Laya head as a local checkpoint directory
#: (``scorer.checkpoint_dir``). An unknown kind is refused at load time: a kind
#: this build cannot read must fail the deploy, not degrade to the wrong model.
SCORER_KINDS: tuple[str, ...] = ("lexicon", "laya")

#: The lowest sensitivity level layer 2 is allowed to assert. ``public`` and
#: ``internal`` are excluded because a semantic finding is only ever a claim that
#: something is *more* sensitive than it looks; a layer that asserted ``public``
#: would be asking to loosen, which :func:`resolve_floor` would ignore anyway.
#: Refusing it at load time keeps the config honest instead of inert.
ASSERTABLE_LEVELS: tuple[str, ...] = ("confidential", "regulated")

#: What an artifact's positive examples may have come from. "production" is
#: absent on purpose: production positives ARE blocked content, and the gate does
#: not train on those. See the bootstrap paradox in the module docstring.
ALLOWED_POSITIVE_SOURCES: frozenset[str] = frozenset({"synthetic", "public", "synthetic+public"})
#: Negatives are the opposite case. Real traffic that the gate let through is the
#: most valuable negative set there is, and using it costs nothing: by definition
#: it was already allowed to leave the building.
ALLOWED_NEGATIVE_SOURCES: frozenset[str] = frozenset(
    {"production", "public", "synthetic", "production+public", "production+synthetic"}
)


class SemanticGateError(RuntimeError):
    """Layer 2 cannot be honoured as configured. Raised at construction, not at request time."""


class SemanticConfigError(SemanticGateError, ValueError):
    """A malformed ``gate.semantic`` / ``gate.blocked_metadata`` section.

    Subclasses :class:`ValueError` so that a policy loader may re-raise it as a
    :class:`~jev_route.policy.PolicyError` without changing the message, and so
    that callers already catching ``ValueError`` for config problems keep working.
    """


class SemanticArtifactError(SemanticGateError):
    """An artifact that is missing, corrupt, or refuses to state its provenance."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EnforceCriteria:
    """The measured gates between shadow and enforce.

    These numbers are the whole reason ``mode: enforce`` is not sufficient on its
    own. Each one is a different way a semantic layer can be wrong, and each has
    a different cost:

    ``min_recall``
        Held-out positives the layer must catch. **This is the one that matters.**
        A miss here is content that left the building. The default is 0.99 and
        :data:`MIN_ALLOWED_RECALL` is the floor below which the config is
        rejected outright -- you may raise the bar, you may not remove it.
    ``max_false_positive_rate``
        Real negatives the layer would have air-gapped. A cost and a credibility
        problem, not a leak: the failure mode is that operators turn the layer
        off, which is why the bound is tight enough to matter and not so tight
        that nobody can ship.
    ``max_disagreement_rate``
        How often the two layers contradict each other on **live traffic**, read
        from the shadow observation (:func:`measure_shadow_log`). High
        disagreement does not mean layer 2 is wrong -- catching what layer 1
        misses is the job -- but it does mean the pair is not yet a coherent
        story, and promoting on top of it makes every future incident ambiguous.
        An offline eval set can stand in for it, but only weakly: there the number
        is dominated by how many positives the set contains, which is a property
        of the set and not of the traffic.
    ``max_semantic_miss_rate``
        Of the examples the deterministic floor *also* catches, how many layer 2
        misses. A model that does not recognise a card number as sensitive is not
        a model you want judging the cases regex cannot see.
    ``min_positive_examples`` / ``min_negative_examples``
        Sample sizes. At 0.99 recall, twenty positives proves nothing: one miss
        is 0.95 and zero misses is a rounding error. The minimums are what stop a
        tiny eval set from manufacturing a promotion.
    ``min_shadow_examples``
        Live requests observed in shadow before promotion is even considered. This
        is the criterion no offline number can substitute for: a model that scores
        0.99 on a held-out set of sentences you wrote has still never seen your
        traffic, and the disagreement rate you are about to bound is a property of
        that traffic.
    """

    min_recall: float = 0.99
    max_false_positive_rate: float = 0.02
    max_disagreement_rate: float = 0.05
    max_semantic_miss_rate: float = 0.02
    min_positive_examples: int = 200
    min_negative_examples: int = 500
    min_shadow_examples: int = 1000

    #: A configured ``min_recall`` below this is rejected at load time rather than
    #: honoured. 0.9 is already generous for a component whose false negative is a
    #: data-egress incident; the knob exists so an operator can demand 0.995, not
    #: so they can accept 0.6.
    MIN_ALLOWED_RECALL: ClassVar[float] = 0.90

    @classmethod
    def parse(cls, raw: Any, *, where: str = "gate.semantic.enforce_requires") -> EnforceCriteria:
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise SemanticConfigError(f"{where} must be a mapping, got {type(raw).__name__}")
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise SemanticConfigError(
                f"{where} has unknown keys {unknown}. Known: {sorted(known)}. "
                "A typo here would silently relax the promotion criteria, so it fails the deploy."
            )
        criteria = cls(**{k: raw[k] for k in raw if k in known})
        criteria._validate(where=where)
        return criteria

    def _validate(self, *, where: str) -> None:
        for name in ("min_recall", "max_false_positive_rate", "max_disagreement_rate", "max_semantic_miss_rate"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise SemanticConfigError(f"{where}.{name} must be a number, got {value!r}")
            if not 0.0 <= float(value) <= 1.0:
                raise SemanticConfigError(f"{where}.{name} must be within [0, 1], got {value!r}")
        for name in ("min_positive_examples", "min_negative_examples", "min_shadow_examples"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise SemanticConfigError(f"{where}.{name} must be a non-negative integer, got {value!r}")
        if self.min_recall < self.MIN_ALLOWED_RECALL:
            raise SemanticConfigError(
                f"{where}.min_recall={self.min_recall} is below the floor of {self.MIN_ALLOWED_RECALL}. "
                "The semantic layer guards a data-egress boundary: the recall requirement is "
                "configurable upward, not downward."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_recall": self.min_recall,
            "max_false_positive_rate": self.max_false_positive_rate,
            "max_disagreement_rate": self.max_disagreement_rate,
            "max_semantic_miss_rate": self.max_semantic_miss_rate,
            "min_positive_examples": self.min_positive_examples,
            "min_negative_examples": self.min_negative_examples,
            "min_shadow_examples": self.min_shadow_examples,
        }


@dataclass(frozen=True)
class SemanticGatePolicy:
    """The ``gate.semantic`` section, parsed and validated at policy load time.

    Validated strictly, unlike ``on_uncertain`` (which drops keys it does not
    know so a newer policy still loads on an older build). The difference is the
    direction a mistake fails in: an ignored ``on_uncertain`` knob degrades toward
    stricter, while an ignored ``mode: enfoce`` typo leaves a gate in shadow that
    an operator believes is enforcing. Every key here is either honoured or fatal.
    """

    mode: str = DEFAULT_SEMANTIC_MODE
    #: Path to a semantic artifact (a JSON file, or a directory holding
    #: ``semantic.json``). ``None`` means "no model has been trained yet", which
    #: in shadow mode makes the layer inert rather than broken.
    artifact: str | None = None
    #: Score at or above which layer 2 asserts sensitivity.
    threshold: float = DEFAULT_SEMANTIC_THRESHOLD
    #: The floor layer 2 asserts when it fires. Only used in enforce mode.
    level: str = DEFAULT_SEMANTIC_LEVEL
    #: JSON file of metrics measured from the live shadow log
    #: (:func:`measure_shadow_log`). Merged over the artifact's offline metrics
    #: before promotion is checked, because the disagreement rate is a property of
    #: production traffic and cannot be measured on a hand-written eval set.
    shadow_metrics: str | None = None
    enforce_requires: EnforceCriteria = field(default_factory=EnforceCriteria)

    @classmethod
    def parse(cls, raw: Any, *, where: str = "gate.semantic") -> SemanticGatePolicy:
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise SemanticConfigError(f"{where} must be a mapping, got {type(raw).__name__}")
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise SemanticConfigError(
                f"{where} has unknown keys {unknown}. Known: {sorted(known)}. "
                "Failing the deploy is the point: a silently ignored key in this "
                "section is a gate that is not doing what its config says."
            )
        shadow_metrics = raw.get("shadow_metrics")
        if shadow_metrics is not None and not isinstance(shadow_metrics, (str, Path)):
            raise SemanticConfigError(
                f"{where}.shadow_metrics must be a path string, got {type(shadow_metrics).__name__}"
            )
        mode = str(raw.get("mode", DEFAULT_SEMANTIC_MODE)).strip().lower()
        if mode not in SEMANTIC_MODES:
            raise SemanticConfigError(f"{where}.mode must be one of {list(SEMANTIC_MODES)}, got {mode!r}")

        artifact = raw.get("artifact")
        if artifact is not None and not isinstance(artifact, (str, Path)):
            raise SemanticConfigError(f"{where}.artifact must be a path string, got {type(artifact).__name__}")
        if artifact is not None and not str(artifact).strip():
            raise SemanticConfigError(f"{where}.artifact must be a non-empty path (or omitted)")

        threshold = raw.get("threshold", DEFAULT_SEMANTIC_THRESHOLD)
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise SemanticConfigError(f"{where}.threshold must be a number, got {threshold!r}")
        if not 0.0 < float(threshold) < 1.0:
            # 0.0 would fire on everything and 1.0 would fire on nothing; both are
            # config mistakes rather than risk postures, and both are silent.
            raise SemanticConfigError(f"{where}.threshold must be within (0, 1) exclusive, got {threshold!r}")

        level = str(raw.get("level", DEFAULT_SEMANTIC_LEVEL)).strip().lower()
        if level not in ASSERTABLE_LEVELS:
            raise SemanticConfigError(
                f"{where}.level must be one of {list(ASSERTABLE_LEVELS)}, got {level!r}. "
                "Layer 2 can only make routing stricter, so it has no business asserting a low level."
            )
        if level not in SENSITIVITY_LEVELS:  # pragma: no cover - ASSERTABLE_LEVELS is a subset
            raise SemanticConfigError(f"{where}.level {level!r} is not on the sensitivity ladder")

        return cls(
            mode=mode,
            artifact=str(artifact) if artifact is not None else None,
            threshold=float(threshold),
            level=level,
            shadow_metrics=str(shadow_metrics) if shadow_metrics is not None else None,
            enforce_requires=EnforceCriteria.parse(raw.get("enforce_requires"), where=f"{where}.enforce_requires"),
        )

    @property
    def runs(self) -> bool:
        """Whether the layer scores traffic at all."""
        return self.mode != "off"

    @property
    def enforces(self) -> bool:
        return self.mode == "enforce"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "artifact": self.artifact,
            "threshold": self.threshold,
            "level": self.level,
            "shadow_metrics": self.shadow_metrics,
            "enforce_requires": self.enforce_requires.to_dict(),
        }


@dataclass(frozen=True)
class BlockedMetadataPolicy:
    """The ``gate.blocked_metadata`` section: what to record about a refusal.

    Exists because "the gate blocked it" is not an operable answer. To improve a
    detector set you need to know *which* detectors fired, on what *shape* of
    request, and how often -- and you need that without keeping the text that
    triggered it. See :class:`~jev_route.schema.GateBlockRecord`.
    """

    enabled: bool = True
    #: Destination for the refusal stream. ``None`` means "next to the decision
    #: log", resolved by :func:`~jev_route.logging_sink.build_blocked_sink`. It is
    #: deliberately a *separate file* from the decision log: the refusal stream
    #: feeds rule improvement and the decision log feeds model training, and
    #: keeping them physically apart means no export path can pick the wrong one
    #: up by accident.
    path: str | None = None

    @classmethod
    def parse(cls, raw: Any, *, where: str = "gate.blocked_metadata") -> BlockedMetadataPolicy:
        if raw is None:
            return cls()
        if isinstance(raw, bool):
            return cls(enabled=raw)
        if not isinstance(raw, Mapping):
            raise SemanticConfigError(f"{where} must be a mapping or a boolean, got {type(raw).__name__}")
        unknown = sorted(set(raw) - {"enabled", "path"})
        if unknown:
            raise SemanticConfigError(f"{where} has unknown keys {unknown}. Known: ['enabled', 'path'].")
        path = raw.get("path")
        if path is not None and not isinstance(path, (str, Path)):
            raise SemanticConfigError(f"{where}.path must be a path string, got {type(path).__name__}")
        if path is not None and not str(path).strip():
            raise SemanticConfigError(f"{where}.path must be a non-empty path (or omitted)")
        return cls(enabled=bool(raw.get("enabled", True)), path=str(path) if path is not None else None)

    def to_dict(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "path": self.path}


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str, *, lower: bool = True, min_token_chars: int = 2, ngram_max: int = 2) -> list[str]:
    """Word unigrams plus bigrams.

    Duplicated from ``distill.artifact.tokenize`` rather than imported, and the
    reason is a dependency direction, not laziness: importing it would execute
    ``jev_route.distill.__init__``, and layer 2 has to load in a container where
    the distill extra is not installed. Ten lines of tokenizing is a cheap price
    for a gate that has no optional dependencies. The parameters are stored in the
    artifact so a scorer and its tokenizer can never drift apart.
    """
    body = text.lower() if lower else text
    words = [t for t in _TOKEN_RE.findall(body) if len(t) >= min_token_chars or t.isdigit()]
    if ngram_max < 2:
        return words
    return [*words, *(a + "_" + b for a, b in itertools.pairwise(words))]


def sigmoid(z: float) -> float:
    """Logistic function, written via ``tanh`` so large ``|z|`` cannot overflow."""
    return 0.5 * (1.0 + math.tanh(z / 2.0))


@runtime_checkable
class SemanticScorer(Protocol):
    """A local model that answers one question: how sensitive is this text?

    Implementations must be local and synchronous. "Local" is load-bearing: this
    layer runs on text the deterministic gate may already have refused to send
    anywhere, so a scorer that reached the network would turn the guard into the
    leak. Return a probability in ``[0, 1]``; raise nothing -- an exception is
    caught by :class:`SemanticLayer` and recorded as a degraded assessment,
    because a broken layer 2 must cost telemetry, not requests.
    """

    #: Stable identifier for logs and the artifact envelope.
    name: str
    #: Version of the trained model, recorded on every assessment.
    model_version: str

    def score(self, text: str, *, features: RequestFeatures | None = None) -> float:
        """Probability that ``text`` is contextually sensitive."""
        ...


class LexiconScorer:
    """Sparse logistic regression over a stored vocabulary. Pure Python.

    Chosen over a neural student for one reason: it evaluates in the interpreter
    that is already running the proxy, with no numpy, so layer 2 costs an import
    nothing and a score a dict walk. The ceiling is lower than a transformer's and
    that trade is stated rather than hidden -- this layer's job is to catch the
    *obvious-in-context* cases regex cannot see ("disciplinary hearing", "the
    patient reports", "as discussed under NDA"), not to out-read a person.

    The vocabulary lives in the artifact as readable JSON. A reviewer can open the
    file and see every token the model weights, which is more than can be said for
    an embedding matrix, and it is the reason this format has no ``.npz`` in it.
    """

    name = "lexicon-logreg"

    def __init__(
        self,
        weights: Mapping[str, float],
        *,
        bias: float = 0.0,
        model_version: str = "semantic-unknown",
        lower: bool = True,
        min_token_chars: int = 2,
        ngram_max: int = 2,
    ) -> None:
        self.weights: dict[str, float] = {str(k): float(v) for k, v in weights.items()}
        self.bias = float(bias)
        self.model_version = model_version
        self.lower = bool(lower)
        self.min_token_chars = int(min_token_chars)
        self.ngram_max = int(ngram_max)
        if not self.weights:
            raise SemanticArtifactError(f"{self.name}: artifact carries an empty weight vector")

    @property
    def n_features(self) -> int:
        return len(self.weights)

    def tokens(self, text: str) -> list[str]:
        return tokenize(text, lower=self.lower, min_token_chars=self.min_token_chars, ngram_max=self.ngram_max)

    def score(self, text: str, *, features: RequestFeatures | None = None) -> float:
        """``sigmoid(bias + sum(weights[token]))`` over the tokens present in ``text``.

        ``features`` is accepted and ignored. The signature carries it because a
        heavier scorer may want the deterministic feature vector, and a Protocol
        whose implementations disagree on their arguments is not a Protocol.
        """
        del features
        if not text:
            return sigmoid(self.bias)
        weights = self.weights
        z = self.bias
        for token in set(self.tokens(text)):
            weight = weights.get(token)
            if weight is not None:
                z += weight
        return sigmoid(z)


class NullScorer:
    """A scorer that never fires. Stands in for "no artifact has been trained yet".

    Distinguished from ``mode: off`` on purpose: ``off`` is an operator's decision
    not to run layer 2, while this is the honest state of a deployment that has
    not trained a model yet. It reports ``available = False`` through the layer so
    the assessment says *why* nothing happened instead of reporting a confident 0.0
    that nobody measured.
    """

    name = "null"
    model_version = "none"

    def __init__(self, reason: str = "no semantic artifact configured") -> None:
        self.reason = reason

    def score(self, text: str, *, features: RequestFeatures | None = None) -> float:
        del text, features
        return 0.0


# --------------------------------------------------------------------------- #
# The artifact
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Provenance:
    """Where an artifact's training examples came from. Checked on load.

    This is the bootstrap paradox made auditable. A reviewer should be able to
    answer "was the gate trained on blocked content?" by reading one JSON object,
    without reading the training code and without trusting anybody's memory.
    """

    positives: str
    negatives: str
    #: Explicit, and explicitly required. An artifact that omits it is refused:
    #: "we did not say we did" is not provenance.
    trained_on_blocked_content: bool
    #: Free-form note (generator version, dataset id, who labelled it).
    note: str = ""

    def validate(self, *, where: str) -> None:
        if self.trained_on_blocked_content:
            raise SemanticArtifactError(
                f"{where}: artifact declares trained_on_blocked_content=true. Refusing to load. "
                "The gate cannot be trained on the content it blocks: those examples only exist "
                "because a request was refused egress, and using them makes the refusal the "
                "training set for the next one. Positives must be synthetic or public."
            )
        if self.positives not in ALLOWED_POSITIVE_SOURCES:
            raise SemanticArtifactError(
                f"{where}: provenance.positives={self.positives!r} is not allowed. "
                f"Layer 2 trains on {sorted(ALLOWED_POSITIVE_SOURCES)} positives only; production "
                "positives are blocked content by definition."
            )
        if self.negatives not in ALLOWED_NEGATIVE_SOURCES:
            raise SemanticArtifactError(
                f"{where}: provenance.negatives={self.negatives!r} is not allowed "
                f"(expected one of {sorted(ALLOWED_NEGATIVE_SOURCES)})."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "positives": self.positives,
            "negatives": self.negatives,
            "trained_on_blocked_content": self.trained_on_blocked_content,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, raw: Any, *, where: str) -> Provenance:
        if not isinstance(raw, Mapping):
            raise SemanticArtifactError(
                f"{where}: the artifact must declare a `provenance` object naming where its "
                "positive and negative examples came from. An artifact that will not say is an "
                "artifact that cannot be audited."
            )
        if "trained_on_blocked_content" not in raw:
            raise SemanticArtifactError(
                f"{where}: provenance.trained_on_blocked_content is required and must be false."
            )
        return cls(
            positives=str(raw.get("positives", "")),
            negatives=str(raw.get("negatives", "")),
            trained_on_blocked_content=bool(raw["trained_on_blocked_content"]),
            note=str(raw.get("note", "")),
        )


def _scorer_from_payload(
    scorer_cfg: Mapping[str, Any], *, model_version: str, where: str
) -> LexiconScorer | LayaScorer:
    """Build the scorer the artifact's ``scorer`` payload describes, validating it.

    This is where the ``kind`` discriminator is read. ``"lexicon"`` is the
    default and may be omitted: it builds the vocabulary scorer from the inline
    weights. ``"laya"`` builds the neural scorer from the checkpoint directory
    pointer -- the weights live in the checkpoint, the artifact carries the
    address and the exact question. Every failure is a
    :class:`SemanticArtifactError` with the reason: the artifact is the only
    source of truth for which model is running.
    """
    kind = str(scorer_cfg.get("kind", "lexicon")).strip().lower()
    if kind not in SCORER_KINDS:
        raise SemanticArtifactError(
            f"{where}: scorer.kind={kind!r} is not supported by this build (supports "
            f"{list(SCORER_KINDS)}). An unknown kind must fail the load, not degrade to the "
            "wrong model."
        )
    if kind == "laya":
        from .backends.laya_scorer import LayaScorer  # lazy: torch/laya stay out of the gate import

        checkpoint_dir = scorer_cfg.get("checkpoint_dir")
        if not isinstance(checkpoint_dir, (str, Path)) or not str(checkpoint_dir).strip():
            raise SemanticArtifactError(
                f"{where}: scorer.checkpoint_dir is required for scorer.kind='laya' "
                "(the trained Laya head's checkpoint directory)."
            )
        question = scorer_cfg.get("question")
        if question is not None and not isinstance(question, Mapping):
            raise SemanticArtifactError(f"{where}: scorer.question must be an object when present")
        return LayaScorer(
            str(checkpoint_dir),
            model_version=model_version,
            device=str(scorer_cfg.get("device", "auto")),
            question=dict(question) if question is not None else None,
        )
    if str(scorer_cfg.get("name", "")) != LexiconScorer.name:
        raise SemanticArtifactError(
            f"{where}: scorer.name={scorer_cfg.get('name')!r} is not supported by this build "
            f"(expects {LexiconScorer.name!r})."
        )
    tokenizer = scorer_cfg.get("tokenizer") or {}
    if not isinstance(tokenizer, Mapping):
        raise SemanticArtifactError(f"{where}: scorer.tokenizer must be an object")
    weights = scorer_cfg.get("weights")
    if not isinstance(weights, Mapping):
        raise SemanticArtifactError(f"{where}: scorer.weights must be an object of token -> weight")
    scorer = LexiconScorer(
        weights,
        bias=float(scorer_cfg.get("bias", 0.0)),
        model_version=model_version,
        lower=bool(tokenizer.get("lower", True)),
        min_token_chars=int(tokenizer.get("min_token_chars", 2)),
        ngram_max=int(tokenizer.get("ngram_max", 2)),
    )
    for value in scorer.weights.values():
        if not math.isfinite(value):
            raise SemanticArtifactError(f"{where}: scorer.weights contains a non-finite value")
    if not math.isfinite(scorer.bias):
        raise SemanticArtifactError(f"{where}: scorer.bias is not finite")
    return scorer


@dataclass(frozen=True)
class SemanticArtifact:
    """A loaded semantic-layer artifact: scorer, threshold, metrics, provenance."""

    scorer: LexiconScorer | LayaScorer
    metadata: dict[str, Any]
    metrics: dict[str, Any]
    provenance: Provenance
    source: str | None = None

    @property
    def model_version(self) -> str:
        return self.scorer.model_version

    @property
    def threshold(self) -> float:
        """The threshold the artifact was measured at.

        Read from the artifact rather than from config when both exist, because
        the promotion metrics were computed *at this threshold*. Letting config
        move the threshold without re-measuring would promote a model on numbers
        that no longer describe it.
        """
        return float(self.metadata.get("threshold", DEFAULT_SEMANTIC_THRESHOLD))

    @property
    def level(self) -> str:
        return str(self.metadata.get("level", DEFAULT_SEMANTIC_LEVEL))

    @property
    def criteria(self) -> dict[str, Any]:
        return dict(self.metadata.get("criteria") or {})

    @property
    def created_at(self) -> str:
        return str(self.metadata.get("created_at", ""))

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.metadata)
        payload["kind"] = SEMANTIC_ARTIFACT_KIND
        payload["artifact_version"] = SEMANTIC_ARTIFACT_VERSION
        payload.setdefault("model_version", self.model_version)
        payload.setdefault("threshold", self.threshold)
        payload.setdefault("level", self.level)
        payload.setdefault("provenance", self.provenance.to_dict())
        if getattr(self.scorer, "kind", "lexicon") == "laya":
            # The Laya head cannot be inlined: its weights are a checkpoint
            # directory, not a JSON table. The artifact carries the pointer and
            # the exact question, and the checkpoint must live next to whatever
            # deployment loads the artifact (layer 2 is local by contract).
            payload["scorer"] = {
                "kind": "laya",
                "name": self.scorer.name,
                "checkpoint_dir": self.scorer.checkpoint_dir,
                "device": self.scorer.device,
                "question": dict(self.scorer.question),
            }
        else:
            payload["scorer"] = {
                "name": self.scorer.name,
                "bias": self.scorer.bias,
                "tokenizer": {
                    "lower": self.scorer.lower,
                    "min_token_chars": self.scorer.min_token_chars,
                    "ngram_max": self.scorer.ngram_max,
                },
                "weights": dict(sorted(self.scorer.weights.items())),
            }
        payload["metrics"] = dict(self.metrics)
        return payload

    def save(self, path: str | Path) -> Path:
        """Write the artifact. A directory gets ``semantic.json``; a file is written directly."""
        target = Path(path)
        if target.suffix == "":
            target.mkdir(parents=True, exist_ok=True)
            target = target / SEMANTIC_ARTIFACT_FILE
        elif target.parent != Path(""):
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> SemanticArtifact:
        """Load and validate. Every failure raises :class:`SemanticArtifactError` with a reason."""
        p = Path(path)
        candidate = p / SEMANTIC_ARTIFACT_FILE if p.is_dir() else p
        if not candidate.exists():
            raise SemanticArtifactError(
                f"no semantic gate artifact at {p}. Expected {SEMANTIC_ARTIFACT_FILE} in a directory, "
                "or a JSON file. Train one with jev_route.gate_semantic.fit_scorer(...) from synthetic "
                "positives and production negatives; until then leave gate.semantic.mode at shadow "
                "with no artifact and layer 2 stays inert."
            )
        try:
            raw = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SemanticArtifactError(f"{candidate} could not be read as JSON: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise SemanticArtifactError(f"{candidate} must contain a JSON object")
        return cls.from_dict(raw, source=str(candidate))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, source: str | None = None) -> SemanticArtifact:
        where = source or "semantic artifact"
        kind = str(raw.get("kind", ""))
        if kind != SEMANTIC_ARTIFACT_KIND:
            raise SemanticArtifactError(
                f"{where} has kind {kind or '(missing)'!r}, expected {SEMANTIC_ARTIFACT_KIND!r}. "
                "This is not a gate artifact -- a distilled-model artifact from `jev-route train` "
                "is a different file with a different job."
            )
        version = str(raw.get("artifact_version", ""))
        if version not in SUPPORTED_SEMANTIC_ARTIFACT_VERSIONS:
            raise SemanticArtifactError(
                f"{where} has artifact_version {version or '(missing)'}, this build supports "
                f"{sorted(SUPPORTED_SEMANTIC_ARTIFACT_VERSIONS)}."
            )
        scorer_cfg = raw.get("scorer")
        if not isinstance(scorer_cfg, Mapping):
            raise SemanticArtifactError(f"{where} has no `scorer` object")
        scorer = _scorer_from_payload(
            scorer_cfg, model_version=str(raw.get("model_version", "semantic-unknown")), where=where
        )
        provenance = Provenance.from_dict(raw.get("provenance"), where=where)
        provenance.validate(where=where)

        metrics = raw.get("metrics")
        if not isinstance(metrics, Mapping):
            raise SemanticArtifactError(
                f"{where} carries no `metrics` object. Promotion to enforce is decided on measured "
                "numbers, so an artifact without them can never be promoted -- retrain it with a "
                "held-out split."
            )

        return cls(
            scorer=scorer,
            metadata={k: v for k, v in raw.items() if k not in {"scorer", "metrics", "provenance"}},
            metrics=dict(metrics),
            provenance=provenance,
            source=source,
        )


# --------------------------------------------------------------------------- #
# Merging the two layers
# --------------------------------------------------------------------------- #
def resolve_floor(*floors: str | None) -> str | None:
    """The strictest of the layers' floors, or ``None`` when none asserted one.

    This function is the invariant "layer 2 may make routing stricter, never
    looser" expressed as arithmetic: it is a maximum, so there is no argument it
    can be given that makes a deterministic ``regulated`` come out as anything
    else. A semantic layer that answers ``public`` contributes ``None`` and
    therefore contributes nothing.
    """
    best: str | None = None
    best_index = -1
    for floor in floors:
        if not floor:
            continue
        index = level_index(str(floor), SENSITIVITY_LEVELS)
        if index > best_index:
            best_index, best = index, str(floor)
    return best


#: The two ways the layers can contradict each other. Named rather than boolean
#: because they mean opposite things -- one is the layer earning its keep, the
#: other is a reason not to promote it.
DISAGREE_SEMANTIC_ONLY = "semantic_only"  # layer 2 fired, layer 1 saw nothing
DISAGREE_SEMANTIC_MISS = "semantic_miss"  # layer 1 fired hard, layer 2 saw nothing
DISAGREEMENT_KINDS: tuple[str, ...] = (DISAGREE_SEMANTIC_ONLY, DISAGREE_SEMANTIC_MISS)


@dataclass(frozen=True)
class SemanticAssessment:
    """What layer 2 said about one request, and whether anybody acted on it.

    Carries no text, no span, and no token list: the score, the claim, the mode,
    and enough of layer 1's verdict to reconstruct the disagreement offline. It is
    written to the decision log next to ``shadow``, and it is the raw material for
    the promotion metrics -- which is why "which layer said what" is a field and
    not a formatted string.
    """

    mode: str
    ran: bool = False
    #: Layer 2's probability that the request is contextually sensitive.
    score: float = 0.0
    threshold: float = DEFAULT_SEMANTIC_THRESHOLD
    fired: bool = False
    #: The level layer 2 asserts when it fires. Recorded in shadow too, because
    #: "what would it have done" is the measurement; :attr:`enforced` is what
    #: separates that from acting on it.
    asserted_level: str | None = None
    #: True only when the assessment actually affected routing (mode ``enforce``
    #: and fired). Shadow is always False here, whatever the score.
    enforced: bool = False
    model_version: str = ""
    latency_ms: float = 0.0
    degraded: bool = False
    reason: str | None = None
    #: Layer 1's side of the comparison. Detector ids only -- the same
    #: identify-without-retaining rule :class:`~jev_route.schema.GateFinding`
    #: follows for spans.
    layer1_fired: bool = False
    layer1_floor: str | None = None
    layer1_detectors: tuple[str, ...] = ()
    layer1_blocks_backend: bool = False
    disagreement: str | None = None

    @property
    def sensitivity_floor(self) -> str | None:
        """The floor the router may apply. ``None`` unless this assessment enforced."""
        return self.asserted_level if self.enforced else None

    @property
    def force_local(self) -> bool:
        """Whether layer 2 forces the local tier on its own authority."""
        return self.enforced

    @classmethod
    def inert(cls, mode: str, reason: str | None = None) -> SemanticAssessment:
        """An assessment for a layer that did not run. Not an error, and says so."""
        return cls(mode=mode, ran=False, reason=reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "ran": self.ran,
            "score": self.score,
            "threshold": self.threshold,
            "fired": self.fired,
            "asserted_level": self.asserted_level,
            "enforced": self.enforced,
            "model_version": self.model_version,
            "latency_ms": self.latency_ms,
            "degraded": self.degraded,
            "reason": self.reason,
            "layer1": {
                "fired": self.layer1_fired,
                "sensitivity_floor": self.layer1_floor,
                "detectors": list(self.layer1_detectors),
                "blocks_backend": self.layer1_blocks_backend,
            },
            "disagreement": self.disagreement,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SemanticAssessment:
        layer1 = data.get("layer1") or {}
        if not isinstance(layer1, Mapping):
            layer1 = {}
        known = {f.name for f in cls.__dataclass_fields__.values()} - {
            "layer1_fired",
            "layer1_floor",
            "layer1_detectors",
            "layer1_blocks_backend",
        }
        kwargs: dict[str, Any] = {k: v for k, v in data.items() if k in known}
        return cls(
            **kwargs,
            layer1_fired=bool(layer1.get("fired", False)),
            layer1_floor=layer1.get("sensitivity_floor"),
            layer1_detectors=tuple(str(d) for d in (layer1.get("detectors") or ())),
            layer1_blocks_backend=bool(layer1.get("blocks_backend", False)),
        )


# --------------------------------------------------------------------------- #
# The layer
# --------------------------------------------------------------------------- #
class SemanticLayer:
    """Layer 2: scores traffic, logs disagreements, and blocks nothing until promoted.

    Constructed once per process, from policy. Construction is where the promotion
    criteria are checked, so a deployment that asks for enforce without earning it
    fails to start rather than starting in a mode its operator did not ask for.
    """

    def __init__(
        self,
        config: SemanticGatePolicy,
        *,
        scorer: SemanticScorer | None = None,
        artifact: SemanticArtifact | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Args:
            config: the parsed ``gate.semantic`` section.
            scorer: an already-built scorer. Supplying one skips artifact loading,
                which is how tests and embedding hosts inject their own model.
            artifact: an already-loaded artifact, when the caller wants the
                provenance and metrics that come with it.
            metrics: measured promotion metrics, when there is no artifact to read
                them from. Only reachable from code: there is no config key that
                supplies metrics, because a number an operator typed is not a
                measurement.
        """
        self.config = config
        self.artifact = artifact
        self.unavailable_reason: str | None = None
        self._counts: dict[str, int] = {
            "assessed": 0,
            "inert": 0,
            "fired": 0,
            "enforced": 0,
            "errors": 0,
            DISAGREE_SEMANTIC_ONLY: 0,
            DISAGREE_SEMANTIC_MISS: 0,
        }

        if not config.runs:
            self.scorer: SemanticScorer | None = None
        elif scorer is not None:
            self.scorer = scorer
        else:
            self.scorer = None
            self._load_scorer()

        # Explicit metrics win over the artifact's, so a host that re-measures at
        # build time can promote on its own numbers rather than on the ones that
        # shipped in the file.
        source_artifact = artifact if artifact is not None else self.artifact
        measured = dict(metrics) if metrics is not None else dict(source_artifact.metrics if source_artifact else {})
        if config.shadow_metrics:
            measured = merge_metrics(
                measured, load_shadow_metrics(config.shadow_metrics), model_version=self.model_version
            )
        self.metrics: dict[str, Any] = measured
        self._require_promotable(measured)

        #: The threshold the scores are compared against. An artifact's own
        #: threshold wins, because that is the threshold its metrics were measured
        #: at; see :attr:`SemanticArtifact.threshold`.
        self.threshold = float(self.artifact.threshold) if self.artifact is not None else config.threshold
        if self.artifact is not None and abs(self.artifact.threshold - config.threshold) > 1e-9:
            self._refuse_enforce(
                f"gate.semantic.threshold={config.threshold} does not match the artifact's measured "
                f"threshold={self.artifact.threshold}. The promotion metrics describe the artifact's "
                "threshold; moving it without re-measuring would promote a model on numbers that are "
                "not about it. Re-measure, or set the threshold to the artifact's value."
            )

    # -- construction ----------------------------------------------------- #
    @classmethod
    def from_policy(cls, policy: Policy) -> SemanticLayer:
        """Build the layer a :class:`~jev_route.policy.Policy` describes."""
        return cls(policy.semantic_gate)

    def _load_scorer(self) -> None:
        if self.config.artifact is None:
            reason = "no semantic artifact configured (gate.semantic.artifact is unset)"
            self._refuse_enforce(
                f"{reason}. Enforce mode requires a trained, measured model: refusing to start "
                "rather than running an unmeasured gate. Leave mode at shadow until an artifact "
                "exists and passes its promotion criteria."
            )
            self.unavailable_reason = reason
            return
        try:
            self.artifact = SemanticArtifact.load(self.config.artifact)
        except SemanticArtifactError as exc:
            self._refuse_enforce(str(exc))
            self.unavailable_reason = str(exc)
            return
        self.scorer = self.artifact.scorer

    def _refuse_enforce(self, message: str) -> None:
        """Raise unless the layer is allowed to be inert.

        Called on every path where the layer cannot do what ``mode: enforce``
        promises. The alternative -- logging a warning and continuing in shadow --
        is exactly the silent downgrade this design exists to prevent: the config
        would say enforce, the metrics dashboard would say enforce, and nothing
        would be enforcing.
        """
        if self.config.enforces:
            raise SemanticGateError(f"gate.semantic.mode: enforce refused. {message}")

    def _require_promotable(self, measured: Mapping[str, Any]) -> None:
        if not self.config.enforces:
            return
        if not measured:
            self._refuse_enforce(
                "the artifact carries no measured metrics, so promotion cannot be checked. "
                "Re-measure it against a held-out set of synthetic/public positives and real "
                "negatives (jev_route.gate_semantic.measure_promotion_metrics) and retrain."
            )
            return
        report = check_promotion(self.config.enforce_requires, measured)
        if not report.ok:
            self._refuse_enforce("\n" + report.message())

    # -- runtime ---------------------------------------------------------- #
    @property
    def mode(self) -> str:
        return self.config.mode

    @property
    def active(self) -> bool:
        """Whether the layer will actually score a request."""
        return self.scorer is not None

    @property
    def model_version(self) -> str:
        return getattr(self.scorer, "model_version", "none") if self.scorer is not None else "none"

    def assess(
        self,
        text: str,
        *,
        features: RequestFeatures | None = None,
        verdict: GateVerdict | None = None,
    ) -> SemanticAssessment:
        """Score one request against layer 1's verdict and record the comparison.

        Runs on the RAW excerpt, including for requests layer 1 already blocked.
        That is not an egress decision: this layer is local by contract, and
        scoring a blocked request is the only way to measure whether it would have
        caught the thing the checksum caught.

        Never raises. A scorer that throws is recorded as ``degraded`` and the
        request continues on layer 1's verdict alone, which is still a floor:
        losing layer 2 loses telemetry and strictness, never protection.
        """
        if self.scorer is None:
            self._counts["inert"] += 1
            return SemanticAssessment.inert(self.config.mode, self.unavailable_reason or "mode: off")

        layer1_fired = bool(verdict is not None and verdict.sensitivity_floor is not None)
        layer1_floor = verdict.sensitivity_floor if verdict is not None else None
        layer1_detectors = tuple(dict.fromkeys(f.detector for f in (verdict.findings if verdict else ())))
        layer1_blocks = bool(verdict is not None and verdict.blocks_backend)

        started = time.perf_counter()
        try:
            raw_score = float(self.scorer.score(text, features=features))
        except Exception as exc:  # a broken layer 2 is telemetry, not an outage
            self._counts["errors"] += 1
            self._counts["assessed"] += 1
            return SemanticAssessment(
                mode=self.config.mode,
                ran=False,
                threshold=self.threshold,
                degraded=True,
                reason=f"{type(exc).__name__}: {exc}",
                model_version=self.model_version,
                latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
                layer1_fired=layer1_fired,
                layer1_floor=layer1_floor,
                layer1_detectors=layer1_detectors,
                layer1_blocks_backend=layer1_blocks,
            )
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)

        if not math.isfinite(raw_score):
            self._counts["errors"] += 1
            self._counts["assessed"] += 1
            return SemanticAssessment(
                mode=self.config.mode,
                ran=False,
                threshold=self.threshold,
                degraded=True,
                reason=f"scorer returned a non-finite score ({raw_score!r})",
                model_version=self.model_version,
                latency_ms=latency_ms,
                layer1_fired=layer1_fired,
                layer1_floor=layer1_floor,
                layer1_detectors=layer1_detectors,
                layer1_blocks_backend=layer1_blocks,
            )

        score = min(1.0, max(0.0, raw_score))
        fired = score >= self.threshold
        enforced = self.config.enforces and fired
        disagreement: str | None = None
        if fired and not layer1_fired:
            disagreement = DISAGREE_SEMANTIC_ONLY
        elif layer1_fired and not fired:
            disagreement = DISAGREE_SEMANTIC_MISS

        self._counts["assessed"] += 1
        if fired:
            self._counts["fired"] += 1
        if enforced:
            self._counts["enforced"] += 1
        if disagreement is not None:
            self._counts[disagreement] += 1

        return SemanticAssessment(
            mode=self.config.mode,
            ran=True,
            score=round(score, 6),
            threshold=self.threshold,
            fired=fired,
            asserted_level=self.config.level if fired else None,
            enforced=enforced,
            model_version=self.model_version,
            latency_ms=latency_ms,
            layer1_fired=layer1_fired,
            layer1_floor=layer1_floor,
            layer1_detectors=layer1_detectors,
            layer1_blocks_backend=layer1_blocks,
            disagreement=disagreement,
        )

    def stats(self) -> dict[str, Any]:
        """In-process counters. The log is the durable copy; this is the live one."""
        assessed = self._counts["assessed"]
        out: dict[str, Any] = {
            "mode": self.config.mode,
            "active": self.active,
            "model_version": self.model_version,
            "threshold": self.threshold,
            **dict(self._counts),
        }
        if self.unavailable_reason:
            out["unavailable_reason"] = self.unavailable_reason
        if assessed:
            out["fire_rate"] = round(self._counts["fired"] / assessed, 6)
            out["disagreement_rate"] = round(
                (self._counts[DISAGREE_SEMANTIC_ONLY] + self._counts[DISAGREE_SEMANTIC_MISS]) / assessed, 6
            )
        return out


# --------------------------------------------------------------------------- #
# Promotion: measured, not asserted
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LabeledExample:
    """One row of a promotion eval set.

    ``sensitive`` is the ground truth, and its provenance matters: positives come
    from synthetic generators or public corpora, negatives from real traffic the
    gate already allowed through. Never from blocked requests -- see the module
    docstring. ``source`` is carried into the metrics so a report can show what it
    was measured on.
    """

    text: str
    sensitive: bool
    source: str = ""


@dataclass(frozen=True)
class PromotionCheck:
    """One criterion, its requirement, and what was actually measured."""

    name: str
    requirement: str
    measured: str
    ok: bool

    def line(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        return f"  [{mark}] {self.name:<26} measured {self.measured:<12} required {self.requirement}"


@dataclass(frozen=True)
class PromotionReport:
    """The answer to "may this artifact enforce?".

    Always carries the measured numbers, pass or fail, because a refusal that does
    not say *why* gets answered by lowering a threshold in config.
    """

    criteria: Mapping[str, Any]
    metrics: Mapping[str, Any]
    checks: tuple[PromotionCheck, ...]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(check.line().strip() for check in self.checks if not check.ok)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "criteria": dict(self.criteria),
            "metrics": dict(self.metrics),
            "checks": [
                {"name": c.name, "requirement": c.requirement, "measured": c.measured, "ok": c.ok} for c in self.checks
            ],
            "failures": list(self.failures),
        }

    def message(self) -> str:
        """The loud refusal: every check, every number, and what to do about it."""
        head = "PROMOTION CRITERIA MET" if self.ok else "PROMOTION CRITERIA NOT MET"
        lines = [
            head,
            f"  artifact metrics: n_positives={self.metrics.get('n_positives')} "
            f"n_negatives={self.metrics.get('n_negatives')} "
            f"threshold={self.metrics.get('threshold')} "
            f"model_version={self.metrics.get('model_version')}",
            *(check.line() for check in self.checks),
        ]
        if not self.ok:
            lines.append(
                "  The semantic layer stays in shadow. Fix the measurement, not the config: "
                "retrain on more (or better) synthetic positives and real negatives, then "
                "re-measure on a fresh held-out set."
            )
        return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "not measured"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def measure_promotion_metrics(
    scorer: SemanticScorer,
    examples: Iterable[LabeledExample],
    *,
    gate: HardGate | None = None,
    threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    model_version: str | None = None,
) -> dict[str, Any]:
    """Score an eval set against both layers and reduce it to promotion numbers.

    Returns counts and rates only. Not one field carries text, a token, or a span,
    so this report can be pasted into an incident channel or committed to the repo
    without anybody having to think about what is in it.

    Both layers are run on every row, which is what makes ``disagreement_rate``
    mean something: it is the fraction of rows where the deterministic floor and
    the semantic model contradict each other in either direction.

    ``gate`` is typed loosely and built on demand so that measuring does not
    require importing the detector module twice or constructing a gate the caller
    already has.
    """
    from .gate import HardGate

    active_gate = gate if gate is not None else HardGate()
    rows = list(examples)
    n_pos = sum(1 for row in rows if row.sensitive)
    n_neg = len(rows) - n_pos

    caught_pos = 0
    false_pos = 0
    disagreements = 0
    semantic_only = 0
    layer1_fired_total = 0
    semantic_miss = 0
    layer1_caught_pos = 0
    positive_scores: list[float] = []
    negative_scores: list[float] = []

    for row in rows:
        score = min(1.0, max(0.0, float(scorer.score(row.text))))
        fired = score >= threshold
        verdict = active_gate.scan(row.text)
        layer1_fired = verdict.sensitivity_floor is not None
        if row.sensitive:
            positive_scores.append(score)
            caught_pos += int(fired)
            layer1_caught_pos += int(layer1_fired)
        else:
            negative_scores.append(score)
            false_pos += int(fired)
        if layer1_fired:
            layer1_fired_total += 1
            semantic_miss += int(not fired)
        if fired != layer1_fired:
            disagreements += 1
            semantic_only += int(fired)

    def mean(values: Sequence[float]) -> float | None:
        return round(sum(values) / len(values), 6) if values else None

    return {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": model_version if model_version is not None else getattr(scorer, "model_version", ""),
        "scorer": getattr(scorer, "name", ""),
        "threshold": float(threshold),
        "n_examples": len(rows),
        "n_positives": n_pos,
        "n_negatives": n_neg,
        "n_layer1_fired": layer1_fired_total,
        "recall": round(caught_pos / n_pos, 6) if n_pos else None,
        "false_positive_rate": round(false_pos / n_neg, 6) if n_neg else None,
        "disagreement_rate": round(disagreements / len(rows), 6) if rows else None,
        "semantic_only_rate": round(semantic_only / len(rows), 6) if rows else None,
        # None, not 0.0, when unmeasurable: "no misses" and "nothing to miss" are
        # different claims and only one of them is evidence.
        "semantic_miss_rate": round(semantic_miss / layer1_fired_total, 6) if layer1_fired_total else None,
        "layer1_recall": round(layer1_caught_pos / n_pos, 6) if n_pos else None,
        "true_positives": caught_pos,
        "false_positives": false_pos,
        "mean_score_positive": mean(positive_scores),
        "mean_score_negative": mean(negative_scores),
        "max_score_negative": round(max(negative_scores), 6) if negative_scores else None,
    }


def check_promotion(criteria: EnforceCriteria, metrics: Mapping[str, Any]) -> PromotionReport:
    """Compare measured metrics against the criteria. No side effects, no fallback."""
    checks: list[PromotionCheck] = []

    def at_least(name: str, key: str, required: float | int) -> None:
        measured = metrics.get(key)
        ok = isinstance(measured, (int, float)) and not isinstance(measured, bool) and measured >= required
        checks.append(PromotionCheck(name, f">= {required}", _fmt(measured), ok))

    def at_most(name: str, key: str, required: float | int) -> None:
        measured = metrics.get(key)
        ok = isinstance(measured, (int, float)) and not isinstance(measured, bool) and measured <= required
        checks.append(PromotionCheck(name, f"<= {required}", _fmt(measured), ok))

    at_least("recall", "recall", criteria.min_recall)
    at_most("false_positive_rate", "false_positive_rate", criteria.max_false_positive_rate)
    at_most("disagreement_rate", "disagreement_rate", criteria.max_disagreement_rate)
    at_least("n_positives", "n_positives", criteria.min_positive_examples)
    at_least("n_negatives", "n_negatives", criteria.min_negative_examples)
    at_least("shadow_examples", "shadow_n_examples", criteria.min_shadow_examples)

    miss = metrics.get("semantic_miss_rate")
    if miss is None:
        # Unmeasurable is not a pass. It means the eval set contains no example the
        # deterministic floor catches, so there is nothing to check layer 2's
        # agreement against -- which is a gap in the eval set, not in the model.
        checks.append(
            PromotionCheck(
                "semantic_miss_rate",
                f"<= {criteria.max_semantic_miss_rate}",
                "unmeasured",
                False,
            )
        )
    else:
        at_most("semantic_miss_rate", "semantic_miss_rate", criteria.max_semantic_miss_rate)

    return PromotionReport(criteria=criteria.to_dict(), metrics=dict(metrics), checks=tuple(checks))


# --------------------------------------------------------------------------- #
# Reading the shadow log back: the live half of the promotion evidence
# --------------------------------------------------------------------------- #
def measure_shadow_log(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Reduce logged :class:`SemanticAssessment` blocks to the live promotion numbers.

    This is the other half of the evidence, and the half that cannot be faked with
    a hand-written eval set: it is what layer 2 actually did to real traffic while
    it was unable to affect anything. ``recall`` and ``false_positive_rate`` are
    absent because live traffic has no ground-truth labels -- which is precisely
    why those two come from the held-out offline set and the disagreement numbers
    come from here.

    Takes the JSON mappings of decision records (``json.loads`` of each log line,
    or ``record.to_dict()``). Records whose layer did not run are counted and
    excluded: an inert layer observed a million times is still zero observations.
    """
    assessed = 0
    skipped = 0
    fired = 0
    enforced = 0
    disagreements = 0
    semantic_only = 0
    semantic_miss = 0
    layer1_fired = 0
    degraded = 0
    versions: set[str] = set()
    score_sum = 0.0

    for row in rows:
        block = row.get("semantic") if isinstance(row, Mapping) else None
        if not isinstance(block, Mapping):
            skipped += 1
            continue
        if not block.get("ran"):
            skipped += 1
            continue
        assessed += 1
        versions.add(str(block.get("model_version", "")))
        fired += int(bool(block.get("fired")))
        enforced += int(bool(block.get("enforced")))
        degraded += int(bool(block.get("degraded")))
        score_sum += float(block.get("score", 0.0))
        layer1 = block.get("layer1") if isinstance(block.get("layer1"), Mapping) else {}
        layer1_fired += int(bool(layer1.get("fired")))
        kind = block.get("disagreement")
        if kind:
            disagreements += 1
            semantic_only += int(kind == DISAGREE_SEMANTIC_ONLY)
            semantic_miss += int(kind == DISAGREE_SEMANTIC_MISS)

    if len(versions) > 1:
        raise SemanticGateError(
            f"the shadow log mixes {len(versions)} semantic model versions {sorted(versions)}. "
            "A disagreement rate averaged over two different models is not evidence about either "
            "of them; re-measure from the point the current artifact was deployed."
        )
    if assessed == 0:
        raise SemanticGateError(
            f"no shadow assessments found in {assessed + skipped} record(s). Layer 2 must have run "
            "in shadow on real traffic before it can be promoted; check gate.semantic.mode and "
            "gate.semantic.artifact for the period this log covers."
        )
    return {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "shadow_model_version": next(iter(versions), ""),
        "shadow_n_examples": assessed,
        "shadow_skipped_records": skipped,
        "shadow_fire_rate": round(fired / assessed, 6),
        "shadow_enforced": enforced,
        "shadow_degraded": degraded,
        "shadow_mean_score": round(score_sum / assessed, 6),
        "disagreement_rate": round(disagreements / assessed, 6),
        "semantic_only_rate": round(semantic_only / assessed, 6),
        "semantic_miss_rate": round(semantic_miss / layer1_fired, 6) if layer1_fired else None,
        "n_layer1_fired": layer1_fired,
    }


def load_shadow_metrics(path: str | Path) -> dict[str, Any]:
    """Read a shadow-metrics JSON file written from :func:`measure_shadow_log`."""
    p = Path(path)
    if not p.exists():
        raise SemanticGateError(
            f"gate.semantic.shadow_metrics points at {p}, which does not exist. Produce it with "
            "jev_route.gate_semantic.measure_shadow_log over the decision log covering the shadow "
            "period; a promotion cannot be checked against numbers that were never measured."
        )
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SemanticGateError(f"{p} could not be read as JSON: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise SemanticGateError(f"{p} must contain a JSON object of metrics")
    return dict(raw)


def merge_metrics(
    offline: Mapping[str, Any], shadow: Mapping[str, Any] | None, *, model_version: str
) -> dict[str, Any]:
    """Overlay live shadow numbers on the artifact's offline numbers.

    Shadow wins where both speak, because live traffic is the better evidence for
    a rate that depends on the traffic. The model versions must match: shadow
    numbers collected against last month's artifact are not evidence about this
    one, and silently averaging them is how a promotion gets approved on data
    describing a model that is no longer deployed.
    """
    merged = dict(offline)
    if not shadow:
        return merged
    shadow_version = str(shadow.get("shadow_model_version", ""))
    if shadow_version != model_version:
        raise SemanticGateError(
            f"shadow metrics describe model_version={shadow_version or '(none)'} but the artifact is "
            f"{model_version!r}. Refusing to combine them: re-run the shadow period against the "
            "artifact you intend to promote."
        )
    merged.update(shadow)
    merged["metrics_sources"] = ["artifact_holdout", "shadow_log"]
    return merged


# --------------------------------------------------------------------------- #
# A minimal trainer, so layer 2 is testable end to end without numpy
# --------------------------------------------------------------------------- #
def _bucket(text: str) -> float:
    """Deterministic ``[0, 1)`` bucket for a string.

    SHA-256 rather than the builtin ``hash()``: the builtin is salted per process,
    so a split computed today would not be the split computed tomorrow, and a
    holdout that changes membership cannot be used to make a promotion decision.
    """
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16) / float(16**12)


def _split_examples(
    positives: Sequence[str], negatives: Sequence[str], holdout_fraction: float
) -> tuple[list[LabeledExample], list[LabeledExample]]:
    """Split by content hash, so the holdout is stable across runs and processes.

    A hash is stable but not balanced: with a handful of unique rows it can land
    every row on one side, which would make the metrics unmeasurable for a reason
    that has nothing to do with the data. When the fraction is strictly inside
    (0, 1) one row is moved deterministically to the empty side so both sides
    exist. An explicit 0.0 or 1.0 is an operator choice and is honoured verbatim;
    the caller is the one that refuses such a set, with the measured numbers.
    """
    train: list[LabeledExample] = []
    holdout: list[LabeledExample] = []
    for text, sensitive, source in [(t, True, "positive") for t in positives] + [
        (t, False, "negative") for t in negatives
    ]:
        row = LabeledExample(text=text, sensitive=sensitive, source=source)
        (holdout if _bucket(text) < holdout_fraction else train).append(row)
    if 0.0 < holdout_fraction < 1.0:
        if not holdout and len(train) >= 2:
            mover = max(train, key=lambda row: _bucket(row.text))
            train.remove(mover)
            holdout.append(mover)
        elif not train and len(holdout) >= 2:
            mover = min(holdout, key=lambda row: _bucket(row.text))
            holdout.remove(mover)
            train.append(mover)
    return train, holdout


def _build_vocab(
    rows: Sequence[LabeledExample], *, min_df: int, max_vocab: int, tokenizer: Mapping[str, Any]
) -> dict[str, int]:
    """Token -> column, ranked by document frequency then alphabetically.

    Deterministic selection matters more than clever selection here: the same
    inputs must produce the same vocabulary, or the artifact's weights cannot be
    reproduced and a reviewer cannot re-derive what the model keys on.
    """
    document_frequency: dict[str, int] = {}
    for row in rows:
        for token in set(tokenize(row.text, **tokenizer)):
            document_frequency[token] = document_frequency.get(token, 0) + 1
    ranked = sorted(
        ((token, df) for token, df in document_frequency.items() if df >= min_df and token),
        key=lambda kv: (-kv[1], kv[0]),
    )
    return {token: index for index, (token, _df) in enumerate(ranked[:max_vocab])}


def _train_weights(
    rows: Sequence[LabeledExample],
    vocab: Mapping[str, int],
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
    tokenizer: Mapping[str, Any],
) -> tuple[dict[str, float], float]:
    """Batch gradient descent on binary cross-entropy with an L2 penalty.

    Presence features, not counts: a token appearing nine times in one prompt is
    not nine times the evidence that it is sensitive, and count features make the
    score a function of prompt length, which is a false-positive machine on long
    support threads.

    Pure Python and O(nnz) per epoch. That is slow next to numpy and fast enough
    for the tens of thousands of rows a promotion set has, which is the right
    trade for a component that must run where numpy does not exist.
    """
    tokenized = [
        (frozenset(t for t in tokenize(row.text, **tokenizer) if t in vocab), 1.0 if row.sensitive else 0.0)
        for row in rows
    ]
    n = len(tokenized) or 1
    positives = sum(label for _tokens, label in tokenized)
    prior = min(max(positives / n, 1e-6), 1.0 - 1e-6)
    bias = math.log(prior / (1.0 - prior))
    weights: dict[str, float] = dict.fromkeys(vocab, 0.0)

    for _epoch in range(max(1, epochs)):
        gradient: dict[str, float] = {}
        bias_gradient = 0.0
        for tokens, label in tokenized:
            z = bias + sum(weights[t] for t in tokens)
            error = sigmoid(z) - label
            if error == 0.0:
                continue
            bias_gradient += error
            for token in tokens:
                gradient[token] = gradient.get(token, 0.0) + error
        step = learning_rate / n
        bias -= step * bias_gradient
        for token, value in gradient.items():
            weights[token] -= step * value + learning_rate * l2 * weights[token]
    # Drop tokens that training left at zero: they cost a dict lookup per score and
    # carry no information, and a reviewer reading the artifact should see the
    # model, not the search space it was trained in.
    return {token: round(weight, 6) for token, weight in weights.items() if abs(weight) > 1e-9}, round(bias, 6)


def fit_scorer(
    positives: Sequence[str],
    negatives: Sequence[str],
    *,
    holdout_fraction: float = 0.25,
    epochs: int = 120,
    learning_rate: float = 1.0,
    l2: float = 1e-3,
    min_df: int = 2,
    max_vocab: int = 3000,
    threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    level: str = DEFAULT_SEMANTIC_LEVEL,
    tokenizer: Mapping[str, Any] | None = None,
    model_version: str | None = None,
    positive_source: str = "synthetic",
    negative_source: str = "production",
    note: str = "",
    gate: HardGate | None = None,
) -> SemanticArtifact:
    """Train a scorer and measure it on its own held-out split.

    A reference implementation, deliberately small: it exists so that the two
    layers can be exercised end to end -- trained, measured, promoted, refused --
    without a numerical stack, and so the artifact format has a producer. A real
    deployment will usually train a stronger student elsewhere and write the same
    artifact format; nothing downstream changes, because promotion is decided on
    the metrics block and not on how the weights were produced.

    The returned artifact's ``metrics`` are measured on the **holdout** only. A
    promotion decided on training-set recall is a promotion decided on a number
    the model was allowed to memorise.
    """
    clean_positives = sorted({str(t).strip() for t in positives if str(t).strip()})
    clean_negatives = sorted({str(t).strip() for t in negatives if str(t).strip()})
    if not clean_positives:
        raise SemanticGateError("fit_scorer needs at least one positive example")
    if not clean_negatives:
        raise SemanticGateError("fit_scorer needs at least one negative example")
    if level not in ASSERTABLE_LEVELS:
        raise SemanticGateError(f"level must be one of {list(ASSERTABLE_LEVELS)}, got {level!r}")
    overlap = sorted(set(clean_positives) & set(clean_negatives))
    if overlap:
        raise SemanticGateError(
            f"{len(overlap)} example(s) appear as both positive and negative; the first is "
            f"{len(overlap[0])} chars long. Contradictory labels make every metric meaningless."
        )

    token_cfg = {"lower": True, "min_token_chars": 2, "ngram_max": 2, **(tokenizer or {})}
    train, holdout = _split_examples(clean_positives, clean_negatives, float(holdout_fraction))
    if not train or not holdout:
        raise SemanticGateError(
            f"the holdout split produced {len(train)} train and {len(holdout)} holdout rows; "
            "promotion cannot be measured without both. Supply more examples or lower "
            "holdout_fraction."
        )
    # A document-frequency floor higher than the training set can never be met:
    # with a handful of unique rows no token repeats, and refusing here would
    # make the smallest demonstrable set untrainable. Fit at min_df=1 instead
    # and record the effective floor in the artifact. This is not a loophole:
    # promotion is decided on the holdout metrics against the deployment's
    # criteria, and a set this small cannot reach the minimum sample sizes.
    effective_min_df = min(int(min_df), max(1, len(train)))
    vocab = _build_vocab(train, min_df=effective_min_df, max_vocab=int(max_vocab), tokenizer=token_cfg)
    if not vocab:
        raise SemanticGateError(
            "no token reached min_df; the positive and negative sets share no repeated vocabulary. "
            "Lower min_df or supply more examples."
        )
    weights, bias = _train_weights(
        train, vocab, epochs=int(epochs), learning_rate=float(learning_rate), l2=float(l2), tokenizer=token_cfg
    )
    version = model_version or f"semantic-logreg-{hashlib.sha256(str(sorted(weights)).encode()).hexdigest()[:8]}"
    scorer = LexiconScorer(
        weights,
        bias=bias,
        model_version=version,
        lower=bool(token_cfg.get("lower", True)),
        min_token_chars=int(token_cfg.get("min_token_chars", 2)),
        ngram_max=int(token_cfg.get("ngram_max", 2)),
    )
    metrics = measure_promotion_metrics(scorer, holdout, gate=gate, threshold=threshold)
    payload: dict[str, Any] = {
        "kind": SEMANTIC_ARTIFACT_KIND,
        "artifact_version": SEMANTIC_ARTIFACT_VERSION,
        "model_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "threshold": float(threshold),
        "level": level,
        "scorer": {
            "name": LexiconScorer.name,
            "bias": bias,
            "tokenizer": dict(token_cfg),
            "weights": weights,
        },
        "metrics": metrics,
        # Informational: the criteria the artifact was measured against at build
        # time. Promotion re-checks against the *deployment's* criteria on load.
        "criteria": EnforceCriteria().to_dict(),
        "provenance": Provenance(
            positives=positive_source, negatives=negative_source, trained_on_blocked_content=False, note=note
        ).to_dict(),
        "training": {
            "algorithm": "logreg-batch-gd",
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "l2": float(l2),
            "min_df": effective_min_df,
            "max_vocab": int(max_vocab),
            "vocab_size": len(vocab),
            "weights_kept": len(weights),
            "train_rows": len(train),
            "holdout_rows": len(holdout),
            "holdout_fraction": float(holdout_fraction),
            "positive_examples": len(clean_positives),
            "negative_examples": len(clean_negatives),
        },
    }
    # Round-tripped through the loader on purpose: a trainer that can produce an
    # artifact its own loader rejects is a bug that should surface here, not in a
    # proxy starting up three weeks later.
    return SemanticArtifact.from_dict(payload, source=f"fit_scorer({version})")


__all__ = [
    "ASSERTABLE_LEVELS",
    "DEFAULT_SEMANTIC_LEVEL",
    "DEFAULT_SEMANTIC_MODE",
    "DEFAULT_SEMANTIC_THRESHOLD",
    "DISAGREEMENT_KINDS",
    "DISAGREE_SEMANTIC_MISS",
    "DISAGREE_SEMANTIC_ONLY",
    "SCORER_KINDS",
    "SEMANTIC_ARTIFACT_FILE",
    "SEMANTIC_ARTIFACT_KIND",
    "SEMANTIC_ARTIFACT_VERSION",
    "SEMANTIC_MODES",
    "BlockedMetadataPolicy",
    "EnforceCriteria",
    "LabeledExample",
    "LexiconScorer",
    "NullScorer",
    "PromotionCheck",
    "PromotionReport",
    "Provenance",
    "SemanticArtifact",
    "SemanticArtifactError",
    "SemanticAssessment",
    "SemanticConfigError",
    "SemanticGateError",
    "SemanticGatePolicy",
    "SemanticLayer",
    "SemanticScorer",
    "check_promotion",
    "fit_scorer",
    "load_shadow_metrics",
    "measure_promotion_metrics",
    "measure_shadow_log",
    "merge_metrics",
    "resolve_floor",
    "sigmoid",
    "tokenize",
]
