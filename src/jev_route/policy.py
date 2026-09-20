"""The policy engine: where calibrated answers become a routing decision.

This is the part of jev-route an operator actually owns. The backend supplies
judgements with probabilities; this module turns those into a tier and a concrete
model, and it does so from a YAML file you can read, review in a pull request, and
change without touching code.

Three deliberate design choices:

**1. Expressions are parsed, not eval'd.** Rule conditions are compiled from a
strictly limited subset of Python's AST -- names, constants, comparisons, boolean
operators, and literal collections. There is no ``eval``, no attribute access into
the interpreter, no calls, no imports. A policy file is configuration that gets
loaded at startup from wherever operators keep config, and "we checked it looked
safe" is not a control. Anything outside the whitelist raises
:class:`PolicyError` at load time, naming the offending node, rather than at
request time.

**2. The local gate is a floor the policy cannot talk itself out of.** Rules see
``gate_force_local``, ``gate_blocks_backend`` and ``gate_sensitivity_floor``, and
the default policy routes on them first. But even if an operator deletes those
rules, the router still merges the gate's sensitivity floor into the effective
sensitivity *before* rules run, so a checksum-validated card number cannot be
routed to a frontier model by a clever condition. The gate is not one rule among
many; it is a constraint on all of them.

**3. Uncertainty is a first-class input.** ``on_uncertain`` bumps the effective
answer *stricter* when the backend's confidence falls below a threshold. "70%
sure it is internal" becomes "treat it as confidential". This is the behaviour a
regex guardrail cannot express and an argmax classifier throws away, and it is the
single most useful knob in the file.
"""

from __future__ import annotations

import ast
import itertools
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .gate_semantic import (
    BlockedMetadataPolicy,
    SemanticConfigError,
    SemanticGatePolicy,
)
from .schema import (
    COMPLEXITY_LEVELS,
    SENSITIVITY_LEVELS,
    TIERS,
    bump_within,
    level_index,
)

POLICY_VERSION = 1

#: Expression variables and where they come from. Documented because a policy
#: author cannot introspect the engine at runtime.
EXPRESSION_VARIABLES: tuple[str, ...] = (
    "complexity",
    "sensitivity",
    "domain",
    "complexity_confidence",
    "sensitivity_confidence",
    "domain_confidence",
    "pii",
    "pii_present",
    "gate_force_local",
    "gate_blocks_backend",
    "gate_sensitivity_floor",
    "gate_detectors",
    "advisory_topics",
    "semantic_fired",
    "semantic_score",
    "semantic_force_local",
    "semantic_mode",
    "degraded",
    "cached",
    "tier",
    "model",
    "char_len",
    "word_count",
    "line_count",
    "code_blocks",
    "question_marks",
    "has_stack_trace",
    "lang",
    "n_prior_turns",
    "requested_model",
    "metadata",
)

_ALLOWED_NODES: frozenset[type[ast.AST]] = frozenset(
    {
        ast.Expression,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.UnaryOp,
        ast.Not,
        ast.USub,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.In,
        ast.NotIn,
        ast.Is,
        ast.IsNot,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.List,
        ast.Tuple,
        ast.Set,
        ast.Subscript,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
    }
)


class PolicyError(ValueError):
    """A policy file that cannot be honoured. Raised at load time."""


# --------------------------------------------------------------------------- #
# Safe expression evaluation
# --------------------------------------------------------------------------- #
def _validate_node(node: ast.AST, expression: str) -> None:
    if type(node) not in _ALLOWED_NODES:
        raise PolicyError(
            f"disallowed syntax {type(node).__name__!r} in rule expression {expression!r}. "
            f"Allowed: names, constants, comparisons, and/or/not, in/not in, "
            f"literal lists/sets/tuples, subscripting, + - *. No calls, no attribute access."
        )
    if isinstance(node, ast.Name) and node.id not in EXPRESSION_VARIABLES:
        raise PolicyError(
            f"unknown variable {node.id!r} in rule expression {expression!r}. "
            f"Available: {', '.join(EXPRESSION_VARIABLES)}"
        )
    if isinstance(node, ast.Subscript) and not isinstance(node.value, ast.Name):
        raise PolicyError(f"only simple names may be subscripted, in {expression!r}")
    for child in ast.iter_child_nodes(node):
        _validate_node(child, expression)


@dataclass(frozen=True)
class Expression:
    """A compiled, side-effect-free boolean condition."""

    source: str
    _tree: ast.Expression = field(compare=False, repr=False)

    @classmethod
    def compile(cls, source: str) -> Expression:
        text = (source or "").strip()
        if not text:
            raise PolicyError("rule expression is empty")
        try:
            tree = ast.parse(text, mode="eval")
        except SyntaxError as exc:
            raise PolicyError(f"rule expression {source!r} is not valid syntax: {exc.msg}") from exc
        _validate_node(tree, source)
        return cls(source=text, _tree=tree)

    def evaluate(self, namespace: Mapping[str, Any]) -> bool:
        code = compile(self._tree, "<jev-route-policy>", "eval")
        try:
            return bool(eval(code, {"__builtins__": {}}, dict(namespace)))
        except Exception as exc:  # a missing key or bad comparison is a policy bug
            raise PolicyError(f"rule expression {self.source!r} failed to evaluate: {exc}") from exc


_COMPILE_CACHE: dict[str, Expression] = {}
_CACHE_LOCK = threading.Lock()


def compile_expression(source: str) -> Expression:
    """Compile and memoize an expression. Policies are parsed once, evaluated often."""
    with _CACHE_LOCK:
        cached = _COMPILE_CACHE.get(source)
        if cached is None:
            cached = Expression.compile(source)
            _COMPILE_CACHE[source] = cached
        return cached


# --------------------------------------------------------------------------- #
# Policy structure
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rule:
    """One ordered condition -> outcome pair. The first match wins."""

    rule_id: str
    tier: str | None
    model: str | None
    expression: Expression | None
    reason: str

    @property
    def is_default(self) -> bool:
        return self.expression is None


@dataclass(frozen=True)
class UncertaintyPolicy:
    """What to do when the backend is not sure. Always in the stricter direction."""

    sensitivity_confidence_below: float | None = 0.8
    sensitivity_bump_levels: int = 1
    complexity_confidence_below: float | None = 0.7
    complexity_bump_levels: int = 1
    #: A noul near 0.5 is a coin flip, not a "no". Treat it as a "yes" so the
    #: stricter rule fires. This is the fail-safe reading of an uncertain PII
    #: judgement and it is the single most important default in the file.
    pii_uncertain_threshold: float = 0.35
    pii_uncertain_counts_as_present: bool = True
    #: Below this sensitivity certainty, stop bumping one level at a time and send
    #: the request to the safest tier outright. ``None`` disables it.
    #:
    #: Read ``confidence`` as CERTAINTY, not as the top probability: it is
    #: normalized entropy over the whole distribution, so a confident-looking
    #: 0.63/0.34 split scores about 0.28. That is why this threshold looks low.
    #:
    #: The default is 0.25 because that is what the shipped 223-prompt eval set
    #: supports, and the measurement is not close. Sweeping the threshold over the
    #: real Jev backend's own certainties (11 baseline leaks, 136 rows labelled for
    #: a cloud tier):
    #:
    #:       T     leaks fixed    cloud rows needlessly air-gapped
    #:     0.25        1 / 11                    0 / 136
    #:     0.30        1 / 11                    2 / 136
    #:     0.40        2 / 11                   10 / 136
    #:     0.50        3 / 11                   18 / 136
    #:     0.60        3 / 11                   27 / 136
    #:
    #: Above 0.25 every step buys one fewer leak at the price of several percent of
    #: legitimate cloud traffic, because low certainty is mostly noise here rather
    #: than a signal about sensitivity: rows labelled for the cloud have a median
    #: certainty of 0.94 but a 10th percentile of 0.46. So 0.25 is the highest
    #: setting with a non-negative trade, and it is the default.
    #:
    #: Raise it deliberately if your risk posture values preventing one egress over
    #: localizing a dozen benign requests -- 0.5 is a defensible choice for
    #: regulated workloads, and costs about 13% of cloud traffic on this dataset.
    #: Keep it well below ``sensitivity_confidence_below``; if the two are equal the
    #: gradient never applies and all uncertain traffic is air-gapped.
    force_local_confidence_below: float | None = 0.25


@dataclass(frozen=True)
class FailurePolicy:
    """What to do when the decision backend cannot answer.

    ``fail_closed`` routes to the safest tier -- for a sensitivity-aware router
    that means the tier whose data never leaves the building. ``fail_open`` routes
    to the most capable tier, which is right when the risk you care about is answer
    quality rather than data egress. Both are defensible; silently picking one for
    the operator is not, so it is explicit config with a safe default.
    """

    mode: str = "fail_closed"
    fail_closed_tier: str = "local"
    fail_open_tier: str = "strong"


def _parse_uncertainty(raw: Any) -> UncertaintyPolicy:
    """Build the uncertainty policy, ignoring keys this version does not know.

    Unknown keys are dropped rather than fatal: a policy written for a newer
    jev-route should still load on an older one, and a typo in an optional knob
    should not take down a proxy at startup. Every knob that decides whether data
    leaves the building has a safe default, so an ignored key degrades toward
    stricter, never toward looser.
    """
    if raw is None:
        return UncertaintyPolicy()
    if not isinstance(raw, Mapping):
        raise PolicyError(f"on_uncertain must be a mapping, got {type(raw).__name__}")
    known = {f.name for f in UncertaintyPolicy.__dataclass_fields__.values()}
    values = {k: v for k, v in raw.items() if k in known}
    # Bumps only ever go stricter. A negative value is not a milder setting --
    # it is the inversion of the knob's purpose: de-escalating a low-confidence
    # sensitivity judgement is how "confidential, not sure" becomes "internal,
    # send it", so a policy that names one fails the deploy, not the request.
    # Zero is the legal "disable this bump" and is honoured.
    for bump_key in ("sensitivity_bump_levels", "complexity_bump_levels"):
        value = values.get(bump_key)
        if value is None:
            continue
        try:
            as_int = int(value)
        except (TypeError, ValueError):
            raise PolicyError(f"on_uncertain.{bump_key} must be an integer, got {value!r}") from None
        if as_int < 0:
            raise PolicyError(
                f"on_uncertain.{bump_key} is {value}; bumps only ever go stricter. Set it to 0 "
                "to disable the bump, or to a positive count to escalate that many levels. A "
                "negative value would move an uncertain judgement DOWN the ladder, which is "
                "exactly the inversion this knob exists to prevent."
            )
    return UncertaintyPolicy(**values)


def _parse_semantic_gate(gate_section: Mapping[str, Any], *, where: str = "gate") -> SemanticGatePolicy:
    """Parse and validate ``gate.semantic`` at load time.

    Two layers of "fail the deploy, not the request", and they are not redundant:

    * **here**, the shape -- mode spelling, threshold range, asserted level,
      unknown keys. A typo in this section is not a harmless no-op: ``mode:
      enfoce`` would leave a gate in shadow that its operator believes is
      enforcing, which is the one failure mode worse than not having the layer.
    * **at layer construction** (:class:`~jev_route.gate_semantic.SemanticLayer`),
      the substance -- whether the artifact exists and whether its measured
      metrics earn enforce. That check needs the artifact, and reading a model
      file is not the policy parser's job.
    """
    try:
        return SemanticGatePolicy.parse(gate_section.get("semantic"), where=f"{where}.semantic")
    except SemanticConfigError as exc:
        raise PolicyError(str(exc)) from exc


def _parse_blocked_metadata(gate_section: Mapping[str, Any], *, where: str = "gate") -> BlockedMetadataPolicy:
    """Parse ``gate.blocked_metadata``: the gate's refusal telemetry stream."""
    try:
        return BlockedMetadataPolicy.parse(gate_section.get("blocked_metadata"), where=f"{where}.blocked_metadata")
    except SemanticConfigError as exc:
        raise PolicyError(str(exc)) from exc


def _parse_failure(raw: Any) -> dict[str, Any]:
    """Normalize ``on_backend_down``, which accepts a mapping or a bare mode string.

    The string shorthand must be checked FIRST. ``dict("fail_open")`` does not
    raise a clear error -- it produces index/character nonsense or fails deep
    inside the dataclass -- so handling the mapping case first turns a documented
    config form into an opaque crash.
    """
    if isinstance(raw, str):
        return {"mode": raw.strip()}
    if isinstance(raw, Mapping):
        return dict(raw)
    if raw is None:
        return {}
    raise PolicyError(f"on_backend_down must be a mapping or a mode string, got {type(raw).__name__}")


@dataclass(frozen=True)
class Policy:
    """A loaded, validated routing policy."""

    tiers: Mapping[str, tuple[str, ...]]
    rules: tuple[Rule, ...]
    tier_order: tuple[str, ...]
    uncertainty: UncertaintyPolicy
    failure: FailurePolicy
    pii_threshold: float = 0.5
    backend: Mapping[str, Any] = field(default_factory=dict)
    gate: Mapping[str, Any] = field(default_factory=dict)
    cache: Mapping[str, Any] = field(default_factory=dict)
    logging: Mapping[str, Any] = field(default_factory=dict)
    shadow: Mapping[str, Any] = field(default_factory=dict)
    #: Parsed ``gate.semantic``: the second gate layer. Shadow by default, and
    #: inert by default, because no artifact has been trained yet.
    semantic_gate: SemanticGatePolicy = field(default_factory=SemanticGatePolicy)
    #: Parsed ``gate.blocked_metadata``: what to record when the gate refuses a
    #: request. Enabled by default; it carries no content.
    blocked_metadata: BlockedMetadataPolicy = field(default_factory=BlockedMetadataPolicy)
    version: int = POLICY_VERSION
    source: str | None = None
    #: The document this policy was parsed from, kept verbatim so
    #: :meth:`with_overrides` can round-trip back to YAML without losing comments
    #: it never understood in the first place.
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)
    #: Round-robin cursor per tier, so a tier with several deployments spreads load.
    _cursors: Any = field(default_factory=lambda: itertools.count(), compare=False, repr=False)
    _cursor_lock: threading.Lock = field(default_factory=threading.Lock, compare=False, repr=False)
    _cursor_state: dict[str, int] = field(default_factory=dict, compare=False, repr=False)

    # -- loading ---------------------------------------------------------- #
    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, source: str | None = None) -> Policy:
        version = int(raw.get("version", POLICY_VERSION))
        if version != POLICY_VERSION:
            raise PolicyError(f"unsupported policy version {version}; this build understands version {POLICY_VERSION}")

        tiers_raw = raw.get("tiers") or {}
        if not isinstance(tiers_raw, Mapping) or not tiers_raw:
            raise PolicyError("policy must define at least one tier under `tiers:`")
        tiers: dict[str, tuple[str, ...]] = {}
        for name, models in tiers_raw.items():
            name = str(name).strip()
            if not name:
                raise PolicyError("tier names must be non-empty")
            if isinstance(models, str):
                resolved: tuple[str, ...] = (models.strip(),)
            elif isinstance(models, Sequence):
                resolved = tuple(str(m).strip() for m in models if str(m).strip())
            else:
                raise PolicyError(f"tier {name!r} must map to a model name or a list of model names")
            if not resolved:
                raise PolicyError(f"tier {name!r} lists no models")
            tiers[name] = resolved

        tier_order_raw = raw.get("tier_order") or []
        tier_order = tuple(str(t).strip() for t in tier_order_raw) or tuple(TIERS)
        unknown_order = [t for t in tier_order if t not in tiers]
        if unknown_order:
            raise PolicyError(f"tier_order names tiers that are not defined: {unknown_order}")
        if len(set(tier_order)) != len(tier_order):
            raise PolicyError("tier_order contains duplicates")

        rules = tuple(_parse_rule(index, item, tiers) for index, item in enumerate(raw.get("rules") or []))
        if not rules:
            raise PolicyError("policy defines no rules")
        defaults = [r for r in rules if r.is_default]
        if len(defaults) > 1:
            raise PolicyError("at most one rule may omit `if:` (the default rule)")
        if defaults and defaults[0] is not rules[-1]:
            raise PolicyError("the default rule (no `if:`) must be last")
        if not defaults:
            # Without a default, a request matching nothing would have no answer.
            # Fail loudly at load time instead of at request time.
            raise PolicyError("policy must end with a default rule that has no `if:`")

        uncertainty = _parse_uncertainty(raw.get("on_uncertain"))
        failure_raw = _parse_failure(raw.get("on_backend_down"))
        failure_fields = {"mode", "fail_closed_tier", "fail_open_tier"}
        failure = FailurePolicy(**{k: v for k, v in failure_raw.items() if k in failure_fields})
        if failure.mode not in ("fail_closed", "fail_open"):
            raise PolicyError(f"on_backend_down.mode must be fail_closed or fail_open, got {failure.mode!r}")
        for tier_name in (failure.fail_closed_tier, failure.fail_open_tier):
            if tier_name not in tiers:
                raise PolicyError(f"failure tier {tier_name!r} is not defined under `tiers:`")

        gate_raw = raw.get("gate") or {}
        if not isinstance(gate_raw, Mapping):
            # ``dict("skip_backend")`` and ``dict([...])`` fail in ways that name a
            # character index rather than a config key, so the section is checked
            # before anything reads from it.
            raise PolicyError(f"`gate:` must be a mapping, got {type(gate_raw).__name__}")

        return cls(
            tiers=tiers,
            rules=rules,
            tier_order=tier_order,
            uncertainty=uncertainty,
            failure=failure,
            pii_threshold=float(raw.get("pii_threshold", 0.5)),
            backend=dict(raw.get("backend") or {}),
            gate=dict(gate_raw),
            cache=dict(raw.get("cache") or {}),
            logging=dict(raw.get("logging") or {}),
            shadow=dict(raw.get("shadow") or {}),
            semantic_gate=_parse_semantic_gate(gate_raw),
            blocked_metadata=_parse_blocked_metadata(gate_raw),
            version=version,
            source=source,
            raw=dict(raw),
        )

    # -- round-tripping --------------------------------------------------- #
    def with_overrides(self, **changes: Any) -> Policy:
        """Return a new Policy with top-level config keys replaced.

        Used by ``jev-route graduate`` to perform the backend swap as a config
        change rather than a code change, and by tests to vary one knob without
        rebuilding a document by hand.
        """
        merged = {**self.raw, **changes}
        return Policy.from_dict(merged, source=self.source)

    def with_backend(self, backend_config: Mapping[str, Any]) -> Policy:
        """Return a new Policy pointing at a different decision backend."""
        return self.with_overrides(backend=dict(backend_config))

    def to_yaml(self) -> str:
        """Serialize back to YAML. Comment-free: this is for programmatic writes."""
        return yaml.safe_dump(dict(self.raw), sort_keys=False, allow_unicode=True, width=100)

    def write_yaml(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_yaml(), encoding="utf-8")
        return p

    @classmethod
    def from_yaml(cls, text: str, *, source: str | None = None) -> Policy:
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise PolicyError(f"policy YAML failed to parse: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise PolicyError("policy YAML must be a mapping at the top level")
        return cls.from_dict(raw, source=source)

    @classmethod
    def from_file(cls, path: str | Path) -> Policy:
        p = Path(path)
        if not p.exists():
            raise PolicyError(f"policy file not found: {p}")
        return cls.from_yaml(p.read_text(encoding="utf-8"), source=str(p))

    # -- resolution ------------------------------------------------------- #
    def models_for_tier(self, tier: str) -> tuple[str, ...]:
        models = self.tiers.get(tier)
        if not models:
            raise PolicyError(f"rule selected undefined tier {tier!r}; defined tiers: {sorted(self.tiers)}")
        return models

    def pick_model(self, tier: str) -> str:
        """Round-robin within a tier. Stateless enough to survive restarts."""
        models = self.models_for_tier(tier)
        if len(models) == 1:
            return models[0]
        with self._cursor_lock:
            index = self._cursor_state.get(tier, 0)
            self._cursor_state[tier] = (index + 1) % len(models)
        return models[index]

    def bump_tier(self, tier: str, steps: int = 1) -> str:
        """Move ``steps`` positions up :attr:`tier_order`, clamped.

        "Up" is the more capable / more expensive direction. Used by
        ``on_uncertain`` for complexity and by operators writing escalation rules.
        """
        if tier not in self.tier_order:
            return tier
        idx = level_index(tier, self.tier_order)
        return self.tier_order[max(0, min(len(self.tier_order) - 1, idx + steps))]

    def escalate_sensitivity(self, level: str, steps: int = 1) -> str:
        return bump_within(level, SENSITIVITY_LEVELS, steps)

    def escalate_complexity(self, level: str, steps: int = 1) -> str:
        return bump_within(level, COMPLEXITY_LEVELS, steps)

    def evaluate(self, namespace: Mapping[str, Any]) -> tuple[Rule, str, str]:
        """First matching rule wins.

        Returns ``(rule, tier, model)``. A rule may name a model directly, which
        bypasses tier round-robin -- useful for pinning one tenant to one
        deployment.
        """
        for rule in self.rules:
            if rule.is_default or rule.expression.evaluate(namespace):
                # A rule that names a model pins it exactly; the tier is then
                # recorded for provenance but does not select anything.
                if rule.model:
                    return rule, (rule.tier or ""), rule.model
                tier = rule.tier or self.tier_order[-1]
                return rule, tier, self.pick_model(tier)
        raise PolicyError("no rule matched and no default rule is defined")  # unreachable by validation


def _parse_rule(index: int, raw: Any, tiers: Mapping[str, tuple[str, ...]]) -> Rule:
    if not isinstance(raw, Mapping):
        raise PolicyError(f"rule #{index} must be a mapping with `if:`/`then:` keys")

    outcome = raw.get("then")
    if isinstance(outcome, str):
        outcome = {"tier": outcome}
    if not isinstance(outcome, Mapping):
        raise PolicyError(f"rule #{index}: `then:` must be a tier name or a mapping")

    tier = outcome.get("tier")
    model = outcome.get("model")
    if tier is None and model is None:
        raise PolicyError(f"rule #{index}: `then:` must set at least one of tier or model")
    if tier is not None:
        tier = str(tier).strip()
        if tier not in tiers:
            raise PolicyError(f"rule #{index}: tier {tier!r} is not defined under `tiers:` ({sorted(tiers)})")
    if model is not None:
        model = str(model).strip()
        if not model:
            raise PolicyError(f"rule #{index}: `then.model` must be non-empty")

    condition = raw.get("if")
    expression = compile_expression(str(condition)) if condition is not None else None

    rule_id = str(raw.get("id") or f"rule-{index}" + ("-default" if expression is None else ""))
    reason = str(raw.get("reason") or (f"matched rule {rule_id}" if expression else f"default rule {rule_id}"))
    return Rule(rule_id=rule_id, tier=tier, model=model, expression=expression, reason=reason)


__all__ = [
    "EXPRESSION_VARIABLES",
    "POLICY_VERSION",
    "BlockedMetadataPolicy",
    "Expression",
    "FailurePolicy",
    "Policy",
    "PolicyError",
    "Rule",
    "SemanticGatePolicy",
    "UncertaintyPolicy",
    "compile_expression",
]
