"""Measuring whether a distilled student is good enough to serve.

Three kinds of number, in increasing order of how much they matter:

**Accuracy and macro-F1 per head.** Necessary, and the least interesting: a
student can match the teacher's argmax on 95% of traffic and still be useless to
this router.

**Calibration (ECE).** This is the headline metric. The entire reason to bootstrap
on a System One model rather than prompting an LLM is that the probabilities mean
something, so ``on_uncertain`` can escalate "probably internal" to
"confidential". A distilled model that is accurate but overconfident silently
disables that machinery: the router keeps routing on answers that no longer
deserve trust. ``evaluate`` therefore reports Expected Calibration Error per head,
plus the mean KL to the teacher, and :mod:`jev_route.distill.graduate` can refuse
a cutover on calibration alone.

**Tier agreement.** The operator does not actually care whether the student agrees
about ``domain``; they care whether the request lands on the same tier. So the
teacher's logged answers and the student's predictions are both pushed through the
*same* :class:`~jev_route.policy.Policy` -- same gate merge, same confidence
floors, same rules -- and the resulting tiers are compared. A student that
disagrees on domain but agrees on tier is fine. One that moves a request across
the ``confidential`` boundary is not, and that specific failure gets its own
number (``boundary_error_rate``) because it is the one that turns into a data
egress incident.

Labels here are the *teacher's*, not ground truth. Everything in this module
measures imitation: how faithfully the student reproduces the calibrated cloud
model on held-out traffic. That is the right question for a graduation decision,
but it is not a claim that the teacher was correct.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..policy import Policy

# Imported from the router rather than reimplemented: tier agreement is only
# meaningful if "the same policy" means literally the same code path the
# production router uses. Duplicating the gate merge and the confidence-floor
# escalation here would drift, and a drift in either direction produces a
# confident, wrong graduation recommendation. If the router renames these, this
# import fails loudly at import time instead of diverging silently.
from ..router import _merge_gate as _router_merge_gate
from ..router import _namespace as _router_namespace
from ..schema import (
    SENSITIVITY_LEVELS,
    DecisionAnswers,
    GateVerdict,
    RequestFeatures,
    certainty_from_probabilities,
)
from .artifact import DistilledArtifact, StudentModel, load_artifact
from .export import CHOICE_HEADS, Dataset, TrainingRow, load_dataset, percentiles

#: Sensitivity levels at or above this index are "the data must not leave".
_BOUNDARY_INDEX = SENSITIVITY_LEVELS.index("confidential")


class EvaluateError(RuntimeError):
    """Evaluation could not be performed. Always actionable."""


# --------------------------------------------------------------------------- #
# Policy replay
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PolicyReplay:
    """The tier a set of answers produces under a policy, and how it got there."""

    tier: str
    rule_id: str
    reason: str
    effective_sensitivity: str
    effective_complexity: str
    pii_present: bool
    escalations: tuple[str, ...]


def replay_policy(
    policy: Policy,
    *,
    answers: DecisionAnswers,
    gate: GateVerdict,
    features: RequestFeatures,
    classified: bool = True,
    degraded: bool = False,
    metadata: Mapping[str, Any] | None = None,
    requested_model: str = "",
) -> PolicyReplay:
    """Run answers through steps 5-7 of the router's chain, without the I/O.

    ``classified`` mirrors the router's own flag: the gate merge always applies,
    but the confidence-floor escalation only applies to answers a backend actually
    produced. Replaying a gate-blocked row as if a model had judged it would
    invent an escalation nobody ever made.
    """
    (complexity, sensitivity, pii), _escalations = _router_merge_gate(answers, gate)
    uncertainty = policy.uncertainty
    escalations: list[str] = []

    if classified and not degraded:
        if (
            uncertainty.complexity_confidence_below is not None
            and answers.complexity.confidence < uncertainty.complexity_confidence_below
        ):
            before = complexity
            complexity = policy.escalate_complexity(complexity, uncertainty.complexity_bump_levels)
            if complexity != before:
                escalations.append(f"complexity {before}->{complexity}")
        if (
            uncertainty.sensitivity_confidence_below is not None
            and answers.sensitivity.confidence < uncertainty.sensitivity_confidence_below
        ):
            before = sensitivity
            sensitivity = policy.escalate_sensitivity(sensitivity, uncertainty.sensitivity_bump_levels)
            if sensitivity != before:
                escalations.append(f"sensitivity {before}->{sensitivity}")

    pii_present = pii >= policy.pii_threshold
    if (
        uncertainty.pii_uncertain_counts_as_present
        and not pii_present
        and pii >= uncertainty.pii_uncertain_threshold
    ):
        pii_present = True
        escalations.append(f"pii treated as present at p={pii:.2f}")

    namespace = _router_namespace(
        complexity=complexity,
        sensitivity=sensitivity,
        answers=answers,
        pii=pii,
        pii_present=pii_present,
        verdict=gate,
        features=features,
        degraded=degraded,
        metadata=metadata or {},
        requested_model=requested_model,
    )
    rule, tier, _model = policy.evaluate(namespace)
    return PolicyReplay(
        tier=tier,
        rule_id=rule.rule_id,
        reason=rule.reason,
        effective_sensitivity=sensitivity,
        effective_complexity=complexity,
        pii_present=pii_present,
        escalations=tuple(escalations),
    )


def replay_row(policy: Policy, row: TrainingRow, answers: DecisionAnswers) -> PolicyReplay:
    """Replay one exported row's context (gate, features, metadata) with ``answers``."""
    return replay_policy(
        policy,
        answers=answers,
        gate=GateVerdict.from_dict(row.gate or {}),
        features=row.features,
        classified=bool(row.context.get("classified", True)),
        degraded=False,
        metadata=row.context.get("metadata") or {},
        requested_model=str(row.context.get("requested_model") or ""),
    )


# --------------------------------------------------------------------------- #
# Metric primitives
# --------------------------------------------------------------------------- #
def expected_calibration_error(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    bins: int = 15,
) -> dict[str, Any]:
    """Standard top-label ECE, plus the curve so a reader can see *where* it breaks.

    ``ECE = sum_b (n_b / N) * |accuracy_b - confidence_b|`` over equal-width
    confidence bins. It answers the only question that matters for this project:
    when the student says 0.8, is it right 80% of the time? A model that is right
    90% of the time and always says 0.99 has a large ECE and would quietly disable
    every ``on_uncertain`` rule in the policy.
    """
    n_bins = max(1, int(bins))
    total = len(labels)
    if total == 0:
        return {"ece": float("nan"), "mce": float("nan"), "bins": [], "n": 0, "bins_requested": n_bins}
    if len(probabilities) != total:
        raise EvaluateError(f"expected {total} distributions, got {len(probabilities)}")

    bucket_acc: list[list[float]] = [[] for _ in range(n_bins)]
    bucket_conf: list[list[float]] = [[] for _ in range(n_bins)]
    for distribution, label in zip(probabilities, labels, strict=True):
        values = [float(v) for v in distribution]
        if not values:
            continue
        predicted = max(range(len(values)), key=lambda i: (values[i], -i))
        confidence = max(values)
        index = min(n_bins - 1, int(confidence * n_bins))
        bucket_acc[index].append(1.0 if predicted == int(label) else 0.0)
        bucket_conf[index].append(confidence)

    ece = 0.0
    mce = 0.0
    curve: list[dict[str, Any]] = []
    for index in range(n_bins):
        count = len(bucket_acc[index])
        low = index / n_bins
        high = (index + 1) / n_bins
        if count == 0:
            curve.append({"bin": index, "range": [round(low, 4), round(high, 4)], "n": 0})
            continue
        accuracy = sum(bucket_acc[index]) / count
        confidence = sum(bucket_conf[index]) / count
        gap = abs(accuracy - confidence)
        ece += gap * count / total
        mce = max(mce, gap)
        curve.append(
            {
                "bin": index,
                "range": [round(low, 4), round(high, 4)],
                "n": count,
                "accuracy": round(accuracy, 6),
                "confidence": round(confidence, 6),
                "gap": round(gap, 6),
            }
        )
    return {
        "ece": round(ece, 6),
        "mce": round(mce, 6),
        "bins": curve,
        "n": total,
        "bins_requested": n_bins,
    }


def classification_metrics(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    label_names: Sequence[str],
    *,
    bins: int = 15,
) -> dict[str, Any]:
    """Accuracy, per-class precision/recall/F1, macro-F1, and the calibration block."""
    total = len(labels)
    n_classes = len(label_names)
    if total == 0 or n_classes == 0:
        return {
            "n": total, "accuracy": float("nan"), "macro_f1": float("nan"),
            "per_class": {}, "calibration": expected_calibration_error([], [], bins=bins),
            "prediction_support": {}, "label_support": {},
        }
    predicted = [max(range(len(p)), key=lambda i: (float(p[i]), -i)) for p in probabilities]
    true_positive = [0] * n_classes
    false_positive = [0] * n_classes
    support = [0] * n_classes
    correct = 0
    for pred, true in zip(predicted, labels, strict=True):
        support[true] += 1
        if pred == true:
            correct += 1
            true_positive[true] += 1
        else:
            false_positive[pred] += 1

    per_class: dict[str, Any] = {}
    f1s: list[float] = []
    for index, name in enumerate(label_names):
        tp, fp, sup = true_positive[index], false_positive[index], support[index]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / sup if sup else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[name] = {
            "support": sup,
            "predicted": tp + fp,
            "accuracy": round(recall, 6),
            "precision": round(precision, 6),
            "f1": round(f1, 6),
        }
        # macro-F1 over classes that appear on either side, which is sklearn's
        # default. Averaging over a class with zero support and zero predictions
        # would report a number that measures the dataset's shape, not the model.
        if sup or (tp + fp):
            f1s.append(f1)

    return {
        "n": total,
        "accuracy": round(correct / total, 6),
        "macro_f1": round(sum(f1s) / len(f1s), 6) if f1s else float("nan"),
        "per_class": per_class,
        "label_support": {name: support[i] for i, name in enumerate(label_names)},
        "prediction_support": {name: true_positive[i] + false_positive[i] for i, name in enumerate(label_names)},
        "calibration": expected_calibration_error(probabilities, labels, bins=bins),
    }


def teacher_calibration_error(
    student: Sequence[Sequence[float]],
    teacher: Sequence[Sequence[float]],
    *,
    bins: int = 15,
) -> dict[str, Any]:
    """Calibration of the student *against the teacher*, binned by student confidence.

    For each held-out row: take the class the student picked, its probability for
    that class, and the teacher's probability for the same class. Bin by the
    student's confidence and compare means. Zero means "when the student says 0.9,
    the teacher also puts 0.9 there" -- which is the property the whole pipeline
    exists to preserve.

    This is the calibration number that matters, and it is deliberately not the
    textbook ECE. Textbook ECE compares confidence against *accuracy*, and the only
    labels available here are the teacher's own argmax, so a perfect student scores
    100% accuracy by construction and its ECE collapses to ``1 - mean confidence``.
    That number measures how peaked the teacher is, not how faithful the student is.
    Both are reported; :attr:`EvaluationReport.worst_ece` uses this one.
    """
    n_bins = max(1, int(bins))
    total = len(student)
    if total == 0:
        return {"ece": float("nan"), "mce": float("nan"), "bins": [], "n": 0}
    bucket_conf: list[list[float]] = [[] for _ in range(n_bins)]
    bucket_ref: list[list[float]] = [[] for _ in range(n_bins)]
    for q_row, p_row in zip(student, teacher, strict=True):
        values = [float(v) for v in q_row]
        if not values:
            continue
        predicted = max(range(len(values)), key=lambda i: (values[i], -i))
        confidence = values[predicted]
        reference = float(p_row[predicted]) if predicted < len(p_row) else 0.0
        index = min(n_bins - 1, int(confidence * n_bins))
        bucket_conf[index].append(confidence)
        bucket_ref[index].append(reference)

    ece = mce = 0.0
    curve: list[dict[str, Any]] = []
    for index in range(n_bins):
        count = len(bucket_conf[index])
        low, high = index / n_bins, (index + 1) / n_bins
        if count == 0:
            continue
        confidence = sum(bucket_conf[index]) / count
        reference = sum(bucket_ref[index]) / count
        gap = abs(confidence - reference)
        ece += gap * count / total
        mce = max(mce, gap)
        curve.append(
            {
                "bin": index,
                "range": [round(low, 4), round(high, 4)],
                "n": count,
                "student_confidence": round(confidence, 6),
                "teacher_probability": round(reference, 6),
                "gap": round(gap, 6),
            }
        )
    return {"ece": round(ece, 6), "mce": round(mce, 6), "bins": curve, "n": total}


def distribution_fidelity(
    student: Sequence[Sequence[float]], teacher: Sequence[Sequence[float]]
) -> dict[str, float]:
    """How close the whole distribution is, not just the winner.

    ``kl`` is the training objective measured on held-out data (the number that
    says distillation worked); ``brier`` is a proper scoring rule a non-expert can
    read; ``tvd`` bounds how often the two models can disagree at all.
    """
    if not student:
        return {"mean_kl": float("nan"), "mean_brier": float("nan"), "mean_tvd": float("nan")}
    kl_total = brier_total = tvd_total = 0.0
    for q_row, p_row in zip(student, teacher, strict=True):
        kl = brier = tvd = 0.0
        for p, q in zip(p_row, q_row, strict=True):
            p = max(float(p), 1e-12)
            q = min(max(float(q), 1e-12), 1.0)
            kl += p * (math.log(p) - math.log(q))
            brier += (q - p) ** 2
            tvd += abs(q - p)
        kl_total += kl
        brier_total += brier
        tvd_total += 0.5 * tvd
    n = len(student)
    return {
        "mean_kl": round(kl_total / n, 6),
        "mean_brier": round(brier_total / n, 6),
        "mean_tvd": round(tvd_total / n, 6),
    }


# --------------------------------------------------------------------------- #
# Report structures
# --------------------------------------------------------------------------- #
def _json_safe(value: Any) -> Any:
    """Recursively replace NaN/Inf with None so reports are valid JSON.

    ``json.dumps`` will happily emit a bare ``NaN``, which is not JSON and breaks
    every downstream consumer (dashboards, CI uploaders, jq). A metric that could
    not be computed is reported as absent, not as a syntax error.
    """
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else round(value, 6)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return _json_safe(float(value)) if isinstance(value, (int, float)) else str(value)


@dataclass(frozen=True)
class HeadMetrics:
    """Everything measured for one output head."""

    head: str
    kind: str
    labels: tuple[str, ...]
    n: int
    accuracy: float
    macro_f1: float
    #: Calibration against the teacher's probabilities. The headline number.
    ece: float
    mce: float
    #: Textbook ECE against the teacher's argmax labels. Biased upward by
    #: construction (see :func:`teacher_calibration_error`); reported so the
    #: familiar number is available and so the teacher's own value next to it
    #: makes the bias visible.
    argmax_ece: float
    argmax_mce: float
    #: The teacher's own textbook ECE against its own argmax: the floor that
    #: ``argmax_ece`` cannot go below, no matter how good the student is.
    teacher_argmax_ece: float
    fidelity: dict[str, float]
    per_class: dict[str, Any]
    label_support: dict[str, int]
    prediction_support: dict[str, int]
    calibration_curve: tuple[dict[str, Any], ...]
    mean_confidence: float
    #: The teacher's reported confidence versus the confidence implied by its own
    #: distribution. :class:`~jev_route.schema.ChoiceAnswer` keeps both, and
    #: comparing them is how you find out whether a backend's self-reported
    #: number is worth thresholding on.
    teacher_confidence: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "head": self.head,
                "kind": self.kind,
                "labels": list(self.labels),
                "n": self.n,
                "accuracy": self.accuracy,
                "macro_f1": self.macro_f1,
                "ece": self.ece,
                "mce": self.mce,
                "argmax_ece": self.argmax_ece,
                "argmax_mce": self.argmax_mce,
                "teacher_argmax_ece": self.teacher_argmax_ece,
                "fidelity": self.fidelity,
                "per_class": self.per_class,
                "label_support": self.label_support,
                "prediction_support": self.prediction_support,
                "calibration_curve": list(self.calibration_curve),
                "mean_confidence": self.mean_confidence,
                "teacher_confidence": self.teacher_confidence,
            }
        )

    def line(self) -> str:
        return (
            f"{self.head:<12} n={self.n:<5} acc={self.accuracy:6.1%}  macroF1={self.macro_f1:6.3f}  "
            f"ECEvsTeacher={self.ece:6.4f}  ECEvsArgmax={self.argmax_ece:6.4f}  "
            f"teacherECE={self.teacher_argmax_ece:6.4f}  "
            f"KL={self.fidelity.get('mean_kl', float('nan')):7.4f}  "
            f"Brier={self.fidelity.get('mean_brier', float('nan')):6.4f}"
        )


@dataclass(frozen=True)
class AgreementMetrics:
    """Student versus teacher, at the label level and at the level the operator feels."""

    n_rows: int
    argmax_agreement: dict[str, float]
    argmax_counts: dict[str, tuple[int, int]]
    tier_agreement: float | None
    tier_counts: tuple[int, int] | None
    #: Tier agreement of a *hypothetically perfect* student: the teacher's own
    #: distributions replayed with derived confidence. Tier agreement cannot
    #: exceed this by much, so a student sitting at the ceiling is done and the
    #: remaining gap belongs to the policy's confidence thresholds.
    ceiling_tier_agreement: float | None
    #: Tier flips where every argmax label agreed -- i.e. caused purely by a
    #: confidence floor firing on one backend's scale and not the other's.
    confidence_only_flips: int
    tier_confusion: dict[str, int]
    rule_agreement: float | None
    boundary_error_rate: float | None
    boundary_errors: int
    pii_agreement: float | None
    teacher_tier_distribution: dict[str, int]
    student_tier_distribution: dict[str, int]
    #: Fraction of rows where replaying the teacher's own answers reproduces the
    #: tier that was logged. Below 1.0 means the policy used for evaluation is not
    #: the policy that produced the log, and every tier number here is suspect.
    replay_fidelity: float | None
    excluded_gate_rows: int
    policy_source: str

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "n_rows": self.n_rows,
                "argmax_agreement": self.argmax_agreement,
                "argmax_counts": {k: list(v) for k, v in self.argmax_counts.items()},
                "tier_agreement": self.tier_agreement,
                "tier_counts": list(self.tier_counts) if self.tier_counts else None,
                "ceiling_tier_agreement": self.ceiling_tier_agreement,
                "confidence_only_flips": self.confidence_only_flips,
                "tier_confusion": self.tier_confusion,
                "rule_agreement": self.rule_agreement,
                "boundary_error_rate": self.boundary_error_rate,
                "boundary_errors": self.boundary_errors,
                "pii_agreement": self.pii_agreement,
                "teacher_tier_distribution": self.teacher_tier_distribution,
                "student_tier_distribution": self.student_tier_distribution,
                "replay_fidelity": self.replay_fidelity,
                "excluded_gate_rows": self.excluded_gate_rows,
                "policy_source": self.policy_source,
            }
        )


@dataclass(frozen=True)
class LatencyMetrics:
    """Student inference cost, measured, next to the teacher's logged cost."""

    n: int
    p50: float
    p95: float
    p99: float
    mean: float
    minimum: float
    maximum: float
    teacher: dict[str, float]
    #: ``student p50 - teacher p50``. Negative means graduating removes latency.
    delta_p50_ms: float | None
    measured_on: str = "cpu"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "n": self.n,
                "p50": self.p50,
                "p95": self.p95,
                "p99": self.p99,
                "mean": self.mean,
                "min": self.minimum,
                "max": self.maximum,
                "teacher": self.teacher,
                "delta_p50_ms": self.delta_p50_ms,
                "measured_on": self.measured_on,
                "note": self.note,
            }
        )

    def line(self) -> str:
        teacher = self.teacher or {}
        return (
            f"student p50={self.p50:.3f}ms p95={self.p95:.3f}ms p99={self.p99:.3f}ms (n={self.n})  "
            f"teacher p50={_fmt(teacher.get('p50'), '.3f')}ms p95={_fmt(teacher.get('p95'), '.3f')}ms  "
            f"delta_p50={_fmt(self.delta_p50_ms, '+.3f')}ms"
        )


@dataclass(frozen=True)
class EvaluationReport:
    """The whole evaluation, structured. :func:`render_report` makes it readable."""

    model: dict[str, Any]
    dataset: dict[str, Any]
    heads: dict[str, HeadMetrics]
    agreement: AgreementMetrics | None
    latency: LatencyMetrics
    notes: tuple[str, ...] = ()
    generated_at: str = ""

    @property
    def worst_ece(self) -> float:
        """The headline calibration number: the worst head, not the average.

        Averaging hides the head that is broken, and one badly calibrated head is
        enough to break the policy rule that reads it. Uses calibration against the
        teacher's probabilities, which is the only unbiased option when the labels
        are the teacher's own (see :func:`teacher_calibration_error`).
        """
        values = [h.ece for h in self.heads.values() if h.n and not math.isnan(h.ece)]
        return max(values) if values else float("nan")

    @property
    def worst_ece_head(self) -> str | None:
        candidates = [(h.ece, name) for name, h in self.heads.items() if h.n and not math.isnan(h.ece)]
        return max(candidates)[1] if candidates else None

    @property
    def tier_agreement(self) -> float | None:
        return self.agreement.tier_agreement if self.agreement else None

    @property
    def n_rows(self) -> int:
        return int(self.dataset.get("rows", 0))

    def to_dict(self) -> dict[str, Any]:
        #: Heads that actually saw rows; an empty head has macro_f1 = nan and would
        #: poison the mean. max(1, ...) keeps the divisor safe when none did.
        scored = [head for head in self.heads.values() if head.n]
        mean_macro_f1 = sum(head.macro_f1 for head in scored) / max(1, len(scored))
        return _json_safe(
            {
                "report_schema": "jev_route.distill.evaluation/1",
                "generated_at": self.generated_at,
                "model": self.model,
                "dataset": self.dataset,
                "heads": {name: head.to_dict() for name, head in self.heads.items()},
                "agreement": self.agreement.to_dict() if self.agreement else None,
                "latency": self.latency.to_dict(),
                "summary": {
                    "worst_ece": self.worst_ece,
                    "worst_ece_head": self.worst_ece_head,
                    "tier_agreement": self.tier_agreement,
                    "mean_macro_f1": mean_macro_f1,
                    "student_p50_ms": self.latency.p50,
                },
                "notes": list(self.notes),
            }
        )


# --------------------------------------------------------------------------- #
# The driver
# --------------------------------------------------------------------------- #
def _resolve_target(target: Any, vectorizer: Any) -> tuple[StudentModel, Any, str, dict[str, Any]]:
    """Accept an artifact, a trained student, or a bare model + vectorizer."""
    if isinstance(target, DistilledArtifact):
        meta = {
            "model_version": target.model_version,
            "created_at": target.created_at,
            "artifact_schema_version": target.artifact_schema_version,
            "source": target.source,
            "teacher_model_versions": list(target.teacher_model_versions),
            **target.model.to_dict(),
        }
        return target.model, target.vectorizer, target.mode, meta
    model = getattr(target, "model", None)
    if model is None and isinstance(target, StudentModel):
        model, target_meta = target, {}
    elif model is None:
        raise EvaluateError(
            f"cannot evaluate {type(target).__name__}: expected a DistilledArtifact, a TrainedStudent, "
            "or a StudentModel (plus vectorizer=)"
        )
    else:
        target_meta = {}
    vec = vectorizer or getattr(target, "vectorizer", None)
    if vec is None:
        raise EvaluateError("evaluating a bare StudentModel needs vectorizer=; load an artifact instead")
    mode = str(getattr(target, "mode", "") or ("text" if vec.kind == "tfidf" else "features"))
    meta = {
        "model_version": str(getattr(target, "model_version", "") or "in-memory"),
        "source": getattr(target, "dataset_source", None),
        "teacher_model_versions": list(getattr(target, "teacher_model_versions", ()) or ()),
        **model.to_dict(),
        **target_meta,
    }
    return model, vec, mode, meta


def with_derived_confidence(answers: DecisionAnswers) -> DecisionAnswers:
    """Same distributions, but ``confidence`` recomputed the way the student reports it.

    A distilled answer's confidence is always
    :func:`~jev_route.schema.certainty_from_probabilities`, while a cloud backend
    may report its own (Jev does; the mock reports top-1 minus top-2). Those are
    *different functions of the same distribution*, so ``on_uncertain`` thresholds
    fire at different places for the two backends even when the student is a perfect
    copy. This is the single most common cause of tier disagreement after
    graduation, and it is a property of the policy, not a defect in the model -- so
    it gets measured separately, as the ceiling in :func:`_agreement`.
    """
    from dataclasses import replace

    from ..schema import ChoiceAnswer

    def derived(choice: ChoiceAnswer) -> ChoiceAnswer:
        return replace(
            choice,
            confidence=certainty_from_probabilities(choice.probabilities),
            confidence_reported=False,
        )

    return DecisionAnswers(
        complexity=derived(answers.complexity),
        sensitivity=derived(answers.sensitivity),
        pii=answers.pii,
        domain=derived(answers.domain),
    )


def teacher_answers_from_row(row: TrainingRow) -> DecisionAnswers:
    """Rebuild the teacher's :class:`DecisionAnswers` exactly as the router saw them.

    The argmax is the teacher's *raw* belief and the confidence is the *logged*
    confidence, because ``on_uncertain`` thresholds were applied to that number in
    production. Feeding a recomputed confidence through the replay would measure a
    policy the operator is not running.
    """
    from ..schema import ChoiceAnswer, NoulAnswer

    layout = {head.name: head.labels for head in _default_layout()}
    conf = dict(row.teacher.get("confidence") or {})
    reported = bool(conf.get("reported", False))

    def build(head: str) -> ChoiceAnswer:
        target = row.targets.get(head)
        labels = layout[head]
        probs = {label: float(target["probs"][label]) for label in labels}
        logged = conf.get(head)
        return ChoiceAnswer(
            choice=str(target["label"]),
            probabilities=probs,
            confidence=float(logged) if isinstance(logged, (int, float)) else certainty_from_probabilities(probs),
            confidence_reported=reported,
        )

    return DecisionAnswers(
        complexity=build("complexity"),
        sensitivity=build("sensitivity"),
        pii=NoulAnswer(value=float(row.targets.pii.get("value", 0.5))),
        domain=build("domain"),
    )


def _default_layout() -> tuple[Any, ...]:
    from .artifact import default_head_layout

    return default_head_layout()


def _head_metrics(
    head_name: str,
    spec: Any,
    rows: Sequence[TrainingRow],
    answers_list: Sequence[DecisionAnswers],
    *,
    bins: int,
) -> tuple[HeadMetrics, str | None]:
    """Every number for one output head, plus a note when the head had nothing to grade.

    Split out of :func:`evaluate` so that function reads as *resolve the target,
    grade each head, replay the policy, time it, report*. The pairing loop below is
    the delicate part: the student's distribution, the teacher's distribution and
    the teacher's argmax label for one row have to be appended together or every
    downstream metric silently grades row *i* against row *j*. Rows whose target is
    not usable for this head are skipped on all four lists at once, so they stay
    aligned. Returns ``(metrics, note)``; ``note`` is None unless the head had no
    usable rows at all.
    """
    labels = tuple(spec.labels)
    student_dists: list[list[float]] = []
    teacher_dists: list[list[float]] = []
    teacher_labels: list[int] = []
    #: Logged confidence (what the router thresholded on) and the confidence
    #: recomputed from the distribution, kept in lockstep so gaps align.
    logged_conf: list[float | None] = []
    computed_conf: list[float] = []
    confidence_reported: list[bool] = []
    student_conf: list[float] = []
    for row, answers in zip(rows, answers_list, strict=True):
        target = row.targets.get(head_name)
        if not target.get("usable", False):
            continue  # the teacher said nothing measurable about this row
        block = row.teacher.get("confidence") or {}
        raw_conf = block.get(head_name)
        reported = block.get("reported")
        flag = bool(reported.get(head_name)) if isinstance(reported, Mapping) else bool(reported)
        if head_name == "pii":
            value = float(answers.pii.value)
            s_dist = [1.0 - value, value]
            t_value = float(target["value"])
            t_dist = [1.0 - t_value, t_value]
            teacher_index = 1 if bool(target["label"]) else 0
            student_conf.append(float(answers.pii.confidence))
        else:
            choice = getattr(answers, head_name)
            s_dist = [float(choice.probabilities.get(label, 0.0)) for label in labels]
            t_dist = [float(target["probs"][label]) for label in labels]
            teacher_index = labels.index(str(target["label"]))
            student_conf.append(float(choice.confidence))
        student_dists.append(s_dist)
        teacher_dists.append(t_dist)
        teacher_labels.append(teacher_index)
        computed_conf.append(certainty_from_probabilities(dict(zip(labels, t_dist, strict=True))))
        logged_conf.append(float(raw_conf) if isinstance(raw_conf, (int, float)) else None)
        confidence_reported.append(flag)

    if not student_dists:
        empty = HeadMetrics(
            head=head_name, kind=spec.kind, labels=labels, n=0,
            accuracy=float("nan"), macro_f1=float("nan"),
            ece=float("nan"), mce=float("nan"),
            argmax_ece=float("nan"), argmax_mce=float("nan"), teacher_argmax_ece=float("nan"),
            fidelity={"mean_kl": float("nan"), "mean_brier": float("nan"), "mean_tvd": float("nan")},
            per_class={}, label_support={}, prediction_support={}, calibration_curve=(),
            mean_confidence=float("nan"),
            teacher_confidence={"n_reported": 0, "logged_mean": float("nan"),
                                "computed_mean": float("nan"), "mean_abs_gap": float("nan"),
                                "student_confidence_mean": float("nan")},
        )
        return empty, f"head {head_name!r} has no usable held-out rows; it could not be evaluated"

    metrics = classification_metrics(student_dists, teacher_labels, labels, bins=bins)
    argmax_calibration = metrics.pop("calibration")
    # The teacher graded against its own argmax: the floor the student's
    # argmax-ECE cannot beat, and the number that makes the bias visible.
    teacher_metrics = classification_metrics(teacher_dists, teacher_labels, labels, bins=bins)
    teacher_calibration = teacher_metrics["calibration"]
    against_teacher = teacher_calibration_error(student_dists, teacher_dists, bins=bins)

    pairs = [(a, b) for a, b in zip(logged_conf, computed_conf, strict=True) if a is not None]
    gaps = [abs(a - b) for a, b in pairs]
    metrics_obj = HeadMetrics(
        head=head_name,
        kind=spec.kind,
        labels=labels,
        n=metrics["n"],
        accuracy=metrics["accuracy"],
        macro_f1=metrics["macro_f1"],
        ece=against_teacher["ece"],
        mce=against_teacher["mce"],
        argmax_ece=argmax_calibration["ece"],
        argmax_mce=argmax_calibration["mce"],
        teacher_argmax_ece=teacher_calibration["ece"],
        fidelity=distribution_fidelity(student_dists, teacher_dists),
        per_class=metrics["per_class"],
        label_support=metrics["label_support"],
        prediction_support=metrics["prediction_support"],
        calibration_curve=tuple(against_teacher["bins"]),
        mean_confidence=(sum(student_conf) / len(student_conf)) if student_conf else float("nan"),
        teacher_confidence={
            "n_reported": sum(1 for flag in confidence_reported if flag),
            "logged_mean": (sum(a for a, _b in pairs) / len(pairs)) if pairs else float("nan"),
            "computed_mean": sum(computed_conf) / len(computed_conf),
            "mean_abs_gap": (sum(gaps) / len(gaps)) if gaps else float("nan"),
            "student_confidence_mean": (sum(student_conf) / len(student_conf)) if student_conf else float("nan"),
        },
    )
    return metrics_obj, None


def evaluate(
    target: Any,
    dataset: Dataset | str | Path,
    *,
    policy: Policy | str | Path | None = None,
    split: str | None = "holdout",
    bins: int = 15,
    latency_rows: int | None = None,
    latency_warmup: int = 5,
    measure_latency: bool = True,
    vectorizer: Any = None,
) -> EvaluationReport:
    """Evaluate a distilled model against the teacher on held-out rows.

    ``target`` may be an artifact directory / :class:`DistilledArtifact` /
    :class:`~jev_route.distill.train.TrainedStudent`. ``policy`` enables tier
    agreement; without it the label-level metrics are still computed, because
    accuracy and calibration do not depend on the policy and an operator should be
    able to see them before deciding which policy to graduate under.
    """
    from .artifact import utcnow

    model, vec, mode, model_meta = _resolve_target(target, vectorizer)
    if isinstance(dataset, (str, Path)):
        dataset = load_dataset(dataset)
    rows = list(dataset.rows if split is None else (r for r in dataset.rows if r.split == split))
    notes: list[str] = []

    if not rows:
        raise EvaluateError(
            f"no rows in split {split!r} of {dataset.source or 'the dataset'}; nothing to evaluate. "
            "Export with a non-zero --holdout-fraction, or pass split=None to evaluate every row."
        )

    usable = [row for row in rows if (row.text or "").strip() or mode == "features"]
    if mode == "text" and len(usable) != len(rows):
        notes.append(
            f"{len(rows) - len(usable)} row(s) carry no text and were skipped in text mode "
            "(the router never logs text for gate-blocked requests)"
        )
    rows = usable

    inputs: list[Any] = [(row.text or "") if mode == "text" else row.features for row in rows]
    answers_list = model.predict_answers(vec.transform(inputs))
    layout = {head.name: head for head in _default_layout()}

    # -- per-head metrics ------------------------------------------------- #
    heads: dict[str, HeadMetrics] = {}
    for head_name in ("complexity", "sensitivity", "domain", "pii"):
        head_metrics, head_note = _head_metrics(head_name, layout[head_name], rows, answers_list, bins=bins)
        heads[head_name] = head_metrics
        if head_note is not None:
            notes.append(head_note)

    # -- policy-level agreement ------------------------------------------- #
    agreement: AgreementMetrics | None = None
    resolved_policy = _resolve_policy(policy)
    if resolved_policy is not None:
        agreement = _agreement(resolved_policy, rows, answers_list, notes)
    else:
        notes.append(
            "no policy supplied: tier agreement was skipped. Pass policy= (or --policy) to measure the "
            "number that actually decides a cutover."
        )

    # -- latency ----------------------------------------------------------- #
    latency = _latency(
        model, vec, inputs, rows, latency_rows=latency_rows, warmup=latency_warmup, measure=measure_latency
    )

    return EvaluationReport(
        model=model_meta,
        dataset={
            "source": dataset.source,
            "mode": dataset.mode,
            "rows": len(rows),
            "split": split or "all",
            "sha256": str((dataset.stats or {}).get("dataset_sha256", "")),
            "teacher_model_versions": list(dataset.teacher_model_versions()),
        },
        heads=heads,
        agreement=agreement,
        latency=latency,
        notes=tuple(notes),
        generated_at=utcnow(),
    )


def _resolve_policy(policy: Policy | str | Path | None) -> Policy | None:
    if policy is None:
        return None
    if isinstance(policy, Policy):
        return policy
    return Policy.from_file(policy)


@dataclass
class _AgreementTally:
    """Running counters for one pass over the held-out rows.

    :func:`_agreement` measures a dozen different agreements over the *same* rows.
    Holding the counters in one mutable object is what lets the per-row work be
    split into two readable steps -- labels, then tiers -- instead of threading a
    dozen locals through both.
    """

    #: ``{head: [matches, pairs]}``: argmax agreement per head, gate rows included.
    agree_counts: dict[str, list[int]] = field(
        default_factory=lambda: {head: [0, 0] for head in (*CHOICE_HEADS, "pii")}
    )
    tier_confusion: dict[str, int] = field(default_factory=dict)
    teacher_tiers: dict[str, int] = field(default_factory=dict)
    student_tiers: dict[str, int] = field(default_factory=dict)
    tier_pairs: int = 0
    tier_matches: int = 0
    ceiling_matches: int = 0
    confidence_only_flips: int = 0
    rule_pairs: int = 0
    rule_matches: int = 0
    boundary_pairs: int = 0
    boundary_errors: int = 0
    pii_pairs: int = 0
    pii_matches: int = 0
    replay_matches: int = 0
    replay_total: int = 0
    gate_rows: int = 0


def _tally_argmax(
    tally: _AgreementTally,
    policy: Policy,
    row: TrainingRow,
    teacher: DecisionAnswers,
    student: DecisionAnswers,
) -> None:
    """Label-level agreement for one row: every choice head, then pii.

    Counted for *every* row including gate-blocked ones, because the argmax
    numbers describe the whole holdout while the tier numbers deliberately do not.
    """
    for head in CHOICE_HEADS:
        if not row.targets.usable(head):
            continue
        tally.agree_counts[head][1] += 1
        if str(getattr(teacher, head).choice) == str(getattr(student, head).choice):
            tally.agree_counts[head][0] += 1
    if row.targets.usable("pii"):
        # Two different agreements on purpose. The argmax one uses the schema's
        # 0.5 (is the answer "yes"?); the policy one uses the operator's
        # threshold (does the same rule fire?). They diverge whenever
        # pii_threshold is not 0.5, and only the second one changes routing.
        tally.agree_counts["pii"][1] += 1
        if teacher.pii.is_true() == student.pii.is_true():
            tally.agree_counts["pii"][0] += 1
        teacher_pii = teacher.pii.is_true(policy.pii_threshold)
        student_pii = student.pii.is_true(policy.pii_threshold)
        tally.pii_pairs += 1
        if teacher_pii == student_pii:
            tally.pii_matches += 1


def _tally_tiers(
    tally: _AgreementTally,
    policy: Policy,
    row: TrainingRow,
    teacher: DecisionAnswers,
    student: DecisionAnswers,
) -> None:
    """Tier-level agreement for one classified row: replay both sides through one policy."""
    teacher_replay = replay_row(policy, row, teacher)
    student_replay = replay_row(policy, row, student)
    # The ceiling: what tier agreement would be if the student reproduced the
    # teacher's distributions *exactly*, and therefore reported derived
    # confidence instead of the teacher's own. Any gap below this is the
    # student's fault; any gap in this number is the policy's.
    ceiling_replay = replay_row(policy, row, with_derived_confidence(teacher))
    logged_tier = str(row.teacher.get("tier", ""))
    if logged_tier:
        tally.replay_total += 1
        if logged_tier == teacher_replay.tier:
            tally.replay_matches += 1

    tally.tier_pairs += 1
    tally.teacher_tiers[teacher_replay.tier] = tally.teacher_tiers.get(teacher_replay.tier, 0) + 1
    tally.student_tiers[student_replay.tier] = tally.student_tiers.get(student_replay.tier, 0) + 1
    if ceiling_replay.tier == teacher_replay.tier:
        tally.ceiling_matches += 1
    if teacher_replay.tier == student_replay.tier:
        tally.tier_matches += 1
    else:
        key = f"{teacher_replay.tier} -> {student_replay.tier}"
        tally.tier_confusion[key] = tally.tier_confusion.get(key, 0) + 1
        if all(
            str(getattr(teacher, head).choice) == str(getattr(student, head).choice)
            for head in CHOICE_HEADS
        ):
            # Every label agrees, so the tier moved because a confidence floor
            # fired on one side and not the other.
            tally.confidence_only_flips += 1
    tally.rule_pairs += 1
    if teacher_replay.rule_id == student_replay.rule_id:
        tally.rule_matches += 1

    tally.boundary_pairs += 1
    teacher_strict = _sensitivity_index(teacher_replay.effective_sensitivity) >= _BOUNDARY_INDEX
    student_strict = _sensitivity_index(student_replay.effective_sensitivity) >= _BOUNDARY_INDEX
    if teacher_strict != student_strict:
        tally.boundary_errors += 1


def _agreement(
    policy: Policy, rows: Sequence[TrainingRow], answers_list: Sequence[DecisionAnswers], notes: list[str]
) -> AgreementMetrics:
    """Compare teacher and student where it matters: on the tier, through one policy."""
    tally = _AgreementTally()

    for row, student in zip(rows, answers_list, strict=True):
        teacher = teacher_answers_from_row(row)
        _tally_argmax(tally, policy, row, teacher, student)

        if not bool(row.context.get("classified", True)):
            # The router never calls a backend for a gate-blocked request, so both
            # sides produce the gate's answer by construction. Counting those rows
            # would inflate tier agreement with decisions the student never made.
            tally.gate_rows += 1
            continue

        _tally_tiers(tally, policy, row, teacher, student)

    if tally.replay_total and tally.replay_matches < tally.replay_total:
        notes.append(
            f"policy replay reproduced the logged tier on {tally.replay_matches}/{tally.replay_total} rows. "
            "The policy used for evaluation is not the policy that produced this log, so treat the tier "
            "numbers as approximate."
        )
    if tally.gate_rows:
        notes.append(
            f"{tally.gate_rows} gate-blocked row(s) were excluded from tier agreement: the router does not call any "
            "backend for them, so teacher and student would agree by construction."
        )
    return AgreementMetrics(
        n_rows=len(rows),
        argmax_agreement={
            head: (counts[0] / counts[1] if counts[1] else float("nan"))
            for head, counts in tally.agree_counts.items()
        },
        argmax_counts={head: tuple(counts) for head, counts in tally.agree_counts.items()},  # type: ignore[misc]
        tier_agreement=(tally.tier_matches / tally.tier_pairs) if tally.tier_pairs else None,
        tier_counts=(tally.tier_matches, tally.tier_pairs) if tally.tier_pairs else None,
        ceiling_tier_agreement=(tally.ceiling_matches / tally.tier_pairs) if tally.tier_pairs else None,
        confidence_only_flips=tally.confidence_only_flips,
        tier_confusion=dict(sorted(tally.tier_confusion.items(), key=lambda kv: -kv[1])),
        rule_agreement=(tally.rule_matches / tally.rule_pairs) if tally.rule_pairs else None,
        boundary_error_rate=(tally.boundary_errors / tally.boundary_pairs) if tally.boundary_pairs else None,
        boundary_errors=tally.boundary_errors,
        pii_agreement=(tally.pii_matches / tally.pii_pairs) if tally.pii_pairs else None,
        teacher_tier_distribution=tally.teacher_tiers,
        student_tier_distribution=tally.student_tiers,
        replay_fidelity=(tally.replay_matches / tally.replay_total) if tally.replay_total else None,
        excluded_gate_rows=tally.gate_rows,
        policy_source=str(policy.source or "in-memory policy"),
    )


def _sensitivity_index(level: str) -> int:
    try:
        return SENSITIVITY_LEVELS.index(level)
    except ValueError:
        return -1


def _latency(
    model: StudentModel,
    vectorizer: Any,
    inputs: Sequence[Any],
    rows: Sequence[TrainingRow],
    *,
    latency_rows: int | None,
    warmup: int,
    measure: bool,
) -> LatencyMetrics:
    """Measure the serve path one row at a time, because that is how it is served.

    Batching would produce a prettier number and a misleading one: the router calls
    ``decide()`` once per request, so the per-call cost -- vectorize plus forward
    pass -- is the thing that gets added to every request in production.
    """
    teacher_values = [float(r.teacher.get("latency_ms") or 0.0) for r in rows]
    teacher = percentiles([v for v in teacher_values if v > 0])
    note = ""
    if not measure or not inputs:
        samples = percentiles([])
        return LatencyMetrics(n=0, p50=float("nan"), p95=float("nan"), p99=float("nan"), mean=float("nan"),
                              minimum=float("nan"), maximum=float("nan"), teacher=teacher, delta_p50_ms=None,
                              note="latency not measured")
    limit = len(inputs) if latency_rows is None else max(1, min(int(latency_rows), len(inputs)))
    subset = list(inputs[:limit])
    for item in subset[: max(0, int(warmup))]:
        model.predict_answers(vectorizer.transform([item]))
    timings: list[float] = []
    for item in subset:
        started = time.perf_counter()
        model.predict_answers(vectorizer.transform([item]))
        timings.append((time.perf_counter() - started) * 1000.0)
    samples = percentiles(timings)
    delta = (samples["p50"] - teacher["p50"]) if teacher.get("n") else None
    if not teacher.get("n"):
        note = (
            "the teacher logged no latency (MockBackend reports 0ms), so delta_p50 is unavailable; "
            "with a real Jev log this is where the cloud round trip shows up"
        )
    return LatencyMetrics(
        n=samples["n"],
        p50=samples["p50"],
        p95=samples["p95"],
        p99=samples["p99"],
        mean=samples["mean"],
        minimum=min(timings) if timings else float("nan"),
        maximum=max(timings) if timings else float("nan"),
        teacher=teacher,
        delta_p50_ms=delta,
        note=note,
    )


def evaluate_artifact(
    artifact: str | Path | DistilledArtifact,
    dataset: Dataset | str | Path,
    **kwargs: Any,
) -> EvaluationReport:
    """Load an artifact from disk and evaluate it. The CLI's one-call entry point."""
    target = artifact if isinstance(artifact, DistilledArtifact) else load_artifact(artifact)
    return evaluate(target, dataset, **kwargs)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _fmt(value: Any, spec: str = ".3f", dash: str = "  n/a") -> str:
    if value is None:
        return dash
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return dash
    return format(number, spec)


def _render_head_detail(report: EvaluationReport) -> list[str]:
    """The ``--verbose`` per-head block: per-class quality, confidence, calibration bins.

    Split out of :func:`render_report` so that function stays one flat list of
    sections. Emitted only on request because it is three lines per class per head,
    which buries the headline numbers an operator decides on.
    """
    lines: list[str] = []
    for name, head in report.heads.items():
        if not head.per_class:
            continue
        lines.append("")
        lines.append(f"  {name}: per class")
        for label, entry in head.per_class.items():
            lines.append(
                f"    {label:<16} support={entry['support']:<5} predicted={entry['predicted']:<5} "
                f"acc={entry['accuracy']:.3f} prec={entry['precision']:.3f} f1={entry['f1']:.3f}"
            )
        tc = head.teacher_confidence
        lines.append(
            f"    confidence: teacher logged={_fmt(tc.get('logged_mean'))} "
            f"teacher recomputed={_fmt(tc.get('computed_mean'))} gap={_fmt(tc.get('mean_abs_gap'))} "
            f"(backend-reported on {tc.get('n_reported', 0)}/{head.n} rows); "
            f"student={_fmt(tc.get('student_confidence_mean'))}"
        )
        if head.calibration_curve:
            lines.append(f"    calibration vs teacher ({len(head.calibration_curve)} non-empty bins):")
            for entry in head.calibration_curve:
                if not entry.get("n"):
                    continue
                lines.append(
                    f"      conf [{entry['range'][0]:.2f},{entry['range'][1]:.2f}) n={entry['n']:<5} "
                    f"student={entry['student_confidence']:.3f} teacher={entry['teacher_probability']:.3f} "
                    f"gap={entry['gap']:.3f}"
                )
    return lines


def _render_agreement(agreement: AgreementMetrics) -> list[str]:
    """The teacher-agreement section, in the order the numbers should be read.

    Tier agreement first because it is what the operator feels, then the two
    failure modes that hide behind it: rows crossing the ``confidential``
    boundary, and tiers that moved only because a confidence floor fired.
    """
    lines: list[str] = ["", f"agreement with the teacher (policy: {agreement.policy_source})"]
    lines.append(
        "  argmax  : "
        + "  ".join(f"{head} {_fmt(rate, '.1%')}" for head, rate in agreement.argmax_agreement.items())
    )
    tier = agreement.tier_counts
    lines.append(
        f"  tier    : {_fmt(agreement.tier_agreement, '.1%')}"
        + (f" ({tier[0]}/{tier[1]})" if tier else "")
        + f"   rule: {_fmt(agreement.rule_agreement, '.1%')}"
    )
    lines.append(
        f"  sensitivity boundary errors: {agreement.boundary_errors} "
        f"({_fmt(agreement.boundary_error_rate, '.2%')})  <-- rows moved across the confidential line"
    )
    lines.append(f"  pii threshold agreement    : {_fmt(agreement.pii_agreement, '.1%')}")
    if agreement.tier_confusion:
        lines.append(
            "  tier changes: "
            + ", ".join(f"{k} x{v}" for k, v in list(agreement.tier_confusion.items())[:6])
        )
    lines.append(
        f"  teacher tiers {agreement.teacher_tier_distribution} -> student tiers "
        f"{agreement.student_tier_distribution}"
    )
    lines.append(f"  policy replay fidelity: {_fmt(agreement.replay_fidelity, '.1%')}")
    return lines


def render_report(report: EvaluationReport, *, verbose: bool = False) -> str:
    """A report an operator can read once and make a decision from.

    Ordered by how much each number should influence that decision: quality per
    head, then calibration (the reason to have used Jev at all), then agreement at
    the tier level (what the operator actually feels), then cost.
    """
    model = report.model or {}
    data = report.dataset or {}
    lines: list[str] = []
    lines.append("distillation evaluation")
    lines.append("=======================")
    lines.append(
        f"student     : {model.get('model_version', '?')}  mode={data.get('mode', '?')}  "
        f"{model.get('family', '?')}  features={model.get('n_features', '?')}  params={model.get('n_parameters', '?')}"
    )
    teachers = data.get("teacher_model_versions") or model.get("teacher_model_versions") or []
    lines.append(f"teacher     : {', '.join(str(t) for t in teachers) or 'unknown'}")
    lines.append(
        f"dataset     : {data.get('rows', 0)} held-out rows "
        f"(split={data.get('split', '?')}) from {data.get('source', '?')}"
    )

    lines.append("")
    lines.append("per-head quality (labels are the teacher's -- this measures imitation, not truth)")
    lines.append("  head           n     acc   macroF1  ECE(vs tea)  ECE(vs argmax)  tea ECE  KL(tea||stu)   Brier")
    for name, head in report.heads.items():
        lines.append(
            f"  {name:<12} {head.n:>5}  {_fmt(head.accuracy, '6.1%')}  {_fmt(head.macro_f1, '7.3f')}  "
            f"{_fmt(head.ece, '10.4f')}  {_fmt(head.argmax_ece, '13.4f')}  {_fmt(head.teacher_argmax_ece, '7.4f')}  "
            f"{_fmt(head.fidelity.get('mean_kl'), '10.4f')}  {_fmt(head.fidelity.get('mean_brier'), '7.4f')}"
        )
    lines.append(
        f"  worst ECE {_fmt(report.worst_ece, '.4f')} on head {report.worst_ece_head!r} -- "
        "the headline number: an accurate but overconfident student silently disables every on_uncertain rule."
    )
    lines.append(
        "  ECE(vs tea) bins by the student's confidence and compares it to the teacher's probability for the"
    )
    lines.append(
        "  same class, so 0 means the student is as sure as the teacher was. ECE(vs argmax) is the textbook"
    )
    lines.append(
        "  number against the teacher's own argmax; it is biased upward (a perfect student scores 100%"
    )
    lines.append(
        "  accuracy by construction), which is why `tea ECE` -- the teacher graded the same way -- sits next "
        "to it."
    )

    if verbose:
        lines.extend(_render_head_detail(report))

    if report.agreement is not None:
        lines.extend(_render_agreement(report.agreement))
    else:
        lines.append("")
        lines.append("agreement : skipped (no policy supplied)")

    lines.append("")
    lines.append("latency (measured one request at a time, the way the router calls it)")
    lines.append(f"  {report.latency.line()}")
    if report.latency.note:
        lines.append(f"  note: {report.latency.note}")

    if report.notes:
        lines.append("")
        lines.append("notes")
        lines.extend(f"  - {note}" for note in report.notes)
    return "\n".join(lines)


__all__ = [
    "AgreementMetrics",
    "EvaluateError",
    "EvaluationReport",
    "HeadMetrics",
    "LatencyMetrics",
    "PolicyReplay",
    "classification_metrics",
    "distribution_fidelity",
    "evaluate",
    "evaluate_artifact",
    "expected_calibration_error",
    "render_report",
    "replay_policy",
    "replay_row",
    "teacher_answers_from_row",
]
