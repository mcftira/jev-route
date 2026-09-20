"""Calibration metrics for jev-route's decision heads.

Why this module exists separately from the accuracy numbers in ``run_eval.py``:
accuracy answers "was the argmax right?", which is the question an *uncalibrated*
classifier can also answer. The entire reason jev-route routes on a System One
model instead of a prompted LLM or a regex is that the answer arrives with a
probability distribution, and the policy engine acts on the *confidence* --
``on_uncertain`` bumps sensitivity stricter when the model is 70% sure rather than
99% sure. That behaviour is only worth anything if the confidence means what it
says. This module measures whether it does.

Four measurements, in the order they matter:

``expected_calibration_error``
    Weighted mean |confidence - accuracy| over equal-width confidence bins. The
    single number that says "0.8 means 80%".
``reliability_diagram``
    The same computation exposed bin by bin, so the report can show *where* the
    model is over- or under-confident instead of hiding it in one scalar.
``brier_score``
    A proper scoring rule over the full distribution, not just the top class.
    ECE only looks at the argmax; Brier punishes a confident wrong distribution
    even when the argmax happened to be right. Reported with a skill score
    against the class-prior baseline, because a raw Brier is not comparable
    between heads with different numbers of classes or different label balance.
``coverage_curve``
    Accuracy as a function of a confidence threshold. This is the empirical
    justification (or refutation) of ``on_uncertain``: if accuracy at
    confidence >= 0.8 is not meaningfully higher than accuracy overall, then the
    escalation policy is paying for noise.

Conventions, stated because they change the numbers:

* **Multiclass ECE uses top-label confidence.** For a head with K classes we bin
  on ``max(probs)`` (or on the backend's own reported ``confidence`` when the
  caller supplies one) and score a sample as correct when the argmax equals the
  gold label. This is the standard formulation and it is the one that matches how
  the policy engine consumes the answer.
* **The binary PII head is reported both ways.** ``confidence = max(p, 1-p)``
  makes it comparable with the choice heads; an event-based curve (``p`` against
  the ``yes`` rate) is also produced, because for a yes/no detector "calibrated"
  most naturally means "when it says 0.7, the thing is there 70% of the time".
* **Empty bins are skipped, not counted as zero error.** A bin with no samples
  carries no evidence. ``n_bins`` is therefore an upper bound on the number of
  terms, and the returned ``populated_bins`` tells you how many actually
  contributed. With 223 rows and coarse reported confidences, a 15-bin ECE can
  easily rest on 4 populated bins -- and a report that hides that is lying.

Stdlib only. No numpy, no matplotlib: the markdown table must stand alone, and a
calibration number that needs a plotting stack to be believed is not a number.
matplotlib is imported lazily and opportunistically by ``run_eval.py`` if present.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: Default bin count. 15 is wide enough to show shape on a few hundred rows and
#: narrow enough that most bins still hold samples.
DEFAULT_BINS = 15

#: Thresholds used for the risk/coverage curve. 0.0 is "everything" and is
#: always present so the curve has an honest left edge.
DEFAULT_COVERAGE_THRESHOLDS: tuple[float, ...] = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


# --------------------------------------------------------------------------- #
# Sample container
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CalibrationSample:
    """One (prediction, gold label) pair with the distribution behind it.

    ``probabilities`` is the full distribution, never just the argmax, so the
    same object can serve ECE (needs top confidence), Brier (needs everything),
    and the reliability diagram (needs the bin). ``confidence`` is what the
    *backend* reported, when it reported one; ``top_probability`` is what the
    distribution implies. They disagree, and the disagreement is itself a result
    worth printing, so both are carried.
    """

    gold: str
    prediction: str
    probabilities: Mapping[str, float]
    #: Backend-reported confidence, or ``None`` when the backend did not supply one.
    confidence: float | None = None
    #: Identifier for tracing a sample back to a dataset row in the JSONL.
    row_id: str = ""

    @property
    def top_probability(self) -> float:
        return max(self.probabilities.values()) if self.probabilities else 0.0

    @property
    def correct(self) -> bool:
        return self.prediction == self.gold

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.probabilities)

    def one_hot(self) -> dict[str, float]:
        return {k: (1.0 if k == self.gold else 0.0) for k in self.probabilities}


@dataclass(frozen=True)
class Bin:
    """One confidence bin of a reliability diagram."""

    lo: float
    hi: float
    count: int
    mean_confidence: float
    accuracy: float
    #: accuracy - mean_confidence. Negative = over-confident, the dangerous side
    #: for a data-residency router, because it is the side that routes sensitive
    #: text to a cloud model on the strength of a confidence that was not earned.
    gap: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "lo": round(self.lo, 6),
            "hi": round(self.hi, 6),
            "count": self.count,
            "mean_confidence": round(self.mean_confidence, 6),
            "accuracy": round(self.accuracy, 6),
            "gap": round(self.gap, 6),
        }


@dataclass(frozen=True)
class CalibrationReport:
    """Everything the report renderer needs for one head."""

    head: str
    n: int
    accuracy: float
    ece: float
    ece_top_probability: float
    mce: float
    brier: float
    #: Brier skill score against the class-prior (climatological) baseline.
    #: ``> 0`` means the distribution beats "always predict the marginal".
    brier_skill: float
    baseline_brier: float
    n_bins: int
    populated_bins: int
    bins: tuple[Bin, ...]
    coverage: tuple[dict[str, Any], ...]
    #: Confidence actually used for ``ece``: ``"reported"`` or ``"top_probability"``.
    confidence_source: str
    #: Mean reported confidence, when the backend supplied one. Useful because a
    #: head that is accurate but *under*-confident will trigger ``on_uncertain``
    #: constantly and silently make the router expensive.
    mean_reported_confidence: float | None = None
    #: Binary-only: event-based reliability of P(yes). Empty for choice heads.
    event_bins: tuple[Bin, ...] = ()
    event_ece: float = 0.0
    notes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CalibrationReport:
        """Rebuild a report from its JSON form.

        Exists so the renderer and the tests can work from the persisted
        ``summary.json`` without re-running anything: a report that can only be
        drawn from live objects cannot be redrawn by whoever reads the results
        file six months later.
        """
        return cls(
            head=str(data.get("head", "")),
            n=int(data.get("n", 0)),
            accuracy=float(data.get("accuracy", 0.0)),
            ece=float(data.get("ece", 0.0)),
            ece_top_probability=float(data.get("ece_top_probability", 0.0)),
            mce=float(data.get("mce", 0.0)),
            brier=float(data.get("brier", 0.0)),
            brier_skill=float(data.get("brier_skill", 0.0)),
            baseline_brier=float(data.get("baseline_brier", 0.0)),
            n_bins=int(data.get("n_bins", DEFAULT_BINS)),
            populated_bins=int(data.get("populated_bins", 0)),
            bins=tuple(Bin(**b) for b in data.get("bins", ())),
            coverage=tuple(data.get("coverage", ())),
            confidence_source=str(data.get("confidence_source", "")),
            mean_reported_confidence=data.get("mean_reported_confidence"),
            event_bins=tuple(Bin(**b) for b in data.get("event_bins", ())),
            event_ece=float(data.get("event_ece", 0.0)),
            notes=tuple(data.get("notes", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "head": self.head,
            "n": self.n,
            "accuracy": round(self.accuracy, 6),
            "ece": round(self.ece, 6),
            "ece_top_probability": round(self.ece_top_probability, 6),
            "mce": round(self.mce, 6),
            "brier": round(self.brier, 6),
            "brier_skill": round(self.brier_skill, 6),
            "baseline_brier": round(self.baseline_brier, 6),
            "n_bins": self.n_bins,
            "populated_bins": self.populated_bins,
            "confidence_source": self.confidence_source,
            "mean_reported_confidence": (
                None if self.mean_reported_confidence is None else round(self.mean_reported_confidence, 6)
            ),
            "bins": [b.to_dict() for b in self.bins],
            "event_bins": [b.to_dict() for b in self.event_bins],
            "event_ece": round(self.event_ece, 6),
            "coverage": list(self.coverage),
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------- #
# Core metrics
# --------------------------------------------------------------------------- #
def _bin_index(confidence: float, n_bins: int) -> int:
    """Equal-width bin for a confidence in ``[0, 1]``, clamped at the top edge.

    ``1.0`` must land in the last bin, not in a nonexistent bin ``n_bins``, which
    is the classic off-by-one that silently drops every perfectly-confident
    sample from the ECE sum and makes the model look better calibrated than it is.
    """
    clamped = min(1.0, max(0.0, float(confidence)))
    return min(n_bins - 1, int(clamped * n_bins))


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], *, n_bins: int = DEFAULT_BINS
) -> tuple[float, float, int]:
    """Top-label ECE. Returns ``(ece, mce, populated_bins)``.

    ``ece`` weights each bin by its share of samples, so a bin holding two rows
    cannot dominate a bin holding eighty. ``mce`` is the largest single-bin
    |gap|, reported alongside because a good mean can hide one terrible bin --
    and with a few hundred rows the mean is often resting on very few samples.

    Empty bins contribute nothing and are not counted as perfectly calibrated.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if len(confidences) != len(correct):
        raise ValueError(f"confidences and correct must be the same length, got {len(confidences)} and {len(correct)}")
    total = len(confidences)
    if total == 0:
        return 0.0, 0.0, 0

    conf_sum = [0.0] * n_bins
    hit_sum = [0.0] * n_bins
    counts = [0] * n_bins
    for conf, ok in zip(confidences, correct, strict=True):
        idx = _bin_index(conf, n_bins)
        conf_sum[idx] += float(conf)
        hit_sum[idx] += 1.0 if ok else 0.0
        counts[idx] += 1

    ece = 0.0
    mce = 0.0
    populated = 0
    for idx in range(n_bins):
        if counts[idx] == 0:
            continue
        populated += 1
        mean_conf = conf_sum[idx] / counts[idx]
        acc = hit_sum[idx] / counts[idx]
        gap = abs(acc - mean_conf)
        ece += gap * counts[idx] / total
        mce = max(mce, gap)
    return ece, mce, populated


def reliability_diagram(
    confidences: Sequence[float], correct: Sequence[bool], *, n_bins: int = DEFAULT_BINS
) -> tuple[Bin, ...]:
    """Per-bin predicted confidence vs observed accuracy.

    Returned as data rather than as a plot so the report can render it as a
    markdown table that survives being pasted into a README, a terminal, or a
    pull request comment.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    conf_sum = [0.0] * n_bins
    hit_sum = [0.0] * n_bins
    counts = [0] * n_bins
    for conf, ok in zip(confidences, correct, strict=True):
        idx = _bin_index(conf, n_bins)
        conf_sum[idx] += float(conf)
        hit_sum[idx] += 1.0 if ok else 0.0
        counts[idx] += 1

    out: list[Bin] = []
    for idx in range(n_bins):
        lo = idx / n_bins
        hi = (idx + 1) / n_bins
        if counts[idx] == 0:
            out.append(Bin(lo=lo, hi=hi, count=0, mean_confidence=0.0, accuracy=0.0, gap=0.0))
            continue
        mean_conf = conf_sum[idx] / counts[idx]
        acc = hit_sum[idx] / counts[idx]
        out.append(Bin(lo=lo, hi=hi, count=counts[idx], mean_confidence=mean_conf, accuracy=acc, gap=acc - mean_conf))
    return tuple(out)


def brier_score(samples: Sequence[CalibrationSample], classes: Sequence[str]) -> float:
    """Multiclass Brier score: mean squared error over the full distribution.

    ``sum_k (p_k - y_k)^2`` averaged over samples. Reported per-sample-normalised
    (not divided by K) so it is directly comparable with the binary form, which
    is the K=2 case of the same expression. Range ``[0, 2]``; lower is better.
    """
    if not samples:
        return 0.0
    total = 0.0
    for sample in samples:
        one_hot = {k: (1.0 if k == sample.gold else 0.0) for k in classes}
        probs = sample.probabilities
        total += sum((float(probs.get(k, 0.0)) - one_hot[k]) ** 2 for k in classes)
    return total / len(samples)


def baseline_brier(samples: Sequence[CalibrationSample], classes: Sequence[str]) -> float:
    """Brier score of the marginal (class-prior) predictor.

    The reference a real distribution has to beat. Without it, "Brier = 0.31"
    means nothing: on a head whose gold label is 60% one class, predicting the
    prior already scores about 0.44, and on a balanced four-class head the
    uniform predictor scores 0.75. ``brier_skill = 1 - brier / baseline`` puts
    every head on the same scale, where 0 is "no better than the marginal" and
    1 is perfect.
    """
    if not samples:
        return 0.0
    counts = dict.fromkeys(classes, 0)
    for sample in samples:
        if sample.gold in counts:
            counts[sample.gold] += 1
    n = len(samples)
    prior = {k: counts[k] / n for k in classes}
    total = 0.0
    for sample in samples:
        one_hot = {k: (1.0 if k == sample.gold else 0.0) for k in classes}
        total += sum((prior[k] - one_hot[k]) ** 2 for k in classes)
    return total / n


def coverage_curve(
    confidences: Sequence[float],
    correct: Sequence[bool],
    thresholds: Sequence[float] = DEFAULT_COVERAGE_THRESHOLDS,
) -> tuple[dict[str, Any], ...]:
    """Accuracy on the subset of samples at or above each confidence threshold.

    This is the measurement that decides whether ``on_uncertain`` is worth its
    cost. Escalating on low confidence is only rational if the low-confidence
    subset is genuinely less accurate -- i.e. if accuracy *rises* as the threshold
    rises. If the curve is flat, the policy is spending money on a confidence
    signal that carries no information, and the report should say so.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")
    total = len(confidences)
    pairs = sorted(zip(confidences, correct, strict=True), key=lambda p: -p[0])
    out: list[dict[str, Any]] = []
    for threshold in sorted(set(thresholds)):
        kept = [(c, ok) for c, ok in pairs if c >= threshold]
        n = len(kept)
        acc = (sum(1 for _, ok in kept if ok) / n) if n else None
        out.append(
            {
                "threshold": round(float(threshold), 6),
                "n": n,
                "coverage": round(n / total, 6) if total else 0.0,
                "accuracy": None if acc is None else round(acc, 6),
            }
        )
    return tuple(out)


def binary_event_curve(
    probabilities: Sequence[float], positives: Sequence[bool], *, n_bins: int = 10
) -> tuple[tuple[Bin, ...], float]:
    """Event-based reliability for a yes/no head: bin ``p(yes)`` against observed yes-rate.

    Distinct from :func:`reliability_diagram`, which bins on ``max(p, 1-p)`` and
    therefore cannot distinguish "confidently yes" from "confidently no". For a
    PII detector the asymmetry matters: over-confidence in *no* is the failure
    that leaks data, and only this view shows it.
    """
    if len(probabilities) != len(positives):
        raise ValueError("probabilities and positives must be the same length")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    total = len(probabilities)
    p_sum = [0.0] * n_bins
    hit = [0.0] * n_bins
    counts = [0] * n_bins
    for p, y in zip(probabilities, positives, strict=True):
        idx = _bin_index(p, n_bins)
        p_sum[idx] += float(p)
        hit[idx] += 1.0 if y else 0.0
        counts[idx] += 1
    bins: list[Bin] = []
    ece = 0.0
    for idx in range(n_bins):
        lo = idx / n_bins
        hi = (idx + 1) / n_bins
        if counts[idx] == 0:
            bins.append(Bin(lo=lo, hi=hi, count=0, mean_confidence=0.0, accuracy=0.0, gap=0.0))
            continue
        mean_p = p_sum[idx] / counts[idx]
        rate = hit[idx] / counts[idx]
        bins.append(Bin(lo=lo, hi=hi, count=counts[idx], mean_confidence=mean_p, accuracy=rate, gap=rate - mean_p))
        ece += abs(rate - mean_p) * counts[idx] / total if total else 0.0
    return tuple(bins), ece


# --------------------------------------------------------------------------- #
# Head-level driver
# --------------------------------------------------------------------------- #
def calibrate_head(
    head: str,
    samples: Sequence[CalibrationSample],
    classes: Sequence[str],
    *,
    n_bins: int = DEFAULT_BINS,
    coverage_thresholds: Sequence[float] = DEFAULT_COVERAGE_THRESHOLDS,
    prefer_reported_confidence: bool = True,
) -> CalibrationReport:
    """Full calibration report for one choice head.

    ``prefer_reported_confidence`` decides which confidence the headline ECE is
    computed on. It defaults to the backend-reported value because that is the
    number the policy engine thresholds against -- measuring the calibration of a
    confidence the router never reads would be measuring the wrong thing. When a
    backend does not report one (MockBackend derives it from the distribution;
    the distilled backend computes normalized entropy) we fall back to
    ``max(probs)`` and record which was used, so a reader never mistakes one for
    the other.
    """
    notes: list[str] = []
    if not samples:
        return CalibrationReport(
            head=head,
            n=0,
            accuracy=0.0,
            ece=0.0,
            ece_top_probability=0.0,
            mce=0.0,
            brier=0.0,
            brier_skill=0.0,
            baseline_brier=0.0,
            n_bins=n_bins,
            populated_bins=0,
            bins=(),
            coverage=(),
            confidence_source="none",
            notes=("no samples",),
        )

    reported = [s.confidence for s in samples if s.confidence is not None]
    use_reported = prefer_reported_confidence and len(reported) == len(samples)
    if not use_reported and reported:
        notes.append(
            f"only {len(reported)}/{len(samples)} samples carried a reported confidence; "
            "falling back to max(probabilities)"
        )
    confidence_source = "reported" if use_reported else "top_probability"

    confidences = [(float(s.confidence) if use_reported else s.top_probability) for s in samples]
    correct = [s.correct for s in samples]

    ece, mce, populated = expected_calibration_error(confidences, correct, n_bins=n_bins)
    top_conf = [s.top_probability for s in samples]
    ece_top, _mce_top, _pop_top = expected_calibration_error(top_conf, correct, n_bins=n_bins)
    bins = reliability_diagram(confidences, correct, n_bins=n_bins)

    brier = brier_score(samples, classes)
    base = baseline_brier(samples, classes)
    skill = (1.0 - brier / base) if base > 0 else 0.0

    coverage = coverage_curve(confidences, correct, coverage_thresholds)
    accuracy = sum(1 for ok in correct if ok) / len(correct)

    if populated < max(3, n_bins // 3):
        notes.append(f"ECE rests on only {populated} populated bin(s) of {n_bins}; treat it as indicative, not precise")
    mean_reported = (sum(reported) / len(reported)) if reported else None

    return CalibrationReport(
        head=head,
        n=len(samples),
        accuracy=accuracy,
        ece=ece,
        ece_top_probability=ece_top,
        mce=mce,
        brier=brier,
        brier_skill=skill,
        baseline_brier=base,
        n_bins=n_bins,
        populated_bins=populated,
        bins=bins,
        coverage=coverage,
        confidence_source=confidence_source,
        mean_reported_confidence=mean_reported,
        notes=tuple(notes),
    )


def calibrate_binary_head(
    head: str,
    probabilities: Sequence[float],
    gold: Sequence[bool],
    *,
    threshold: float = 0.5,
    n_bins: int = DEFAULT_BINS,
    event_bins: int = 10,
    coverage_thresholds: Sequence[float] = DEFAULT_COVERAGE_THRESHOLDS,
) -> CalibrationReport:
    """Calibration for a noul (yes/no) head such as ``pii_present``.

    Two views are produced from the same data because they answer different
    questions. The top-label view (``bins``, ``ece``) treats the head as a
    two-class choice so its ECE is comparable with complexity/sensitivity/domain.
    The event view (``event_bins``, ``event_ece``) asks the question an operator
    actually cares about: when the model says P(PII) = 0.7, is PII there 70% of
    the time? Brier here is the binary form ``mean((p - y)^2)``, range ``[0, 1]``,
    with the marginal predictor as its baseline.
    """
    if len(probabilities) != len(gold):
        raise ValueError("probabilities and gold must be the same length")
    n = len(probabilities)
    if n == 0:
        return CalibrationReport(
            head=head,
            n=0,
            accuracy=0.0,
            ece=0.0,
            ece_top_probability=0.0,
            mce=0.0,
            brier=0.0,
            brier_skill=0.0,
            baseline_brier=0.0,
            n_bins=n_bins,
            populated_bins=0,
            bins=(),
            coverage=(),
            confidence_source="abs(2p-1)",
            notes=("no samples",),
        )

    # Top-label view: confidence is distance from the coin flip, prediction is
    # the thresholded value. Mirrors NoulAnswer.confidence in the schema so the
    # number measured is the number the policy thresholds.
    confidences = [abs(2.0 * float(p) - 1.0) for p in probabilities]
    predictions = [float(p) >= threshold for p in probabilities]
    correct = [pred == g for pred, g in zip(predictions, gold, strict=True)]

    ece, mce, populated = expected_calibration_error(confidences, correct, n_bins=n_bins)
    bins = reliability_diagram(confidences, correct, n_bins=n_bins)
    coverage = coverage_curve(confidences, correct, coverage_thresholds)

    # Binary Brier + marginal baseline.
    brier = sum((float(p) - (1.0 if g else 0.0)) ** 2 for p, g in zip(probabilities, gold, strict=True)) / n
    rate = sum(1 for g in gold if g) / n
    base = (rate * (1.0 - rate) ** 2) + ((1.0 - rate) * rate**2)
    skill = (1.0 - brier / base) if base > 0 else 0.0

    ev_bins, ev_ece = binary_event_curve(probabilities, gold, n_bins=event_bins)

    notes = [
        f"top-label view bins on |2p-1| with a decision threshold of {threshold}; "
        "event view bins p(yes) against the observed yes-rate",
    ]
    if populated < max(3, n_bins // 3):
        notes.append(f"ECE rests on only {populated} populated bin(s) of {n_bins}")

    return CalibrationReport(
        head=head,
        n=n,
        accuracy=sum(1 for ok in correct if ok) / n,
        ece=ece,
        ece_top_probability=ece,
        mce=mce,
        brier=brier,
        brier_skill=skill,
        baseline_brier=base,
        n_bins=n_bins,
        populated_bins=populated,
        bins=bins,
        coverage=coverage,
        confidence_source="abs(2p-1)",
        mean_reported_confidence=sum(confidences) / n,
        event_bins=ev_bins,
        event_ece=ev_ece,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------- #
# ASCII rendering -- the table must stand alone without a plotting stack
# --------------------------------------------------------------------------- #
def render_reliability_table(
    report: CalibrationReport, *, width: int = 22, event: bool = False, include_empty: bool = False
) -> str:
    """Render a reliability diagram as a markdown table with an ASCII bar.

    The bar shows observed accuracy; the ``^`` marker shows predicted confidence
    on the same scale. Where the marker sits right of the bar, the head is
    over-confident. That is readable in a terminal, in a diff, and in a GitHub
    comment, which a PNG is not.
    """
    bins = report.event_bins if event else report.bins
    title = (
        f"{report.head}: predicted p(yes) vs observed yes-rate"
        if event
        else f"{report.head}: predicted confidence vs observed accuracy"
    )
    lines = [
        title,
        "",
        "| bin | n | predicted | observed | gap | ``observed`` (``^`` = predicted) |",
        "| --- | ---: | ---: | ---: | ---: | :--- |",
    ]
    for b in bins:
        if b.count == 0:
            # Empty bins are skipped by default: on a few hundred rows with a
            # coarse reported confidence, most bins hold nothing, and printing
            # thirteen blank rows buries the two that carry the measurement.
            # ``include_empty=True`` restores the full grid for the JSON-driven
            # chart renderer, which wants every bin edge.
            if include_empty:
                lines.append(f"| {b.lo:.2f}-{b.hi:.2f} | 0 | - | - | - | `{'.' * width}` |")
            continue
        filled = round(b.accuracy * width)
        marker = round(min(1.0, max(0.0, b.mean_confidence)) * (width - 1))
        cells = ["#" if i < filled else "." for i in range(width)]
        # The predicted-confidence caret overwrites one cell so both quantities
        # share a single scale. Caret right of the bars = over-confident.
        cells[marker] = "^"
        bar = "".join(cells)
        lines.append(
            f"| {b.lo:.2f}-{b.hi:.2f} | {b.count} | {b.mean_confidence:.3f} | "
            f"{b.accuracy:.3f} | {b.gap:+.3f} | `{bar}` |"
        )
    if event:
        ece = report.event_ece
        populated = sum(1 for b in bins if b.count > 0)
        total_bins = len(bins)
        source = "p(yes)"
        mce = max((abs(b.gap) for b in bins if b.count > 0), default=0.0)
    else:
        ece = report.ece
        populated = report.populated_bins
        total_bins = report.n_bins
        source = report.confidence_source
        mce = report.mce
    lines.append("")
    lines.append(
        f"n={report.n}  ECE={ece:.4f}  MCE={mce:.4f}  "
        f"populated bins={populated}/{total_bins}  confidence source: {source}"
    )
    return "\n".join(lines)


def render_coverage_table(report: CalibrationReport) -> str:
    """Risk/coverage as a markdown table: what does abstaining below T buy?"""
    lines = [
        f"{report.head}: accuracy vs confidence threshold",
        "",
        "| confidence >= | n | coverage | accuracy | delta vs all |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    base_acc = next((c["accuracy"] for c in report.coverage if c["threshold"] == 0.0), None)
    for point in report.coverage:
        acc = point["accuracy"]
        delta = "-" if (acc is None or base_acc is None) else f"{acc - base_acc:+.3f}"
        acc_s = "-" if acc is None else f"{acc:.3f}"
        lines.append(f"| {point['threshold']:.2f} | {point['n']} | {point['coverage']:.3f} | {acc_s} | {delta} |")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_BINS",
    "DEFAULT_COVERAGE_THRESHOLDS",
    "Bin",
    "CalibrationReport",
    "CalibrationSample",
    "baseline_brier",
    "binary_event_curve",
    "brier_score",
    "calibrate_binary_head",
    "calibrate_head",
    "coverage_curve",
    "expected_calibration_error",
    "reliability_diagram",
    "render_coverage_table",
    "render_reliability_table",
]
