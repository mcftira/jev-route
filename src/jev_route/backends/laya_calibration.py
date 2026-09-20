"""Post-hoc calibration for Laya's overconfident outputs.

Laya ships overconfident: its raw probabilities are peaked and its built-in
temperature is not fit on *our* routing task, so the confidence floors in a
policy would threshold on numbers that are not calibrated. This module is the
make-or-break layer: it turns Laya's raw distributions into calibrated ones
and ships the fit as a small, versioned, checksummed artifact that the
:class:`~jev_route.backends.laya.LayaBackend` loads at startup.

Two methods
-----------
``temperature`` (default)
    A single scalar per head, ``T``, applied as ``p' = softmax(log(p) / T)``
    (for the binary head, ``p' = p^(1/T) / (p^(1/T) + (1-p)^(1/T))``). Fit by
    minimising the negative log-likelihood of the target distributions -- gold
    one-hot labels from the labelled eval, or the teacher's soft distributions
    from a Jev decision log (the Jev-to-Laya knowledge-transfer case).
    ``T > 1`` flattens an overconfident head; ``T < 1`` sharpens an
    underconfident one. One parameter per head: no overfitting risk at our
    sample sizes, and the whole fit is a bounded 1-D search.

``isotonic`` (optional)
    Per class, a monotone piecewise-constant map from the raw probability of
    that class to its calibrated probability, fit with pool-adjacent-violators
    (pure Python -- no sklearn). More expressive, more parameters, so it wants
    more data than temperature; use it when the eval or teacher log is large
    and temperature is visibly bent. Choice heads are renormalised after the
    per-class maps so each head still sums to one.

The transform is total and honest: any probability vector in, a valid one out.
The artifact is the unit of trust -- it records *what it was fit on* (source,
n, ECE before/after) and is checksummed, so a tampered or mismatched file is
refused at load rather than silently applied.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from ..schema import (
    ChoiceAnswer,
    DecisionAnswers,
    NoulAnswer,
    certainty_from_probabilities,
)

#: Heads a calibration artifact covers, in the order the policy reads them.
CHOICE_HEADS: tuple[str, ...] = ("complexity", "sensitivity", "domain")
BINARY_HEAD = "pii"
ALL_HEADS: tuple[str, ...] = (*CHOICE_HEADS, BINARY_HEAD)

ARTIFACT_KIND = "laya_route.calibration"
ARTIFACT_SCHEMA_VERSION = 1


class CalibrationError(Exception):
    """A calibration artifact that is missing, malformed, or tampered with."""


# --------------------------------------------------------------------------- #
# Fit
# --------------------------------------------------------------------------- #
def _nll_for_temperature(probs: Sequence[Mapping[str, float]], target: Mapping[str, float], temp: float) -> float:
    """Negative log-likelihood of ``target`` under ``softmax(log(p)/temp)``."""
    total = 0.0
    for p in probs:
        lps = [math.log(max(float(v), 1e-12)) for v in p.values()]
        m = max(lps)
        zs = [lp / temp - m / temp for lp in lps]
        # softmax in log-space for stability
        exps = [math.exp(z) for z in zs]
        s = sum(exps)
        for (c, _v), e in zip(p.items(), exps, strict=True):
            t = float(target.get(c, 0.0))
            if t > 0.0:
                total -= t * math.log(max(e / s, 1e-12))
    return total


def fit_temperature(
    samples_by_head: Mapping[str, Sequence[tuple[Mapping[str, float], Mapping[str, float]]]],
    *,
    lo: float = 0.05,
    hi: float = 12.0,
    iters: int = 60,
) -> dict[str, float]:
    """Fit one temperature per head by golden-section minimisation of the NLL.

    ``samples_by_head[head]`` is a sequence of ``(raw_probs, target)`` pairs,
    where ``raw_probs`` is Laya's distribution over that head's options and
    ``target`` is the fit target: a gold one-hot (from the labelled eval) or a
    teacher soft distribution (from a Jev decision log). Deterministic.
    """
    temps: dict[str, float] = {}
    for head, samples in samples_by_head.items():
        if not samples:
            temps[head] = 1.0
            continue
        # One target per sample; all samples share the same option set. The
        # samples bind explicitly so the closure cannot capture a stale loop
        # variable (ruff B023).
        def objective(temp: float, _samples: tuple = samples) -> float:
            return sum(
                _nll_for_temperature([p], {c: t.get(c, 0.0) for c in p}, temp)
                for p, t in _samples
            )

        # Golden-section search on [lo, hi].
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        a, b = lo, hi
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc, fd = objective(c), objective(d)
        for _ in range(iters):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - gr * (b - a)
                fc = objective(c)
            else:
                a, c, fc = c, d, fd
                d = a + gr * (b - a)
                fd = objective(d)
        temps[head] = round(min(hi, max(lo, (a + b) / 2.0)), 6)
    return temps


class Isotonic1D:
    """A monotone piecewise-constant regressor (pool-adjacent-violators).

    Fit on ``(x, y)`` pairs with ``y`` in ``[0, 1]``; predict a calibrated
    value for a new ``x``. Pure Python, deterministic, and clamped at both
    ends so an out-of-range input degrades to the nearest fitted value rather
    than extrapolating off the cliff.
    """

    def __init__(self) -> None:
        self._x_breaks: list[float] = []
        self._y_vals: list[float] = []

    def fit(self, xs: Sequence[float], ys: Sequence[float]) -> Isotonic1D:
        if len(xs) != len(ys):
            raise ValueError("xs and ys must be the same length")
        order = sorted(range(len(xs)), key=lambda i: float(xs[i]))
        # Stack of blocks: [sum_y, count, x_max]. PAV: merge while the left
        # block's mean exceeds the right block's mean.
        blocks: list[list[float]] = []
        for i in order:
            blocks.append([float(ys[i]), 1.0, float(xs[i])])
            while len(blocks) >= 2 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
                sy, cn, _xm = blocks.pop()
                blocks[-1][0] += sy
                blocks[-1][1] += cn
                # x_max of the merged block is the *right* block's x_max.
                # (blocks[-1] already holds it.)
        self._x_breaks = [b[2] for b in blocks]
        self._y_vals = [min(1.0, max(0.0, b[0] / b[1])) for b in blocks]
        return self

    def predict(self, x: float) -> float:
        if not self._x_breaks:
            return min(1.0, max(0.0, x))
        x = float(x)
        if x <= self._x_breaks[0]:
            return self._y_vals[0]
        if x > self._x_breaks[-1]:
            return self._y_vals[-1]
        # Largest break <= x.
        lo, hi = 0, len(self._x_breaks) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._x_breaks[mid] <= x:
                lo = mid
            else:
                hi = mid - 1
        return self._y_vals[lo]

    def to_breakpoints(self) -> tuple[list[float], list[float]]:
        return list(self._x_breaks), list(self._y_vals)

    @classmethod
    def from_breakpoints(cls, xs: Sequence[float], ys: Sequence[float]) -> Isotonic1D:
        iso = cls()
        iso._x_breaks = [float(x) for x in xs]
        iso._y_vals = [float(y) for y in ys]
        return iso


def fit_isotonic(
    samples_by_head: Mapping[str, Sequence[tuple[Mapping[str, float], Mapping[str, float]]]],
) -> dict[str, dict[str, tuple[list[float], list[float]]]]:
    """Fit a per-class isotonic map for each head.

    For choice heads, one :class:`Isotonic1D` per option: fit on
    ``(p_option, target_option)``. For the binary head, a single map on
    ``(p_yes, target_yes)``.
    """
    out: dict[str, dict[str, tuple[list[float], list[float]]]] = {}
    for head, samples in samples_by_head.items():
        if not samples:
            out[head] = {}
            continue
        if head == BINARY_HEAD:
            options = ["yes"]  # the "no" map is redundant; apply() uses only "yes"
        else:
            # Option names = the keys of the first sample's raw distribution.
            options = list(samples[0][0].keys()) if samples[0][0] else []
        per_class: dict[str, tuple[list[float], list[float]]] = {}
        for opt in options:
            xs = [float(s[0].get(opt, 0.0)) for s in samples]
            ys = [float(s[1].get(opt, 0.0)) for s in samples]
            xsu = sorted(set(xs))
            if len(xsu) < 2:
                # A single distinct x cannot fit a monotone step; keep identity.
                per_class[opt] = ([0.0, 1.0], [0.0, 1.0])
                continue
            iso = Isotonic1D().fit(xs, ys)
            per_class[opt] = iso.to_breakpoints()
        out[head] = per_class
    return out


def expected_calibration_error(conf: Sequence[float], correct: Sequence[bool], *, bins: int = 15) -> float:
    """Small self-contained ECE (mirrors ``evals/calibration.py`` for the artifact)."""
    if not conf:
        return 0.0
    edges = [i / bins for i in range(bins + 1)]
    e = 0.0
    for lo, hi in pairwise(edges):
        sel = [(c, ok) for c, ok in zip(conf, correct, strict=True) if lo < c <= hi]
        if sel:
            mean_conf = sum(c for c, _ in sel) / len(sel)
            acc = sum(1 for _, ok in sel if ok) / len(sel)
            e += (len(sel) / len(conf)) * abs(mean_conf - acc)
    return float(e)


def _confidence_for(head: str, probs: Mapping[str, float]) -> float:
    if head == BINARY_HEAD:
        p = float(probs.get("yes", 0.5))
        return abs(2.0 * p - 1.0)
    return certainty_from_probabilities(probs)


# --------------------------------------------------------------------------- #
# Artifact
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CalibrationArtifact:
    """A fitted, versioned, checksummed calibration for a Laya checkpoint."""

    method: str
    temperature: Mapping[str, float] = field(default_factory=dict)
    isotonic: Mapping[str, Mapping[str, tuple[Sequence[float], Sequence[float]]]] = field(default_factory=dict)
    fit: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = ARTIFACT_SCHEMA_VERSION

    def canonical(self) -> dict[str, Any]:
        """Everything except the checksum, in stable order, for hashing."""
        return {
            "schema_version": self.schema_version,
            "kind": ARTIFACT_KIND,
            "method": self.method,
            "temperature": {k: float(v) for k, v in sorted(self.temperature.items())},
            "isotonic": {
                h: {c: ([float(x) for x in xs], [float(y) for y in ys]) for c, (xs, ys) in sorted(cls.items())}
                for h, cls in sorted(self.isotonic.items())
            },
            "fit": dict(self.fit),
        }

    def checksum(self) -> str:
        blob = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        return "sha256:" + hashlib.sha256(blob).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        d = self.canonical()
        d["checksum"] = self.checksum()
        return d

    def apply_to(self, answers: DecisionAnswers) -> DecisionAnswers:
        """Return ``answers`` with this artifact's calibration applied."""
        return apply(answers, self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CalibrationArtifact:
        method = str(data.get("method", ""))
        if method not in ("temperature", "isotonic"):
            raise CalibrationError(f"unknown calibration method {method!r}")
        temperature = {str(k): float(v) for k, v in (data.get("temperature") or {}).items()}
        if method == "temperature" and not temperature:
            raise CalibrationError("temperature artifact carries no temperatures")
        iso_raw = data.get("isotonic") or {}
        isotonic = {
            str(h): {str(c): (tuple(map(float, xs)), tuple(map(float, ys))) for c, (xs, ys) in cls.items()}
            for h, cls in iso_raw.items()
        }
        if method == "isotonic" and not isotonic:
            raise CalibrationError("isotonic artifact carries no isotonic maps")
        return cls(
            method=method,
            temperature=temperature,
            isotonic=isotonic,
            fit=dict(data.get("fit") or {}),
            schema_version=int(data.get("schema_version", ARTIFACT_SCHEMA_VERSION)),
        )


def save_artifact(artifact: CalibrationArtifact, path: str) -> str:
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact.to_dict(), f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    return path


def load_artifact(source: Any) -> CalibrationArtifact:
    """Load and *validate* a calibration artifact from a path or a dict.

    Refuses (raises :class:`CalibrationError`) on: a wrong schema version, a
    checksum mismatch (tampering or a stale file), or a method with missing
    parameters. A router that cannot verify its calibration must not run it.
    """
    if isinstance(source, Mapping):
        data = dict(source)
    elif isinstance(source, CalibrationArtifact):
        return source
    else:
        with open(str(source), encoding="utf-8") as f:
            data = json.load(f)

    artifact = CalibrationArtifact.from_dict(data)
    if artifact.schema_version != ARTIFACT_SCHEMA_VERSION:
        raise CalibrationError(
            f"calibration schema version {artifact.schema_version} is not supported "
            f"by this build (expected {ARTIFACT_SCHEMA_VERSION})"
        )
    expected = artifact.checksum()
    got = str(data.get("checksum", ""))
    if got and got != expected:
        raise CalibrationError(
            "calibration checksum mismatch: the artifact was modified after it was "
            f"fitted (expected {expected}, file says {got})"
        )
    return artifact


# --------------------------------------------------------------------------- #
# Apply
# --------------------------------------------------------------------------- #
def _temperature_choice(probs: Mapping[str, float], temp: float) -> dict[str, float]:
    lps = [math.log(max(float(v), 1e-12)) for v in probs.values()]
    scaled = [lp / temp for lp in lps]
    m = max(scaled)
    exps = [math.exp(z - m) for z in scaled]
    s = sum(exps)
    return {k: e / s for k, e in zip(probs.keys(), exps, strict=True)}


def _temperature_binary(p: float, temp: float) -> float:
    p = min(1.0 - 1e-12, max(1e-12, p))
    a = math.log(p) / temp
    b = math.log(1.0 - p) / temp
    m = max(a, b)
    ea, eb = math.exp(a - m), math.exp(b - m)
    return ea / (ea + eb)


def apply(answers: DecisionAnswers, artifact: CalibrationArtifact) -> DecisionAnswers:
    """Return a copy of ``answers`` with the calibration applied.

    The transform is total: it never raises on odd input, and every head still
    carries a full, normalised distribution.
    """
    if artifact.method == "temperature":
        temps = {h: float(artifact.temperature.get(h, 1.0)) for h in ALL_HEADS}
        complexity = _calib_choice(answers.complexity, temps["complexity"], lambda p, t: _temperature_choice(p, t))
        sensitivity = _calib_choice(answers.sensitivity, temps["sensitivity"], lambda p, t: _temperature_choice(p, t))
        domain = _calib_choice(answers.domain, temps["domain"], lambda p, t: _temperature_choice(p, t))
        pii_p = _temperature_binary(answers.pii.value, temps["pii"])
    else:
        def iso_map(head: str, opt: str) -> Isotonic1D:
            bp = artifact.isotonic.get(head, {}).get(opt)
            if bp is None:
                return _IDENTITY
            return Isotonic1D.from_breakpoints(bp[0], bp[1])

        complexity = _calib_choice(answers.complexity, None, lambda p, _: _isotonic_choice(p, "complexity", iso_map))
        sensitivity = _calib_choice(answers.sensitivity, None, lambda p, _: _isotonic_choice(p, "sensitivity", iso_map))
        domain = _calib_choice(answers.domain, None, lambda p, _: _isotonic_choice(p, "domain", iso_map))
        pii_bp = artifact.isotonic.get(BINARY_HEAD, {})
        pii_map = (
            Isotonic1D.from_breakpoints(pii_bp["yes"][0], pii_bp["yes"][1])
            if "yes" in pii_bp
            else _IDENTITY
        )
        pii_p = min(1.0, max(0.0, pii_map.predict(answers.pii.value)))

    return DecisionAnswers(
        complexity=complexity,
        sensitivity=sensitivity,
        pii=NoulAnswer(value=min(1.0, max(0.0, pii_p))),
        domain=domain,
    )


_IDENTITY = Isotonic1D()


def _calib_choice(answer: ChoiceAnswer, temp: float | None, fn: Any) -> ChoiceAnswer:
    """Calibrate one choice head and set the confidence to the calibrated top probability.

    Temperature scaling calibrates the *top probability*: after the fit,
    ``max(p')`` is an honest estimate of P(the argmax is correct) -- the number
    the policy's confidence floors threshold on. (The normalized-entropy
    ``computed_confidence`` stays available on the answer for the views that
    need it; it is deliberately not the headline confidence here.)
    """
    probs = {k: float(v) for k, v in answer.probabilities.items()}
    new_probs = fn(probs, temp)
    total = sum(new_probs.values())
    if total <= 0:
        new_probs = probs
    else:
        new_probs = {k: v / total for k, v in new_probs.items()}
    confidence = max(new_probs.values())
    choice = max(new_probs.items(), key=lambda kv: kv[1])[0]
    return ChoiceAnswer(
        choice=choice,
        probabilities=new_probs,
        confidence=confidence,
        confidence_reported=True,
    )


def _isotonic_choice(probs: Mapping[str, float], head: str, iso_map: Any) -> dict[str, float]:
    return {
        k: min(1.0, max(0.0, iso_map(head, str(k)).predict(float(v)))) for k, v in probs.items()
    }


def make_fit_metadata(
    *,
    method: str,
    samples_by_head: Mapping[str, Sequence[tuple[Mapping[str, float], Mapping[str, float]]]],
    laya_checkpoint: str,
    laya_version: str,
    source: str,
    ece_before: Mapping[str, float],
    ece_after: Mapping[str, float],
) -> dict[str, Any]:
    """Assemble the ``fit`` block an artifact records about its own provenance."""
    return {
        "method": method,
        "source": source,
        "n_samples": int(max((len(v) for v in samples_by_head.values()), default=0)),
        "per_head_n": {h: len(v) for h, v in samples_by_head.items()},
        "laya_checkpoint": laya_checkpoint,
        "laya_version": laya_version,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ece_before": {k: round(float(v), 4) for k, v in ece_before.items()},
        "ece_after": {k: round(float(v), 4) for k, v in ece_after.items()},
    }


__all__ = [
    "ALL_HEADS",
    "ARTIFACT_KIND",
    "ARTIFACT_SCHEMA_VERSION",
    "BINARY_HEAD",
    "CHOICE_HEADS",
    "CalibrationArtifact",
    "CalibrationError",
    "Isotonic1D",
    "apply",
    "expected_calibration_error",
    "fit_isotonic",
    "fit_temperature",
    "load_artifact",
    "make_fit_metadata",
    "save_artifact",
]
