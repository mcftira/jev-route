"""Live shadow metrics: the per-decision agreement window between layer 1 and the semantic head.

The promotion machinery in :mod:`jev_route.gate_semantic` has always had two
halves of evidence. The offline half is measured on a held-out set
(:func:`~jev_route.gate_semantic.measure_promotion_metrics`). The live half is
what the semantic head did to real traffic while it could not affect anything.
This module is the *rolling* version of that live half: instead of re-reading
the whole decision log each time, the layer feeds one line per decided request
into a small JSONL window that rolls forward as traffic accumulates.

Each record is one decision, and nothing else::

    {"decision_fired": true, "shadow_score": 0.91, "threshold": 0.5, "ts": 1750000000.0}

``decision_fired`` is the deterministic layer\'s verdict (its floor is not
``None``), ``shadow_score`` is the semantic head\'s probability, and the
agreement is ``decision_fired == (shadow_score >= threshold)``. No text, no
spans, no tokens: a window that could leak a prompt is not telemetry, it is the
incident. The file is runtime data, gitignored, and may be deleted: the window
re-accumulates, which is exactly what a shadow period is.

What the window answers:

* ``window_status()`` -- how many decisions, how many days, the agreement rate,
  a 15-bin rolling ECE (shadow probability against the binary outcome enforce
  mode would produce), and the drift alarm.
* drift -- the disagreement rate of the first 200 records is the baseline; the
  disagreement rate of everything after that is the current rate; when the
  current rate exceeds twice the baseline, the head is no longer describing the
  traffic it was measured on, and :func:`maybe_demote` writes a
  ``jev_route.demotion`` event.
* ``maybe_demote()`` -- the auto-demotion check the enforcing layer calls on
  every assessed request. It writes the event record and returns ``True`` only
  when the alarm is set.

Stdlib-only, for the same reason the rest of the gate is: the window is
appended on the request path.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: How many decisions the rolling window keeps. The window is evidence about the
#: *current* head; ten thousand records from before a redeploy are not, so the
#: file rolls forward rather than growing without bound.
DEFAULT_WINDOW_SIZE = 5000
#: A shadow period is complete only with at least this many decided requests...
REQUIRED_DECISIONS = 500
#: ...spanning at least this many days. Volume without time is a burst, and a
#: burst says nothing about the traffic the head would enforce on tomorrow.
REQUIRED_DAYS = 7.0
#: The baseline disagreement rate is measured over the first this many records.
BASELINE_SIZE = 200
#: Drift alarms when the post-baseline disagreement rate exceeds this multiple
#: of the baseline. Twice, not ten: the head and layer 1 are two systems that
#: must tell one coherent story, and a doubling of contradictions is the story
#: breaking.
DRIFT_FACTOR = 2.0
#: Bin count for the rolling ECE.
ECE_BINS = 15
#: The live agreement bar the graduation gate applies to the window.
MIN_AGREEMENT_RATE = 0.95
#: The ``kind`` field of the auto-demotion event record.
DEMOTION_EVENT_KIND = "jev_route.demotion"

_SECONDS_PER_DAY = 86400.0


def _finite(value: Any, name: str) -> float:
    """Coerce to a finite float, or refuse: a NaN in the window poisons every rate."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _shadow_fired(shadow_score: float, threshold: float) -> bool:
    """The binary outcome enforce mode would produce from this record."""
    return shadow_score >= threshold


def _disagreement(records: Sequence[Mapping[str, Any]]) -> float:
    """The fraction of ``records`` where layer 1 and the shadow head contradict."""
    if not records:
        return 0.0
    disagree = sum(1 for r in records if r["decision_fired"] != _shadow_fired(r["shadow_score"], r["threshold"]))
    return disagree / len(records)


def _ece(records: Sequence[Mapping[str, Any]], bins: int) -> float:
    """Expected calibration error over ``bins`` equal-width bins on the score.

    Confidence is the shadow score; the outcome is the binary enforce decision
    derived from that same record. A head whose 0.9 sits in a bin where it fires
    90% of the time is calibrated; a head that says 0.9 and fires 50% of the
    time is confident in the wrong proportion, and the number here says so.
    """
    n = len(records)
    counts = [0] * bins
    conf_sums = [0.0] * bins
    outcome_sums = [0.0] * bins
    for record in records:
        index = min(bins - 1, int(record["shadow_score"] * bins))
        counts[index] += 1
        conf_sums[index] += record["shadow_score"]
        outcome_sums[index] += 1.0 if _shadow_fired(record["shadow_score"], record["threshold"]) else 0.0
    ece = 0.0
    for index in range(bins):
        if counts[index]:
            ece += (counts[index] / n) * abs(conf_sums[index] / counts[index] - outcome_sums[index] / counts[index])
    return round(ece, 6)


@dataclass(frozen=True)
class WindowStatus:
    """The window reduced to the numbers a promotion or a demotion decides on."""

    n_decisions: int
    days_covered: float
    #: Fraction of records where layer 1 and the shadow head agree; ``None`` on an empty window.
    agreement_rate: float | None
    #: 15-bin ECE of the shadow score against the enforce outcome; ``None`` on an empty window.
    ece: float | None
    #: Disagreement rate over the first ``BASELINE_SIZE`` records; ``None`` until there are enough.
    baseline_disagreement: float | None
    #: Disagreement rate over the records after the baseline; ``None`` until there are any.
    current_disagreement: float | None
    #: True when the current disagreement rate exceeds ``DRIFT_FACTOR`` x the baseline.
    drift_alarm: bool
    #: True when ``n_decisions >= REQUIRED_DECISIONS`` and ``days_covered >= REQUIRED_DAYS``.
    window_complete: bool
    required_decisions: int
    required_days: float
    window_size: int
    first_ts: float | None
    last_ts: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_decisions": self.n_decisions,
            "days_covered": round(self.days_covered, 6),
            "agreement_rate": None if self.agreement_rate is None else round(self.agreement_rate, 6),
            "ece": self.ece,
            "baseline_disagreement": None
            if self.baseline_disagreement is None
            else round(self.baseline_disagreement, 6),
            "current_disagreement": None
            if self.current_disagreement is None
            else round(self.current_disagreement, 6),
            "drift_alarm": self.drift_alarm,
            "window_complete": self.window_complete,
            "required_decisions": self.required_decisions,
            "required_days": self.required_days,
            "window_size": self.window_size,
            "first_ts": self.first_ts,
            "last_ts": self.last_ts,
        }


class ShadowMetrics:
    """A rolling JSONL window of per-decision agreement, plus the drift check.

    Args:
        path: the window file. Created on the first :meth:`record`; read (as an
            empty window) when absent, so a freshly deployed shadow period starts
            at zero instead of erroring.
        window_size: how many records the file keeps; older ones roll off.
        min_decisions / min_days: the completion bar (see :data:`REQUIRED_DECISIONS`,
            :data:`REQUIRED_DAYS`).
        baseline_size: how many leading records define the baseline disagreement.
        drift_factor: the multiple of the baseline that trips the alarm.
        ece_bins: bin count for the rolling ECE.
        now: clock, ``() -> epoch seconds``. Injectable so tests run without
            sleeping; defaults to the wall clock.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        window_size: int = DEFAULT_WINDOW_SIZE,
        min_decisions: int = REQUIRED_DECISIONS,
        min_days: float = REQUIRED_DAYS,
        baseline_size: int = BASELINE_SIZE,
        drift_factor: float = DRIFT_FACTOR,
        ece_bins: int = ECE_BINS,
        now: Callable[[], float] | None = None,
    ) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if min_decisions < 1:
            raise ValueError(f"min_decisions must be >= 1, got {min_decisions}")
        if baseline_size < 1:
            raise ValueError(f"baseline_size must be >= 1, got {baseline_size}")
        if ece_bins < 1:
            raise ValueError(f"ece_bins must be >= 1, got {ece_bins}")
        self.path = Path(path)
        self.window_size = int(window_size)
        self.min_decisions = int(min_decisions)
        self.min_days = float(min_days)
        self.baseline_size = int(baseline_size)
        self.drift_factor = float(drift_factor)
        self.ece_bins = int(ece_bins)
        self._now = now if now is not None else (lambda: time.time())
        #: The last demotion event this instance wrote (set by :func:`maybe_demote`).
        self.last_event: dict[str, Any] | None = None
        self._records: list[dict[str, Any]] = self._load()

    # -- persistence ------------------------------------------------------ #
    @property
    def events_path(self) -> Path:
        """Where demotion events are appended, next to the window they belong to."""
        return self.path.with_suffix(".demotions.jsonl")

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        records: list[dict[str, Any]] = []
        for lineno, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            where = f"{self.path}:{lineno}"
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{where}: the window file is not valid JSONL ({exc}). A promotion reads "
                    "this file, so a corrupt line is a refusal, not a skip: repair or delete it "
                    "and let the shadow period re-accumulate."
                ) from exc
            records.append(self._parse_record(raw, where=where))
        if len(records) > self.window_size:
            # The file may hold more than the cap (it was written by an older
            # window size); status is always computed over the newest records.
            records = records[-self.window_size :]
        return records

    @staticmethod
    def _parse_record(raw: Any, *, where: str) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{where}: a window record must be a JSON object, got {type(raw).__name__}")
        missing = [key for key in ("decision_fired", "shadow_score", "threshold", "ts") if key not in raw]
        if missing:
            raise ValueError(f"{where}: record is missing field(s) {missing}")
        fired = raw["decision_fired"]
        if not isinstance(fired, bool):
            raise ValueError(f"{where}: decision_fired must be a boolean, got {fired!r}")
        return {
            "decision_fired": fired,
            "shadow_score": _finite(raw["shadow_score"], "shadow_score"),
            "threshold": _finite(raw["threshold"], "threshold"),
            "ts": _finite(raw["ts"], "ts"),
        }

    def _trim(self) -> None:
        """Rewrite the file with only the newest ``window_size`` records (atomic)."""
        keep = self._records[-self.window_size :]
        self._records = keep
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in keep), encoding="utf-8")
        tmp.replace(self.path)

    # -- collection --------------------------------------------------------- #
    def record(
        self,
        decision_fired: bool,
        shadow_score: float,
        threshold: float,
        *,
        ts: float | None = None,
    ) -> dict[str, Any]:
        """Append one decided request to the window. Returns the stored record."""
        record = {
            "decision_fired": bool(decision_fired),
            "shadow_score": _finite(shadow_score, "shadow_score"),
            "threshold": _finite(threshold, "threshold"),
            "ts": _finite(ts if ts is not None else self._now(), "ts"),
        }
        self._records.append(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if len(self._records) > self.window_size:
            self._trim()
        return record

    # -- status ------------------------------------------------------------- #
    def n_records(self) -> int:
        """How many records the window currently holds."""
        return len(self._records)

    def window_status(self) -> WindowStatus:
        """Reduce the window to :class:`WindowStatus`. No side effects."""
        records = self._records
        n = len(records)
        if n == 0:
            return WindowStatus(
                n_decisions=0,
                days_covered=0.0,
                agreement_rate=None,
                ece=None,
                baseline_disagreement=None,
                current_disagreement=None,
                drift_alarm=False,
                window_complete=False,
                required_decisions=self.min_decisions,
                required_days=self.min_days,
                window_size=self.window_size,
                first_ts=None,
                last_ts=None,
            )
        timestamps = [record["ts"] for record in records]
        # max-min, not last-first: a wall clock can step backwards and must not
        # be able to shrink the measured coverage of the period.
        days_covered = max(0.0, (max(timestamps) - min(timestamps)) / _SECONDS_PER_DAY)
        agreement = sum(
            1
            for record in records
            if record["decision_fired"] == _shadow_fired(record["shadow_score"], record["threshold"])
        ) / n
        baseline = _disagreement(records[: self.baseline_size]) if n >= self.baseline_size else None
        current = _disagreement(records[self.baseline_size :]) if n > self.baseline_size else None
        drift_alarm = (
            baseline is not None and current is not None and current > self.drift_factor * baseline
        )
        return WindowStatus(
            n_decisions=n,
            days_covered=days_covered,
            agreement_rate=round(agreement, 6),
            ece=_ece(records, self.ece_bins),
            baseline_disagreement=baseline,
            current_disagreement=current,
            drift_alarm=drift_alarm,
            window_complete=(n >= self.min_decisions and days_covered >= self.min_days),
            required_decisions=self.min_decisions,
            required_days=self.min_days,
            window_size=self.window_size,
            first_ts=records[0]["ts"],
            last_ts=records[-1]["ts"],
        )


def maybe_demote(metrics: ShadowMetrics, *, ts: str | None = None) -> bool:
    """The auto-demotion check. Writes the event when -- and only when -- drift alarms.

    Reads the window, and when the post-baseline disagreement rate has exceeded
    ``DRIFT_FACTOR`` x the baseline, appends one event record to
    :attr:`ShadowMetrics.events_path`::

        {"kind": "jev_route.demotion", "ts": "<iso-8601>", "reason": "...", ...}

    and returns ``True``. The event is what an operator (or a host that polls
    the layer) acts on; the layer that called this is responsible for stopping
    enforcement in-process. ``ts`` is injectable so tests never sleep.
    """
    status = metrics.window_status()
    if not status.drift_alarm:
        return False
    reason = (
        f"live disagreement drifted: post-baseline rate {status.current_disagreement:.4f} exceeds "
        f"{metrics.drift_factor:g}x the baseline {status.baseline_disagreement:.4f} "
        f"(baseline = first {metrics.baseline_size} decisions, "
        f"current = following {status.n_decisions - metrics.baseline_size})"
    )
    event = {
        "kind": DEMOTION_EVENT_KIND,
        "ts": ts if ts is not None else datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reason": reason,
        "baseline_disagreement": round(status.baseline_disagreement, 6),
        "current_disagreement": round(status.current_disagreement, 6),
        "n_decisions": status.n_decisions,
    }
    metrics.events_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics.events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    metrics.last_event = event
    return True


__all__ = [
    "BASELINE_SIZE",
    "DEFAULT_WINDOW_SIZE",
    "DEMOTION_EVENT_KIND",
    "DRIFT_FACTOR",
    "ECE_BINS",
    "MIN_AGREEMENT_RATE",
    "REQUIRED_DAYS",
    "REQUIRED_DECISIONS",
    "ShadowMetrics",
    "WindowStatus",
    "maybe_demote",
]
