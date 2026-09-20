"""Turning the decision log into a training set.

The log written by :mod:`jev_route.logging_sink` is already a dataset -- that was
the point of storing full distributions instead of argmax labels -- but it is not
yet a *good* one. This module does four jobs, and each one exists because getting
it wrong silently produces a worse model:

1. **Pick the mode the log can actually support.** ``text`` mode needs stored
   redacted excerpts (``logging.excerpt_mode: redacted``); ``features`` mode needs
   only the deterministic :class:`~jev_route.schema.RequestFeatures`. Asking for
   text from a text-free log is an error, not a fallback: silently training a
   weaker model than the operator asked for is exactly the kind of surprise this
   project should not have.
2. **Drop the rows that carry no signal.** A degraded decision is a uniform
   distribution -- the shape of "the backend was down". Training on it teaches the
   student to be uncertain about everything.
3. **Keep the rows the gate decided, with the heads they can label masked.** A
   gate-blocked request never reached a backend, so complexity and domain are
   uniform and unusable, but the sensitivity label is deterministic and locally
   produced. Those rows are *included*, tagged, and masked per head.
4. **Split reproducibly.** The train/holdout assignment is a hash of
   ``request_id``, so re-running export on a grown log keeps every previous row on
   the same side. Nothing that was trained on ever moves into the holdout, which
   is what makes the graduation numbers mean something.

Labels are the teacher's *raw* belief, not the policy-adjusted effective answer.
The router re-applies the gate floor and the ``on_uncertain`` confidence bump to
whatever a backend returns, so baking those adjustments into the student would
apply them twice and destroy the calibration that made the cloud phase worth
paying for. The effective values are still exported, under ``teacher``, for
evaluation and debugging.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ..logging_sink import iter_all_records, iter_records
from ..schema import (
    COMPLEXITY_LEVELS,
    DOMAINS,
    SCHEMA_VERSION,
    SENSITIVITY_LEVELS,
    DecisionRecord,
    RequestFeatures,
)

#: Bump when the row layout changes incompatibly. Stored on every row.
ROW_SCHEMA = "jev_route.distill.row/1"
SUPPORTED_ROW_SCHEMAS: frozenset[str] = frozenset({"jev_route.distill.row/1"})

ExportMode = Literal["text", "features", "auto"]

CHOICE_HEADS: tuple[str, ...] = ("complexity", "sensitivity", "domain")
LADDERS: dict[str, tuple[str, ...]] = {
    "complexity": COMPLEXITY_LEVELS,
    "sensitivity": SENSITIVITY_LEVELS,
    "domain": DOMAINS,
}

#: Salt so a request_id cannot be reused as a split key elsewhere by accident.
_SPLIT_SALT = "jev-route-distill-split\x00"

#: A distribution whose spread is below this is "uniform for practical purposes".
_UNIFORM_EPS = 1e-6


class ExportError(ValueError):
    """The log cannot produce the dataset that was asked for. Always actionable."""


# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TargetSet:
    """Soft targets for one record: distributions, hard labels, and per-head usability."""

    complexity: Mapping[str, Any]
    sensitivity: Mapping[str, Any]
    domain: Mapping[str, Any]
    pii: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "complexity": dict(self.complexity),
            "sensitivity": dict(self.sensitivity),
            "domain": dict(self.domain),
            "pii": dict(self.pii),
        }

    def get(self, head: str) -> Mapping[str, Any]:
        return getattr(self, head)

    def usable(self, head: str) -> bool:
        return bool(self.get(head).get("usable", False))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TargetSet:
        return cls(**{head: dict(data[head]) for head in ("complexity", "sensitivity", "domain", "pii")})


@dataclass(frozen=True)
class TrainingRow:
    """One exported training example."""

    request_id: str
    split: str
    targets: TargetSet
    features: RequestFeatures
    teacher: Mapping[str, Any]
    gate: Mapping[str, Any]
    context: Mapping[str, Any]
    provenance: Mapping[str, Any]
    tags: tuple[str, ...] = ()
    text: str | None = None
    excerpt_hash: str = ""
    timestamp: str = ""
    weight: float = 1.0
    row_schema: str = ROW_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_schema": self.row_schema,
            "split": self.split,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "excerpt_hash": self.excerpt_hash,
            "text": self.text,
            "features": self.features.to_dict(),
            "targets": self.targets.as_dict(),
            "teacher": dict(self.teacher),
            "gate": dict(self.gate),
            "context": dict(self.context),
            "provenance": dict(self.provenance),
            "tags": list(self.tags),
            "weight": self.weight,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrainingRow:
        schema = str(data.get("row_schema", ROW_SCHEMA))
        if schema not in SUPPORTED_ROW_SCHEMAS:
            raise ExportError(
                f"dataset row schema {schema!r} is not supported by this build "
                f"(supports {sorted(SUPPORTED_ROW_SCHEMAS)}); re-export the dataset"
            )
        return cls(
            request_id=str(data["request_id"]),
            split=str(data["split"]),
            targets=TargetSet.from_dict(data["targets"]),
            features=RequestFeatures.from_dict(data.get("features") or {}),
            teacher=dict(data.get("teacher") or {}),
            gate=dict(data.get("gate") or {}),
            context=dict(data.get("context") or {}),
            provenance=dict(data.get("provenance") or {}),
            tags=tuple(str(t) for t in (data.get("tags") or ())),
            text=data.get("text"),
            excerpt_hash=str(data.get("excerpt_hash", "")),
            timestamp=str(data.get("timestamp", "")),
            weight=float(data.get("weight", 1.0)),
            row_schema=schema,
        )

    @classmethod
    def from_json(cls, line: str) -> TrainingRow:
        return cls.from_dict(json.loads(line))

    @property
    def is_gate_row(self) -> bool:
        return "gate-fired" in self.tags


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def split_bucket(request_id: str) -> float:
    """Deterministic ``[0, 1)`` bucket for a request id.

    SHA-256 rather than Python's ``hash()``: the builtin is salted per process, so
    a split computed today would not be reproducible tomorrow. Reproducibility is
    the whole requirement here -- a holdout that changes membership between runs
    cannot be used to make a cutover decision.
    """
    digest = hashlib.sha256((_SPLIT_SALT + str(request_id)).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def assign_split(request_id: str, holdout_fraction: float) -> str:
    fraction = min(max(float(holdout_fraction), 0.0), 0.9)
    return "holdout" if split_bucket(request_id) < fraction else "train"


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #
def _choice_target(probs: Mapping[str, float] | None, ladder: Sequence[str], source: str) -> dict[str, Any]:
    """Normalize a logged distribution onto ``ladder`` and judge whether it carries signal.

    Probabilities are stored at full precision, not rounded: an exported
    distribution should be bit-identical to the logged one, both because "the soft
    targets survived" is a property worth testing exactly and because rounding 6
    decimal places repeatedly is how calibration quietly drifts.
    """
    from ..schema import certainty_from_probabilities

    raw = {str(k): float(v) for k, v in (probs or {}).items() if str(k) in ladder}
    total = sum(raw.values())
    if total <= 0.0:
        # No mass anywhere on the ladder: nothing was asked, or the backend
        # answered with options we do not know. Uniform, and unusable.
        even = 1.0 / len(ladder)
        return {
            "probs": dict.fromkeys(ladder, even),
            "label": ladder[len(ladder) // 2],
            "usable": False,
            "source": "none",
            "confidence": 0.0,
        }
    ordered = {level: raw.get(level, 0.0) / total for level in ladder}
    even = 1.0 / len(ladder)
    uniform = all(abs(value - even) <= _UNIFORM_EPS for value in ordered.values())
    label = max(ladder, key=lambda level: (ordered[level], -ladder.index(level)))
    return {
        "probs": ordered,
        "label": label,
        # A uniform distribution is the shape of "nobody classified this": the
        # gate blocked the call, or the backend degraded. Either way it is not a
        # label, and training on it would teach the student to be uncertain.
        "usable": not uniform,
        "source": source,
        "confidence": certainty_from_probabilities(ordered),
    }


def _pii_target(value: float | None, source: str) -> dict[str, Any]:
    v = 0.5 if value is None else min(1.0, max(0.0, float(value)))
    return {
        "value": v,
        "label": v >= 0.5,
        # Exactly 0.5 is NoulAnswer.unknown(): a coin flip, i.e. no measurement.
        "usable": abs(v - 0.5) > 1e-9,
        "source": source,
        "confidence": abs(2.0 * v - 1.0),
    }


def _extract_targets(record: DecisionRecord) -> tuple[TargetSet, int]:
    """Build the row's targets. Returns ``(targets, n_usable_heads)``."""
    answers = record.decision.answers
    gate_only = record.decision.backend == "gate"
    source = "gate" if gate_only else "backend"
    heads = {
        head: _choice_target(getattr(answers, head).probabilities, LADDERS[head], source)
        for head in CHOICE_HEADS
    }
    # The effective (post-gate, post-bump) label is recorded separately: it is what
    # the policy reasoned about, and it is NOT what we train on. See module docstring.
    heads["sensitivity"]["effective_label"] = record.decision.effective_sensitivity
    heads["complexity"]["effective_label"] = record.decision.effective_complexity
    pii = _pii_target(answers.pii.value, source)
    targets = TargetSet(
        complexity=heads["complexity"],
        sensitivity=heads["sensitivity"],
        domain=heads["domain"],
        pii=pii,
    )
    return targets, sum(1 for head in (*CHOICE_HEADS, "pii") if targets.usable(head))


# --------------------------------------------------------------------------- #
# Reading the log
# --------------------------------------------------------------------------- #
def iter_source_records(
    source: str | Path | Iterable[DecisionRecord], *, pattern: str = "decisions*.jsonl"
) -> Iterator[DecisionRecord]:
    """Yield records from a JSONL file, a directory of rotated logs, or an iterable.

    Accepting all three keeps the pipeline scriptable: the CLI passes a path, the
    tests pass records they just built, and a rotated log directory works without
    the operator concatenating anything.
    """
    if isinstance(source, DecisionRecord):
        yield source
        return
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            yield from iter_all_records(path, pattern=pattern)
        else:
            yield from iter_records(path)
        return
    yield from source


def percentiles(values: Sequence[float]) -> dict[str, float]:
    """p50/p95/p99/mean by linear interpolation. Public: evaluate and graduate reuse it."""
    if not values:
        return {"n": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    ordered = sorted(float(v) for v in values)

    def pick(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = q * (len(ordered) - 1)
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {
        "n": len(ordered),
        "p50": round(pick(0.50), 3),
        "p95": round(pick(0.95), 3),
        "p99": round(pick(0.99), 3),
        "mean": round(sum(ordered) / len(ordered), 3),
    }


@dataclass(frozen=True)
class LogInspection:
    """What a decision log can support, measured before anything is exported."""

    source: str
    records: int = 0
    degraded: int = 0
    gate_fired: int = 0
    gate_blocked: int = 0
    gate_only: int = 0
    with_excerpt: int = 0
    without_excerpt: int = 0
    schema_versions: dict[str, int] = field(default_factory=dict)
    backends: dict[str, int] = field(default_factory=dict)
    teacher_model_versions: dict[str, int] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def text_mode_possible(self) -> bool:
        return self.with_excerpt > 0

    @property
    def features_mode_possible(self) -> bool:
        return self.records - self.degraded > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "records": self.records,
            "degraded": self.degraded,
            "gate_fired": self.gate_fired,
            "gate_blocked": self.gate_blocked,
            "gate_only": self.gate_only,
            "with_excerpt": self.with_excerpt,
            "without_excerpt": self.without_excerpt,
            "schema_versions": dict(self.schema_versions),
            "backends": dict(self.backends),
            "teacher_model_versions": dict(self.teacher_model_versions),
            "latency_ms": dict(self.latency_ms),
            "text_mode_possible": self.text_mode_possible,
            "features_mode_possible": self.features_mode_possible,
            "notes": list(self.notes),
        }


def inspect_log(source: str | Path | Iterable[DecisionRecord], **kwargs: Any) -> LogInspection:
    """Stream the log once and report what it contains.

    Streaming rather than loading: a production log is the largest file in the
    project by a wide margin, and answering "can I train a text model?" must not
    require holding it in memory.
    """
    counts: dict[str, int] = dict.fromkeys(
        ("records", "degraded", "gate_fired", "gate_blocked", "gate_only", "with_excerpt"), 0
    )
    schema_versions: dict[str, int] = {}
    backends: dict[str, int] = {}
    versions: dict[str, int] = {}
    latencies: list[float] = []
    for record in iter_source_records(source, **kwargs):
        counts["records"] += 1
        schema_versions[record.schema_version] = schema_versions.get(record.schema_version, 0) + 1
        backend = record.decision.backend
        backends[backend] = backends.get(backend, 0) + 1
        versions[record.decision.backend_model_version or "(unknown)"] = (
            versions.get(record.decision.backend_model_version or "(unknown)", 0) + 1
        )
        if record.decision.degraded:
            counts["degraded"] += 1
        if record.decision.gate.fired:
            counts["gate_fired"] += 1
        if record.decision.gate.blocks_backend:
            counts["gate_blocked"] += 1
        if backend == "gate":
            counts["gate_only"] += 1
        if (record.excerpt or "").strip():
            counts["with_excerpt"] += 1
        if record.backend_latency_ms > 0:
            latencies.append(record.backend_latency_ms)

    notes: list[str] = []
    if counts["records"] == 0:
        notes.append("the log is empty: route some traffic first")
    if counts["with_excerpt"] == 0:
        notes.append("no record carries an excerpt: text mode is impossible (logging.excerpt_mode is not 'redacted')")
    if counts["degraded"]:
        notes.append(
            f"{counts['degraded']} degraded record(s) will be skipped: a failed backend logged uniform "
            "distributions"
        )
    if counts["gate_only"]:
        notes.append(
            f"{counts['gate_only']} record(s) never reached a backend (the local gate blocked the call); "
            "their sensitivity label is kept, their complexity/domain heads are masked"
        )
    return LogInspection(
        source=str(source),
        records=counts["records"],
        degraded=counts["degraded"],
        gate_fired=counts["gate_fired"],
        gate_blocked=counts["gate_blocked"],
        gate_only=counts["gate_only"],
        with_excerpt=counts["with_excerpt"],
        without_excerpt=counts["records"] - counts["with_excerpt"],
        schema_versions=schema_versions,
        backends=backends,
        teacher_model_versions=versions,
        latency_ms=percentiles(latencies),
        notes=tuple(notes),
    )


def resolve_mode(requested: str, inspection: LogInspection) -> str:
    """Decide ``text`` vs ``features``, or explain precisely why neither is possible."""
    wanted = str(requested or "auto").lower()
    if wanted not in ("text", "features", "auto"):
        raise ExportError(f"unknown export mode {requested!r}; expected text, features, or auto")

    if inspection.records == 0:
        raise ExportError(
            f"no decision records found in {inspection.source}. Point --log at your decisions.jsonl "
            "(or the directory holding rotated logs) and make sure logging.enabled is true in the policy."
        )

    if wanted == "text" and not inspection.text_mode_possible:
        raise ExportError(_TEXT_MODE_IMPOSSIBLE.format(source=inspection.source, records=inspection.records))
    if wanted == "features":
        return "features"
    if wanted == "text":
        return "text"
    # auto: text when the log has it (higher ceiling), features otherwise.
    return "text" if inspection.text_mode_possible else "features"


_TEXT_MODE_IMPOSSIBLE = (
    # The head sentence stays its own literal so no physical line of this
    # multi-line message exceeds the line limit; the rendered text is unchanged.
    "text mode needs stored prompt excerpts, and none of the {records} records in {source} carry one.\n"
    "\n"
    """The decision log was written with `logging.excerpt_mode: hash` (the default), which retains a hash and the
deterministic request features but no text. There is nothing to train a text classifier on, and this export
refuses to silently train a weaker model than the one you asked for.

Two ways forward:
  1. Train on features now (zero text retention, lower accuracy ceiling):
       jev-route export --log {source} --mode features
  2. Start retaining redacted excerpts, collect more traffic, then export in text mode:
       set `logging.excerpt_mode: redacted` in your policy and re-run the router.
     Note that requests the local hard gate blocks are never written as text under any setting."""
)


# --------------------------------------------------------------------------- #
# Row building
# --------------------------------------------------------------------------- #
def _tags_for(record: DecisionRecord) -> tuple[str, ...]:
    """Provenance tags. Kept on the row so a trainer can weight or filter by origin."""
    tags: list[str] = []
    gate = record.decision.gate
    if gate.fired:
        tags.append("gate-fired")
    if gate.blocks_backend:
        tags.append("gate-blocked")
    if record.decision.backend == "gate":
        tags.append("gate-only")
    if gate.force_local:
        tags.append("force-local")
    if record.decision.escalated:
        tags.append("escalated")
    if record.decision.cached:
        tags.append("cached")
    return tuple(tags)


def build_row(
    record: DecisionRecord,
    *,
    mode: str,
    holdout_fraction: float,
    exported_at: str,
) -> TrainingRow:
    """One record -> one training row. Assumes the record has already passed the skip checks."""
    targets, _usable = _extract_targets(record)
    answers = record.decision.answers
    text = (record.excerpt or None) if mode == "text" else None
    if mode == "text" and text is not None and not text.strip():
        text = None
    return TrainingRow(
        request_id=record.request_id,
        split=assign_split(record.request_id, holdout_fraction),
        targets=targets,
        features=record.features,
        text=text,
        excerpt_hash=record.excerpt_hash,
        timestamp=record.timestamp,
        teacher={
            "backend": record.decision.backend,
            "model_version": record.decision.backend_model_version,
            "latency_ms": record.backend_latency_ms,
            "total_latency_ms": record.total_latency_ms,
            "degraded": record.decision.degraded,
            "tier": record.decision.tier,
            "model": record.decision.model,
            "rule_id": record.decision.rule_id,
            "effective_sensitivity": record.decision.effective_sensitivity,
            "effective_complexity": record.decision.effective_complexity,
            "escalated": list(record.decision.escalated),
            # Confidence as the backend reported it (or as the router derived it).
            # evaluate.py compares this against the student's derived confidence:
            # a distilled answer must be usable by the same on_uncertain floors.
            "confidence": {
                "complexity": answers.complexity.confidence,
                "sensitivity": answers.sensitivity.confidence,
                "domain": answers.domain.confidence,
                "pii": answers.pii.confidence,
                # Per head, because a backend may report one and derive another.
                # A noul has no flag: for pii the probability *is* the calibration.
                "reported": {
                    "complexity": bool(answers.complexity.confidence_reported),
                    "sensitivity": bool(answers.sensitivity.confidence_reported),
                    "domain": bool(answers.domain.confidence_reported),
                    "pii": False,
                },
            },
        },
        gate=record.decision.gate.to_dict(),
        context={
            "requested_model": record.requested_model or "",
            "metadata": dict(record.metadata or {}),
            # "classified" is the router's own notion: did a backend actually look
            # at this request? Policy replay needs it to reproduce the logged tier.
            "classified": record.decision.backend != "gate",
        },
        provenance={
            "row_schema": ROW_SCHEMA,
            "record_schema_version": record.schema_version,
            "record_kind": record.kind,
            "exported_at": exported_at,
            "export_mode": mode,
            "excerpt_retained": text is not None,
            "decision_schema_version": SCHEMA_VERSION,
        },
        tags=_tags_for(record),
        weight=1.0,
    )


def skip_reason(record: DecisionRecord, *, mode: str) -> str | None:
    """Why this record must not become a training row, or ``None`` if it should."""
    if record.schema_version != SCHEMA_VERSION:
        # A migration would go here. There is only one schema version so far, so
        # the honest behaviour is to refuse the row and say which version it was.
        return "unsupported_schema_version"
    if record.decision.degraded:
        return "degraded"
    if mode == "text" and not (record.excerpt or "").strip():
        return "no_text"
    _targets, usable = _extract_targets(record)
    if usable == 0:
        return "no_signal"
    return None


@dataclass(frozen=True)
class ExportStats:
    """Everything an operator needs to judge the dataset without re-reading it."""

    source: str
    requested_mode: str
    resolved_mode: str
    records_read: int = 0
    rows_written: int = 0
    train_rows: int = 0
    holdout_rows: int = 0
    holdout_fraction: float = 0.2
    skipped: dict[str, int] = field(default_factory=dict)
    gate_rows: int = 0
    gate_blocked_rows: int = 0
    usable_rows_per_head: dict[str, int] = field(default_factory=dict)
    label_support: dict[str, dict[str, int]] = field(default_factory=dict)
    teacher_backends: dict[str, int] = field(default_factory=dict)
    teacher_model_versions: dict[str, int] = field(default_factory=dict)
    record_schema_versions: dict[str, int] = field(default_factory=dict)
    teacher_latency_ms: dict[str, float] = field(default_factory=dict)
    contains_prompt_text: bool = False
    dataset_sha256: str = ""
    row_schema: str = ROW_SCHEMA
    exported_at: str = ""
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            out[key] = list(value) if isinstance(value, tuple) else value
        return out

    @property
    def skipped_total(self) -> int:
        return sum(self.skipped.values())

    def summary(self) -> str:
        """Human-readable line count, for the CLI and for tests to assert on."""
        lines = [
            f"exported {self.rows_written} rows from {self.records_read} records "
            f"({self.resolved_mode} mode): {self.train_rows} train / {self.holdout_rows} holdout",
        ]
        if self.skipped_total:
            detail = ", ".join(f"{n} {reason}" for reason, n in sorted(self.skipped.items()) if n)
            lines.append(f"skipped {self.skipped_total}: {detail}")
        if self.gate_rows:
            lines.append(
                f"{self.gate_rows} row(s) came from gate-fired requests "
                f"({self.gate_blocked_rows} never reached a backend); unusable heads are masked, not dropped"
            )
        for head in (*CHOICE_HEADS, "pii"):
            support = self.label_support.get(head) or {}
            lines.append(
                f"  {head:<12} usable={self.usable_rows_per_head.get(head, 0):<6} "
                + " ".join(f"{k}={v}" for k, v in sorted(support.items()))
            )
        lines.extend(f"warning: {w}" for w in self.warnings)
        return "\n".join(lines)


@dataclass(frozen=True)
class ExportResult:
    rows_path: Path
    stats_path: Path
    stats: ExportStats
    inspection: LogInspection


@dataclass
class _ExportTally:
    """Running counts over the rows written so far.

    :func:`export_dataset` reports nineteen numbers about the file it just wrote.
    Holding them in one mutable object is what lets the per-row accounting move
    into :func:`_tally_row`, so the write loop reads as what it is: skip, build,
    write, hash, count.
    """

    written: int = 0
    train_rows: int = 0
    holdout_rows: int = 0
    gate_rows: int = 0
    gate_blocked_rows: int = 0
    #: Rows repeating an ``excerpt_hash`` already written. The split is keyed on
    #: ``request_id``, so the same prompt can land on both sides of it.
    duplicate_excerpts: int = 0
    backends: dict[str, int] = field(default_factory=dict)
    versions: dict[str, int] = field(default_factory=dict)
    schemas: dict[str, int] = field(default_factory=dict)
    latencies: list[float] = field(default_factory=list)
    usable: dict[str, int] = field(default_factory=lambda: dict.fromkeys((*CHOICE_HEADS, "pii"), 0))
    support: dict[str, dict[str, int]] = field(
        default_factory=lambda: {head: {} for head in (*CHOICE_HEADS, "pii")}
    )
    seen_ids: set[str] = field(default_factory=set)
    seen_excerpts: set[str] = field(default_factory=set)


def _bump(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _tally_row(tally: _ExportTally, row: TrainingRow) -> None:
    """Fold one written row into the counts that become :class:`ExportStats`.

    ``seen_excerpts`` is updated here rather than in the caller because duplicate
    detection is a property of the rows written, and because the count has to be
    incremented *before* the hash is added, or the first repeat would be missed.
    """
    if row.excerpt_hash:
        if row.excerpt_hash in tally.seen_excerpts:
            tally.duplicate_excerpts += 1
        tally.seen_excerpts.add(row.excerpt_hash)

    tally.written += 1
    if row.split == "holdout":
        tally.holdout_rows += 1
    else:
        tally.train_rows += 1
    if "gate-fired" in row.tags:
        tally.gate_rows += 1
    if "gate-blocked" in row.tags:
        tally.gate_blocked_rows += 1
    _bump(tally.backends, row.teacher.get("backend", ""))
    _bump(tally.versions, str(row.teacher.get("model_version", "")))
    _bump(tally.schemas, str(row.provenance.get("record_schema_version", "")))
    if float(row.teacher.get("latency_ms") or 0.0) > 0:
        tally.latencies.append(float(row.teacher["latency_ms"]))
    for head in (*CHOICE_HEADS, "pii"):
        if row.targets.usable(head):
            tally.usable[head] += 1
            if head == "pii":
                _bump(tally.support[head], "true" if row.targets.pii["label"] else "false")
            else:
                _bump(tally.support[head], str(row.targets.get(head)["label"]))


def _dataset_paths(out_path: str | Path) -> tuple[Path, Path]:
    """Where the rows go, and where the stats sidecar goes.

    A directory argument means "``dataset.jsonl`` inside it", and any other suffix
    is coerced to ``.jsonl`` so :func:`load_dataset` can trust the file name. The
    parent directory is created here because both files are written into it.
    """
    target = Path(out_path)
    if target.is_dir():
        target = target / "dataset.jsonl"
    if target.suffix != ".jsonl":
        target = target.with_suffix(target.suffix + ".jsonl") if target.suffix else target.with_suffix(".jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target, target.with_name(target.stem + ".stats.json")


def _export_warnings(
    *, resolved: str, inspection: LogInspection, skipped: Mapping[str, int], tally: _ExportTally
) -> list[str]:
    """What an operator should be told about the dataset that was just written.

    Warnings rather than errors on purpose: none of these makes the file unusable,
    and each one changes how the graduation numbers should be read. They land in
    ``ExportStats.warnings`` and are printed by the CLI.
    """
    warnings: list[str] = []
    if resolved == "text" and inspection.without_excerpt:
        warnings.append(
            f"{skipped.get('no_text', 0)} record(s) carry no excerpt and were excluded from the text dataset. "
            "The router never writes text for gate-blocked requests, so those rows can only be trained in "
            "features mode; use --mode features to include them."
        )
    if tally.duplicate_excerpts:
        warnings.append(
            f"{tally.duplicate_excerpts} row(s) repeat an excerpt_hash already seen. The split is keyed on "
            "request_id, so identical prompts can land on both sides and make holdout metrics optimistic. "
            "Deduplicate traffic or read the numbers with that in mind."
        )
    if resolved == "text":
        warnings.append(
            "this dataset contains redacted prompt text. Treat the file with the same care as the prompts "
            "themselves, and delete it after training if you do not need it."
        )
    if any(tally.usable[head] == 0 for head in (*CHOICE_HEADS, "pii")):
        empty = [head for head in (*CHOICE_HEADS, "pii") if tally.usable[head] == 0]
        warnings.append(
            f"no usable signal for head(s) {empty}: every teacher distribution for them was uniform. "
            "Those heads will be trained on nothing and will stay at their prior."
        )
    if tally.holdout_rows == 0:
        warnings.append(
            "the holdout split is empty, so evaluate and graduate have nothing to measure against. "
            "Collect more traffic or lower --holdout-fraction."
        )
    return warnings


def export_dataset(
    source: str | Path | Iterable[DecisionRecord],
    out_path: str | Path,
    *,
    mode: ExportMode = "auto",
    holdout_fraction: float = 0.2,
    include_gate_rows: bool = True,
    skip_duplicate_request_ids: bool = True,
    pattern: str = "decisions*.jsonl",
    progress: Any = None,
) -> ExportResult:
    """Export a decision log to a training dataset (JSONL + a stats sidecar).

    Two passes over the log by design: the first measures what the log can support
    (so ``mode="auto"`` and the text-mode error are both decided from evidence),
    the second writes. Memory stays flat no matter how big the log gets.
    """
    inspection = inspect_log(source, pattern=pattern)
    resolved = resolve_mode(mode, inspection)
    if not (0.0 <= float(holdout_fraction) < 0.9):
        raise ExportError(f"holdout_fraction must be in [0, 0.9), got {holdout_fraction}")

    target, stats_path = _dataset_paths(out_path)

    exported_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    skipped: dict[str, int] = {}
    tally = _ExportTally()
    digest = hashlib.sha256()

    with open(target, "w", encoding="utf-8") as handle:
        for record in iter_source_records(source, pattern=pattern):
            reason = skip_reason(record, mode=resolved)
            if reason is None and skip_duplicate_request_ids and record.request_id in tally.seen_ids:
                reason = "duplicate_request_id"
            if reason is None and not include_gate_rows and record.decision.gate.blocks_backend:
                reason = "gate_blocked_excluded"
            if reason is not None:
                _bump(skipped, reason)
                continue

            row = build_row(record, mode=resolved, holdout_fraction=holdout_fraction, exported_at=exported_at)
            line = row.to_json() + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
            tally.seen_ids.add(record.request_id)
            _tally_row(tally, row)
            if progress is not None and tally.written % 1000 == 0:
                progress(tally.written)

    if tally.written == 0:
        raise ExportError(
            f"every one of the {inspection.records} records in {inspection.source} was skipped "
            f"({skipped}); nothing to train on. "
            + (
                "All of them were degraded (the backend was down), so the log holds uniform distributions "
                "and no signal. Route traffic while the backend is healthy and re-export."
                if skipped.get("degraded")
                else "Check the skip reasons above."
            )
        )
    warnings = _export_warnings(resolved=resolved, inspection=inspection, skipped=skipped, tally=tally)

    stats = ExportStats(
        source=str(source),
        requested_mode=str(mode),
        resolved_mode=resolved,
        records_read=inspection.records,
        rows_written=tally.written,
        train_rows=tally.train_rows,
        holdout_rows=tally.holdout_rows,
        holdout_fraction=float(holdout_fraction),
        skipped=skipped,
        gate_rows=tally.gate_rows,
        gate_blocked_rows=tally.gate_blocked_rows,
        usable_rows_per_head=tally.usable,
        label_support=tally.support,
        teacher_backends=tally.backends,
        teacher_model_versions=tally.versions,
        record_schema_versions=tally.schemas,
        teacher_latency_ms=percentiles(tally.latencies),
        contains_prompt_text=resolved == "text",
        dataset_sha256=digest.hexdigest(),
        exported_at=exported_at,
        warnings=tuple(warnings),
    )
    stats_path.write_text(
        json.dumps(
            {"inspection": inspection.as_dict(), "stats": stats.as_dict()},
            sort_keys=True,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return ExportResult(rows_path=target, stats_path=stats_path, stats=stats, inspection=inspection)


# --------------------------------------------------------------------------- #
# Reading a dataset back
# --------------------------------------------------------------------------- #
def _dataset_file(path: str | Path) -> Path:
    p = Path(path)
    if p.is_dir():
        for candidate in ("dataset.jsonl", "train.jsonl"):
            if (p / candidate).exists():
                return p / candidate
        found = sorted(p.glob("*.jsonl"))
        if not found:
            raise ExportError(f"{p} holds no .jsonl dataset; run `jev-route export` to export one")
        return found[0]
    if not p.exists():
        raise ExportError(
            f"dataset not found: {p}. Export one with `jev-route export --log <decisions.jsonl> --out {p}`."
        )
    return p


@dataclass(frozen=True)
class Dataset:
    """An exported dataset held in memory, with the mode it was exported in.

    Rows are the unit of everything downstream: the trainer vectorizes
    :meth:`texts` or :meth:`feature_rows` depending on :attr:`mode`, and evaluate
    reads the soft targets straight back off the rows. Keeping one object for both
    is what stops the trainer and the evaluator from drifting apart about which
    rows are usable.
    """

    rows: tuple[TrainingRow, ...]
    mode: str
    stats: dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[TrainingRow]:
        return iter(self.rows)

    def __getitem__(self, index: int) -> TrainingRow:
        return self.rows[index]

    def split(self, name: str) -> Dataset:
        if name not in ("train", "holdout"):
            raise ExportError(f"unknown split {name!r}; expected 'train' or 'holdout'")
        return Dataset(
            rows=tuple(r for r in self.rows if r.split == name),
            mode=self.mode,
            stats=dict(self.stats),
            source=self.source,
        )

    @property
    def train(self) -> Dataset:
        return self.split("train")

    @property
    def holdout(self) -> Dataset:
        return self.split("holdout")

    def texts(self) -> list[str]:
        """Redacted excerpts, in row order. Raises in features mode."""
        if self.mode != "text":
            raise ExportError(
                f"this dataset was exported in {self.mode!r} mode and carries no prompt text; "
                "use feature_rows() or re-export with --mode text"
            )
        return [r.text or "" for r in self.rows]

    def feature_rows(self) -> list[RequestFeatures]:
        return [r.features for r in self.rows]

    def usable_counts(self) -> dict[str, int]:
        return {
            head: sum(1 for r in self.rows if r.targets.usable(head))
            for head in (*CHOICE_HEADS, "pii")
        }

    def label_support(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {head: {} for head in (*CHOICE_HEADS, "pii")}
        for row in self.rows:
            for head in CHOICE_HEADS:
                if row.targets.usable(head):
                    label = str(row.targets.get(head)["label"])
                    out[head][label] = out[head].get(label, 0) + 1
            if row.targets.usable("pii"):
                label = "true" if row.targets.pii["label"] else "false"
                out["pii"][label] = out["pii"].get(label, 0) + 1
        return out

    def teacher_latencies(self) -> list[float]:
        return [float(r.teacher.get("latency_ms") or 0.0) for r in self.rows]

    def teacher_model_versions(self) -> tuple[str, ...]:
        return tuple(sorted({str(r.teacher.get("model_version") or "") for r in self.rows} - {""}))

    def describe(self) -> str:
        usable = self.usable_counts()
        support = self.label_support()
        lines = [
            f"dataset: {len(self.rows)} rows, mode={self.mode}, source={self.source}",
            f"  train={sum(1 for r in self.rows if r.split == 'train')} "
            f"holdout={sum(1 for r in self.rows if r.split == 'holdout')} "
            f"gate_rows={sum(1 for r in self.rows if r.is_gate_row)}",
        ]
        for head in (*CHOICE_HEADS, "pii"):
            lines.append(
                f"  {head:<12} usable={usable[head]:<6} "
                + " ".join(f"{k}={v}" for k, v in sorted(support[head].items()))
            )
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        """Write rows back to JSONL. Used by tests and by dataset subsetting."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            for row in self.rows:
                handle.write(row.to_json() + "\n")
        return target


def load_dataset(path: str | Path, *, split: str | None = None, mode: str | None = None) -> Dataset:
    """Load an exported dataset, optionally filtering to one split."""
    rows_file = _dataset_file(path)
    stats: dict[str, Any] = {}
    sidecar = rows_file.with_name(rows_file.stem + ".stats.json")
    if sidecar.exists():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            stats = dict(payload.get("stats") or payload) if isinstance(payload, Mapping) else {}
        except json.JSONDecodeError:
            stats = {}

    rows: list[TrainingRow] = []
    with open(rows_file, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ExportError(f"{rows_file} contains a malformed line: {exc}") from exc
            row = TrainingRow.from_dict(data)
            if split is None or row.split == split:
                rows.append(row)

    resolved_mode = mode or str(stats.get("resolved_mode") or "")
    if resolved_mode not in ("text", "features"):
        # Infer from the rows themselves, so a dataset file is usable even if its
        # sidecar was deleted. Text wins when present: that is what was trained on.
        resolved_mode = "text" if any(r.text for r in rows) else "features"
    dataset = Dataset(rows=tuple(rows), mode=resolved_mode, stats=stats, source=str(rows_file))
    if not dataset.rows:
        raise ExportError(
            f"{rows_file} contains no rows"
            + (f" in split {split!r}" if split else "")
            + "; re-export the decision log"
        )
    return dataset


__all__ = [
    "CHOICE_HEADS",
    "LADDERS",
    "ROW_SCHEMA",
    "SUPPORTED_ROW_SCHEMAS",
    "Dataset",
    "ExportError",
    "ExportMode",
    "ExportResult",
    "ExportStats",
    "LogInspection",
    "TargetSet",
    "TrainingRow",
    "assign_split",
    "build_row",
    "export_dataset",
    "inspect_log",
    "iter_source_records",
    "load_dataset",
    "percentiles",
    "resolve_mode",
    "skip_reason",
    "split_bucket",
]
