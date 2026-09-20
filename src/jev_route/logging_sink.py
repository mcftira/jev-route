"""Decision log: the append-only record of every routing decision.

This is not an audit trail that happens to be useful. **It is the training set.**
``jev_route.distill`` reads these records and produces the local model that
eventually replaces the cloud backend, which means the log has to be held to
dataset standards rather than logging standards:

* **Schema-versioned.** Every record carries ``schema_version``. A field may be
  added; it may not be silently renamed or retyped. Old records must stay
  readable, because the value of the log is that it accumulates.
* **Soft targets, not just argmax.** The full probability distribution behind
  every answer is stored. Distilling from argmax labels throws away exactly the
  calibration that made the bootstrap worth paying for.
* **Append-only.** Records are never rewritten in place. Rotation creates a new
  file; it does not edit the old one.
* **Provable without being leaky.** A gate finding stores a hash of the matched
  span, so the log can prove a card number was present without keeping it.

A second, physically separate stream carries the gate's refusals
(:class:`~jev_route.schema.GateBlockRecord`, built by :func:`build_blocked_sink`):
which detectors fired, on what shape of request, how often, and never what the
text was. It is not part of the training dataset and its filename is deliberately
not matched by :func:`iter_all_records`' glob, so no export can sweep it up.

The privacy contract, stated plainly: by default a record contains the hash of the
excerpt and deterministic features, and **no prompt text**. Setting
``logging.excerpt_mode: redacted`` stores the redacted excerpt, which enables
text-based distillation and is a real increase in what you retain. Requests the
local gate blocked from reaching a cloud backend are never written as text under
any setting -- the gate's judgement that the content is too sensitive to send
somewhere also applies to your own log file.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .gate_semantic import BlockedMetadataPolicy
from .schema import GATE_BLOCK_RECORD_KIND, RECORD_KIND, DecisionRecord, GateBlockRecord

#: What a sink may be handed. Both are JSONL records with a ``kind``
#: discriminator and a ``to_json()``; they are separate streams with separate
#: readers, and :func:`iter_records` / :func:`iter_gate_blocks` each yield only
#: their own kind, so a shared file stays filterable rather than merely shared.
LoggedRecord = DecisionRecord | GateBlockRecord


@runtime_checkable
class DecisionSink(Protocol):
    """Where decision records go. Implementations must tolerate concurrent writes."""

    def write(self, record: LoggedRecord) -> None: ...
    def close(self) -> None: ...


class NullSink:
    """Discards records. For tests, and for operators who want routing without logging."""

    def __init__(self) -> None:
        self.count = 0

    def write(self, record: LoggedRecord) -> None:
        self.count += 1

    def close(self) -> None:
        return None

    def stats(self) -> dict[str, Any]:
        return {"backend": "null", "written": self.count}


class JsonlSink:
    """Append-only JSON Lines file with size-based rotation.

    One ``write()`` syscall per record, under a lock. On a regular file opened
    with ``O_APPEND`` that is effectively atomic per line, which is why a
    multi-worker proxy can share one path without interleaving records. Rotation
    renames the current file rather than truncating it: nothing is destroyed.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_bytes: int = 256 * 1024 * 1024,
        flush_every: int = 1,
        create_dirs: bool = True,
    ) -> None:
        self.path = Path(path)
        self.max_bytes = max(0, int(max_bytes))
        self.flush_every = max(1, int(flush_every))
        self._lock = threading.Lock()
        self._written = 0
        self._rotations = 0
        self._errors = 0
        self._handle: Any = None
        if create_dirs and self.path.parent != Path(""):
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def _open(self) -> Any:
        if self._handle is None:
            # Line-buffered text append. Buffering is handled by flush_every,
            # because a lost tail on crash costs training rows.
            self._handle = open(self.path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
        return self._handle

    def write(self, record: LoggedRecord) -> None:
        line = record.to_json() + "\n"
        with self._lock:
            try:
                handle = self._open()
                if self.max_bytes and self.path.exists() and self.path.stat().st_size + len(line) > self.max_bytes:
                    self._rotate_locked()
                    handle = self._open()
                handle.write(line)
                self._written += 1
                if self._written % self.flush_every == 0:
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError:
                # A full disk must not take down the request path. Count it and
                # carry on: routing still works, the log has a hole.
                self._errors += 1

    def _rotate_locked(self) -> None:
        assert self._handle is not None
        self._handle.close()
        self._handle = None
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        target = self.path.with_name(f"{self.path.stem}.{stamp}{self.path.suffix}")
        suffix = 1
        while target.exists():
            target = self.path.with_name(f"{self.path.stem}.{stamp}.{suffix}{self.path.suffix}")
            suffix += 1
        self.path.rename(target)
        self._rotations += 1

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                try:
                    self._handle.flush()
                    self._handle.close()
                except OSError:
                    self._errors += 1
                self._handle = None

    def stats(self) -> dict[str, Any]:
        return {
            "backend": "jsonl",
            "path": str(self.path),
            "written": self._written,
            "rotations": self._rotations,
            "write_errors": self._errors,
            "size_bytes": self.path.stat().st_size if self.path.exists() else 0,
        }


class CallbackSink:
    """Hand each record to a callable. For shipping to a warehouse or a stream."""

    def __init__(self, callback: Callable[[LoggedRecord], None]) -> None:
        self.callback = callback
        self._written = 0
        self._errors = 0

    def write(self, record: LoggedRecord) -> None:
        try:
            self.callback(record)
            self._written += 1
        except Exception:  # a broken sink must not break routing
            self._errors += 1

    def close(self) -> None:
        return None

    def stats(self) -> dict[str, Any]:
        return {"backend": "callback", "written": self._written, "errors": self._errors}


class CompositeSink:
    """Fan out to several sinks. A failure in one does not affect the others."""

    def __init__(self, sinks: Iterable[DecisionSink]) -> None:
        self.sinks = tuple(sinks)
        self._errors = 0

    def write(self, record: LoggedRecord) -> None:
        for sink in self.sinks:
            # Guarded per member, because the whole reason to compose sinks is
            # that a warehouse shipper being down should not cost you the local
            # JSONL copy too. An unguarded loop starves every member after the
            # first failure -- and propagates into Router._log, taking the
            # request path down with it.
            try:
                sink.write(record)
            except Exception:
                self._errors += 1

    def close(self) -> None:
        for sink in self.sinks:
            try:
                sink.close()
            except Exception:
                self._errors += 1

    def stats(self) -> dict[str, Any]:
        return {
            "backend": "composite",
            "member_errors": self._errors,
            "sinks": [s.stats() for s in self.sinks if hasattr(s, "stats")],
        }


def build_sink(config: Mapping[str, Any] | None) -> DecisionSink:
    """Construct the sink named by ``logging`` policy config."""
    cfg = dict(config or {})
    if not cfg.get("enabled", True):
        return NullSink()
    path = cfg.get("path")
    if not path:
        return NullSink()
    return JsonlSink(
        path,
        max_bytes=int(cfg.get("max_bytes", 256 * 1024 * 1024)),
        flush_every=int(cfg.get("flush_every", 1)),
    )


#: Filename of the refusal stream. Deliberately NOT matched by the
#: ``decisions*.jsonl`` default of :func:`iter_all_records`: the refusal stream is
#: for tuning detectors and must never be swept into a training export by a glob.
BLOCKED_METADATA_FILENAME = "gate-blocks.jsonl"


def default_blocked_metadata_path(logging_path: Any) -> Path | None:
    """Where the refusal stream lives when the operator did not say.

    A sibling of the decision log rather than a fixed relative path, so a
    deployment that puts its log on a particular volume puts this next to it -- and
    a deployment with no log path at all gets ``None`` and no refusal stream either,
    because "logging is off" should mean logging is off.

    Anything that is not a path yields ``None`` rather than raising. ``logging.path``
    is operator config and this package has seen it hold a dict (an unsubstituted
    YAML flow mapping parses happily and only fails when somebody treats it as a
    filename). Refusal telemetry is never a reason for a router to refuse to start.
    """
    if not logging_path or not isinstance(logging_path, (str, os.PathLike)):
        return None
    path = Path(logging_path)
    return path.parent / BLOCKED_METADATA_FILENAME


def build_blocked_sink(
    config: BlockedMetadataPolicy | None, *, logging_config: Mapping[str, Any] | None = None
) -> DecisionSink:
    """Construct the sink for :class:`~jev_route.schema.GateBlockRecord`.

    Separate from :func:`build_sink` on purpose. The decision log is the training
    dataset; the refusal stream is rule-improvement telemetry about content that
    was judged too sensitive to send anywhere. Keeping them in different files
    means no reader, glob, or future exporter can pick up the wrong one by
    accident -- the separation is physical, not a filter somebody has to remember.
    """
    cfg = config or BlockedMetadataPolicy()
    if not cfg.enabled:
        return NullSink()
    log_cfg = dict(logging_config or {})
    if not log_cfg.get("enabled", True):
        return NullSink()
    path = Path(cfg.path) if cfg.path else default_blocked_metadata_path(log_cfg.get("path"))
    if path is None:
        return NullSink()
    try:
        return JsonlSink(
            path,
            max_bytes=int(log_cfg.get("max_bytes", 256 * 1024 * 1024)),
            flush_every=int(log_cfg.get("flush_every", 1)),
        )
    except (OSError, TypeError, ValueError):
        # An unwritable destination costs the operator telemetry, not requests.
        # Same posture JsonlSink takes toward a full disk on write.
        return NullSink()


# --------------------------------------------------------------------------- #
# Reading the log back
# --------------------------------------------------------------------------- #
def _iter_kind(path: str | Path, kind: str) -> Iterator[Mapping[str, Any]]:
    """Yield the payloads of one ``kind`` from a JSONL file, skipping foreign lines.

    Tolerates a partial trailing line from an interrupted write, because a log that
    cannot be read after a crash is worse than no log. Filtering here rather than in
    the caller is what makes a mixed stream safe: every reader gets its own kind and
    nothing else, whatever else shares the file.
    """
    p = Path(path)
    if not p.exists():
        return
    with open(p, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, Mapping) or data.get("kind") != kind:
                continue
            yield data


def iter_records(path: str | Path) -> Iterator[DecisionRecord]:
    """Yield every decision record in a JSONL file, skipping foreign lines."""
    for data in _iter_kind(path, RECORD_KIND):
        yield DecisionRecord.from_dict(data)


def iter_gate_blocks(path: str | Path) -> Iterator[GateBlockRecord]:
    """Yield every gate-refusal record in a JSONL file, skipping foreign lines.

    The refusal stream's reader. Kept apart from :func:`iter_records` so that the
    training pipeline -- which reads decisions -- cannot ingest refusal telemetry,
    and a rule-improvement tool cannot mistake a decision for a refusal.
    """
    for data in _iter_kind(path, GATE_BLOCK_RECORD_KIND):
        yield GateBlockRecord.from_dict(data)


def iter_all_records(directory: str | Path, *, pattern: str = "decisions*.jsonl") -> Iterator[DecisionRecord]:
    """Yield records across a directory of rotated log files, oldest first."""
    d = Path(directory)
    if not d.exists():
        return
    for path in sorted(d.glob(pattern), key=lambda p: p.stat().st_mtime):
        yield from iter_records(path)


__all__ = [
    "BLOCKED_METADATA_FILENAME",
    "CallbackSink",
    "CompositeSink",
    "DecisionSink",
    "JsonlSink",
    "LoggedRecord",
    "NullSink",
    "build_blocked_sink",
    "build_sink",
    "default_blocked_metadata_path",
    "iter_all_records",
    "iter_gate_blocks",
    "iter_records",
]
