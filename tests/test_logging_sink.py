"""Tests for :mod:`jev_route.logging_sink` -- the decision log, which is the dataset.

Most logging libraries get tested for "does a line appear". This module cannot be
tested that way, because it is not a log: it is the training set that
``jev_route.distill`` later turns into the local model. A log with a hole is an
inconvenience. A dataset with a hole is a silently biased model. So the properties
pinned here are the ones the module docstring promises, and each one is a promise
somebody downstream is relying on:

* **Schema-versioned.** Every line carries ``schema_version`` and ``kind``. Old
  records must stay readable, because the value of the log is that it accumulates.
* **Soft targets, not just argmax.** The whole distribution behind every answer is
  on disk. This is the single most important assertion in the file: if a future
  refactor writes ``{"complexity": "hard"}`` instead of the four probabilities, the
  suite still "passes" every naive logging test while destroying the calibration
  that made the bootstrap worth paying for.
* **Append-only.** Records are never rewritten in place; rotation renames a file
  rather than truncating it. Two tests assert this at the byte level, because
  "append-only" is a claim about bytes and not about intent.
* **A sink must never break routing.** Every failure mode -- missing directory,
  path that is a directory, disk full, a callback that raises -- is counted in
  ``stats()`` and swallowed. The request path is more important than the log.
* **Provable without being leaky.** A gate finding stores a hash of the matched
  span, so a record can prove a card number was present without keeping it.

Reads are tested as adversarially as writes: a log that cannot be parsed after a
crash is worse than no log, so ``iter_records`` has to survive truncated trailing
lines, blank lines, and foreign event kinds sharing the same stream.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from jev_route.gate import HardGate
from jev_route.logging_sink import (
    CallbackSink,
    CompositeSink,
    DecisionSink,
    JsonlSink,
    NullSink,
    build_sink,
    iter_all_records,
    iter_records,
)
from jev_route.prompts import compute_features, hash_text
from jev_route.schema import (
    COMPLEXITY_LEVELS,
    DOMAINS,
    RECORD_KIND,
    SCHEMA_VERSION,
    SENSITIVITY_LEVELS,
    ChoiceAnswer,
    DecisionAnswers,
    DecisionRecord,
    GateVerdict,
    NoulAnswer,
    RoutingDecision,
)

# --------------------------------------------------------------------------- #
# Fixtures-in-waiting: one record shape, reused everywhere
# --------------------------------------------------------------------------- #
#: Non-uniform on purpose. A helper that produced ``{a: 1.0}`` would make every
#: "the full distribution is on disk" assertion vacuous: one key would look
#: exactly like a preserved distribution of one.
COMPLEXITY_PROBS: dict[str, float] = {"trivial": 0.11, "standard": 0.55, "hard": 0.29, "frontier": 0.05}
SENSITIVITY_PROBS: dict[str, float] = {"public": 0.2, "internal": 0.5, "confidential": 0.24, "regulated": 0.06}
DOMAIN_PROBS: dict[str, float] = {
    "code": 0.1,
    "writing": 0.55,
    "analysis": 0.2,
    "chat": 0.05,
    "data-extraction": 0.1,
}
PII_VALUE = 0.37

PROMPT_TEXT = "Summarize the platform team status update and draft the release note."

#: Rotated siblings are named ``<stem>.<UTC stamp>[.<n>].jsonl``.
ROTATED_NAME = re.compile(r"^decisions\.\d{8}T\d{6}(?:\.\d+)?\.jsonl$")

#: A record is ~1.7 KB, so 200 bytes forces a rotation on every single write.
TINY_MAX_BYTES = 200

#: 256 MiB, the documented default. Referenced rather than retyped so that a
#: change to the default shows up as a test failure instead of a silent drift.
DEFAULT_MAX_BYTES = 256 * 1024 * 1024


def make_answers(
    *,
    complexity: Mapping[str, float] | None = None,
    sensitivity: Mapping[str, float] | None = None,
    domain: Mapping[str, float] | None = None,
    pii: float = PII_VALUE,
) -> DecisionAnswers:
    """Answers with real, non-degenerate distributions."""
    cx = dict(complexity or COMPLEXITY_PROBS)
    sens = dict(sensitivity or SENSITIVITY_PROBS)
    dom = dict(domain or DOMAIN_PROBS)
    return DecisionAnswers(
        complexity=_choice(cx, COMPLEXITY_LEVELS),
        sensitivity=_choice(sens, SENSITIVITY_LEVELS),
        pii=NoulAnswer(value=pii),
        domain=_choice(dom, DOMAINS),
    )


def _choice(probabilities: Mapping[str, float], ladder: Sequence[str]) -> ChoiceAnswer:
    """A :class:`ChoiceAnswer` whose confidence is derived, not asserted into being."""
    if tuple(probabilities) != tuple(ladder):
        raise ValueError(f"distribution keys {tuple(probabilities)} != ladder {tuple(ladder)}")
    values = sorted(probabilities.values(), reverse=True)
    return ChoiceAnswer(
        choice=max(probabilities.items(), key=lambda kv: kv[1])[0],
        probabilities=dict(probabilities),
        confidence=round(values[0] - values[1], 6),
        confidence_reported=False,
    )


def make_record(
    index: int = 0,
    *,
    text: str = PROMPT_TEXT,
    answers: DecisionAnswers | None = None,
    gate: GateVerdict | None = None,
    excerpt: str | None = None,
) -> DecisionRecord:
    """One complete record, with a distinct ``request_id`` per ``index``."""
    decision = RoutingDecision(
        tier="cheap",
        model="cheap-model",
        rule_id="default",
        reason="routine task, no sensitivity signal",
        answers=answers if answers is not None else make_answers(),
        gate=gate if gate is not None else GateVerdict(),
        backend="mock",
        backend_model_version="mock-1.0.0",
        effective_sensitivity="internal",
        effective_complexity="standard",
        escalated=("sensitivity",),
        degraded=False,
        degrade_reason=None,
        latency_ms=1.5,
        cached=False,
    )
    return DecisionRecord(
        request_id=f"req-{index}",
        timestamp=f"2026-01-01T00:00:{index:02d}Z",
        decision=decision,
        features=compute_features(text),
        excerpt_hash=hash_text(text),
        backend_latency_ms=1.0,
        total_latency_ms=2.5,
        questions_sent={"complexity": {"type": "choice"}},
        excerpt=excerpt,
        metadata={"tenant": "acme"},
    )


def tmp_files(path: Path) -> list[Path]:
    """Every ``decisions*.jsonl`` in ``path``'s directory, sorted by name."""
    return sorted(path.parent.glob("decisions*.jsonl"))


def raw_lines(path: Path) -> list[str]:
    """The file as bytes-turned-lines, with no parsing and no filtering."""
    return path.read_text(encoding="utf-8").splitlines()


def raw_records(path: Path) -> list[dict[str, Any]]:
    """Every line of ``path``, parsed as JSON. Fails loudly if one is not JSON."""
    return [json.loads(line) for line in raw_lines(path)]


def answers_on_disk(path: Path, index: int = 0) -> dict[str, Any]:
    """The ``answers`` object of one line, read straight off the disk."""
    return raw_records(path)[index]["decision"]["answers"]  # type: ignore[no-any-return]


class _HandleSpy:
    """Wraps a real file handle and counts ``flush()``/``close()``.

    Reaching into ``JsonlSink._handle`` is ugly, and deliberate: "close() flushes"
    cannot be observed from the outside, because the sink opens its file
    line-buffered (``buffering=1``) so the bytes are already on disk before
    ``close()`` runs. Counting the flush call is the only honest way to pin the
    promise without faking a disk-full condition.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.flushes = 0
        self.closes = 0

    def write(self, data: str) -> int:
        return int(self._inner.write(data))

    def flush(self) -> None:
        self.flushes += 1
        self._inner.flush()

    def fileno(self) -> int:
        return int(self._inner.fileno())

    def close(self) -> None:
        self.closes += 1
        self._inner.close()


class _BrokenHandle:
    """A handle whose every operation fails, for the close-error path."""

    def flush(self) -> None:
        raise OSError(5, "Input/output error")

    def close(self) -> None:
        raise OSError(5, "Input/output error")


# --------------------------------------------------------------------------- #
# The protocol
# --------------------------------------------------------------------------- #
class TestSinkProtocol:
    """``DecisionSink`` is structural, so conformance is worth asserting explicitly.

    The router only ever holds a ``DecisionSink``; nothing type-checks the concrete
    classes at runtime. A sink that quietly lost ``close()`` would still import and
    would only fail during teardown in production.
    """

    def test_every_shipped_sink_satisfies_the_protocol(self, tmp_path: Path) -> None:
        sinks: list[Any] = [
            NullSink(),
            CallbackSink(lambda record: None),
            CompositeSink([]),
            JsonlSink(tmp_path / "decisions.jsonl"),
        ]
        for sink in sinks:
            assert isinstance(sink, DecisionSink), type(sink).__name__

    def test_a_recording_sink_from_conftest_also_satisfies_it(self, sink: Any) -> None:
        # The conftest double is what most router tests write into, so it has to be
        # a legitimate DecisionSink or those tests are exercising a different contract.
        assert isinstance(sink, DecisionSink)


# --------------------------------------------------------------------------- #
# NullSink
# --------------------------------------------------------------------------- #
class TestNullSink:
    """The sink for operators who want routing without logging. It must still count."""

    def test_write_counts_records(self) -> None:
        sink = NullSink()
        sink.write(make_record(0))
        sink.write(make_record(1))
        assert sink.count == 2

    def test_stats_reports_backend_and_written(self) -> None:
        sink = NullSink()
        sink.write(make_record(0))
        # Exact dict, not a subset: the CLI prints this and an unexpected key
        # would land in an operator's monitoring payload.
        assert sink.stats() == {"backend": "null", "written": 1}

    def test_a_fresh_sink_has_written_nothing(self) -> None:
        assert NullSink().stats() == {"backend": "null", "written": 0}

    def test_close_returns_none_and_is_idempotent(self) -> None:
        sink = NullSink()
        assert sink.close() is None
        assert sink.close() is None

    def test_a_null_sink_has_nowhere_to_write(self, tmp_path: Path) -> None:
        sink = NullSink()
        for i in range(3):
            sink.write(make_record(i))
        sink.close()
        # "Null" has to mean null. No path attribute and no path in stats, so there
        # is no way for it to be retaining anything, not even relative to the cwd.
        assert not hasattr(sink, "path")
        assert set(sink.stats()) == {"backend", "written"}
        assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
# CallbackSink
# --------------------------------------------------------------------------- #
class TestCallbackSink:
    """Hand records to a callable -- and never let the callable break routing."""

    def test_the_callback_receives_every_record_in_order(self) -> None:
        seen: list[DecisionRecord] = []
        sink = CallbackSink(seen.append)
        for i in range(3):
            sink.write(make_record(i))
        assert [r.request_id for r in seen] == ["req-0", "req-1", "req-2"]

    def test_the_callback_receives_the_record_object_not_a_copy(self) -> None:
        seen: list[DecisionRecord] = []
        record = make_record(0)
        CallbackSink(seen.append).write(record)
        # Identity, not equality: a sink that re-parsed the record would lose the
        # mapping types and quietly change what a warehouse callback receives.
        assert seen[0] is record

    def test_stats_counts_successful_writes(self) -> None:
        sink = CallbackSink(lambda record: None)
        sink.write(make_record(0))
        sink.write(make_record(1))
        assert sink.stats() == {"backend": "callback", "written": 2, "errors": 0}

    def test_a_raising_callback_does_not_propagate(self) -> None:
        def explode(record: DecisionRecord) -> None:
            raise RuntimeError("warehouse unreachable")

        sink = CallbackSink(explode)
        sink.write(make_record(0))  # must not raise

    def test_a_raising_callback_is_counted_as_an_error(self) -> None:
        def explode(record: DecisionRecord) -> None:
            raise RuntimeError("warehouse unreachable")

        sink = CallbackSink(explode)
        sink.write(make_record(0))
        sink.write(make_record(1))
        # A failed write is not a written record: counting it as one would make the
        # log look complete while the warehouse silently has a hole.
        assert sink.stats() == {"backend": "callback", "written": 0, "errors": 2}

    def test_routing_continues_after_a_callback_failure(self) -> None:
        seen: list[DecisionRecord] = []
        state = {"fail": True}

        def flaky(record: DecisionRecord) -> None:
            if state["fail"]:
                raise RuntimeError("transient")
            seen.append(record)

        sink = CallbackSink(flaky)
        sink.write(make_record(0))
        state["fail"] = False
        sink.write(make_record(1))
        assert [r.request_id for r in seen] == ["req-1"]
        assert sink.stats() == {"backend": "callback", "written": 1, "errors": 1}

    def test_a_base_exception_is_not_swallowed(self) -> None:
        def interrupt(record: DecisionRecord) -> None:
            raise KeyboardInterrupt

        sink = CallbackSink(interrupt)
        # Verified behaviour, and the right one: ``write`` catches ``Exception``,
        # not ``BaseException``. A sink that ate Ctrl-C or ``SystemExit`` would make
        # a hung warehouse callback uninterruptible, which is worse than a lost row.
        with pytest.raises(KeyboardInterrupt):
            sink.write(make_record(0))
        assert sink.stats()["errors"] == 0

    def test_close_returns_none(self) -> None:
        assert CallbackSink(lambda record: None).close() is None


# --------------------------------------------------------------------------- #
# CompositeSink
# --------------------------------------------------------------------------- #
class TestCompositeSink:
    """Fan-out: file plus warehouse plus stream, from one router."""

    def test_write_reaches_every_member(self, sink: Any) -> None:
        null = NullSink()
        composite = CompositeSink([null, sink])
        composite.write(make_record(0))
        assert null.count == 1
        assert sink.count == 1

    def test_members_are_called_in_construction_order(self) -> None:
        order: list[str] = []

        class Named:
            def __init__(self, label: str) -> None:
                self.label = label

            def write(self, record: DecisionRecord) -> None:
                order.append(self.label)

            def close(self) -> None:
                order.append(f"close:{self.label}")

        CompositeSink([Named("a"), Named("b"), Named("c")]).write(make_record(0))
        # Order matters for operators who put the durable file first: on a crash the
        # file should be the one that got the record.
        assert order == ["a", "b", "c"]

    def test_close_closes_every_member(self, sink: Any) -> None:
        null = NullSink()
        CompositeSink([null, sink]).close()
        assert sink.closed == 1

    def test_stats_lists_member_stats_in_order(self, sink: Any) -> None:
        null = NullSink()
        null.write(make_record(0))
        stats = CompositeSink([null, sink]).stats()
        assert stats["backend"] == "composite"
        assert stats["sinks"] == [
            {"backend": "null", "written": 1},
            {"backend": "recording", "written": 0},
        ]

    def test_members_without_stats_are_omitted(self, sink: Any) -> None:
        class Bare:
            def write(self, record: DecisionRecord) -> None:
                return None

            def close(self) -> None:
                return None

        stats = CompositeSink([Bare(), sink]).stats()
        # hasattr-filtered, so a third-party sink without stats() must not turn
        # ``Router.stats()`` into an AttributeError.
        assert stats["sinks"] == [{"backend": "recording", "written": 0}]

    def test_an_empty_composite_is_inert(self) -> None:
        composite = CompositeSink([])
        composite.write(make_record(0))
        composite.close()
        assert composite.stats() == {
            "backend": "composite",
            "member_errors": 0,
            "sinks": [],
        }

    def test_a_failing_member_does_not_stop_the_others(self, sink: Any) -> None:
        def explode(record: DecisionRecord) -> None:
            raise RuntimeError("warehouse down")

        CompositeSink([CallbackSink(explode), sink]).write(make_record(0))
        # The realistic failure: a member that guards itself (as CallbackSink does)
        # must not take the durable file down with it.
        assert sink.count == 1

    def test_a_member_that_raises_should_not_starve_later_members(self, sink: Any) -> None:
        """The documented fan-out promise, for a member that does not guard itself.

        Minimal reproduction::

            class Boom:
                def write(self, record): raise RuntimeError("boom")
                def close(self): pass
            CompositeSink([Boom(), recording]).write(record)  # raises; recording stays empty

        Suggested fix: wrap the loop body in ``except Exception`` (mirroring
        ``CallbackSink.write``), count the failures, and expose them in ``stats()``
        as ``{"backend": "composite", "errors": n, "sinks": [...]}``. ``BaseException``
        should still propagate, so Ctrl-C is not swallowed.
        """

        class Boom:
            def write(self, record: DecisionRecord) -> None:
                raise RuntimeError("boom")

            def close(self) -> None:
                return None

        CompositeSink([Boom(), sink]).write(make_record(0))
        assert sink.count == 1


# --------------------------------------------------------------------------- #
# JsonlSink: writing
# --------------------------------------------------------------------------- #
class TestJsonlSinkWriting:
    """One write, one line, whole record. This is the dataset contract."""

    @pytest.fixture
    def path(self, tmp_path: Path) -> Path:
        return tmp_path / "decisions.jsonl"

    def test_each_write_appends_exactly_one_line(self, path: Path) -> None:
        sink = JsonlSink(path)
        for i in range(3):
            sink.write(make_record(i))
        sink.close()
        assert len(raw_lines(path)) == 3

    def test_every_line_is_newline_terminated(self, path: Path) -> None:
        sink = JsonlSink(path)
        sink.write(make_record(0))
        sink.close()
        content = path.read_text(encoding="utf-8")
        # No trailing newline would glue the next record onto this one after a
        # restart, corrupting both.
        assert content.endswith("\n")
        assert content.count("\n") == 1

    def test_the_line_parses_back_into_an_equal_record(self, path: Path) -> None:
        record = make_record(0)
        sink = JsonlSink(path)
        sink.write(record)
        sink.close()
        assert DecisionRecord.from_json(raw_lines(path)[0]) == record

    def test_schema_version_and_kind_are_on_every_line(self, path: Path) -> None:
        sink = JsonlSink(path)
        for i in range(3):
            sink.write(make_record(i))
        sink.close()
        for data in raw_records(path):
            assert data["schema_version"] == SCHEMA_VERSION
            assert data["kind"] == RECORD_KIND

    @pytest.mark.parametrize(
        ("field", "ladder", "expected"),
        [
            pytest.param("complexity", COMPLEXITY_LEVELS, COMPLEXITY_PROBS, id="complexity"),
            pytest.param("sensitivity", SENSITIVITY_LEVELS, SENSITIVITY_PROBS, id="sensitivity"),
            pytest.param("domain", DOMAINS, DOMAIN_PROBS, id="domain"),
        ],
    )
    def test_the_full_distribution_is_on_disk(
        self, path: Path, field: str, ladder: Sequence[str], expected: Mapping[str, float]
    ) -> None:
        """Every ladder option, with its own probability -- not just the winner.

        This is the assertion that makes distillation possible. Training on the
        argmax throws away exactly the calibration that made the cloud bootstrap
        worth paying for, so a regression here is a silent one: the log still looks
        fine and the distilled model is simply worse.
        """
        sink = JsonlSink(path)
        sink.write(make_record(0))
        sink.close()
        stored = answers_on_disk(path)[field]["probabilities"]
        # Set equality, not sequence equality: to_json() writes with sort_keys=True
        # so the on-disk key order is alphabetical rather than ladder order. What a
        # trainer needs is "every option is present with its probability", and that
        # is order-independent. The sortedness itself is pinned separately below.
        assert set(stored) == set(ladder)
        for option in ladder:
            assert stored[option] == pytest.approx(expected[option]), option

    def test_lines_are_key_sorted_so_diffs_and_hashes_are_stable(self, path: Path) -> None:
        record = make_record(0)
        sink = JsonlSink(path)
        sink.write(record)
        sink.close()
        line = raw_lines(path)[0]
        # Documented in DecisionRecord.to_json: sort_keys keeps diffs and content
        # hashes stable. Two runs over the same decisions must produce byte-equal
        # logs, or a dataset checksum stops meaning anything.
        parsed = json.loads(line)
        assert line == json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        keys = list(parsed["decision"]["answers"]["complexity"]["probabilities"])
        assert keys == sorted(keys)
        assert keys != list(COMPLEXITY_LEVELS)

    def test_non_argmax_mass_is_actually_stored(self, path: Path) -> None:
        sink = JsonlSink(path)
        sink.write(make_record(0))
        sink.close()
        answers = answers_on_disk(path)
        for field in ("complexity", "sensitivity", "domain"):
            stored = answers[field]["probabilities"]
            winner = answers[field]["choice"]
            losers = [v for k, v in stored.items() if k != winner]
            # A one-hot distribution would also satisfy "all keys present".
            assert max(losers) > 0.01, field

    def test_confidence_and_its_provenance_survive(self, path: Path) -> None:
        sink = JsonlSink(path)
        sink.write(make_record(0))
        sink.close()
        stored = answers_on_disk(path)["complexity"]
        assert stored["confidence"] == pytest.approx(0.55 - 0.29)
        # confidence_reported distinguishes a calibrated backend number from one we
        # derived; losing it would make distill.evaluate compare a number to itself.
        assert stored["confidence_reported"] is False

    def test_the_noul_answer_is_stored_as_a_probability(self, path: Path) -> None:
        sink = JsonlSink(path)
        sink.write(make_record(0))
        sink.close()
        assert answers_on_disk(path)["pii"] == {"noul": PII_VALUE}

    def test_no_prompt_text_is_stored_by_default(self, path: Path) -> None:
        sink = JsonlSink(path)
        sink.write(make_record(0))
        sink.close()
        line = raw_lines(path)[0]
        assert PROMPT_TEXT not in line
        assert raw_records(path)[0]["excerpt"] is None
        # The hash is what makes the record linkable without being readable.
        assert hash_text(PROMPT_TEXT) in line

    def test_the_excerpt_is_stored_when_explicitly_set(self, path: Path) -> None:
        sink = JsonlSink(path)
        sink.write(make_record(0, excerpt="[payment_card] redacted form"))
        sink.close()
        assert raw_records(path)[0]["excerpt"] == "[payment_card] redacted form"

    def test_a_gate_finding_is_provable_without_being_leaky(self, path: Path, samples: dict[str, Any]) -> None:
        card = samples["card_valid"]
        verdict = HardGate().scan(f"Charge {card} to the customer account")
        assert verdict.findings, "the sample card number should fire the gate"
        sink = JsonlSink(path)
        sink.write(make_record(0, text=f"Charge {card} to the customer account", gate=verdict))
        sink.close()
        line = raw_lines(path)[0]
        # Provable: the finding, its floor and its span hash are all retained.
        stored_findings = raw_records(path)[0]["decision"]["gate"]["findings"]
        assert stored_findings[0]["span_hash"] in line
        assert stored_findings[0]["force_local"] is True
        # Not leaky: the thing that fired the detector is nowhere in the file.
        assert card not in line
        assert card.replace(" ", "") not in line

    def test_a_second_sink_appends_instead_of_truncating(self, path: Path) -> None:
        JsonlSink(path).write(make_record(0))
        before = path.read_bytes()
        JsonlSink(path).write(make_record(1))
        after = path.read_bytes()
        # Byte-level append-only: the first record must survive verbatim, which is
        # what lets a proxy restart without losing the rows it already wrote.
        assert after.startswith(before)
        assert len(raw_lines(path)) == 2

    def test_parent_directories_are_created_by_default(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "deeper" / "decisions.jsonl"
        assert not path.parent.exists()
        JsonlSink(path)
        assert path.parent.is_dir()

    def test_create_dirs_false_does_not_create_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "decisions.jsonl"
        JsonlSink(path, create_dirs=False)
        assert not path.parent.exists()

    def test_flush_every_leaves_the_file_readable_before_close(self, path: Path) -> None:
        sink = JsonlSink(path, flush_every=50)
        for i in range(5):
            sink.write(make_record(i))
        # Verified reality: the handle is opened with buffering=1, so each line
        # reaches the OS immediately and flush_every only controls the extra
        # fsync. Asserting "5 records readable while the handle is still open" is
        # therefore the crash-safety property that actually matters.
        assert len(list(iter_records(path))) == 5
        sink.close()

    def test_concurrent_writers_do_not_interleave_records(self, path: Path) -> None:
        sink = JsonlSink(path)
        threads_n, per_thread = 8, 20

        def worker(k: int) -> None:
            for i in range(per_thread):
                sink.write(make_record(k * 100 + i))

        threads = [threading.Thread(target=worker, args=(k,)) for k in range(threads_n)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        sink.close()
        records = list(iter_records(path))
        # The docstring claims a multi-worker proxy can share one path. Interleaved
        # writes would show up here as unparseable lines or as missing records.
        assert len(records) == threads_n * per_thread
        assert len({r.request_id for r in records}) == threads_n * per_thread
        assert sink.stats()["written"] == threads_n * per_thread

    def test_write_after_close_reopens_and_appends(self, path: Path) -> None:
        sink = JsonlSink(path)
        sink.write(make_record(0))
        sink.close()
        sink.write(make_record(1))
        sink.close()
        assert [r.request_id for r in iter_records(path)] == ["req-0", "req-1"]


# --------------------------------------------------------------------------- #
# JsonlSink: rotation
# --------------------------------------------------------------------------- #
class TestJsonlSinkRotation:
    """Rotation renames. It never rewrites and never destroys."""

    @pytest.fixture
    def path(self, tmp_path: Path) -> Path:
        return tmp_path / "decisions.jsonl"

    def _write_n(self, path: Path, n: int, *, max_bytes: int = TINY_MAX_BYTES) -> JsonlSink:
        sink = JsonlSink(path, max_bytes=max_bytes)
        for i in range(n):
            sink.write(make_record(i))
        return sink

    def test_a_small_max_bytes_rotates_on_every_write(self, path: Path) -> None:
        sink = self._write_n(path, 6)
        sink.close()
        # Verified: 6 writes at max_bytes=200 with ~1.7 KB records rotate six times,
        # because the size check runs after the handle is opened -- so even the very
        # first write rotates, leaving one empty sibling behind.
        assert sink.stats()["rotations"] == 6
        assert len(list(tmp_files(path))) == 7

    def test_rotated_siblings_carry_a_utc_timestamp(self, path: Path) -> None:
        self._write_n(path, 3).close()
        siblings = [p.name for p in tmp_files(path) if p != path]
        assert siblings, "expected at least one rotated sibling"
        for name in siblings:
            assert ROTATED_NAME.match(name), name
        # Same-second collisions get a numeric suffix rather than overwriting.
        assert any(re.search(r"\.\d+\.jsonl$", name) for name in siblings)

    def test_the_current_file_keeps_the_newest_record(self, path: Path) -> None:
        self._write_n(path, 6).close()
        records = list(iter_records(path))
        assert len(records) == 1
        assert records[0].request_id == "req-5"

    def test_rotation_never_destroys_data(self, path: Path, tmp_path: Path) -> None:
        self._write_n(path, 6).close()
        recovered = list(iter_all_records(tmp_path))
        assert len(recovered) == 6
        assert {r.request_id for r in recovered} == {f"req-{i}" for i in range(6)}

    def test_rotation_renames_the_bytes_untouched(self, path: Path, tmp_path: Path) -> None:
        sink = JsonlSink(path, max_bytes=DEFAULT_MAX_BYTES)
        sink.write(make_record(0))
        before = path.read_bytes()
        sink.close()

        rotator = JsonlSink(path, max_bytes=10)
        rotator.write(make_record(1))
        rotator.close()
        sibling = next(p for p in tmp_files(path) if p != path)
        # Append-only at the byte level: the rotated file is the old file, renamed.
        # A rewrite would be invisible to a record-count assertion.
        assert sibling.read_bytes() == before
        assert rotator.stats()["rotations"] == 1

    def test_max_bytes_zero_disables_rotation(self, path: Path, tmp_path: Path) -> None:
        sink = self._write_n(path, 30, max_bytes=0)
        sink.close()
        assert sink.stats()["rotations"] == 0
        assert len(list(tmp_files(path))) == 1
        assert len(list(iter_records(path))) == 30

    def test_a_negative_max_bytes_is_clamped_to_zero(self, tmp_path: Path) -> None:
        # A policy typo like max_bytes: -1 must mean "never rotate", not "rotate on
        # every write and shred the log into 1-record files".
        sink = JsonlSink(tmp_path / "decisions.jsonl", max_bytes=-1)
        assert sink.max_bytes == 0

    def test_a_zero_flush_every_is_clamped_to_one(self, tmp_path: Path) -> None:
        # ``written % 0`` would be a ZeroDivisionError on the first record.
        assert JsonlSink(tmp_path / "decisions.jsonl", flush_every=0).flush_every == 1

    def test_size_bytes_in_stats_is_the_current_file_only(self, path: Path) -> None:
        sink = self._write_n(path, 6)
        sink.close()
        # Not the sum of the rotated siblings: stats() describes the live file, so
        # an operator watching it sees the size that is about to trigger a rotation.
        assert sink.stats()["size_bytes"] == path.stat().st_size
        assert sink.stats()["size_bytes"] == len(make_record(5).to_json().encode("utf-8")) + 1


# --------------------------------------------------------------------------- #
# JsonlSink: failures
# --------------------------------------------------------------------------- #
class TestJsonlSinkFailures:
    """A full disk must not take down the request path."""

    def test_a_missing_parent_directory_is_counted_not_raised(self, tmp_path: Path) -> None:
        sink = JsonlSink(tmp_path / "absent" / "decisions.jsonl", create_dirs=False)
        sink.write(make_record(0))
        stats = sink.stats()
        assert stats["write_errors"] == 1
        assert stats["written"] == 0

    def test_a_directory_as_the_log_path_is_counted_not_raised(self, tmp_path: Path) -> None:
        target = tmp_path / "decisions.jsonl"
        target.mkdir()
        sink = JsonlSink(target)
        sink.write(make_record(0))
        stats = sink.stats()
        assert stats["write_errors"] == 1
        assert stats["written"] == 0
        # Documented oddity, asserted so it cannot change unnoticed: size_bytes stats
        # the path, and a directory stats non-zero, so it is not "bytes of log".
        assert stats["size_bytes"] > 0

    def test_a_regular_file_as_the_parent_is_counted_not_raised(self, tmp_path: Path) -> None:
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("x", encoding="utf-8")
        sink = JsonlSink(blocker / "decisions.jsonl", create_dirs=False)
        sink.write(make_record(0))
        assert sink.stats()["write_errors"] == 1

    def test_creating_dirs_under_a_regular_file_raises_at_construction(self, tmp_path: Path) -> None:
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("x", encoding="utf-8")
        # Verified: with the default create_dirs=True the failure surfaces in the
        # constructor, not in write(). That is the better place for it -- a broken
        # log path is a configuration error and should fail at policy load, not
        # three requests in.
        with pytest.raises(OSError):
            JsonlSink(blocker / "decisions.jsonl")

    def test_a_transient_open_failure_is_swallowed_and_counted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "decisions.jsonl"
        sink = JsonlSink(path)
        original = sink._open
        attempts = {"n": 0}

        def flaky_open() -> Any:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise OSError(28, "No space left on device")
            return original()

        monkeypatch.setattr(sink, "_open", flaky_open)
        sink.write(make_record(0))  # ENOSPC: must not escape
        assert sink.stats()["write_errors"] == 1
        assert sink.stats()["written"] == 0

    def test_writing_recovers_after_a_transient_open_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "decisions.jsonl"
        sink = JsonlSink(path)
        original = sink._open
        attempts = {"n": 0}

        def flaky_open() -> Any:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise OSError(28, "No space left on device")
            return original()

        monkeypatch.setattr(sink, "_open", flaky_open)
        sink.write(make_record(0))
        sink.write(make_record(1))
        sink.close()
        # The hole is exactly one record: the log stays usable after the disk does.
        assert [r.request_id for r in iter_records(path)] == ["req-1"]
        assert sink.stats() == {
            "backend": "jsonl",
            "path": str(path),
            "written": 1,
            "rotations": 0,
            "write_errors": 1,
            "size_bytes": path.stat().st_size,
        }

    def test_a_close_failure_is_counted_not_raised(self, tmp_path: Path) -> None:
        sink = JsonlSink(tmp_path / "decisions.jsonl")
        sink._handle = _BrokenHandle()
        sink.close()  # must not raise
        assert sink.stats()["write_errors"] == 1
        assert sink._handle is None

    def test_stats_after_an_error_still_reports_the_path(self, tmp_path: Path) -> None:
        path = tmp_path / "absent" / "decisions.jsonl"
        sink = JsonlSink(path, create_dirs=False)
        sink.write(make_record(0))
        # The path stays in stats even though no file exists, so an operator can see
        # which configured log is broken rather than just that one is.
        assert sink.stats()["path"] == str(path)
        assert sink.stats()["size_bytes"] == 0


# --------------------------------------------------------------------------- #
# JsonlSink: close
# --------------------------------------------------------------------------- #
class TestJsonlSinkClose:
    """close() flushes, and is safe to call as often as teardown likes."""

    @pytest.fixture
    def path(self, tmp_path: Path) -> Path:
        return tmp_path / "decisions.jsonl"

    def test_close_flushes_the_open_handle(self, path: Path) -> None:
        sink = JsonlSink(path, flush_every=100)
        sink.write(make_record(0))
        spy = _HandleSpy(sink._handle)
        sink._handle = spy
        sink.close()
        assert spy.flushes == 1
        assert spy.closes == 1
        assert sink._handle is None

    def test_close_is_idempotent(self, path: Path) -> None:
        sink = JsonlSink(path)
        sink.write(make_record(0))
        sink.close()
        sink.close()
        sink.close()
        # A double close that raised would break Router.aclose(), which is itself
        # documented as never raising during teardown.
        assert sink.stats()["write_errors"] == 0

    def test_every_record_is_on_disk_after_close(self, path: Path) -> None:
        sink = JsonlSink(path, flush_every=100)
        for i in range(4):
            sink.write(make_record(i))
        sink.close()
        assert len(list(iter_records(path))) == 4

    def test_close_on_a_never_written_sink_is_a_noop(self, path: Path) -> None:
        sink = JsonlSink(path)
        sink.close()
        assert sink.stats()["written"] == 0
        assert not path.exists()


# --------------------------------------------------------------------------- #
# build_sink
# --------------------------------------------------------------------------- #
class TestBuildSink:
    """The ``logging`` policy section, turned into a sink."""

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param(None, id="no-logging-section"),
            pytest.param({}, id="empty-section"),
            pytest.param({"enabled": False, "path": "/tmp/never.jsonl"}, id="explicitly-disabled"),
            pytest.param({"enabled": False}, id="disabled-without-path"),
            pytest.param({"enabled": True}, id="enabled-but-no-path"),
            pytest.param({"path": ""}, id="empty-path"),
            pytest.param({"path": None}, id="null-path"),
        ],
    )
    def test_configs_without_a_usable_path_build_a_null_sink(self, config: Any) -> None:
        # A disabled or pathless log must build a NullSink rather than raise: the
        # router constructs its sink at policy-load time, so a logging typo would
        # otherwise take down routing entirely. Note "explicitly-disabled" also
        # carries a path -- enabled: false has to win over a configured path, or
        # turning logging off would keep writing.
        assert isinstance(build_sink(config), NullSink)

    def test_a_path_builds_a_jsonl_sink(self, tmp_path: Path) -> None:
        path = tmp_path / "d.jsonl"
        sink = build_sink({"path": str(path)})
        assert isinstance(sink, JsonlSink)
        assert sink.path == path

    def test_the_built_sink_actually_writes_there(self, tmp_path: Path) -> None:
        path = tmp_path / "d.jsonl"
        sink = build_sink({"path": str(path)})
        sink.write(make_record(0))
        sink.close()
        assert len(list(iter_records(path))) == 1

    def test_max_bytes_and_flush_every_come_from_config(self, tmp_path: Path) -> None:
        sink = build_sink({"path": str(tmp_path / "d.jsonl"), "max_bytes": 4096, "flush_every": 7})
        assert isinstance(sink, JsonlSink)
        assert sink.max_bytes == 4096
        assert sink.flush_every == 7

    def test_the_defaults_match_the_documented_ones(self, tmp_path: Path) -> None:
        sink = build_sink({"path": str(tmp_path / "d.jsonl")})
        assert isinstance(sink, JsonlSink)
        assert sink.max_bytes == DEFAULT_MAX_BYTES
        assert sink.flush_every == 1

    def test_a_path_object_is_accepted(self, tmp_path: Path) -> None:
        path = tmp_path / "d.jsonl"
        sink = build_sink({"enabled": True, "path": path})
        assert isinstance(sink, JsonlSink)
        assert sink.path == path

    def test_config_keys_are_coerced_from_strings(self, tmp_path: Path) -> None:
        # YAML gives us ints, but a config coming from an env var or a ConfigMap
        # can be a string; int() coercion is what keeps that from exploding.
        sink = build_sink({"path": str(tmp_path / "d.jsonl"), "max_bytes": "4096", "flush_every": "3"})
        assert isinstance(sink, JsonlSink)
        assert sink.max_bytes == 4096
        assert sink.flush_every == 3


# --------------------------------------------------------------------------- #
# iter_records
# --------------------------------------------------------------------------- #
class TestIterRecords:
    """Reading the log back -- including after a crash."""

    @pytest.fixture
    def path(self, tmp_path: Path) -> Path:
        return tmp_path / "decisions.jsonl"

    def test_records_come_back_in_file_order(self, path: Path) -> None:
        sink = JsonlSink(path)
        for i in range(4):
            sink.write(make_record(i))
        sink.close()
        assert [r.request_id for r in iter_records(path)] == ["req-0", "req-1", "req-2", "req-3"]

    @pytest.mark.parametrize("blank", ["", "   ", "\t", " \n "])
    def test_blank_lines_are_skipped(self, path: Path, blank: str) -> None:
        record = make_record(0)
        path.write_text(f"{blank}\n{record.to_json()}\n{blank}\n", encoding="utf-8")
        assert [r.request_id for r in iter_records(path)] == ["req-0"]

    @pytest.mark.parametrize("junk", ["{not json at all", "{'single': 'quotes'}", "\x00\x01binary", "}"])
    def test_invalid_json_is_skipped(self, path: Path, junk: str) -> None:
        record = make_record(0)
        path.write_text(f"{junk}\n{record.to_json()}\n", encoding="utf-8")
        assert [r.request_id for r in iter_records(path)] == ["req-0"]

    def test_a_truncated_trailing_line_is_skipped(self, path: Path) -> None:
        """The crash case: a record cut off mid-write by SIGKILL or a full disk."""
        sink = JsonlSink(path)
        for i in range(3):
            sink.write(make_record(i))
        sink.close()
        content = path.read_text(encoding="utf-8")
        path.write_text(content + make_record(9).to_json()[:120], encoding="utf-8")
        # Two good records and one torn tail must yield the two good ones. A reader
        # that raised here would make the log unusable exactly when it matters.
        assert [r.request_id for r in iter_records(path)] == ["req-0", "req-1", "req-2"]

    def test_a_foreign_kind_is_skipped(self, path: Path) -> None:
        foreign = json.dumps({"kind": "some.other.event", "payload": 1}, sort_keys=True)
        path.write_text(f"{foreign}\n{make_record(0).to_json()}\n", encoding="utf-8")
        assert [r.request_id for r in iter_records(path)] == ["req-0"]

    def test_a_line_without_a_kind_is_skipped(self, path: Path) -> None:
        kindless = json.dumps({"request_id": "nope", "schema_version": SCHEMA_VERSION})
        path.write_text(kindless + "\n" + make_record(0).to_json() + "\n", encoding="utf-8")
        assert [r.request_id for r in iter_records(path)] == ["req-0"]

    @pytest.mark.parametrize("line", ["[1, 2, 3]", '"a bare string"', "42", "true", "null"])
    def test_non_object_json_is_skipped(self, path: Path, line: str) -> None:
        path.write_text(f"{line}\n{make_record(0).to_json()}\n", encoding="utf-8")
        assert [r.request_id for r in iter_records(path)] == ["req-0"]

    def test_a_nonexistent_path_yields_nothing(self, tmp_path: Path) -> None:
        # Not an exception: reading a log that has not been written yet is the
        # normal state of a fresh deployment, and distill runs over it.
        assert list(iter_records(tmp_path / "missing.jsonl")) == []

    def test_our_own_kind_that_cannot_be_parsed_is_loud(self, path: Path) -> None:
        broken = json.dumps({"kind": RECORD_KIND, "request_id": "x"})
        path.write_text(broken + "\n", encoding="utf-8")
        # Verified and deliberate: foreign lines are skipped silently, but a line
        # claiming to be one of ours and missing its decision is real corruption.
        # Skipping it would drop training rows without a sound.
        with pytest.raises(KeyError):
            list(iter_records(path))

    def test_a_round_trip_preserves_the_full_distributions(self, path: Path) -> None:
        record = make_record(0)
        sink = JsonlSink(path)
        sink.write(record)
        sink.close()
        back = next(iter(iter_records(path)))
        assert back.decision.answers.complexity.probabilities == pytest.approx(dict(COMPLEXITY_PROBS))
        assert back.decision.answers.sensitivity.probabilities == pytest.approx(dict(SENSITIVITY_PROBS))
        assert back.decision.answers.domain.probabilities == pytest.approx(dict(DOMAIN_PROBS))
        assert back.decision.answers.pii.value == PII_VALUE
        assert back == record

    def test_a_round_trip_preserves_gate_findings_and_escalation(self, path: Path) -> None:
        verdict = HardGate().scan("Charge 4111 1111 1111 1111 now")
        record = make_record(0, text="Charge 4111 1111 1111 1111 now", gate=verdict)
        sink = JsonlSink(path)
        sink.write(record)
        sink.close()
        back = next(iter(iter_records(path)))
        assert back.decision.gate.fired is True
        assert back.decision.gate.blocks_backend is True
        assert back.decision.gate.findings == verdict.findings
        assert back.decision.gate.advisory_topics == verdict.advisory_topics
        # The span hash is what makes the finding provable later; losing it would
        # leave a floor with no evidence attached.
        assert all(f.span_hash for f in back.decision.gate.findings)
        assert back.decision.escalated == ("sensitivity",)

    def test_features_survive_the_round_trip(self, path: Path) -> None:
        record = make_record(0)
        sink = JsonlSink(path)
        sink.write(record)
        sink.close()
        back = next(iter(iter_records(path)))
        # Features are how an operator trains without ever storing prompt text, so
        # they are part of the dataset contract rather than incidental debug data.
        assert back.features == record.features
        assert back.features.char_len == len(PROMPT_TEXT)

    def test_metadata_and_questions_sent_survive(self, path: Path) -> None:
        record = make_record(0)
        sink = JsonlSink(path)
        sink.write(record)
        sink.close()
        back = next(iter(iter_records(path)))
        assert back.metadata == {"tenant": "acme"}
        assert back.questions_sent == {"complexity": {"type": "choice"}}


# --------------------------------------------------------------------------- #
# iter_all_records
# --------------------------------------------------------------------------- #
class TestIterAllRecords:
    """Reading a whole directory of rotated logs, oldest first."""

    def _write(self, directory: Path, name: str, index: int, mtime: float | None = None) -> Path:
        path = directory / name
        path.write_text(make_record(index).to_json() + "\n", encoding="utf-8")
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        return path

    def test_a_nonexistent_directory_yields_nothing(self, tmp_path: Path) -> None:
        assert list(iter_all_records(tmp_path / "nope")) == []

    def test_an_empty_directory_yields_nothing(self, tmp_path: Path) -> None:
        logs = tmp_path / "logs"
        logs.mkdir()
        assert list(iter_all_records(logs)) == []

    def test_files_are_read_oldest_first_by_mtime(self, tmp_path: Path) -> None:
        logs = tmp_path / "logs"
        logs.mkdir()
        now = time.time()
        # Written newest-first and named out of order, so only mtime can produce the
        # expected sequence. Distillation reads history in order; a name sort here
        # would silently train on a shuffled timeline.
        self._write(logs, "decisions.3.jsonl", 3, mtime=now - 300)
        self._write(logs, "decisions.1.jsonl", 1, mtime=now - 100)
        self._write(logs, "decisions.2.jsonl", 2, mtime=now - 200)
        assert [r.request_id for r in iter_all_records(logs)] == ["req-3", "req-2", "req-1"]

    def test_the_default_pattern_ignores_foreign_files(self, tmp_path: Path) -> None:
        logs = tmp_path / "logs"
        logs.mkdir()
        self._write(logs, "decisions.1.jsonl", 1)
        self._write(logs, "audit.jsonl", 99)
        self._write(logs, "decisions.jsonl.gz", 98)
        assert [r.request_id for r in iter_all_records(logs)] == ["req-1"]

    def test_a_custom_pattern_is_honoured(self, tmp_path: Path) -> None:
        logs = tmp_path / "logs"
        logs.mkdir()
        self._write(logs, "decisions.jsonl", 1)
        self._write(logs, "shadow.jsonl", 2)
        assert [r.request_id for r in iter_all_records(logs, pattern="shadow*.jsonl")] == ["req-2"]

    def test_a_rotated_log_directory_reads_back_complete(self, tmp_path: Path) -> None:
        """End to end: rotate on every write, then read the whole history back."""
        logs = tmp_path / "logs"
        sink = JsonlSink(logs / "decisions.jsonl", max_bytes=TINY_MAX_BYTES)
        for i in range(6):
            sink.write(make_record(i))
        sink.close()
        recovered = list(iter_all_records(logs))
        assert len(recovered) == 6
        assert {r.request_id for r in recovered} == {f"req-{i}" for i in range(6)}
        for record in recovered:
            assert record.schema_version == SCHEMA_VERSION
            assert set(record.decision.answers.complexity.probabilities) == set(COMPLEXITY_LEVELS)

    def test_records_within_a_file_keep_their_order(self, tmp_path: Path) -> None:
        logs = tmp_path / "logs"
        logs.mkdir()
        path = logs / "decisions.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for i in range(3):
                handle.write(make_record(i).to_json() + "\n")
        assert [r.request_id for r in iter_all_records(logs)] == ["req-0", "req-1", "req-2"]
