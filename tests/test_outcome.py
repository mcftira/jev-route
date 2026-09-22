"""Outcome verification: labels appended as their own record kind, never
duplicating decision rows; blocked content never verified; failures log null."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from jev_route.backends.mock import MockBackend
from jev_route.outcome import OUTCOME_RECORD_KIND, OutcomeVerifier


class RecSink:
    def __init__(self) -> None:
        self.records: list[Any] = []

    def write(self, record: Any) -> None:
        self.records.append(record)


async def test_non_empty_response_verifies_completed() -> None:
    sink = RecSink()
    v = OutcomeVerifier(MockBackend(), sink)
    rec = await v.maybe_verify(
        request_id="r1", request_summary="say hi", response_excerpt="Hello there!",
        model_served="cheap-model", gate_blocked=False,
    )
    assert rec is not None and rec.completed_p == pytest.approx(0.9)
    assert rec.kind == OUTCOME_RECORD_KIND
    assert sink.records[0].request_id == "r1"


async def test_gate_blocked_is_never_verified() -> None:
    sink = RecSink()
    v = OutcomeVerifier(MockBackend(), sink)
    rec = await v.maybe_verify(
        request_id="r2", request_summary="ssn 000-12-3456", response_excerpt="...",
        model_served="m", gate_blocked=True,
    )
    assert rec is None
    assert sink.records == []


async def test_empty_response_scores_low() -> None:
    v = OutcomeVerifier(MockBackend(), RecSink())
    rec = await v.maybe_verify(
        request_id="r3", request_summary="q", response_excerpt="", model_served="m", gate_blocked=False,
    )
    assert rec is not None and rec.completed_p == pytest.approx(0.1)


async def test_verification_failure_logs_null_never_raises() -> None:
    class BrokenBackend:
        name = "broken"

        def noul(self, state, instructions):
            raise RuntimeError("backend down")

    sink = RecSink()
    v = OutcomeVerifier(BrokenBackend(), sink)
    rec = await v.maybe_verify(
        request_id="r4", request_summary="q", response_excerpt="ok", model_served="m", gate_blocked=False,
    )
    assert rec is not None and rec.completed_p is None
    assert sink.records[0].completed_p is None


async def test_backend_without_noul_is_skipped() -> None:
    class Bare:
        name = "bare"

    v = OutcomeVerifier(Bare(), RecSink())
    rec = await v.maybe_verify(
        request_id="r5", request_summary="q", response_excerpt="ok", model_served="m", gate_blocked=False,
    )
    assert rec is None


async def test_sample_rate_zero_never_verifies() -> None:
    v = OutcomeVerifier(MockBackend(), RecSink(), sample_rate=0.0)
    rec = await v.maybe_verify(
        request_id="r6", request_summary="q", response_excerpt="ok", model_served="m", gate_blocked=False,
    )
    assert rec is None


def test_record_is_json_roundtrip(tmp_path: Path) -> None:
    from jev_route.outcome import OutcomeRecord

    rec = OutcomeRecord(request_id="r", completed_p=0.9, model_served="m", backend="mock",
                        checked_at=1.0, sampled=True)
    line = rec.to_json()
    back = json.loads(line)
    assert back["kind"] == OUTCOME_RECORD_KIND
    assert back["completed_p"] == pytest.approx(0.9)
