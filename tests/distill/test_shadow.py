"""Tests for :mod:`jev_route.backends.shadow`.

No network, no API key, no torch, no scikit-learn: primaries and shadows are
either :class:`~jev_route.backends.mock.MockBackend` or the hand-written
:class:`StubBackend` below. Timing assertions use real (short) sleeps with
generous bounds -- the contract under test is asyncio semantics ("the shadow
never delays the primary by more than its bound"), not clock arithmetic, so
mocking the loop would measure the wrong thing.

Every test settles its shadow tasks (``join()`` / ``aclose()``) before ending,
so a failed assertion cannot leave a pending task behind to emit
"Task was destroyed but it is pending" noise into the rest of the suite.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from jev_route.backends import build_backend
from jev_route.backends.base import BackendResult, DecisionBackend, DecisionRequest
from jev_route.backends.mock import MockBackend
from jev_route.backends.shadow import ShadowBackend
from jev_route.schema import (
    COMPLEXITY_LEVELS,
    DOMAINS,
    SENSITIVITY_LEVELS,
    ChoiceAnswer,
    DecisionAnswers,
    NoulAnswer,
    RequestFeatures,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _choice(choice: str, ladder: tuple[str, ...]) -> ChoiceAnswer:
    return ChoiceAnswer(
        choice=choice,
        probabilities={level: (1.0 if level == choice else 0.0) for level in ladder},
        confidence=1.0,
        confidence_reported=False,
    )


def _answers(
    *,
    complexity: str = "standard",
    sensitivity: str = "internal",
    pii: float = 0.1,
    domain: str = "code",
) -> DecisionAnswers:
    return DecisionAnswers(
        complexity=_choice(complexity, COMPLEXITY_LEVELS),
        sensitivity=_choice(sensitivity, SENSITIVITY_LEVELS),
        pii=NoulAnswer(value=pii),
        domain=_choice(domain, DOMAINS),
    )


def _result(
    answers: DecisionAnswers | None = None,
    *,
    model_version: str = "stub-1",
    degraded: bool = False,
    degrade_reason: str | None = None,
) -> BackendResult:
    return BackendResult(
        answers=answers if answers is not None else _answers(),
        model_version=model_version,
        degraded=degraded,
        degrade_reason=degrade_reason,
    )


class StubBackend:
    """A scripted ``DecisionBackend``: fixed result, optional delay, optional raise.

    Delays are real ``asyncio.sleep`` calls so timeouts and cancellation are
    exercised the way production would exercise them.
    """

    def __init__(
        self,
        *,
        name: str = "stub",
        model_version: str = "stub-1",
        result: BackendResult | None = None,
        delay: float = 0.0,
        error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.name = name
        self.model_version = model_version
        self.result = result if result is not None else _result(model_version=model_version)
        self.delay = delay
        self.error = error
        self.close_error = close_error
        self.calls = 0
        self.closed = 0

    async def decide(self, request: DecisionRequest) -> BackendResult:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.result

    async def aclose(self) -> None:
        self.closed += 1
        if self.close_error is not None:
            raise self.close_error


def _request(
    text: str = "Summarize the incident review and email the draft to the team.",
    request_id: str = "req-1",
) -> DecisionRequest:
    return DecisionRequest(
        redacted_excerpt=text,
        features=RequestFeatures(char_len=len(text), word_count=len(text.split())),
        request_id=request_id,
    )


_BASE_REPORT_KEYS = {
    "request_id",
    "primary_backend",
    "shadow_backend",
    "shadow_model_version",
    "agrees",
    "disagreements",
    "shadow_latency_ms",
    "shadow_degraded",
    "sampled",
    "mode",
}


# --------------------------------------------------------------------------- #
# 1. the primary's answer is the product
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_returns_the_primary_result_object_when_shadow_agrees() -> None:
    primary_result = _result(model_version="primary-9")
    primary = StubBackend(name="primary", model_version="primary-9", result=primary_result)
    shadow = StubBackend(name="shadow", model_version="shadow-3", result=_result(model_version="shadow-3"))
    backend = ShadowBackend(primary=primary, shadow=shadow, mode="await")

    out = await backend.decide(_request())

    assert out is primary_result  # the object itself, not a copy or a merge
    report = backend.reports()[0]
    assert report["request_id"] == "req-1"
    assert report["primary_backend"] == "primary"
    assert report["shadow_backend"] == "shadow"
    assert report["shadow_model_version"] == "shadow-3"
    assert report["agrees"] is True
    assert report["disagreements"] == {}
    assert report["shadow_degraded"] is False
    assert report["shadow_latency_ms"] >= 0.0
    assert report["sampled"] is True
    assert report["mode"] == "await"
    stats = backend.stats()
    assert stats["completed"] == 1
    assert stats["disagreements"] == 0
    assert stats["disagreement_rate"] == 0.0
    assert backend.disagreements() == ()
    await backend.aclose()


@pytest.mark.asyncio
async def test_returns_the_primary_result_object_when_shadow_disagrees() -> None:
    primary_result = _result(_answers(complexity="standard", pii=0.1))
    primary = StubBackend(name="primary", result=primary_result)
    shadow = StubBackend(name="shadow", result=_result(_answers(complexity="frontier", pii=0.9)))
    backend = ShadowBackend(primary, shadow, mode="await")

    out = await backend.decide(_request())

    assert out is primary_result  # the shadow's opinion changes nothing
    report = backend.reports()[0]
    assert report["agrees"] is False
    assert report["disagreements"]["complexity"] == {"primary": "standard", "shadow": "frontier"}
    assert report["disagreements"]["pii"] == {"primary": 0.1, "shadow": 0.9}
    assert backend.disagreements() == (report,)
    stats = backend.stats()
    assert stats["disagreements"] == 1
    assert stats["disagreement_rate"] == 1.0
    assert stats["fields"]["complexity"] == 1
    assert stats["fields"]["pii"] == 1
    await backend.aclose()


@pytest.mark.asyncio
async def test_mock_primary_with_colder_mock_shadow() -> None:
    primary = MockBackend(temperature=0.55)
    shadow = MockBackend(temperature=0.02)
    backend = ShadowBackend(primary=primary, shadow=shadow, mode="await")
    req = _request("Prove the sharding refactor is safe, then debug the race condition in these logs.")

    out = await backend.decide(req)

    assert out.model_version == "mock-1.0.0"
    assert out.answers == primary.decide_sync(req)  # exactly what the primary says
    report = backend.reports()[0]
    assert report["primary_backend"] == "mock"
    assert report["shadow_backend"] == "mock"
    await backend.aclose()


def test_model_version_mirrors_primary_live() -> None:
    primary = StubBackend(model_version="primary-A")
    backend = ShadowBackend(primary, StubBackend())
    assert backend.name == "shadow"
    assert backend.model_version == "primary-A"
    primary.model_version = "primary-B"  # a cloud backend learns this only after first response
    assert backend.model_version == "primary-B"


def test_satisfies_the_decision_backend_protocol() -> None:
    assert isinstance(ShadowBackend(StubBackend(), StubBackend()), DecisionBackend)


def test_default_mode_is_background() -> None:
    backend = ShadowBackend(StubBackend(), StubBackend())
    assert backend.mode == "background"
    assert backend.shadow_timeout_seconds == 5.0
    assert backend.sample_rate == 1.0
    assert backend.log_disagreements is True
    assert backend.max_reports == 256


def test_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="background"):
        ShadowBackend(StubBackend(), StubBackend(), mode="inline")


def test_rejects_negative_max_reports() -> None:
    with pytest.raises(ValueError, match="max_reports"):
        ShadowBackend(StubBackend(), StubBackend(), max_reports=-1)


# --------------------------------------------------------------------------- #
# 2. a broken shadow is telemetry, never an error
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["background", "await"])
async def test_shadow_exception_never_escapes(mode: str) -> None:
    primary_result = _result()
    primary = StubBackend(name="primary", result=primary_result)
    shadow = StubBackend(name="shadow", error=RuntimeError("shadow exploded"))
    backend = ShadowBackend(primary, shadow, mode=mode)  # type: ignore[arg-type]

    out = await backend.decide(_request())
    if mode == "background":
        assert await backend.join(timeout=2.0) == 1

    assert out is primary_result
    report = backend.reports()[0]
    assert report["error"] == "RuntimeError: shadow exploded"
    assert report["agrees"] is None  # no opinion, not "agrees"
    assert report["disagreements"] == {}
    assert "timeout" not in report
    assert backend.disagreements() == ()
    stats = backend.stats()
    assert stats["errors"] == 1
    assert stats["completed"] == 0
    assert stats["timeouts"] == 0
    assert stats["disagreements"] == 0
    await backend.aclose()


# --------------------------------------------------------------------------- #
# 3. + 5. a hung shadow is bounded in both modes
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["background", "await"])
async def test_hung_shadow_times_out_without_delaying_primary(mode: str) -> None:
    primary = StubBackend(name="primary")
    shadow = StubBackend(name="shadow", delay=10.0)
    backend = ShadowBackend(primary, shadow, mode=mode, shadow_timeout_seconds=0.05)  # type: ignore[arg-type]
    started = time.monotonic()
    try:
        out = await backend.decide(_request())
        elapsed = time.monotonic() - started

        assert out is primary.result
        assert elapsed < 1.0
        if mode == "background":
            assert backend.reports() == ()  # the shadow has not settled yet
            assert await backend.join(timeout=2.0) == 1
        report = backend.reports()[0]
        assert report["timeout"] is True
        assert report["agrees"] is None
        assert "error" not in report
        assert report["shadow_latency_ms"] >= 40.0  # it waited out the bound
        stats = backend.stats()
        assert stats["timeouts"] == 1
        assert stats["completed"] == 0
        assert stats["errors"] == 0
    finally:
        await backend.aclose()
    await asyncio.sleep(0)  # flush done-callbacks
    assert not backend._tasks  # nothing pending -> no "Task was destroyed" warning


@pytest.mark.asyncio
async def test_await_mode_bounds_added_latency() -> None:
    shadow = StubBackend(name="shadow", delay=10.0)
    backend = ShadowBackend(StubBackend(), shadow, mode="await", shadow_timeout_seconds=0.05)
    started = time.monotonic()

    out = await backend.decide(_request())
    elapsed = time.monotonic() - started

    assert elapsed < 1.0  # bounded by shadow_timeout_seconds, not by the 10 s hang
    assert out is not None
    assert backend.reports()[0]["timeout"] is True
    assert backend.stats()["timeouts"] == 1
    await backend.aclose()


# --------------------------------------------------------------------------- #
# 4. background mode adds zero latency
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_background_mode_does_not_delay_primary() -> None:
    primary = StubBackend(name="primary")
    shadow = StubBackend(name="shadow", delay=0.5)
    backend = ShadowBackend(primary, shadow)  # default mode: background
    started = time.monotonic()

    out = await backend.decide(_request())
    elapsed = time.monotonic() - started

    assert elapsed < 0.2  # the 0.5 s shadow did not slow the primary down
    assert out is primary.result
    assert backend.reports() == ()  # still in flight; nothing recorded yet

    assert await backend.join(timeout=2.0) == 1
    report = backend.reports()[0]
    assert report["agrees"] is True
    assert report["mode"] == "background"
    assert report["shadow_latency_ms"] >= 400.0  # the shadow really did run
    await backend.aclose()


@pytest.mark.asyncio
async def test_join_waits_for_multiple_background_tasks() -> None:
    shadow = StubBackend(name="shadow", delay=0.05)
    backend = ShadowBackend(StubBackend(), shadow)
    for i in range(3):
        await backend.decide(_request(request_id=f"r{i}"))

    assert await backend.join(timeout=2.0) == 3
    assert len(backend.reports()) == 3
    assert shadow.calls == 3
    assert await backend.join() == 0  # idempotent when idle
    await backend.aclose()


@pytest.mark.asyncio
async def test_join_returns_zero_when_idle() -> None:
    backend = ShadowBackend(StubBackend(), StubBackend())
    assert await backend.join() == 0
    assert await backend.join(timeout=0.01) == 0
    await backend.aclose()


# --------------------------------------------------------------------------- #
# 6. degraded shadows
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_degraded_shadow_is_recorded_but_not_a_disagreement() -> None:
    primary = StubBackend(name="primary", result=_result())
    degraded = _result(
        _answers(complexity="frontier", sensitivity="regulated", pii=0.99, domain="chat"),
        degraded=True,
        degrade_reason="stub-outage",
    )
    shadow = StubBackend(name="shadow", result=degraded)
    backend = ShadowBackend(primary, shadow, mode="await")

    out = await backend.decide(_request())

    assert out is primary.result
    report = backend.reports()[0]
    assert report["shadow_degraded"] is True
    assert report["agrees"] is None  # a shadow that could not answer tells you nothing
    assert report["disagreements"] == {}
    assert backend.disagreements() == ()
    stats = backend.stats()
    assert stats["degraded"] == 1
    assert stats["completed"] == 1
    assert stats["disagreements"] == 0
    assert stats["disagreement_rate"] == 0.0
    await backend.aclose()


# --------------------------------------------------------------------------- #
# 7. comparison semantics
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("shadow_pii", "expect_agrees"),
    [(0.5, True), (0.74, True), (0.75, False), (0.76, False)],
)
async def test_pii_disagreement_tolerance(shadow_pii: float, expect_agrees: bool) -> None:
    primary = StubBackend(result=_result(_answers(pii=0.5)))
    shadow = StubBackend(result=_result(_answers(pii=shadow_pii)))
    backend = ShadowBackend(primary, shadow, mode="await")

    await backend.decide(_request())
    report = backend.reports()[0]

    assert report["agrees"] is expect_agrees
    if expect_agrees:
        assert "pii" not in report["disagreements"]
    else:
        # |delta| >= 0.25 counts, and both sides are recorded for triage
        assert report["disagreements"]["pii"] == {"primary": 0.5, "shadow": shadow_pii}
    await backend.aclose()


@pytest.mark.asyncio
async def test_choice_disagreements_compare_argmax_and_record_both() -> None:
    primary = StubBackend(result=_result(_answers(sensitivity="internal", domain="code")))
    shadow = StubBackend(result=_result(_answers(sensitivity="regulated", domain="writing")))
    backend = ShadowBackend(primary, shadow, mode="await")

    await backend.decide(_request())
    report = backend.reports()[0]

    assert report["agrees"] is False
    assert report["disagreements"] == {
        "sensitivity": {"primary": "internal", "shadow": "regulated"},
        "domain": {"primary": "code", "shadow": "writing"},
    }
    stats = backend.stats()
    assert stats["fields"]["sensitivity"] == 1
    assert stats["fields"]["domain"] == 1
    assert stats["fields"]["pii"] == 0
    await backend.aclose()


# --------------------------------------------------------------------------- #
# 8. deterministic sampling
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_sample_rate_zero_never_calls_shadow() -> None:
    shadow = StubBackend(name="shadow")
    backend = ShadowBackend(StubBackend(name="primary"), shadow, mode="await", sample_rate=0.0)
    for i in range(3):
        out = await backend.decide(_request(request_id=f"req-{i}"))
        assert out is not None

    assert shadow.calls == 0
    assert backend.reports() == ()
    stats = backend.stats()
    assert stats["requests"] == 3
    assert stats["sampled"] == 0
    await backend.aclose()


@pytest.mark.asyncio
async def test_sample_rate_one_always_calls_shadow() -> None:
    shadow = StubBackend(name="shadow")
    backend = ShadowBackend(StubBackend(name="primary"), shadow, mode="await", sample_rate=1.0)
    for i in range(3):
        await backend.decide(_request(request_id=f"req-{i}"))

    assert shadow.calls == 3
    assert backend.stats()["sampled"] == 3
    assert len(backend.reports()) == 3
    await backend.aclose()


@pytest.mark.asyncio
async def test_partial_sampling_is_deterministic_per_request_id() -> None:
    async def sampled_once(request_id: str) -> int:
        backend = ShadowBackend(StubBackend(), StubBackend(), mode="await", sample_rate=0.5)
        await backend.decide(_request(request_id=request_id))
        verdict = int(backend.stats()["sampled"])
        await backend.aclose()
        return verdict

    first = await sampled_once("req-fixed")
    second = await sampled_once("req-fixed")
    assert first == second  # same key, same verdict, fresh instance
    assert first in (0, 1)


@pytest.mark.asyncio
async def test_partial_sampling_splits_traffic() -> None:
    primary, shadow = StubBackend(), StubBackend()
    backend = ShadowBackend(primary, shadow, mode="await", sample_rate=0.5)
    for i in range(64):
        await backend.decide(_request(request_id=f"req-{i}"))

    sampled = backend.stats()["sampled"]
    assert 0 < sampled < 64  # sha256 bucketing: a real split, identical in every run
    assert shadow.calls == sampled
    assert len(backend.reports()) == sampled
    assert all(report["sampled"] is True for report in backend.reports())
    await backend.aclose()


@pytest.mark.asyncio
async def test_sampling_falls_back_to_excerpt_hash_without_request_id() -> None:
    primary, shadow = StubBackend(), StubBackend()
    backend = ShadowBackend(primary, shadow, mode="await", sample_rate=0.5)
    await backend.decide(_request(text="identical excerpt", request_id=""))
    await backend.decide(_request(text="identical excerpt", request_id=""))

    sampled = backend.stats()["sampled"]
    assert sampled in (0, 2)  # same excerpt -> same verdict, both times
    assert shadow.calls == sampled
    await backend.aclose()


# --------------------------------------------------------------------------- #
# 9. on_disagreement callback
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_on_disagreement_callback_receives_report_copy() -> None:
    seen: list[dict[str, Any]] = []
    primary = StubBackend(result=_result(_answers()))
    shadow = StubBackend(result=_result(_answers(domain="chat")))
    backend = ShadowBackend(primary, shadow, mode="await", on_disagreement=seen.append)

    await backend.decide(_request())

    assert len(seen) == 1
    recorded = backend.disagreements()[0]
    assert seen[0] == recorded
    assert seen[0] is not recorded  # the callback cannot mutate the ring buffer
    assert seen[0]["disagreements"]["domain"] == {"primary": "code", "shadow": "chat"}
    await backend.aclose()


@pytest.mark.asyncio
async def test_callback_fires_in_background_mode_after_join() -> None:
    seen: list[dict[str, Any]] = []
    shadow = StubBackend(result=_result(_answers(complexity="frontier")), delay=0.05)
    backend = ShadowBackend(StubBackend(), shadow, on_disagreement=seen.append)

    await backend.decide(_request())
    assert seen == []  # the shadow is still in flight
    assert await backend.join(timeout=2.0) == 1

    assert len(seen) == 1
    assert seen[0]["agrees"] is False
    await backend.aclose()


@pytest.mark.asyncio
async def test_callback_not_invoked_when_shadow_agrees() -> None:
    seen: list[dict[str, Any]] = []
    backend = ShadowBackend(StubBackend(), StubBackend(), mode="await", on_disagreement=seen.append)

    await backend.decide(_request())

    assert seen == []
    assert backend.stats()["callback_errors"] == 0
    await backend.aclose()


@pytest.mark.asyncio
async def test_broken_callback_is_swallowed_and_counted() -> None:
    def boom(report: dict[str, Any]) -> None:
        raise ValueError("callback bug")

    primary_result = _result()
    primary = StubBackend(result=primary_result)
    shadow = StubBackend(result=_result(_answers(complexity="frontier")))
    backend = ShadowBackend(primary, shadow, mode="await", on_disagreement=boom)

    out = await backend.decide(_request())  # must not raise

    assert out is primary_result
    stats = backend.stats()
    assert stats["callback_errors"] == 1
    assert stats["disagreements"] == 1  # recorded despite the broken callback
    assert len(backend.disagreements()) == 1
    await backend.aclose()


@pytest.mark.asyncio
async def test_log_disagreements_false_gates_the_callback_only() -> None:
    seen: list[dict[str, Any]] = []
    primary = StubBackend(result=_result())
    shadow = StubBackend(result=_result(_answers(complexity="frontier")))
    backend = ShadowBackend(primary, shadow, mode="await", log_disagreements=False, on_disagreement=seen.append)

    await backend.decide(_request())

    assert seen == []  # nothing exported...
    assert len(backend.reports()) == 1  # ...but introspection still works
    assert backend.stats()["disagreements"] == 1
    await backend.aclose()


# --------------------------------------------------------------------------- #
# 10. bounded ring buffer + report/stats contracts
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_reports_ring_buffer_is_bounded_by_max_reports() -> None:
    primary, shadow = StubBackend(), StubBackend()
    backend = ShadowBackend(primary, shadow, mode="await", max_reports=3)
    for i in range(5):
        await backend.decide(_request(request_id=f"req-{i}"))

    reports = backend.reports()
    assert len(reports) == 3
    assert [report["request_id"] for report in reports] == ["req-2", "req-3", "req-4"]
    stats = backend.stats()
    assert stats["requests"] == 5  # counters are cumulative, not derived from the buffer
    assert stats["completed"] == 5
    await backend.aclose()


@pytest.mark.asyncio
async def test_report_key_contract() -> None:
    primary = StubBackend()

    ok = ShadowBackend(primary, StubBackend(), mode="await")
    await ok.decide(_request(request_id="ok"))
    assert set(ok.reports()[0]) == _BASE_REPORT_KEYS
    await ok.aclose()

    timed_out = ShadowBackend(primary, StubBackend(delay=5.0), mode="await", shadow_timeout_seconds=0.05)
    await timed_out.decide(_request(request_id="slow"))
    assert set(timed_out.reports()[0]) == _BASE_REPORT_KEYS | {"timeout"}
    await timed_out.aclose()

    failed = ShadowBackend(primary, StubBackend(error=KeyError("k")), mode="await")
    await failed.decide(_request(request_id="bad"))
    assert set(failed.reports()[0]) == _BASE_REPORT_KEYS | {"error"}
    await failed.aclose()


@pytest.mark.asyncio
async def test_stats_key_contract() -> None:
    backend = ShadowBackend(StubBackend(), StubBackend(), mode="await")
    await backend.decide(_request())
    stats = backend.stats()
    for key in (
        "requests",
        "sampled",
        "completed",
        "timeouts",
        "errors",
        "degraded",
        "disagreements",
        "disagreement_rate",
        "callback_errors",
        "fields",
    ):
        assert key in stats
    assert set(stats["fields"]) == {"complexity", "sensitivity", "domain", "pii"}
    await backend.aclose()


@pytest.mark.asyncio
async def test_disagreement_rate_is_disagreements_over_completed() -> None:
    primary = StubBackend(name="primary")
    shadow = StubBackend(name="shadow")
    backend = ShadowBackend(primary, shadow, mode="await")

    await backend.decide(_request(request_id="agree"))
    shadow.result = _result(_answers(complexity="frontier"))
    await backend.decide(_request(request_id="disagree"))

    stats = backend.stats()
    assert stats["completed"] == 2
    assert stats["disagreements"] == 1
    assert stats["disagreement_rate"] == 0.5
    assert stats["fields"] == {"complexity": 1, "sensitivity": 0, "domain": 0, "pii": 0}
    await backend.aclose()


@pytest.mark.asyncio
async def test_reset_clears_reports_and_counters() -> None:
    shadow = StubBackend(result=_result(_answers(complexity="frontier")))
    backend = ShadowBackend(StubBackend(), shadow, mode="await")
    await backend.decide(_request())
    assert backend.reports()
    assert backend.stats()["requests"] == 1

    backend.reset()

    assert backend.reports() == ()
    assert backend.disagreements() == ()
    stats = backend.stats()
    assert stats["requests"] == 0
    assert stats["disagreements"] == 0
    assert stats["disagreement_rate"] == 0.0
    assert set(stats["fields"].values()) == {0}
    await backend.aclose()


# --------------------------------------------------------------------------- #
# primary failures, lifecycle
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["background", "await"])
async def test_primary_failure_propagates_and_leaves_no_tasks(mode: str) -> None:
    primary = StubBackend(name="primary", error=RuntimeError("primary down"))
    shadow = StubBackend(name="shadow", delay=1.0)
    backend = ShadowBackend(primary, shadow, mode=mode)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="primary down"):
        await backend.decide(_request())

    assert backend.reports() == ()  # no comparison exists, so no report exists
    stats = backend.stats()
    assert stats["requests"] == 1
    assert stats["sampled"] == 1
    assert stats["completed"] == 0
    await backend.aclose()
    await asyncio.sleep(0)
    assert not backend._tasks  # the orphaned shadow task was cancelled, not leaked


@pytest.mark.asyncio
async def test_aclose_closes_both_children_and_is_idempotent() -> None:
    primary = StubBackend(name="primary")
    shadow = StubBackend(name="shadow", delay=1.0)
    backend = ShadowBackend(primary, shadow, mode="background")
    await backend.decide(_request())  # leaves a shadow task in flight

    await backend.aclose()

    assert primary.closed == 1
    assert shadow.closed == 1
    await asyncio.sleep(0)
    assert not backend._tasks

    await backend.aclose()  # safe to call twice
    assert primary.closed == 2
    assert shadow.closed == 2


@pytest.mark.asyncio
async def test_aclose_swallows_child_close_errors() -> None:
    primary = StubBackend(name="primary")
    shadow = StubBackend(name="shadow", close_error=RuntimeError("close boom"))
    backend = ShadowBackend(primary, shadow)

    await backend.aclose()  # must not raise

    assert primary.closed == 1
    assert shadow.closed == 1


# --------------------------------------------------------------------------- #
# 11. factory integration
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_build_backend_returns_working_shadow_backend() -> None:
    backend = build_backend(
        {
            "backend": {
                "name": "shadow",
                "primary": {"name": "mock"},
                "shadow": {"name": "mock"},
                "shadow_timeout_seconds": 0.2,
            }
        }
    )
    assert isinstance(backend, ShadowBackend)
    assert backend.name == "shadow"
    assert backend.model_version == "mock-1.0.0"
    assert backend.mode == "background"  # the graduate-time default
    assert backend.log_disagreements is True
    assert backend.shadow_timeout_seconds == 0.2

    out = await backend.decide(_request("Draft a blog post about the migration."))

    assert isinstance(out, BackendResult)
    assert out.model_version == "mock-1.0.0"
    assert await backend.join(timeout=2.0) == 1
    report = backend.reports()[0]
    assert report["agrees"] is True  # mock vs mock at the same temperature
    assert report["sampled"] is True
    await backend.aclose()
