"""ShadowBackend: serve one backend's answer, measure another's.

Why this exists
---------------
The cutover from the cloud bootstrap (``JevBackend``) to a locally distilled
model (``DistilledBackend``) is only trustworthy if the local model can be
shown to agree with the cloud model *on your own live traffic*, not just on a
static eval set. :class:`ShadowBackend` is the instrument for that evidence.
It is a drop-in :class:`~jev_route.backends.base.DecisionBackend` that always
answers with the **primary** backend's result, object unchanged, while running
a **shadow** backend alongside purely for telemetry. Whatever the shadow says,
and whatever it does -- disagrees, throws, hangs, returns degraded mush -- the
caller receives the primary's answer. The primary path is the product; the
shadow is measurement, and measurement must never become the product.

How this differs from router-level shadow (``Router._run_shadow``)
------------------------------------------------------------------
Both exist, deliberately, and they compose (a router with ``shadow.enabled``
can wrap a ShadowBackend), but they answer different questions:

``Router._run_shadow`` (router-level)
    Configured under ``shadow:`` in the policy file. It runs *inline* -- the
    router awaits the shadow before returning the decision -- and writes the
    comparison into the decision record's ``shadow`` field. That is
    per-request provenance inside the training log, paid for with the
    shadow's latency on every sampled request.

``ShadowBackend`` (backend-level, this module)
    Selected via ``backend: {name: shadow}`` in the policy file -- the shape
    ``jev-route graduate`` writes at cutover, with the cloud model as primary
    and the freshly distilled local model as shadow. Telemetry lives in a
    bounded in-process ring buffer (``reports()`` / ``stats()``) instead of
    the decision log, and the shadow runs as a background asyncio task by
    default, so it adds **zero** latency to any request. That default is the
    reason this class exists separately from the router's shadow: a config you
    put in front of production traffic at cutover cannot tax every request
    for the sake of measuring it.

Latency contract
----------------
``mode="background"`` (default)
    ``decide()`` returns as soon as the primary answers. The shadow task
    outlives the call, still bounded by ``shadow_timeout_seconds`` via
    ``asyncio.wait_for`` (a hung shadow is *cancelled*, not merely ignored),
    and its report lands when it settles. Call :meth:`join` to make that
    deterministic -- tests and ``graduate --replay`` do; production does not
    need to.
``mode="await"``
    ``decide()`` runs primary and shadow concurrently and returns only after
    both settle. Added latency is at most ``shadow_timeout_seconds`` (the
    primary's own latency is unaffected by the shadow). Useful for offline
    replay, where determinism matters more than throughput.

Failure contract
----------------
A shadow timeout, exception, or degraded result is recorded in the report --
as ``timeout``, ``error``, or ``shadow_degraded`` -- and never changes or
raises out of :meth:`decide`. Such reports carry ``agrees: None`` rather than
``True`` or ``False``: a shadow that could not answer tells you nothing, and
counting its silence as agreement (or as disagreement) would corrupt the one
number the whole exercise exists to produce, ``stats()["disagreement_rate"]``.

Sampling is deterministic per request (``request_id`` when present, else the
excerpt hash, via ``hashlib.sha256`` -- never the salted builtin ``hash()``),
so a replay of the same log shadows exactly the same slice of traffic.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import time
from collections import deque
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from ..schema import DecisionAnswers
from .base import BackendResult, DecisionBackend, DecisionRequest, call_with_timeout

#: Absolute difference in the PII noul value at or above which the two backends
#: are said to disagree. Mirrors ``Router._run_shadow``; defined locally because
#: importing :mod:`jev_route.router` from a backend would close the import cycle
#: router -> backends -> shadow -> router.
PII_DISAGREEMENT_TOLERANCE = 0.25

#: The categorical fields compared by argmax choice. The full probability
#: distributions legitimately differ between backends (calibration is not a
#: fingerprint), and both are already preserved elsewhere; the *decision* is the
#: choice, so the choice is what a disagreement means.
CHOICE_FIELDS: tuple[str, ...] = ("complexity", "sensitivity", "domain")

#: All fields a disagreement can be recorded against: the choice fields plus the
#: PII noul. Fixed shape so ``stats()["fields"]`` is predictable for the CLI.
DISAGREEMENT_FIELDS: tuple[str, ...] = (*CHOICE_FIELDS, "pii")

ShadowMode = Literal["background", "await"]

_T = TypeVar("_T")

#: Counters tracked cumulatively in ``stats()``. Cumulative, not derived from the
#: ring buffer, because the buffer is bounded and old reports legitimately fall
#: out of it while the counts must not.
_COUNTER_KEYS: tuple[str, ...] = (
    "requests",
    "sampled",
    "completed",
    "timeouts",
    "errors",
    "degraded",
    "disagreements",
    "callback_errors",
)


@dataclass(frozen=True)
class _ShadowOutcome:
    """What one bounded shadow call produced.

    Exactly one of ``result`` / ``error`` / ``timeout`` is set. A dataclass
    rather than a tuple because the three failure shapes are read in several
    places and a boolean-plus-optional soup is how telemetry bugs hide.
    """

    result: BackendResult | None = None
    error: str | None = None
    timeout: bool = False
    #: Wall-clock milliseconds of the attempt, measured by *this* class rather
    #: than trusting ``result.latency_ms``: a timed-out or raising shadow has no
    #: self-reported latency, and the operator cares about the bound spent.
    latency_ms: float = 0.0


def _compare_answers(primary: DecisionAnswers, shadow: DecisionAnswers) -> dict[str, Any]:
    """Disagreement map between two answers. Mirrors ``Router._run_shadow``.

    PII is compared with a tolerance (:data:`PII_DISAGREEMENT_TOLERANCE`)
    because 0.62 vs 0.58 is calibration noise while 0.9 vs 0.1 is a real
    disagreement worth an alert; the categorical fields need no tolerance --
    a different argmax *is* a different routing decision.
    """
    disagreements: dict[str, Any] = {}
    for field_name in CHOICE_FIELDS:
        a = getattr(primary, field_name).choice
        b = getattr(shadow, field_name).choice
        if a != b:
            disagreements[field_name] = {"primary": a, "shadow": b}
    delta = abs(primary.pii.value - shadow.pii.value)
    if delta >= PII_DISAGREEMENT_TOLERANCE:
        disagreements["pii"] = {
            "primary": round(primary.pii.value, 4),
            "shadow": round(shadow.pii.value, 4),
        }
    return disagreements


class ShadowBackend:
    """A drop-in backend that always answers as ``primary`` and measures ``shadow``.

    Implements :class:`~jev_route.backends.base.DecisionBackend`, so anything in
    the system that takes a backend -- the router, the LiteLLM integrations, the
    CLI -- can sit on top of a shadow pair without knowing. ``name`` is
    ``"shadow"`` and ``model_version`` mirrors the *primary's current* version,
    because the primary is what actually answered: a decision record stamped
    with the shadow's version would be a lie about provenance.

    Not thread-safe, by the same design decision as
    :class:`~jev_route.backends.base.CircuitBreaker`: one asyncio event loop
    owns an instance.
    """

    #: Stable identifier recorded on every decision. Config-facing.
    name = "shadow"

    def __init__(
        self,
        primary: DecisionBackend,
        shadow: DecisionBackend,
        *,
        log_disagreements: bool = True,
        shadow_timeout_seconds: float = 5.0,
        mode: ShadowMode = "background",
        sample_rate: float = 1.0,
        max_reports: int = 256,
        on_disagreement: Callable[[dict[str, Any]], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if mode not in ("background", "await"):
            raise ValueError(f"mode must be background|await, got {mode!r}")
        if int(max_reports) < 0:
            raise ValueError(f"max_reports must be >= 0, got {max_reports!r}")
        self.primary = primary
        self.shadow = shadow
        #: Gates the *outbound* side effect only: with ``False``, no
        #: ``on_disagreement`` callback fires. The in-memory ring buffer and
        #: ``stats()`` keep recording regardless -- they are this backend's own
        #: bounded introspection API, not an external log destination, and
        #: silently emptying them would break ``graduate`` rather than protect
        #: anything.
        self.log_disagreements = bool(log_disagreements)
        #: Upper bound on one shadow attempt. ``<= 0`` means unbounded, matching
        #: :func:`~jev_route.backends.base.call_with_timeout` everywhere else in
        #: the package; ``aclose()`` is then the only thing that stops a straggler.
        self.shadow_timeout_seconds = float(shadow_timeout_seconds)
        self.mode: ShadowMode = mode
        self.sample_rate = float(sample_rate)
        self.max_reports = int(max_reports)
        self.on_disagreement = on_disagreement
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic
        self._reports: deque[dict[str, Any]] = deque(maxlen=self.max_reports)
        #: Strong references to every live shadow task. The event loop holds only
        #: weak references, so without this set a background task could be garbage
        #: collected mid-flight ("Task was destroyed but it is pending"); the done
        #: callback discards entries, which is what keeps the set itself bounded.
        self._tasks: set[asyncio.Task[Any]] = set()
        self._counts: dict[str, int] = dict.fromkeys(_COUNTER_KEYS, 0)
        self._field_counts: dict[str, int] = dict.fromkeys(DISAGREEMENT_FIELDS, 0)

    @property
    def model_version(self) -> str:
        """The primary's *current* model version.

        A property, not a snapshot taken in ``__init__``: a cloud backend learns
        the exact version that served a request only from the response, so the
        value can legitimately change after construction, and every decision
        record stamped through this backend must carry the truth as of the read.
        """
        return self.primary.model_version

    # -- the DecisionBackend interface ------------------------------------ #
    async def decide(self, request: DecisionRequest) -> BackendResult:
        """Answer via the primary; measure the shadow; never let it interfere.

        Returns the primary's :class:`BackendResult` object itself -- not a copy,
        not a merge -- so downstream provenance (``questions_sent``,
        ``latency_ms``, ``degraded``) is exactly what the primary produced.
        """
        self._counts["requests"] += 1
        if not self._sampled(request):
            return await self.primary.decide(request)
        self._counts["sampled"] += 1
        if self.mode == "await":
            return await self._decide_await(request)
        return await self._decide_background(request)

    async def aclose(self) -> None:
        """Cancel in-flight shadow tasks, then close both children (each guarded).

        Tasks are cancelled *before* the children close so no shadow call can be
        mid-flight against an already-closed backend. Safe to call more than
        once: the second call finds no tasks, and children are closed again
        behind a guard -- the :class:`DecisionBackend` protocol already requires
        ``aclose()`` to be idempotent, and a child that violates that (or fails
        to close at all) is swallowed here because a shutdown path that can
        raise is a shutdown path operators learn to skip. Mirrors
        ``Router.aclose``.
        """
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for child in (self.primary, self.shadow):
            with contextlib.suppress(Exception):  # shutdown must not raise; see docstring
                await child.aclose()

    # -- introspection (graduate and the CLI read these) ------------------- #
    def reports(self) -> tuple[dict[str, Any], ...]:
        """Every retained report, oldest first, as a snapshot tuple."""
        return tuple(self._reports)

    def disagreements(self) -> tuple[dict[str, Any], ...]:
        """Only the reports where the shadow gave a real, differing answer.

        ``agrees is False`` -- not merely falsy: reports for timed-out, raising,
        or degraded shadows carry ``agrees=None`` and are excluded, because a
        shadow that could not answer is not a shadow that objected.
        """
        return tuple(report for report in self._reports if report["agrees"] is False)

    def stats(self) -> dict[str, Any]:
        """Cumulative counters since construction (or last :meth:`reset`).

        ``completed`` counts shadow attempts that returned a result (degraded or
        not); timeouts and errors are tracked separately and are *not*
        completions. ``disagreement_rate`` is ``disagreements / completed`` --
        the fraction of shadowed traffic where both backends actually answered
        and gave different decisions. It is the number ``graduate`` compares
        against its threshold, which is exactly why degraded shadows count as
        completed-but-not-disagreeing rather than being dropped from the
        denominator silently: a candidate model that degrades constantly shows
        up in ``degraded``, not as a flattering 0% disagreement rate.

        ``fields`` breaks disagreements down per field, so an operator can see
        whether a candidate model is wrong about complexity or about PII --
        those have very different blast radii.
        """
        counts = dict(self._counts)
        completed = counts["completed"]
        counts["disagreement_rate"] = round(counts["disagreements"] / completed, 6) if completed else 0.0
        counts["fields"] = dict(self._field_counts)
        return counts

    def reset(self) -> None:
        """Forget every report and counter (a fresh stats window).

        In-flight shadow tasks are deliberately untouched: one that is already
        running records into the fresh buffer when it lands, which is the least
        surprising behaviour for a live dashboard and avoids ``reset()``
        silently deleting telemetry that was already paid for.
        """
        self._reports.clear()
        for key in self._counts:
            self._counts[key] = 0
        for key in self._field_counts:
            self._field_counts[key] = 0

    async def join(self, timeout: float | None = None) -> int:
        """Await outstanding background shadow tasks; return how many settled.

        Exists because background mode is deliberately fire-and-forget, and
        tests, ``graduate --replay``, and any CLI stats dump need a way to make
        that telemetry deterministic without changing production behaviour. If
        ``timeout`` expires, returns the partial count and leaves the stragglers
        running -- :meth:`aclose` is the way to stop them. In ``await`` mode
        there is never anything outstanding, so this returns 0.
        """
        tasks = [task for task in list(self._tasks) if not task.done()]
        if not tasks:
            return 0
        done, _pending = await asyncio.wait(tasks, timeout=timeout)
        return len(done)

    # -- the two modes ------------------------------------------------------ #
    async def _decide_background(self, request: DecisionRequest) -> BackendResult:
        """Primary in the foreground, shadow as a tracked background task.

        The shadow task is spawned *before* the primary is awaited, so an
        expensive (possibly cloud) shadow call overlaps the primary instead of
        trailing it. The task receives the primary's result through a future set
        when the primary lands; comparing after its own call finishes keeps the
        recorded shadow latency about the shadow.
        """
        loop = asyncio.get_running_loop()
        primary_done: asyncio.Future[BackendResult] = loop.create_future()
        task = self._spawn(self._shadow_then_record(request, primary_done))
        try:
            result = await self.primary.decide(request)
        except BaseException:
            # The primary failed, so there is no comparison to make -- and no
            # telemetry task may outlive the call that created it. Cancel the
            # task rather than resolving the future with an exception: an
            # exception nobody retrieves would surface later as asyncio's
            # "Future exception was never retrieved" noise.
            task.cancel()
            raise
        if not primary_done.done():
            primary_done.set_result(result)
        return result

    async def _decide_await(self, request: DecisionRequest) -> BackendResult:
        """Both settle before returning; added latency <= ``shadow_timeout_seconds``.

        The shadow is still a tracked task (not a bare ``await``), so a
        cancellation of *this* coroutine -- e.g. a client disconnect propagating
        through the router -- cancels the shadow with it instead of orphaning it.
        """
        task = self._spawn(self._attempt(request))
        try:
            result = await self.primary.decide(request)
            outcome = await task
        except BaseException:
            task.cancel()
            raise
        self._record(request, result, outcome)
        return result

    async def _shadow_then_record(
        self,
        request: DecisionRequest,
        primary_done: asyncio.Future[BackendResult],
    ) -> None:
        """Background task body: bounded shadow call, then compare, then record."""
        outcome = await self._attempt(request)
        try:
            primary_result = await primary_done
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - defensive; see _decide_background
            # decide() cancels this task when the primary fails; this branch is
            # the belt to that suspenders, so an unexpected future state can
            # never turn into an unretrieved-task-exception warning.
            return
        self._record(request, primary_result, outcome)

    async def _attempt(self, request: DecisionRequest) -> _ShadowOutcome:
        """One shadow call under the timeout bound. Never raises: that is the job.

        ``call_with_timeout`` cancels the inner coroutine on expiry, so a hung
        shadow stops consuming its own resources (and, for a cloud shadow, stops
        billing) rather than merely being ignored. :class:`asyncio.CancelledError`
        is a ``BaseException`` and deliberately propagates: cancellation is the
        one signal a telemetry task must obey.
        """
        started = self._clock()
        try:
            result = await call_with_timeout(self.shadow.decide(request), self.shadow_timeout_seconds)
        except asyncio.TimeoutError:
            return _ShadowOutcome(timeout=True, latency_ms=self._elapsed_ms(started))
        except Exception as exc:  # recorded into the report; never re-raised
            return _ShadowOutcome(error=f"{type(exc).__name__}: {exc}", latency_ms=self._elapsed_ms(started))
        return _ShadowOutcome(result=result, latency_ms=self._elapsed_ms(started))

    # -- sampling, recording, tasks ----------------------------------------- #
    def _sampled(self, request: DecisionRequest) -> bool:
        """Deterministic per-request sampling of *which* requests get shadowed.

        Keyed on ``request_id`` when present, else on the redacted excerpt, and
        hashed with ``hashlib.sha256`` -- never the builtin ``hash()``, whose
        per-process salt would sample a different slice of traffic on every
        restart and make replays incomparable. Same key, same verdict, in every
        process and every run: that is what lets ``graduate --replay`` reproduce
        a shadow evaluation exactly. Mirrors ``Router._sample_hit``.
        """
        rate = self.sample_rate
        if rate >= 1.0:
            return True
        if rate <= 0.0:
            return False
        key = request.request_id or hashlib.sha256((request.redacted_excerpt or "").encode("utf-8")).hexdigest()
        bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        return bucket < rate

    def _build_report(
        self,
        request: DecisionRequest,
        primary_result: BackendResult,
        outcome: _ShadowOutcome,
    ) -> dict[str, Any]:
        """Assemble the report dict. Key set is a contract; see ``stats()``.

        ``agrees`` is tri-state on purpose: ``True``/``False`` only when the
        shadow produced a real (non-degraded) answer, ``None`` when it timed
        out, raised, or degraded -- all three of which mean "no opinion", not
        "no objection".
        """
        report: dict[str, Any] = {
            "request_id": request.request_id,
            "primary_backend": self.primary.name,
            "shadow_backend": self.shadow.name,
            "shadow_model_version": outcome.result.model_version if outcome.result is not None else None,
            "agrees": None,
            "disagreements": {},
            "shadow_latency_ms": outcome.latency_ms,
            "shadow_degraded": False,
            "sampled": True,
            "mode": self.mode,
        }
        if outcome.timeout:
            report["timeout"] = True
        elif outcome.error is not None:
            report["error"] = outcome.error
        elif outcome.result is not None:
            if outcome.result.degraded:
                # Recorded, not compared: a degraded result is maximum-uncertainty
                # defaults by construction, so "disagreeing" with it is an artifact
                # of the outage, not a property of the model.
                report["shadow_degraded"] = True
            else:
                disagreements = _compare_answers(primary_result.answers, outcome.result.answers)
                report["disagreements"] = disagreements
                report["agrees"] = not disagreements
        return report

    def _record(
        self,
        request: DecisionRequest,
        primary_result: BackendResult,
        outcome: _ShadowOutcome,
    ) -> None:
        """Append one report and fold it into the cumulative counters."""
        report = self._build_report(request, primary_result, outcome)
        self._reports.append(report)
        counts = self._counts
        if report.get("timeout"):
            counts["timeouts"] += 1
        elif report.get("error") is not None:
            counts["errors"] += 1
        else:
            counts["completed"] += 1
            if report["shadow_degraded"]:
                counts["degraded"] += 1
        if report["agrees"] is False:
            counts["disagreements"] += 1
            for field_name in report["disagreements"]:
                self._field_counts[field_name] += 1
            self._notify(report)

    def _notify(self, report: dict[str, Any]) -> None:
        """Hand a disagreement to the operator's callback, if one is wired up.

        Guarded because this fires on the routing path (in background mode,
        inside the telemetry task): a broken callback must not break routing,
        and must not lose the record of the disagreement it was called to
        report -- hence a counter rather than a raise. ``callback_errors`` going
        up means the callback is broken, not the shadow.

        The callback receives a shallow copy, so it cannot mutate the buffered
        report; it is called synchronously on the event loop, so it must be
        fast and non-blocking (schedule your own task if it is not).
        """
        if not self.log_disagreements or self.on_disagreement is None:
            return
        try:
            self.on_disagreement(dict(report))
        except Exception:  # counted, never propagated: see docstring
            self._counts["callback_errors"] += 1

    def _spawn(self, coro: Coroutine[Any, Any, _T]) -> asyncio.Task[_T]:
        """Create a task on the running loop and keep a strong reference to it."""
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Future[Any]) -> None:
        """Drop the finished task from the tracking set; consume any exception.

        Retrieving the exception (always ``None`` by construction -- nothing in
        a shadow task may raise) is the seatbelt against asyncio logging
        "Task exception was never retrieved" should that construction ever be
        violated by a future edit.
        """
        self._tasks.discard(task)
        if not task.cancelled():
            task.exception()

    def _elapsed_ms(self, started: float) -> float:
        return round((self._clock() - started) * 1000.0, 3)


__all__ = [
    "CHOICE_FIELDS",
    "DISAGREEMENT_FIELDS",
    "PII_DISAGREEMENT_TOLERANCE",
    "ShadowBackend",
    "ShadowMode",
]
