"""The DecisionBackend interface: the architectural center of jev-route.

Everything above this module -- the policy engine, the router, the LiteLLM
integrations, the CLI -- speaks to a :class:`DecisionBackend` and has no idea
whether the answer came from TypeSafe's cloud, from a deterministic mock, or from
a model distilled out of your own traffic on your own hardware.

That indirection is not for tidiness. It is the mechanism that makes the project's
promise real: you bootstrap on a calibrated cloud model, your traffic accumulates
as labeled decisions, you distill a local model from them, and then you swap one
config value and the cloud dependency is gone. Nothing else in the system changes,
because nothing else in the system ever knew.

Four implementations ship with the package:

``JevBackend``
    Cloud. The day-one path: needs an API key, no dataset, no training.
``MockBackend``
    Deterministic, offline, no key. Runs the whole system end to end so tests,
    demos, and CI never depend on a network or a paid account.
``DistilledBackend``
    Local. The end state: serves a model trained on your own decision log.
``LayaBackend``
    Local, pretrained. The air-gapped edition: serves the open-source Laya
    System One model (arXiv:2503.23303) on local hardware, no key, no egress.
    Its calibrated probabilities are a hard precondition of ``enforce`` mode.

Implementations must obey two invariants:

* **Return the full schema.** A backend that cannot answer a question returns
  that question at maximum uncertainty, never omits it. The policy engine and the
  log format both assume the schema is total.
* **Never raise for an expected outage.** Timeouts, rate limits, and circuit-open
  conditions come back as a degraded :class:`BackendResult`; only programmer
  errors propagate. The router's fail-closed behaviour depends on this.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..schema import DecisionAnswers, RequestFeatures


@dataclass(frozen=True)
class DecisionRequest:
    """Everything a backend is allowed to see about one request.

    Deliberately contains no raw prompt text: only the *redacted* excerpt, the
    deterministic features, the gate's advisory topic hints, and non-payload
    metadata. A backend that receives this cannot leak what the gate already
    removed, which is the point of building the request object this way.
    """

    redacted_excerpt: str
    features: RequestFeatures
    #: Topic keywords the local gate matched. Hints, not verdicts: the backend
    #: may weigh them, and the policy engine will not let them relax a floor.
    advisory_topics: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    request_id: str = ""

    def state(self) -> dict[str, Any]:
        """The structured ``state`` payload handed to a System One model."""
        meta = dict(self.metadata or {})
        for unsafe in ("messages", "prompt", "content", "input", "raw", "body"):
            meta.pop(unsafe, None)
        return {
            "prompt_excerpt": self.redacted_excerpt,
            "request_features": self.features.to_dict(),
            "local_gate_topic_hints": list(self.advisory_topics),
            "caller_metadata": meta,
        }


@dataclass(frozen=True)
class BackendResult:
    """A backend's answer, plus the provenance the decision log needs."""

    answers: DecisionAnswers
    model_version: str
    #: The questions actually sent, verbatim, so a logged decision can be
    #: reproduced. Empty for backends that do not send questions.
    questions_sent: Mapping[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    #: True when the backend could not answer and returned maximum-uncertainty
    #: defaults. The router treats this as fail-closed input, not as a real read.
    degraded: bool = False
    degrade_reason: str | None = None


class BackendError(Exception):
    """Raised only for programmer errors, never for outages."""


@runtime_checkable
class DecisionBackend(Protocol):
    """The interface every backend implements."""

    #: Stable identifier recorded on every decision. Config-facing.
    name: str
    #: Version of the underlying model, recorded for reproducibility. A cloud
    #: backend learns this from the response; a distilled one from its artifact.
    model_version: str

    async def decide(self, request: DecisionRequest) -> BackendResult:
        """Answer the routing questions for one request. Must not raise on outage."""
        ...

    async def aclose(self) -> None:
        """Release resources. Safe to call more than once."""
        ...


# --------------------------------------------------------------------------- #
# Circuit breaker
# --------------------------------------------------------------------------- #
class CircuitBreaker:
    """Minimal three-state breaker: closed -> open -> half-open -> closed.

    Exists so a decisioning outage degrades the router instead of adding a full
    timeout to every single request. Without it, a dead backend turns a 5 ms
    routing decision into ``N x timeout`` and the proxy looks hung rather than
    fail-closed.

    Not thread-safe by design: it is used from one asyncio event loop.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_seconds: float = 30.0,
        half_open_max_calls: int = 1,
        clock: Any = time.monotonic,
    ) -> None:
        self.failure_threshold = max(1, failure_threshold)
        self.recovery_seconds = max(0.0, recovery_seconds)
        self.half_open_max_calls = max(1, half_open_max_calls)
        self._clock = clock
        self._failures = 0
        self._state = "closed"
        self._opened_at = 0.0
        self._half_open_in_flight = 0

    @property
    def state(self) -> str:
        """Current state, recomputed so an expired open breaker reads half-open."""
        if self._state == "open" and (self._clock() - self._opened_at) >= self.recovery_seconds:
            self._state = "half-open"
            self._half_open_in_flight = 0
        return self._state

    def allow(self) -> bool:
        """Whether a call may proceed right now."""
        state = self.state
        if state == "closed":
            return True
        if state == "half-open":
            return self._half_open_in_flight < self.half_open_max_calls
        return False

    def on_start(self) -> None:
        if self.state == "half-open":
            self._half_open_in_flight += 1

    def on_success(self) -> None:
        self._failures = 0
        self._half_open_in_flight = 0
        self._state = "closed"

    def on_failure(self) -> None:
        self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
        if self._state == "half-open":
            self._trip()
            return
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._trip()

    def _trip(self) -> None:
        self._state = "open"
        self._opened_at = self._clock()
        self._failures = 0

    def reset(self) -> None:
        self._failures = 0
        self._half_open_in_flight = 0
        self._state = "closed"


async def call_with_timeout(coro: Any, timeout_seconds: float) -> Any:
    """Await ``coro`` under a timeout, cancelling it cleanly on expiry."""
    if timeout_seconds <= 0:
        return await coro
    return await asyncio.wait_for(coro, timeout=timeout_seconds)


__all__ = [
    "BackendError",
    "BackendResult",
    "CircuitBreaker",
    "DecisionBackend",
    "DecisionRequest",
    "call_with_timeout",
]
