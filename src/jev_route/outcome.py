"""Outcome verification: the decision log records what the router decided;
this module records whether it was RIGHT.

A post-call hook builds state from the request summary plus a truncated
response excerpt (capped), asks ONE noul question -- ``task_completed`` -- on
the configured DecisionBackend, and appends an :class:`OutcomeRecord` to the
SAME decision log, linked by request id. The distill pipeline joins on
``request_id``.

Standing rules:
* requests the sensitivity gate blocked or diverted are NEVER verified --
  blocked content goes nowhere, including to Jev for verification;
* verification failures log ``outcome: null`` -- never retry-storm, never
  affect the already-returned response;
* ``outcome_verification.enabled`` is default-false in the policy.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

#: Discriminator for the outcome stream -- appended to the same JSONL, joined
#: by request_id downstream.
OUTCOME_RECORD_KIND = "jev_route.outcome"

#: Response excerpts are capped at this many characters (~500 tokens) before
#: being shown to the verification backend.
MAX_RESPONSE_CHARS = 2000

OUTCOME_INSTRUCTIONS = (
    "Given a request summary and the model's response, decide whether the "
    "response adequately completes the request. Answer true only if the "
    "response addresses what was asked at a usable level of quality."
)


@dataclass(frozen=True)
class OutcomeRecord:
    """One verification result, linked to a decision by ``request_id``.

    ``completed_p`` is None on verification failure (the record still lands,
    so operators can see the verification layer itself failing).
    """

    request_id: str
    completed_p: float | None
    model_served: str
    backend: str
    checked_at: float
    sampled: bool
    kind: str = OUTCOME_RECORD_KIND
    schema_version: str = "1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


class OutcomeVerifier:
    """Ask the decision backend whether the response completed the task.

    The backend seam is an OPTIONAL ``noul(state, instructions) -> float``
    method (JevBackend, MockBackend, LayaBackend implement it; backends that
    do not are simply skipped -- verification is additive).
    """

    def __init__(self, backend: Any, sink: Any, *, sample_rate: float = 1.0) -> None:
        self.backend = backend
        self.sink = sink
        self.sample_rate = max(0.0, min(1.0, sample_rate))

    def _sampled(self, request_id: str) -> bool:
        if self.sample_rate >= 1.0:
            return True
        # deterministic on the request id: the same request always (never)
        # verifies, so a replayed trace is reproducible.
        return (hash(request_id) % 10_000) / 10_000.0 < self.sample_rate

    async def maybe_verify(
        self,
        *,
        request_id: str,
        request_summary: str,
        response_excerpt: str,
        model_served: str,
        gate_blocked: bool,
    ) -> OutcomeRecord | None:
        """Verify if enabled conditions hold; return the record (or None when
        not sampled). Never raises: a broken verifier must not break serving."""
        if gate_blocked:
            return None  # blocked content is never sent anywhere, ever
        sampled = self._sampled(request_id)
        if not sampled:
            return None
        noul = getattr(self.backend, "noul", None)
        if noul is None:
            return None
        state = {
            "request_summary": request_summary[:500],
            "response_excerpt": response_excerpt[:MAX_RESPONSE_CHARS],
        }
        completed_p: float | None
        try:
            if callable(getattr(noul, "__wrapped__", None)):  # pragma: no cover
                pass
            import inspect

            if inspect.iscoroutinefunction(noul):
                completed_p = float(await noul(state, OUTCOME_INSTRUCTIONS))
            else:
                completed_p = float(noul(state, OUTCOME_INSTRUCTIONS))
            completed_p = 0.0 if math.isnan(completed_p) else max(0.0, min(1.0, completed_p))
        except Exception:
            completed_p = None
        record = OutcomeRecord(
            request_id=request_id,
            completed_p=completed_p,
            model_served=model_served,
            backend=str(getattr(self.backend, "name", "unknown")),
            checked_at=time.time(),
            sampled=True,
        )
        import contextlib

        with contextlib.suppress(Exception):
            # logging must never take serving down either
            self.sink.write(record)
        return record
