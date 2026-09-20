"""The jev-route evaluation harness: run the labelled dataset through the router.

What this script is for
-----------------------
The project claims that routing on a *calibrated* decision model beats routing on
a regex, and that the uncertainty signal is worth paying for. Both claims are
falsifiable, so this harness falsifies-or-confirms them against
``evals/data/labeled_prompts.jsonl`` and writes the numbers down where anyone can
check them against the raw per-prompt results.

Three design rules shaped the implementation, and each one exists because the
obvious alternative produces a number that is technically computed and actually
meaningless:

**1. One API pass, many policy passes.** The escalation A/B (``on_uncertain``
enabled vs disabled) and the gate-coverage variant all run over the *same* raw
backend answers. Calling Jev once per configuration would measure API
nondeterminism alongside the policy change and would triple the spend, so
:class:`RecordingBackend` memoizes on the same key the router's own cache uses.
A memo hit is literally "the router would have served this from cache", which
makes the A/B an exact counterfactual rather than a second sample.

**2. Failures become rows, never silence.** A prompt that exhausts its retries
is written to the JSONL with ``error`` set and is excluded from the accuracy
denominator *by name*, with the excluded count printed in the report. An
evaluation of 223 prompts that quietly scored 210 is how benchmark numbers
become fiction.

**3. Ground truth never reaches the model.** The router forwards caller
``metadata`` into the decision request's ``state``. Only the row id is passed, so
nothing in ``labels``/``expected_tier``/``difficulty`` can leak into the
judgement being measured. See :func:`_eval_metadata`.

Usage
-----
::

    # full run against the real Jev backend AND the offline mock, then report
    .venv/bin/python evals/run_eval.py

    # re-analyse the persisted JSONL without spending another API call
    .venv/bin/python evals/run_eval.py --reuse

    # offline only
    .venv/bin/python evals/run_eval.py --backend mock

Outputs land in ``evals/results/``: one JSONL per (backend, configuration), a
machine-readable ``summary.json``, ``reliability.json`` for chart rendering, and
the human-readable ``REPORT.md``.

The API key is read from the environment (``.env`` via python-dotenv if present)
and is never printed, logged, or written to any output file.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVALS_DIR.parent

# The harness has to work before `pip install -e .` has been run, and it has to
# be able to import its sibling metric modules when executed as a plain script
# (`python evals/run_eval.py`) rather than as part of a package. Both shims are
# explicit rather than relying on cwd, so the script works from anywhere.
for _path in (str(EVALS_DIR), str(REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import calibration  # noqa: E402  (path shim above must run first)
import cost_model  # noqa: E402

from jev_route.backends.base import (  # noqa: E402
    BackendResult,
    CircuitBreaker,
    DecisionBackend,
    DecisionRequest,
)
from jev_route.backends.jev import JevBackend  # noqa: E402
from jev_route.backends.mock import MockBackend  # noqa: E402
from jev_route.cache import InMemoryTTLCache, NullCache  # noqa: E402
from jev_route.logging_sink import NullSink  # noqa: E402
from jev_route.policy import Policy  # noqa: E402
from jev_route.router import Router  # noqa: E402
from jev_route.schema import (  # noqa: E402
    COMPLEXITY_LEVELS,
    DOMAINS,
    SENSITIVITY_LEVELS,
    TIERS,
)

DATASET_PATH = EVALS_DIR / "data" / "labeled_prompts.jsonl"
POLICY_PATH = REPO_ROOT / "policies" / "default.yaml"
RESULTS_DIR = EVALS_DIR / "results"

#: The three policy configurations measured per backend, in the order they must
#: run. Order matters: the first populates the memo, the rest are free.
CONFIG_DEFAULT = "default"
CONFIG_NO_ESCALATION = "no_escalation"
CONFIG_STILL_CLASSIFY = "still_classify"

#: Complexity/sensitivity confidence floors fully disabled, and the "an uncertain
#: PII noul counts as present" rule off. This is the ablation that isolates what
#: ``on_uncertain`` buys: same gate, same rules, same backend answers, no bumps.
NO_UNCERTAIN = {
    "sensitivity_confidence_below": None,
    "sensitivity_bump_levels": 0,
    "complexity_confidence_below": None,
    "complexity_bump_levels": 0,
    "pii_uncertain_threshold": 0.0,
    "pii_uncertain_counts_as_present": False,
}


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DatasetRow:
    """One labelled prompt. Immutable: nothing downstream may edit a label."""

    id: str
    text: str
    labels: Mapping[str, Any]
    expected_tier: str
    difficulty: str
    notes: str

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def words(self) -> int:
        return len(self.text.split())


def load_dataset(path: str | Path = DATASET_PATH) -> tuple[DatasetRow, ...]:
    """Read the labelled JSONL. Refuses to run on a file the validator would reject.

    Only the six documented keys are read. A row missing ``expected_tier`` or
    carrying an unknown tier is a hard error here rather than a skipped row,
    because silently dropping rows is exactly the failure mode rule 2 above
    exists to prevent.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"dataset not found: {p}")
    rows: list[DatasetRow] = []
    seen: set[str] = set()
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{p}:{lineno}: blank line in dataset")
        raw = json.loads(line)
        rid = str(raw["id"])
        if rid in seen:
            raise ValueError(f"{p}:{lineno}: duplicate id {rid!r}")
        seen.add(rid)
        tier = str(raw["expected_tier"])
        if tier not in TIERS:
            raise ValueError(f"{p}:{lineno}: unknown expected_tier {tier!r}")
        rows.append(
            DatasetRow(
                id=rid,
                text=str(raw["text"]),
                labels=dict(raw["labels"]),
                expected_tier=tier,
                difficulty=str(raw.get("difficulty", "")),
                notes=str(raw.get("notes", "")),
            )
        )
    if not rows:
        raise ValueError(f"{p}: dataset is empty")
    return tuple(rows)


def _eval_metadata(row: DatasetRow, config: str, backend_name: str) -> dict[str, Any]:
    """Caller metadata for one eval request.

    Deliberately carries no ground truth. The router passes ``metadata`` straight
    into ``DecisionRequest.state()``, which the backend shows the model, so any
    label copied in here would contaminate the measurement. Only opaque tracing
    fields are allowed, and this function is the single place that decides what
    those are.
    """
    return {"eval_id": row.id, "eval_config": config, "eval_backend": backend_name}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Recording + memoizing backend wrapper
# --------------------------------------------------------------------------- #
@dataclass
class CallTrace:
    """What one logical decision cost, in attempts."""

    attempts: int = 0
    memo_hit: bool = False
    degrade_reasons: list[str] = field(default_factory=list)


class RecordingBackend:
    """Wraps a real :class:`DecisionBackend` and captures what it actually said.

    Two jobs, neither of which the router can do for us:

    *Capture.* The router records the *effective* answers -- the argmax after the
    gate floor and the confidence bumps have been applied -- while keeping the
    original distribution. Calibration and component-label accuracy both need the
    pre-merge argmax, so the raw :class:`BackendResult` is stored per request id
    as it passes through.

    *Memoize.* Several policy configurations are evaluated over one backend pass
    (see the module docstring). The memo key is the router's own cache key minus
    the policy version, so a hit means "identical redacted excerpt and identical
    gate hints", which is precisely the condition under which the router itself
    would have served a cached answer.

    Retries live here rather than in the runner because a degraded
    :class:`BackendResult` is indistinguishable from a real answer once it is
    inside the router: the router turns it into a fail-closed ``local`` tier,
    which would then be scored as a *correct* routing decision on every
    sensitive row. Retrying until the backend genuinely gives up, and recording
    every attempt, keeps that contamination out of the accuracy numbers.
    """

    def __init__(
        self,
        inner: DecisionBackend,
        *,
        max_attempts: int = 4,
        retry_base_seconds: float = 0.5,
        seed: int | None = 0,
    ) -> None:
        self.inner = inner
        self.name = inner.name
        self.max_attempts = max(1, max_attempts)
        self.retry_base_seconds = retry_base_seconds
        #: Seeded RNG so a re-run of a failed harness is reproducible. Jitter is
        #: still applied -- without it, N concurrent retries re-fire in lockstep
        #: and turn a rate limit into a thundering herd that never clears.
        self._rng = random.Random(seed)
        self.raw_by_request: dict[str, BackendResult] = {}
        self.traces: dict[str, CallTrace] = {}
        self._memo: dict[str, BackendResult] = {}
        self._lock = asyncio.Lock()
        self.api_calls = 0
        self.memo_hits = 0
        self.model_versions: Counter[str] = Counter()

    @property
    def model_version(self) -> str:
        return getattr(self.inner, "model_version", self.name)

    @staticmethod
    def _memo_key(request: DecisionRequest) -> str:
        payload = "\x00".join([request.redacted_excerpt, ",".join(request.advisory_topics)])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    async def decide(self, request: DecisionRequest) -> BackendResult:
        key = self._memo_key(request)
        async with self._lock:
            cached = self._memo.get(key)
        if cached is not None:
            self.memo_hits += 1
            self.raw_by_request[request.request_id] = cached
            self.traces[request.request_id] = CallTrace(attempts=0, memo_hit=True)
            return cached

        result, trace = await self._decide_with_retries(request)
        async with self._lock:
            # Last writer wins on a genuine duplicate; the answers are the same
            # call either way, and the memo exists to avoid paying twice.
            self._memo[key] = result
        self.raw_by_request[request.request_id] = result
        self.traces[request.request_id] = trace
        if not result.degraded:
            self.model_versions[result.model_version] += 1
        return result

    async def _decide_with_retries(self, request: DecisionRequest) -> tuple[BackendResult, CallTrace]:
        trace = CallTrace()
        result: BackendResult | None = None
        for attempt in range(self.max_attempts):
            trace.attempts = attempt + 1
            self.api_calls += 1
            result = await self.inner.decide(request)
            if not result.degraded:
                return result, trace
            trace.degrade_reasons.append(result.degrade_reason or "unknown")
            if attempt + 1 < self.max_attempts:
                # Exponential backoff plus jitter. JevBackend already retried the
                # retryable HTTP statuses internally, so reaching here means the
                # whole internal budget was spent; waiting longer is the only
                # lever left.
                delay = self.retry_base_seconds * (2**attempt) + self._rng.uniform(0.0, 0.35)
                await asyncio.sleep(min(delay, 8.0))
        assert result is not None
        return result, trace

    async def aclose(self) -> None:
        await self.inner.aclose()

    def stats(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "api_calls": self.api_calls,
            "logical_requests": len(self.raw_by_request),
            "memo_hits": self.memo_hits,
            "model_versions": dict(self.model_versions),
            "model_version_final": self.model_version,
            "retry_attempts_total": sum(t.attempts for t in self.traces.values()),
            "requests_needing_retry": sum(1 for t in self.traces.values() if t.attempts > 1),
            "degraded_final": sum(1 for r in self.raw_by_request.values() if r.degraded),
        }


# --------------------------------------------------------------------------- #
# Router construction per configuration
# --------------------------------------------------------------------------- #
def _policy_for(config: str, base: Policy) -> Policy:
    """Derive one configuration's policy from the shipped default.

    ``with_overrides`` is used rather than editing YAML text so the derivation is
    exact and reviewable: the same loader, validator, and rule compiler produce
    every variant, and a typo in an override fails at load time.
    """
    if config == CONFIG_DEFAULT:
        return base
    if config == CONFIG_NO_ESCALATION:
        return base.with_overrides(on_uncertain=dict(NO_UNCERTAIN))
    if config == CONFIG_STILL_CLASSIFY:
        gate_cfg = {**dict(base.gate), "on_force_local": "still_classify"}
        return base.with_overrides(gate=gate_cfg)
    raise ValueError(f"unknown eval configuration {config!r}")


def _build_router(policy: Policy, backend: RecordingBackend, *, cache: Any = None) -> Router:
    """A router wired for evaluation.

    ``NullSink`` because the harness persists a strictly richer JSONL of its own;
    letting the router also append to ``./decision-log/decisions.jsonl`` would
    write three copies of every prompt into the repo's working tree. ``NullCache``
    for the measured passes so every latency figure is a real backend latency and
    every row gets its own recorded raw answer; the cache is exercised separately
    by :func:`measure_cache`.
    """
    return Router(
        policy,
        backend,
        cache=cache if cache is not None else NullCache(),
        sink=NullSink(),
        excerpt_mode="hash",
    )


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #
def _row_dict(
    row: DatasetRow,
    config: str,
    backend_name: str,
    *,
    request_id: str,
    decision: Any,
    record: Any,
    raw: BackendResult | None,
    trace: CallTrace | None,
    wall_ms: float,
    error: str | None,
) -> dict[str, Any]:
    """Flatten one evaluation into a JSONL row.

    Contains three views of the same decision on purpose:

    ``raw_answers``
        what the backend said before the gate floor and the confidence bumps.
        This is the view calibration and component-label accuracy are computed
        on, because it is the model's judgement rather than the policy's.
    ``decision.answers`` (inside ``record``)
        the *effective* answers the policy reasoned about. Comparing the two is
        how the report shows what escalation actually changed.
    ``record``
        the full :class:`DecisionRecord`, i.e. the same object the production
        decision log would contain. ``questions_sent`` is stripped -- it is
        identical boilerplate on every row and would multiply the file size by
        four; its SHA-256 is recorded once in the run metadata instead.

    Nothing here contains the prompt text. Only its length and hash survive,
    matching the router's own ``excerpt_mode: hash`` default, so the results file
    can be committed.
    """
    record_dict = record.to_dict() if record is not None else None
    if record_dict is not None:
        record_dict.pop("questions_sent", None)
    gate_dict = decision.gate.to_dict() if decision is not None else None
    return {
        "id": row.id,
        "config": config,
        "backend": backend_name,
        "request_id": request_id,
        "timestamp": (record.timestamp if record is not None else datetime.now(timezone.utc).isoformat()),
        "text_chars": row.chars,
        "text_words": row.words,
        "difficulty": row.difficulty,
        "labels": dict(row.labels),
        "expected_tier": row.expected_tier,
        "classified": bool(decision is not None and decision.backend != "gate"),
        "error": error,
        "attempts": trace.attempts if trace else 0,
        "memo_hit": bool(trace.memo_hit) if trace else False,
        "degraded": bool(decision.degraded) if decision is not None else False,
        "degrade_reasons": list(trace.degrade_reasons) if trace else [],
        "tier": decision.tier if decision is not None else None,
        "model": decision.model if decision is not None else None,
        "rule_id": decision.rule_id if decision is not None else None,
        "reason": decision.reason if decision is not None else None,
        "escalated": list(decision.escalated) if decision is not None else [],
        "effective": (
            {
                "complexity": decision.effective_complexity,
                "sensitivity": decision.effective_sensitivity,
                "pii": decision.answers.pii.value,
                "pii_present": decision.answers.pii.value >= 0.5,
            }
            if decision is not None
            else None
        ),
        "raw_answers": raw.answers.to_dict() if raw is not None else None,
        "raw_model_version": raw.model_version if raw is not None else None,
        "gate": gate_dict,
        "backend_latency_ms": (record.backend_latency_ms if record is not None else 0.0),
        "total_latency_ms": (record.total_latency_ms if record is not None else 0.0),
        "wall_ms": round(wall_ms, 3),
        "cached": bool(decision.cached) if decision is not None else False,
        "record": record_dict,
    }


async def run_config(
    rows: Sequence[DatasetRow],
    config: str,
    backend: RecordingBackend,
    base_policy: Policy,
    *,
    concurrency: int = 12,
    on_progress: Any = None,
) -> list[dict[str, Any]]:
    """Run every dataset row through the real router under one policy config.

    Bounded concurrency rather than serial: at ~0.8 s per Jev decision, 223
    prompts serially is three minutes of wall clock for no benefit, and the API
    tolerates parallelism. The semaphore is what keeps "parallel" from becoming
    "223 simultaneous requests", which would trip rate limits and turn a latency
    measurement into a queueing measurement.

    Exceptions are caught per row and recorded as ``error`` rows. A single bad
    row must not abort a run that has already spent two minutes of API calls, and
    a dropped row must never pass silently.
    """
    policy = _policy_for(config, base_policy)
    router = _build_router(policy, backend)
    sem = asyncio.Semaphore(max(1, concurrency))
    done = 0

    async def one(row: DatasetRow) -> dict[str, Any]:
        nonlocal done
        request_id = f"eval-{config}-{row.id}"
        async with sem:
            started = time.perf_counter()
            try:
                decision = await router.route_text(
                    row.text,
                    request_id=request_id,
                    metadata=_eval_metadata(row, config, backend.name),
                )
                error: str | None = None
            except Exception as exc:  # recorded, never swallowed
                decision = None
                error = f"{type(exc).__name__}: {exc}"
            wall_ms = (time.perf_counter() - started) * 1000.0
        raw = backend.raw_by_request.get(request_id)
        trace = backend.traces.get(request_id)
        # Reconstruct the DecisionRecord the sink would have written. The sink is
        # Null here (see _build_router), so the record is rebuilt from the same
        # pieces to keep the JSONL a faithful stand-in for the production log.
        record = _rebuild_record(row, request_id, decision, raw, wall_ms)
        done += 1
        if on_progress is not None and (done % 25 == 0 or done == len(rows)):
            on_progress(done, len(rows), config, backend.name)
        return _row_dict(
            row,
            config,
            backend.name,
            request_id=request_id,
            decision=decision,
            record=record,
            raw=raw,
            trace=trace,
            wall_ms=wall_ms,
            error=error,
        )

    results = await asyncio.gather(*(one(r) for r in rows))
    # The router is deliberately not closed here: aclose() would close the shared
    # backend, and later configurations still need it.
    return list(results)


def _rebuild_record(
    row: DatasetRow,
    request_id: str,
    decision: Any,
    raw: BackendResult | None,
    wall_ms: float,
) -> Any:
    """Build the :class:`DecisionRecord` for a routed row.

    Only the fields an evaluator can legitimately know are filled in: the
    decision, the excerpt hash, and the latencies. ``features`` are recomputed
    from the raw text with the same helper the router uses, so a consumer of the
    JSONL sees the identical shape it would see from the production sink.
    """
    if decision is None:
        return None
    from jev_route.prompts import compute_features, excerpt_from_text, hash_text
    from jev_route.schema import DecisionRecord

    raw_excerpt = excerpt_from_text(row.text)
    features = compute_features(
        raw_excerpt,
        messages=None,
        gate_detectors=decision.gate.detectors(),
        n_gate_findings=len(decision.gate.findings),
        gate_force_local=decision.gate.force_local,
    )
    return DecisionRecord(
        request_id=request_id,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        decision=decision,
        features=features,
        excerpt_hash=hash_text(raw_excerpt, salt=""),
        backend_latency_ms=round(raw.latency_ms if raw is not None else 0.0, 3),
        total_latency_ms=round(wall_ms, 3),
        questions_sent={},
        excerpt=None,
        metadata=_eval_metadata(row, "record", ""),
        requested_model=None,
        shadow=None,
    )


async def measure_cache(
    rows: Sequence[DatasetRow],
    backend: RecordingBackend,
    base_policy: Policy,
    *,
    concurrency: int = 12,
    ttl_seconds: float = 900.0,
) -> dict[str, Any]:
    """Exercise the decision cache and report what it actually saves.

    Two passes over the same prompts with one warm ``InMemoryTTLCache``. Because
    :class:`RecordingBackend` has already memoized every answer, neither pass
    touches the network: what is being measured is the cache's hit rate and the
    router-side latency with and without it, which is the part an operator feels.
    The backend latency of a *cold* call is taken from the measured run, not from
    this synthetic replay, and the report says so.
    """
    policy = _policy_for(CONFIG_DEFAULT, base_policy)
    cache = InMemoryTTLCache(ttl_seconds=ttl_seconds, max_entries=max(1024, len(rows) * 2))

    async def one(row: DatasetRow, router: Router, sem: asyncio.Semaphore, tag: str) -> float:
        """Route one row and return the router-side latency. Args are explicit so
        the coroutine cannot close over a loop variable from the pass below."""
        async with sem:
            decision = await router.route_text(
                row.text,
                request_id=f"cache-{tag}-{row.id}",
                metadata=_eval_metadata(row, f"cache_pass_{tag}", backend.name),
            )
        return decision.latency_ms

    passes: list[dict[str, Any]] = []
    for pass_index in (1, 2):
        router = _build_router(policy, backend, cache=cache)
        sem = asyncio.Semaphore(max(1, concurrency))
        latencies = await asyncio.gather(*(one(r, router, sem, f"p{pass_index}") for r in rows))
        stats = cache.stats()
        passes.append(
            {
                "pass": pass_index,
                "cache_stats": dict(stats),
                "latency": cost_model.latency_summary(list(latencies), percentiles=(50.0, 95.0, 99.0)),
            }
        )
    warm = passes[1]["cache_stats"]
    hits = int(warm.get("hits", 0))
    misses = int(warm.get("misses", 0))
    return {
        "passes": passes,
        "warm_pass_hit_rate": round(hits / (hits + misses), 6) if (hits + misses) else None,
        "warm_pass_hits": hits,
        "warm_pass_misses": misses,
        "note": (
            "both passes served from RecordingBackend's memo, so no API calls were made; "
            "cold backend latency comes from the measured run, not from this replay"
        ),
    }


# --------------------------------------------------------------------------- #
# Metrics: classification
# --------------------------------------------------------------------------- #
def confusion_matrix(pairs: Iterable[tuple[str, str]], labels: Sequence[str]) -> dict[str, dict[str, int]]:
    """``expected -> predicted -> count`` over every label in ``labels``.

    Built as a dense matrix including zero cells, so a class the model never
    predicted still appears as a row of zeros. A sparse matrix silently hides the
    most informative failure a router can have: a level it never routes to at all.
    """
    matrix = {gold: dict.fromkeys(labels, 0) for gold in labels}
    other: dict[str, int] = {}
    for gold, predicted in pairs:
        if gold not in matrix:
            other[f"unknown_gold:{gold}"] = other.get(f"unknown_gold:{gold}", 0) + 1
            continue
        if predicted not in matrix[gold]:
            other[f"unknown_pred:{predicted}"] = other.get(f"unknown_pred:{predicted}", 0) + 1
            continue
        matrix[gold][predicted] += 1
    if other:
        matrix["_unlabelled"] = other  # type: ignore[assignment]
    return matrix


def per_class_metrics(pairs: Iterable[tuple[str, str]], labels: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Per-class precision/recall/F1/support plus macro-F1.

    Macro-F1 is the headline rather than accuracy because the dataset is
    deliberately unbalanced across levels (``frontier`` is 12% of rows,
    ``standard`` is 36%). A router that answered ``standard`` to everything would
    score 36% accuracy and 0% macro-F1, and only the second number tells you that
    it learned nothing.
    """
    pairs = list(pairs)
    total = len(pairs)
    out: dict[str, dict[str, Any]] = {}
    f1s: list[float] = []
    for label in labels:
        tp = sum(1 for g, p in pairs if g == label and p == label)
        fp = sum(1 for g, p in pairs if g != label and p == label)
        fn = sum(1 for g, p in pairs if g == label and p != label)
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1s.append(f1)
        out[label] = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            #: Share of this class's rows the model got right. Reported per class
            #: because macro-F1 alone does not say *which* level is weak.
            "accuracy": round(tp / support, 6) if support else None,
        }
    out["_macro"] = {
        "macro_f1": round(sum(f1s) / len(f1s), 6) if f1s else 0.0,
        "accuracy": round(sum(1 for g, p in pairs if g == p) / total, 6) if total else 0.0,
        "n": total,
    }
    return out


def binary_metrics(pairs: Iterable[tuple[bool, bool]]) -> dict[str, Any]:
    """Precision/recall/F1 for a yes/no head, with the full 2x2 counts.

    For the PII head a false negative is a data-egress event and a false positive
    is a wasted local route, so both are reported as named counts rather than
    being collapsed into an error rate.
    """
    pairs = list(pairs)
    tp = sum(1 for g, p in pairs if g and p)
    tn = sum(1 for g, p in pairs if not g and not p)
    fp = sum(1 for g, p in pairs if not g and p)
    fn = sum(1 for g, p in pairs if g and not p)
    n = len(pairs)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round((tp + tn) / n, 6) if n else 0.0,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round((2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0, 6),
        #: The number that matters for a residency router: PII present and the
        #: model said no.
        "missed_pii": fn,
        #: PII absent and the model said yes: costs a local route, leaks nothing.
        "false_pii": fp,
    }


# --------------------------------------------------------------------------- #
# Metrics: tier routing
# --------------------------------------------------------------------------- #
def _is_scoreable(row: Mapping[str, Any]) -> bool:
    """A row counts toward accuracy only if the router actually produced a tier."""
    return row.get("error") is None and row.get("tier") in TIERS


def tier_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Tier accuracy, confusion, and the unsafe/expensive split for one run.

    Rows that errored or degraded are excluded from the denominator and counted
    out loud in ``excluded``. They are *not* scored as wrong and *not* dropped
    silently: a degraded backend fails closed to ``local``, which on this dataset
    would score as a correct routing decision for 39% of rows and inflate accuracy
    by accident. That is the specific lie this function refuses to tell.
    """
    excluded = {
        "error": sum(1 for r in rows if r.get("error") is not None),
        "degraded": sum(1 for r in rows if r.get("degraded") and r.get("error") is None),
        "missing_tier": sum(1 for r in rows if r.get("error") is None and r.get("tier") not in TIERS),
    }
    scoreable = [r for r in rows if _is_scoreable(r) and not r.get("degraded")]
    pairs = [(str(r["expected_tier"]), str(r["tier"])) for r in scoreable]
    kinds = Counter(cost_model.classify_error(g, p) for g, p in pairs)

    per_expected: dict[str, dict[str, Any]] = {}
    for tier in TIERS:
        subset = [(g, p) for g, p in pairs if g == tier]
        per_expected[tier] = {
            "n": len(subset),
            "correct": sum(1 for g, p in subset if g == p),
            "accuracy": round(sum(1 for g, p in subset if g == p) / len(subset), 6) if subset else None,
            "unsafe": sum(1 for g, p in subset if cost_model.classify_error(g, p) == "unsafe"),
            "expensive": sum(1 for g, p in subset if cost_model.classify_error(g, p) == "expensive"),
            "overspend": sum(1 for g, p in subset if cost_model.classify_error(g, p) == "overspend"),
            "underpowered": sum(1 for g, p in subset if cost_model.classify_error(g, p) == "underpowered"),
        }

    n = len(pairs)
    #: ``unsafe`` is the headline. It counts rows labelled ``local`` -- data that
    #: must not leave the building -- that were routed to a cloud tier.
    unsafe = kinds.get("unsafe", 0)
    by_difficulty: dict[str, dict[str, Any]] = {}
    for difficulty in sorted({str(r.get("difficulty", "")) for r in scoreable}):
        subset = [r for r in scoreable if str(r.get("difficulty", "")) == difficulty]
        hits = sum(1 for r in subset if r["expected_tier"] == r["tier"])
        by_difficulty[difficulty] = {
            "n": len(subset),
            "accuracy": round(hits / len(subset), 6) if subset else None,
            "unsafe": sum(
                1 for r in subset if cost_model.classify_error(str(r["expected_tier"]), str(r["tier"])) == "unsafe"
            ),
        }

    return {
        "n_rows": len(rows),
        "n_scored": n,
        "excluded": excluded,
        "accuracy": round(sum(1 for g, p in pairs if g == p) / n, 6) if n else None,
        "confusion": confusion_matrix(pairs, list(TIERS)),
        "per_class": per_class_metrics(pairs, list(TIERS)),
        "error_kinds": {k: kinds.get(k, 0) for k in cost_model.ERROR_KINDS},
        #: Rates are given against total scored rows so they are comparable
        #: between runs of different sizes.
        "unsafe_rate": round(unsafe / n, 6) if n else None,
        "unsafe_errors": unsafe,
        "expensive_errors": kinds.get("expensive", 0),
        "per_expected_tier": per_expected,
        "by_difficulty": by_difficulty,
        "tier_distribution": dict(Counter(str(r["tier"]) for r in scoreable)),
        "escalated_rows": sum(1 for r in scoreable if r.get("escalated")),
        "gate_forced_rows": sum(1 for r in scoreable if (r.get("gate") or {}).get("force_local")),
        "gate_blocked_rows": sum(1 for r in scoreable if (r.get("gate") or {}).get("blocks_backend")),
        "classified_rows": sum(1 for r in scoreable if r.get("classified")),
        "wrong_rows": [
            {
                "id": r["id"],
                "expected": r["expected_tier"],
                "predicted": r["tier"],
                "kind": cost_model.classify_error(str(r["expected_tier"]), str(r["tier"])),
                "rule_id": r.get("rule_id"),
                "labels": r.get("labels"),
                "effective": r.get("effective"),
                "escalated": r.get("escalated"),
                "difficulty": r.get("difficulty"),
            }
            for r in scoreable
            if r["expected_tier"] != r["tier"]
        ],
    }


# --------------------------------------------------------------------------- #
# Metrics: component labels and calibration samples
# --------------------------------------------------------------------------- #
def _raw_choice(row: Mapping[str, Any], head: str) -> tuple[str, Mapping[str, float], float] | None:
    """The backend's own answer for one head, or ``None`` when it never answered."""
    raw = row.get("raw_answers")
    if not raw:
        return None
    block = raw.get(head)
    if not block:
        return None
    probs = {str(k): float(v) for k, v in (block.get("probabilities") or {}).items()}
    confidence = block.get("confidence")
    return str(block.get("choice")), probs, (float(confidence) if confidence is not None else 0.0)


def component_metrics(rows: Sequence[Mapping[str, Any]], head: str, labels: Sequence[str]) -> dict[str, Any]:
    """Compare one backend head against the dataset's label for it.

    Uses the *raw* backend argmax (pre-gate-merge, pre-escalation) because that
    is the model's judgement. Scoring the effective label here would credit the
    gate and the policy for the backend's accuracy and debit them for its
    mistakes, and the resulting number would describe neither.

    Rows the backend never saw (gate blocked the call) are excluded and counted in
    ``n_unclassified``. That exclusion is a real selection effect -- the blocked
    rows are the ones with checksum-valid identifiers in them -- so it is reported
    rather than buried.
    """
    pairs: list[tuple[str, str]] = []
    skipped = 0
    argmax_mismatch = 0
    for row in rows:
        got = _raw_choice(row, head)
        gold = str((row.get("labels") or {}).get(head, ""))
        if got is None:
            skipped += 1
            continue
        choice, probs, _conf = got
        if probs and choice != max(probs.items(), key=lambda kv: kv[1])[0]:
            argmax_mismatch += 1
        pairs.append((gold, choice))
    result: dict[str, Any] = {
        "head": head,
        "labels": list(labels),
        "n_scored": len(pairs),
        "n_unclassified": skipped,
        "argmax_mismatch_rows": argmax_mismatch,
        "confusion": confusion_matrix(pairs, list(labels)),
        "per_class": per_class_metrics(pairs, list(labels)),
    }
    result["macro_f1"] = result["per_class"]["_macro"]["macro_f1"]
    result["accuracy"] = result["per_class"]["_macro"]["accuracy"]
    return result


def pii_metrics(rows: Sequence[Mapping[str, Any]], *, threshold: float = 0.5) -> dict[str, Any]:
    """Binary PII metrics on the raw noul, plus the policy's effective decision.

    Two views because they answer different questions. ``raw`` asks whether the
    model can tell PII from non-PII. ``effective`` asks whether the *router*
    treated PII as present after the gate floor and the uncertain-band rule -- the
    thing that actually decides whether text leaves the building.
    """
    raw_pairs: list[tuple[bool, bool]] = []
    eff_pairs: list[tuple[bool, bool]] = []
    skipped = 0
    for row in rows:
        gold = bool((row.get("labels") or {}).get("pii", False))
        raw = row.get("raw_answers")
        if raw and raw.get("pii") is not None:
            raw_pairs.append((gold, float(raw["pii"]["noul"]) >= threshold))
        else:
            skipped += 1
        effective = row.get("effective")
        if effective is not None:
            eff_pairs.append((gold, bool(effective.get("pii_present"))))
    return {
        "threshold": threshold,
        "raw": binary_metrics(raw_pairs),
        "effective": binary_metrics(eff_pairs),
        "n_unclassified": skipped,
        "note": (
            "raw = backend noul thresholded at 0.5; effective = after the local gate floor and the "
            "policy's uncertain-band rule. The effective view is what decides data egress."
        ),
    }


def calibration_samples(
    rows: Sequence[Mapping[str, Any]], head: str, labels: Sequence[str]
) -> list[calibration.CalibrationSample]:
    """Build calibration samples for one choice head from raw backend answers."""
    out: list[calibration.CalibrationSample] = []
    for row in rows:
        got = _raw_choice(row, head)
        gold = str((row.get("labels") or {}).get(head, ""))
        if got is None or gold not in labels:
            continue
        choice, probs, _reported = got
        block = (row.get("raw_answers") or {}).get(head) or {}
        out.append(
            calibration.CalibrationSample(
                gold=gold,
                prediction=choice,
                probabilities={k: float(probs.get(k, 0.0)) for k in labels},
                confidence=(float(block["confidence"]) if block.get("confidence_reported") else None),
                row_id=str(row.get("id", "")),
            )
        )
    return out


def calibration_block(rows: Sequence[Mapping[str, Any]], *, n_bins: int = calibration.DEFAULT_BINS) -> dict[str, Any]:
    """Calibration for all four heads, plus the metadata needed to interpret it.

    ``populated_bins`` is carried into the report on purpose: with 186-206 rows
    and a backend that reports coarse confidences, an ECE computed over 15 bins
    can rest on very few of them, and printing "ECE = 0.041" without saying so
    would overstate the precision of the measurement.
    """
    reports: dict[str, Any] = {}
    for head, ladder in (
        ("complexity", COMPLEXITY_LEVELS),
        ("sensitivity", SENSITIVITY_LEVELS),
        ("domain", DOMAINS),
    ):
        samples = calibration_samples(rows, head, ladder)
        report = calibration.calibrate_head(head, samples, list(ladder), n_bins=n_bins)
        reports[head] = report.to_dict()
        reports[head]["_report"] = report  # kept for rendering, stripped before JSON

    pii_probs: list[float] = []
    pii_gold: list[bool] = []
    for row in rows:
        raw = row.get("raw_answers")
        if raw and raw.get("pii") is not None:
            pii_probs.append(float(raw["pii"]["noul"]))
            pii_gold.append(bool((row.get("labels") or {}).get("pii", False)))
    pii_report = calibration.calibrate_binary_head("pii", pii_probs, pii_gold, n_bins=n_bins)
    reports["pii"] = pii_report.to_dict()
    reports["pii"]["_report"] = pii_report
    return reports


# --------------------------------------------------------------------------- #
# Strategies that need no extra backend pass
# --------------------------------------------------------------------------- #
def gate_only_tier(row: Mapping[str, Any]) -> str:
    """The "why not just regex?" baseline: local hard gate, then always cheap.

    Derived from the gate verdict the router already recorded, so it is the real
    gate on the real text with no model involved anywhere. It forces ``local``
    exactly when the gate forced local (a structured identifier, a credential, or
    a personal identifier matched) and sends everything else to the cheapest tier.
    """
    gate = row.get("gate") or {}
    if gate.get("force_local") or gate.get("blocks_backend"):
        return "local"
    return "cheap"


def _cost_rows(rows: Sequence[Mapping[str, Any]], tier_of: Any) -> list[cost_model.CostRow]:
    """Reduce scored rows to :class:`cost_model.CostRow`, applying one tier function."""
    out: list[cost_model.CostRow] = []
    for row in rows:
        out.append(
            cost_model.CostRow(
                id=str(row["id"]),
                chars=int(row.get("text_chars") or 0),
                words=int(row.get("text_words") or 0),
                tier=tier_of(row),
                expected_tier=str(row.get("expected_tier") or ""),
            )
        )
    return out


def cost_block(rows: Sequence[Mapping[str, Any]], policy: Policy) -> dict[str, Any]:
    """Price four strategies over the same prompts and compare them.

    ``jev_route`` and ``gate_only`` both read tiers off the *same* measured rows;
    the difference is whether a model classified the request. That is the honest
    comparison, because it holds the prompt set, the token proxy, and the price
    table fixed and varies only the thing under test.
    """
    scored = [r for r in rows if r.get("error") is None and not r.get("degraded")]
    model_for_tier = {tier: models[0] for tier, models in policy.tiers.items()}
    cost_rows: dict[str, list[cost_model.CostRow]] = {
        "jev_route": _cost_rows(scored, lambda r: str(r["tier"])),
        "gate_only": _cost_rows(scored, gate_only_tier),
        "always_strong": _cost_rows(scored, lambda _r: "strong"),
        "always_cheap": _cost_rows(scored, lambda _r: "cheap"),
    }
    costs = {name: cost_model.simulate(name, cost_rows[name], model_for_tier=model_for_tier) for name in cost_rows}
    return {
        "n_scored": len(scored),
        "model_for_tier": model_for_tier,
        "strategies": {name: cost.to_dict() for name, cost in costs.items()},
        "vs_always_strong": cost_model.compare_strategies(costs, baseline="always_strong"),
        "vs_gate_only": cost_model.compare_strategies(costs, baseline="gate_only"),
        #: The negative control, in both directions. ``jev_route_only_correct`` is
        #: the set of prompts a regex gate gets wrong and the router gets right --
        #: the entire justification for paying a model to classify.
        "negative_control": cost_model.negative_control(
            cost_rows["jev_route"], cost_rows["gate_only"], name_a="jev_route", name_b="gate_only"
        ),
        "assumptions": cost_model.simulate(
            "always_cheap", cost_rows["always_cheap"], model_for_tier=model_for_tier
        ).assumptions,
    }


def latency_block(rows: Sequence[Mapping[str, Any]], backend_name: str) -> dict[str, Any]:
    """Measured routing overhead.

    Only rows where the backend was genuinely called (not gate-skipped, not
    served from the memo, not degraded) contribute to ``backend_latency``, because
    those are the only rows where the number is a real network round trip. Gate
    skips are reported separately: their near-zero cost is a feature of the
    design, and averaging it into the backend latency would understate the tax on
    everything else.
    """
    real_calls = [
        r
        for r in rows
        if r.get("classified") and not r.get("memo_hit") and not r.get("degraded") and r.get("error") is None
    ]
    backend_ms = [float(r.get("backend_latency_ms") or 0.0) for r in real_calls]
    total_ms = [float(r.get("total_latency_ms") or 0.0) for r in rows if r.get("error") is None]
    skipped = [r for r in rows if not r.get("classified") and r.get("error") is None]
    skipped_ms = [float(r.get("total_latency_ms") or 0.0) for r in skipped]

    backend_summary = cost_model.latency_summary(backend_ms)
    total_summary = cost_model.latency_summary(total_ms)
    p50 = backend_summary.get("p50_ms")
    p95 = backend_summary.get("p95_ms")
    return {
        "backend": backend_name,
        "n_real_backend_calls": len(real_calls),
        "n_gate_skipped": len(skipped),
        "backend_latency_ms": backend_summary,
        "router_total_latency_ms": total_summary,
        "gate_skipped_latency_ms": cost_model.latency_summary(skipped_ms, percentiles=(50.0, 95.0)),
        #: Overhead expressed against an assumed completion latency. The
        #: assumption is printed next to every one of these numbers.
        "representative_completion_ms": cost_model.REPRESENTATIVE_COMPLETION_MS,
        "overhead_pct_of_completion": {
            "p50": cost_model.overhead_as_pct(p50),
            "p95": cost_model.overhead_as_pct(p95),
            "mean": cost_model.overhead_as_pct(backend_summary.get("mean_ms")),
        },
        "note": (
            "backend_latency is the Jev/mock decision call only; router_total adds the local gate, "
            "redaction, feature extraction, merge, escalation and policy evaluation. "
            "representative_completion_ms is an assumption, not a measurement."
        ),
    }


# --------------------------------------------------------------------------- #
# Per-backend analysis
# --------------------------------------------------------------------------- #
def analyse_backend(
    backend_name: str,
    config_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    policy: Policy,
    backend_stats: Mapping[str, Any] | None = None,
    cache_report: Mapping[str, Any] | None = None,
    run_started: str = "",
    run_finished: str = "",
    n_bins: int = calibration.DEFAULT_BINS,
) -> dict[str, Any]:
    """Every number for one backend, in one JSON-serialisable dict.

    Which configuration supplies which measurement is a deliberate choice and is
    recorded in ``provenance`` so no reader has to guess:

    * tier accuracy and cost come from ``default`` -- the shipped policy;
    * the escalation A/B comes from ``default`` vs ``no_escalation``;
    * component labels and calibration come from ``still_classify``, because it is
      the configuration that lets the backend answer the largest number of rows
      (only gate-*blocked* rows stay unclassified there, versus gate-*forced* too).
    """
    default_rows = list(config_rows.get(CONFIG_DEFAULT) or [])
    noesc_rows = list(config_rows.get(CONFIG_NO_ESCALATION) or [])
    still_rows = list(config_rows.get(CONFIG_STILL_CLASSIFY) or default_rows)
    label_source = CONFIG_STILL_CLASSIFY if config_rows.get(CONFIG_STILL_CLASSIFY) else CONFIG_DEFAULT

    tier_default = tier_metrics(default_rows)
    tier_noesc = tier_metrics(noesc_rows) if noesc_rows else None

    def _delta(a: Mapping[str, Any], b: Mapping[str, Any] | None) -> dict[str, Any]:
        if b is None:
            return {}
        return {
            "accuracy_delta": (
                None
                if a.get("accuracy") is None or b.get("accuracy") is None
                else round(float(a["accuracy"]) - float(b["accuracy"]), 6)
            ),
            "unsafe_delta": int(a.get("unsafe_errors", 0)) - int(b.get("unsafe_errors", 0)),
            "expensive_delta": int(a.get("expensive_errors", 0)) - int(b.get("expensive_errors", 0)),
        }

    escalation_ab = {
        "enabled": {
            "accuracy": tier_default.get("accuracy"),
            "unsafe_errors": tier_default.get("unsafe_errors"),
            "expensive_errors": tier_default.get("expensive_errors"),
            "error_kinds": tier_default.get("error_kinds"),
            "escalated_rows": tier_default.get("escalated_rows"),
            "tier_distribution": tier_default.get("tier_distribution"),
            "n_scored": tier_default.get("n_scored"),
        },
        "disabled": (
            None
            if tier_noesc is None
            else {
                "accuracy": tier_noesc.get("accuracy"),
                "unsafe_errors": tier_noesc.get("unsafe_errors"),
                "expensive_errors": tier_noesc.get("expensive_errors"),
                "error_kinds": tier_noesc.get("error_kinds"),
                "escalated_rows": tier_noesc.get("escalated_rows"),
                "tier_distribution": tier_noesc.get("tier_distribution"),
                "n_scored": tier_noesc.get("n_scored"),
            }
        ),
        "enabled_minus_disabled": _delta(tier_default, tier_noesc),
    }

    components = {
        "complexity": component_metrics(still_rows, "complexity", COMPLEXITY_LEVELS),
        "sensitivity": component_metrics(still_rows, "sensitivity", SENSITIVITY_LEVELS),
        "domain": component_metrics(still_rows, "domain", DOMAINS),
        "pii": pii_metrics(still_rows),
    }
    calibration_reports = calibration_block(still_rows, n_bins=n_bins)

    lat = latency_block(default_rows, backend_name)
    costs = cost_block(default_rows, policy)

    # Split the live report objects out before serialising: the dicts go into
    # summary.json, the objects go to the renderer, and nothing non-serialisable
    # is left hiding inside a dict that is about to be json.dump'd.
    live_reports = {head: data.pop("_report") for head, data in calibration_reports.items()}

    model_versions = sorted(
        {str(r.get("raw_model_version")) for r in default_rows + still_rows if r.get("raw_model_version")}
    )

    return {
        "backend": backend_name,
        "run_started": run_started,
        "run_finished": run_finished,
        "n_rows": len(default_rows),
        "model_versions_observed": model_versions,
        "backend_stats": dict(backend_stats or {}),
        "provenance": {
            "tier_accuracy": CONFIG_DEFAULT,
            "cost": CONFIG_DEFAULT,
            "latency": CONFIG_DEFAULT,
            "escalation_ab": [CONFIG_DEFAULT, CONFIG_NO_ESCALATION],
            "component_labels": label_source,
            "calibration": label_source,
        },
        "tier": tier_default,
        "escalation_ab": escalation_ab,
        "components": components,
        "calibration": dict(calibration_reports),
        "cost": costs,
        "latency": lat,
        "cache": dict(cache_report or {}),
        "_calibration_reports": live_reports,
    }


def mock_vs_jev_gap(summary: Mapping[str, Any]) -> dict[str, Any]:
    """What the cloud bootstrap buys, as a difference rather than an adjective.

    Computed only when both backends were run. The interesting direction is not
    guaranteed: the mock is a deterministic keyword-and-shape reader, and on a
    dataset whose sensitive rows are full of keywords it can look surprisingly
    competitive. If it does, that is a finding about the dataset, and this block
    is where it becomes visible instead of being averaged away.
    """
    backends = summary.get("backends") or {}
    if "jev" not in backends or "mock" not in backends:
        return {"available": False, "reason": "only one backend was run"}
    jev, mock = backends["jev"], backends["mock"]

    def _diff(path_j: Any, path_m: Any) -> Any:
        if path_j is None or path_m is None:
            return None
        return round(float(path_j) - float(path_m), 6)

    heads = {}
    for head in ("complexity", "sensitivity", "domain"):
        heads[head] = {
            "macro_f1_jev": jev["components"][head]["macro_f1"],
            "macro_f1_mock": mock["components"][head]["macro_f1"],
            "macro_f1_gap": _diff(jev["components"][head]["macro_f1"], mock["components"][head]["macro_f1"]),
            "accuracy_jev": jev["components"][head]["accuracy"],
            "accuracy_mock": mock["components"][head]["accuracy"],
            "ece_jev": jev["calibration"][head]["ece"],
            "ece_mock": mock["calibration"][head]["ece"],
        }
    return {
        "available": True,
        "tier_accuracy_jev": jev["tier"]["accuracy"],
        "tier_accuracy_mock": mock["tier"]["accuracy"],
        "tier_accuracy_gap": _diff(jev["tier"]["accuracy"], mock["tier"]["accuracy"]),
        "unsafe_errors_jev": jev["tier"]["unsafe_errors"],
        "unsafe_errors_mock": mock["tier"]["unsafe_errors"],
        "unsafe_errors_gap": int(jev["tier"]["unsafe_errors"]) - int(mock["tier"]["unsafe_errors"]),
        "pii_f1_jev": jev["components"]["pii"]["raw"]["f1"],
        "pii_f1_mock": mock["components"]["pii"]["raw"]["f1"],
        "heads": heads,
    }


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def results_paths(backend_name: str, results_dir: Path) -> dict[str, Path]:
    """One JSONL per configuration.

    The ``default`` configuration keeps the unprefixed name because it is the run
    the README quotes and the one a reader will look for first; the ablations are
    suffixed so they can never be mistaken for it.
    """
    base = f"{backend_name}_labeled_run"
    return {
        CONFIG_DEFAULT: results_dir / f"{base}.jsonl",
        CONFIG_NO_ESCALATION: results_dir / f"{base}.{CONFIG_NO_ESCALATION}.jsonl",
        CONFIG_STILL_CLASSIFY: results_dir / f"{base}.{CONFIG_STILL_CLASSIFY}.jsonl",
    }


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, separators=(",", ":"), default=str))
            fh.write("\n")
    return len(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: malformed JSON: {exc}") from exc
    return out


def _progress(done: int, total: int, config: str, backend_name: str) -> None:
    """Progress to stderr. Never the API key, never prompt text, never a label."""
    print(f"  [{backend_name}/{config}] {done}/{total}", file=sys.stderr, flush=True)


async def run_backend_eval(
    backend_name: str,
    dataset: Sequence[DatasetRow],
    policy: Policy,
    *,
    results_dir: Path,
    concurrency: int,
    max_attempts: int,
    reuse: bool,
    cache_replay: bool,
    timeout_seconds: float,
    breaker_threshold: int,
    quiet: bool,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Run (or reload) every configuration for one backend.

    ``reuse`` is the idempotence switch: the whole point of persisting the raw
    rows is that re-analysis must never require re-spending the API calls. When
    any of the three files is missing, reuse refuses rather than partially
    re-running, because a summary assembled from two configurations of one run
    and one configuration of another would be comparing different backends.
    """
    paths = results_paths(backend_name, results_dir)
    if reuse:
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"--reuse asked for but these results files are missing: {missing}. "
                "Run without --reuse once to produce them."
            )
        rows = {cfg: read_jsonl(path) for cfg, path in paths.items()}
        return rows, {"reused": True, "paths": {k: str(v) for k, v in paths.items()}}

    if backend_name == "jev":
        api_key = os.environ.get("TYPESAFE_API_KEY", "")
        if not api_key:
            raise SystemExit(
                "TYPESAFE_API_KEY is not set. Put it in .env or the environment, "
                "or run with --backend mock for the offline evaluation."
            )
        inner = JevBackend(
            timeout_seconds=timeout_seconds,
            # More retries than the library default: a 223-row sweep at
            # concurrency 12 will see the occasional 429, and giving up after two
            # attempts would turn a rate limit into a row of fabricated
            # fail-closed decisions.
            max_retries=3,
            # The library default of 5 consecutive failures is right for a proxy
            # serving live traffic and wrong for a benchmark: at concurrency 12 a
            # one-second rate-limit burst trips it, and every remaining row then
            # fails closed to `local` -- which on this dataset would score as
            # CORRECT on 39% of rows. The threshold is raised, not removed, so a
            # genuinely dead backend still trips and the run reports a wall of
            # degraded rows instead of a flattering accuracy number.
            breaker=CircuitBreaker(failure_threshold=breaker_threshold, recovery_seconds=3.0),
        )
    elif backend_name == "mock":
        inner = MockBackend()
    else:
        raise ValueError(f"unknown backend {backend_name!r}")

    recorder = RecordingBackend(inner, max_attempts=max_attempts)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    config_rows: dict[str, list[dict[str, Any]]] = {}
    try:
        for config in (CONFIG_DEFAULT, CONFIG_NO_ESCALATION, CONFIG_STILL_CLASSIFY):
            if not quiet:
                print(f"[{backend_name}] running configuration {config!r}", file=sys.stderr, flush=True)
            rows = await run_config(
                dataset,
                config,
                recorder,
                policy,
                concurrency=concurrency,
                on_progress=None if quiet else _progress,
            )
            written = write_jsonl(paths[config], rows)
            config_rows[config] = rows
            if not quiet:
                print(
                    f"[{backend_name}] {config}: wrote {written} rows -> {paths[config].name}",
                    file=sys.stderr,
                    flush=True,
                )
        cache_report = await measure_cache(dataset, recorder, policy, concurrency=concurrency) if cache_replay else None
    finally:
        await recorder.aclose()
    finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return config_rows, {
        "reused": False,
        "paths": {k: str(v) for k, v in paths.items()},
        "backend_stats": recorder.stats(),
        "cache": cache_report,
        "run_started": started,
        "run_finished": finished,
        "concurrency": concurrency,
        "max_attempts": max_attempts,
        #: Fingerprint of the exact question set sent to Jev, so a reader can
        #: tell two runs apart even though the questions themselves are stripped
        #: from every row to keep the JSONL small.
        "questions_sha256": _questions_fingerprint(recorder),
    }


def _questions_fingerprint(recorder: RecordingBackend) -> str | None:
    """SHA-256 of one representative ``questions_sent`` payload.

    The questions are identical for every row, so storing one fingerprint is
    enough to prove which prompt set a run used without repeating ~5 KB of
    boilerplate 223 times.
    """
    for result in recorder.raw_by_request.values():
        if result.questions_sent:
            blob = json.dumps(result.questions_sent, sort_keys=True, separators=(",", ":"))
            return hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return None


# --------------------------------------------------------------------------- #
# Optional chart output
# --------------------------------------------------------------------------- #
def maybe_write_pngs(summary: Mapping[str, Any], results_dir: Path) -> list[str]:
    """Write reliability-diagram PNGs *if* matplotlib happens to be installed.

    Deliberately optional and deliberately last. The markdown tables in
    ``REPORT.md`` are the deliverable; a PNG is a convenience for whoever writes
    the docs. Requiring a plotting stack would mean the calibration numbers could
    not be reproduced in a bare venv, which is the opposite of what an
    open-source benchmark should do.
    """
    written: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return written
    for backend_name, analysis in (summary.get("backends") or {}).items():
        for head, data in (analysis.get("calibration") or {}).items():
            report = calibration.CalibrationReport.from_dict(data)
            bins = [b for b in report.bins if b.count > 0]
            if not bins:
                continue
            fig, ax = plt.subplots(figsize=(4.2, 3.4), dpi=140)
            centres = [(b.lo + b.hi) / 2 for b in bins]
            ax.bar(
                centres,
                [b.accuracy for b in bins],
                width=1.0 / max(1, report.n_bins),
                alpha=0.75,
                label="observed accuracy",
                color="#4c78a8",
            )
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
            ax.scatter(
                centres,
                [b.mean_confidence for b in bins],
                s=18,
                color="#e45756",
                zorder=3,
                label="mean predicted confidence",
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("predicted confidence")
            ax.set_ylabel("observed accuracy")
            ax.set_title(f"{backend_name} / {head}  (ECE={report.ece:.3f}, n={report.n})")
            ax.legend(loc="lower right", fontsize=7)
            fig.tight_layout()
            out = results_dir / f"reliability_{backend_name}_{head}.png"
            fig.savefig(out)
            plt.close(fig)
            written.append(str(out))
    return written


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _git_commit() -> str | None:
    """Best-effort commit id, so a report can be tied to the code that produced it."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _load_dotenv() -> bool:
    """Load ``.env`` if python-dotenv is available. Returns whether a key is set.

    python-dotenv is a dev dependency, not a core one, so its absence is not an
    error: the key may already be in the environment. The value itself is never
    printed, logged, or written to any output file -- only its presence is
    reported.
    """
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_file, override=False)
        except ImportError:
            pass
    return bool(os.environ.get("TYPESAFE_API_KEY"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_eval.py",
        description="Evaluate jev-route against evals/data/labeled_prompts.jsonl and write REPORT.md.",
    )
    parser.add_argument(
        "--backend",
        choices=("both", "jev", "mock"),
        default="both",
        help="which decision backend(s) to evaluate (default: both)",
    )
    parser.add_argument("--dataset", default=str(DATASET_PATH), help="labelled JSONL to evaluate")
    parser.add_argument("--policy", default=str(POLICY_PATH), help="policy file under test")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR), help="where to write outputs")
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="analyse the existing results files instead of calling any backend",
    )
    parser.add_argument("--concurrency", type=int, default=12, help="requests in flight (default 12)")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=4,
        help="attempts per prompt before it is recorded as an error row (default 4)",
    )
    parser.add_argument("--timeout-seconds", type=float, default=20.0, help="per-attempt Jev timeout (default 20)")
    parser.add_argument(
        "--breaker-threshold",
        type=int,
        default=50,
        help="consecutive failures before the circuit opens (default 50; see run_backend_eval)",
    )
    parser.add_argument("--bins", type=int, default=calibration.DEFAULT_BINS, help="ECE/reliability bins")
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N rows (debugging)")
    parser.add_argument("--no-cache-replay", action="store_true", help="skip the warm-cache replay measurement")
    parser.add_argument("--no-report", action="store_true", help="write JSON but not REPORT.md")
    parser.add_argument(
        "--png", action="store_true", help="also write PNG reliability diagrams if matplotlib is installed"
    )
    parser.add_argument("--quiet", action="store_true", help="no progress output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    key_present = _load_dotenv()
    dataset = load_dataset(args.dataset)
    if args.limit > 0:
        dataset = dataset[: args.limit]
    policy = Policy.from_file(args.policy)

    backends = ("jev", "mock") if args.backend == "both" else (args.backend,)
    if "jev" in backends and not key_present and not args.reuse:
        print(
            "warning: TYPESAFE_API_KEY is not set; the jev backend will fail to start",
            file=sys.stderr,
        )

    dataset_path = Path(args.dataset)
    policy_path = Path(args.policy)
    meta: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "path": str(dataset_path),
            "sha256_16": _file_sha256(dataset_path) if dataset_path.exists() else None,
            "n_rows": len(dataset),
        },
        "policy": {
            "path": str(policy_path),
            "sha256_16": _file_sha256(policy_path) if policy_path.exists() else None,
            "version": policy.version,
            "tiers": {k: list(v) for k, v in policy.tiers.items()},
            "on_uncertain": {
                "sensitivity_confidence_below": policy.uncertainty.sensitivity_confidence_below,
                "sensitivity_bump_levels": policy.uncertainty.sensitivity_bump_levels,
                "complexity_confidence_below": policy.uncertainty.complexity_confidence_below,
                "complexity_bump_levels": policy.uncertainty.complexity_bump_levels,
                "pii_uncertain_threshold": policy.uncertainty.pii_uncertain_threshold,
                "pii_uncertain_counts_as_present": policy.uncertainty.pii_uncertain_counts_as_present,
            },
            "gate_on_force_local": policy.gate.get("on_force_local"),
            "failure_mode": policy.failure.mode,
        },
        "git_commit": _git_commit(),
        "argv": list(argv or sys.argv[1:]),
        "typesafe_api_key_present": key_present,
        "calibration_bins": args.bins,
        "concurrency": args.concurrency,
        "reuse": args.reuse,
    }

    summary: dict[str, Any] = {"meta": meta, "backends": {}}
    render_inputs: dict[str, Any] = {}
    for backend_name in backends:
        config_rows, run_info = asyncio.run(
            run_backend_eval(
                backend_name,
                dataset,
                policy,
                results_dir=results_dir,
                concurrency=args.concurrency,
                max_attempts=args.max_attempts,
                reuse=args.reuse,
                cache_replay=not args.no_cache_replay,
                timeout_seconds=args.timeout_seconds,
                breaker_threshold=args.breaker_threshold,
                quiet=args.quiet,
            )
        )
        merged = {**meta, **{k: v for k, v in run_info.items() if k != "paths"}}
        analysis = analyse_backend(
            backend_name,
            config_rows,
            policy=policy,
            backend_stats=run_info.get("backend_stats"),
            cache_report=run_info.get("cache"),
            run_started=str(run_info.get("run_started", meta["generated_at"])),
            run_finished=str(run_info.get("run_finished", meta["generated_at"])),
            n_bins=args.bins,
        )
        analysis["run_info"] = {k: v for k, v in run_info.items() if k != "cache"}
        analysis["run_info"].update({k: v for k, v in merged.items() if k in ("questions_sha256", "reused")})
        render_inputs[backend_name] = analysis.pop("_calibration_reports")
        summary["backends"][backend_name] = analysis
        if not args.quiet:
            _print_headline(backend_name, analysis)

    summary["mock_vs_jev"] = mock_vs_jev_gap(summary)

    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=False, default=str), encoding="utf-8")

    reliability = {
        backend_name: dict(analysis.get("calibration") or {}) for backend_name, analysis in summary["backends"].items()
    }
    (results_dir / "reliability.json").write_text(json.dumps(reliability, indent=2, default=str), encoding="utf-8")

    if args.png:
        written = maybe_write_pngs(summary, results_dir)
        if not args.quiet:
            print(f"pngs: {written or 'none (matplotlib not installed)'}", file=sys.stderr)

    if not args.no_report:
        import report

        markdown = report.render(summary, calibration_reports=render_inputs)
        (results_dir / "REPORT.md").write_text(markdown, encoding="utf-8")

    if not args.quiet:
        print(f"\nwrote {summary_path}", file=sys.stderr)
        print(f"wrote {results_dir / 'reliability.json'}", file=sys.stderr)
        if not args.no_report:
            print(f"wrote {results_dir / 'REPORT.md'}", file=sys.stderr)
    return 0


def _print_headline(backend_name: str, analysis: Mapping[str, Any]) -> None:
    """One-line summary to stderr, so a long run always ends with a number."""
    tier = analysis.get("tier") or {}
    sens = (analysis.get("calibration") or {}).get("sensitivity") or {}
    cost = (analysis.get("cost") or {}).get("vs_always_strong") or {}
    lat = (analysis.get("latency") or {}).get("backend_latency_ms") or {}
    print(
        f"[{backend_name}] tier accuracy={tier.get('accuracy')} "
        f"UNSAFE={tier.get('unsafe_errors')} "
        f"sensitivity ECE={sens.get('ece')} "
        f"cost vs always-strong={cost.get('jev_route', {}).get('vs_baseline_pct')}% "
        f"jev latency p50/p95/p99="
        f"{lat.get('p50_ms')}/{lat.get('p95_ms')}/{lat.get('p99_ms')} ms "
        f"errored={(tier.get('excluded') or {}).get('error')}",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
