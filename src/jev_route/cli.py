"""The ``jev-route`` command line.

Every verb here is a thin wrapper over an importable function, so anything the CLI
can do is also available to a program. The CLI exists to make the lifecycle
legible: route a prompt, look at what was decided, inspect the log you are
accumulating, then export / train / evaluate / package / graduate it.

The verbs fall into three groups:

``route``, ``demo``, ``doctor``
    Day one. No API key required -- they default to MockBackend.
``log-stats``, ``explain``
    The log you are accumulating. This is the dataset growing.
``export``, ``train``, ``evaluate``, ``package``, ``graduate``
    The graduation pipeline: turn that log into a local model and cut over.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .policy import Policy
from .router import Router
from .schema import DecisionRecord

DEFAULT_POLICY = "policies/default.yaml"
PROG = "jev-route"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _find_policy(explicit: str | None) -> Path:
    """Locate a policy file, with a couple of forgiving fallbacks."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("JEV_ROUTE_POLICY")
    if env:
        candidates.append(Path(env))
    candidates += [Path(DEFAULT_POLICY), Path(__file__).resolve().parents[2] / DEFAULT_POLICY]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(c) for c in candidates)
    raise SystemExit(f"no policy file found (tried: {tried}); pass --policy")


def _load_router(args: argparse.Namespace, *, backend_override: str | None = None) -> tuple[Router, Policy, Path]:
    """Build a Router from a policy file, optionally forcing the backend."""
    from .backends import build_backend

    path = _find_policy(getattr(args, "policy", None))
    policy = Policy.from_file(path)
    backend_cfg = dict(policy.backend)
    if backend_override:
        backend_cfg["name"] = backend_override
    policy = policy.with_overrides(backend=backend_cfg)
    return Router(policy, build_backend(policy)), policy, path


def _print_decision(decision: Any, *, verbose: bool) -> None:
    """Human-readable one-liner, with the distributions when asked for."""
    answers = decision.answers
    print(f"  tier      : {decision.tier}")
    print(f"  model     : {decision.model}")
    print(f"  rule      : {decision.rule_id} -- {decision.reason}")
    print(f"  complexity: {answers.complexity.choice:9s} (confidence {answers.complexity.confidence:.2f})")
    print(f"  sensitivity: {answers.sensitivity.choice:12s} (confidence {answers.sensitivity.confidence:.2f})")
    print(f"  pii       : {answers.pii.value:.3f}")
    print(f"  domain    : {answers.domain.choice}")
    print(f"  backend   : {decision.backend} ({decision.backend_model_version})")
    print(f"  latency   : {decision.latency_ms:.1f} ms")
    if decision.degraded:
        print(f"  DEGRADED  : {decision.degrade_reason}")
    for note in decision.escalated:
        print(f"  escalated : {note}")
    if verbose:
        print("  distributions:")
        for name in ("complexity", "sensitivity", "domain"):
            dist = getattr(answers, name).probabilities
            print(f"    {name:12s} " + "  ".join(f"{k}={v:.3f}" for k, v in sorted(dist.items())))
        if decision.gate.fired:
            print(
                f"  gate      : floor={decision.gate.sensitivity_floor} "
                f"force_local={decision.gate.force_local} blocks_cloud={decision.gate.blocks_backend}"
            )
            for finding in decision.gate.findings:
                print(
                    f"              {finding.detector} x{finding.count} [{finding.category}] span={finding.span_hash}"
                )
            if decision.gate.advisory_topics:
                print(f"  topics    : {', '.join(decision.gate.advisory_topics)}")


def _load_records(args: argparse.Namespace) -> list[DecisionRecord]:
    """Read the decision log the policy points at."""
    from .logging_sink import iter_all_records, iter_records

    policy = Policy.from_file(_find_policy(getattr(args, "policy", None)))
    configured = str((policy.logging or {}).get("path") or "")
    path = Path(getattr(args, "log", None) or configured)
    if not path or not path.exists():
        raise SystemExit(
            f"decision log not found at {path or '<unset>'}. "
            "Route some traffic first (`jev-route route ...`), or pass --log."
        )
    if path.is_dir():
        return list(iter_all_records(path))
    return list(iter_records(path))


def _emit(data: Any, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(data, indent=2, default=str, sort_keys=True))
    else:
        print(data)


# --------------------------------------------------------------------------- #
# verbs
# --------------------------------------------------------------------------- #
def cmd_route(args: argparse.Namespace) -> int:
    """Route one prompt and print the decision."""
    text = args.text if args.text is not None else sys.stdin.read().strip()
    if not text:
        raise SystemExit("nothing to route: pass TEXT or pipe it on stdin")

    router, _policy, _path = _load_router(args, backend_override=args.backend)
    metadata = json.loads(args.metadata) if args.metadata else {}

    async def go() -> Any:
        try:
            return await router.route_text(text, metadata=metadata)
        finally:
            await router.aclose()

    decision = asyncio.run(go())
    if args.json:
        print(json.dumps(decision.to_dict(), indent=2, default=str, sort_keys=True))
    else:
        print(f"routed: {text[:70]!r}{'...' if len(text) > 70 else ''}")
        _print_decision(decision, verbose=args.verbose)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Run a canned set of prompts through the router and show the tiers.

    Defaults to MockBackend, so this works with no API key and no network. That is
    deliberate: the first thing anyone does after cloning is run the demo, and it
    must not fail because they have not signed up for anything yet.
    """
    # Labels describe the PROMPT, never the expected answer. Printing an expected
    # tier here would quietly turn the demo into a test that passes by assertion,
    # and the two backends legitimately disagree on some of these -- that is the
    # point of showing them side by side.
    cases = [
        ("greeting", "hi there, thanks!"),
        ("routine single-step coding task", "Write a Python function that reads a CSV and sums one column."),
        (
            "multi-step system design with a proof obligation",
            "Design a sharded write path with consensus for our event store and prove the "
            "safety invariant holds under network partition.",
        ),
        (
            "regulated DATA present -> the local gate blocks the cloud call entirely",
            "Patient John Smith, SSN 666-45-1234, MRN: ABC-9931, diagnosed with type 2 "
            "diabetes. Summarize his chart for the insurer.",
        ),
        (
            "regulated TOPIC, no data -> the case keyword routers get wrong",
            "Explain how HIPAA actually works, who it applies to, and the part everyone gets wrong.",
        ),
        (
            "credential material in the prompt",
            "Debug this: our deploy uses sk-live-abcdefghijklmnopqrstuvwx and it 401s.",
        ),
        (
            "compliance question about an ordinary product decision",
            "Our new HR portal needs a GDPR consent banner. What should the copy say?",
        ),
    ]
    backend = args.backend or "mock"
    router, policy, path = _load_router(args, backend_override=backend)
    print(f"policy : {path}")
    print(f"backend: {backend}")
    print(f"tiers  : {json.dumps({k: list(v) for k, v in policy.tiers.items()})}")
    if backend == "mock":
        # Honest disclosure rather than a demo that quietly looks broken. The mock
        # is underconfident by construction, so the policy's on_uncertain floors
        # escalate more often than they would with a calibrated backend. That is
        # fail-safe working as designed, not a bug -- but it should be said.
        print("note   : MockBackend is deliberately underconfident, so you will see more")
        print("         [escalated] lines than a calibrated backend produces. Escalating on")
        print("         uncertainty is the fail-safe; run `--backend jev` to see it calibrated.")
    print()

    async def go() -> None:
        try:
            for label, text in cases:
                decision = await router.route_text(text)
                gate_note = " [gate]" if decision.gate.fired else ""
                esc = " [escalated]" if decision.escalated else ""
                print(f"{label}{gate_note}{esc}")
                print(
                    f"   -> tier={decision.tier} model={decision.model} "
                    f"rule={decision.rule_id} {decision.latency_ms:.0f}ms"
                )
                print(
                    f"   complexity={decision.effective_complexity} "
                    f"sensitivity={decision.effective_sensitivity} "
                    f"pii={decision.answers.pii.value:.2f}"
                )
                print()
        finally:
            await router.aclose()

    asyncio.run(go())
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report whether this machine can actually run each backend. Never raises."""
    policy_path = _find_policy(args.policy)
    policy = Policy.from_file(policy_path)
    print(f"policy file      : {policy_path}")
    print(f"policy backend   : {policy.backend.get('name', 'mock')}")
    print(f"tiers            : {json.dumps({k: list(v) for k, v in policy.tiers.items()})}")
    print(f"failure mode     : {policy.failure.mode} -> {policy.failure.fail_closed_tier}")
    print(f"excerpt mode     : {(policy.logging or {}).get('excerpt_mode', 'hash')}")
    print(f"log path         : {(policy.logging or {}).get('path', '<unset>')}")

    checks: list[tuple[str, bool, str]] = []
    checks.append(("TYPESAFE_API_KEY set", bool(os.environ.get("TYPESAFE_API_KEY")), "required for backend.name: jev"))

    from .gate import HardGate

    gate = HardGate()
    probe = gate.scan("card 4111 1111 1111 1111 and dana@northside-health.org")
    checks.append(
        ("gate detects a card + email", probe.force_local and probe.blocks_backend, f"{len(probe.findings)} findings")
    )
    topic = gate.scan("Explain how HIPAA works.")
    checks.append(
        ("gate treats a HIPAA *question* as advisory", not topic.force_local, f"topics={list(topic.advisory_topics)}")
    )

    from .backends.mock import MockBackend

    try:
        asyncio.run(MockBackend().aclose())
        checks.append(("MockBackend usable offline", True, ""))
    except Exception as exc:
        checks.append(("MockBackend usable offline", False, str(exc)))

    if os.environ.get("TYPESAFE_API_KEY") and not args.offline:
        checks.append(_probe_jev())

    width = max(len(name) for name, _ok, _n in checks)
    failures = 0
    print()
    for name, ok, note in checks:
        failures += 0 if ok else 1
        print(f"  [{'ok ' if ok else 'FAIL'}] {name.ljust(width)}  {note}")
    print(f"\n{len(checks) - failures}/{len(checks)} checks passed")
    return 1 if failures else 0


def _probe_jev() -> tuple[str, bool, str]:
    """One real Jev call, so `doctor` proves the key works rather than just exists."""
    from .backends.base import DecisionRequest
    from .backends.jev import JevBackend
    from .prompts import compute_features

    async def go() -> tuple[bool, str]:
        backend = JevBackend(timeout_seconds=float(os.environ.get("JEV_ROUTE_PROBE_TIMEOUT", "20")))
        try:
            result = await backend.decide(
                DecisionRequest(redacted_excerpt="Is this a test?", features=compute_features("Is this a test?"))
            )
            if result.degraded:
                return False, f"degraded: {result.degrade_reason}"
            return True, f"{result.model_version} in {result.latency_ms:.0f}ms"
        finally:
            await backend.aclose()

    try:
        ok, note = asyncio.run(go())
    except Exception as exc:
        return ("Jev API reachable", False, f"{type(exc).__name__}: {exc}")
    return ("Jev API reachable", ok, note)


def cmd_log_stats(args: argparse.Namespace) -> int:
    """Summarize the decision log: what you are accumulating, and what it can train."""
    records = _load_records(args)
    if not records:
        print("log is empty")
        return 0

    tiers = Counter(r.decision.tier for r in records)
    backends = Counter(r.decision.backend for r in records)
    rules = Counter(r.decision.rule_id for r in records)
    complexity = Counter(r.decision.answers.complexity.choice for r in records)
    sensitivity = Counter(r.decision.answers.sensitivity.choice for r in records)
    versions = Counter(r.decision.backend_model_version for r in records)
    schema_versions = Counter(r.schema_version for r in records)
    escalated = sum(1 for r in records if r.decision.escalated)
    degraded = sum(1 for r in records if r.decision.degraded)
    gate_forced = sum(1 for r in records if r.decision.gate.force_local)
    gate_blocked = sum(1 for r in records if r.decision.gate.blocks_backend)
    with_text = sum(1 for r in records if r.excerpt)
    latencies = sorted(r.total_latency_ms for r in records)

    def pct(q: float) -> float:
        if not latencies:
            return 0.0
        return latencies[min(len(latencies) - 1, int(q * len(latencies)))]

    n = len(records)
    report: dict[str, Any] = {
        "records": n,
        "schema_versions": dict(schema_versions),
        "tiers": dict(tiers.most_common()),
        "backends": dict(backends.most_common()),
        "rules": dict(rules.most_common()),
        "complexity": dict(complexity.most_common()),
        "sensitivity": dict(sensitivity.most_common()),
        "teacher_model_versions": dict(versions.most_common()),
        "escalated": escalated,
        "degraded": degraded,
        "gate_forced_local": gate_forced,
        "gate_blocked_cloud": gate_blocked,
        "records_with_excerpt_text": with_text,
        "latency_ms": {"p50": round(pct(0.50), 1), "p95": round(pct(0.95), 1), "p99": round(pct(0.99), 1)},
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    print(f"decision records : {n}")
    print(f"schema versions  : {dict(schema_versions)}")
    print(f"teacher versions : {dict(versions.most_common())}")
    print(f"tiers            : {dict(tiers.most_common())}")
    print(f"backends         : {dict(backends.most_common())}")
    print(f"rules fired      : {dict(rules.most_common(8))}")
    print(f"complexity       : {dict(complexity.most_common())}")
    print(f"sensitivity      : {dict(sensitivity.most_common())}")
    print(f"escalated        : {escalated} ({escalated / n:.1%})")
    print(f"degraded         : {degraded} ({degraded / n:.1%})")
    print(f"gate forced local: {gate_forced} ({gate_forced / n:.1%})")
    print(f"gate blocked cloud: {gate_blocked} ({gate_blocked / n:.1%})")
    print(f"latency ms p50/p95/p99: {report['latency_ms']}")
    print()
    # The one number that decides which graduation path is open.
    if with_text:
        print(f"DISTILLABLE: {with_text}/{n} records carry redacted excerpt text.")
        print("  -> text-mode distillation is available (best accuracy).")
    else:
        print("No record carries excerpt text (logging.excerpt_mode is 'hash' or 'none').")
        print("  -> features-mode distillation only. To enable text mode, set")
        print("     logging.excerpt_mode: redacted and collect more traffic.")
        print("     Note: gate-blocked requests are never stored as text, by design.")
    degraded_rows = sum(1 for r in records if r.decision.degraded)
    usable = n - degraded_rows
    print(f"  -> usable training rows (excluding degraded): {usable}")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    """Replay one logged decision and show why it landed where it did."""
    records = _load_records(args)
    match = [r for r in records if r.request_id == args.request_id]
    if not match:
        print(f"no record with request_id {args.request_id!r}. Recent ids:")
        for r in records[-10:]:
            print(f"  {r.request_id}  tier={r.decision.tier} rule={r.decision.rule_id}")
        return 1
    record = match[-1]
    print(json.dumps(record.to_dict(), indent=2, default=str, sort_keys=True))
    return 0


# --------------------------------------------------------------------------- #
# graduation pipeline verbs
# --------------------------------------------------------------------------- #
def _require_distill() -> Any:
    try:
        from . import distill
    except ImportError as exc:
        raise SystemExit(
            "the graduation pipeline needs extra dependencies:\n"
            "    pip install 'jev-route[distill]'\n"
            f"(import failed with: {exc})"
        ) from exc
    return distill


def cmd_export(args: argparse.Namespace) -> int:
    distill = _require_distill()
    return distill.export_cli(args)


def cmd_train(args: argparse.Namespace) -> int:
    distill = _require_distill()
    return distill.train_cli(args)


def cmd_evaluate(args: argparse.Namespace) -> int:
    distill = _require_distill()
    return distill.evaluate_cli(args)


def cmd_package(args: argparse.Namespace) -> int:
    distill = _require_distill()
    return distill.package_cli(args)


def cmd_graduate(args: argparse.Namespace) -> int:
    distill = _require_distill()
    return distill.graduate_cli(args, load_router=_load_router, find_policy=_find_policy)


def cmd_label_sensitivity(args: argparse.Namespace) -> int:
    distill = _require_distill()
    return distill.label_sensitivity_cli(args)


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Calibrated LLM routing you bootstrap in the cloud and end up owning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "typical lifecycle:\n"
            f"  {PROG} doctor                       can I run this at all?\n"
            f"  {PROG} demo                         route canned prompts, no API key needed\n"
            f"  {PROG} route 'summarize this thread'  one decision\n"
            "  ... run your traffic for a while; the decision log fills up ...\n"
            f"  {PROG} log-stats                    what have I accumulated?\n"
            f"  {PROG} export --out data/ds         log -> training set\n"
            f"  {PROG} train --data data/ds --out artifacts/v1\n"
            f"  {PROG} evaluate --artifact artifacts/v1\n"
            f"  {PROG} graduate --artifact artifacts/v1 --policy policies/default.yaml\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {_version()}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_policy_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--policy", default=None, help=f"policy YAML (default: {DEFAULT_POLICY} or $JEV_ROUTE_POLICY)")
        p.add_argument(
            "--backend",
            default=None,
            choices=[None, "mock", "jev", "distilled", "shadow"],
            help="override the policy's backend for this invocation",
        )
        p.add_argument("--json", action="store_true", help="emit machine-readable JSON")
        p.add_argument("-v", "--verbose", action="store_true")

    p = sub.add_parser("optimize-questions", help="GEPA-compile routing question wording (manual, keyed; never in CI)")
    p.add_argument("--trace", default="traces/demo_500.jsonl")
    p.add_argument("--policy", default="policies/default.yaml")
    p.add_argument("--out", default="policy/compiled_questions.yaml")
    p.add_argument("--max-metric-calls", type=int, default=150)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--reflection-model", default="openai/w/models/Qwen3.8-27B-Q8_0.gguf",
                   help="litellm model string for GEPA reflection")
    p.add_argument("--reflection-base", default="http://192.168.1.124:8010/v1",
                   help="OpenAI-compatible endpoint for the reflection model")
    p.add_argument("--dry-run", action="store_true", help="validate seed/dataset/evaluator without spending calls")
    p.set_defaults(func=_cmd_optimize_questions)

    p = sub.add_parser("backtest", help="replay a request trace through a policy and measure cost delta")
    p.add_argument("--trace", required=True, help="JSONL trace (e.g. traces/demo_500.jsonl)")
    p.add_argument("--policy", default="policies/default.yaml")
    p.add_argument("--report", default=None, help="write a markdown report here (else print summary)")
    p.set_defaults(func=_cmd_backtest)

    p = sub.add_parser("route", help="route one prompt and print the decision")
    p.add_argument("text", nargs="?", default=None, help="prompt text (or pipe it on stdin)")
    p.add_argument("--metadata", default=None, help="JSON object of caller metadata")
    add_policy_args(p)
    p.set_defaults(func=cmd_route)

    p = sub.add_parser("demo", help="route a canned set of prompts (defaults to MockBackend)")
    add_policy_args(p)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("doctor", help="check that this machine can run each backend")
    p.add_argument("--policy", default=None)
    p.add_argument("--offline", action="store_true", help="skip the live Jev probe")
    p.set_defaults(func=cmd_doctor)

    def add_log_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--policy", default=None)
        p.add_argument("--log", default=None, help="decision log path (default: from policy)")
        p.add_argument("--json", action="store_true")

    p = sub.add_parser("log-stats", help="summarize the decision log you are accumulating")
    add_log_args(p)
    p.set_defaults(func=cmd_log_stats)

    p = sub.add_parser("explain", help="replay one logged decision")
    p.add_argument("request_id")
    add_log_args(p)
    p.set_defaults(func=cmd_explain)

    # --- graduation pipeline. Argument names are fixed here so the distill
    #     module's *_cli(args) functions have a stable contract.
    p = sub.add_parser("export", help="decision log -> training dataset")
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["auto", "text", "features"], default="auto")
    p.add_argument("--holdout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    add_log_args(p)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("train", help="distill a local model with soft-target KL")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--trainer", choices=["auto", "numpy", "torch"], default="auto")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--l2", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--head-weights", default=None, help="JSON, e.g. '{\"sensitivity\":2.0}'")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="accuracy, ECE, teacher agreement, latency")
    p.add_argument("--artifact", required=True)
    p.add_argument("--data", default=None, help="dataset (default: the one recorded in the artifact)")
    p.add_argument("--policy", default=None)
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("package", help="write a loadable artifact from a trained model")
    # `--artifact` is the spelling every other verb uses for this same object
    # (`evaluate --artifact`, `graduate --artifact`). `--model` stays as a
    # deprecated alias for one release so a script written against 0.1 keeps
    # working instead of breaking silently; distill.package_cli enforces
    # exactly-one-of and prints the deprecation warning.
    p.add_argument("--artifact", default=None, help="trained model to package (artifact directory or zip)")
    p.add_argument("--model", default=None, help="DEPRECATED alias for --artifact")
    p.add_argument("--out", required=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_package)

    p = sub.add_parser("graduate", help="report readiness and (optionally) cut over to the local model")
    p.add_argument(
        "--track",
        choices=["router", "gate"],
        default="router",
        help="what graduates: the routing model (default) or the semantic gate layer (shadow -> enforce)",
    )
    p.add_argument(
        "--artifact", required=True, help="router track: packaged routing artifact; gate track: semantic artifact"
    )
    p.add_argument("--policy", default=None)
    p.add_argument("--data", default=None)
    p.add_argument(
        "--shadow-metrics", default=None, help="gate track: live shadow metrics JSON (from measure_shadow_log)"
    )
    p.add_argument("--log", default=None, help="gate track: measure live shadow metrics from this decision log")
    p.add_argument(
        "--replay", action="store_true", help="re-run held-out prompts through the live teacher (needs a Jev key)"
    )
    p.add_argument("--min-tier-agreement", type=float, default=0.95)
    p.add_argument("--max-ece", type=float, default=0.10)
    p.add_argument("--max-latency-ms", type=float, default=50.0)
    p.add_argument("--min-samples", type=int, default=500)
    p.add_argument("--write", action="store_true", help="actually perform the backend swap")
    p.add_argument("--in-place", action="store_true", help="overwrite the policy file (default: write a new one)")
    p.add_argument("--out-policy", default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_graduate)

    p = sub.add_parser("label-sensitivity", help="build + label a sensitivity dataset (allowed sources only)")
    p.add_argument("--out", required=True, help="output directory (dataset.jsonl + stats sidecar + audit)")
    p.add_argument("--kinds", default=None, help="comma-separated synthetic PII kinds (default: all)")
    p.add_argument("--per-kind", type=int, default=6)
    p.add_argument("--contextual", type=int, default=2, help="contextual templates per category")
    p.add_argument("--production-log", default=None, help="decision log to harvest cleared negatives from")
    p.add_argument("--production-limit", type=int, default=None)
    p.add_argument(
        "--include-blocked-metadata", action="store_true", help="blocked records contribute a text-free projection"
    )
    p.add_argument("--mode", choices=["text", "features"], default="text")
    p.add_argument(
        "--backend", choices=["mock", "jev"], default="mock", help="teacher (mock is offline and deterministic)"
    )
    p.add_argument("--seed", type=int, default=20260919)
    p.add_argument("--exported-at", default=None)
    p.add_argument("--audit", default=None, help="labeling audit path (default: <out>/labeling-audit.jsonl)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_label_sensitivity)

    return parser


def _version() -> str:
    try:
        from . import __version__

        return __version__
    except Exception:
        return "unknown"


def _cmd_optimize_questions(args: Any) -> int:
    import os

    if not os.environ.get("TYPESAFE_API_KEY"):
        print("error: TYPESAFE_API_KEY is not set. Export it first "
              "(the key lives in the untracked .env; source it or export it manually).", file=sys.stderr)
        return 2
    from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything
    from optimize.questions import QuestionEvaluator, load_stratified, provenance_header, seed_candidate, write_compiled

    from .policy import Policy

    policy = Policy.from_file(args.policy)
    train, val = load_stratified(args.trace, n_train=80, n_val=60, seed=args.seed)
    evaluator = QuestionEvaluator(policy=policy)
    seed = seed_candidate()
    if args.dry_run:
        probe = evaluator.evaluate(seed, train[:4])
        print(f"dry-run ok: train={len(train)} val={len(val)}; seed probe score {probe['score']:.3f} "
              f"(accuracy {probe['accuracy']:.2f}, savings {probe['savings']:.2f})")
        return 0

    print(f"compiling: train={len(train)} val={len(val)} budget={args.max_metric_calls} metric calls")
    def _per_example(candidate: dict, example: dict | None = None, **_kw: object):
        """GEPA's per-example contract: one trace row in, (score, outputs) out;
        oa.log carries the misroute feedback that becomes the gradient."""
        import gepa.optimize_anything as oa

        assert example is not None
        outcome = evaluator.evaluate_one(candidate, example)
        if outcome["miss"]:
            oa.log(
                f"{example['id']}: policy intends {outcome['expected']}, candidate questions "
                f"produced {outcome['got']} (rule {outcome['rule_id']})"
            )
        return outcome["score"], {"id": example["id"], "expected": outcome["expected"], "got": outcome["got"]}

    result = optimize_anything(
        seed_candidate=seed,
        evaluator=_per_example,
        dataset=[{"id": r["id"], "input": r["text"], "expected": r["category"]} for r in train],
        valset=[{"id": r["id"], "input": r["text"], "expected": r["category"]} for r in val],
        objective=(
            "Optimize routing question wording to maximize tier accuracy on the synthetic trace, "
            "with cost as the tiebreaker. Misroute feedback lists expected-vs-produced tiers."
        ),
        background=(
            "The artifact is routing question TEXT for an LLM router. Expected tiers come from a "
            "synthetic trace with category labels (trivial_completion->local, code_gen->cheap, "
            "multi_file_refactor->strong, hungarian_support->cheap, pii_fake->local, "
            "long_context->strong)."
        ),
        config=GEPAConfig(
            engine=EngineConfig(seed=args.seed, max_metric_calls=args.max_metric_calls),
            reflection=ReflectionConfig(
                reflection_lm=args.reflection_model,
                reflection_lm_kwargs={"api_base": args.reflection_base, "api_key": "sk-local"},
                reflection_minibatch_size=12,
            ),
        ),
    )
    best = getattr(result, "best_candidate", None) or seed
    blob = best.get("questions_yaml", seed["questions_yaml"]) if isinstance(best, dict) else seed["questions_yaml"]
    baseline = evaluator.evaluate(seed, val)
    compiled = evaluator.evaluate(best, val)
    write_compiled(
        args.out,
        blob,
        provenance_header(
            baseline_score=baseline["score"], compiled_score=compiled["score"],
            seed=args.seed, max_metric_calls=args.max_metric_calls,
        ),
    )
    print(f"baseline {baseline['score']:.4f} -> compiled {compiled['score']:.4f} (valset n={len(val)})")
    print(f"wrote {args.out}")
    return 0


def _cmd_backtest(args: Any) -> int:
    import asyncio

    from .backtest import render_report, run_backtest
    from .policy import Policy

    policy = Policy.from_file(args.policy)
    report = asyncio.run(run_backtest(trace_path=args.trace, policy=policy))
    if args.report:
        from pathlib import Path

        Path(args.report).write_text(render_report(report, trace_path=args.trace, policy_name=args.policy))
        print(f"report written to {args.report}")
    print(
        f"{report.total} requests | ours ${report.cost_ours:.4f} vs baseline ${report.cost_baseline:.4f} "
        f"| savings {report.savings_pct:.1f}% | gate fires {report.gate_fires} | tiers {report.by_tier}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
