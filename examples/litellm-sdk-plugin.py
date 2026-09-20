#!/usr/bin/env python
"""Mode 1: jev-route as a ``Router(plugins=[...])`` routing plugin, from the SDK.

Run it::

    PYTHONPATH=src .venv/bin/python examples/litellm-sdk-plugin.py            # offline, stubbed upstream
    PYTHONPATH=src .venv/bin/python examples/litellm-sdk-plugin.py --live     # real backends

The offline run needs no network and no API key of any kind: the decision
backend is the deterministic MockBackend, and the *upstream* LLM call is stubbed
(see ``install_stub_upstream`` -- it monkeypatches ``litellm.acompletion``, which
is what ``Router`` calls internally, and records what it was asked for).
Everything up to that boundary is real: a real ``litellm.Router``, a real
``RoutingContext``, the real plugin pipeline, and a real decision log written to
disk.

What this demonstrates, in order:

1. **The plugin narrows LiteLLM's candidate deployments.** ``Router`` seeds
   ``context.candidate_models`` from ``resolved_litellm_models(model)``, runs
   every plugin, then keeps only the healthy deployments whose
   ``litellm_params.model`` survived. So the tier -> models mapping is in the
   *provider* namespace (``openai/qwen3.8``), not the operator-facing
   ``model_name`` namespace (``qwen38``) that the jev-route policy file uses.
   That mismatch is the one thing everybody gets wrong here, and the script
   prints both namespaces side by side so you can see it.
2. **The signals survive into LiteLLM's own metadata.** ``Router`` copies
   ``context.signals`` into ``request_kwargs[...]["routing_plugin_signals"]``,
   which is where a downstream guardrail, logger or spend tracker reads them.
3. **Every decision is logged, soft distributions included.** The last section
   reads the JSONL back and prints the probability distributions, because that
   file -- not this script's stdout -- is the product. It is what
   ``jev-route train`` distils a local model from later.

Only ``Router(plugins=[...])`` is used here. For a proxy deployment, see
``examples/litellm-proxy-classifier/config.yaml`` (native complexity router +
ClassifierPlugin) and ``examples/litellm-proxy-hook/config.yaml`` (pre-call hook).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import litellm
from litellm import Router

from jev_route.integrations import _shared
from jev_route.integrations.litellm_plugin import JevRouteRoutingPlugin

# --------------------------------------------------------------------------- #
# The deployment pool
# --------------------------------------------------------------------------- #
#: THE SHAPE THAT MAKES MODE 1 WORK.
#:
#: ``Router._run_routing_plugins`` seeds ``context.candidate_models`` from
#: ``resolved_litellm_models(model)`` -- the ``litellm_params.model`` strings of
#: every deployment registered under the *requested* model name. So a group with
#: one deployment gives the plugin one candidate and nothing to narrow: the
#: decision is made, logged, and then ignored. Put all the tier deployments in
#: ONE group and the plugin has a real choice to make.
#:
#: The three named entries below the group are there so a client can still pin a
#: tier directly; the plugin never sees those requests, because it only runs for
#: the group it is asked about.
LOCAL = {
    "model": "openai/qwen3.8",
    "api_base": os.environ.get("JEV_ROUTE_LOCAL_API_BASE", "http://192.168.1.124:8010/v1"),
    # llama.cpp ignores the key; litellm requires a non-empty value.
    "api_key": "EMPTY",
}
ALIBABA_BASE = "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
#: Read from the environment, never inlined. The stubbed run never sends it anywhere.
ALIBABA_KEY = os.environ.get("ALIBABA_API_KEY", "unset-offline")

MODEL_LIST: list[dict[str, Any]] = [
    # --- the routed group: all three tiers, one model_name -------------------
    {"model_name": "auto", "litellm_params": dict(LOCAL)},
    {
        "model_name": "auto",
        "litellm_params": {"model": "openai/qwen3.8-flash", "api_base": ALIBABA_BASE, "api_key": ALIBABA_KEY},
    },
    {
        "model_name": "auto",
        "litellm_params": {"model": "openai/qwen3.8-max", "api_base": ALIBABA_BASE, "api_key": ALIBABA_KEY},
    },
    # --- direct access to each tier, for pinning and for comparison ----------
    {"model_name": "qwen38", "litellm_params": dict(LOCAL)},
    {
        "model_name": "qwen3.8-flash",
        "litellm_params": {"model": "openai/qwen3.8-flash", "api_base": ALIBABA_BASE, "api_key": ALIBABA_KEY},
    },
    {
        "model_name": "qwen3.8-max",
        "litellm_params": {"model": "openai/qwen3.8-max", "api_base": ALIBABA_BASE, "api_key": ALIBABA_KEY},
    },
]

#: tier -> the ``litellm_params.model`` strings that serve it. This is the
#: mapping the plugin narrows with, and it is in the *provider* namespace on
#: purpose -- see point 1 in the module docstring. The policy file's own
#: ``tiers:`` uses ``model_name`` values (``qwen38``), which is what a proxy
#: deployment wants; the plugin needs this one instead.
TIER_MODELS: dict[str, tuple[str, ...]] = {
    "local": ("openai/qwen3.8",),
    "cheap": ("openai/qwen3.8-flash",),
    "strong": ("openai/qwen3.8-max",),
}

#: Four prompts that land in different places, so one run shows the whole ladder.
PROMPTS: list[tuple[str, str]] = [
    ("greeting", "Say hello and thank me for the recipe."),
    ("routine", "Rewrite this paragraph of public documentation so it reads better."),
    (
        "hard",
        "Design the architecture for a distributed consensus layer, derive the proof that it is "
        "safe under partition, then optimize the p99 latency and explain the trade-offs of "
        "sharding versus replication.",
    ),
    (
        "sensitive",
        "Patient record for John Smith, SSN 123-45-6789, card 4111 1111 1111 1111. "
        "Summarize the clinical notes and the prescription.",
    ),
]


# --------------------------------------------------------------------------- #
# The offline upstream stub
# --------------------------------------------------------------------------- #
def install_stub_upstream() -> list[dict[str, Any]]:
    """Monkeypatch ``litellm.acompletion`` so the run needs no network.

    This is the *only* thing stubbed, and it is stubbed at the boundary rather
    than inside jev-route: everything the plugin pipeline does -- context
    construction, narrowing, deployment filtering, health checks -- is real
    LiteLLM code running against a real ``Router``. ``Router.acompletion`` ends
    in ``litellm.acompletion(**kwargs)``, so replacing that module attribute
    intercepts the call without touching the routing path.

    Returns the list the stub appends to, so the caller can assert on what was
    actually dispatched.
    """
    from litellm.types.utils import Choices, Message, ModelResponse, Usage

    calls: list[dict[str, Any]] = []

    async def fake_acompletion(*args: Any, **kwargs: Any) -> ModelResponse:
        model = str(kwargs.get("model") or (args[0] if args else "unknown"))
        calls.append({"model": model, "api_base": kwargs.get("api_base"), "messages": kwargs.get("messages")})
        return ModelResponse(
            id=f"stub-{len(calls)}",
            choices=[
                Choices(
                    index=0,
                    message=Message(content=f"[stubbed upstream: {model}]", role="assistant"),
                    finish_reason="stop",
                )
            ],
            model=model,
            created=int(time.time()),
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    litellm.acompletion = fake_acompletion  # type: ignore[assignment]
    return calls


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def build_router(plugin: JevRouteRoutingPlugin) -> Router:
    """A real LiteLLM router with jev-route in its plugin pipeline.

    ``routing_strategy`` must be one of the async-native ones (``simple-shuffle``
    is the default and is fine): LiteLLM *raises* if plugins are configured and a
    call resolves to the synchronous deployment-selection path, because silently
    skipping a policy plugin would let a deny-all rule be bypassed. That is a
    good failure, and it is why this example uses ``acompletion`` and
    ``async_get_available_deployment`` rather than their sync twins.
    """
    return Router(
        model_list=[dict(entry, litellm_params=dict(entry["litellm_params"])) for entry in MODEL_LIST],
        routing_strategy="simple-shuffle",
        plugins=[plugin],
    )


def print_deployment_step(plugin: JevRouteRoutingPlugin) -> None:
    print("=" * 78)
    print("STEP 1 -- deployment selection through the plugin pipeline")
    print("=" * 78)
    routed = sorted({str(entry["litellm_params"]["model"]) for entry in MODEL_LIST if entry["model_name"] == "auto"})
    print(f"candidates for model_name='auto': {json.dumps(routed)}")
    print(f"policy tiers (model_name namespace): {json.dumps(_tier_view(plugin))}")
    print(f"plugin mapping (litellm_params.model namespace): {json.dumps(TIER_MODELS)}")
    print()


def _tier_view(plugin: JevRouteRoutingPlugin) -> dict[str, list[str]]:
    try:
        return {tier: list(models) for tier, models in plugin.router.policy.tiers.items()}
    except Exception as exc:  # broad except, deliberately: display only
        return {"<unavailable>": [str(exc)]}


async def run_selection(router: Router) -> list[tuple[str, Any, dict[str, Any]]]:
    """Ask LiteLLM which deployment would serve each prompt, and show why."""
    results = []
    for label, prompt in PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        request_kwargs: dict[str, Any] = {"metadata": {"model_group": "auto", "user_api_key_alias": "example"}}
        deployment = await router.async_get_available_deployment(
            model="auto", messages=messages, request_kwargs=request_kwargs
        )
        signals = (request_kwargs.get("metadata") or {}).get("routing_plugin_signals", {})
        jev = signals.get("jev_route", {}) if isinstance(signals, dict) else {}
        litellm_params = deployment.get("litellm_params", {})
        print(
            f"[{label:9}] -> model_name={deployment.get('model_name'):<14} "
            f"litellm_params.model={litellm_params.get('model')}"
        )
        print(f"            jev-route: tier={jev.get('tier')} model={jev.get('model')} rule={jev.get('rule_id')}")
        print(f"            why: {jev.get('reason')}")
        print(
            f"            complexity={jev.get('complexity')} (p={jev.get('complexity_confidence')}) "
            f"sensitivity={jev.get('sensitivity')} (p={jev.get('sensitivity_confidence')}) "
            f"pii={jev.get('pii')} gate_force_local={jev.get('gate_force_local')}"
        )
        print(f"            candidates: {jev.get('candidates_before')} available -> kept {jev.get('candidates_after')}")
        if jev.get("fallback_applied"):
            print(f"            fallback_applied: {jev['fallback_applied']}")
        print()
        results.append((label, deployment, jev))
    return results


async def run_completion(router: Router, stubbed: bool) -> list[dict[str, Any]]:
    """Push one request all the way through ``Router.acompletion``."""
    print("=" * 78)
    print(
        "STEP 2 -- a full Router.acompletion through the plugin"
        + (" (upstream STUBBED: litellm.acompletion monkeypatched)" if stubbed else " (live upstream)")
    )
    print("=" * 78)
    calls = install_stub_upstream() if stubbed else []
    for label, prompt in PROMPTS:
        response = await router.acompletion(model="auto", messages=[{"role": "user", "content": prompt}])
        served = getattr(response, "model", None)
        text = response.choices[0].message.content if getattr(response, "choices", None) else ""
        dispatched = calls[-1]["model"] if calls else served
        print(f"[{label:9}] answered by litellm_params.model={dispatched}  response.model={served}")
        if not stubbed:
            print(f"            {str(text)[:100]}")
    if stubbed:
        print()
        print(f"stubbed upstream dispatched {len(calls)} call(s): {json.dumps([c['model'] for c in calls])}")
    print()
    return calls


def print_dataset(log_path: Path) -> None:
    """Read the decision log back: this file is the actual product."""
    print("=" * 78)
    print("STEP 3 -- the decision log (this is the training dataset)")
    print("=" * 78)
    if not log_path.exists():
        print(f"nothing at {log_path}")
        return
    from jev_route import iter_records

    rows = list(iter_records(log_path))
    print(f"{log_path}: {len(rows)} record(s), schema_version={rows[0].schema_version if rows else '-'}")
    for row in rows[-4:]:
        answers = row.decision.answers
        print(
            f"  {row.request_id[:12]} tier={row.decision.tier:<6} rule={row.decision.rule_id:<20} "
            f"backend={row.decision.backend}"
        )
        print(f"      complexity  {json.dumps(dict(answers.complexity.probabilities))}")
        print(f"      sensitivity {json.dumps(dict(answers.sensitivity.probabilities))}")
        print(f"      pii         p={answers.pii.value:.4f}   domain={answers.domain.choice}")
    print()
    print("  Every row carries the FULL soft distribution, not just the argmax.")
    print("  That calibration is what `jev-route train` distils a local model from,")
    print("  and it is the reason to bootstrap on a calibrated cloud model at all.")
    print()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="call the real backends instead of the stub")
    parser.add_argument("--policy", default=None, help=f"path to a policy file (default: ${_shared.POLICY_ENV_VAR})")
    parser.add_argument(
        "--log",
        default=os.environ.get(_shared.DECISION_LOG_ENV_VAR, "./decision-log/decisions.jsonl"),
        help="where to write (and then read back) the decision log",
    )
    args = parser.parse_args(argv)

    # Pin the log path for this run. The env var is honoured by _shared exactly
    # the way a deployment would use it, which is what makes STEP 3 read the file
    # this run actually wrote.
    os.environ[_shared.DECISION_LOG_ENV_VAR] = str(args.log)
    log_path = Path(args.log)
    before = log_path.stat().st_size if log_path.exists() else 0

    router = _shared.build_router(args.policy)
    print(f"decision backend : {router.backend.name} ({router.backend.model_version})")
    print(f"policy           : {router.policy.source or '<built-in fallback>'}")
    print(f"decision log     : {log_path}")
    if not os.environ.get("TYPESAFE_API_KEY"):
        print(
            "note             : TYPESAFE_API_KEY is not set, so the mock backend is judging."
            " Set it and use a policy with backend.name: jev for calibrated cloud decisions."
        )
    print()

    plugin = JevRouteRoutingPlugin(router, tier_models=TIER_MODELS)
    litellm_router = build_router(plugin)

    print_deployment_step(plugin)
    asyncio.run(_run(litellm_router, live=args.live))

    # Only show the records this run appended, so a re-run does not reprint the
    # whole accumulated log.
    if before and log_path.exists():
        with log_path.open("rb") as handle:
            handle.seek(before)
            fresh = handle.read().decode("utf-8", "replace")
        tmp = log_path.with_suffix(".tail.jsonl")
        tmp.write_text(fresh, encoding="utf-8")
        print_dataset(tmp)
        tmp.unlink(missing_ok=True)
    else:
        print_dataset(log_path)
    return 0


async def _run(litellm_router: Router, *, live: bool) -> None:
    await run_selection(litellm_router)
    await run_completion(litellm_router, stubbed=not live)


if __name__ == "__main__":
    sys.exit(main())
