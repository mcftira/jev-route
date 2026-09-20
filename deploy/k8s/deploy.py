#!/usr/bin/env python3
"""Deploy jev-route + LiteLLM to a Kubernetes cluster.

What it does, in order:

1. Regenerates the code and policy ConfigMaps from ``src/`` (never trust a stale one).
2. Creates the API-key Secret from the environment or a dotenv file. Keys are never
   written into a manifest, and the generated Secret is applied from stdin so it does
   not land on disk.
3. Injects the ConfigMap ``items[]`` mapping into the Deployment's code volume, which
   is what restores ``backends/`` and ``integrations/`` as real subdirectories.
4. Applies everything and waits for the rollout.
5. Smoke-tests the proxy: liveness, ``/v1/models``, one completion per tier, and --
   with ``--hook`` -- that jev-route actually rewrote the model.

Usage:
    export TYPESAFE_API_KEY=... ALIBABA_API_KEY=...
    python3 deploy/k8s/deploy.py --context spark-local
    python3 deploy/k8s/deploy.py --context spark-local --no-hook   # LiteLLM alone
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
K8S = REPO / "deploy" / "k8s"

REQUIRED_SECRETS = ("TYPESAFE_API_KEY", "ALIBABA_API_KEY")


def sh(argv: list[str], *, input_text: str | None = None, check: bool = True, capture: bool = True) -> str:
    """Run a command, returning stdout. Errors carry the command and its output."""
    printable = " ".join(shlex.quote(a) for a in argv)
    proc = subprocess.run(
        argv,
        input=input_text,
        capture_output=capture,
        text=True,
    )
    if check and proc.returncode != 0:
        sys.exit(f"command failed ({proc.returncode}): {printable}\n{proc.stdout}\n{proc.stderr}")
    return (proc.stdout or "") + (proc.stderr or "")


def kubectl(ctx: str, *args: str, input_text: str | None = None, check: bool = True) -> str:
    return sh(["kubectl", "--context", ctx, *args], input_text=input_text, check=check)


def load_env_file(path: Path) -> None:
    """Minimal dotenv loader, so the script has no dependencies beyond PyYAML."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def cluster_master_key(ctx: str, namespace: str) -> str | None:
    """The proxy master key already stored in the cluster Secret, if there is one.

    Reusing it is not tidiness, it is correctness. The LiteLLM proxy reads the
    master key ONCE, at boot, and this deployment has no ``DATABASE_URL``. So if a
    redeploy rotates the key in the Secret but the pod does not restart -- which is
    exactly what happens on a ConfigMap-only change -- the running proxy still
    enforces the old key. A client presenting the new one is not recognized as the
    master key, LiteLLM falls through to a virtual-key database lookup, and every
    request fails with a baffling ``400 {"error": {"message": "No connected db."}}``
    that looks like a database problem and is really a key mismatch.
    """
    out = kubectl(
        ctx,
        "-n",
        namespace,
        "get",
        "secret",
        "jev-route-secrets",
        "-o",
        "jsonpath={.data.JEV_ROUTE_MASTER_KEY}",
        check=False,
    ).strip()
    if not out or "NotFound" in out or "not found" in out or "Error" in out:
        return None
    try:
        decoded = base64.b64decode(out, validate=True).decode("utf-8")
    except Exception:
        return None
    return decoded or None


def build_secret(namespace: str, *, reuse_key: str | None = None) -> str:
    """Render the Secret YAML from the environment. Never written to disk."""
    missing = [k for k in REQUIRED_SECRETS if not os.environ.get(k)]
    if missing:
        sys.exit(
            f"missing required environment variables: {', '.join(missing)}\n"
            f"Export them, or put them in {REPO / '.env'} (gitignored)."
        )
    pinned = os.environ.get("JEV_ROUTE_MASTER_KEY")
    master_key = pinned or reuse_key or "sk-jev-route-" + os.urandom(9).hex()
    if not pinned:
        if reuse_key:
            # Deliberately not printed. It is already in the cluster, and an
            # operator who needs it can read the Secret.
            print("  reusing the proxy master key already in the cluster Secret")
        else:
            # Printed once, because an operator who does not set it has no other way
            # to learn what the proxy will accept. Redacted from every other output.
            print(f"  generated proxy master key (set JEV_ROUTE_MASTER_KEY to pin it): {master_key}")
    data = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": "jev-route-secrets",
            "namespace": namespace,
            "labels": {"app.kubernetes.io/name": "jev-route"},
        },
        "type": "Opaque",
        "stringData": {
            "TYPESAFE_API_KEY": os.environ["TYPESAFE_API_KEY"],
            "ALIBABA_API_KEY": os.environ["ALIBABA_API_KEY"],
            "JEV_ROUTE_MASTER_KEY": master_key,
        },
    }
    return yaml.safe_dump(data, sort_keys=False)


def redact_secrets(text: str) -> str:
    """Replace secret values in rendered manifests with a placeholder.

    Applied to anything printed. The values are matched by key rather than by a
    fixed prefix, so an unusual key format is still caught.
    """
    secret_keys = (*REQUIRED_SECRETS, "JEV_ROUTE_MASTER_KEY")
    out_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        key = stripped.split(":", 1)[0].strip() if ":" in stripped else ""
        if key in secret_keys and len(stripped.split(":", 1)) > 1 and stripped.split(":", 1)[1].strip():
            indent = line[: len(line) - len(line.lstrip())]
            out_lines.append(f"{indent}{key}: <redacted>")
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


def render_deployment(namespace: str, checksums: Mapping[str, str] | None = None) -> str:
    """Deployment YAML with the code-volume items[] mapping injected.

    Takes no ``hook`` flag: enabling the jev-route integration changes the
    LiteLLM *config* ConfigMap, not the Deployment. The pod spec is identical
    either way, which is what makes the phase-1/phase-2 cutover a config change
    rather than a redeploy of a different workload.

    ``checksums`` are stamped into the POD TEMPLATE, and that is what makes this
    script actually deploy anything. Kubernetes does not restart pods when a
    ConfigMap they mount changes -- the new content appears in the volume within a
    minute, but a Python process that already imported the package keeps running
    the old one, and a proxy that already parsed its config keeps using the old
    config. Without these annotations, editing a routing rule or the router source
    and re-running deploy.py would report success and change nothing observable,
    which is the worst possible failure mode for a deploy script. Changing the
    template hash forces the rollout.
    """
    doc = yaml.safe_load((K8S / "20-deployment.yaml").read_text(encoding="utf-8"))
    items_doc = yaml.safe_load((K8S / "16-code-volume-items.yaml").read_text(encoding="utf-8"))
    volumes = doc["spec"]["template"]["spec"]["volumes"]
    for volume in volumes:
        if volume.get("name") == "code":
            volume["configMap"]["items"] = items_doc["items"]
            break
    else:
        sys.exit("deployment has no `code` volume to inject items into")
    doc["metadata"]["namespace"] = namespace
    if checksums:
        annotations = doc["spec"]["template"]["metadata"].setdefault("annotations", {})
        for name, digest in checksums.items():
            annotations[f"jev-route/{name}-checksum"] = digest
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)


def render_litellm_config(hook: bool) -> str:
    """The LiteLLM ConfigMap, with the jev-route callback removed when --no-hook."""
    doc = yaml.safe_load((K8S / "10-litellm-config.yaml").read_text(encoding="utf-8"))
    config = yaml.safe_load(doc["data"]["config.yaml"])
    if not hook:
        (config.get("litellm_settings") or {}).pop("callbacks", None)
    doc["data"]["config.yaml"] = yaml.safe_dump(config, sort_keys=False, allow_unicode=True, width=10**6)
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=10**6)


def wait_rollout(ctx: str, namespace: str, timeout: int = 300) -> None:
    print(f"waiting for rollout (timeout {timeout}s)...")
    kubectl(ctx, "-n", namespace, "rollout", "status", "deployment/jev-route-litellm", f"--timeout={timeout}s")


def proxy_base(ctx: str, namespace: str) -> tuple[str, str]:
    """Resolve the proxy URL and master key from the cluster."""
    out = kubectl(
        ctx, "-n", namespace, "get", "svc", "jev-route", "-o", "jsonpath={.status.loadBalancer.ingress[0].ip}"
    )
    ip = out.strip()
    if not ip:
        # Not all clusters hand out an LB IP; fall back to port-forward territory.
        sys.exit("service jev-route has no LoadBalancer IP yet; re-run in a moment")
    key = kubectl(
        ctx, "-n", namespace, "get", "secret", "jev-route-secrets", "-o", "jsonpath={.data.JEV_ROUTE_MASTER_KEY}"
    )
    return f"http://{ip}:4100", base64.b64decode(key).decode()


def smoke_test(base: str, key: str, hook: bool) -> int:
    """Hit the proxy once per tier plus the managed alias. Returns failures."""
    import urllib.error
    import urllib.request

    def call(model: str, prompt: str, max_tokens: int = 24) -> dict[str, Any]:
        """One completion. Returns the RESOLVED upstream, not the alias we asked for.

        ``payload["model"]`` echoes the request, so it says ``auto`` no matter what
        the router decided -- checking it would make the routing assertion below
        unfalsifiable. The proxy reports the deployment it actually used in the
        ``x-litellm-model-group`` response header, which is the only field here that
        can fail.
        """
        body = json.dumps(
            {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
        ).encode()
        req = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        started = time.time()
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                payload = json.loads(resp.read().decode())
                headers = {k.lower(): v for k, v in resp.headers.items()}
        except urllib.error.HTTPError as exc:
            return {"ok": False, "status": exc.code, "error": exc.read().decode()[:300]}
        except Exception as exc:  # URLError, TimeoutError, socket.timeout, bad JSON
            # Caught broadly on purpose. Both upstream tiers are reasoning models and
            # a hard prompt can outrun any fixed client timeout; an uncaught
            # TimeoutError here used to abort the deploy script after the rollout had
            # already succeeded, which reads as "the deploy failed" when it did not.
            return {"ok": False, "status": 0, "error": f"{type(exc).__name__}: {exc}"[:300]}
        elapsed = (time.time() - started) * 1000
        choice = (payload.get("choices") or [{}])[0]
        return {
            "ok": True,
            "ms": round(elapsed),
            "model": payload.get("model"),
            "resolved": headers.get("x-litellm-model-group") or payload.get("model"),
            "api_base": headers.get("x-litellm-model-api-base", ""),
            "text": (choice.get("message") or {}).get("content", "")[:80].replace("\n", " "),
        }

    def tier_models() -> dict[str, str]:
        """tier -> first model serving it, read from the policy being deployed."""
        doc = yaml.safe_load((K8S / "policy-cluster.yaml").read_text(encoding="utf-8"))
        return {name: str(models[0]) for name, models in (doc.get("tiers") or {}).items() if models}

    failures = 0
    print("\n=== smoke test ===")
    for model in ("qwen38", "qwen3.8-flash", "qwen3.8-max"):
        result = call(model, "Reply with exactly: OK")
        status = "PASS" if result.get("ok") else "FAIL"
        failures += 0 if result.get("ok") else 1
        print(f"  [{status}] {model:16s} {result.get('ms', '-'):>6}ms -> {result.get('text') or result.get('error')}")

    if hook:
        cases = [
            ("auto", "hi there, thanks!", "cheap"),
            (
                "auto",
                "Prove that no algorithm can sort n items in fewer than log2(n!) comparisons "
                "in the worst case, then design a distributed sharded write path with consensus "
                "and justify the safety invariant under partition.",
                "strong",
            ),
            (
                "auto",
                "Patient John Smith, SSN 666-45-1234, MRN: ABC-9931, diagnosed with type 2 "
                "diabetes. Summarize his chart for the insurer.",
                "local",
            ),
        ]
        print("\n=== jev-route routing through the proxy ===")
        tiers = tier_models()
        for model, prompt, expected_tier in cases:
            result = call(model, prompt, max_tokens=16)
            resolved = result.get("resolved")
            wanted = tiers.get(expected_tier)
            # Three separate things can be wrong and they are reported separately:
            # the call failed, the router picked the wrong tier, or a sensitive
            # prompt was served by a cloud endpoint instead of the cluster.
            ok = bool(result.get("ok")) and resolved == wanted
            if not result.get("ok"):
                reason = f"request failed: {result.get('error')}"
            elif resolved != wanted:
                reason = f"expected {expected_tier}={wanted}, resolved {resolved}"
            else:
                reason = ""
            egress = (
                "cluster"
                if ":8010" in result.get("api_base", "") or "svc.cluster.local" in result.get("api_base", "")
                else "cloud"
            )
            if expected_tier == "local" and result.get("ok") and egress != "cluster":
                ok = False
                reason = f"sensitive request was served by a cloud endpoint ({result.get('api_base')})"
            failures += 0 if ok else 1
            print(
                f"  [{'PASS' if ok else 'FAIL'}] asked={model} expected_tier={expected_tier:6s} "
                f"resolved={resolved} egress={egress} {result.get('ms', '-')}ms"
            )
            if reason:
                print(f"        {reason}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--context", default=os.environ.get("KUBE_CONTEXT", "spark-local"))
    parser.add_argument("--namespace", default="jev-route")
    parser.add_argument("--env-file", default=str(REPO / ".env"))
    parser.add_argument(
        "--no-hook", action="store_true", help="deploy LiteLLM without the jev-route callback (phase 1 / debugging)"
    )
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    hook = not args.no_hook

    load_env_file(Path(args.env_file))
    ctx, ns = args.context, args.namespace

    print(f"target: context={ctx} namespace={ns} hook={hook}")
    print("regenerating ConfigMaps from src/ ...")
    sh([sys.executable, str(K8S / "build_configmap.py")])

    generated = (K8S / "15-generated.yaml").read_text(encoding="utf-8")
    litellm_config = render_litellm_config(hook)
    manifests = [
        (K8S / "00-namespace.yaml").read_text(encoding="utf-8"),
        build_secret(ns, reuse_key=cluster_master_key(ctx, ns)),
        generated,
        litellm_config,
        render_deployment(
            ns,
            checksums={
                "generated": hashlib.sha256(generated.encode("utf-8")).hexdigest()[:16],
                "litellm-config": hashlib.sha256(litellm_config.encode("utf-8")).hexdigest()[:16],
            },
        ),
        (K8S / "30-service.yaml").read_text(encoding="utf-8"),
    ]
    combined = "---\n".join(manifests)
    if args.dry_run:
        # Never echo credentials, not even to a terminal. --dry-run exists to read
        # the manifests, and a script that prints secrets into shell scrollback (and
        # therefore into CI logs and pasted bug reports) is a leak with extra steps.
        print(redact_secrets(combined))
        return 0

    print("applying ...")
    # Server-side apply, and this is not a stylistic choice. Client-side `kubectl
    # apply` records the entire submitted object in a
    # `kubectl.kubernetes.io/last-applied-configuration` annotation, which is
    # capped at 256 KiB. The jev-route-code ConfigMap carries the whole package,
    # so client-side apply fails with "metadata.annotations: Too long" the moment
    # the source grows past that. Server-side apply keeps no such annotation.
    # --force-conflicts is required to take over fields a previous client-side
    # apply already owns.
    kubectl(ctx, "apply", "--server-side", "--force-conflicts", "-f", "-", input_text=combined)
    wait_rollout(ctx, ns)

    if args.skip_smoke:
        return 0
    base, key = proxy_base(ctx, ns)
    print(f"proxy: {base}")
    failures = smoke_test(base, key, hook)
    print(f"\nsmoke test failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
