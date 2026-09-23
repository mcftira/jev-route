"""Decision backends, and the factory that picks one from policy config.

The factory lives here rather than in :mod:`jev_route.router` so that the router
stays ignorant of which backends exist. Adding a backend means adding a branch in
:func:`build_backend` and nothing else.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..policy import Policy
from .base import (
    BackendError,
    BackendResult,
    CircuitBreaker,
    DecisionBackend,
    DecisionRequest,
)
from .jev import DEFAULT_API_URL, DEFAULT_MODEL, DEFAULT_TIMEOUT_SECONDS, JevBackend
from .mock import MockBackend


def _compiled_question_overrides(policy: Policy | Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Load the GEPA artifact when the policy selects ``questions: compiled``.

    v0.4: adopted only when the gate has passed. The file's provenance header is
    comment lines; the YAML body is the questions map. A missing or unparsable
    artifact is a loud failure, never a silent fallback to handwritten -- an
    operator who selects compiled must get compiled or an error.
    """
    if not isinstance(policy, Policy) or policy.questions != "compiled":
        return None
    import yaml

    artifact = Path(policy.compiled_questions_path)
    if not artifact.exists():
        raise BackendError(
            f"policy.questions is 'compiled' but {artifact} does not exist; "
            "run `jev-route optimize-questions` or select 'handwritten'"
        )
    overrides = yaml.safe_load(artifact.read_text())
    if not isinstance(overrides, dict):
        raise BackendError(f"{artifact} did not parse to a questions mapping")
    return overrides


def build_backend(policy: Policy | Mapping[str, Any] | None = None, **overrides: Any) -> DecisionBackend:
    """Construct the backend named by ``backend.name`` in policy config.

    ``overrides`` are passed through to the backend constructor, which is how the
    CLI and tests inject a transport or an artifact path without inventing a
    second configuration mechanism.
    """
    cfg: Mapping[str, Any] = {}
    if isinstance(policy, Policy):
        cfg = policy.backend or {}
    elif isinstance(policy, Mapping):
        cfg = policy.get("backend", policy) or {}
    cfg = {**cfg, **overrides}

    name = str(cfg.get("name", "mock")).lower()
    if name == "mock":
        return MockBackend(temperature=float(cfg.get("temperature", 0.55)))
    if name == "jev":
        return JevBackend(
            api_key=cfg.get("api_key") or os.environ.get(str(cfg.get("api_key_env", "TYPESAFE_API_KEY")), ""),
            # Defaults are imported from .jev rather than repeated here. The whole
            # point of confining the cloud dependency to one module is that deleting
            # that module removes egress; a second copy of the endpoint literal in
            # the factory quietly defeats it. Do not spell the hostname out in this
            # file, not even in a comment -- the invariant is checked by searching
            # every source file for it, comments included.
            api_url=str(cfg.get("api_url", DEFAULT_API_URL)),
            model=str(cfg.get("model", DEFAULT_MODEL)),
            timeout_seconds=float(cfg.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)),
            max_retries=int(cfg.get("max_retries", 2)),
            include_domain=bool(cfg.get("include_domain", True)),
            # v0.5 air-gapped mode: the same backend, pointed at a local
            # OpenAI-compatible systemone server, with no cloud credential
            # read or required. The flag, not the URL, decides whether a
            # missing key is a startup error.
            airgapped=bool(cfg.get("airgapped", False)),
            question_overrides=_compiled_question_overrides(policy),
        )
    if name == "distilled":
        # Imported lazily: the distilled backend needs the artifact loader and
        # optionally numpy, and a deployment using Jev should not pay for either.
        from .distilled import DistilledBackend

        return DistilledBackend(
            artifact=str(cfg.get("artifact", "./artifacts/jev-route-distilled")),
            feature_mode=bool(cfg.get("feature_mode", False)),
        )
    if name == "laya":
        # Imported lazily: the laya backend needs a local Laya checkpoint (and,
        # for the real path, torch/transformers/laya). A deployment on Jev or
        # the mock should not pay for any of it -- the same reason `distilled`
        # above is lazy. The factory only resolves the calibration artifact and
        # passes config through; loading the weights happens on first decide.
        from .laya import LayaBackend
        from .laya_calibration import load_artifact

        calibration = cfg.get("calibration")
        artifact = None
        if isinstance(calibration, Mapping) and calibration.get("artifact"):
            artifact = load_artifact(str(calibration["artifact"]))
        elif isinstance(calibration, (str, bytes)):
            artifact = load_artifact(calibration)
        return LayaBackend(
            model=str(cfg.get("model", "english")),
            model_path=cfg.get("model_path"),
            device=str(cfg.get("device", "auto")),
            token_budget=int(cfg.get("token_budget", 448)),
            max_len=int(cfg.get("max_len", 512)),
            head_max_len=int(cfg.get("head_max_len", 192)),
            role=str(cfg.get("role", "enforce")),
            calibration=artifact,
            include_domain=bool(cfg.get("include_domain", True)),
        )
    if name == "shadow":
        from .shadow import ShadowBackend

        # A shadow backend without its two sides is a config error, not a KeyError:
        # a bare key error out of Router construction names the index, not the
        # fix. Name the missing keys the way every other config error here does.
        missing = [side for side in ("primary", "shadow") if not isinstance(cfg.get(side), Mapping)]
        if missing:
            raise BackendError(
                f"backend.name: shadow requires {', '.join(missing)} side(s); each needs at least "
                f"'name'. Got: {sorted(cfg.keys())}"
            )
        return ShadowBackend(
            primary=build_backend({"backend": {**cfg.get("primary", {}), "name": str(cfg["primary"]["name"])}}),
            shadow=build_backend({"backend": {**cfg.get("shadow", {}), "name": str(cfg["shadow"]["name"])}}),
            log_disagreements=bool(cfg.get("log_disagreements", True)),
            shadow_timeout_seconds=float(cfg.get("shadow_timeout_seconds", 5.0)),
        )
    raise BackendError(f"unknown backend {name!r}; expected one of: mock, jev, distilled, laya, shadow")


__all__ = [
    "BackendError",
    "BackendResult",
    "CircuitBreaker",
    "DecisionBackend",
    "DecisionRequest",
    "JevBackend",
    "MockBackend",
    "build_backend",
]
