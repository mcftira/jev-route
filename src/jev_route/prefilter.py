"""Deterministic tier pre-filter: rule out what could never serve the request,
BEFORE any DecisionBackend call.

Adapted from the Jevonian pattern (code filters candidates before any model
call) to jev-route's shape: the "candidates" here are the routing TIERS
(local/cheap/strong). Every check is a pure function of the request's already
computed features, the hard-gate verdict, the semantic assessment, and policy
data. Nothing in this module talks to a model.

Three outcomes:
* **0 tiers left** -> fail closed to the local fallback tier.
* **exactly 1 tier left** -> the degenerate case: routing is free. The router
  skips the backend call entirely and logs the decision with
  ``backend: "prefilter"`` so the distill pipeline sees it like any other
  decision (deterministic labels are high-confidence training targets).
* **2+ tiers left** -> the backend decides as usual.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: Which tiers count as "cloud" for the sensitivity floor. Local is the only
#: tier that may receive content the gate refuses to let leave the machine.
CLOUD_TIERS: tuple[str, ...] = ("cheap", "strong")

#: Default per-tier limits, used when the policy's `prefilter:` block does not
#: override them. `max_tokens` is the rough context ceiling for the routing
#: decision's purposes (a conservative estimate, not the vendor number).
DEFAULT_TIER_LIMITS: dict[str, dict[str, Any]] = {
    "local": {"max_tokens": 8_192, "capabilities": frozenset()},
    "cheap": {"max_tokens": 131_072, "capabilities": frozenset({"json"})},
    "strong": {"max_tokens": 1_000_000, "capabilities": frozenset({"json", "tools", "vision"})},
}

#: Estimated tokens per character of excerpt for the routing estimate. Four
#: chars per token is the standard English approximation and intentionally
#: generous for code (which tokenizes denser).
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class PrefilterResult:
    surviving: tuple[str, ...]
    removed: tuple[tuple[str, str], ...]  # (tier, reason) -- the audit trail
    single_candidate: bool
    fallback_tier: str


def estimate_tokens(excerpt: str) -> int:
    return max(1, len(excerpt) // CHARS_PER_TOKEN + 1)


def _limits(config: Mapping[str, Any], tier: str) -> Mapping[str, Any]:
    override = (config.get("tier_limits") or {}).get(tier) or {}
    base = DEFAULT_TIER_LIMITS.get(tier, {"max_tokens": 0, "capabilities": frozenset()})
    merged = dict(base)
    merged.update(override)
    return merged


def filter_tiers(
    *,
    tiers: Sequence[str],
    excerpt: str,
    features: Any,
    verdict: Any,
    assessment: Any | None = None,
    config: Mapping[str, Any] | None = None,
) -> PrefilterResult:
    """Remove every tier that deterministically cannot serve this request.

    ``config`` is the policy's ``prefilter:`` block (``policy.raw.get("prefilter", {})``).
    ``verdict`` is the hard-gate verdict; ``assessment`` the semantic layer's.
    """
    config = config or {}
    allow = {str(t) for t in config.get("allow_tiers", tiers)}
    deny = {str(t) for t in config.get("deny_tiers", [])}
    quota_exhausted = {str(t) for t in config.get("quota_exhausted", [])}
    fallback_tier = str(config.get("fallback_tier", "local"))
    # `features` is a RequestFeatures dataclass, not a Mapping: capability
    # requirements live in policy config or as an optional attribute.
    required = set(config.get("required_capabilities", []))
    if isinstance(features, Mapping):
        required |= set(features.get("required_capabilities", ()) or ())
    else:
        required |= set(getattr(features, "required_capabilities", ()) or ())

    removed: list[tuple[str, str]] = []
    surviving: list[str] = []

    blocks_cloud = bool(getattr(verdict, "blocks_backend", False) or getattr(verdict, "force_local", False))
    semantic_local = bool(assessment is not None and getattr(assessment, "force_local", False))
    tokens = estimate_tokens(excerpt)

    for tier in tiers:
        limits = _limits(config, tier)
        if tier not in allow:
            removed.append((tier, "not in allow_tiers"))
        elif tier in deny:
            removed.append((tier, "in deny_tiers"))
        elif tier in quota_exhausted:
            removed.append((tier, "quota exhausted"))
        elif blocks_cloud and tier in CLOUD_TIERS:
            # The hard gate refused egress; the backend never even sees these.
            removed.append((tier, "sensitivity floor (gate blocks cloud)"))
        elif semantic_local and tier in CLOUD_TIERS:
            removed.append((tier, "sensitivity floor (semantic layer forces local)"))
        elif limits.get("max_tokens", 0) and tokens > int(limits["max_tokens"]):
            removed.append((tier, f"context window < ~{tokens} tokens"))
        elif required and not required.issubset(set(limits.get("capabilities", frozenset()))):
            missing = sorted(required - set(limits.get("capabilities", frozenset())))
            removed.append((tier, f"missing capabilities: {missing}"))
        else:
            surviving.append(tier)

    if not surviving:
        return PrefilterResult((fallback_tier,), tuple(removed), True, fallback_tier)
    return PrefilterResult(tuple(surviving), tuple(removed), len(surviving) == 1, fallback_tier)
