"""jev-route: calibrated LLM routing you bootstrap in the cloud and end up owning.

Quick start::

    from jev_route import Router

    router = Router.from_policy_file("policies/default.yaml")
    decision = await router.route_text("Summarize this customer support thread")
    print(decision.tier, decision.model, decision.reason)

With no API key at all, that runs on the deterministic MockBackend. Set
``backend.name: jev`` and ``TYPESAFE_API_KEY`` for calibrated cloud decisions, or
``backend.name: distilled`` once you have trained a local model from your own
decision log.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .backends.base import BackendResult, DecisionBackend, DecisionRequest
from .backends.jev import JevBackend
from .backends.mock import MockBackend
from .cache import DecisionCache, InMemoryTTLCache, NullCache
from .gate import DEFAULT_DETECTORS, Detector, HardGate
from .logging_sink import DecisionSink, JsonlSink, NullSink, iter_records
from .policy import Policy, PolicyError
from .router import Router
from .schema import (
    COMPLEXITY_LEVELS,
    DOMAINS,
    SCHEMA_VERSION,
    SENSITIVITY_LEVELS,
    TIERS,
    ChoiceAnswer,
    DecisionAnswers,
    DecisionRecord,
    GateFinding,
    GateVerdict,
    NoulAnswer,
    RequestFeatures,
    RoutingDecision,
)

__all__ = [
    "COMPLEXITY_LEVELS",
    "DEFAULT_DETECTORS",
    "DOMAINS",
    "SCHEMA_VERSION",
    "SENSITIVITY_LEVELS",
    "TIERS",
    "BackendResult",
    "ChoiceAnswer",
    "DecisionAnswers",
    "DecisionBackend",
    "DecisionCache",
    "DecisionRecord",
    "DecisionRequest",
    "DecisionSink",
    "Detector",
    "GateFinding",
    "GateVerdict",
    "HardGate",
    "InMemoryTTLCache",
    "JevBackend",
    "JsonlSink",
    "MockBackend",
    "NoulAnswer",
    "NullCache",
    "NullSink",
    "Policy",
    "PolicyError",
    "RequestFeatures",
    "Router",
    "RoutingDecision",
    "__version__",
    "iter_records",
]
