"""Shared plumbing for the LiteLLM integrations.

Three integration surfaces (a routing plugin, a classifier plugin, a pre-call
hook) all need the same four things: a process-wide :class:`~jev_route.router.Router`,
a way to pull a message list out of whichever wire shape LiteLLM handed us, a way
to pull *identity* metadata out of that same payload without pulling prompt text
along with it, and a compact JSON view of a decision that is safe to stash in
LiteLLM's own structures.

This module has **no LiteLLM import**. That is deliberate: the router singleton
and the payload helpers are pure jev-route, so they are testable without LiteLLM
installed and reusable from the CLI. Only the three ``litellm_*`` modules touch
LiteLLM types, and they are the only place a version bump can break us.

Two decisions here are worth stating before the code, because both look like
over-engineering until you know what they protect:

**The router is built lazily and never raises.** LiteLLM resolves plugin dotted
paths at proxy startup, and a module-level instance that raises in its
constructor takes the whole proxy down. A router that cannot find its policy or
its API key falls back to the built-in policy on the MockBackend and logs
loudly, because a booted proxy making coarse decisions beats a proxy that will
not start -- and the fallback is the *safe* direction, not the cheap one.

**Configuration precedence is explicit, and the environment only ever moves a
file.** The policy file (``JEV_ROUTE_POLICY``, else ``./policies/default.yaml``,
else a copy of it found next to the package, else a built-in copy) owns every
routing decision: rules, tiers, gate, uncertainty bumps, failure mode, excerpt
mode. Exactly one thing may be overridden from the environment:
``JEV_ROUTE_DECISION_LOG`` redirects ``logging.path``, because where writable
storage is mounted is a property of the deployment and not of the policy. It
changes nothing else -- not ``excerpt_mode``, not ``hash_salt`` -- so an
environment variable cannot be used to start retaining prompt text without a
diff somebody reviews.

**Metadata is allowlisted, not denylisted.** ``data["metadata"]`` flows into the
logged :class:`~jev_route.schema.DecisionRecord` (the training dataset) *and*
into the :class:`~jev_route.backends.base.DecisionRequest` that may be sent to a
cloud backend. A denylist of "keys that might contain prompt text" is a list
that has to be complete forever, against a payload LiteLLM owns and changes. An
allowlist of identity keys is a list that has to be *correct* once. So: known
identity keys, scalar values only, truncated, everything else dropped.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import uuid
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..policy import Policy
from ..router import Router
from ..schema import RoutingDecision

#: Logged under the ``LiteLLM`` namespace on purpose. The proxy configures that
#: logger and its handlers; a ``jev_route.*`` logger would propagate to a root
#: that a proxy deployment may never have configured, and the operator would not
#: see why their traffic is being routed by the fallback policy. Standalone SDK
#: users get the same messages through the same hierarchy.
LOGGER = logging.getLogger("LiteLLM.jev_route.integrations")

#: Policy file location. Default: search the working directory, then the source
#: checkout this package was imported from, then the built-in fallback.
POLICY_ENV_VAR = "JEV_ROUTE_POLICY"
#: Comma-separated model names the pre-call hook is allowed to rewrite. ``*``
#: means every model. Never rewrite a model the operator did not opt in to.
MANAGED_MODELS_ENV_VAR = "JEV_ROUTE_MANAGED_MODELS"
#: Per-request budget for one routing decision, in milliseconds.
TIMEOUT_ENV_VAR = "JEV_ROUTE_TIMEOUT_MS"
#: Overrides ``logging.path`` in whichever policy the router ends up using.
DECISION_LOG_ENV_VAR = "JEV_ROUTE_DECISION_LOG"

DEFAULT_TIMEOUT_MS = 3000
DEFAULT_MANAGED_MODELS = "*"

#: The namespace jev-route writes into LiteLLM's own structures:
#: ``context.signals[SIGNALS_KEY]`` for a routing plugin and
#: ``data["metadata"][SIGNALS_KEY]`` for a proxy hook. One name in both places
#: means one grep finds every trace jev-route leaves in someone else's objects.
SIGNALS_KEY = "jev_route"

#: Call types that carry a completion-shaped payload we can route. Everything
#: else (``embedding``, ``rerank``, ``image_generation``, ``transcription``,
#: file/batch/vector-store endpoints, ...) is ignored: routing a text-completion
#: decision onto an embedding call would rewrite a model that has no chat tier
#: behind it, and there is no prompt to judge anyway.
ROUTABLE_CALL_TYPES: frozenset[str] = frozenset(
    {
        "completion",
        "acompletion",
        "text_completion",
        "atext_completion",
        "generate_content",
        "agenerate_content",
        "anthropic_messages",
        "aanthropic_messages",
        "responses",
        "aresponses",
    }
)

#: Identity keys copied out of a request's metadata. Allowlist: see the module
#: docstring. This list was written against the real key set LiteLLM 1.101 hands
#: a plugin, captured with :func:`log_metadata_keys` from a running proxy -- 51
#: keys, of which exactly these are identity.
#:
#: Deliberately NOT here, with the reason, because "why isn't X allowed" is the
#: first question a sceptical reader asks:
#:   ``user_api_key``, ``user_api_key_hash``, ``user_api_key_auth``,
#:   ``user_api_key_auth_metadata``  -- the caller's credential and its hash. A
#:       training dataset that accumulates key hashes is a breach waiting to be
#:       published, and it is in that metadata blob on every single request.
#:   ``headers``  -- carries ``Authorization``, ``Cookie`` and client IPs.
#:   ``user_api_key_metadata``, ``user_api_key_team_metadata``  -- free-form
#:       operator JSON; may hold anything, including prompt fragments.
#:   ``user_api_key_user_email``, ``requester_ip_address``, ``user_agent``
#:       -- personal data (an email, and the two classic re-identification
#:       signals). jev-route's whole position is that a routing decision does not
#:       need to retain who a person is, so it does not.
#:   ``endpoint``, ``litellm_received_at``, ``queue_time_seconds``,
#:   ``attempted_retries``, ...  -- neither identity nor useful to a policy rule.
METADATA_ALLOWLIST: tuple[str, ...] = (
    "user_id",
    "end_user",
    # The OpenAI end-user field. LiteLLM carries it at the top level of the body
    # rather than in metadata, and :func:`extract_metadata` copies it in under
    # this name; it is listed here so the allowlist stays the single authority.
    "end_user_id",
    "user_api_key_alias",
    "user_api_key_team_id",
    "user_api_key_team_alias",
    "user_api_key_org_id",
    "user_api_key_org_alias",
    "user_api_key_project_id",
    "user_api_key_project_alias",
    "user_api_key_end_user_id",
    "team_id",
    "team_alias",
    "org_id",
    "request_id",
    "litellm_call_id",
    "session_id",
    # Routing provenance: which alias the caller used and which group it resolved
    # to. Useful in a policy rule (`requested_model == "auto"`) and harmless.
    "model_group",
    "model_group_alias",
    "original_model_group",
)
#: Extra identity keys an operator may allowlist without editing this file.
#: Comma-separated. Values still have to be scalars to survive.
METADATA_EXTRA_KEYS_ENV_VAR = "JEV_ROUTE_METADATA_EXTRA_KEYS"

#: Cap on any single metadata value. Identity strings are short; anything longer
#: is either a mistake or a payload smuggled into an identity field.
MAX_METADATA_VALUE_CHARS = 256
#: Cap on list-valued metadata (``tags``), for the same reason.
MAX_METADATA_LIST_ITEMS = 32

#: Roles a normalizer maps onto the OpenAI chat-completions role vocabulary, so
#: the router sees one shape no matter which API surface produced the request.
_ROLE_ALIASES = {"model": "assistant", "bot": "assistant", "human": "user"}

#: Built-in policy used when no policy file can be found. It mirrors
#: ``policies/default.yaml`` -- the file the project documents -- with the
#: backend forced to ``mock``, because a router with no policy also has no
#: credential and must still make the *safe* decision offline. Tier model names
#: are the ones the shipped examples use; set ``JEV_ROUTE_POLICY`` to your own
#: file and they come from there instead.
BUILTIN_FALLBACK_POLICY: dict[str, Any] = {
    "version": 1,
    "backend": {"name": "mock"},
    "gate": {"on_force_local": "skip_backend"},
    "tiers": {
        "local": ["qwen38"],
        "cheap": ["qwen3.8-flash"],
        "strong": ["qwen3.8-max"],
    },
    "tier_order": ["cheap", "strong"],
    "pii_threshold": 0.5,
    "rules": [
        {
            "id": "gate.force-local",
            "if": "gate_force_local",
            "then": {"tier": "local"},
            "reason": "local hard gate matched a structured identifier or credential",
        },
        {
            "id": "data.sensitive",
            "if": 'sensitivity in ["confidential", "regulated"] or pii_present',
            "then": {"tier": "local"},
            "reason": "sensitive or personal data must not leave the infrastructure",
        },
        {
            "id": "complexity.frontier",
            "if": 'complexity == "frontier"',
            "then": {"tier": "strong"},
            "reason": "task needs frontier-class reasoning",
        },
        {
            "id": "complexity.hard",
            "if": 'complexity == "hard"',
            "then": {"tier": "strong"},
            "reason": "task needs a strong model",
        },
        {"id": "default", "then": {"tier": "cheap"}, "reason": "routine task, no sensitivity signal"},
    ],
    "on_uncertain": {
        "sensitivity_confidence_below": 0.8,
        "sensitivity_bump_levels": 1,
        "complexity_confidence_below": 0.7,
        "complexity_bump_levels": 1,
        "pii_uncertain_threshold": 0.35,
        "pii_uncertain_counts_as_present": True,
        # Mirrors policies/default.yaml: without this, an integration that loses
        # its policy file also loses the "too uncertain to egress" escalation.
        "force_local_confidence_below": 0.25,
    },
    "on_backend_down": {"mode": "fail_closed", "fail_closed_tier": "local", "fail_open_tier": "strong"},
    "cache": {"enabled": True, "ttl_seconds": 900, "max_entries": 8192},
    "logging": {"enabled": True, "path": "./decision-log/decisions.jsonl", "excerpt_mode": "hash"},
}

#: Where to look for a policy file when ``JEV_ROUTE_POLICY`` is unset.
#:
#: ``__file__`` is wrapped because it is not guaranteed to exist: a frozen or
#: zip-imported package has no meaningful one, and this is a *discovery* helper
#: only. It must never be the reason the module fails to import, because LiteLLM
#: imports it at proxy startup. Nothing else in the integrations reads the
#: filesystem relative to this module -- a deployment that mounts the package from
#: a ConfigMap, with no source tree and no installed distribution metadata around
#: it, has to work, and it does: the policy path comes from the environment or the
#: working directory, and the version string is never looked up here.
try:
    _PACKAGE_ROOT = Path(__file__).resolve().parent
except Exception:  # pragma: no cover - frozen/zip imports, or an odd mount
    _PACKAGE_ROOT = Path.cwd()


def _policy_search_paths() -> tuple[Path, ...]:
    """Candidate policy files, most specific first.

    The walk up out of the package matters for the two ways this code is
    actually run: from a source checkout (``PYTHONPATH=src``), where
    ``policies/default.yaml`` sits three levels up, and from an installed wheel
    inside ``site-packages``, where it does not exist and the built-in fallback
    takes over.
    """
    candidates = [Path.cwd() / "policies" / "default.yaml"]
    for parent in _PACKAGE_ROOT.parents:
        candidates.append(parent / "policies" / "default.yaml")
        # A flat install layout puts the package one level down from the root.
        candidates.append(parent / "jev-route" / "policies" / "default.yaml")
    seen: dict[Path, None] = {}
    for candidate in candidates:
        seen.setdefault(candidate, None)
    return tuple(seen)


def resolve_policy_path(explicit: str | os.PathLike[str] | None = None) -> Path | None:
    """First existing policy file, or ``None`` when there is nowhere to look.

    An explicit path (or ``JEV_ROUTE_POLICY``) that does not exist is *not*
    silently replaced by a discovered one: an operator who named a file meant
    that file, and routing on a different policy than the one they reviewed is
    worse than failing loudly. Returning ``None`` lets the caller decide, and
    :func:`build_router` logs the exact path it could not find.
    """
    named = explicit if explicit is not None else os.environ.get(POLICY_ENV_VAR)
    if named:
        path = Path(str(named)).expanduser()
        return path if path.exists() else None
    for candidate in _policy_search_paths():
        if candidate.is_file():
            return candidate
    return None


# --------------------------------------------------------------------------- #
# Router singleton
# --------------------------------------------------------------------------- #
_ROUTER: Router | None = None
_ROUTER_LOCK = threading.Lock()


def decision_log_override() -> str | None:
    """The ``JEV_ROUTE_DECISION_LOG`` value, when the operator set one."""
    value = os.environ.get(DECISION_LOG_ENV_VAR, "")
    return value.strip() or None


def apply_decision_log_override(policy: Policy) -> Policy:
    """Point a policy's decision log at ``JEV_ROUTE_DECISION_LOG``, if it is set.

    Precedence, and the reasoning behind an environment variable outranking a
    versioned config file:

    1. ``JEV_ROUTE_DECISION_LOG`` wins, because it names *where writable storage
       is mounted*, which is a property of the deployment and not of the routing
       policy. A Kubernetes manifest knows it has an ``emptyDir`` at
       ``/var/log/jev-route``; the policy file in git does not, and must not.
    2. ``logging.path`` from the policy file, everywhere else.

    The override touches ``logging.path`` and nothing else: ``excerpt_mode``,
    ``hash_salt`` and rotation stay the policy's, because those are privacy and
    dataset decisions an operator reviews in a pull request, and letting an
    environment variable change them would create a way to start retaining prompt
    text without a diff anyone reads. That asymmetry is the whole argument for
    allowing the override at all -- it moves a file, it does not change a policy.
    """
    override = decision_log_override()
    if not override:
        return policy
    logging_config = {**dict(policy.logging or {}), "path": override}
    return policy.with_overrides(logging=logging_config)


def load_policy(policy_path: str | os.PathLike[str] | None = None, *, strict: bool = False) -> Policy:
    """Resolve and parse the policy, degrading to the built-in one instead of raising.

    ``strict=True`` propagates every error and is what the CLI and the tests use.
    ``strict=False`` is what an integration entry point uses, because LiteLLM
    resolves plugin dotted paths at proxy startup and a constructor that raises
    there takes the whole proxy down.
    """
    path = Path(str(policy_path)) if policy_path else resolve_policy_path(policy_path)
    if path is None or not path.is_file():
        wanted = policy_path or os.environ.get(POLICY_ENV_VAR)
        if strict:
            raise FileNotFoundError(
                f"jev-route policy not found: {wanted or '(no path given)'}; searched "
                + ", ".join(str(p) for p in _policy_search_paths()[:3])
            )
        LOGGER.error(
            "jev-route: no policy file%s; using the built-in fallback policy on MockBackend. "
            "Set %s=/path/to/policy.yaml to route on your own rules.",
            f" at {wanted!r}" if wanted else " (set JEV_ROUTE_POLICY or run from a checkout)",
            POLICY_ENV_VAR,
        )
        return apply_decision_log_override(Policy.from_dict(BUILTIN_FALLBACK_POLICY, source="<builtin-fallback>"))
    if strict:
        return apply_decision_log_override(Policy.from_file(path))
    try:
        return apply_decision_log_override(Policy.from_file(path))
    except Exception as exc:  # broad except, deliberately: an unparseable policy must not stop the process
        LOGGER.error(
            "jev-route: policy %s could not be parsed (%s: %s); using the built-in fallback policy "
            "on MockBackend. Fix the file or set %s.",
            path,
            type(exc).__name__,
            exc,
            POLICY_ENV_VAR,
        )
        return apply_decision_log_override(Policy.from_dict(BUILTIN_FALLBACK_POLICY, source="<builtin-fallback>"))


def build_router(policy_path: str | os.PathLike[str] | None = None, *, strict: bool = False) -> Router:
    """Construct a :class:`Router`, degrading instead of raising unless ``strict``.

    Two independent downgrades, each logged, because they answer different
    questions:

    1. **No readable policy** -> the built-in fallback policy (see
       :func:`load_policy`).
    2. **A policy whose backend cannot be constructed** -- no ``TYPESAFE_API_KEY``
       for ``backend.name: jev`` is the common one, and the backend refuses to
       start without a key rather than quietly stop classifying -> *the same
       policy*, on MockBackend. Keeping the operator's rules, tiers, gate and
       logging config and changing only the source of the judgements is the
       smallest possible downgrade, and it is the one that preserves the
       invariant that matters: sensitive data still routes local.
    """
    policy = load_policy(policy_path, strict=strict)
    from ..backends import build_backend  # deferred import: deferred: avoids an import cycle

    if strict:
        return Router(policy, build_backend(policy))
    try:
        return Router(policy, build_backend(policy))
    except Exception as exc:  # broad except, deliberately: startup must not take the proxy down
        LOGGER.error(
            "jev-route: the %s decision backend could not be built (%s: %s); falling back to "
            "MockBackend with the same policy. Decisions will be coarse but safe.",
            (policy.backend or {}).get("name", "configured"),
            type(exc).__name__,
            exc,
        )
        return Router(policy.with_backend({"name": "mock"}), _mock_backend())


def _mock_backend() -> Any:
    from ..backends.mock import MockBackend  # deferred import: local import keeps this module import-light

    return MockBackend()


def get_router(*, strict: bool = False) -> Router:
    """The process-wide router, built on first use.

    One router per process, not one per request: the router owns the decision
    cache, the circuit breaker, and the append-only log handle. Building it per
    request would defeat all three and produce an interleaved decision log.
    """
    global _ROUTER  # module-level singleton by design: module singleton by design
    if _ROUTER is not None:
        return _ROUTER
    with _ROUTER_LOCK:
        if _ROUTER is None:
            _ROUTER = build_router(strict=strict)
    return _ROUTER


def configure_router(
    policy_path: str | os.PathLike[str] | None = None,
    *,
    router: Router | None = None,
    strict: bool = False,
) -> Router:
    """Install the process-wide router explicitly.

    Call this before the first request -- at proxy startup, or in a test fixture
    -- to pin the policy instead of relying on discovery. Passing ``router=``
    installs an already-built router, which is how the tests inject a
    MockBackend and a temporary JSONL sink without touching the environment.
    """
    global _ROUTER  # module-level singleton by design
    with _ROUTER_LOCK:
        _ROUTER = router if router is not None else build_router(policy_path, strict=strict)
        return _ROUTER


def reset_router() -> None:
    """Drop the singleton. Test-only: it deliberately does not close the router."""
    global _ROUTER  # module-level singleton by design
    with _ROUTER_LOCK:
        _ROUTER = None


async def aclose_router() -> None:
    """Close the singleton's backend and log handle. Safe to call twice."""
    global _ROUTER  # module-level singleton by design
    with _ROUTER_LOCK:
        router, _ROUTER = _ROUTER, None
    if router is not None:
        await router.aclose()


def timeout_ms(default: int = DEFAULT_TIMEOUT_MS) -> float:
    """The per-request routing budget, from ``JEV_ROUTE_TIMEOUT_MS``."""
    raw = os.environ.get(TIMEOUT_ENV_VAR)
    if not raw:
        return float(default)
    try:
        return max(0.0, float(raw))
    except ValueError:
        LOGGER.warning("jev-route: %s=%r is not a number; using %dms.", TIMEOUT_ENV_VAR, raw, default)
        return float(default)


# --------------------------------------------------------------------------- #
# Managed-model opt-in
# --------------------------------------------------------------------------- #
def parse_managed_models(raw: str | Sequence[str] | None = None) -> frozenset[str]:
    """Parse ``JEV_ROUTE_MANAGED_MODELS`` into a set of names, or the ``*`` wildcard."""
    source = raw if raw is not None else os.environ.get(MANAGED_MODELS_ENV_VAR, DEFAULT_MANAGED_MODELS)
    items = source if isinstance(source, str) else ",".join(source)
    names = frozenset(part.strip() for part in items.split(",") if part.strip())
    return names or frozenset({DEFAULT_MANAGED_MODELS})


def is_managed(model: Any, managed: Iterable[str]) -> bool:
    """Whether ``model`` is one this integration is allowed to rewrite."""
    allowed = frozenset(managed)
    if "*" in allowed:
        return True
    return isinstance(model, str) and model in allowed


# --------------------------------------------------------------------------- #
# Payload extraction
# --------------------------------------------------------------------------- #
def extract_messages(data: Mapping[str, Any] | None, call_type: str = "") -> list[dict[str, Any]]:  # noqa: ARG001
    """Normalize any LiteLLM request payload to an OpenAI chat-completions list.

    The shapes this copes with, in the order they are tried, because a payload
    can carry more than one key and the first is not always the right one:

    ``messages``
        Chat completions, and Anthropic ``/v1/messages`` (which adds a separate
        top-level ``system`` that is folded in as a system message).
    ``input`` (+ ``instructions``)
        The Responses API, where ``input`` is either a bare string or a list of
        typed items.
    ``prompt``
        Text completions: a string, or a list of tokens/strings.
    ``contents``
        Gemini ``generate_content``: ``parts`` with ``text``, roles ``user``/``model``.

    Returns ``[]`` when nothing routable is present. An empty list is a valid
    answer, not an error: the caller routes it, the excerpt is empty, the local
    gate finds nothing, and the backend answers at maximum uncertainty -- which
    the policy engine then escalates. Failing closed on a payload we could not
    parse is the correct behaviour and needs no special case here.

    ``call_type`` is accepted for symmetry with the hook signature and for
    callers that want to log it; the payload shape is detected from the data
    itself, because the same shape reaches us under both the sync and async
    spelling of a call type (``completion`` / ``acompletion``) and because
    trusting a caller-supplied label over the payload would let a mismatch route
    nothing at all.
    """
    if not isinstance(data, Mapping):
        return []

    out: list[dict[str, Any]] = []
    # Anthropic carries the system prompt at the top level; the Responses API
    # calls the same thing `instructions`. A payload never means both.
    system_text = _flatten_blocks(data.get("system")) or _flatten_blocks(data.get("instructions"))
    if system_text:
        out.append({"role": "system", "content": system_text})

    for key, converter in _PAYLOAD_SHAPES:
        turns = converter(data.get(key))
        if turns:
            out.extend(turns)
            return out
    return out


def _from_messages(value: Any) -> list[dict[str, Any]]:
    """Chat-completions / Anthropic ``messages``."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [m for m in (_normalize_message(item) for item in value) if m is not None]


def _from_input(value: Any) -> list[dict[str, Any]]:
    """Responses API ``input``: a bare string, or a list of typed items."""
    if isinstance(value, str):
        return [{"role": "user", "content": value}] if value.strip() else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [m for m in (_normalize_message(item) for item in value) if m is not None]
    return []


def _from_prompt(value: Any) -> list[dict[str, Any]]:
    """Text-completion ``prompt``: a string, or a list of pieces/tokens."""
    if isinstance(value, str):
        return [{"role": "user", "content": value}] if value.strip() else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        # A tokenized prompt has no text to judge. Joining the string parts is
        # the honest best effort; ints are skipped rather than stringified,
        # because a token id is not a word and pretending otherwise would put
        # invented text into the excerpt the gate scans.
        joined = " ".join(part for part in value if isinstance(part, str))
        return [{"role": "user", "content": joined}] if joined.strip() else []
    return []


def _from_contents(value: Any) -> list[dict[str, Any]]:
    """Gemini ``contents``: ``[{"role": "user", "parts": [{"text": ...}]}]``."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [m for m in (_normalize_gemini_content(item) for item in value) if m is not None]


#: Tried in order; the first that yields turns wins. Chat completions first
#: because it is the overwhelmingly common shape and the cheapest to test.
_PAYLOAD_SHAPES: tuple[tuple[str, Any], ...] = (
    ("messages", _from_messages),
    ("input", _from_input),
    ("prompt", _from_prompt),
    ("contents", _from_contents),
)


def extract_prompt_text(data: Mapping[str, Any] | None, call_type: str = "") -> str | None:  # noqa: ARG001
    """The bare prompt string of a text-completion payload, else ``None``.

    Exists so the hook can call :meth:`Router.route_text` for ``/v1/completions``
    instead of wrapping the prompt in a synthetic user message. The excerpt is
    nearly identical either way, but ``n_messages`` and ``n_prior_turns`` are
    features in the training set, and a request that carried one prompt should
    not be logged as a request that carried a conversation.
    """
    if not isinstance(data, Mapping):
        return None
    if data.get("messages") or data.get("input") or data.get("contents"):
        return None
    turns = _from_prompt(data.get("prompt"))
    if len(turns) == 1 and isinstance(turns[0].get("content"), str):
        return turns[0]["content"]
    return None


def _normalize_message(item: Any) -> dict[str, Any] | None:
    """One message from any dialect, or ``None`` when it carries nothing at all."""
    if not isinstance(item, Mapping):
        return None
    role = str(item.get("role") or item.get("type") or "").strip().lower()
    content = item.get("content")
    if content is None:
        # Responses API items put text in `text` and tool output in `output`.
        content = item.get("text") if item.get("text") is not None else item.get("output")
    if not role and content is None:
        return None
    if role in ("", "message"):
        role = "user"
    role = _ROLE_ALIASES.get(role, role)
    # An empty-content tool call still becomes a message: prompts.compute_features
    # counts the role, and "this request has tool output" is a real routing
    # feature even when the output itself is not excerpted.
    return {"role": role, "content": "" if content is None else content}


def _normalize_gemini_content(item: Any) -> dict[str, Any] | None:
    """A Gemini ``contents`` entry (``{"role": ..., "parts": [{"text": ...}]}``)."""
    if not isinstance(item, Mapping):
        return None
    parts = item.get("parts")
    if not isinstance(parts, Sequence) or isinstance(parts, (str, bytes)):
        return None
    text = "\n".join(str(part["text"]) for part in parts if isinstance(part, Mapping) and part.get("text") is not None)
    if not text.strip() and not item.get("role"):
        return None
    role = str(item.get("role") or "user").strip().lower()
    return {"role": _ROLE_ALIASES.get(role, role), "content": text}


def _flatten_blocks(value: Any) -> str:
    """Flatten Anthropic-style ``system`` blocks (a string, or a list of text blocks)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        text = value.get("text")
        return str(text) if text is not None else ""
    if isinstance(value, Sequence):
        return "\n".join(part for part in (_flatten_blocks(item) for item in value) if part)
    return ""


# --------------------------------------------------------------------------- #
# Metadata extraction
# --------------------------------------------------------------------------- #
def metadata_allowlist() -> frozenset[str]:
    """The identity keys we copy, plus anything ``JEV_ROUTE_METADATA_EXTRA_KEYS`` adds."""
    extra = os.environ.get(METADATA_EXTRA_KEYS_ENV_VAR, "")
    return frozenset(METADATA_ALLOWLIST) | frozenset(p.strip() for p in extra.split(",") if p.strip())


def extract_metadata(data: Mapping[str, Any] | None) -> dict[str, Any]:
    """Caller identity from a LiteLLM *request payload*, with nothing else.

    Thin wrapper over :func:`filter_metadata` that knows where LiteLLM puts
    things: identity in ``data["metadata"]``, plus the OpenAI ``user`` field at
    the top level of the body.
    """
    if not isinstance(data, Mapping):
        return {}
    raw = data.get("metadata")
    raw = dict(raw) if isinstance(raw, Mapping) else {}
    # The OpenAI end-user field sits at the top level of the body, not in
    # metadata. It is identity, not payload, so it is worth keeping.
    if "end_user" not in raw and isinstance(data.get("user"), str):
        raw["end_user"] = data["user"]
    return filter_metadata(raw)


def filter_metadata(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Reduce a metadata mapping to allowlisted identity, filtered twice.

    The result goes into the decision record (the training dataset) and into the
    :class:`~jev_route.backends.base.DecisionRequest` that a cloud backend may
    see, so it is filtered once by allowlist and once by shape: only scalars and
    short lists of scalars survive, and strings are truncated. ``tags`` is
    handled specially because it is the one list operators actually write policy
    rules against, and it is caller-supplied -- so it is capped in both item
    count and length rather than trusted.
    """
    if not isinstance(raw, Mapping):
        return {}
    allowed = metadata_allowlist()

    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in allowed:
            continue
        sanitized = _sanitize_metadata_value(value)
        if sanitized is not None:
            out[str(key)] = sanitized

    # `tags` is handled outside the loop above on purpose: it is list-valued, so
    # the scalar filter would drop it, but `"prod" in tags` is a policy rule
    # operators write. Cap it rather than trust it.
    # LiteLLM spells the caller's tags `caller_tags` in a router/plugin context
    # and `tags` in a proxy body; accept both, emit one name, so a policy rule
    # reads `"prod" in tags` whichever surface the request came through.
    tags = raw.get("tags")
    if tags is None:
        tags = raw.get("caller_tags")
    if isinstance(tags, Sequence) and not isinstance(tags, (str, bytes)):
        clean = [
            str(tag)[:MAX_METADATA_VALUE_CHARS]
            for tag in list(tags)[:MAX_METADATA_LIST_ITEMS]
            if isinstance(tag, (str, int, float, bool))
        ]
        if clean:
            out["tags"] = clean
    return out


def _sanitize_metadata_value(value: Any) -> Any:
    """Scalars only, strings truncated. ``None`` means "drop this key"."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        return text[:MAX_METADATA_VALUE_CHARS]
    return None


def log_metadata_keys(where: str, metadata: Mapping[str, Any] | None) -> None:
    """DEBUG-log the *keys* present in a metadata mapping, never their values.

    Exists because the one thing an operator cannot work out from documentation is
    which identity keys LiteLLM actually put in the payload at the point jev-route
    sees it -- it differs between the proxy's pre-call hook (a full body, with
    ``litellm_call_id``), the routing-plugin pipeline (the router's metadata
    variable) and the classifier plugin (a smaller snapshot). Key names are not
    sensitive; values are. So this logs the shape and nothing else, and it is how
    you find out why a decision record did not get the join key you expected.
    """
    if not LOGGER.isEnabledFor(logging.DEBUG):
        return
    keys = sorted(str(k) for k in metadata) if isinstance(metadata, Mapping) else []
    LOGGER.debug("jev-route: %s metadata keys: %s", where, keys or "<none>")


def request_id_for(data: Mapping[str, Any] | None, metadata: Mapping[str, Any] | None = None) -> str:
    """A decision id that LiteLLM's own logs can be joined back to.

    Reusing ``litellm_call_id`` (falling back to the proxy's ``request_id``) is
    what makes the decision log and LiteLLM's spend log two views of one event
    instead of two unrelated files. Without a shared key, "which decisions
    belonged to the requests that were slow" is unanswerable, and answering it is
    half the reason to keep the log.
    """
    sources: list[Any] = [metadata, data.get("metadata") if isinstance(data, Mapping) else None, data]
    # One level of nesting, because LiteLLM sometimes hands a router a metadata
    # dict that *contains* its own metadata blob (`litellm_metadata`,
    # `requester_metadata`). Deeper than that stops being a lookup and starts
    # being a search, and a wrong id is worse than a generated one.
    for source in list(sources):
        if isinstance(source, Mapping):
            sources.extend(source.get(key) for key in ("litellm_metadata", "metadata", "requester_metadata"))
    for source in sources:
        if isinstance(source, Mapping):
            for key in ("litellm_call_id", "request_id"):
                value = source.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()[:MAX_METADATA_VALUE_CHARS]
    return uuid.uuid4().hex


# --------------------------------------------------------------------------- #
# One routing decision, with a budget
# --------------------------------------------------------------------------- #
async def decide(
    router: Router,
    *,
    messages: Sequence[Mapping[str, Any]] | None = None,
    prompt: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    requested_model: str | None = None,
    request_id: str | None = None,
    timeout_ms: float | None = None,
) -> RoutingDecision | None:
    """:func:`decide_with_reason`, discarding the reason.

    The three shipped integrations all use the two-value form, because each has
    somewhere to put a reason (``context.signals``, the spend-log metadata, a
    log line). This one-argument-shorter form exists for operators writing their
    own integration against :func:`decide_with_reason`'s semantics without having
    to unpack a tuple they will not read.
    """
    decision, _reason = await decide_with_reason(
        router,
        messages=messages,
        prompt=prompt,
        metadata=metadata,
        requested_model=requested_model,
        request_id=request_id,
        timeout_ms=timeout_ms,
    )
    return decision


async def decide_with_reason(
    router: Router,
    *,
    messages: Sequence[Mapping[str, Any]] | None = None,
    prompt: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    requested_model: str | None = None,
    request_id: str | None = None,
    timeout_ms: float | None = None,
) -> tuple[RoutingDecision | None, str | None]:
    """Run one routing decision under a budget.

    Returns ``(decision, reason)``. ``decision is None`` means "no decision", and
    ``reason`` says why in one line -- a string an operator can read in LiteLLM's
    spend-log metadata instead of having to go and grep the proxy's stderr for the
    matching warning.

    All three integrations call this instead of the router directly, so that the
    timeout and error semantics exist once and are argued once:

    * **``None`` on timeout, on cancellation, and on any internal error.** The
      router itself fails *closed* -- a backend outage produces a decision for
      the safest tier. What ``None`` covers is the case where jev-route produced
      no decision at all, and there the caller must fall back to LiteLLM's own
      behaviour rather than invent a tier. Failing open *at the integration
      layer* is correct precisely because the decision layer already failed
      closed; the two layers are not redundant, they cover different failures.
    * **A budget exists because LiteLLM only gives one to classifier plugins.**
      ``Router(plugins=[...])`` awaits a routing plugin with no timeout at all,
      and a pre-call hook has none either, so a hung decision backend would hang
      the request. The router's own circuit breaker usually prevents that; this
      is the backstop for the case where it has not tripped yet.
    * **Cancelling a decision can lose its log record.** The budget is therefore
      generous by default (3 s) and the decision cache plus circuit breaker make
      a slow backend rare. That tradeoff is stated here rather than discovered in
      a gap in someone's dataset.
    """
    budget = (timeout_ms if timeout_ms is not None else DEFAULT_TIMEOUT_MS) / 1000.0
    if messages is not None:
        coroutine = router.route_messages(
            messages, metadata=metadata, requested_model=requested_model, request_id=request_id
        )
    else:
        coroutine = router.route_text(
            prompt or "", metadata=metadata, requested_model=requested_model, request_id=request_id
        )
    if budget <= 0:
        try:
            return await coroutine, None
        except Exception as exc:  # broad except, deliberately
            LOGGER.warning("jev-route: routing decision failed (%s: %s); declining to route.", type(exc).__name__, exc)
            return None, f"routing decision failed: {type(exc).__name__}: {exc}"
    try:
        return await asyncio.wait_for(coroutine, timeout=budget), None
    except (TimeoutError, asyncio.TimeoutError):  # the alias differs before 3.11
        LOGGER.warning("jev-route: routing decision exceeded %.0fms; declining to route this request.", budget * 1000)
        return None, f"routing decision exceeded {budget * 1000:.0f}ms"
    except Exception as exc:  # broad except, deliberately: an integration must not fail the caller's request
        LOGGER.warning("jev-route: routing decision failed (%s: %s); declining to route.", type(exc).__name__, exc)
        return None, f"routing decision failed: {type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Decision -> compact JSON
# --------------------------------------------------------------------------- #
def decision_signals(
    decision: RoutingDecision,
    *,
    request_id: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A small JSON-serializable summary of a decision.

    Used for LiteLLM's ``context.signals``, the proxy's spend-log metadata, and
    the response headers an operator might want. It is deliberately *not* the
    full record: the soft probability distributions belong in the JSONL decision
    log and nowhere else, because copying a 4-question distribution into
    LiteLLM's per-request metadata would put it in every downstream observability
    system too. What stays here is enough to answer "why did this route there"
    from a spend log, plus the id that joins to the full record.
    """
    answers = decision.answers
    signals: dict[str, Any] = {
        "tier": decision.tier,
        "model": decision.model,
        "rule_id": decision.rule_id,
        # The outcome verifier needs this: gate-blocked content is never sent
        # anywhere, including to a verification backend.
        "gate_blocked": bool(decision.gate.blocks_backend or decision.gate.force_local),
        "reason": decision.reason,
        "complexity": answers.complexity.choice,
        "complexity_confidence": round(float(answers.complexity.confidence), 4),
        "sensitivity": answers.sensitivity.choice,
        "sensitivity_confidence": round(float(answers.sensitivity.confidence), 4),
        "domain": answers.domain.choice,
        "pii": round(float(answers.pii.value), 4),
        "escalated": list(decision.escalated),
        "gate_force_local": bool(decision.gate.force_local),
        "gate_blocks_backend": bool(decision.gate.blocks_backend),
        "degraded": bool(decision.degraded),
        "backend": decision.backend,
        "latency_ms": round(float(decision.latency_ms), 3),
    }
    if request_id:
        signals["request_id"] = request_id
    if extra:
        signals.update({str(k): v for k, v in extra.items()})
    return signals


__all__ = [
    "BUILTIN_FALLBACK_POLICY",
    "DECISION_LOG_ENV_VAR",
    "DEFAULT_TIMEOUT_MS",
    "LOGGER",
    "MANAGED_MODELS_ENV_VAR",
    "METADATA_ALLOWLIST",
    "METADATA_EXTRA_KEYS_ENV_VAR",
    "POLICY_ENV_VAR",
    "ROUTABLE_CALL_TYPES",
    "SIGNALS_KEY",
    "TIMEOUT_ENV_VAR",
    "aclose_router",
    "apply_decision_log_override",
    "build_router",
    "configure_router",
    "decide",
    "decide_with_reason",
    "decision_log_override",
    "decision_signals",
    "extract_messages",
    "extract_metadata",
    "extract_prompt_text",
    "filter_metadata",
    "get_router",
    "is_managed",
    "load_policy",
    "log_metadata_keys",
    "parse_managed_models",
    "request_id_for",
    "reset_router",
    "resolve_policy_path",
    "timeout_ms",
]
