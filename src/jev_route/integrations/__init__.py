"""LiteLLM integrations for jev-route.

Three ways to put jev-route in front of LiteLLM, in increasing order of coupling
to LiteLLM's own routing machinery:

``litellm_hook.JevRoutePreCallHook``
    Proxy pre-call hook. Rewrites ``data["model"]`` before dispatch. No router
    plugins, no complexity router, works on any proxy version with
    ``async_pre_call_hook``. The lowest-risk way to try this.
``litellm_classifier.JevRouteClassifier``
    ``classifier_type: custom`` inside LiteLLM's native complexity router.
    Returns a *tier name* and lets LiteLLM's tier machinery handle deployment
    selection, cooldowns, fallbacks and load balancing.
``litellm_plugin.JevRouteRoutingPlugin``
    ``Router(plugins=[...])`` from the SDK. Narrows ``candidate_models`` to the
    deployments of the chosen tier. The only one of the three that works outside
    a proxy.

All three are thin. The local hard gate, the fail-closed defaults, the calibrated
decision, the policy engine and the decision log all live in the core and are
shared: an integration decides *where the request text comes from* and *what
happens to the answer*, and nothing else. That is why every one of them can be
read in a sitting, and why swapping between them does not change what gets
logged -- the dataset is the same dataset.

Imports here are lazy on purpose. The jev-route core has no LiteLLM dependency
(stdlib + PyYAML + httpx only), and ``import jev_route`` must not start failing
because an optional extra is missing -- nor should it pay LiteLLM's import cost
in a process that only wants the CLI. So this package imports nothing at module
load; attribute access resolves the submodule and, if LiteLLM is absent, raises
an ``ImportError`` that names the extra to install.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

#: Attribute name -> the submodule that defines it.
_LAZY_EXPORTS: dict[str, str] = {
    "JevRoutePreCallHook": "litellm_hook",
    "proxy_handler_instance": "litellm_hook",
    "JevRouteClassifier": "litellm_classifier",
    "classifier": "litellm_classifier",
    "JevRouteRoutingPlugin": "litellm_plugin",
    "routing_plugin": "litellm_plugin",
    # Shared helpers are re-exported because an operator writing a custom
    # integration needs the same payload/metadata extraction, and duplicating it
    # is how two integrations drift into disagreeing about what is PII.
    "get_router": "_shared",
    "configure_router": "_shared",
    "build_router": "_shared",
    "aclose_router": "_shared",
    "extract_messages": "_shared",
    "extract_metadata": "_shared",
    "decide": "_shared",
}

#: Literal, not ``sorted(_LAZY_EXPORTS)``: tooling (and ruff's PLE0605) reads
#: ``__all__`` statically, and a computed one is invisible to it.
__all__ = [
    "JevRouteClassifier",
    "JevRoutePreCallHook",
    "JevRouteRoutingPlugin",
    "aclose_router",
    "build_router",
    "classifier",
    "configure_router",
    "decide",
    "extract_messages",
    "extract_metadata",
    "get_router",
    "proxy_handler_instance",
    "routing_plugin",
]


def __getattr__(name: str) -> Any:
    """Resolve a submodule or one of its exports on first access."""
    if name in _LAZY_EXPORTS:
        module = _import(_LAZY_EXPORTS[name])
        value = getattr(module, name)
        globals()[name] = value  # cache, so PEP 562 lookup happens once
        return value
    if name in {"litellm_hook", "litellm_classifier", "litellm_plugin", "_shared"}:
        module = _import(name)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


def _import(name: str) -> Any:
    try:
        return import_module(f"{__name__}.{name}")
    except ImportError as exc:  # pragma: no cover - depends on the environment
        if "litellm" in str(exc) or exc.name == "litellm":
            raise ImportError(
                f"jev-route's LiteLLM integrations need LiteLLM: pip install 'jev-route[litellm]' "
                f"(or `pip install litellm`). Original error: {exc}"
            ) from exc
        raise
