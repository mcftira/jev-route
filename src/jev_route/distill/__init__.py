"""The graduation pipeline: ``export`` -> ``train`` -> ``evaluate`` -> ``package`` -> ``graduate``.

This package is the second half of the project's claim -- "Run it. Log it.
Distill it. Own it." -- and it is optional in the strictest sense: the router
runs in containers that have no numerical stack installed at all, so
``import jev_route`` must never pull one in.

That constraint shapes this file. The five verbs :mod:`jev_route.cli` calls live
in :mod:`jev_route.distill.cli` and are re-exported through :func:`__getattr__`
(PEP 562) rather than imported here, so ``import jev_route.distill`` costs one
small module and nothing else. The first *attribute access* -- ``distill.train_cli``
-- is what loads the implementation, and even then numpy stays out until the verb
actually needs it: every heavy import in this package sits inside a function, an
invariant :mod:`tests.test_invariants` enforces both statically (no heavy import
outside a function body) and dynamically (a subprocess with numpy, torch, pandas,
scipy, sklearn, redis and transformers blocked at the import system, in which the
core still imports and routes a prompt).

The submodules are public API in their own right and are listed in ``__dir__`` so
they are discoverable; importing one directly (``from jev_route.distill.export
import export_dataset``) is the normal way to use them from a program.
"""

from __future__ import annotations

from typing import Any

#: The CLI verbs. :mod:`jev_route.cli` reaches them as ``distill.export_cli(args)``
#: and friends; the argument contract is fixed there.
#: Listed in lifecycle order rather than alphabetically, because that order IS the
#: documentation. ``label_sensitivity`` leads the gate track (it builds the data
#: layer 2 trains on); the rest are the router track.
__all__ = [  # noqa: RUF022
    "label_sensitivity_cli",
    "export_cli",
    "train_cli",
    "evaluate_cli",
    "package_cli",
    "graduate_cli",
]

#: Submodules a program may want through this namespace rather than by full path.
_SUBMODULES = ("artifact", "cli", "evaluate", "export", "train")


def __getattr__(name: str) -> Any:
    """Lazily resolve the verbs, and the submodules, on first access."""
    if name in __all__:
        from . import cli as _cli

        return getattr(_cli, name)
    if name in _SUBMODULES:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__, *_SUBMODULES})
