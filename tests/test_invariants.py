"""The project's promises, as executable assertions.

Everything else in this suite tests a module. This file tests *claims* -- the
sentences in the README and the module docstrings that a sceptical engineer would
read first and believe last:

* "No code path calls the Jev API except inside JevBackend."
* "The local hard gate runs first, always locally, never model-decided, never bypassed."
* "Fail-closed by default."
* "Every decision is logged with full soft probability distributions."
* "The whole system runs end-to-end with MockBackend and no API key."

Each is enforced mechanically -- by grepping the source tree, by walking the AST, by
running a subprocess with a scrubbed environment, a blocked import system and a
denied socket -- rather than by re-asserting behaviour that a unit test already
covers. A claim that cannot be checked by a machine is a claim that will rot.

One test is marked ``xfail(strict=True)`` because the source tree currently violates
its own documented invariant: the cloud endpoint is named in two modules instead of
one. It is left failing-on-purpose and named in the report so the violation stays
visible instead of being quietly deleted -- and ``strict=True`` means that on the day
the source is fixed, the test turns red until the marker is removed. That is the
point of the marker, not a defect in it.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from jev_route.backends.base import BackendResult, DecisionRequest
from jev_route.backends.mock import MockBackend
from jev_route.cache import NullCache
from jev_route.gate import DEFAULT_DETECTORS, HardGate
from jev_route.logging_sink import CallbackSink
from jev_route.policy import FailurePolicy, Policy
from jev_route.router import Router
from jev_route.schema import (
    COMPLEXITY_LEVELS,
    DOMAINS,
    SCHEMA_VERSION,
    SENSITIVITY_LEVELS,
    DecisionAnswers,
    DecisionRecord,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src" / "jev_route"
PYTHON = sys.executable

#: Modules whose whole job is optional heavy lifting. ``integrations`` may import
#: litellm at module level because that IS the integration; nothing else may.
CORE_EXEMPTIONS: dict[str, set[str]] = {"integrations/litellm_hook.py": {"litellm"}}

HEAVY_DEPENDENCIES = {"numpy", "torch", "redis", "transformers", "pandas", "sklearn", "scipy"}

#: Libraries that could carry a prompt off the machine. The offline probes block
#: these at the import system, which is what turns "no egress" from an assumption
#: about whichever venv happens to run the suite into a checked property.
EGRESS_CLIENTS = ("httpx", "requests", "aiohttp", "urllib3", "http.client", "urllib.request")

#: ``JEV_PROBE_BLOCK`` value that also hides every optional extra, i.e. the import
#: environment of a bare ``pip install jev-route``.
NO_EXTRAS = ",".join(sorted(HEAVY_DEPENDENCIES))


# --------------------------------------------------------------------------- #
# Source-tree scanning
# --------------------------------------------------------------------------- #
def source_files() -> list[Path]:
    files = sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)
    assert files, f"no python sources found under {SRC}"
    return files


def relative(path: Path) -> str:
    return str(path.relative_to(SRC)).replace(os.sep, "/")


def parsed(path: Path) -> ast.Module:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:  # pragma: no cover - only during a broken edit
        pytest.fail(f"{relative(path)} does not parse: {exc}")


@pytest.fixture(scope="module")
def sources() -> dict[str, tuple[Path, ast.Module, str]]:
    """Every src module as (path, AST, text), parsed once.

    A syntax error anywhere in src fails here with the filename, which is the
    useful message -- the alternative is twenty confusing collection errors.
    """
    return {relative(p): (p, parsed(p), p.read_text(encoding="utf-8")) for p in source_files()}


def module_level_imports(tree: ast.Module) -> list[str]:
    """Top-level names imported at module scope (not inside a function)."""
    out: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            out.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append(node.module.split(".")[0])
    return out


def imported_module_paths(tree: ast.Module) -> set[str]:
    """Every module path an import names, relative imports included.

    Unlike :func:`all_imports` this keeps the whole dotted path, because collapsing
    to the first component -- correct for "is this a heavy dependency" -- destroys
    the only information a "may this module be named here" check needs:
    ``from .backends.jev import JevBackend`` reduces to ``backends``.
    """
    paths: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            paths.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            paths.add(node.module)
    return paths


def all_imports(tree: ast.Module) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend((alias.name.split(".")[0], node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append((node.module.split(".")[0], node.lineno))
    return out


#: Only function bodies count as lazy. An import in a *class* body, or inside a
#: module-level ``try:``/``if:`` block, still executes when the module is imported,
#: so it costs exactly what a top-level import costs and must be reported as one.
FUNCTION_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef)


def nested_import_lines(tree: ast.Module, kinds: tuple[type[ast.AST], ...] = FUNCTION_SCOPES) -> set[int]:
    """Line numbers of every import that sits inside one of *kinds*.

    Containment is decided by line *ranges* -- ``(lineno, end_lineno)`` of each scope
    -- and not by comparing an import's line to the scopes' own ``lineno``. The
    second reading looks natural and is wrong for every import in the tree, because
    a body never starts on its ``def`` line: it reports genuinely lazy imports as
    module level, and would wave through a real module-level import whenever a
    function happened to be defined on the same line.
    """
    ranges = [
        (node.lineno, node.end_lineno or node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, kinds)
    ]
    return {
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and any(start <= node.lineno <= end for start, end in ranges)
    }


def dotted(node: ast.expr) -> str:
    """Render a call target back to its dotted source form (``self.gate.scan``)."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def call_lines(scope: ast.AST, target: str) -> set[int]:
    """Lines where *scope* calls *target*, e.g. ``"self.backend.decide"``."""
    return {
        node.lineno
        for node in ast.walk(scope)
        if isinstance(node, ast.Call) and dotted(node.func) == target
    }


def self_calls(scope: ast.AST) -> set[str]:
    """Names of methods *scope* calls on ``self`` -- its edges in the call graph."""
    return {
        node.func.attr
        for node in ast.walk(scope)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    }


def reaches(start: str, graph: dict[str, set[str]], target: str) -> bool:
    """Whether *start* can reach *target* by following ``self.`` calls."""
    seen: set[str] = set()
    stack = [start]
    while stack:
        for callee in graph.get(stack.pop(), set()):
            if callee == target:
                return True
            if callee not in seen:
                seen.add(callee)
                stack.append(callee)
    return False


#: Provider key prefixes. ``gate.py``'s ``provider_api_key`` detector has to spell
#: all of these out in order to find them in a user's prompt, so a scan for the bare
#: prefix can never pass against the one module whose job is recognising
#: credentials -- it would fire on the detector and never on a leak.
CREDENTIAL_PREFIXES = ("sk-ant-", "AKIA", "ghp_", "xoxb-", "AIzaSy", "hf_")

#: Realistic key material behind a prefix. Every key these prefixes describe carries
#: at least 16 more characters; a regex character class such as ``AKIA[0-9A-Z]{16}``
#: does not. That difference is the whole basis of the scan.
KEY_MATERIAL = r"[A-Za-z0-9_.\-]{16,}"


def find_planted_credentials(text: str) -> list[str]:
    """Credential-shaped *values* in ``text``, reported as ``"<prefix>... (line N)"``.

    Never the value itself: a scanner that echoes what it found turns one committed
    key into one committed key in every CI log and every pasted failure message.
    """
    found: list[str] = []
    for prefix in CREDENTIAL_PREFIXES:
        for match in re.finditer(re.escape(prefix) + KEY_MATERIAL, text):
            line = text.count("\n", 0, match.start()) + 1
            found.append(f"{prefix}... (line {line})")
    return sorted(found)


# --------------------------------------------------------------------------- #
# Policy documents
# --------------------------------------------------------------------------- #
def policy_doc(**overrides: Any) -> dict[str, Any]:
    """A policy document that is minimal but *loadable*.

    ``Policy.from_dict`` validates two things that make "minimal" a moving target,
    and a test that trips over either reports a ``PolicyError`` instead of the
    invariant it meant to check:

    * ``tier_order`` defaults to ``schema.TIERS`` and every name in it must be
      defined under ``tiers:``, so all three standard tiers have to appear even when
      the test only cares about one;
    * ``FailurePolicy`` validates *both* of its tiers at load time, so ``strong``
      must exist even in a policy that can never fail open.

    The subprocess probes spell their documents out inline instead: a child process
    cannot import from this test module, and there the document is part of what is
    being run rather than part of the harness.
    """
    doc: dict[str, Any] = {
        "version": 1,
        "backend": {"name": "mock"},
        "tiers": {"local": ["local-model"], "cheap": ["cheap-model"], "strong": ["strong-model"]},
        "tier_order": ["cheap", "strong", "local"],
        "rules": [
            {"id": "gate", "if": "gate_force_local", "then": {"tier": "local"}},
            {"id": "default", "then": {"tier": "cheap"}},
        ],
    }
    doc.update(overrides)
    return doc


# --------------------------------------------------------------------------- #
# 1. The egress boundary
# --------------------------------------------------------------------------- #
class TestEgressBoundary:
    def test_the_typesafe_endpoint_appears_in_exactly_one_module(self, sources: dict[str, Any]) -> None:
        """The cloud hostname must appear in backends/jev.py and nowhere else in src/."""
        offenders = sorted(name for name, (_, _, text) in sources.items() if "api.typesafe.ai" in text)
        assert offenders == ["backends/jev.py"], f"api.typesafe.ai also appears in: {offenders}"

    def test_no_module_names_the_endpoint_outside_jev_backend(self, sources: dict[str, Any]) -> None:
        """The weaker, currently-true form: only jev.py and the backend factory mention it."""
        offenders = sorted(
            name
            for name, (_, _, text) in sources.items()
            if "typesafe.ai" in text and name not in {"backends/jev.py", "backends/__init__.py"}
        )
        assert offenders == [], f"the cloud endpoint leaked into: {offenders}"

    def test_only_jev_backend_references_the_systemone_path(self, sources: dict[str, Any]) -> None:
        offenders = sorted(
            name
            for name, (_, _, text) in sources.items()
            if "/v1/systemone" in text and name not in {"backends/jev.py", "backends/__init__.py"}
        )
        assert offenders == []

    def test_jev_backend_is_the_only_module_that_can_egress(self, sources: dict[str, Any]) -> None:
        """No other module may hold an HTTP client or a socket at all.

        This is the mechanical version of "delete one class and the cloud dependency
        is gone": if egress could happen anywhere, deleting JevBackend would prove
        nothing.
        """
        forbidden = {"httpx", "requests", "aiohttp", "urllib3"}
        for name, (_, tree, _) in sources.items():
            if name == "backends/jev.py":
                continue
            imported = {mod for mod, _ in all_imports(tree)}
            assert not (imported & forbidden), f"{name} imports an HTTP client: {sorted(imported & forbidden)}"

    def test_urllib_is_not_used_as_a_side_door(self, sources: dict[str, Any]) -> None:
        for name, (_, tree, _) in sources.items():
            imported = {mod for mod, _ in all_imports(tree)}
            assert "urllib" not in imported, f"{name} imports urllib"


class TestHttpxIsLazy:
    def test_no_module_imports_httpx_at_module_level(self, sources: dict[str, Any]) -> None:
        """``import jev_route`` must work in an environment with no httpx installed."""
        offenders = sorted(name for name, (_, tree, _) in sources.items() if "httpx" in module_level_imports(tree))
        assert offenders == []

    def test_httpx_is_only_imported_inside_jev_backend(self, sources: dict[str, Any]) -> None:
        offenders = sorted(
            name for name, (_, tree, _) in sources.items() if "httpx" in {m for m, _ in all_imports(tree)}
        )
        assert offenders == ["backends/jev.py"]

    def test_the_lazy_import_lives_inside_a_function(self, sources: dict[str, Any]) -> None:
        """``import httpx`` must be inside a function *body*, not merely below one.

        An import placed after the class definition is still paid for by every
        ``import jev_route``, so position in the file proves nothing; containment in
        a function's line range is what makes it lazy. Both spellings are checked --
        ``import httpx`` and ``from httpx import ...`` cost the same.
        """
        _, tree, _ = sources["backends/jev.py"]
        lines = {
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "httpx" for a in node.names)
        } | {
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "httpx"
        }
        assert lines, "jev.py no longer imports httpx at all"
        assert lines <= nested_import_lines(tree), "the httpx import escaped to module level"


class TestHeavyDependenciesAreLazy:
    def test_the_core_imports_and_routes_with_no_extras_installed(self) -> None:
        """numpy/torch/redis belong behind an extra, so the core must run without them.

        This is the dynamic half of the invariant, and the half that cannot be fooled
        by an import that is merely *spelled* lazily: every extra is blocked at the
        import system, then the package is imported and a whole route is run offline.
        A machine with torch installed and a bare ``pip install jev-route`` are
        indistinguishable from in here, which is the promise being made.
        """
        code = """
import asyncio, json, os
import jev_route
from jev_route import MockBackend, Policy, Router
from jev_route.logging_sink import CallbackSink

# Prove the environment this probe claims to simulate: each extra really is absent.
for extra in os.environ["JEV_PROBE_BLOCK"].split(","):
    try:
        __import__(extra)
    except ImportError as exc:
        assert "blocked in this probe" in str(exc), f"{extra} came from somewhere else: {exc}"
    else:
        raise AssertionError(f"{extra} is importable; this probe proves nothing about the core")

records = []
policy = Policy.from_dict({
    "version": 1,
    "backend": {"name": "mock"},
    "tiers": {"local": ["l"], "cheap": ["c"], "strong": ["s"]},
    "tier_order": ["cheap", "strong", "local"],
    "rules": [
        {"id": "gate", "if": "gate_force_local", "then": {"tier": "local"}},
        {"id": "default", "then": {"tier": "cheap"}},
    ],
})
router = Router(policy, MockBackend(), sink=CallbackSink(records.append))
decision = asyncio.run(router.route_text("Summarize the roadmap for the team."))
print(json.dumps({"tier": decision.tier, "records": len(records), "version": jev_route.__version__}))
"""
        result = run_probe(code, timeout=120, JEV_PROBE_BLOCK=NO_EXTRAS)
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        assert payload["records"] == 1, "the route did not complete without the extras"
        assert payload["tier"] in {"cheap", "strong", "local"}

    def test_heavy_imports_are_nested(self, sources: dict[str, Any]) -> None:
        """The static half: no heavy import may sit outside a function body.

        Stricter than a scan of ``tree.body``, and deliberately so -- an import under
        a module-level ``try:`` or ``if:`` is invisible to that scan yet still runs,
        and still breaks the laptop the extra was never installed on.
        """
        offenders = [
            f"{name}:{lineno} imports {module_name}"
            for name, (_, tree, _) in sources.items()
            for module_name, lineno in all_imports(tree)
            if module_name in HEAVY_DEPENDENCIES and lineno not in nested_import_lines(tree)
        ]
        assert not offenders, f"heavy imports outside a function body: {offenders}"

    def test_third_party_module_level_imports_are_only_the_documented_ones(self, sources: dict[str, Any]) -> None:
        """Core rule: stdlib + PyYAML + httpx. Integrations may need their framework."""
        stdlib = set(sys.stdlib_module_names)
        for name, (_, tree, _) in sources.items():
            external = {m for m in module_level_imports(tree) if m not in stdlib and m != "jev_route"}
            allowed = {"yaml"} | CORE_EXEMPTIONS.get(name, set())
            assert external <= allowed, f"{name} has module-level third-party imports {sorted(external - allowed)}"


# --------------------------------------------------------------------------- #
# 2. It runs with no key and no network
# --------------------------------------------------------------------------- #
#: Preamble every offline probe runs under. Three guards, then a self-check that
#: they are armed:
#:
#: * outbound HTTP clients cannot be imported, so the lazy ``import httpx`` inside
#:   ``JevBackend`` fails loudly instead of reaching the cloud;
#: * ``create_connection``/``getaddrinfo``/``socket.connect``/``socket.connect_ex``
#:   raise, so a client built straight on a socket fails loudly too -- and with
#:   ``asyncio``'s own connector, which resolves through ``getaddrinfo``;
#: * the blocklist is a ``sys.meta_path`` finder and the socket *methods* are
#:   patched, because the obvious alternative -- blocking the ``socket`` module or
#:   replacing ``socket.socket`` -- breaks the standard library instead of proving
#:   anything: ``import asyncio`` pulls in ``socket`` and ``selectors``, and
#:   ``ssl.SSLSocket`` subclasses ``socket.socket``.
#:
#: ``@EGRESS@`` is substituted from ``EGRESS_CLIENTS`` below so this string and that
#: constant cannot drift apart.
SUBPROCESS_PREAMBLE = """
import os, socket, sys


# Raised in place of a connection attempt, so a probe fails loudly.
class NetworkAccess(RuntimeError):
    pass


_BLOCKED = {@EGRESS@}
_BLOCKED |= {name for name in os.environ.get("JEV_PROBE_BLOCK", "").split(",") if name}


class _BlockedImportFinder:
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == blocked or fullname.startswith(blocked + ".") for blocked in _BLOCKED):
            raise ImportError(f"{fullname!r} is blocked in this probe")
        return None


sys.meta_path.insert(0, _BlockedImportFinder())


def _deny(*args, **kwargs):
    raise NetworkAccess("this probe attempted to open a network connection")


socket.create_connection = _deny
socket.getaddrinfo = _deny
socket.socket.connect = _deny
socket.socket.connect_ex = _deny


def _must_raise(call, error, what):
    # A guard that cannot fire is not a guard: without this, a rename or a finder
    # that stops running first would leave every probe below passing forever while
    # proving nothing.
    try:
        call()
    except error:
        return
    except OSError as exc:
        raise AssertionError(f"{what} is not armed; it reached the OS: {exc}") from None
    raise AssertionError(f"{what} is not armed; it succeeded")


_BLOCKED.add("jev_route_probe_sentinel")
_must_raise(lambda: __import__("jev_route_probe_sentinel"), ImportError, "the import blocklist")
# ``.invalid`` is reserved by RFC 2606 and can never resolve, so a disarmed DNS
# guard fails fast here instead of hanging on a real lookup.
_must_raise(lambda: socket.getaddrinfo("probe.invalid", 443), NetworkAccess, "the DNS guard")
""".replace("@EGRESS@", ", ".join(repr(module) for module in EGRESS_CLIENTS))


def scrubbed_env(**extra: str) -> dict[str, str]:
    """An environment with every credential removed, so a test cannot pass by accident."""
    env = {k: v for k, v in os.environ.items() if not k.upper().endswith(("API_KEY", "TOKEN", "SECRET"))}
    env.pop("TYPESAFE_API_KEY", None)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env.update(extra)
    return env


def run_probe(body: str, *, timeout: int = 60, **blocked_or_env: str) -> subprocess.CompletedProcess[str]:
    """Run ``body`` in a child process that cannot reach the network or read a key.

    :data:`SUBPROCESS_PREAMBLE` is prepended *here* rather than at each call site so
    that no probe can forget it -- a probe without the guards still passes, and goes
    on passing for the wrong reason long after the reason is gone.

    ``blocked_or_env`` goes straight into the child environment; pass
    ``JEV_PROBE_BLOCK=NO_EXTRAS`` to hide the optional extras as well.
    """
    return subprocess.run(
        [PYTHON, "-c", SUBPROCESS_PREAMBLE + body],
        cwd=REPO_ROOT,
        env=scrubbed_env(**blocked_or_env),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class TestNoKeyNoNetwork:
    """Importing and running jev_route must open no connection and read no key.

    Each test below runs in a *subprocess* under ``SUBPROCESS_PREAMBLE`` rather than
    in-process with a monkeypatched socket, because the claim is about the whole
    import graph: an in-process test inherits whatever this suite already imported,
    and so cannot see a module-scope ``import httpx`` that the probe would hit cold.
    """

    def test_import_succeeds_with_no_key_and_no_socket(self) -> None:
        code = """
import os
assert not os.environ.get("TYPESAFE_API_KEY"), "the key should have been scrubbed"
import jev_route
from jev_route import Router, MockBackend, Policy, HardGate
print("imported", jev_route.__version__)

# Importing the package must not have pulled the HTTP client in on the way, and the
# blocklist must still be able to stop it: our finder's message, not the interpreter's
# plain ModuleNotFoundError, is what proves the guard fired.
try:
    import httpx
except ImportError as exc:
    print("httpx blocked by the probe:", "blocked in this probe" in str(exc))
else:
    print("httpx blocked by the probe: False")
"""
        result = run_probe(code)
        assert result.returncode == 0, result.stderr
        assert "imported" in result.stdout
        assert "httpx blocked by the probe: True" in result.stdout

    def test_a_full_route_runs_with_no_key_and_no_socket(self) -> None:
        """The definition of done: gate -> decide -> merge -> evaluate -> log, offline."""
        code = """
import asyncio, json, os, tempfile
from jev_route import Router, Policy, MockBackend
from jev_route.logging_sink import CallbackSink

async def main():
    records = []
    policy = Policy.from_dict({
        "version": 1,
        "backend": {"name": "mock"},
        "tiers": {"local": ["l"], "cheap": ["c"], "strong": ["s"]},
        "tier_order": ["cheap", "strong", "local"],
        "rules": [
            {"id": "gate", "if": "gate_force_local", "then": {"tier": "local"}},
            {"id": "sensitive",
             "if": 'sensitivity in ["confidential", "regulated"] or pii_present',
             "then": {"tier": "local"}},
            {"id": "default", "then": {"tier": "cheap"}},
        ],
        "logging": {"enabled": True, "path": os.path.join(tempfile.mkdtemp(), "d.jsonl")},
    })
    router = Router(policy, MockBackend(), sink=CallbackSink(records.append))
    plain = await router.route_text("Summarize the roadmap for the team.")
    secret = await router.route_text("Charge card 4111 1111 1111 1111 now.")
    print(json.dumps({
        "plain_tier": plain.tier,
        "secret_tier": secret.tier,
        "secret_backend": secret.backend,
        "records": len(records),
        "api_key_present": bool(os.environ.get("TYPESAFE_API_KEY")),
    }))

asyncio.run(main())
"""
        result = run_probe(code, timeout=120)
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        assert payload["api_key_present"] is False
        assert payload["secret_tier"] == "local"
        assert payload["secret_backend"] == "gate"
        assert payload["records"] == 2
        assert payload["plain_tier"] in {"cheap", "strong", "local"}

    def test_jev_backend_is_not_constructed_at_import_time(self) -> None:
        """Importing the package must not require -- or read -- a key."""
        code = """
import jev_route, jev_route.backends.jev as jev
try:
    jev.JevBackend()
except ValueError as exc:
    print("refused:", "MockBackend" in str(exc))
else:
    print("refused: False")
"""
        result = run_probe(code)
        assert result.returncode == 0, result.stderr
        assert "refused: True" in result.stdout

    def test_no_module_reads_a_dotenv_file(self, sources: dict[str, Any]) -> None:
        """Secrets come from the environment. A library that opens .env is a surprise."""
        for name, (_, tree, _) in sources.items():
            imported = {m for m, _ in all_imports(tree)}
            assert "dotenv" not in imported, f"{name} imports dotenv"
        for name, (_, _, text) in sources.items():
            assert "load_dotenv" not in text, f"{name} calls load_dotenv"

    async def test_router_works_in_process_with_the_key_removed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        policy = Policy.from_dict(policy_doc())
        records: list[DecisionRecord] = []
        router = Router(policy, MockBackend(), sink=CallbackSink(records.append), cache=NullCache())
        decision = await router.route_text("Summarize the roadmap for the team.")
        assert decision.tier == "cheap"
        assert decision.backend == "mock"
        assert len(records) == 1

    async def test_from_policy_file_needs_no_key_when_the_backend_is_mock(self, tmp_path: Path) -> None:
        path = tmp_path / "policy.yaml"
        path.write_text(
            "version: 1\n"
            "backend:\n  name: mock\n"
            "tiers:\n  local: [l]\n  cheap: [c]\n  strong: [s]\n"
            "rules:\n  - id: default\n    then:\n      tier: cheap\n"
            f"logging:\n  enabled: true\n  path: {tmp_path / 'd.jsonl'}\n",
            encoding="utf-8",
        )
        router = Router.from_policy_file(path)
        assert (await router.route_text("hello there")).tier == "cheap"
        await router.aclose()


# --------------------------------------------------------------------------- #
# 3. The gate runs first and is never model-decided
# --------------------------------------------------------------------------- #
class TestGateIsFirst:
    def test_router_calls_gate_scan_before_backend_decide(self, sources: dict[str, Any]) -> None:
        """Source order is the enforcement: the gate line must precede the backend line."""
        _, _, text = sources["router.py"]
        scan_at = text.index("self.gate.scan(")
        decide_at = text.index("await self.backend.decide(")
        assert scan_at < decide_at, "the backend is called before the gate has run"

    def test_no_router_path_can_skip_the_gate(self, sources: dict[str, Any]) -> None:
        """Every public entry point funnels through ``_route``, where the gate lives.

        Read off the AST rather than out of sliced source text: the claim is about
        call structure, and a text slice breaks the moment a method is reordered,
        decorated, or renamed. Three properties are needed, because each on its own
        is satisfiable by a router that leaks:

        * every public ``route*`` method reaches ``_route``, following ``self.``
          calls so a sync wrapper delegating to an async one still counts;
        * inside ``_route``, ``self.gate.scan`` is called before
          ``self.backend.decide`` -- the gate sees the RAW excerpt, so calling it
          *afterwards* would be a log entry, not a gate;
        * no other method calls the backend directly. A second call site is a second
          path, and a second path is a way to skip the gate.
        """
        _, tree, _ = sources["router.py"]
        router_class = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Router"), None)
        assert router_class is not None, "router.py no longer defines a Router class"
        methods = {
            node.name: node
            for node in router_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        entrypoints = sorted(name for name in methods if not name.startswith("_") and name.startswith("route"))
        assert {"route_text", "route_messages"} <= set(entrypoints), (
            f"the public routing API changed shape; re-check the gate for: {entrypoints}"
        )

        graph = {name: self_calls(node) for name, node in methods.items()}
        unreachable = [name for name in entrypoints if not reaches(name, graph, "_route")]
        assert not unreachable, f"these entry points can decide without _route: {unreachable}"

        scan_lines = call_lines(methods["_route"], "self.gate.scan")
        decide_lines = call_lines(methods["_route"], "self.backend.decide")
        assert scan_lines and decide_lines, "_route no longer calls both the gate and the backend"
        assert min(scan_lines) < min(decide_lines), "the backend is called before the gate has run"

        side_doors = sorted(
            name
            for name, node in methods.items()
            if name != "_route" and call_lines(node, "self.backend.decide")
        )
        assert not side_doors, f"these methods call the backend directly, bypassing _route: {side_doors}"

    def test_the_gate_is_never_constructed_from_a_backend_answer(self, sources: dict[str, Any]) -> None:
        """No HardGate construction may take its input from anything a model produced."""
        _, _, text = sources["router.py"]
        assert "HardGate(" not in text or "def _gate_from_policy" in text
        # The gate is built from policy config only: disabled detectors and the
        # placeholder-domain flag. Neither is a model output.
        factory = text.split("def _gate_from_policy", 1)[1].split("\ndef ", 1)[0]
        assert "answers" not in factory
        assert "backend" not in factory

    def test_router_always_has_a_real_gate(self) -> None:
        policy = Policy.from_dict(policy_doc())
        for gate_argument in (None,):
            router = Router(policy, MockBackend(), gate=gate_argument, sink=CallbackSink(lambda r: None))
            assert isinstance(router.gate, HardGate)
            assert router.gate.detectors, "a gate with no detectors is not a gate"
            assert router.gate.scan("4111 1111 1111 1111").blocks_backend is True

    def test_gate_config_cannot_disable_the_gate_itself(self) -> None:
        """``disabled_detectors`` silences rules; there is no key that stops the scan."""
        policy = Policy.from_dict(
            policy_doc(gate={"disabled_detectors": [d.name for d in DEFAULT_DETECTORS]})
        )
        router = Router(policy, MockBackend(), sink=CallbackSink(lambda r: None))
        assert isinstance(router.gate, HardGate)
        assert router.gate.scan("anything").fired is False  # every rule silenced...
        assert router.gate.scan("anything") is not None  # ...and the gate still ran

    def test_every_detector_floor_is_on_the_schema_ladder(self) -> None:
        """A floor nobody can compare against would silently drop out of the merge."""
        for detector in DEFAULT_DETECTORS:
            assert detector.sensitivity_floor in SENSITIVITY_LEVELS, detector.name

    def test_advisory_detectors_never_force_or_block(self) -> None:
        for detector in DEFAULT_DETECTORS:
            if detector.advisory:
                assert detector.force_local is False, detector.name
                assert detector.blocks_backend is False, detector.name
                assert detector.pii is False, detector.name

    def test_hard_detectors_do_force_or_block(self) -> None:
        """The converse: a non-advisory detector that does neither is dead weight."""
        for detector in DEFAULT_DETECTORS:
            if not detector.advisory:
                assert detector.force_local or detector.blocks_backend, detector.name

    def test_blocking_detectors_cover_every_secret_and_regulated_category(self) -> None:
        categories = {d.category for d in DEFAULT_DETECTORS if d.blocks_backend}
        assert "secret" in categories
        assert "regulated" in categories

    def test_the_default_gate_instance_ships_every_detector(self) -> None:
        from jev_route.gate import default_gate

        assert {d.name for d in default_gate.detectors} == {d.name for d in DEFAULT_DETECTORS}

    def test_detector_names_are_a_stable_dataset_vocabulary(self) -> None:
        """Findings are logged by name, so the names are part of the dataset contract."""
        expected = {
            "payment_card", "us_ssn", "iban", "uk_nhs_number", "uk_national_insurance",
            "private_key_block", "provider_api_key", "inline_credential", "basic_auth_url",
            "medical_record_number", "email_address", "phone_number", "date_of_birth",
            "street_address", "named_individual", "kw_health_regulation", "kw_legal_privilege",
            "kw_financial_regulation", "kw_minors", "kw_confidential_business",
            "kw_security_vulnerability", "kw_internal_only",
        }
        assert expected <= {d.name for d in DEFAULT_DETECTORS}


# --------------------------------------------------------------------------- #
# 4. Fail closed by default
# --------------------------------------------------------------------------- #
class DownBackend:
    """A backend that is honestly unavailable, for the fail-closed checks.

    Defined here rather than borrowed from ``conftest`` because what it returns *is*
    part of the claim under test: ``degraded=True`` with maximum-uncertainty answers
    is the shape the router is required to treat as an outage and not as a read.
    """

    name = "down"
    model_version = "down-0.0.0"

    async def decide(self, request: DecisionRequest) -> BackendResult:
        return BackendResult(
            answers=DecisionAnswers.unknown(),
            model_version=self.model_version,
            questions_sent={},
            latency_ms=0.0,
            degraded=True,
            degrade_reason="probe backend is down",
        )

    async def aclose(self) -> None:
        return None


class TestFailClosedByDefault:
    def test_the_dataclass_default_is_fail_closed(self) -> None:
        assert FailurePolicy().mode == "fail_closed"
        assert FailurePolicy().fail_closed_tier == "local"

    def test_the_shipped_policy_is_fail_closed(self) -> None:
        policy = Policy.from_file(REPO_ROOT / "policies" / "default.yaml")
        assert policy.failure.mode == "fail_closed"
        assert policy.failure.fail_closed_tier == "local"

    def test_a_policy_without_the_section_is_fail_closed(self) -> None:
        """Omitting ``on_backend_down`` must buy the safe default, not the fast one.

        Also pins the two spellings the loader accepts -- a mapping and a bare mode
        string. A shorthand that silently failed to parse would leave an operator who
        wrote ``on_backend_down: fail_open`` with a fail-closed router (or the reverse,
        which is a data-egress surprise), so both directions have to be observable.
        The absent section is checked against the dataclass default itself, not just
        against the string ``"fail_closed"``, so the two cannot drift.
        """
        absent = Policy.from_dict(policy_doc())
        assert absent.failure == FailurePolicy(), "the omitted section is not the dataclass default"
        assert absent.failure.mode == "fail_closed"
        assert absent.failure.fail_closed_tier == "local"

        for shorthand in ("fail_closed", "fail_open"):
            parsed = Policy.from_dict(policy_doc(on_backend_down=shorthand))
            assert parsed.failure.mode == shorthand, f"the {shorthand!r} shorthand did not parse"

        explicit = Policy.from_dict(
            policy_doc(on_backend_down={"mode": "fail_open", "fail_open_tier": "strong"})
        )
        assert explicit.failure.mode == "fail_open"
        assert explicit.failure.fail_open_tier == "strong"
        assert explicit.failure.fail_closed_tier == "local"

    async def test_a_down_backend_with_no_section_lands_on_the_safest_tier(self) -> None:
        """The behavioural half: an outage must *route* to ``local``, not merely parse to it.

        Asserting ``policy.failure.mode`` proves what the loader produced. This drives
        a router whose backend reports itself down, with the section absent, and reads
        the tier the request actually landed on -- the only version of "fail closed" an
        operator ever experiences. The logged record is checked too: the training
        dataset must say the decision came from the outage path, not from a rule.
        """
        records: list[DecisionRecord] = []
        policy = Policy.from_dict(policy_doc())
        assert "on_backend_down" not in policy.raw, "policy_doc must omit the section"
        router = Router(policy, DownBackend(), sink=CallbackSink(records.append), cache=NullCache())

        decision = await router.route_text("Summarize the roadmap for the team.")

        assert decision.degraded is True, "the double did not report an outage"
        assert decision.tier == "local" == policy.failure.fail_closed_tier
        assert decision.rule_id == "backend.down"
        assert len(records) == 1
        assert json.loads(records[0].to_json())["decision"]["tier"] == "local"

    def test_an_uncertain_answer_is_treated_as_pii_present_by_default(self) -> None:
        policy = Policy.from_file(REPO_ROOT / "policies" / "default.yaml")
        assert policy.uncertainty.pii_uncertain_counts_as_present is True
        assert policy.uncertainty.pii_uncertain_threshold > 0.0

    def test_uncertainty_bumps_are_never_negative(self) -> None:
        """A bump that loosened an answer would invert the whole design."""
        policy = Policy.from_file(REPO_ROOT / "policies" / "default.yaml")
        assert policy.uncertainty.sensitivity_bump_levels >= 0
        assert policy.uncertainty.complexity_bump_levels >= 0


# --------------------------------------------------------------------------- #
# 5. Every decision is logged, with the whole distribution
# --------------------------------------------------------------------------- #
class TestLogIsTheDataset:
    async def test_records_carry_the_full_ladder_for_every_question(self) -> None:
        records: list[DecisionRecord] = []
        policy = Policy.from_file(REPO_ROOT / "policies" / "default.yaml").with_overrides(
            logging={"enabled": False}, cache={"enabled": False}
        )
        router = Router(policy, MockBackend(), sink=CallbackSink(records.append), cache=NullCache())
        await router.route_text("Summarize the roadmap for the team.")
        await router.route_text("Charge card 4111 1111 1111 1111 now.")

        assert len(records) == 2
        for record in records:
            payload = json.loads(record.to_json())
            answers = payload["decision"]["answers"]
            assert sorted(answers["complexity"]["probabilities"]) == sorted(COMPLEXITY_LEVELS)
            assert sorted(answers["sensitivity"]["probabilities"]) == sorted(SENSITIVITY_LEVELS)
            assert sorted(answers["domain"]["probabilities"]) == sorted(DOMAINS)
            assert "noul" in answers["pii"]

    async def test_json_keys_are_sorted_and_the_version_is_always_present(self) -> None:
        records: list[DecisionRecord] = []
        policy = Policy.from_file(REPO_ROOT / "policies" / "default.yaml").with_overrides(
            logging={"enabled": False}, cache={"enabled": False}
        )
        router = Router(policy, MockBackend(), sink=CallbackSink(records.append), cache=NullCache())
        await router.route_text("hello there")
        await router.route_text("Charge card 4111 1111 1111 1111 now.")

        for record in records:
            line = record.to_json()
            assert "\n" not in line, "one record per line, always"
            data = json.loads(line)
            assert list(data) == sorted(data), "to_json must sort keys so diffs and hashes are stable"
            assert data["schema_version"] == SCHEMA_VERSION
            assert data["kind"] == "jev_route.decision"

    def test_schema_version_is_stamped_by_default(self) -> None:
        record = DecisionRecord.from_dict(
            {
                "request_id": "r",
                "timestamp": "t",
                "decision": {
                    "tier": "cheap", "model": "m", "rule_id": "d", "reason": "r",
                    "answers": {
                        "complexity": {"choice": "trivial", "probabilities": {"trivial": 1.0}, "confidence": 1.0},
                        "sensitivity": {"choice": "public", "probabilities": {"public": 1.0}, "confidence": 1.0},
                        "pii": {"noul": 0.0},
                        "domain": {"choice": "chat", "probabilities": {"chat": 1.0}, "confidence": 1.0},
                    },
                    "backend": "mock", "effective_sensitivity": "public", "effective_complexity": "trivial",
                },
            }
        )
        assert record.schema_version == SCHEMA_VERSION

    def test_gate_findings_keep_only_a_hash(self) -> None:
        gate = HardGate()
        verdict = gate.scan("Charge card 4111 1111 1111 1111 now.")
        blob = json.dumps(verdict.to_dict())
        assert "4111" not in blob
        assert all(len(f.span_hash) == 16 for f in verdict.findings)

    async def test_every_routed_request_produces_exactly_one_record(self) -> None:
        records: list[DecisionRecord] = []
        policy = Policy.from_file(REPO_ROOT / "policies" / "default.yaml").with_overrides(
            logging={"enabled": False}, cache={"enabled": False}
        )
        router = Router(policy, MockBackend(), sink=CallbackSink(records.append), cache=NullCache())
        texts = [
            "hello",
            "prove the spectral theorem",
            "Charge card 4111 1111 1111 1111",
            "mail dana.kovacs@northside-health.org",
        ]
        for text in texts:
            await router.route_text(text)
        assert len(records) == len(texts)
        assert len({r.request_id for r in records}) == len(texts)


# --------------------------------------------------------------------------- #
# 6. House style, enforced mechanically
# --------------------------------------------------------------------------- #
class TestHouseStyle:
    def test_every_module_has_a_docstring(self, sources: dict[str, Any]) -> None:
        """This is a public repo; a module without a stated purpose will be misread."""
        missing = [name for name, (_, tree, _) in sources.items() if module_docstring(tree) is None]
        assert not missing, f"modules without a docstring: {missing}"

    def test_every_module_uses_future_annotations(self, sources: dict[str, Any]) -> None:
        missing = [name for name, (_, tree, _) in sources.items() if not imports_future_annotations(tree)]
        assert not missing, f"modules missing `from __future__ import annotations`: {missing}"

    def test_no_bare_excepts(self, sources: dict[str, Any]) -> None:
        """A bare except swallows KeyboardInterrupt and hides the failure you came for."""
        offenders = [
            f"{name}:{node.lineno}"
            for name, (_, tree, _) in sources.items()
            for node in ast.walk(tree)
            if isinstance(node, ast.ExceptHandler) and node.type is None
        ]
        assert not offenders, f"bare excepts at {offenders}"

    def test_eval_is_confined_to_the_policy_sandbox(self, sources: dict[str, Any]) -> None:
        """One eval in the whole tree, behind the AST whitelist, with no builtins."""
        offenders = [name for name, (_, tree, _) in sources.items() if name != "policy.py" and _calls(tree, "eval")]
        assert not offenders, f"eval() outside policy.py in {offenders}"
        _, tree, text = sources["policy.py"]
        assert _calls(tree, "eval")
        assert '{"__builtins__": {}}' in text, "the sandbox must clear builtins"

    def test_exec_and_compile_of_user_input_are_absent(self, sources: dict[str, Any]) -> None:
        for name, (_, tree, _) in sources.items():
            if name == "policy.py":
                continue  # compile() of the whitelisted AST is the sandbox itself
            assert not _calls(tree, "exec"), f"{name} calls exec()"

    def test_no_hardcoded_credentials(self, sources: dict[str, Any]) -> None:
        """The repo is public. A key committed here is a key revoked tomorrow.

        The scan wants a provider prefix *followed by realistic key material*, not the
        bare prefix. ``gate.py`` is a credential detector: its ``provider_api_key``
        regex has to spell out ``AKIA``, ``ghp_`` and the rest in order to find them in
        a user's prompt, so a prefix-only scan can never pass against the one module
        whose job is recognising credentials. A check that fires on the detector and
        not on a leak is worse than no check -- the usual outcome is that someone
        allowlists the detector and stops reading the failure.
        """
        offenders = [
            f"{name}: {where}" for name, (_, _, text) in sources.items() for where in find_planted_credentials(text)
        ]
        assert not offenders, f"hardcoded credentials: {offenders}"

    def test_the_credential_scan_still_finds_a_credential(self, tmp_path: Path) -> None:
        """A scanner that cannot find a secret is worse than no scanner at all.

        The check above is a negative control, and negatives are cheap to satisfy:
        loosening a regex until it passes is the failure mode this test exists to
        catch. So plant three synthetic keys in a temp file and require all three to
        be flagged, then require the detector's own character classes to stay quiet.

        The planted values are assembled from parts, so *this* file never holds a
        contiguous credential literal -- a repo-wide scanner reading the test suite
        would otherwise (rightly) flag the file that is complaining about them.
        """
        planted_keys = {
            "AKIA": "AKIA" + "Q7X9M2KLP4ZRW8TJ",
            "sk-ant-": "sk-ant-" + "api03-H4J7K9M2P5Q8R1T4V6",
            "ghp_": "ghp_" + "aB3dE5fG7hJ9kL1mN3pQ5rS7tU9vW1xZ3bD5",
            "xoxb-": "xoxb-" + "123456789012-Q7X9M2KLP4ZRW8",
            "AIzaSy": "AIzaSy" + "A1B2C3D4E5F6G7H8I9J0K1",
            "hf_": "hf_" + "A1B2C3D4E5F6G7H8I9J0K1L2M3N4",
        }
        planted = tmp_path / "leak.py"
        planted.write_text(
            "".join(f'leaked_{index} = "{value}"\n' for index, value in enumerate(planted_keys.values())),
            encoding="utf-8",
        )
        found = find_planted_credentials(planted.read_text(encoding="utf-8"))
        assert len(found) == len(planted_keys), f"the scan missed a planted key: {found}"
        assert sorted(planted_keys) == sorted(where.split("...")[0] for where in found)

        detector_source = (
            'r"|AKIA[0-9A-Z]{16}"      # AWS access key id\n'
            'r"|ghp_[A-Za-z0-9]{36}"   # GitHub PAT\n'
            'r"|hf_[A-Za-z0-9]{34}"    # Hugging Face\n'
            'r"|sk-ant-[A-Za-z0-9_\\-]{20,}"  # Anthropic\n'
        )
        assert find_planted_credentials(detector_source) == [], "the scan fired on a detector's regex"

    def test_no_module_reads_the_env_file(self, sources: dict[str, Any]) -> None:
        for name, (_, _, text) in sources.items():
            assert '".env"' not in text and "'.env'" not in text, f"{name} references a .env file"


def _calls(tree: ast.Module, func_name: str) -> bool:
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == func_name
        for node in ast.walk(tree)
    )


def module_docstring(tree: ast.Module) -> str | None:
    """The module docstring, or ``None`` when the first statement is not one."""
    first = tree.body[0] if tree.body else None
    if not isinstance(first, ast.Expr):
        return None
    value = first.value
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    return None


def imports_future_annotations(tree: ast.Module) -> bool:
    """Whether the module carries ``from __future__ import annotations``."""
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )


# --------------------------------------------------------------------------- #
# 7. The graduation story holds together
# --------------------------------------------------------------------------- #
class TestGraduationStory:
    def test_the_backend_is_swappable_by_config_alone(self) -> None:
        """Bootstrap on the cloud, graduate to local: one key in the policy file."""
        from jev_route.backends import build_backend

        policy = Policy.from_file(REPO_ROOT / "policies" / "default.yaml")
        assert isinstance(build_backend(policy), MockBackend)
        graduated = policy.with_backend({"name": "mock", "temperature": 0.3})
        assert isinstance(build_backend(graduated), MockBackend)
        assert build_backend(graduated).temperature == 0.3

    def test_the_router_does_not_know_which_backend_exists(self, sources: dict[str, Any]) -> None:
        """``router.py`` must not name JevBackend: the swap has to be config-only."""
        _, _, text = sources["router.py"]
        assert "JevBackend" not in text
        assert "MockBackend" not in text
        assert "DistilledBackend" not in text

    def test_the_router_imports_no_backend_implementation(self, sources: dict[str, Any]) -> None:
        """``router.py`` may touch the backend protocol and the factory, and nothing else.

        The previous form of this check asked whether ``"jev_route.backends.jev"`` was
        among the *first components* of the imports, which is a set that can only ever
        hold ``"jev_route"`` -- it passed on exactly the violation it existed to catch.
        Full dotted paths make it a real check, and naming the allowed surface keeps a
        future ``from .backends.mock import MockBackend`` from slipping through as
        "just another backends import".
        """
        _, tree, _ = sources["router.py"]
        paths = imported_module_paths(tree)
        implementations = {"jev", "mock", "distilled", "shadow"}
        named = sorted(path for path in paths if path.rsplit(".", 1)[-1] in implementations)
        assert not named, f"router.py imports backend implementations: {named}"
        backend_surface = sorted(path for path in paths if "backends" in path.split("."))
        assert set(backend_surface) <= {"backends", "backends.base"}, (
            f"router.py must only see the protocol and the factory, got: {backend_surface}"
        )

    def test_the_decision_schema_is_backend_agnostic(self) -> None:
        """The same four answers whichever backend produced them -- that is the point."""
        from jev_route.schema import DecisionAnswers

        unknown = DecisionAnswers.unknown()
        assert set(unknown.complexity.probabilities) == set(COMPLEXITY_LEVELS)
        assert set(unknown.sensitivity.probabilities) == set(SENSITIVITY_LEVELS)
        assert set(unknown.domain.probabilities) == set(DOMAINS)
        assert unknown.pii.value == 0.5

    def test_schema_version_is_a_string_and_documented(self) -> None:
        assert isinstance(SCHEMA_VERSION, str)
        assert SCHEMA_VERSION == "1"
