"""Tests for :mod:`jev_route.backends.distilled` -- the local end state.

The backend is small; what it promises is not. Three promises get most of the
attention here, because each one is a claim the README makes and a sceptical
operator would test first:

1. **It never raises.** A missing, truncated, checksum-mismatched or
   schema-stale artifact becomes ``degraded=True`` with maximum-uncertainty
   answers, so the policy's ``on_backend_down`` fails *closed*. A routing
   decision that 500s because a model file is bad is the one failure mode this
   package is not allowed to have.
2. **numpy is lazy.** Checked statically (no heavy import outside a function
   body, no ``distill`` import at module level at all) and dynamically, in a
   subprocess where numpy is blocked at the import system -- the same trick
   ``tests/test_invariants.py`` uses for the core.
3. **Nothing leaves.** No HTTP client, no socket, no URL, no hostname.

Artifacts are built directly from the public ``distill.artifact`` API rather
than by running a training job: these tests are about *serving* an artifact, and
tying them to the trainer would make every failure here ambiguous between two
modules. ``tests/distill`` covers training.

No network and no API key anywhere, which ``conftest.py``'s autouse ``_offline``
fixture enforces mechanically.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from jev_route import Policy
from jev_route.backends import build_backend
from jev_route.backends.base import DecisionRequest
from jev_route.backends.distilled import UNLOADED_MODEL_VERSION, DistilledBackend
from jev_route.prompts import compute_features
from jev_route.schema import COMPLEXITY_LEVELS, DOMAINS, SENSITIVITY_LEVELS, DecisionAnswers, RequestFeatures

MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "jev_route" / "backends" / "distilled.py"
HEAVY = {"numpy", "torch", "scipy", "sklearn", "pandas", "redis", "transformers"}
EGRESS = {"httpx", "requests", "aiohttp", "urllib3", "http", "socket", "urllib"}

PROMPTS: tuple[str, ...] = (
    "hi there, thanks!",
    "Write a Python function that reads a CSV and sums one column.",
    "Design a sharded write path with consensus and prove the safety invariant holds.",
    "Summarize this customer support thread and extract the account ids.",
    "Explain how HIPAA actually works and who it applies to.",
    "Translate the release notes into German and keep the code blocks intact.",
)


# --------------------------------------------------------------------------- #
# helpers and fixtures
# --------------------------------------------------------------------------- #
def _require_numpy() -> Any:
    """numpy belongs to the ``distill`` extra, so skip rather than fail without it."""
    try:
        import numpy
    except ImportError:  # pragma: no cover - depends on the environment
        pytest.skip("the distill extra is not installed: pip install 'jev-route[distill]'")
    return numpy


def _feature_rows(texts: Sequence[str] = PROMPTS) -> list[RequestFeatures]:
    return [compute_features(text) for text in texts]


def _metadata(model: Any, *, mode: str, version: str) -> dict[str, Any]:
    """The smallest envelope ``DistilledArtifact.load`` accepts."""
    from jev_route.distill.artifact import ARTIFACT_KIND, ARTIFACT_SCHEMA_VERSION

    return {
        "kind": ARTIFACT_KIND,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_version": version,
        "created_at": "2026-09-19T00:00:00+00:00",
        "mode": mode,
        "contains_prompt_text": mode == "text",
        "model": model.to_dict(),
        "teacher": {"name": "mock", "model_versions": ["mock-1.0.0"]},
        "dataset": {"source": "tests", "rows": len(PROMPTS)},
    }


def _build_artifact(target: Path, *, mode: str, version: str = "test-distilled-1") -> Path:
    from jev_route.distill.artifact import (
        DistilledArtifact,
        FeatureVectorizer,
        StudentModel,
        TextVectorizer,
    )

    if mode == "text":
        vectorizer: Any = TextVectorizer(max_features=256, ngram_max=2).fit(list(PROMPTS))
    else:
        vectorizer = FeatureVectorizer().fit(_feature_rows())
    model = StudentModel.initialize(vectorizer.n_features, hidden=0, seed=7)
    artifact = DistilledArtifact(
        model=model,
        vectorizer=vectorizer,
        metadata=_metadata(model, mode=mode, version=version),
        metrics=None,
        dataset={"rows": len(PROMPTS), "mode": mode, "train_rows": 5, "holdout_rows": 1},
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    return artifact.save(target)


@pytest.fixture(scope="module")
def features_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    _require_numpy()
    return _build_artifact(tmp_path_factory.mktemp("artifacts") / "features-v1", mode="features")


@pytest.fixture(scope="module")
def text_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    _require_numpy()
    return _build_artifact(tmp_path_factory.mktemp("artifacts") / "text-v1", mode="text")


@pytest.fixture
def request_features() -> DecisionRequest:
    """A normal, non-gated request: the case that must be answered, not degraded."""
    text = PROMPTS[1]
    return DecisionRequest(redacted_excerpt=text, features=compute_features(text))


@pytest.fixture
def broken_artifact(tmp_path: Path) -> Path:
    """A directory that looks like an artifact and holds no ``artifact.json``."""
    target = tmp_path / "not-an-artifact"
    target.mkdir()
    (target / "README.txt").write_text("this is not an artifact\n", encoding="utf-8")
    return target


@pytest.fixture
def corrupt_artifact(features_artifact: Path, tmp_path: Path) -> Path:
    """A real artifact with one weight byte flipped, so the checksum no longer matches."""
    _require_numpy()
    import shutil

    target = tmp_path / "corrupt"
    shutil.copytree(features_artifact, target)
    weights = target / "W.npy"
    blob = bytearray(weights.read_bytes())
    blob[-1] ^= 0xFF
    weights.write_bytes(bytes(blob))
    return target


def _policy_with(backend: dict[str, Any], *, fail_mode: str = "fail_closed") -> Policy:
    from tests.conftest import make_policy_doc

    return Policy.from_dict(make_policy_doc(backend=backend, on_backend_down={"mode": fail_mode}))


# --------------------------------------------------------------------------- #
# 1. construction never raises
# --------------------------------------------------------------------------- #
class TestConstructionNeverRaises:
    """A bad artifact path is a runtime condition, not a programming error."""

    @pytest.mark.parametrize(
        "path",
        [
            pytest.param("does/not/exist", id="missing"),
            pytest.param("", id="empty-string"),
        ],
    )
    def test_a_missing_artifact_constructs(self, path: str) -> None:
        backend = DistilledBackend(path)
        assert isinstance(backend, DistilledBackend)
        assert backend.is_loaded is False
        assert backend.model_version == UNLOADED_MODEL_VERSION

    def test_a_directory_that_is_not_an_artifact_constructs(self, broken_artifact: Path) -> None:
        backend = DistilledBackend(broken_artifact)
        assert backend.is_loaded is False

    def test_a_corrupt_artifact_constructs(self, corrupt_artifact: Path) -> None:
        backend = DistilledBackend(corrupt_artifact)
        assert backend.is_loaded is False

    def test_a_plain_text_file_constructs(self, tmp_path: Path) -> None:
        target = tmp_path / "artifact.txt"
        target.write_text("definitely not a zip\n", encoding="utf-8")
        assert DistilledBackend(target).is_loaded is False

    def test_construction_does_not_read_the_artifact(self, features_artifact: Path) -> None:
        """Loading is deferred to the first decision, so startup stays cheap."""
        backend = DistilledBackend(features_artifact)
        assert backend.is_loaded is False
        assert backend.preload() is True
        assert backend.is_loaded is True


# --------------------------------------------------------------------------- #
# 2. the degraded path: fail closed, never raise
# --------------------------------------------------------------------------- #
class TestDegradesInsteadOfRaising:
    """Every broken-artifact case must answer, at maximum uncertainty."""

    @pytest.mark.parametrize(
        "fixture_name",
        ["broken_artifact", "corrupt_artifact"],
    )
    async def test_a_bad_artifact_degrades(self, request: pytest.FixtureRequest, fixture_name: str) -> None:
        backend = DistilledBackend(request.getfixturevalue(fixture_name))
        result = await backend.decide(
            DecisionRequest(redacted_excerpt=PROMPTS[1], features=compute_features(PROMPTS[1]))
        )
        assert result.degraded is True
        assert result.degrade_reason
        assert result.answers == DecisionAnswers.unknown()
        assert result.model_version == UNLOADED_MODEL_VERSION

    async def test_a_missing_artifact_degrades_with_the_path_in_the_reason(self, tmp_path: Path) -> None:
        missing = tmp_path / "gone"
        backend = DistilledBackend(missing)
        result = await backend.decide(
            DecisionRequest(redacted_excerpt=PROMPTS[0], features=compute_features(PROMPTS[0]))
        )
        assert result.degraded is True
        assert "gone" in (result.degrade_reason or "")

    async def test_a_missing_features_object_degrades(self, features_artifact: Path) -> None:
        """A features-mode artifact handed a request with no features at all."""
        backend = DistilledBackend(features_artifact)
        result = await backend.decide(DecisionRequest(redacted_excerpt=PROMPTS[1], features=None))  # type: ignore[arg-type]
        assert result.degraded is True
        assert result.answers == DecisionAnswers.unknown()
        assert "RequestFeatures" in (result.degrade_reason or "")

    async def test_a_feature_vector_the_vectorizer_cannot_build_degrades(self, features_artifact: Path) -> None:
        """The vectorizer blowing up mid-transform must not escape as an exception."""

        class Exploding:
            """Quacks like RequestFeatures, then fails on the first attribute read."""

            def __getattr__(self, name: str) -> Any:
                raise RuntimeError(f"cannot compute {name}")

        backend = DistilledBackend(features_artifact)
        result = await backend.decide(
            DecisionRequest(redacted_excerpt=PROMPTS[1], features=Exploding())  # type: ignore[arg-type]
        )
        assert result.degraded is True
        assert result.answers == DecisionAnswers.unknown()
        assert "cannot compute" in (result.degrade_reason or "")

    async def test_a_text_artifact_with_no_excerpt_degrades(self, text_artifact: Path) -> None:
        """Gate-blocked requests are never stored as text, so there is nothing to read."""
        backend = DistilledBackend(text_artifact)
        result = await backend.decide(DecisionRequest(redacted_excerpt="", features=compute_features(PROMPTS[1])))
        assert result.degraded is True
        assert "text mode" in (result.degrade_reason or "")

    async def test_the_degrade_reason_is_stable_across_calls(self, broken_artifact: Path) -> None:
        backend = DistilledBackend(broken_artifact)
        request = DecisionRequest(redacted_excerpt=PROMPTS[0], features=compute_features(PROMPTS[0]))
        first = await backend.decide(request)
        second = await backend.decide(request)
        assert first.degrade_reason == second.degrade_reason
        assert first.answers == second.answers


class TestFailClosedThroughTheRouter:
    """The point of degrading: the policy then keeps the data local."""

    async def test_a_broken_artifact_lands_on_the_fail_closed_tier(
        self, broken_artifact: Path, router_factory: Any, sink: Any
    ) -> None:
        policy = _policy_with({"name": "distilled", "artifact": str(broken_artifact)})
        router = router_factory(DistilledBackend(broken_artifact), pol=policy)
        decision = await router.route_text(PROMPTS[1])
        await router.aclose()
        assert decision.degraded is True
        assert decision.tier == "local"
        assert sink.count == 1
        assert sink.last.decision.degraded is True

    async def test_the_policy_decides_the_degraded_tier_not_the_backend(
        self, broken_artifact: Path, router_factory: Any
    ) -> None:
        """fail_open is the operator's choice; the backend only reports it cannot answer."""
        policy = _policy_with({"name": "distilled", "artifact": str(broken_artifact)}, fail_mode="fail_open")
        router = router_factory(DistilledBackend(broken_artifact), pol=policy)
        decision = await router.route_text(PROMPTS[1])
        await router.aclose()
        assert decision.tier == "strong"
        assert decision.degraded is True


# --------------------------------------------------------------------------- #
# 3. serving
# --------------------------------------------------------------------------- #
class TestServingAnArtifact:
    async def test_a_features_artifact_answers(
        self, features_artifact: Path, request_features: DecisionRequest
    ) -> None:
        _require_numpy()
        backend = DistilledBackend(features_artifact)
        result = await backend.decide(request_features)
        assert result.degraded is False, result.degrade_reason
        assert result.model_version == "test-distilled-1"
        assert set(result.answers.complexity.probabilities) == set(COMPLEXITY_LEVELS)
        assert set(result.answers.sensitivity.probabilities) == set(SENSITIVITY_LEVELS)
        assert set(result.answers.domain.probabilities) == set(DOMAINS)
        assert 0.0 <= result.answers.pii.value <= 1.0
        assert result.latency_ms >= 0.0
        assert result.questions_sent == {}

    async def test_the_distributions_sum_to_one(
        self, features_artifact: Path, request_features: DecisionRequest
    ) -> None:
        _require_numpy()
        result = await DistilledBackend(features_artifact).decide(request_features)
        for answer in (result.answers.complexity, result.answers.sensitivity, result.answers.domain):
            assert sum(answer.probabilities.values()) == pytest.approx(1.0, abs=1e-5)
            assert answer.choice in answer.probabilities

    async def test_answers_are_deterministic(self, features_artifact: Path, request_features: DecisionRequest) -> None:
        _require_numpy()
        backend = DistilledBackend(features_artifact)
        first = await backend.decide(request_features)
        second = await backend.decide(request_features)
        assert first.answers.to_dict() == second.answers.to_dict()

    async def test_a_text_artifact_answers_from_the_excerpt(self, text_artifact: Path) -> None:
        _require_numpy()
        backend = DistilledBackend(text_artifact)
        result = await backend.decide(
            DecisionRequest(redacted_excerpt=PROMPTS[2], features=compute_features(PROMPTS[2]))
        )
        assert result.degraded is False, result.degrade_reason
        assert backend.mode == "text"

    async def test_feature_mode_true_serves_a_features_artifact(self, features_artifact: Path) -> None:
        _require_numpy()
        backend = DistilledBackend(features_artifact, feature_mode=True)
        result = await backend.decide(
            DecisionRequest(redacted_excerpt=PROMPTS[1], features=compute_features(PROMPTS[1]))
        )
        assert result.degraded is False, result.degrade_reason

    async def test_feature_mode_true_against_a_text_artifact_degrades_with_a_fix(self, text_artifact: Path) -> None:
        """The config asserts something the artifact contradicts: say so, do not guess."""
        backend = DistilledBackend(text_artifact, feature_mode=True)
        result = await backend.decide(
            DecisionRequest(redacted_excerpt=PROMPTS[1], features=compute_features(PROMPTS[1]))
        )
        assert result.degraded is True
        reason = result.degrade_reason or ""
        assert "feature_mode" in reason
        assert "jev-route" in reason

    def test_info_reports_what_an_operator_needs(self, features_artifact: Path) -> None:
        _require_numpy()
        backend = DistilledBackend(features_artifact)
        assert backend.info()["loaded"] is False
        backend.preload()
        info = backend.info()
        assert info["loaded"] is True
        assert info["mode"] == "features"
        assert info["model_version"] == "test-distilled-1"
        assert info["contains_prompt_text"] is False
        assert info["teacher_model_versions"] == ["mock-1.0.0"]
        assert isinstance(info["n_features"], int) and info["n_features"] > 0

    def test_info_reports_a_load_failure(self, broken_artifact: Path) -> None:
        backend = DistilledBackend(broken_artifact)
        assert backend.preload() is False
        assert backend.info()["load_error"]

    def test_preload_retries_a_cached_failure(self, tmp_path: Path) -> None:
        """An artifact mounted after startup becomes servable without a restart."""
        _require_numpy()
        target = tmp_path / "later"
        backend = DistilledBackend(target)
        assert backend.preload() is False
        _build_artifact(target, mode="features", version="test-distilled-2")
        assert backend.preload() is True
        assert backend.model_version == "test-distilled-2"

    async def test_aclose_is_idempotent_and_does_not_drop_the_model(self, features_artifact: Path) -> None:
        _require_numpy()
        backend = DistilledBackend(features_artifact)
        backend.preload()
        await backend.aclose()
        await backend.aclose()
        result = await backend.decide(
            DecisionRequest(redacted_excerpt=PROMPTS[1], features=compute_features(PROMPTS[1]))
        )
        assert result.degraded is False


class TestTheFactoryAndTheRouter:
    def test_build_backend_resolves_the_name(self, features_artifact: Path) -> None:
        backend = build_backend({"name": "distilled", "artifact": str(features_artifact)})
        assert isinstance(backend, DistilledBackend)
        assert backend.name == "distilled"
        assert backend.artifact_path == str(features_artifact)
        assert backend.feature_mode is False

    def test_build_backend_passes_feature_mode_through(self, features_artifact: Path) -> None:
        backend = build_backend({"name": "distilled", "artifact": str(features_artifact), "feature_mode": True})
        assert isinstance(backend, DistilledBackend)
        assert backend.feature_mode is True

    def test_the_default_artifact_path_is_config_facing(self) -> None:
        backend = build_backend({"name": "distilled"})
        assert isinstance(backend, DistilledBackend)
        assert backend.artifact_path.endswith("artifacts/jev-route-distilled")

    async def test_the_router_stamps_the_student_as_the_teacher_of_record(
        self, features_artifact: Path, router_factory: Any, sink: Any
    ) -> None:
        """The log is the next dataset, so provenance has to name the local model."""
        _require_numpy()
        policy = _policy_with({"name": "distilled", "artifact": str(features_artifact)})
        router = router_factory(DistilledBackend(features_artifact), pol=policy)
        decision = await router.route_text(PROMPTS[1])
        await router.aclose()
        assert decision.backend == "distilled"
        assert decision.backend_model_version == "test-distilled-1"
        assert decision.degraded is False
        assert decision.tier in {"cheap", "strong", "local"}
        record = sink.last
        assert record.decision.backend == "distilled"
        assert len(record.decision.answers.complexity.probabilities) == len(COMPLEXITY_LEVELS)
        assert record.decision.answers.pii.value == pytest.approx(decision.answers.pii.value)

    async def test_the_gate_still_runs_before_the_student(self, features_artifact: Path, router_factory: Any) -> None:
        """Graduating changes the classifier, never the gate."""
        _require_numpy()
        policy = _policy_with({"name": "distilled", "artifact": str(features_artifact)})
        backend = DistilledBackend(features_artifact)
        router = router_factory(backend, pol=policy)
        decision = await router.route_text("Patient John Smith, SSN 666-45-1234, MRN: ABC-9931.")
        await router.aclose()
        assert decision.gate.force_local is True
        assert decision.tier == "local"
        assert decision.backend == "gate"
        assert backend.is_loaded is False, "the gate blocked the request, so the student was never called"

    async def test_a_broken_artifact_never_reaches_the_cloud(self, broken_artifact: Path, router_factory: Any) -> None:
        """fail_closed means local hardware, which is the whole privacy argument."""
        policy = _policy_with({"name": "distilled", "artifact": str(broken_artifact)})
        router = router_factory(DistilledBackend(broken_artifact), pol=policy)
        decision = await router.route_text("Summarize the roadmap and mention the patient record.")
        await router.aclose()
        assert decision.tier == "local"
        assert decision.degraded is True


# --------------------------------------------------------------------------- #
# 4. numpy stays out of the import path
# --------------------------------------------------------------------------- #
def _module_tree() -> ast.Module:
    return ast.parse(MODULE_PATH.read_text(encoding="utf-8"), filename=str(MODULE_PATH))


def _function_scoped_import_lines(tree: ast.Module) -> set[int]:
    ranges = [
        (node.lineno, node.end_lineno or node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    return {
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom)) and any(start <= node.lineno <= end for start, end in ranges)
    }


def _imported_roots(tree: ast.Module) -> list[tuple[str, int]]:
    """Absolute imports only, as ``(top-level name, line)``.

    Relative imports are excluded on purpose: ``from ..schema import ...`` has a
    ``module`` of ``"schema"``, which is a package sibling and not a third-party
    root, and reporting it as one would make every check below useless. The
    relative surface -- which is where a stray ``distill`` import would hide -- is
    scanned separately by :meth:`TestHeavyDependenciesAreLazy.test_no_distill_import_at_module_level`.
    """
    out: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend((alias.name.split(".")[0], node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append((node.module.split(".")[0], node.lineno))
    return out


def _string_literals(tree: ast.Module) -> list[tuple[int, str]]:
    return [
        (node.lineno, str(node.value))
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


class TestHeavyDependenciesAreLazy:
    """Mirrors ``tests/test_invariants.py``, scoped to this one module."""

    def test_no_heavy_import_at_module_level(self) -> None:
        tree = _module_tree()
        lazy = _function_scoped_import_lines(tree)
        offenders = [
            f"line {lineno} imports {name}"
            for name, lineno in _imported_roots(tree)
            if name in HEAVY and lineno not in lazy
        ]
        assert not offenders, f"heavy imports outside a function body: {offenders}"

    def test_the_module_does_not_import_numpy_at_all(self) -> None:
        """numpy is reached only through ``distill.artifact.require_numpy``."""
        names = {name for name, _ in _imported_roots(_module_tree())}
        assert "numpy" not in names

    def test_no_distill_import_at_module_level(self) -> None:
        """``import jev_route.backends.distilled`` must not drag the pipeline in."""
        tree = _module_tree()
        lazy = _function_scoped_import_lines(tree)
        offenders = [
            f"line {lineno}" for name, lineno in _imported_roots(tree) if name == "jev_route" and lineno not in lazy
        ]
        relative = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level and (node.module or "").startswith("distill")
        ]
        assert not offenders, f"absolute jev_route imports at module level: {offenders}"
        assert not [line for line in relative if line not in lazy], "a distill import escaped to module level"

    def test_module_level_imports_are_stdlib_and_local_only(self) -> None:
        stdlib = set(sys.stdlib_module_names)
        external = [
            name
            for name, lineno in _imported_roots(_module_tree())
            if lineno not in _function_scoped_import_lines(_module_tree())
            and name not in stdlib
            and name != "jev_route"
        ]
        assert not external, f"module-level third-party imports: {external}"

    def test_importing_the_module_does_not_pull_in_numpy(self) -> None:
        code = (
            "import sys; import jev_route.backends.distilled as d;"
            "b = d.DistilledBackend('does/not/exist');"
            "print('numpy' in sys.modules, b.name, d.UNLOADED_MODEL_VERSION)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(MODULE_PATH.parents[3]),
        )
        assert result.returncode == 0, result.stderr
        pulled, name, placeholder = result.stdout.strip().split()
        assert pulled == "False", "importing the backend pulled numpy in"
        assert name == "distilled"
        assert placeholder == UNLOADED_MODEL_VERSION

    def test_a_degraded_decision_needs_no_numpy(self) -> None:
        """With the extra absent, the backend still answers -- degraded, fail closed."""
        code = """
import asyncio, sys

BLOCKED = {"numpy", "torch", "scipy", "sklearn", "pandas", "redis", "transformers"}


class Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ImportError(f"{fullname!r} is blocked in this probe")
        return None


sys.meta_path.insert(0, Blocker())

from jev_route.backends.distilled import DistilledBackend
from jev_route.backends.base import DecisionRequest
from jev_route.prompts import compute_features
from jev_route.schema import DecisionAnswers

backend = DistilledBackend("whatever")
result = asyncio.run(backend.decide(DecisionRequest(redacted_excerpt="hi", features=compute_features("hi"))))
print(result.degraded, result.answers == DecisionAnswers.unknown(), "numpy" in sys.modules)
print("distill" in (result.degrade_reason or ""))
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(MODULE_PATH.parents[3]),
        )
        assert result.returncode == 0, result.stderr
        first, second = result.stdout.strip().splitlines()
        degraded, unknown, numpy_loaded = first.split()
        assert (degraded, unknown, numpy_loaded) == ("True", "True", "False")
        assert second == "True", "the degrade reason must name the extra to install"


class TestNoEgress:
    def test_no_network_client_or_socket(self) -> None:
        names = {name for name, _ in _imported_roots(_module_tree())}
        assert not (names & EGRESS), f"the local backend imports a way off the machine: {sorted(names & EGRESS)}"

    def test_no_string_literal_names_a_network_location(self) -> None:
        """Scan the literals, not the prose.

        A whole-file substring scan is the wrong instrument here: this module's
        own docstring has to *say* "no HTTP client, no socket, no URL", and a
        needle for "http" would fail on the sentence that makes the promise. What
        can actually egress is a string that ends up in a request, so those are
        what gets checked -- for a URL scheme, and for a hostname-shaped token.
        """
        hostname = re.compile(r"\b[a-z0-9][\w-]*(?:\.[\w-]+)*\.(?:com|ai|io|org|net|dev|cloud)\b", re.IGNORECASE)
        offenders: list[str] = []
        for lineno, text in _string_literals(_module_tree()):
            if "://" in text:
                offenders.append(f"line {lineno}: a URL scheme")
                continue
            match = hostname.search(text)
            if match:
                offenders.append(f"line {lineno}: {match.group(0)!r}")
        assert not offenders, f"the local end state names a network location: {offenders}"

    def test_no_egress_helper_is_imported_even_lazily(self) -> None:
        """Not even inside a function: there is no code path that could reach out."""
        names = {name for name, _ in _imported_roots(_module_tree())}
        assert not (names & EGRESS), f"a network module is imported somewhere: {sorted(names & EGRESS)}"
