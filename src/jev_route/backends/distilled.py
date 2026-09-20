"""DistilledBackend: serve routing decisions from a model you trained yourself.

This is the local end state of the lifecycle the rest of the package exists to
feed. Day one the router asks a cloud teacher; the decision log accumulates;
``jev-route export`` / ``train`` / ``package`` turn that log into an artifact
directory; and then ``backend.name: distilled`` in the policy file -- one key, no
code change -- makes this module answer instead. After that the egress stops.

It is a :class:`~jev_route.backends.base.DecisionBackend` like every other
backend, which is the whole reason the cutover is a config change: the router,
the gate, the policy engine and the decision log cannot tell the difference
between this and a cloud call, and do not need to.

Three properties are load-bearing, and each one has a test:

**It never raises.** A routing decision must not fail because a model file is
missing, truncated, checksum-mismatched, trained against an older schema, or
because numpy is not installed. Every one of those becomes
``BackendResult(degraded=True)`` carrying
:func:`~jev_route.schema.DecisionAnswers.unknown`, which the policy's
``on_backend_down`` section then fails *closed* -- ``fail_closed_tier: local``, so
a broken local model sends traffic to your own hardware rather than to a cloud
API. :class:`~jev_route.distill.artifact.ArtifactError` documents this contract
from the other side. Raising instead would turn a bad file into a 500 on the
request path, which is the one failure mode a router is not allowed to have.

**numpy is lazy.** ``import jev_route.backends.distilled`` costs nothing but
stdlib, and the numerical stack is pulled in by
:func:`~jev_route.distill.artifact.require_numpy` only when an artifact is
actually loaded. A deployment that routes on Jev or the mock never pays for it,
and ``tests/test_invariants.py`` enforces the property in a subprocess where
numpy is blocked at the import system.

**Nothing leaves.** There is no client, no socket and no URL in this module. The
only thing it reads is a directory of JSON and raw ``.npy`` arrays on local disk,
which is what "you end up owning it" has to mean mechanically.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..schema import DecisionAnswers
from .base import BackendResult, DecisionRequest

#: Stamped on a decision when the artifact could not be read at all, so the log
#: stays honest about provenance instead of claiming a version that never answered.
#: ``export`` skips degraded records, so these rows never reach a training set.
UNLOADED_MODEL_VERSION = "distilled-unavailable"


class _CannotDecideError(Exception):
    """This artifact cannot answer this particular request. Always carries the reason."""


class DistilledBackend:
    """A decision backend served by a locally trained artifact.

    ``artifact`` is the directory (or zip) written by ``jev-route package`` or
    ``jev-route train``. It is loaded on first use and then held for the life of
    the instance, because a Router is constructed once per process and the weights
    are a few tens of kilobytes.

    ``feature_mode`` is an *assertion about the artifact*, not a switch that
    changes how it is read: what the model consumes is decided by the vectorizer
    stored inside the artifact, and nothing else can make a TF-IDF model read
    feature vectors. Setting it ``true`` against a text-mode artifact is therefore
    a configuration error and degrades with an actionable reason, rather than
    silently serving something other than what the operator configured. Leaving it
    at the default ``false`` asserts nothing, so a self-describing features-mode
    artifact still serves -- which matters because
    :func:`~jev_route.backends.build_backend` supplies ``false`` to everyone who
    does not set the key.

    Not thread-safe, by the same decision as
    :class:`~jev_route.backends.base.CircuitBreaker`: one asyncio event loop owns
    an instance. Two coroutines racing the first load would each read the same
    file and keep one result, which is wasteful but not wrong.
    """

    #: Stable identifier recorded on every decision. Config-facing.
    name = "distilled"

    def __init__(self, artifact: str | Path, *, feature_mode: bool = False) -> None:
        self.artifact_path = str(artifact)
        self.feature_mode = bool(feature_mode)
        self._artifact: Any = None
        self._load_error: str | None = None
        self._attempted = False

    # -- provenance ------------------------------------------------------- #
    @property
    def model_version(self) -> str:
        """The artifact's own version, or a placeholder while it is unreadable.

        A property rather than a snapshot in ``__init__`` for the same reason
        :class:`~jev_route.backends.shadow.ShadowBackend` uses one: the value is
        only known once the file has been read, and every decision record stamped
        through this backend has to carry the truth as of the read.
        """
        artifact = self._artifact
        return str(artifact.model_version) if artifact is not None else UNLOADED_MODEL_VERSION

    @property
    def is_loaded(self) -> bool:
        return self._artifact is not None

    @property
    def load_error(self) -> str | None:
        """Why the artifact did not load, or ``None``. The degrade reason verbatim."""
        return self._load_error

    @property
    def mode(self) -> str:
        """``"text"`` or ``"features"`` once loaded; ``""`` before that."""
        artifact = self._artifact
        return str(artifact.mode) if artifact is not None else ""

    def info(self) -> dict[str, Any]:
        """Introspection for ``doctor``-style checks and for tests."""
        artifact = self._artifact
        return {
            "name": self.name,
            "artifact": self.artifact_path,
            "feature_mode": self.feature_mode,
            "loaded": artifact is not None,
            "load_error": self._load_error,
            "mode": self.mode,
            "model_version": self.model_version,
            "n_features": int(artifact.n_features) if artifact is not None else None,
            "contains_prompt_text": bool(artifact.contains_prompt_text) if artifact is not None else None,
            "teacher_model_versions": list(artifact.teacher_model_versions) if artifact is not None else [],
        }

    # -- loading ---------------------------------------------------------- #
    def preload(self) -> bool:
        """Load now, and retry a previously cached failure. True when it worked.

        Exists so a deployment can fail visibly at startup -- or so
        ``jev-route doctor`` can check the artifact -- instead of discovering on
        the first request that the file it was pointed at does not exist.
        """
        self._attempted = False
        return self._ensure_loaded() is not None

    def _ensure_loaded(self) -> Any:
        """Load once, and remember why it failed if it did.

        Every failure is recorded as a string instead of propagating. That
        includes the ones that are not :class:`ArtifactError`: a missing numpy
        (:class:`~jev_route.distill.artifact.DistillDependencyError`), a zip that
        is not a zip, a permission error, or a weights file this numpy refuses to
        read. The promise is "a routing decision never fails because a model file
        is bad", and a narrow ``except`` would keep that promise only for the
        failure modes somebody thought of.
        """
        if self._attempted:
            return self._artifact
        self._attempted = True
        from ..distill.artifact import load_artifact

        try:
            self._artifact = load_artifact(self.artifact_path)
            self._load_error = None
        except Exception as exc:  # fail closed: see the docstring, this is the contract
            self._artifact = None
            self._load_error = f"{type(exc).__name__}: {exc}"
        return self._artifact

    # -- the DecisionBackend interface ------------------------------------ #
    async def decide(self, request: DecisionRequest) -> BackendResult:
        """Answer from the artifact. ``async`` because the Protocol is; there is no I/O."""
        return self.decide_sync(request)

    def decide_sync(self, request: DecisionRequest) -> BackendResult:
        """The synchronous core. Public so tests and the CLI can call it directly."""
        started = time.perf_counter()
        artifact = self._ensure_loaded()
        if artifact is None:
            return self._degraded(started, self._load_error or f"artifact {self.artifact_path!r} did not load")
        mismatch = self._mode_assertion(artifact)
        if mismatch is not None:
            return self._degraded(started, mismatch)
        try:
            answers = self._answers(artifact, request)
        except _CannotDecideError as exc:
            return self._degraded(started, str(exc))
        except Exception as exc:  # a malformed feature vector must degrade, not raise
            return self._degraded(started, f"{type(exc).__name__}: {exc}")
        return BackendResult(
            answers=answers,
            model_version=str(artifact.model_version),
            # Empty by definition: a local model asks no questions and sends
            # nothing anywhere, so there is no payload to reproduce from the log.
            questions_sent={},
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    async def aclose(self) -> None:
        """Nothing to release, and safe to call repeatedly.

        The artifact holds numpy arrays in process memory -- no sockets, no file
        handles (the loader reads and closes), no background tasks. Dropping the
        reference here would only make a reuse-after-close router fail, so the
        weights live exactly as long as the backend does.
        """

    # -- internals -------------------------------------------------------- #
    def _mode_assertion(self, artifact: Any) -> str | None:
        """``feature_mode`` contradicts the artifact, or ``None`` when it does not."""
        if not self.feature_mode or artifact.mode == "features":
            return None
        return (
            f"the policy asserts feature_mode: true but {self.artifact_path} was trained in "
            f"{artifact.mode} mode: it reads a TF-IDF vector over the redacted excerpt and cannot be "
            "served from request features alone. Set feature_mode: false, or retrain in features mode "
            "(`jev-route export --mode features`, then `jev-route train`)."
        )

    def _answers(self, artifact: Any, request: DecisionRequest) -> DecisionAnswers:
        """Feed the artifact the input its own vectorizer was fitted on."""
        if artifact.mode == "text":
            text = str(request.redacted_excerpt or "")
            if not text.strip():
                # Under gate.on_force_local: still_classify the router does call a
                # backend for a request the gate blocked, and it never sends text
                # for one. Guessing from an empty vector would be a confident
                # answer about nothing, so this degrades and the policy fails closed.
                raise _CannotDecideError(
                    "this artifact was trained in text mode and the request carries no excerpt to read "
                    "(gate-blocked requests are never stored as text). Serve a features-mode artifact "
                    "for this traffic, or set gate.on_force_local: skip_backend."
                )
            return artifact.answers_for(text=text)
        if request.features is None:
            raise _CannotDecideError(
                f"this artifact was trained in {artifact.mode} mode and needs the request's "
                "RequestFeatures, which this request does not carry."
            )
        return artifact.answers_for(features=request.features)

    def _degraded(self, started: float, reason: str) -> BackendResult:
        """Maximum-uncertainty answers plus the reason, so the router fails closed."""
        return BackendResult(
            answers=DecisionAnswers.unknown(),
            model_version=self.model_version,
            questions_sent={},
            latency_ms=(time.perf_counter() - started) * 1000.0,
            degraded=True,
            degrade_reason=reason,
        )


__all__ = ["UNLOADED_MODEL_VERSION", "DistilledBackend"]
