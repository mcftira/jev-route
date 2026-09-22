"""LayaScorer: the neural variant of the semantic layer's scorer.

The semantic layer (``jev_route.gate_semantic``) scores every request with a
:class:`~jev_route.gate_semantic.SemanticScorer`. The built-in one is a sparse
logistic regression over a stored vocabulary: it reads with ``json.load``, it
runs in pure Python, and it catches the obvious-in-context cases. It is also a
ceiling -- a vocabulary model cannot read an HR disciplinary narrative that
happens to use none of the tokens it was trained on.

This module is the heavier scorer. v0.3 fine-tuned a Laya *head* (the decision
layers of the open-source Laya System One model on a frozen multilingual
encoder) on exactly the layer-2 training set -- synthetic positives and
cleared production negatives -- and saved the result as a Laya checkpoint
directory (``model.safetensors``, ``rl_agent_config.json``, ``tokenizer/``,
``encoder/``). :class:`LayaScorer` loads that directory with ``laya.load`` and
asks the head one question, phrased as a ``noul``:

    "Does the text contain data that is sensitive or confidential in itself?"

The answer's ``noul`` field is P(sensitive) -- the layer's score.

The gate's invariants for a scorer hold here, in the same order:

**Local, and offline.** The checkpoint is a local directory, so
``laya.load`` resolves nothing against the hub. Loading one forces
``HF_HUB_OFFLINE=1`` for the load window: a layer-2 scorer that reached the
network would turn the guard into the leak, and that failure belongs here, in
the scorer, not in egress.

**Lazy.** The weights are not read until the first score. Constructing the
scorer -- including from an artifact on disk -- must stay cheap, because the
policy loads the artifact at process start; a large checkpoint that loads at
import time is a startup cost the config file should not impose.

**Deterministic.** ``system_one`` is one forward pass through a model in eval
mode with a temperature-scaled softmax. No sampling, no randomness, no
network: the same text always gets the same probability.

**Total, and fail-safe.** :meth:`LayaScorer.score` never raises. A missing
checkpoint, a torch crash, or an unparseable model body all return ``0.0`` and
mark the scorer degraded (:attr:`LayaScorer.degraded`,
:attr:`LayaScorer.last_error`, the counters). A broken layer 2 is telemetry
loss, not a request failure: the request continues on layer 1's verdict alone,
which is still a floor.

``laya`` -- and with it torch/transformers -- is imported inside the load
method, exactly like :mod:`.laya` does. The scorer module itself stays
stdlib-only, so a deployment that never uses the Laya scorer pays nothing for
it.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from ..schema import RequestFeatures

#: The gate's semantic question, in Laya's noul format. The head is fine-tuned
#: on exactly this instruction (``laya_route.gate_head.GATE_QUESTION`` in the
#: laya-route training repo); rewording it here changes what the checkpoint
#: answers, so the text is copied verbatim and pinned by tests.
GATE_QUESTION: dict[str, Any] = {
    "type": "noul",
    "instructions": (
        "Does the text contain data that is sensitive or confidential in itself -- "
        "personal data about an identifiable person, credentials, financial or health "
        "records, trade secrets, or regulated identifiers -- as opposed to merely "
        "discussing a sensitive topic in the abstract?"
    ),
    "criteria": {
        "true": "Sensitive or confidential data is present in the text.",
        "false": "No sensitive data is present in the text.",
    },
}

#: The question id under which the scorer asks its one question.
QUESTION_ID = "sensitivity"


def _noul_probability(raw: Any, question_id: str) -> float | None:
    """Extract a noul probability from a ``system_one`` body. Total function.

    Mirrors the coercion rules of ``backends.jev._normalize_noul`` so that a
    Laya answer and a Jev answer degrade the same way: a JSON ``true``/``false``
    reads as 1.0/0.0, a NaN degrades, and anything else that is not a number --
    including the string ``"0.9"`` -- degrades to ``None`` rather than being
    parsed. A malformed body must cost a degraded score, not a request.
    """
    if not isinstance(raw, Mapping):
        return None
    answers = raw.get("answers")
    answer = answers.get(question_id) if isinstance(answers, Mapping) else None
    if not isinstance(answer, Mapping):
        return None
    value = answer.get("noul")
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if not isinstance(value, (int, float)) or math.isnan(float(value)):
        return None
    return float(value)


class LayaScorer:
    """Layer-2 scorer served by a local Laya checkpoint.

    Implements the :class:`~jev_route.gate_semantic.SemanticScorer` protocol:
    :attr:`name`, :attr:`model_version`, and
    :meth:`score <LayaScorer.score>(text, *, features=None) -> float`.

    Args:
        checkpoint_dir: the trained Laya checkpoint directory (``model.safetensors``,
            ``rl_agent_config.json``, ``tokenizer/``, ``encoder/``). Must be a local
            directory: layer 2 is local by contract, and a hub id is refused at load
            time rather than resolved.
        model_version: version stamped on every assessment. Defaults to a hash of the
            checkpoint path.
        device: ``auto`` / ``cpu`` / ``mps`` / ``cuda`` (passed to ``laya.load``).
        question: the noul question to ask. Defaults to :data:`GATE_QUESTION` -- the
            instruction the head was fine-tuned on.
        agent_factory: injectable loader, ``factory(checkpoint_dir) -> agent``, so
            tests run without torch/transformers/laya installed. When set, the
            local-directory and offline checks are the factory's responsibility.
    """

    #: Stable identifier for logs and the artifact envelope.
    name = "laya-sensitivity-head"
    #: The artifact payload discriminator (``scorer.kind``).
    kind = "laya"

    def __init__(
        self,
        checkpoint_dir: str | os.PathLike[str],
        *,
        model_version: str | None = None,
        device: str = "auto",
        question: Mapping[str, Any] | None = None,
        agent_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.checkpoint_dir = os.fspath(checkpoint_dir)
        if not self.checkpoint_dir.strip():
            raise ValueError(f"{self.name}: checkpoint_dir must be a non-empty path")
        self.device = str(device).lower()
        self.question: dict[str, Any] = (
            copy.deepcopy(dict(question)) if question is not None else copy.deepcopy(GATE_QUESTION)
        )
        if str(self.question.get("type", "")) != "noul":
            raise ValueError(
                f"{self.name}: the question must be a noul question (P(true) is the score), "
                f"got type={self.question.get('type')!r}"
            )
        if not isinstance(self.question.get("instructions"), str) or not str(self.question["instructions"]).strip():
            raise ValueError(f"{self.name}: the question needs non-empty string instructions")
        self.model_version = (
            model_version or f"{self.name}-{hashlib.sha256(self.checkpoint_dir.encode('utf-8')).hexdigest()[:12]}"
        )
        self._agent_factory = agent_factory
        self._agent: Any = None
        self._load_lock = threading.Lock()
        self._load_attempted = False
        self.load_latency_ms: float | None = None
        # Degradation state: the honest record that this scorer is (or is not)
        # answering. ``degraded`` describes the most recent score -- a success
        # clears it -- so a host polling the scorer sees the current condition,
        # not a ratchet.
        self.degraded = False
        self.last_error: str | None = None
        self.score_calls = 0
        self.error_count = 0

    # -- loading (lazy) ---------------------------------------------------- #
    def _create_agent(self) -> Any:
        if self._agent_factory is not None:
            return self._agent_factory(self.checkpoint_dir)
        if not os.path.isdir(self.checkpoint_dir):
            raise FileNotFoundError(
                f"{self.name}: checkpoint_dir {self.checkpoint_dir!r} is not a local directory. "
                "Layer 2 is local by contract: point the artifact at the trained "
                "checkpoint directory, not at a hub id."
            )
        from laya import load as laya_load  # lazy: torch/transformers/laya are heavy

        device = None if self.device == "auto" else self.device
        # A local checkpoint: force offline for the load window so nothing in the
        # tokenizer/model path can resolve against the hub. The env value is
        # restored afterwards, whatever the load does.
        previous = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            return laya_load(self.checkpoint_dir, device=device)
        finally:
            if previous is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = previous

    def _load_agent(self) -> Any:
        """Load once (under a lock, on first use) and return the agent.

        Returns ``None`` when loading has failed; the failure is already recorded
        as degraded. A failed load is not retried per request: retrying a
        checkpoint load inside the request path would make a broken checkpoint
        an outage. Use :meth:`reload` to force a fresh attempt.
        """
        with self._load_lock:
            if self._agent is not None:
                return self._agent
            if self._load_attempted:
                return None
            self._load_attempted = True
            started = time.perf_counter()
            try:
                self._agent = self._create_agent()
                self.load_latency_ms = (time.perf_counter() - started) * 1000.0
                self.degraded = False
                self.last_error = None
                return self._agent
            except Exception as exc:  # a broken checkpoint is telemetry, not an outage
                self._mark_degraded(f"{type(exc).__name__}: {exc}")
                self._agent = None
                return None

    def preload(self) -> None:
        """Load the checkpoint now so the first score is not a cold one.

        Never raises: a failure is recorded as degraded, exactly like a failure
        inside :meth:`score`.
        """
        self._load_agent()

    def reload(self) -> None:
        """Force a fresh load attempt on the next score (operator action, not a request path)."""
        with self._load_lock:
            self._agent = None
            self._load_attempted = False

    # -- scoring ----------------------------------------------------------- #
    def score(self, text: str, *, features: RequestFeatures | None = None) -> float:
        """Probability that ``text`` contains data sensitive or confidential in itself.

        ``features`` is accepted and ignored. The signature carries it because a
        scorer that disagrees with its Protocol's arguments is not an
        implementation of it.

        Never raises: on any load or inference error this returns ``0.0`` and
        records the degradation on the scorer (:attr:`degraded`,
        :attr:`last_error`, :attr:`error_count`). A broken layer 2 costs
        telemetry, not the request.
        """
        del features
        self.score_calls += 1
        try:
            agent = self._load_agent()
            if agent is None:
                return 0.0  # the load failure is already recorded as degraded
            raw = agent.system_one(str(text), {QUESTION_ID: self.question})
            value = _noul_probability(raw, QUESTION_ID)
            if value is None:
                self._mark_degraded(f"malformed system_one body: no usable noul probability for {QUESTION_ID!r}")
                return 0.0
            self.degraded = False
            self.last_error = None
            return min(1.0, max(0.0, value))
        except Exception as exc:  # a local crash is telemetry, not a request failure
            self._mark_degraded(f"{type(exc).__name__}: {exc}")
            return 0.0

    def _mark_degraded(self, reason: str) -> None:
        self.degraded = True
        self.last_error = reason[:200]
        self.error_count += 1

    # -- observability ----------------------------------------------------- #
    def stats(self) -> dict[str, Any]:
        """What a host can poll without scoring: is the head up, and what broke?"""
        return {
            "name": self.name,
            "kind": self.kind,
            "model_version": self.model_version,
            "checkpoint_dir": self.checkpoint_dir,
            "device": self.device,
            "loaded": self._agent is not None,
            "load_latency_ms": self.load_latency_ms,
            "score_calls": self.score_calls,
            "error_count": self.error_count,
            "degraded": self.degraded,
            "last_error": self.last_error,
        }


__all__ = [
    "GATE_QUESTION",
    "QUESTION_ID",
    "LayaScorer",
]
