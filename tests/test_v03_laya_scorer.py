"""Tests for the v0.3 Laya sensitivity head as the layer-2 scorer.

v0.3 fine-tuned a Laya head (frozen multilingual encoder, trained decision
head) on the layer-2 dataset and saved it as a Laya checkpoint directory
(``model.safetensors``, ``rl_agent_config.json``, ``tokenizer/``,
``encoder/``). :class:`LayaScorer` wraps that checkpoint in the
:class:`~jev_route.gate_semantic.SemanticScorer` protocol; this file verifies
the promises around the model, with a fake agent in its place -- no GPU, no
real weights, no network (``conftest.py`` blocks outbound sockets):

1. **Protocol.** ``LayaScorer`` is a ``SemanticScorer`` and construction is
   lazy: an artifact on disk can be loaded without touching the weights.
2. **Plumbing.** The text reaches ``system_one`` as the state, the exact
   noul gate question goes with it, and the answer's ``noul`` field comes
   back as the score.
3. **Total.** A load failure, an inference error, or a malformed model body
   all return ``0.0`` and record degradation on the scorer -- never a raise.
4. **Offline.** Loading a local checkpoint directory forces
   ``HF_HUB_OFFLINE=1`` for the load window (checked with a fake ``laya``
   module injected through ``sys.modules``).
5. **Artifact.** The scorer payload's ``kind: "laya"`` variant round-trips
   through ``SemanticArtifact.save``/``load``; the lexicon path is unchanged.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from jev_route.backends.laya_scorer import GATE_QUESTION, QUESTION_ID, LayaScorer
from jev_route.gate_semantic import (
    SEMANTIC_ARTIFACT_KIND,
    LexiconScorer,
    SemanticArtifact,
    SemanticArtifactError,
    SemanticGatePolicy,
    SemanticLayer,
    SemanticScorer,
)
from jev_route.schema import RequestFeatures

CHECKPOINT = "/tmp/v03-laya-head2"


# --------------------------------------------------------------------------- #
# Doubles
# --------------------------------------------------------------------------- #
def canned_body(noul: Any) -> dict[str, Any]:
    """A well-formed ``system_one`` body with one noul answer."""
    return {
        "model": "laya-rl-agent",
        "answers": {QUESTION_ID: {"type": "noul", "noul": noul, "confidence": 0.9}},
        "usage": {"input_tokens": 10, "output_tokens": 0},
    }


class FakeLayaAgent:
    """A ``laya.Agent`` stand-in: records every call, answers from a canned body."""

    def __init__(self, body: Any, calls: list[dict[str, Any]]) -> None:
        self.body = body
        self.calls = calls
        self.fail: Exception | None = None

    def system_one(self, state: Any, questions: Any) -> Any:
        self.calls.append({"state": state, "questions": questions})
        if self.fail is not None:
            raise self.fail
        return self.body


def make_scorer(agent: FakeLayaAgent, *, loads: list[str] | None = None, **kwargs: Any) -> LayaScorer:
    """A scorer whose load goes through a recorded factory (no torch, no weights)."""

    def factory(checkpoint_dir: str) -> FakeLayaAgent:
        if loads is not None:
            loads.append(checkpoint_dir)
        return agent

    return LayaScorer(CHECKPOINT, agent_factory=factory, **kwargs)


def make_broken_factory(loads: list[str]) -> Any:
    def factory(checkpoint_dir: str) -> Any:
        loads.append(checkpoint_dir)
        raise FileNotFoundError(f"checkpoint {checkpoint_dir!r} is missing model.safetensors")

    return factory


def laya_payload(**overrides: Any) -> dict[str, Any]:
    """A loadable artifact envelope carrying the ``kind: "laya"`` scorer."""
    payload: dict[str, Any] = {
        "kind": SEMANTIC_ARTIFACT_KIND,
        "artifact_version": "1",
        "model_version": "laya-gate-head-v03",
        "created_at": "2026-01-01T00:00:00+00:00",
        "threshold": 0.5,
        "level": "confidential",
        "scorer": {
            "kind": "laya",
            "name": LayaScorer.name,
            "checkpoint_dir": CHECKPOINT,
            "device": "cpu",
            "question": GATE_QUESTION,
        },
        "metrics": {
            "recall": 0.99,
            "false_positive_rate": 0.01,
            "threshold": 0.5,
            "model_version": "laya-gate-head-v03",
        },
        "provenance": {
            "positives": "synthetic",
            "negatives": "production",
            "trained_on_blocked_content": False,
            "note": "v0.3 laya head",
        },
    }
    payload.update(overrides)
    return payload


def write_artifact(tmp_path: Path, payload: dict[str, Any], *, name: str = "semantic.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# 1. Protocol and lazy construction
# --------------------------------------------------------------------------- #
class TestProtocolAndConstruction:
    def test_it_implements_the_semantic_scorer_protocol(self) -> None:
        scorer = LayaScorer(CHECKPOINT, agent_factory=lambda cp: FakeLayaAgent(canned_body(0.5), []))
        assert isinstance(scorer, SemanticScorer)
        assert scorer.name == LayaScorer.name
        assert scorer.kind == "laya"

    def test_construction_is_lazy(self) -> None:
        calls: list[dict[str, Any]] = []
        loads: list[str] = []
        scorer = make_scorer(FakeLayaAgent(canned_body(0.5), calls), loads=loads)
        assert loads == [] and calls == []  # nothing happened at construction
        scorer.score("a prompt")
        assert loads == [CHECKPOINT]  # exactly one load
        scorer.score("another prompt")
        assert loads == [CHECKPOINT]  # ...on the first score only

    def test_an_explicit_model_version_is_recorded(self) -> None:
        scorer = LayaScorer(
            CHECKPOINT, model_version="v03-head2", agent_factory=lambda cp: FakeLayaAgent(canned_body(0.5), [])
        )
        assert scorer.model_version == "v03-head2"

    def test_the_default_model_version_is_deterministic_per_checkpoint(self) -> None:
        def factory(cp: str) -> FakeLayaAgent:
            return FakeLayaAgent(canned_body(0.5), [])

        a = LayaScorer(CHECKPOINT, agent_factory=factory)
        b = LayaScorer(CHECKPOINT, agent_factory=factory)
        other = LayaScorer("/elsewhere/ckpt", agent_factory=factory)
        assert a.model_version == b.model_version
        assert a.model_version != other.model_version

    def test_a_non_noul_question_is_refused(self) -> None:
        with pytest.raises(ValueError, match="noul"):
            LayaScorer(CHECKPOINT, question={"type": "choice", "instructions": "pick one", "criteria": ["a"]})

    def test_an_empty_checkpoint_dir_is_refused(self) -> None:
        with pytest.raises(ValueError, match="checkpoint_dir"):
            LayaScorer("   ")


# --------------------------------------------------------------------------- #
# 2. Plumbing: text in, noul out
# --------------------------------------------------------------------------- #
class TestScoringPlumbing:
    def test_the_score_is_the_noul_probability(self) -> None:
        scorer = make_scorer(FakeLayaAgent(canned_body(0.83), []))
        assert scorer.score("The patient reports chest pain.") == pytest.approx(0.83)

    def test_the_text_reaches_the_model_as_state(self) -> None:
        calls: list[dict[str, Any]] = []
        scorer = make_scorer(FakeLayaAgent(canned_body(0.2), calls))
        text = "Please draft the settlement for the disciplinary hearing."
        scorer.score(text)
        assert calls[0]["state"] == text

    def test_the_model_is_asked_the_gate_question(self) -> None:
        calls: list[dict[str, Any]] = []
        scorer = make_scorer(FakeLayaAgent(canned_body(0.2), calls))
        scorer.score("anything")
        questions = calls[0]["questions"]
        assert set(questions) == {QUESTION_ID}
        assert questions[QUESTION_ID] == GATE_QUESTION
        assert questions[QUESTION_ID]["type"] == "noul"
        assert "sensitive or confidential in itself" in questions[QUESTION_ID]["instructions"]

    def test_scoring_is_deterministic(self) -> None:
        scorer = make_scorer(FakeLayaAgent(canned_body(0.62), []))
        scores = [scorer.score("the same text") for _ in range(3)]
        assert scores == [0.62, 0.62, 0.62]

    def test_features_are_accepted_and_ignored(self) -> None:
        scorer = make_scorer(FakeLayaAgent(canned_body(0.4), []))
        assert scorer.score("x", features=RequestFeatures()) == pytest.approx(0.4)

    @pytest.mark.parametrize("noul,expected", [(1.7, 1.0), (-0.2, 0.0), (0, 0.0), (1, 1.0)])
    def test_out_of_range_noul_is_clamped(self, noul: Any, expected: float) -> None:
        scorer = make_scorer(FakeLayaAgent(canned_body(noul), []))
        assert scorer.score("x") == expected

    def test_a_non_numeric_noul_degrades_rather_than_parses(self) -> None:
        # Same total-function rule as the Jev backend: "0.9" is a schema
        # violation, and the fail-safe direction is a degraded zero.
        scorer = make_scorer(FakeLayaAgent(canned_body("0.9"), []))
        assert scorer.score("x") == 0.0
        assert scorer.degraded is True
        assert "noul" in scorer.last_error


# --------------------------------------------------------------------------- #
# 3. Total: broken means 0.0 + recorded, never a raise
# --------------------------------------------------------------------------- #
class TestTotalAndFailSafe:
    def test_a_load_failure_returns_zero_and_records_degraded(self) -> None:
        loads: list[str] = []
        scorer = LayaScorer(CHECKPOINT, agent_factory=make_broken_factory(loads))
        assert scorer.score("x") == 0.0
        assert scorer.degraded is True
        assert scorer.error_count == 1
        assert "FileNotFoundError" in (scorer.last_error or "")

    def test_a_failed_load_is_not_retried_on_every_request(self) -> None:
        loads: list[str] = []
        scorer = LayaScorer(CHECKPOINT, agent_factory=make_broken_factory(loads))
        assert [scorer.score("x") for _ in range(3)] == [0.0, 0.0, 0.0]
        assert loads == [CHECKPOINT]  # one attempt, not one per request

    def test_a_failed_load_can_be_retried_explicitly(self) -> None:
        loads: list[str] = []
        factory = make_broken_factory(loads)
        scorer = LayaScorer(CHECKPOINT, agent_factory=factory)
        assert scorer.score("x") == 0.0
        scorer.reload()
        assert scorer.score("x") == 0.0
        assert loads == [CHECKPOINT, CHECKPOINT]

    def test_an_inference_error_returns_zero_and_records_degraded(self) -> None:
        agent = FakeLayaAgent(canned_body(0.9), [])
        agent.fail = RuntimeError("torch out of memory")
        scorer = make_scorer(agent)
        assert scorer.score("x") == 0.0
        assert scorer.degraded is True
        assert "RuntimeError" in (scorer.last_error or "")

    @pytest.mark.parametrize(
        "body",
        [
            {"model": "laya-rl-agent"},  # no answers at all
            {"answers": {"other": {"type": "noul", "noul": 0.9}}},  # wrong question id
            {"answers": {QUESTION_ID: {"type": "noul", "confidence": 0.9}}},  # no noul field
            {"answers": {QUESTION_ID: {"type": "noul", "noul": None}}},  # null noul
            42,  # garbage
            "nope",  # garbage
        ],
        ids=["no-answers", "wrong-id", "no-noul", "null-noul", "int", "str"],
    )
    def test_a_malformed_body_returns_zero_and_records_degraded(self, body: Any) -> None:
        scorer = make_scorer(FakeLayaAgent(body, []))
        assert scorer.score("x") == 0.0
        assert scorer.degraded is True
        assert "malformed" in (scorer.last_error or "")

    def test_a_recovered_scorer_clears_the_degradation(self) -> None:
        agent = FakeLayaAgent(canned_body(0.83), [])
        agent.fail = RuntimeError("transient")
        scorer = make_scorer(agent)
        assert scorer.score("x") == 0.0
        assert scorer.degraded is True
        agent.fail = None
        assert scorer.score("x") == pytest.approx(0.83)
        assert scorer.degraded is False
        assert scorer.last_error is None

    def test_preload_records_degradation_without_raising(self) -> None:
        loads: list[str] = []
        scorer = LayaScorer(CHECKPOINT, agent_factory=make_broken_factory(loads))
        scorer.preload()  # must not raise
        stats = scorer.stats()
        assert stats["loaded"] is False
        assert stats["degraded"] is True
        assert stats["checkpoint_dir"] == CHECKPOINT


# --------------------------------------------------------------------------- #
# 4. Offline: a local checkpoint never touches the hub
# --------------------------------------------------------------------------- #
class _FakeLayaModule:
    """Stands in for the ``laya`` package in ``sys.modules``."""

    def __init__(self, load_calls: list[dict[str, Any]], agent: FakeLayaAgent) -> None:
        self.load_calls = load_calls
        self.agent = agent
        self.module = types.SimpleNamespace(load=self.load)

    def load(self, path: str, device: Any = None, subfolder: Any = None) -> FakeLayaAgent:
        self.load_calls.append({"path": path, "device": device, "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE")})
        return self.agent


@pytest.fixture
def fake_laya(tmp_path: Path) -> tuple[_FakeLayaModule, list[dict[str, Any]], list[dict[str, Any]], Path]:
    """A fake ``laya`` module plus a genuine-looking local checkpoint directory."""
    load_calls: list[dict[str, Any]] = []
    agent_calls: list[dict[str, Any]] = []
    agent = FakeLayaAgent(canned_body(0.42), agent_calls)
    fake = _FakeLayaModule(load_calls, agent)
    ckpt = tmp_path / "v03-laya-head2"
    ckpt.mkdir()
    (ckpt / "rl_agent_config.json").write_text('{"encoder": "mmBERT-base", "head_layers": 2}', encoding="utf-8")
    return fake, load_calls, agent_calls, ckpt


class TestOfflineLoad:
    def test_a_local_checkpoint_is_loaded_with_hf_hub_offline(self, fake_laya, monkeypatch: pytest.MonkeyPatch) -> None:
        fake, load_calls, agent_calls, ckpt = fake_laya
        monkeypatch.setitem(sys.modules, "laya", fake.module)
        scorer = LayaScorer(ckpt)
        assert scorer.score("The candidate's salary review is pending.") == pytest.approx(0.42)
        assert len(load_calls) == 1
        assert load_calls[0]["path"] == str(ckpt)
        assert load_calls[0]["hf_hub_offline"] == "1"  # forced for the load window
        assert os.environ.get("HF_HUB_OFFLINE") is None  # ...and restored afterwards
        assert len(agent_calls) == 1

    def test_a_preexisting_offline_flag_is_restored(self, fake_laya, monkeypatch: pytest.MonkeyPatch) -> None:
        fake, load_calls, _agent_calls, ckpt = fake_laya
        monkeypatch.setitem(sys.modules, "laya", fake.module)
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        scorer = LayaScorer(ckpt)
        assert scorer.score("x") == pytest.approx(0.42)
        assert load_calls[0]["hf_hub_offline"] == "1"
        assert os.environ.get("HF_HUB_OFFLINE") == "1"  # the caller's value, not our default

    def test_a_non_local_checkpoint_is_refused_without_reaching_laya(
        self, fake_laya, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake, load_calls, _agent_calls, _ckpt = fake_laya
        monkeypatch.setitem(sys.modules, "laya", fake.module)
        scorer = LayaScorer("/nonexistent/checkpoint")
        assert scorer.score("x") == 0.0
        assert scorer.degraded is True
        assert "local directory" in (scorer.last_error or "")
        assert load_calls == []  # refused before the model was ever reached


# --------------------------------------------------------------------------- #
# 5. Artifact: the kind "laya" payload
# --------------------------------------------------------------------------- #
class TestArtifactRoundTrip:
    def test_a_laya_artifact_round_trips_through_a_directory(self, tmp_path: Path) -> None:
        artifact = SemanticArtifact.from_dict(laya_payload())
        assert isinstance(artifact.scorer, LayaScorer)
        written = artifact.save(tmp_path / "sem")
        assert written.name == "semantic.json"

        on_disk = json.loads(written.read_text(encoding="utf-8"))
        assert on_disk["scorer"]["kind"] == "laya"
        assert on_disk["scorer"]["checkpoint_dir"] == CHECKPOINT
        assert on_disk["scorer"]["question"] == GATE_QUESTION
        assert "weights" not in on_disk["scorer"]  # a checkpoint is a pointer, not a table

        loaded = SemanticArtifact.load(tmp_path / "sem")
        assert isinstance(loaded.scorer, LayaScorer)
        assert loaded.scorer.checkpoint_dir == CHECKPOINT
        assert loaded.scorer.device == "cpu"
        assert loaded.scorer.question == GATE_QUESTION
        assert loaded.model_version == "laya-gate-head-v03"
        assert loaded.metrics == artifact.metrics
        assert loaded.provenance == artifact.provenance

    def test_a_laya_artifact_round_trips_through_a_bare_json_file(self, tmp_path: Path) -> None:
        path = write_artifact(tmp_path, laya_payload(), name="gate-semantic.json")
        loaded = SemanticArtifact.load(path)
        assert isinstance(loaded.scorer, LayaScorer)
        assert loaded.scorer.checkpoint_dir == CHECKPOINT

    def test_loading_an_artifact_does_not_load_the_checkpoint(self, tmp_path: Path) -> None:
        # The scorer must stay lazy all the way down: an artifact pointing at a
        # checkpoint that does not exist yet must still load.
        payload = laya_payload(scorer={"kind": "laya", "checkpoint_dir": str(tmp_path / "absent")})
        artifact = SemanticArtifact.from_dict(payload)
        assert isinstance(artifact.scorer, LayaScorer)
        assert artifact.scorer.stats()["loaded"] is False

    def test_omitting_the_question_defaults_to_the_gate_question(self, tmp_path: Path) -> None:
        scorer_cfg = {"kind": "laya", "checkpoint_dir": CHECKPOINT}
        artifact = SemanticArtifact.from_dict(laya_payload(scorer=scorer_cfg))
        assert artifact.scorer.question == GATE_QUESTION

    def test_the_checkpoint_dir_is_required_for_the_laya_kind(self, tmp_path: Path) -> None:
        payload = laya_payload(scorer={"kind": "laya", "name": LayaScorer.name})
        path = write_artifact(tmp_path, payload, name="no-ckpt.json")
        with pytest.raises(SemanticArtifactError, match="checkpoint_dir"):
            SemanticArtifact.load(path)

    def test_an_unknown_scorer_kind_is_refused(self, tmp_path: Path) -> None:
        payload = laya_payload(scorer={"kind": "transformer", "checkpoint_dir": CHECKPOINT})
        path = write_artifact(tmp_path, payload, name="kind.json")
        with pytest.raises(SemanticArtifactError, match=r"scorer\.kind"):
            SemanticArtifact.load(path)

    def test_a_laya_question_must_be_an_object(self, tmp_path: Path) -> None:
        payload = laya_payload(scorer={"kind": "laya", "checkpoint_dir": CHECKPOINT, "question": "noul"})
        path = write_artifact(tmp_path, payload, name="badq.json")
        with pytest.raises(SemanticArtifactError, match=r"scorer\.question"):
            SemanticArtifact.load(path)

    def test_the_lexicon_path_is_unchanged_without_a_kind(self) -> None:
        # A payload with no ``scorer.kind`` at all is still the lexicon one,
        # byte for byte: existing artifacts keep loading exactly as before.
        payload = {
            "kind": SEMANTIC_ARTIFACT_KIND,
            "artifact_version": "1",
            "model_version": "test-semantic-1",
            "threshold": 0.5,
            "level": "confidential",
            "scorer": {
                "name": "lexicon-logreg",
                "bias": -2.0,
                "tokenizer": {"lower": True, "min_token_chars": 2, "ngram_max": 2},
                "weights": {"disciplinary": 4.0},
            },
            "metrics": {"recall": 0.99},
            "provenance": {
                "positives": "synthetic",
                "negatives": "production",
                "trained_on_blocked_content": False,
            },
        }
        artifact = SemanticArtifact.from_dict(payload)
        assert isinstance(artifact.scorer, LexiconScorer)
        assert artifact.scorer.weights == {"disciplinary": 4.0}


# --------------------------------------------------------------------------- #
# 6. The layer: artifact -> scorer -> assessment
# --------------------------------------------------------------------------- #
class TestLayerIntegration:
    def test_the_layer_builds_from_a_laya_artifact(self, tmp_path: Path) -> None:
        path = write_artifact(tmp_path, laya_payload())
        config = SemanticGatePolicy(mode="shadow", artifact=str(path))
        layer = SemanticLayer(config)
        assert layer.active is True
        assert isinstance(layer.scorer, LayaScorer)
        assert layer.scorer.checkpoint_dir == CHECKPOINT
        assert layer.model_version == "laya-gate-head-v03"

    def test_the_layer_scores_through_an_injected_scorer(self) -> None:
        config = SemanticGatePolicy(mode="shadow")
        scorer = make_scorer(FakeLayaAgent(canned_body(0.83), []))
        layer = SemanticLayer(config, scorer=scorer)
        assessment = layer.assess("My SSN is 123-45-6789 and my GP is Dr. Jones.")
        assert assessment.ran is True
        assert assessment.degraded is False
        assert assessment.score == pytest.approx(0.83)
        assert assessment.fired is True
        assert assessment.asserted_level == "confidential"
        assert assessment.model_version == scorer.model_version

    def test_a_broken_head_is_telemetry_not_an_outage(self) -> None:
        config = SemanticGatePolicy(mode="shadow")
        loads: list[str] = []
        scorer = LayaScorer(CHECKPOINT, agent_factory=make_broken_factory(loads))
        layer = SemanticLayer(config, scorer=scorer)
        assessment = layer.assess("hello")  # must not raise
        assert assessment.score == 0.0
        assert assessment.fired is False
        assert scorer.degraded is True
        assert scorer.error_count == 1
        assert layer.stats()["active"] is True  # the layer is up; the head is not
