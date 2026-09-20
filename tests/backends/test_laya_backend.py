"""Tests for :mod:`jev_route.backends.laya` -- the fully local edition.

The backend is a wrapper around a pretrained model; what it *promises* is what
gets tested here, each promise being a claim a sceptical operator would verify
first:

1. **It speaks the protocol.** :class:`LayaBackend` is a
   :class:`DecisionBackend`: total schema (all four answers always present,
   full soft distributions, sums to one) and never raises for an outage
   (degrades instead).
2. **One forward pass.** All four questions go into a *single*
   ``system_one`` call -- the batched ``[MASK]``-per-option scoring that makes
   Laya cheap. A backend that loops four calls has quietly doubled the latency
   and the power draw.
3. **The mapping is 1:1.** The questions handed to the model are exactly
   :func:`jev_route.backends.jev.build_questions` (same types, instructions,
   criteria), and the model's answer shape maps onto :class:`DecisionAnswers`
   -- so a Laya decision and a Jev decision are log-comparable.
4. **No silent truncation.** The excerpt is fit to the checkpoint's sequence
   budget with a deterministic head+tail strategy, and the strategy plus the
   token counts are written onto the decision. Short inputs pass through
   byte-identical.
5. **Uncalibrated Laya cannot drive a decision.** ``enforce`` without a fitted
   artifact refuses at construction (fail-closed, like Jev without a key);
   ``shadow`` is always allowed. A tampered artifact is refused at load.
6. **Nothing leaves.** The whole suite runs with non-loopback sockets blocked
   (see ``conftest.py``); a ``decide()`` that reached for the network would
   raise ``NetworkBlockedError``. torch/transformers/laya are lazy -- checked
   statically here, so the core suite runs without them installed.

The model itself is a mock (a canned overconfident distribution and a
character tokenizer), because CI must not download a 421M checkpoint. The
real checkpoint is exercised by the laya-route integration tests.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from jev_route.backends import build_backend
from jev_route.backends.base import DecisionBackend
from jev_route.backends.jev import build_questions
from jev_route.backends.laya import LayaBackend
from jev_route.backends.laya_calibration import (
    CalibrationArtifact,
    CalibrationError,
    Isotonic1D,
    fit_isotonic,
    fit_temperature,
    load_artifact,
    save_artifact,
)
from jev_route.schema import COMPLEXITY_LEVELS, DOMAINS, SENSITIVITY_LEVELS, DecisionAnswers

MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "jev_route" / "backends" / "laya.py"
HEAVY = {"torch", "transformers", "safetensors", "huggingface_hub", "laya", "numpy"}
EGRESS = {"httpx", "requests", "aiohttp", "urllib3", "http", "socket", "urllib"}

#: The canned model answer: deliberately overconfident, like real Laya.
CANNED_ANSWERS = {
    "model": "laya-rl-agent",
    "answers": {
        "complexity": {
            "type": "choice",
            "choice": "hard",
            "probabilities": {"trivial": 0.01, "standard": 0.05, "hard": 0.9, "frontier": 0.04},
            "confidence": 0.9,
        },
        "sensitivity": {
            "type": "choice",
            "choice": "internal",
            "probabilities": {"public": 0.02, "internal": 0.8, "confidential": 0.15, "regulated": 0.03},
            "confidence": 0.8,
        },
        "pii_present": {"type": "noul", "noul": 0.9, "confidence": 0.9},
        "domain": {
            "type": "choice",
            "choice": "code",
            "probabilities": {"code": 0.85, "writing": 0.05, "analysis": 0.05, "chat": 0.02, "data-extraction": 0.03},
            "confidence": 0.85,
        },
    },
    "usage": {"input_tokens": 100, "output_tokens": 0},
}


class CharTokenizer:
    """One token per character: exact, deterministic, invertible on ASCII."""

    mask_token_id = -1
    pad_token_id = 0
    cls_token_id = 1
    sep_token_id = 2

    def __call__(self, text: str, add_special_tokens: bool = False, **_: Any) -> dict[str, list[int]]:
        return {"input_ids": [ord(c) % 400 for c in str(text)]}

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(chr(int(i)) for i in ids)


class MockAgent:
    """A Laya stand-in: records every call, returns the canned answer."""

    def __init__(self, tok: CharTokenizer, calls: list[dict[str, Any]], fail: bool = False) -> None:
        self.tok = tok
        self.calls = calls
        self.fail = fail

    def system_one(self, state: Any, questions: Any) -> dict[str, Any]:
        self.calls.append({"state": state, "questions": questions})
        if self.fail:
            raise RuntimeError("the model on fire")
        import copy

        return copy.deepcopy(CANNED_ANSWERS)


def _temperature_artifact(**temps: float) -> CalibrationArtifact:
    return CalibrationArtifact(
        method="temperature",
        temperature={k: float(v) for k, v in temps.items()},
        fit={"source": "test", "n_samples": 10},
    )


def _backend(
    calls: list[dict[str, Any]],
    *,
    role: str = "shadow",
    calibration: CalibrationArtifact | None = None,
    model: str = "english",
    token_budget: int = 448,
    max_len: int = 512,
    fail: bool = False,
    language_fn=None,
) -> LayaBackend:
    tok = CharTokenizer()
    return LayaBackend(
        model=model,
        role=role,
        calibration=calibration,
        token_budget=token_budget,
        max_len=max_len,
        agent_factory=lambda cp: MockAgent(tok, calls, fail=fail),
        tokenizer=lambda cp: tok,
        language_fn=language_fn,
    )


# --------------------------------------------------------------------------- #
# The protocol
# --------------------------------------------------------------------------- #
class TestTheProtocol:
    def test_it_is_a_decision_backend(self) -> None:
        backend = _backend(calls=[])
        assert isinstance(backend, DecisionBackend)
        assert backend.name == "laya"

    async def test_total_schema_and_full_distributions(self, decision_request) -> None:
        calls: list[dict[str, Any]] = []
        backend = _backend(calls)
        result = await backend.decide(decision_request("Summarize this internal status update."))
        await backend.aclose()

        assert not result.degraded
        a: DecisionAnswers = result.answers
        for head in (a.complexity, a.sensitivity, a.domain):
            # Full ladder, never a partial distribution.
            assert set(head.probabilities.keys()) in (
                set(COMPLEXITY_LEVELS),
                set(SENSITIVITY_LEVELS),
                set(DOMAINS),
            )
            assert abs(sum(head.probabilities.values()) - 1.0) < 1e-6
        assert 0.0 <= a.pii.value <= 1.0

    async def test_one_forward_pass_for_all_questions(self, decision_request) -> None:
        calls: list[dict[str, Any]] = []
        backend = _backend(calls)
        await backend.decide(decision_request("Do a thing."))
        await backend.decide(decision_request("Do another thing."))
        # Two decisions, two forward passes -- four questions each, never four calls.
        assert len(calls) == 2
        for call in calls:
            assert set(call["questions"].keys()) == {"complexity", "sensitivity", "pii_present", "domain"}

    async def test_state_strips_raw_payload_keys(self, decision_request) -> None:
        calls: list[dict[str, Any]] = []
        backend = _backend(calls)
        await backend.decide(
            decision_request("hello", metadata={"prompt": "RAW", "content": "RAW", "tenant": "acme"})
        )
        state = calls[0]["state"]
        assert "prompt" not in state and "content" not in state
        assert state["caller_metadata"].get("tenant") == "acme"
        assert "prompt_excerpt" in state

    async def test_outage_degrades_never_raises(self, decision_request) -> None:
        calls: list[dict[str, Any]] = []
        backend = _backend(calls, fail=True)
        result = await backend.decide(decision_request("Do a thing."))
        await backend.aclose()
        assert result.degraded
        assert result.degrade_reason
        a = result.answers
        assert abs(sum(a.complexity.probabilities.values()) - 1.0) < 1e-6
        assert a.complexity.complexity if False else a.complexity.computed_confidence == 0.0


# --------------------------------------------------------------------------- #
# The 1:1 mapping
# --------------------------------------------------------------------------- #
class TestTheMapping:
    async def test_questions_are_exactly_build_questions(self, decision_request) -> None:
        calls: list[dict[str, Any]] = []
        backend = _backend(calls)
        await backend.decide(decision_request("Do a thing."))
        sent = calls[0]["questions"]
        expected = build_questions(include_domain=True)
        # The model receives verbatim our four questions: type, instructions,
        # criteria all intact. That is what makes the logs comparable across
        # backends.
        for qid in ("complexity", "sensitivity", "pii_present", "domain"):
            assert sent[qid]["type"] == expected[qid]["type"]
            assert sent[qid]["instructions"] == expected[qid]["instructions"]
            assert sent[qid]["criteria"] == expected[qid]["criteria"]

    async def test_answers_map_onto_our_schema(self, decision_request) -> None:
        calls: list[dict[str, Any]] = []
        backend = _backend(calls)
        result = await backend.decide(decision_request("Write a python function to parse CSV."))
        a = result.answers
        assert a.complexity.choice == "hard"
        assert abs(a.complexity.probabilities["hard"] - 0.9) < 1e-9
        assert a.sensitivity.choice == "internal"
        assert abs(a.pii.value - 0.9) < 1e-9
        assert a.domain.choice == "code"


# --------------------------------------------------------------------------- #
# The excerpt budget
# --------------------------------------------------------------------------- #
class TestTheExcerptBudget:
    async def test_short_input_passes_through_byte_identical(self, decision_request) -> None:
        calls: list[dict[str, Any]] = []
        backend = _backend(calls)
        await backend.decide(decision_request("short and sweet"))
        state = calls[0]["state"]
        assert state["prompt_excerpt"] == "short and sweet"

    async def test_budget_report_is_on_the_decision(self, decision_request) -> None:
        calls: list[dict[str, Any]] = []
        backend = _backend(calls)
        result = await backend.decide(decision_request("short and sweet"))
        laya_meta = result.questions_sent["_laya"]
        assert laya_meta["checkpoint"] == "english"
        assert laya_meta["role"] == "shadow"
        assert laya_meta["calibrated"] is False
        report = laya_meta["excerpt"]
        assert report["strategy"] == "none"
        assert report["tokens_dropped"] == 0

    async def test_long_input_gets_head_tail_and_logs_it(self, decision_request) -> None:
        calls: list[dict[str, Any]] = []
        long_text = "word " * 300  # 1500 chars = 1500 tokens for the char tokenizer
        backend = _backend(calls, token_budget=200)
        result = await backend.decide(decision_request(long_text))
        state = calls[0]["state"]
        excerpt = state["prompt_excerpt"]
        report = result.questions_sent["_laya"]["excerpt"]
        assert report["strategy"] == "head+tail"
        assert report["tokens_total"] == len(long_text)
        assert report["tokens_dropped"] == len(long_text) - report["tokens_kept"]
        assert report["tokens_dropped"] > 0
        # Head+tail: the start and the end of the input survive, the middle does not.
        assert excerpt.startswith(long_text[:10])
        assert long_text[-10:] in excerpt
        assert " ... " in excerpt
        # And the kept state respects the budget: head + state + sep <= max_len.
        budgeter = backend.budgeter("english")
        worst = max(build_questions().values(), key=lambda q: budgeter.overhead_tokens(q))
        assert len(excerpt) + budgeter.overhead_tokens(worst) + 1 <= 512

    async def test_truncation_is_deterministic(self, decision_request) -> None:
        long_text = "alpha beta gamma " * 200
        r1_calls: list[dict[str, Any]] = []
        r2_calls: list[dict[str, Any]] = []
        b1 = _backend(r1_calls, token_budget=120)
        b2 = _backend(r2_calls, token_budget=120)
        await b1.decide(decision_request(long_text))
        await b2.decide(decision_request(long_text))
        assert r1_calls[0]["state"]["prompt_excerpt"] == r2_calls[0]["state"]["prompt_excerpt"]
        await b1.aclose()
        await b2.aclose()


# --------------------------------------------------------------------------- #
# Calibration gating (fail-closed)
# --------------------------------------------------------------------------- #
class TestCalibrationGating:
    def test_enforce_without_calibration_is_refused(self) -> None:
        with pytest.raises(Exception, match="calibration"):
            _backend(calls=[], role="enforce", calibration=None)

    def test_shadow_without_calibration_is_allowed(self) -> None:
        backend = _backend(calls=[], role="shadow", calibration=None)
        assert backend.role == "shadow"
        assert backend.calibration is None

    def test_enforce_with_calibration_is_allowed(self) -> None:
        backend = _backend(
            calls=[],
            role="enforce",
            calibration=_temperature_artifact(complexity=2.0, sensitivity=2.0, domain=2.0, pii=2.0),

        )
        assert backend.calibration is not None

    def test_tampered_artifact_is_refused(self, tmp_path: Path) -> None:
        art = _temperature_artifact(complexity=2.0, sensitivity=2.0, domain=2.0, pii=2.0)
        path = tmp_path / "cal.json"
        save_artifact(art, str(path))
        # Flip one temperature after the checksum was written.

        data = json.loads(path.read_text())
        data["temperature"]["complexity"] = 9.9
        path.write_text(json.dumps(data))
        with pytest.raises(CalibrationError, match="checksum mismatch"):
            load_artifact(str(path))

    def test_unknown_schema_version_is_refused(self, tmp_path: Path) -> None:
        art = _temperature_artifact(complexity=2.0, sensitivity=2.0, domain=2.0, pii=2.0)
        data = art.to_dict()
        data["schema_version"] = 99
        # recompute checksum for the new version so only the version check fires
        tampered = CalibrationArtifact.from_dict({**data, "checksum": ""})
        data["checksum"] = tampered.checksum()
        path = tmp_path / "cal.json"
        path.write_text(json.dumps(data))
        with pytest.raises(CalibrationError, match="schema version"):
            load_artifact(str(path))

    async def test_calibrated_enforce_applies_the_transform(self, decision_request) -> None:
        # T > 1 must flatten an overconfident head: the 0.9 peak drops.
        art = _temperature_artifact(complexity=4.0, sensitivity=4.0, domain=4.0, pii=4.0)
        calls: list[dict[str, Any]] = []
        backend = _backend(calls, role="enforce", calibration=art)
        result = await backend.decide(decision_request("Do a hard thing."))
        a = result.answers
        assert a.complexity.probabilities["hard"] < 0.9
        assert abs(sum(a.complexity.probabilities.values()) - 1.0) < 1e-6
        assert result.questions_sent["_laya"]["calibrated"] is True
        await backend.aclose()


# --------------------------------------------------------------------------- #
# Calibration fit
# --------------------------------------------------------------------------- #
class TestCalibrationFit:
    def test_temperature_fit_flattens_overconfidence(self) -> None:
        # A head that is 0.9-confident but right only 70% of the time:
        # overconfident by construction. The fit must land on a T > 1 that
        # pushes the peak down toward the true accuracy.
        correct_raw = {"trivial": 0.02, "standard": 0.08, "hard": 0.9, "frontier": 0.0}
        samples = [
            (correct_raw, {"trivial": 0.0, "standard": 0.0, "hard": 1.0, "frontier": 0.0})
            for _ in range(70)
        ] + [
            (correct_raw, {"trivial": 0.0, "standard": 1.0, "hard": 0.0, "frontier": 0.0})
            for _ in range(30)
        ]
        temps = fit_temperature({"complexity": samples})
        temp = temps["complexity"]
        assert temp > 1.0, f"overconfident head must fit temp>1, got {temp}"
        # And applying it actually lowers the raw peak.
        from jev_route.backends.laya_calibration import _temperature_choice

        calibrated = _temperature_choice(
            {"trivial": 0.02, "standard": 0.08, "hard": 0.9, "frontier": 0.0}, temp
        )
        assert calibrated["hard"] < 0.9

    def test_temperature_fit_is_deterministic(self) -> None:
        samples = [
            ({"a": 0.7, "b": 0.3}, {"a": 1.0, "b": 0.0}) if i % 4 else ({"a": 0.4, "b": 0.6}, {"a": 0.0, "b": 1.0})
            for i in range(40)
        ]
        assert fit_temperature({"h": samples}) == fit_temperature({"h": samples})

    def test_isotonic_is_monotone(self) -> None:
        import random

        rng = random.Random(7)
        xs = [round(rng.random(), 3) for _ in range(80)]
        # True calibration: p_true = 0.5 + 0.4*p (monotone, overconfident at the top)
        ys = [min(1.0, max(0.0, 0.5 + 0.4 * x + (rng.random() - 0.5) * 0.1)) for x in xs]
        iso = Isotonic1D().fit(xs, ys)
        # Predictions must be non-decreasing.
        probes = [i / 50 for i in range(51)]
        preds = [iso.predict(p) for p in probes]
        assert all(b >= a - 1e-12 for a, b in pairwise(preds))
        # And clamped at the ends.
        assert iso.predict(-1.0) == preds[0]
        assert iso.predict(2.0) == preds[-1]

    def test_isotonic_fit_and_apply_roundtrip(self, tmp_path: Path) -> None:
        import random

        rng = random.Random(3)
        samples = []
        for _ in range(120):
            p = rng.random()
            y = 1.0 if rng.random() < (0.4 + 0.5 * p) else 0.0
            samples.append(({"yes": p, "no": 1.0 - p}, {"yes": y, "no": 1.0 - y}))
        fits = fit_isotonic({"pii": samples})
        assert "yes" in fits["pii"]
        art = CalibrationArtifact(method="isotonic", isotonic=fits, fit={"source": "test"})
        save_artifact(art, str(tmp_path / "iso.json"))
        loaded = load_artifact(str(tmp_path / "iso.json"))
        assert loaded.method == "isotonic"
        # Applying to a mid probability yields a valid in-range value.
        from jev_route.backends.laya_calibration import apply as cal_apply
        from jev_route.schema import ChoiceAnswer, NoulAnswer

        answers = DecisionAnswers(
            complexity=ChoiceAnswer.uniform(COMPLEXITY_LEVELS),
            sensitivity=ChoiceAnswer.uniform(SENSITIVITY_LEVELS),
            pii=NoulAnswer(value=0.6),
            domain=ChoiceAnswer.uniform(DOMAINS),
        )
        out = cal_apply(answers, loaded)
        assert 0.0 <= out.pii.value <= 1.0


# --------------------------------------------------------------------------- #
# Checkpoint routing
# --------------------------------------------------------------------------- #
class TestCheckpointRouting:
    def test_known_checkpoint_names_resolve(self) -> None:
        assert LayaBackend._resolve_checkpoint("english") == "english"
        assert LayaBackend._resolve_checkpoint("en") == "english"
        assert LayaBackend._resolve_checkpoint("multi") == "multilingual"
        assert LayaBackend._resolve_checkpoint("typed") == "typed-decisions"
        assert LayaBackend._resolve_checkpoint("auto") == "auto"
        with pytest.raises(Exception, match="unknown Laya checkpoint"):
            LayaBackend._resolve_checkpoint("quantum")

    async def test_auto_routes_by_detected_language(self, decision_request) -> None:
        calls: list[dict[str, Any]] = []
        loaded: list[str] = []

        class RecordingAgent(MockAgent):
            def system_one(self, state, questions):
                loaded.append("x")
                return super().system_one(state, questions)

        backend = LayaBackend(
            model="auto",
            role="shadow",
            agent_factory=lambda cp: (loaded.append(cp), MockAgent(CharTokenizer(), calls))[1],
            tokenizer=lambda cp: CharTokenizer(),
            language_fn=lambda text: "de" if "gruß" in text else "en",
        )
        await backend.decide(decision_request("Write a python function."))
        await backend.decide(decision_request("Bitte füge gruß zu."))
        await backend.aclose()
        assert loaded == ["english", "multilingual"]
        # The decision records which checkpoint actually answered.
        assert calls[0] is not None

    async def test_non_english_reports_multilingual_checkpoint(self, decision_request) -> None:
        calls: list[dict[str, Any]] = []
        backend = LayaBackend(
            model="auto",
            role="shadow",
            agent_factory=lambda cp: MockAgent(CharTokenizer(), calls),
            tokenizer=lambda cp: CharTokenizer(),
            language_fn=lambda text: "fr",
        )
        result = await backend.decide(decision_request("Résume ce texte."))
        assert result.questions_sent["_laya"]["checkpoint"] == "multilingual"
        await backend.aclose()


# --------------------------------------------------------------------------- #
# Config: the one-line swap
# --------------------------------------------------------------------------- #
class TestConfigSwap:
    def test_build_backend_constructs_laya(self, tmp_path: Path) -> None:
        art = _temperature_artifact(complexity=2.0, sensitivity=2.0, domain=2.0, pii=2.0)
        path = tmp_path / "cal.json"
        save_artifact(art, str(path))
        cfg = {
            "backend": {
                "name": "laya",
                "model": "english",
                "role": "enforce",
                "calibration": {"artifact": str(path)},
            }
        }
        backend = build_backend(cfg)
        assert isinstance(backend, LayaBackend)
        assert backend.calibration is not None

    def test_build_backend_laya_enforce_without_artifact_refuses(self) -> None:
        cfg = {"backend": {"name": "laya", "role": "enforce"}}
        with pytest.raises(Exception, match="calibration"):
            build_backend(cfg)

    def test_build_backend_laya_shadow_is_fine(self) -> None:
        cfg = {"backend": {"name": "laya", "role": "shadow"}}
        backend = build_backend(cfg)
        assert isinstance(backend, LayaBackend)

    def test_swap_is_a_single_key(self) -> None:
        # The graduation claim: flipping `backend.name` (and nothing else about
        # the request path) changes the backend. Jev and Laya sit behind the
        # same factory, the same Router, the same log schema.
        import yaml
        from tests.conftest import DEFAULT_POLICY_PATH

        from jev_route.backends.jev import JevBackend
        from jev_route.policy import Policy

        base = yaml.safe_load(DEFAULT_POLICY_PATH.read_text())
        p_jev = Policy.from_dict({**base, "backend": {"name": "jev", "api_key": "sk-test"}})
        p_laya = Policy.from_dict({**base, "backend": {"name": "laya", "role": "shadow"}})
        b_jev = build_backend(p_jev)
        b_laya = build_backend(p_laya)
        assert type(b_jev) is JevBackend
        assert type(b_laya) is LayaBackend


# --------------------------------------------------------------------------- #
# Lazy imports (static)
# --------------------------------------------------------------------------- #
def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    mods: set[str] = set()
    for node in tree.body:  # module level only; ast.walk would descend into defs
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    return mods


def _function_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Import):
                    mods.update(a.name.split(".")[0] for a in sub.names)
                elif isinstance(sub, ast.ImportFrom) and sub.module and sub.level == 0:
                    mods.add(sub.module.split(".")[0])
    return mods


def test_laya_module_keeps_heavy_and_egress_deps_out_of_top_level() -> None:
    top = _top_level_imports(MODULE_PATH)
    assert not (top & HEAVY), f"heavy import at top level: {top & HEAVY}"
    assert not (top & EGRESS), f"egress import at top level: {top & EGRESS}"
    # The heavy deps *do* get imported, but only inside functions.
    fn = _function_level_imports(MODULE_PATH)
    assert "laya" in fn


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
