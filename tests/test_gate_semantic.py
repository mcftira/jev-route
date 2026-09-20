"""Layer 2 of the gate: config, artifact, promotion, and the layer itself.

The tests here are written as claims about a security boundary rather than as
coverage of a module, because that is what this layer is. The three that matter
most, and that the rest support:

* a semantic layer that says "public" cannot cancel a deterministic floor;
* ``mode: enforce`` without measured evidence refuses to start, loudly;
* an artifact that will not swear it was not trained on blocked content does not
  load at all.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from jev_route.gate import HardGate
from jev_route.gate_semantic import (
    ASSERTABLE_LEVELS,
    DISAGREE_SEMANTIC_MISS,
    DISAGREE_SEMANTIC_ONLY,
    SEMANTIC_ARTIFACT_KIND,
    BlockedMetadataPolicy,
    EnforceCriteria,
    LabeledExample,
    LexiconScorer,
    NullScorer,
    SemanticArtifact,
    SemanticArtifactError,
    SemanticAssessment,
    SemanticConfigError,
    SemanticGateError,
    SemanticGatePolicy,
    SemanticLayer,
    check_promotion,
    fit_scorer,
    load_shadow_metrics,
    measure_promotion_metrics,
    measure_shadow_log,
    merge_metrics,
    resolve_floor,
)
from jev_route.policy import Policy, PolicyError
from jev_route.schema import RequestFeatures

# --------------------------------------------------------------------------- #
# Fixtures and builders
# --------------------------------------------------------------------------- #
CARD_PROMPT = "Charge card 4111 1111 1111 1111 and wire the rest."
CONTEXTUAL_PROMPT = (
    "After three warnings we agreed to let him go before the announcement, "
    "and the discussion with counsel about the settlement stays between us."
)
BENIGN_PROMPT = "Summarize the roadmap for the team and list the open questions."


def good_metrics(**overrides: Any) -> dict[str, Any]:
    """A metrics block that passes the default criteria, so tests can break one number."""
    base: dict[str, Any] = {
        "measured_at": "2026-01-01T00:00:00+00:00",
        "model_version": "test-semantic-1",
        "scorer": "lexicon-logreg",
        "threshold": 0.5,
        "n_examples": 1500,
        "n_positives": 400,
        "n_negatives": 1100,
        "n_layer1_fired": 120,
        "recall": 0.9975,
        "false_positive_rate": 0.008,
        "disagreement_rate": 0.02,
        "semantic_only_rate": 0.018,
        "semantic_miss_rate": 0.008,
        "layer1_recall": 0.3,
        "shadow_n_examples": 4200,
        "shadow_model_version": "test-semantic-1",
    }
    base.update(overrides)
    return base


def artifact_payload(**overrides: Any) -> dict[str, Any]:
    """A minimal but loadable artifact envelope."""
    payload: dict[str, Any] = {
        "kind": SEMANTIC_ARTIFACT_KIND,
        "artifact_version": "1",
        "model_version": "test-semantic-1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "threshold": 0.5,
        "level": "confidential",
        "scorer": {
            "name": "lexicon-logreg",
            "bias": -2.0,
            "tokenizer": {"lower": True, "min_token_chars": 2, "ngram_max": 2},
            "weights": {"disciplinary": 4.0, "settlement": 3.0, "patient": 3.0, "privileged": 3.0},
        },
        "metrics": good_metrics(),
        "provenance": {
            "positives": "synthetic",
            "negatives": "production",
            "trained_on_blocked_content": False,
            "note": "test fixture",
        },
    }
    payload.update(overrides)
    return payload


def write_artifact(tmp_path: Path, payload: dict[str, Any] | None = None, *, name: str = "semantic") -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_json = json.dumps(payload if payload is not None else artifact_payload(), sort_keys=True)
    path.write_text(payload_json, encoding="utf-8")
    return path


class StubScorer:
    """A scorer whose answer the test chooses. Stands in for a trained model."""

    name = "stub"
    model_version = "stub-1"

    def __init__(self, score: float = 0.9) -> None:
        self.score_value = score
        self.seen: list[str] = []

    def score(self, text: str, *, features: RequestFeatures | None = None) -> float:
        del features
        self.seen.append(text)
        return self.score_value


class ExplodingScorer:
    name = "exploding"
    model_version = "exploding-1"

    def score(self, text: str, *, features: RequestFeatures | None = None) -> float:
        del text, features
        raise RuntimeError("model file vanished mid-request")


@pytest.fixture
def artifact_path(tmp_path: Path) -> Path:
    return write_artifact(tmp_path)


@pytest.fixture
def promotable_config(artifact_path: Path) -> SemanticGatePolicy:
    """Config whose artifact genuinely meets the default criteria."""
    return SemanticGatePolicy(mode="enforce", artifact=str(artifact_path))


# --------------------------------------------------------------------------- #
# 1. Configuration: a typo fails the deploy, not the request
# --------------------------------------------------------------------------- #
class TestSemanticConfig:
    def test_the_default_is_shadow_and_inert(self) -> None:
        config = SemanticGatePolicy()
        assert config.mode == "shadow"
        assert config.artifact is None
        assert config.runs is True
        assert config.enforces is False

    def test_off_does_not_run(self) -> None:
        assert SemanticGatePolicy(mode="off").runs is False

    @pytest.mark.parametrize("mode", ["off", "shadow", "enforce", "SHADOW", " Enforce "])
    def test_mode_is_case_insensitive_and_validated(self, mode: str) -> None:
        assert SemanticGatePolicy.parse({"mode": mode}).mode == mode.strip().lower()

    @pytest.mark.parametrize("mode", ["enfoce", "on", "blocking", "", "true"])
    def test_an_unknown_mode_is_fatal(self, mode: str) -> None:
        with pytest.raises(SemanticConfigError, match="mode"):
            SemanticGatePolicy.parse({"mode": mode})

    def test_an_unknown_key_is_fatal(self) -> None:
        # `on_uncertain` drops keys it does not know; this section must not, because
        # an ignored key here is a gate that is not doing what its config says.
        with pytest.raises(SemanticConfigError, match="unknown keys"):
            SemanticGatePolicy.parse({"mode": "shadow", "artifct": "/tmp/x"})

    @pytest.mark.parametrize("threshold", [0.0, 1.0, -0.1, 1.5, "0.5", None, True])
    def test_threshold_must_be_a_number_strictly_inside_the_unit_interval(self, threshold: Any) -> None:
        with pytest.raises(SemanticConfigError, match="threshold"):
            SemanticGatePolicy.parse({"threshold": threshold})

    def test_threshold_accepts_the_open_interval(self) -> None:
        assert SemanticGatePolicy.parse({"threshold": 0.05}).threshold == 0.05
        assert SemanticGatePolicy.parse({"threshold": 0.95}).threshold == 0.95

    @pytest.mark.parametrize("level", ASSERTABLE_LEVELS)
    def test_assertable_levels_are_accepted(self, level: str) -> None:
        assert SemanticGatePolicy.parse({"level": level}).level == level

    @pytest.mark.parametrize("level", ["public", "internal", "secret", ""])
    def test_a_level_that_would_loosen_is_refused(self, level: str) -> None:
        # Layer 2 may only make routing stricter, so it has no low levels to assert.
        with pytest.raises(SemanticConfigError, match="level"):
            SemanticGatePolicy.parse({"level": level})

    def test_artifact_must_be_a_path(self) -> None:
        with pytest.raises(SemanticConfigError, match="artifact"):
            SemanticGatePolicy.parse({"artifact": 17})
        with pytest.raises(SemanticConfigError, match="artifact"):
            SemanticGatePolicy.parse({"artifact": "   "})

    def test_a_non_mapping_section_is_fatal(self) -> None:
        with pytest.raises(SemanticConfigError, match="mapping"):
            SemanticGatePolicy.parse("shadow")

    def test_parse_of_nothing_is_the_default(self) -> None:
        assert SemanticGatePolicy.parse(None) == SemanticGatePolicy()

    def test_config_round_trips_through_to_dict(self) -> None:
        config = SemanticGatePolicy.parse({"mode": "enforce", "artifact": "/a/b", "threshold": 0.7})
        assert SemanticGatePolicy.parse(config.to_dict()) == config


class TestEnforceCriteriaConfig:
    def test_the_defaults_are_conservative(self) -> None:
        criteria = EnforceCriteria()
        assert criteria.min_recall == 0.99
        assert criteria.max_false_positive_rate == 0.02
        assert criteria.max_disagreement_rate == 0.05
        assert criteria.max_semantic_miss_rate == 0.02
        assert criteria.min_positive_examples == 200
        assert criteria.min_negative_examples == 500
        assert criteria.min_shadow_examples == 1000

    def test_the_recall_bar_may_be_raised_but_not_removed(self) -> None:
        assert EnforceCriteria.parse({"min_recall": 0.999}).min_recall == 0.999
        with pytest.raises(SemanticConfigError, match="floor"):
            EnforceCriteria.parse({"min_recall": 0.6})
        with pytest.raises(SemanticConfigError, match="floor"):
            EnforceCriteria.parse({"min_recall": EnforceCriteria.MIN_ALLOWED_RECALL - 0.001})

    def test_the_floor_itself_is_accepted(self) -> None:
        assert EnforceCriteria.parse({"min_recall": EnforceCriteria.MIN_ALLOWED_RECALL}).min_recall == 0.9

    def test_an_unknown_criterion_is_fatal(self) -> None:
        # A misspelled criterion would silently vanish from the promotion check,
        # which is the same failure as not having it.
        with pytest.raises(SemanticConfigError, match="unknown keys"):
            EnforceCriteria.parse({"min_recalls": 0.99})

    @pytest.mark.parametrize("key", ["min_recall", "max_false_positive_rate", "max_disagreement_rate"])
    def test_rates_must_be_probabilities(self, key: str) -> None:
        with pytest.raises(SemanticConfigError):
            EnforceCriteria.parse({key: 1.5})
        with pytest.raises(SemanticConfigError):
            EnforceCriteria.parse({key: -0.1})

    def test_sample_sizes_must_be_non_negative_integers(self) -> None:
        with pytest.raises(SemanticConfigError):
            EnforceCriteria.parse({"min_positive_examples": -1})
        with pytest.raises(SemanticConfigError):
            EnforceCriteria.parse({"min_shadow_examples": "many"})

    def test_a_boolean_is_not_a_number(self) -> None:
        # True == 1 in Python, and `min_recall: true` reading as 1.0 is exactly the
        # kind of silent nonsense a validated config section exists to refuse.
        with pytest.raises(SemanticConfigError):
            EnforceCriteria.parse({"min_recall": True})


class TestBlockedMetadataConfig:
    def test_defaults_to_on_with_a_derived_path(self) -> None:
        config = BlockedMetadataPolicy.parse(None)
        assert config.enabled is True
        assert config.path is None

    def test_boolean_shorthand(self) -> None:
        assert BlockedMetadataPolicy.parse(False).enabled is False
        assert BlockedMetadataPolicy.parse(True).enabled is True

    def test_unknown_key_is_fatal(self) -> None:
        with pytest.raises(SemanticConfigError, match="unknown keys"):
            BlockedMetadataPolicy.parse({"enable": True})

    def test_path_must_be_a_path(self) -> None:
        with pytest.raises(SemanticConfigError, match="path"):
            BlockedMetadataPolicy.parse({"path": 3})


class TestPolicyIntegration:
    def policy_doc(self, gate: Any) -> dict[str, Any]:
        return {
            "version": 1,
            "tiers": {"local": ["l"], "cheap": ["c"], "strong": ["s"]},
            "rules": [{"id": "default", "then": {"tier": "cheap"}}],
            "gate": gate,
        }

    def test_the_sections_are_parsed_at_load_time(self) -> None:
        policy = Policy.from_dict(
            self.policy_doc(
                {
                    "semantic": {"mode": "shadow", "threshold": 0.62, "level": "regulated"},
                    "blocked_metadata": {"enabled": False},
                }
            )
        )
        assert policy.semantic_gate.mode == "shadow"
        assert policy.semantic_gate.threshold == 0.62
        assert policy.semantic_gate.level == "regulated"
        assert policy.blocked_metadata.enabled is False

    def test_a_policy_without_the_sections_gets_the_safe_defaults(self) -> None:
        policy = Policy.from_dict(self.policy_doc({"on_force_local": "skip_backend"}))
        assert policy.semantic_gate == SemanticGatePolicy()
        assert policy.blocked_metadata == BlockedMetadataPolicy()

    def test_a_typo_in_the_semantic_section_is_a_policy_error(self) -> None:
        with pytest.raises(PolicyError, match=r"gate\.semantic\.mode"):
            Policy.from_dict(self.policy_doc({"semantic": {"mode": "enfoce"}}))

    def test_a_typo_in_the_criteria_is_a_policy_error(self) -> None:
        with pytest.raises(PolicyError, match="enforce_requires"):
            Policy.from_dict(self.policy_doc({"semantic": {"enforce_requires": {"min_recall": 0.5}}}))

    def test_a_gate_section_that_is_not_a_mapping_is_a_policy_error(self) -> None:
        with pytest.raises(PolicyError, match="must be a mapping"):
            Policy.from_dict(self.policy_doc("skip_backend"))

    def test_overrides_revalidate(self) -> None:
        policy = Policy.from_dict(self.policy_doc({}))
        with pytest.raises(PolicyError):
            policy.with_overrides(gate={"semantic": {"mode": "enforce", "level": "public"}})


# --------------------------------------------------------------------------- #
# 2. The invariant: layer 2 can raise the floor, never lower it
# --------------------------------------------------------------------------- #
class TestResolveFloor:
    @pytest.mark.parametrize(
        ("layer1", "layer2", "expected"),
        [
            ("regulated", "public", "regulated"),
            ("regulated", None, "regulated"),
            ("regulated", "confidential", "regulated"),
            ("confidential", "regulated", "regulated"),
            (None, "confidential", "confidential"),
            (None, None, None),
            ("internal", "public", "internal"),
            ("", "", None),
        ],
    )
    def test_the_maximum_wins(self, layer1: str | None, layer2: str | None, expected: str | None) -> None:
        assert resolve_floor(layer1, layer2) == expected

    def test_a_semantic_public_cannot_cancel_a_deterministic_regulated(self) -> None:
        # The sentence from the design review, as an assertion: layer 1 said
        # "regulated" because a checksum matched, layer 2 said "public" because a
        # model was unconvinced. The checksum is the floor.
        assert resolve_floor("regulated", "public") == "regulated"

    def test_order_does_not_matter(self) -> None:
        assert resolve_floor("confidential", "regulated") == resolve_floor("regulated", "confidential")

    def test_an_unknown_level_cannot_outrank_a_known_one(self) -> None:
        # level_index returns -1 for anything off the ladder, so a garbage string
        # is inert rather than maximally strict or maximally lax.
        assert resolve_floor("internal", "nonsense") == "internal"

    def test_three_layers_still_take_the_maximum(self) -> None:
        assert resolve_floor("internal", "confidential", "regulated") == "regulated"


# --------------------------------------------------------------------------- #
# 3. The artifact: load-time provenance is the bootstrap paradox, enforced
# --------------------------------------------------------------------------- #
class TestArtifact:
    def test_round_trip_through_a_directory(self, tmp_path: Path) -> None:
        artifact = SemanticArtifact.from_dict(artifact_payload())
        written = artifact.save(tmp_path / "sem")
        assert written.name == "semantic.json"
        loaded = SemanticArtifact.load(tmp_path / "sem")
        assert loaded.model_version == artifact.model_version
        assert loaded.scorer.weights == artifact.scorer.weights
        assert loaded.metrics == artifact.metrics
        assert loaded.provenance == artifact.provenance

    def test_round_trip_through_a_bare_json_file(self, tmp_path: Path) -> None:
        path = write_artifact(tmp_path, name="gate-semantic.json")
        assert SemanticArtifact.load(path).model_version == "test-semantic-1"

    def test_the_saved_file_is_readable_json_with_no_pickle(self, tmp_path: Path) -> None:
        # A reviewer must be able to see every token the model weights without
        # running anything, and a pickle in an artifact directory is arbitrary code
        # execution the moment somebody loads a file they downloaded.
        path = SemanticArtifact.from_dict(artifact_payload()).save(tmp_path / "sem")
        raw = path.read_bytes()
        assert json.loads(raw)["scorer"]["weights"]["disciplinary"] == 4.0
        assert b"\x80" not in raw  # pickle protocol marker
        assert b"numpy" not in raw

    def test_a_missing_artifact_says_how_to_get_one(self, tmp_path: Path) -> None:
        with pytest.raises(SemanticArtifactError, match="fit_scorer"):
            SemanticArtifact.load(tmp_path / "does-not-exist")

    def test_a_distilled_model_artifact_is_not_a_gate_artifact(self, tmp_path: Path) -> None:
        path = write_artifact(tmp_path, artifact_payload(kind="jev_route.distill.artifact"), name="distilled.json")
        with pytest.raises(SemanticArtifactError, match="kind"):
            SemanticArtifact.load(path)

    def test_an_unknown_artifact_version_is_refused(self, tmp_path: Path) -> None:
        path = write_artifact(tmp_path, artifact_payload(artifact_version="99"), name="v99.json")
        with pytest.raises(SemanticArtifactError, match="artifact_version"):
            SemanticArtifact.load(path)

    def test_an_artifact_without_metrics_can_never_be_promoted(self, tmp_path: Path) -> None:
        payload = artifact_payload()
        del payload["metrics"]
        path = write_artifact(tmp_path, payload, name="no-metrics.json")
        with pytest.raises(SemanticArtifactError, match="metrics"):
            SemanticArtifact.load(path)

    def test_a_non_finite_weight_is_refused(self, tmp_path: Path) -> None:
        payload = artifact_payload()
        payload["scorer"]["weights"] = {"disciplinary": float("nan")}
        path = write_artifact(tmp_path, payload, name="nan.json")
        # json.dumps writes NaN as a bare `NaN` token, which json.loads reads back;
        # the artifact check is what stops it reaching a score.
        with pytest.raises(SemanticArtifactError, match="non-finite"):
            SemanticArtifact.load(path)

    def test_an_unknown_scorer_family_is_refused(self, tmp_path: Path) -> None:
        payload = artifact_payload()
        payload["scorer"]["name"] = "transformer"
        path = write_artifact(tmp_path, payload, name="family.json")
        with pytest.raises(SemanticArtifactError, match=r"scorer\.name"):
            SemanticArtifact.load(path)

    def test_corrupt_json_is_refused_with_a_reason(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(SemanticArtifactError, match="JSON"):
            SemanticArtifact.load(path)


class TestProvenanceIsTheBootstrapParadox:
    def test_an_artifact_trained_on_blocked_content_does_not_load(self, tmp_path: Path) -> None:
        payload = artifact_payload()
        payload["provenance"]["trained_on_blocked_content"] = True
        path = write_artifact(tmp_path, payload, name="tainted.json")
        with pytest.raises(SemanticArtifactError, match="blocked"):
            SemanticArtifact.load(path)

    def test_production_positives_are_blocked_content_and_are_refused(self, tmp_path: Path) -> None:
        # Production positives are, by construction, the requests the gate caught.
        payload = artifact_payload()
        payload["provenance"]["positives"] = "production"
        path = write_artifact(tmp_path, payload, name="prod-pos.json")
        with pytest.raises(SemanticArtifactError, match="positives"):
            SemanticArtifact.load(path)

    def test_an_artifact_that_will_not_say_is_refused(self, tmp_path: Path) -> None:
        payload = artifact_payload()
        del payload["provenance"]
        path = write_artifact(tmp_path, payload, name="mute.json")
        with pytest.raises(SemanticArtifactError, match="provenance"):
            SemanticArtifact.load(path)

    def test_an_artifact_that_omits_the_one_field_that_matters_is_refused(self, tmp_path: Path) -> None:
        payload = artifact_payload()
        del payload["provenance"]["trained_on_blocked_content"]
        path = write_artifact(tmp_path, payload, name="partial.json")
        with pytest.raises(SemanticArtifactError, match="trained_on_blocked_content"):
            SemanticArtifact.load(path)

    @pytest.mark.parametrize("source", ["synthetic", "public", "synthetic+public"])
    def test_the_allowed_positive_sources_load(self, tmp_path: Path, source: str) -> None:
        payload = artifact_payload()
        payload["provenance"]["positives"] = source
        path = write_artifact(tmp_path, payload, name=f"src-{source.replace('+', '-')}.json")
        assert SemanticArtifact.load(path).provenance.positives == source

    def test_real_negatives_are_allowed_because_they_already_left(self, tmp_path: Path) -> None:
        payload = artifact_payload()
        payload["provenance"]["negatives"] = "production"
        path = write_artifact(tmp_path, payload, name="prod-neg.json")
        assert SemanticArtifact.load(path).provenance.negatives == "production"


# --------------------------------------------------------------------------- #
# 4. The scorer
# --------------------------------------------------------------------------- #
class TestLexiconScorer:
    def test_evidence_moves_the_score_up(self) -> None:
        scorer = LexiconScorer({"disciplinary": 3.0, "hearing": 2.0}, bias=-3.0)
        assert scorer.score("a routine build log") < 0.5
        assert scorer.score("the disciplinary hearing was scheduled") > 0.5

    def test_more_evidence_is_more_score(self) -> None:
        scorer = LexiconScorer({"disciplinary": 2.0, "hearing": 2.0}, bias=-3.0)
        assert scorer.score("disciplinary") < scorer.score("disciplinary hearing")

    def test_repeating_a_token_does_not_multiply_the_evidence(self) -> None:
        # Presence, not count: nine mentions of "patient" in a long support thread
        # are not nine times the evidence, and count features make the score a
        # function of prompt length.
        scorer = LexiconScorer({"patient": 2.0}, bias=-1.0)
        assert scorer.score("patient") == pytest.approx(scorer.score("patient patient patient"))

    def test_empty_text_scores_the_prior(self) -> None:
        scorer = LexiconScorer({"disciplinary": 3.0}, bias=-1.5)
        assert scorer.score("") == pytest.approx(1.0 / (1.0 + math.exp(1.5)))

    def test_a_scorer_with_no_weights_is_an_error_not_a_silent_zero(self) -> None:
        with pytest.raises(SemanticArtifactError):
            LexiconScorer({})

    def test_unseen_tokens_are_ignored(self) -> None:
        scorer = LexiconScorer({"disciplinary": 3.0}, bias=-3.0)
        assert scorer.score("kubernetes pods scheduling") == pytest.approx(scorer.score(""))

    def test_the_null_scorer_never_fires_and_says_why(self) -> None:
        scorer = NullScorer("nothing trained yet")
        assert scorer.score(CONTEXTUAL_PROMPT) == 0.0
        assert scorer.reason == "nothing trained yet"


# --------------------------------------------------------------------------- #
# 5. The assessment
# --------------------------------------------------------------------------- #
class TestSemanticAssessment:
    def test_a_shadow_assessment_projects_no_floor(self) -> None:
        assessment = SemanticAssessment(
            mode="shadow", ran=True, score=0.97, fired=True, asserted_level="confidential", enforced=False
        )
        # The claim is recorded; acting on it is not.
        assert assessment.asserted_level == "confidential"
        assert assessment.sensitivity_floor is None
        assert assessment.force_local is False

    def test_an_enforced_assessment_projects_its_floor(self) -> None:
        assessment = SemanticAssessment(
            mode="enforce", ran=True, score=0.97, fired=True, asserted_level="regulated", enforced=True
        )
        assert assessment.sensitivity_floor == "regulated"
        assert assessment.force_local is True

    def test_a_quiet_assessment_asserts_nothing(self) -> None:
        assessment = SemanticAssessment(mode="enforce", ran=True, score=0.1, fired=False, enforced=False)
        assert assessment.asserted_level is None
        assert assessment.sensitivity_floor is None

    def test_round_trip(self) -> None:
        assessment = SemanticAssessment(
            mode="shadow",
            ran=True,
            score=0.4242,
            fired=False,
            layer1_fired=True,
            layer1_floor="regulated",
            layer1_detectors=("us_ssn", "payment_card"),
            layer1_blocks_backend=True,
            disagreement=DISAGREE_SEMANTIC_MISS,
            model_version="test-1",
        )
        assert SemanticAssessment.from_dict(json.loads(json.dumps(assessment.to_dict()))) == assessment

    def test_the_serialized_assessment_carries_no_content(self) -> None:
        assessment = SemanticAssessment(
            mode="shadow",
            ran=True,
            score=0.9,
            fired=True,
            asserted_level="confidential",
            layer1_fired=True,
            layer1_detectors=("payment_card",),
            disagreement=DISAGREE_SEMANTIC_ONLY,
        )
        blob = json.dumps(assessment.to_dict())
        for fragment in ("4111", CARD_PROMPT, CONTEXTUAL_PROMPT, "Charge"):
            assert fragment not in blob

    def test_inert_says_it_did_not_run(self) -> None:
        assessment = SemanticAssessment.inert("shadow", "no artifact")
        assert assessment.ran is False
        assert assessment.fired is False
        assert assessment.reason == "no artifact"
        assert assessment.sensitivity_floor is None


# --------------------------------------------------------------------------- #
# 6. The layer: shadow measures, enforce acts, nothing in between
# --------------------------------------------------------------------------- #
class TestSemanticLayerModes:
    def test_off_does_not_score(self) -> None:
        scorer = StubScorer(0.99)
        layer = SemanticLayer(SemanticGatePolicy(mode="off"), scorer=scorer)
        assessment = layer.assess(CONTEXTUAL_PROMPT)
        assert layer.active is False
        assert assessment.ran is False
        assert assessment.mode == "off"
        assert scorer.seen == []  # never even looked at the text

    def test_shadow_with_no_artifact_is_inert_and_not_an_error(self) -> None:
        # The default state of a fresh deployment: configured, untrained, harmless.
        layer = SemanticLayer(SemanticGatePolicy())
        assert layer.active is False
        assessment = layer.assess(CONTEXTUAL_PROMPT)
        assert assessment.ran is False
        assert "artifact" in (assessment.reason or "")

    def test_enforce_with_no_artifact_refuses_to_build(self) -> None:
        with pytest.raises(SemanticGateError, match="refused"):
            SemanticLayer(SemanticGatePolicy(mode="enforce"))

    def test_enforce_with_an_unloadable_artifact_refuses_to_build(self, tmp_path: Path) -> None:
        config = SemanticGatePolicy(mode="enforce", artifact=str(tmp_path / "missing"))
        with pytest.raises(SemanticGateError, match="refused"):
            SemanticLayer(config)

    def test_shadow_with_an_unloadable_artifact_degrades_to_inert(self, tmp_path: Path) -> None:
        # Shadow is a measurement. A missing model file costs telemetry, not requests.
        layer = SemanticLayer(SemanticGatePolicy(mode="shadow", artifact=str(tmp_path / "missing")))
        assert layer.active is False
        assert layer.assess(CONTEXTUAL_PROMPT).ran is False

    def test_shadow_scores_and_records_the_disagreement(self) -> None:
        layer = SemanticLayer(SemanticGatePolicy(mode="shadow"), scorer=StubScorer(0.93))
        assessment = layer.assess(CONTEXTUAL_PROMPT, verdict=HardGate().scan(BENIGN_PROMPT))
        assert assessment.ran is True
        assert assessment.fired is True
        assert assessment.score == pytest.approx(0.93)
        assert assessment.disagreement == DISAGREE_SEMANTIC_ONLY
        assert assessment.layer1_fired is False
        # ...and it still projected nothing the router could act on.
        assert assessment.enforced is False
        assert assessment.sensitivity_floor is None
        assert assessment.force_local is False

    def test_enforce_projects_the_floor_it_asserts(self) -> None:
        layer = SemanticLayer(
            SemanticGatePolicy(mode="enforce", level="regulated"), scorer=StubScorer(0.93), metrics=good_metrics()
        )
        assessment = layer.assess(CONTEXTUAL_PROMPT, verdict=HardGate().scan(BENIGN_PROMPT))
        assert assessment.enforced is True
        assert assessment.sensitivity_floor == "regulated"
        assert assessment.force_local is True

    def test_a_quiet_request_projects_nothing_even_in_enforce(self) -> None:
        layer = SemanticLayer(
            SemanticGatePolicy(mode="enforce"), scorer=StubScorer(0.02), metrics=good_metrics()
        )
        assessment = layer.assess(BENIGN_PROMPT, verdict=HardGate().scan(BENIGN_PROMPT))
        assert assessment.fired is False
        assert assessment.enforced is False
        assert assessment.sensitivity_floor is None

    def test_layer1_detectors_are_recorded_by_id_only(self) -> None:
        layer = SemanticLayer(SemanticGatePolicy(), scorer=StubScorer(0.9))
        verdict = HardGate().scan(CARD_PROMPT)
        assessment = layer.assess(CARD_PROMPT, verdict=verdict)
        assert "payment_card" in assessment.layer1_detectors
        assert assessment.layer1_floor == "regulated"
        assert assessment.layer1_blocks_backend is True
        blob = json.dumps(assessment.to_dict())
        assert "4111" not in blob
        assert CARD_PROMPT not in blob

    def test_a_semantic_miss_is_named_as_such(self) -> None:
        # Layer 1 caught a checksum; layer 2 saw nothing. That is the disagreement
        # that should stop a promotion, and it is not the same fact as the other one.
        layer = SemanticLayer(SemanticGatePolicy(), scorer=StubScorer(0.01))
        assessment = layer.assess(CARD_PROMPT, verdict=HardGate().scan(CARD_PROMPT))
        assert assessment.layer1_fired is True
        assert assessment.fired is False
        assert assessment.disagreement == DISAGREE_SEMANTIC_MISS

    def test_advisory_only_hits_do_not_count_as_layer1_firing(self) -> None:
        # "Explain how HIPAA works" sets no floor, so a layer-2 fire there is the
        # layer doing its job, not a contradiction of a deterministic finding.
        layer = SemanticLayer(SemanticGatePolicy(), scorer=StubScorer(0.9))
        text = "Explain how HIPAA actually works and who it applies to."
        verdict = HardGate().scan(text)
        assert verdict.sensitivity_floor is None
        assessment = layer.assess(text, verdict=verdict)
        assert assessment.layer1_fired is False
        assert assessment.disagreement == DISAGREE_SEMANTIC_ONLY

    def test_a_scorer_that_raises_is_telemetry_not_an_outage(self) -> None:
        layer = SemanticLayer(SemanticGatePolicy(mode="enforce"), scorer=ExplodingScorer(), metrics=good_metrics())
        assessment = layer.assess(CONTEXTUAL_PROMPT)
        assert assessment.ran is False
        assert assessment.degraded is True
        assert "model file vanished" in (assessment.reason or "")
        assert assessment.sensitivity_floor is None  # layer 1 is still the whole floor

    def test_a_non_finite_score_is_refused(self) -> None:
        layer = SemanticLayer(SemanticGatePolicy(), scorer=StubScorer(float("nan")))
        assessment = layer.assess(CONTEXTUAL_PROMPT)
        assert assessment.degraded is True
        assert "non-finite" in (assessment.reason or "")

    def test_an_out_of_range_score_is_clamped(self) -> None:
        layer = SemanticLayer(SemanticGatePolicy(), scorer=StubScorer(3.5))
        assert layer.assess(CONTEXTUAL_PROMPT).score == 1.0

    def test_the_artifact_threshold_wins_over_config(self, artifact_path: Path) -> None:
        payload = artifact_payload(threshold=0.75)
        path = write_artifact(artifact_path.parent, payload, name="t75.json")
        layer = SemanticLayer(SemanticGatePolicy(mode="shadow", artifact=str(path), threshold=0.75))
        assert layer.threshold == 0.75

    def test_enforce_refuses_a_threshold_the_metrics_do_not_describe(self, tmp_path: Path) -> None:
        path = write_artifact(tmp_path, artifact_payload(threshold=0.75), name="t75.json")
        config = SemanticGatePolicy(mode="enforce", artifact=str(path), threshold=0.5)
        with pytest.raises(SemanticGateError, match="threshold"):
            SemanticLayer(config)

    def test_stats_count_what_happened(self) -> None:
        class KeyedScorer(StubScorer):
            """Fires on one word, so a test can choose which row disagrees how."""

            def score(self, text: str, *, features: RequestFeatures | None = None) -> float:
                super().score(text, features=features)
                return 0.9 if "settlement" in text else 0.05

        layer = SemanticLayer(SemanticGatePolicy(mode="shadow"), scorer=KeyedScorer())
        gate = HardGate()
        # Fires where layer 1 saw nothing, and stays quiet where layer 1 found a card.
        layer.assess(CONTEXTUAL_PROMPT, verdict=gate.scan(BENIGN_PROMPT))
        layer.assess(CARD_PROMPT, verdict=gate.scan(CARD_PROMPT))
        stats = layer.stats()
        assert stats["assessed"] == 2
        assert stats["fired"] == 1
        assert stats[DISAGREE_SEMANTIC_ONLY] == 1
        assert stats[DISAGREE_SEMANTIC_MISS] == 1
        assert stats["enforced"] == 0
        assert stats["errors"] == 0
        assert stats["fire_rate"] == pytest.approx(0.5)
        assert stats["disagreement_rate"] == pytest.approx(1.0)

    def test_features_are_passed_through_to_the_scorer(self) -> None:
        seen: list[Any] = []

        class Watching(StubScorer):
            def score(self, text: str, *, features: RequestFeatures | None = None) -> float:
                seen.append(features)
                return super().score(text, features=features)

        layer = SemanticLayer(SemanticGatePolicy(), scorer=Watching())
        features = RequestFeatures(char_len=17)
        layer.assess(CONTEXTUAL_PROMPT, features=features)
        assert seen == [features]


# --------------------------------------------------------------------------- #
# 7. Promotion is a measurement, not a flag
# --------------------------------------------------------------------------- #
class TestPromotion:
    def test_a_measured_artifact_passes(self) -> None:
        report = check_promotion(EnforceCriteria(), good_metrics())
        assert report.ok is True
        assert report.failures == ()
        assert "PROMOTION CRITERIA MET" in report.message()

    def test_enforce_builds_when_the_artifact_earns_it(self, tmp_path: Path) -> None:
        path = write_artifact(tmp_path, artifact_payload(), name="good.json")
        layer = SemanticLayer(SemanticGatePolicy(mode="enforce", artifact=str(path)))
        assert layer.active is True
        assert layer.mode == "enforce"

    @pytest.mark.parametrize(
        ("key", "value", "needle"),
        [
            ("recall", 0.85, "recall"),
            ("false_positive_rate", 0.4, "false_positive_rate"),
            ("disagreement_rate", 0.31, "disagreement_rate"),
            ("semantic_miss_rate", 0.2, "semantic_miss_rate"),
            ("n_positives", 12, "n_positives"),
            ("n_negatives", 30, "n_negatives"),
            ("shadow_n_examples", 40, "shadow_examples"),
        ],
    )
    def test_each_criterion_can_refuse_on_its_own(self, key: str, value: Any, needle: str) -> None:
        report = check_promotion(EnforceCriteria(), good_metrics(**{key: value}))
        assert report.ok is False
        assert needle in " ".join(report.failures)

    def test_the_refusal_names_the_failing_metric_and_its_measured_value(self) -> None:
        # The point of refusing loudly: an operator has to be able to read the
        # number that stopped the promotion without re-running anything.
        report = check_promotion(EnforceCriteria(), good_metrics(recall=0.8512))
        message = report.message()
        assert "NOT MET" in message
        assert "recall" in message
        assert "0.8512" in message
        assert ">= 0.99" in message

    def test_a_missing_metric_fails_rather_than_passing(self) -> None:
        metrics = good_metrics()
        del metrics["recall"]
        report = check_promotion(EnforceCriteria(), metrics)
        assert report.ok is False
        assert "not measured" in report.message()

    def test_an_unmeasurable_miss_rate_is_a_refusal_not_a_free_pass(self) -> None:
        # None means "the eval set contained nothing layer 1 catches", which is a
        # gap in the evidence. Treating it as 0.0 would promote a model that has
        # never been compared against the floor.
        report = check_promotion(EnforceCriteria(), good_metrics(semantic_miss_rate=None))
        assert report.ok is False
        assert "unmeasured" in report.message()

    def test_recall_below_the_absolute_floor_is_refused_whatever_else_passes(self) -> None:
        everything_else_perfect = good_metrics(
            recall=0.89,
            false_positive_rate=0.0,
            disagreement_rate=0.0,
            semantic_miss_rate=0.0,
            n_positives=100_000,
            n_negatives=100_000,
            shadow_n_examples=100_000,
        )
        # Even with the operator's own bar lowered to the minimum the code allows,
        # a measured recall of 0.89 is below it and nothing else can compensate.
        report = check_promotion(
            EnforceCriteria(min_recall=EnforceCriteria.MIN_ALLOWED_RECALL), everything_else_perfect
        )
        assert report.ok is False
        assert [c.name for c in report.checks if not c.ok] == ["recall"]

    def test_a_lowered_bar_below_the_floor_cannot_even_be_configured(self) -> None:
        with pytest.raises(SemanticConfigError, match="floor"):
            EnforceCriteria(min_recall=0.5).parse({"min_recall": 0.5})

    def test_the_layer_refuses_enforce_with_the_measured_numbers(self, tmp_path: Path) -> None:
        payload = artifact_payload(metrics=good_metrics(recall=0.734, n_positives=41))
        path = write_artifact(tmp_path, payload, name="weak.json")
        with pytest.raises(SemanticGateError) as excinfo:
            SemanticLayer(SemanticGatePolicy(mode="enforce", artifact=str(path)))
        message = str(excinfo.value)
        assert "refused" in message
        assert "0.734" in message
        assert "41" in message
        assert "NOT MET" in message

    def test_an_artifact_with_no_metrics_cannot_be_promoted(self, tmp_path: Path) -> None:
        payload = artifact_payload()
        payload["metrics"] = {}
        path = write_artifact(tmp_path, payload, name="empty-metrics.json")
        with pytest.raises(SemanticGateError, match="no measured metrics"):
            SemanticLayer(SemanticGatePolicy(mode="enforce", artifact=str(path)))

    def test_the_report_is_serializable_and_carries_no_text(self) -> None:
        report = check_promotion(EnforceCriteria(), good_metrics(recall=0.5))
        payload = json.loads(json.dumps(report.to_dict()))
        assert payload["ok"] is False
        assert payload["criteria"]["min_recall"] == 0.99
        assert payload["metrics"]["recall"] == 0.5
        assert CARD_PROMPT not in json.dumps(payload)


# --------------------------------------------------------------------------- #
# 8. Measuring an eval set: the arithmetic, pinned
# --------------------------------------------------------------------------- #
class KeyScorer:
    """Fires when a marker token is present, so every count below is chosen."""

    name = "keyed"
    model_version = "keyed-1"

    def __init__(self, marker: str = "zzz") -> None:
        self.marker = marker

    def score(self, text: str, *, features: RequestFeatures | None = None) -> float:
        del features
        return 1.0 if self.marker in text else 0.0


#: Six rows whose every count is known before the measurement runs.
#:
#:   row  text                                    label  layer2  layer1
#:   1    "zzz one"                                 +      fire    -
#:   2    "plain two"                               +      quiet   -
#:   3    "zzz card 4111 1111 1111 1111"            +      fire    FIRE
#:   4    "card 4111 1111 1111 1111 only"           +      quiet   FIRE
#:   5    "zzz benign"                              -      fire    -
#:   6    "benign text"                             -      quiet   -
MEASURED_ROWS = (
    LabeledExample("zzz one", True),
    LabeledExample("plain two", True),
    LabeledExample(f"zzz card {CARD_PROMPT.split('card ')[1].split(' and', maxsplit=1)[0]}", True),
    LabeledExample(CARD_PROMPT, True),
    LabeledExample("zzz benign", False),
    LabeledExample("benign text", False),
)


class TestMeasurePromotionMetrics:
    @pytest.fixture
    def metrics(self) -> dict[str, Any]:
        return measure_promotion_metrics(KeyScorer(), MEASURED_ROWS)

    def test_the_counts_are_what_the_table_says(self, metrics: dict[str, Any]) -> None:
        assert metrics["n_examples"] == 6
        assert metrics["n_positives"] == 4
        assert metrics["n_negatives"] == 2
        assert metrics["true_positives"] == 2
        assert metrics["false_positives"] == 1

    def test_recall_is_measured_on_positives_only(self, metrics: dict[str, Any]) -> None:
        assert metrics["recall"] == pytest.approx(0.5)

    def test_the_false_positive_rate_is_measured_on_negatives_only(self, metrics: dict[str, Any]) -> None:
        assert metrics["false_positive_rate"] == pytest.approx(0.5)

    def test_the_disagreement_rate_counts_both_directions(self, metrics: dict[str, Any]) -> None:
        # Rows 1, 4 and 5 contradict layer 1; rows 2, 3 and 6 agree with it.
        assert metrics["disagreement_rate"] == pytest.approx(0.5)
        assert metrics["semantic_only_rate"] == pytest.approx(2 / 6)

    def test_the_miss_rate_is_conditional_on_layer1_firing(self, metrics: dict[str, Any]) -> None:
        # Two rows fire layer 1 and one of them is missed: 1/2, not 1/6.
        assert metrics["n_layer1_fired"] == 2
        assert metrics["semantic_miss_rate"] == pytest.approx(0.5)

    def test_layer1_recall_is_reported_alongside(self, metrics: dict[str, Any]) -> None:
        # Useful context for a promotion review: how much of the positive set the
        # checksums already catch, i.e. how much layer 2 is actually being asked to add.
        assert metrics["layer1_recall"] == pytest.approx(0.5)

    def test_score_summaries_are_present(self, metrics: dict[str, Any]) -> None:
        assert metrics["mean_score_positive"] == pytest.approx(0.5)
        assert metrics["mean_score_negative"] == pytest.approx(0.5)
        assert metrics["max_score_negative"] == pytest.approx(1.0)

    def test_provenance_of_the_measurement_is_recorded(self, metrics: dict[str, Any]) -> None:
        assert metrics["threshold"] == 0.5
        assert metrics["model_version"] == "keyed-1"
        assert metrics["scorer"] == "keyed"
        assert metrics["measured_at"]

    def test_the_threshold_moves_the_recall(self) -> None:
        # A scorer that answers exactly 0.5 is on the boundary: >= fires.
        metrics = measure_promotion_metrics(KeyScorer(), MEASURED_ROWS, threshold=0.5)
        strict = measure_promotion_metrics(KeyScorer(), MEASURED_ROWS, threshold=1.0)
        assert metrics["recall"] == strict["recall"]
        quiet = measure_promotion_metrics(NullScorer(), MEASURED_ROWS, threshold=0.5)
        assert quiet["recall"] == 0.0
        assert quiet["false_positive_rate"] == 0.0

    def test_an_unmeasurable_miss_rate_is_none_not_zero(self) -> None:
        rows = [LabeledExample("zzz one", True), LabeledExample("plain text", False)]
        metrics = measure_promotion_metrics(KeyScorer(), rows)
        assert metrics["n_layer1_fired"] == 0
        assert metrics["semantic_miss_rate"] is None

    def test_an_empty_set_reports_none_rather_than_dividing_by_zero(self) -> None:
        metrics = measure_promotion_metrics(KeyScorer(), [])
        assert metrics["n_examples"] == 0
        assert metrics["recall"] is None
        assert metrics["false_positive_rate"] is None
        assert metrics["disagreement_rate"] is None

    def test_the_report_carries_no_example_text(self) -> None:
        # These numbers get pasted into pull requests and incident channels. A
        # measurement that leaks one prompt leaks a prompt somebody blocked.
        blob = json.dumps(measure_promotion_metrics(KeyScorer(), MEASURED_ROWS), default=str)
        for row in MEASURED_ROWS:
            assert row.text not in blob
        assert "4111" not in blob


# --------------------------------------------------------------------------- #
# 9. Reading a shadow log back: the live half of the evidence
# --------------------------------------------------------------------------- #
def shadow_record(
    *,
    ran: bool = True,
    fired: bool = False,
    layer1_fired: bool = False,
    enforced: bool = False,
    degraded: bool = False,
    score: float = 0.1,
    model_version: str = "shadow-model-1",
) -> dict[str, Any]:
    """One logged decision record, reduced to the parts measure_shadow_log reads."""
    disagreement = None
    if ran and fired != layer1_fired:
        disagreement = DISAGREE_SEMANTIC_ONLY if fired else DISAGREE_SEMANTIC_MISS
    return {
        "kind": "jev_route.decision",
        "semantic": {
            "ran": ran,
            "fired": fired,
            "enforced": enforced,
            "degraded": degraded,
            "score": score,
            "model_version": model_version,
            "layer1": {"fired": layer1_fired},
            "disagreement": disagreement,
        }
        if ran
        else {"ran": False, "model_version": model_version},
    }


#: Eight assessed rows and two the layer never ran on.
#:   fired: 3 of 8. disagreements: 3 (two semantic_only, one semantic_miss).
#:   layer1 fired on 2 rows, and missed-by-layer2 on 1 of those 2.
SHADOW_LOG = (
    shadow_record(),
    shadow_record(fired=True, score=0.9),
    shadow_record(layer1_fired=True),
    shadow_record(fired=True, layer1_fired=True, score=0.8),
    shadow_record(),
    shadow_record(fired=True, score=0.7),
    shadow_record(),
    shadow_record(degraded=True),
    shadow_record(ran=False),
    shadow_record(ran=False),
)


class TestMeasureShadowLog:
    @pytest.fixture
    def metrics(self) -> dict[str, Any]:
        return measure_shadow_log(SHADOW_LOG)

    def test_rows_the_layer_never_ran_on_are_not_observations(self, metrics: dict[str, Any]) -> None:
        assert metrics["shadow_n_examples"] == 8
        assert metrics["shadow_skipped_records"] == 2

    def test_the_fire_rate_is_over_assessed_rows(self, metrics: dict[str, Any]) -> None:
        assert metrics["shadow_fire_rate"] == pytest.approx(3 / 8)

    def test_the_disagreement_rate_is_over_assessed_rows(self, metrics: dict[str, Any]) -> None:
        assert metrics["disagreement_rate"] == pytest.approx(3 / 8)
        assert metrics["semantic_only_rate"] == pytest.approx(2 / 8)

    def test_the_miss_rate_is_conditional_on_layer1(self, metrics: dict[str, Any]) -> None:
        assert metrics["n_layer1_fired"] == 2
        assert metrics["semantic_miss_rate"] == pytest.approx(0.5)

    def test_it_records_the_model_it_observed(self, metrics: dict[str, Any]) -> None:
        assert metrics["shadow_model_version"] == "shadow-model-1"
        assert metrics["shadow_enforced"] == 0
        assert metrics["shadow_degraded"] == 1
        assert metrics["shadow_mean_score"] == pytest.approx((0.1 * 5 + 0.9 + 0.8 + 0.7) / 8)

    def test_a_log_mixing_two_models_is_refused(self) -> None:
        # Averaging two models' disagreement rates produces a number that is not
        # evidence about either, and it is the number that would decide a promotion.
        rows = [*SHADOW_LOG, shadow_record(fired=True, model_version="shadow-model-2")]
        with pytest.raises(SemanticGateError, match="mixes 2 semantic model versions"):
            measure_shadow_log(rows)

    def test_a_log_with_no_observations_is_refused(self) -> None:
        with pytest.raises(SemanticGateError, match="no shadow assessments"):
            measure_shadow_log([shadow_record(ran=False), {"kind": "jev_route.decision"}])

    def test_an_empty_log_is_refused(self) -> None:
        with pytest.raises(SemanticGateError, match="no shadow assessments"):
            measure_shadow_log([])

    def test_the_report_carries_no_content(self, metrics: dict[str, Any]) -> None:
        assert "prompt" not in json.dumps(metrics)


class TestShadowMetricsMerge:
    def test_shadow_numbers_override_the_offline_ones(self) -> None:
        offline = good_metrics(disagreement_rate=0.0)
        shadow = measure_shadow_log(SHADOW_LOG)
        merged = merge_metrics(offline, shadow, model_version="shadow-model-1")
        assert merged["disagreement_rate"] == pytest.approx(3 / 8)
        assert merged["recall"] == offline["recall"]  # offline-only: live traffic has no labels
        assert merged["metrics_sources"] == ["artifact_holdout", "shadow_log"]

    def test_no_shadow_metrics_is_not_an_error(self) -> None:
        assert merge_metrics(good_metrics(), None, model_version="x") == good_metrics()

    def test_numbers_from_a_different_model_are_refused(self) -> None:
        shadow = measure_shadow_log(SHADOW_LOG)
        with pytest.raises(SemanticGateError, match="Refusing to combine"):
            merge_metrics(good_metrics(), shadow, model_version="some-other-model")

    def test_a_file_that_does_not_exist_says_what_to_do(self, tmp_path: Path) -> None:
        with pytest.raises(SemanticGateError, match="measure_shadow_log"):
            load_shadow_metrics(tmp_path / "nope.json")

    def test_a_layer_promotes_on_live_numbers(self, tmp_path: Path) -> None:
        # The artifact's own holdout disagrees too often; the live shadow log does
        # not. Promotion follows the better evidence, and the versions have to match.
        payload = artifact_payload(
            model_version="shadow-model-1", metrics=good_metrics(model_version="shadow-model-1", disagreement_rate=0.4)
        )
        path = write_artifact(tmp_path, payload, name="live.json")
        shadow_path = tmp_path / "shadow-metrics.json"
        shadow_path.write_text(json.dumps(measure_shadow_log(SHADOW_LOG)), encoding="utf-8")

        strict = SemanticGatePolicy(
            mode="enforce", artifact=str(path), enforce_requires=EnforceCriteria(max_disagreement_rate=0.05)
        )
        with pytest.raises(SemanticGateError, match="disagreement_rate"):
            SemanticLayer(strict)

        promoted = SemanticLayer(
            SemanticGatePolicy(
                mode="enforce",
                artifact=str(path),
                shadow_metrics=str(shadow_path),
                # Sized to this fixture's eight-row log; the point is which
                # disagreement number the check read, not the bars themselves.
                enforce_requires=EnforceCriteria(
                    max_disagreement_rate=0.4, max_semantic_miss_rate=0.6, min_shadow_examples=5
                ),
            )
        )
        assert promoted.active is True
        assert promoted.metrics["metrics_sources"] == ["artifact_holdout", "shadow_log"]

    def test_a_shadow_file_from_another_model_refuses_the_promotion(self, tmp_path: Path) -> None:
        path = write_artifact(tmp_path, artifact_payload(), name="a.json")
        shadow_path = tmp_path / "shadow.json"
        shadow_path.write_text(json.dumps(measure_shadow_log(SHADOW_LOG)), encoding="utf-8")
        with pytest.raises(SemanticGateError, match="Refusing to combine"):
            SemanticLayer(SemanticGatePolicy(mode="enforce", artifact=str(path), shadow_metrics=str(shadow_path)))


# --------------------------------------------------------------------------- #
# 10. The reference trainer
# --------------------------------------------------------------------------- #
CONTEXT_TEMPLATES = (
    "the employee {who} was placed on administrative leave pending the disciplinary hearing {ref}",
    "the performance improvement plan for {who} documents repeated incidents raised by the manager {ref}",
    "clinical note {ref}: the patient {who} denies fever but describes worsening symptoms",
    "counsel advised that memorandum {ref} is privileged and prepared in anticipation of litigation",
    "the acquisition target {who} has not been announced and the terms stay between the parties {ref}",
    "the whistleblower complaint {ref} alleges the finance director approved the payment to {who}",
    "the settlement discussion {ref} is confidential and must not be shared outside the deal team",
    "the internal investigation {ref} interviewed {who} about the alleged harassment complaint",
)
ROUTINE_TEMPLATES = (
    "summarize the roadmap {ref} for the team and list the open questions for {who}",
    "write a python function {ref} that flattens a nested list of dictionaries",
    "explain how kubernetes schedules pods {ref} across the nodes of a cluster",
    "draft a blog post {ref} about the new release and its headline features",
    "what is the time complexity {ref} of mergesort and why is it a stable sort",
    "review pull request {ref} and suggest clearer names for the helpers in {who}",
    "translate paragraph {ref} from english into german and keep the tone",
    "benchmark the json parser {ref} against the csv parser and chart the result",
)


def make_examples(templates: tuple[str, ...], count: int, *, seed: int = 11) -> list[str]:
    """Template-filled sentences, so the holdout holds *unseen* instances."""
    import random

    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < count:
        out.append(rng.choice(templates).format(who=f"person{rng.randrange(40)}", ref=f"case{rng.randrange(50)}"))
    return sorted(set(out))[:count]


class TestFitScorer:
    @pytest.fixture(scope="class")
    @classmethod
    def trained(cls) -> SemanticArtifact:
        # classmethod, not an instance method: a class-scoped instance fixture
        # is deprecated (and errors in pytest 10) even when it sets no state.
        return fit_scorer(
            make_examples(CONTEXT_TEMPLATES, 240),
            make_examples(ROUTINE_TEMPLATES, 320),
            epochs=120,
            learning_rate=2.0,
            min_df=3,
        )

    def test_it_learns_the_distinction_on_unseen_text(self, trained: SemanticArtifact) -> None:
        # Recall and FPR are both measured on the holdout, i.e. on sentences the
        # trainer never saw. A number computed on the training rows would only
        # prove the model can memorise.
        assert trained.metrics["recall"] >= 0.95
        assert trained.metrics["false_positive_rate"] <= 0.05

    def test_the_metrics_come_from_the_holdout_only(self, trained: SemanticArtifact) -> None:
        training = trained.metadata["training"]
        assert trained.metrics["n_examples"] == training["holdout_rows"]
        assert training["holdout_rows"] < training["train_rows"] + training["holdout_rows"]
        assert trained.metrics["n_examples"] < training["positive_examples"] + training["negative_examples"]

    def test_the_artifact_it_writes_is_one_it_can_load(self, trained: SemanticArtifact, tmp_path: Path) -> None:
        path = trained.save(tmp_path / "fitted")
        loaded = SemanticArtifact.load(path)
        assert loaded.model_version == trained.model_version
        assert loaded.metrics == trained.metrics

    def test_it_declares_its_own_provenance(self, trained: SemanticArtifact) -> None:
        # The trainer is the first place the bootstrap paradox could be violated,
        # so it is the first place that has to record not violating it.
        assert trained.provenance.trained_on_blocked_content is False
        assert trained.provenance.positives == "synthetic"
        assert trained.provenance.negatives == "production"

    def test_the_trained_scorer_separates_the_two_kinds_of_prompt(self, trained: SemanticArtifact) -> None:
        contextual = make_examples(CONTEXT_TEMPLATES, 240, seed=99)[0]
        routine = make_examples(ROUTINE_TEMPLATES, 320, seed=99)[0]
        assert trained.scorer.score(contextual) > 0.5
        assert trained.scorer.score(routine) < 0.5

    def test_training_is_deterministic(self) -> None:
        positives = make_examples(CONTEXT_TEMPLATES, 40)
        negatives = make_examples(ROUTINE_TEMPLATES, 40)
        first = fit_scorer(positives, negatives, epochs=20)
        second = fit_scorer(positives, negatives, epochs=20)
        assert first.scorer.weights == second.scorer.weights
        assert first.model_version == second.model_version

    def test_it_refuses_to_train_on_nothing(self) -> None:
        with pytest.raises(SemanticGateError, match="positive"):
            fit_scorer([], ["ordinary prompt text"])
        with pytest.raises(SemanticGateError, match="negative"):
            fit_scorer(["disciplinary hearing"], [])

    def test_it_refuses_contradictory_labels(self) -> None:
        both = "the disciplinary hearing was scheduled for monday"
        with pytest.raises(SemanticGateError, match="both positive and negative"):
            fit_scorer([both], [both, "an ordinary prompt"])

    def test_it_refuses_a_split_that_cannot_measure_anything(self) -> None:
        with pytest.raises(SemanticGateError, match="holdout"):
            fit_scorer(["disciplinary hearing"], ["ordinary prompt"], holdout_fraction=0.0)

    def test_it_refuses_an_unassertable_level(self) -> None:
        with pytest.raises(SemanticGateError, match="level"):
            fit_scorer(["disciplinary hearing"], ["ordinary prompt"], level="public")

    def test_duplicate_examples_are_collapsed(self) -> None:
        artifact = fit_scorer(["disciplinary hearing"] * 25, ["ordinary prompt text"] * 25, epochs=5)
        training = artifact.metadata["training"]
        assert training["positive_examples"] == 1
        assert training["negative_examples"] == 1


# --------------------------------------------------------------------------- #
# 11. Through the router: shadow measures, enforce acts, blocked content stays
# --------------------------------------------------------------------------- #
class TestTheRouterIntegration:
    def policy_doc(self, log_path: Path | None = None, **gate: Any) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "version": 1,
            "backend": {"name": "mock"},
            "tiers": {"local": ["local-model"], "cheap": ["cheap-model"], "strong": ["strong-model"]},
            "tier_order": ["cheap", "strong", "local"],
            "gate": {"on_force_local": "skip_backend", **gate},
            "cache": {"enabled": False},
            "logging": {"enabled": True, "path": str(log_path or "./unused.jsonl"), "excerpt_mode": "hash"},
            "rules": [
                {"id": "gate.force-local", "if": "gate_force_local", "then": {"tier": "local"}},
                {
                    "id": "data.sensitive",
                    "if": 'sensitivity in ["confidential", "regulated"] or pii_present',
                    "then": {"tier": "local"},
                },
                {"id": "default", "then": {"tier": "cheap"}},
            ],
        }
        return doc

    def router(self, tmp_path: Path, layer: SemanticLayer | None, *, blocked_sink: Any = None, **gate: Any) -> Any:
        from jev_route import MockBackend, Router
        from jev_route.cache import NullCache
        from jev_route.logging_sink import CallbackSink

        policy = Policy.from_dict(self.policy_doc(tmp_path / "decisions.jsonl", **gate))
        return Router(
            policy,
            MockBackend(),
            cache=NullCache(),
            sink=CallbackSink(lambda record: None),
            blocked_sink=blocked_sink if blocked_sink is not None else CallbackSink(lambda record: None),
            semantic=layer,
        )

    async def test_shadow_never_changes_the_served_tier(self, tmp_path: Path) -> None:
        # A layer that scores every prompt at 0.99 and asserts "regulated" -- the
        # most aggressive shadow imaginable -- must still serve the same tier as no
        # layer at all. If this fails, shadow is not a measurement.
        from jev_route import MockBackend, Router
        from jev_route.cache import NullCache
        from jev_route.logging_sink import CallbackSink

        policy = Policy.from_dict(self.policy_doc(tmp_path / "decisions.jsonl"))
        plain_records: list[Any] = []
        shadow_records: list[Any] = []
        plain = Router(policy, MockBackend(), cache=NullCache(), sink=CallbackSink(plain_records.append),
                       semantic=SemanticLayer(SemanticGatePolicy(mode="off")))
        shadow = Router(
            policy,
            MockBackend(),
            cache=NullCache(),
            sink=CallbackSink(shadow_records.append),
            semantic=SemanticLayer(
                SemanticGatePolicy(mode="shadow", level="regulated"), scorer=StubScorer(0.99)
            ),
        )
        prompts = [BENIGN_PROMPT, "Explain how HIPAA works and who it applies to.", CONTEXTUAL_PROMPT]
        for prompt in prompts:
            await plain.route_text(prompt)
            await shadow.route_text(prompt)

        for off, on in zip(plain_records, shadow_records, strict=True):
            assert off.decision.tier == on.decision.tier
            assert off.decision.rule_id == on.decision.rule_id
            assert off.decision.effective_sensitivity == on.decision.effective_sensitivity
            assert off.decision.backend == on.decision.backend
            assert on.decision.gate == off.decision.gate  # layer 1 is untouched by layer 2

        # ...but the disagreement was still recorded, which is the whole point.
        assert [r.semantic["fired"] for r in shadow_records] == [True, True, True]
        assert all(r.semantic["enforced"] is False for r in shadow_records)
        assert shadow_records[0].semantic["disagreement"] == DISAGREE_SEMANTIC_ONLY

    async def test_enforce_raises_the_floor_and_the_tier_with_it(self, tmp_path: Path) -> None:
        records: list[Any] = []
        layer = SemanticLayer(
            SemanticGatePolicy(mode="enforce", level="confidential"),
            scorer=StubScorer(0.97),
            metrics=good_metrics(),
        )
        from jev_route import MockBackend, Router
        from jev_route.cache import NullCache
        from jev_route.logging_sink import CallbackSink

        policy = Policy.from_dict(self.policy_doc(tmp_path / "decisions.jsonl"))
        router = Router(policy, MockBackend(), cache=NullCache(), sink=CallbackSink(records.append), semantic=layer)
        decision = await router.route_text(BENIGN_PROMPT)
        assert decision.tier == "local"
        assert decision.effective_sensitivity == "confidential"
        assert decision.rule_id == "data.sensitive"
        assert any("semantic gate layer 2" in note for note in decision.escalated), decision.escalated
        assert records[0].semantic["enforced"] is True

    async def test_a_semantic_public_cannot_cancel_a_deterministic_regulated(self, tmp_path: Path) -> None:
        # The invariant, end to end: layer 2 is certain the prompt is harmless,
        # layer 1 found a Luhn-valid card number. The card number wins, the request
        # stays local, and no backend is called.
        from jev_route import MockBackend, Router
        from jev_route.cache import NullCache
        from jev_route.logging_sink import CallbackSink

        records: list[Any] = []
        layer = SemanticLayer(
            SemanticGatePolicy(mode="enforce", level="confidential"),
            scorer=StubScorer(0.0),  # "public", with total confidence
            metrics=good_metrics(),
        )
        policy = Policy.from_dict(self.policy_doc(tmp_path / "decisions.jsonl"))
        router = Router(policy, MockBackend(), cache=NullCache(), sink=CallbackSink(records.append), semantic=layer)
        decision = await router.route_text(CARD_PROMPT)
        assert decision.gate.sensitivity_floor == "regulated"
        assert decision.effective_sensitivity == "regulated"
        assert decision.tier == "local"
        assert decision.backend == "gate"  # never reached a backend at all
        assert records[0].semantic["disagreement"] == DISAGREE_SEMANTIC_MISS
        assert records[0].semantic["enforced"] is False

    async def test_rules_cannot_read_a_shadow_layer(self, tmp_path: Path) -> None:
        # The policy engine sees the actionable projection, so a rule written
        # against layer 2 is inert until the layer is promoted. `semantic_mode`
        # still tells the truth, which is how a policy can tell the cases apart.
        from jev_route import MockBackend, Router
        from jev_route.cache import NullCache
        from jev_route.logging_sink import CallbackSink

        doc = self.policy_doc(tmp_path / "decisions.jsonl")
        doc["rules"].insert(
            0, {"id": "semantic.rule", "if": "semantic_fired", "then": {"tier": "strong"}, "reason": "test rule"}
        )
        policy = Policy.from_dict(doc)
        layer = SemanticLayer(SemanticGatePolicy(mode="shadow"), scorer=StubScorer(0.99))
        router = Router(policy, MockBackend(), cache=NullCache(), sink=CallbackSink(lambda r: None), semantic=layer)
        assert (await router.route_text(BENIGN_PROMPT)).rule_id == "default"
        assert router.semantic.stats()["mode"] == "shadow"

    async def test_rules_can_read_an_enforcing_layer(self, tmp_path: Path) -> None:
        from jev_route import MockBackend, Router
        from jev_route.cache import NullCache
        from jev_route.logging_sink import CallbackSink

        doc = self.policy_doc(tmp_path / "decisions.jsonl")
        doc["rules"].insert(
            0, {"id": "semantic.rule", "if": "semantic_fired", "then": {"tier": "strong"}, "reason": "test rule"}
        )
        policy = Policy.from_dict(doc)
        layer = SemanticLayer(
            SemanticGatePolicy(mode="enforce"), scorer=StubScorer(0.99), metrics=good_metrics()
        )
        router = Router(policy, MockBackend(), cache=NullCache(), sink=CallbackSink(lambda r: None), semantic=layer)
        decision = await router.route_text(BENIGN_PROMPT)
        assert decision.rule_id == "semantic.rule"

    async def test_a_broken_layer_does_not_break_the_request(self, tmp_path: Path) -> None:
        from jev_route import MockBackend, Router
        from jev_route.cache import NullCache
        from jev_route.logging_sink import CallbackSink

        layer = SemanticLayer(SemanticGatePolicy(mode="enforce"), scorer=ExplodingScorer(), metrics=good_metrics())
        router = Router(
            Policy.from_dict(self.policy_doc(tmp_path / "decisions.jsonl")),
            MockBackend(),
            cache=NullCache(),
            sink=CallbackSink(lambda r: None),
            semantic=layer,
        )
        decision = await router.route_text(BENIGN_PROMPT)
        assert decision.tier == "cheap"
        assert layer.stats()["errors"] == 1

    async def test_an_unpromoted_enforce_config_refuses_to_start(self, tmp_path: Path) -> None:
        from jev_route import MockBackend, Router
        from jev_route.cache import NullCache

        path = write_artifact(tmp_path, artifact_payload(metrics=good_metrics(recall=0.42)), name="weak.json")
        doc = self.policy_doc(tmp_path / "decisions.jsonl", semantic={"mode": "enforce", "artifact": str(path)})
        policy = Policy.from_dict(doc)
        assert policy.semantic_gate.mode == "enforce"  # the flag parsed fine...
        with pytest.raises(SemanticGateError, match=r"0\.42"):  # ...and was still refused
            Router(policy, MockBackend(), cache=NullCache())
