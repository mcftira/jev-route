"""Tests for the numpy trainer's refusal to silently drop ``TrainConfig.encoder``.

WHY THIS FILE EXISTS
--------------------
``TrainConfig.encoder`` names a transformer student. This project's artifact
format cannot hold one: it stores raw weight arrays plus a vectorizer config, so
a fine-tuned encoder would need a tokenizer and hundreds of megabytes that the
format has no slot for. Both trainers therefore have to refuse the request, and
until now only the torch trainer did
(:func:`jev_route.distill.train._require_torch`). The numpy trainer -- the
default -- accepted ``encoder`` and ignored it: it trained the small student and
then sealed the *config* into the artifact through
:meth:`jev_route.distill.train.TrainConfig.to_dict`, so a checksummed,
``graduate``-gated file recorded ``training.encoder = 'answerai-engbert-small-v1'``
while containing no transformer at all. An artifact lying about its own
provenance is the one failure this format exists to prevent.

:func:`jev_route.distill.train.train` now raises
:class:`~jev_route.distill.train.TrainError` for ``encoder`` plus the numpy
framework, before any training happens. These tests pin:

* the raise, and that its message names the value, the *reason* (provenance, not
  serving cost) and both ways out of it;
* every spelling of "the numpy framework" that reaches the refusal, because
  ``train()`` normalises ``None``, ``""`` and case to ``numpy`` and the answer
  must not depend on how the operator wrote it;
* that the torch refusal kept its own wording: the two reasons differ, so merging
  the messages would lose the one thing an operator needs to decide what to do;
* that an unknown framework is still reported as an unknown framework, so the new
  refusal cannot mask a typo;
* that the refusal happens before training, not after it;
* the positive case: an artifact trained by the numpy trainer records
  ``encoder: None`` and holds the student it claims to hold.

Offline and deterministic. The dataset is built by :mod:`synthlog`, the synthetic
decision-log fixture in this directory, with a fixed seed, and exported in
features mode. Nothing here needs torch -- the torch case is refused before the
import, exactly as it is in production.
"""

from __future__ import annotations

import pytest
import synthlog

from jev_route.distill.artifact import ARTIFACT_KIND, ARTIFACT_SCHEMA_VERSION, DistilledArtifact
from jev_route.distill.export import Dataset, export_dataset, load_dataset
from jev_route.distill.train import TrainConfig, TrainError, train, train_torch

#: The encoder name an operator would reach for first: the small ModernBERT-class
#: model this project's docs mention as the hypothetical transformer student.
ENCODER = "answerai-engbert-small-v1"

pytestmark = pytest.mark.distill


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    """A real exported dataset: routed through the production log writer, not hand-waved.

    Features mode on purpose. Text mode would need ``excerpt_mode="redacted"`` and
    a vocabulary build, and none of that is what these tests are about; the refusal
    fires on the config, not on the rows.
    """
    work = tmp_path_factory.mktemp("encoder-refusal")
    log = work / "decisions.jsonl"
    synthlog.build_log(log, n=120, excerpt_mode="hash", seed=11)
    exported = export_dataset(log, work / "ds", mode="features", holdout_fraction=0.25)
    return load_dataset(exported.rows_path)


def test_the_numpy_trainer_refuses_an_encoder_student(dataset: Dataset) -> None:
    """The default trainer must not accept a student it cannot represent."""
    with pytest.raises(TrainError) as excinfo:
        train(dataset, TrainConfig(mode="features", epochs=1, encoder=ENCODER, framework="numpy"))

    message = str(excinfo.value)
    # Names the value: an operator who set it by accident has to see which setting fired.
    assert ENCODER in message
    # Names the real reason. "Not supported" alone invites "then ignore it quietly";
    # the sealed config is why quiet ignoring is a defect rather than a courtesy.
    assert "metadata['training']" in message
    assert "does not contain" in message
    # And both ways out, matching the torch refusal's shape.
    assert "unset `encoder`" in message.lower()
    assert "decisionbackend" in message.lower()


@pytest.mark.parametrize("framework", ["numpy", "NumPy", "NUMPY", None, ""])
def test_the_refusal_covers_every_way_of_saying_numpy(dataset: Dataset, framework: str | None) -> None:
    """``train()`` normalises the framework, so the refusal has to see the normalised value.

    ``None`` and ``""`` are the default trainer: a config loaded from JSON that
    omits ``framework`` lands here, and that is exactly the case where a silently
    ignored ``encoder`` would go unnoticed.
    """
    with pytest.raises(TrainError) as excinfo:
        train(dataset, TrainConfig(mode="features", epochs=1, encoder=ENCODER, framework=framework))
    assert "numpy trainer" in str(excinfo.value)


def test_the_torch_refusal_kept_its_own_wording(dataset: Dataset) -> None:
    """Two refusals, two reasons. The torch one is about serving cost, not provenance."""
    with pytest.raises(TrainError) as excinfo:
        train(dataset, TrainConfig(mode="features", epochs=1, encoder=ENCODER, framework="torch"))

    message = str(excinfo.value)
    assert "is not supported by the distillation artifact format" in message
    assert "request path" in message
    # The numpy wording must not have leaked in: if it had, the two messages were
    # merged and the operator lost the distinction.
    assert "cannot be trained by the numpy trainer" not in message
    assert "metadata['training']" not in message


def test_an_unknown_framework_is_still_reported_as_unknown(dataset: Dataset) -> None:
    """The framework check runs first, so a typo is not reported as an encoder problem."""
    with pytest.raises(TrainError) as excinfo:
        train(dataset, TrainConfig(mode="features", epochs=1, encoder=ENCODER, framework="jax"))

    message = str(excinfo.value)
    assert "unknown training framework" in message
    assert "'jax'" in message
    assert ENCODER not in message


def test_the_refusal_happens_before_training_rather_than_after(dataset: Dataset) -> None:
    """Failing fast is part of the contract: no epochs burned, no progress reported."""
    epochs_run: list[tuple[int, float, float]] = []

    def progress(epoch: int, train_loss: float, val_loss: float) -> None:
        epochs_run.append((epoch, train_loss, val_loss))

    with pytest.raises(TrainError):
        train(
            dataset,
            TrainConfig(mode="features", epochs=3, encoder=ENCODER, framework="numpy"),
            progress=progress,
        )
    assert epochs_run == []


def test_both_public_trainers_refuse_and_neither_can_be_bypassed(dataset: Dataset) -> None:
    """``train()`` and ``train_torch()`` are the two public doors; the loops behind them are not.

    ``_train_numpy`` trusts its caller -- that is normal for a private helper, and
    it is why the guard has to sit in the public entry point rather than in the
    loop. This pins that no *public* name reaches the numpy loop without passing
    one of the two refusals, so a future export cannot quietly reopen the hole.
    """
    with pytest.raises(TrainError) as excinfo:
        train_torch(dataset, TrainConfig(mode="features", epochs=1, encoder=ENCODER))
    assert "is not supported by the distillation artifact format" in str(excinfo.value)

    import jev_route.distill.train as train_module

    public = sorted(train_module.__all__)
    assert "train" in public
    assert "train_torch" in public
    assert not [name for name in public if "numpy" in name.lower() or name.startswith("_")]


def test_a_packaged_numpy_artifact_records_no_encoder(dataset: Dataset, tmp_path) -> None:
    """The positive case: what the numpy trainer seals is what the numpy trainer trained.

    This is the assertion the bug violated. Before the refusal existed, the same
    packaging call with ``encoder=ENCODER`` produced a loadable artifact whose
    ``training_config`` named a transformer while ``model`` held a small student.
    """
    student = train(dataset, TrainConfig(mode="features", epochs=3, seed=7))
    assert student.config.encoder is None

    artifact = DistilledArtifact(
        model=student.model,
        vectorizer=student.vectorizer,
        metadata={
            "kind": ARTIFACT_KIND,
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "mode": student.mode,
            "model_version": "encoder-refusal-test",
            "model": student.model.to_dict(),
            "training": student.config.to_dict(),
            "teacher": {"model_versions": list(student.teacher_model_versions)},
            "contains_prompt_text": student.mode == "text",
        },
    )
    out = tmp_path / "artifact"
    artifact.save(out)
    reloaded = DistilledArtifact.load(out)

    # The recorded provenance makes no transformer claim...
    assert reloaded.training_config.get("encoder") is None
    assert reloaded.training_config.get("framework") == "numpy"
    # ...and the weights it ships are the ones that were trained. Without this
    # second half the first assertion could pass on an empty artifact.
    assert reloaded.model.n_parameters == student.n_parameters
    assert reloaded.model.is_linear == student.model.is_linear
    assert reloaded.n_features == student.model.n_features
