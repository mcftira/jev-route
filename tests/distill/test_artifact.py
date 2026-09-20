"""The distilled artifact format: what it is made of, and what refuses it.

The graduation docs promise a security reviewer that a loaded artifact cannot be
a pickle, cannot be tampered with silently, cannot outlive a schema change, and
is reproducible. This file pins those promises at the artifact layer, where the
lifecycle CLI tests only reach them indirectly:

* **No pickle, anywhere.** The directory is JSON plus raw ``.npy`` weight files;
  the loader never passes ``allow_pickle=True``.
* **Loading validates.** A flipped weight byte fails the checksum; a relabelled
  head fails the live-schema ladder check; an unknown schema version refuses to
  load. Each refusal names what it found.
* **Deterministic.** The same dataset and seed produce the same weight bytes.
* **The split is stable.** A row that was on the train side before the log grew
  is still on the train side after, because the split is keyed on ``request_id``.
* **Gate-blocked rows are masked per head.** The heads a blocked request never
  answered are unusable on that row; the heads it did answer (sensitivity, pii)
  train on it.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import shutil
from collections.abc import Sequence
from pathlib import Path

import pytest
from tests.distill.synthlog import build_log

from jev_route.cli import main


def _require_numpy() -> None:
    try:
        import numpy  # noqa: F401
    except ImportError:  # pragma: no cover - depends on the environment
        pytest.skip("the distill extra is not installed: pip install 'jev-route[distill]'")


def run(argv: Sequence[str]) -> tuple[int, str, str]:
    """Run the CLI in-process and capture both streams (see test_distill_cli.run)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(argv))
    return int(code), out.getvalue(), err.getvalue()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strip_seconds(value):
    """Drop wall-clock fields from a parsed artifact envelope, recursively.

    The trainer records per-epoch and total ``*_seconds`` timings; they are
    properties of this machine, not of the model, so they are stripped before
    the content comparison.
    """
    if isinstance(value, dict):
        return {
            k: _strip_seconds(v) for k, v in value.items() if not (k == "seconds" or k.endswith("_seconds"))
        }
    if isinstance(value, list):
        return [_strip_seconds(v) for v in value]
    return value


@pytest.fixture(scope="module")
def log_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    _require_numpy()
    target = tmp_path_factory.mktemp("artifact-logs") / "decisions.jsonl"
    # Two passes over the full corpus: the 12 blocking prompts are sparse in the
    # 158-prompt corpus, and the gate-masking test below needs them in the log.
    stats = build_log(target, seed=7, excerpt_mode="hash", repeats=2)
    assert stats.records == 2 * 158
    return target


@pytest.fixture(scope="module")
def dataset_path(tmp_path_factory: pytest.TempPathFactory, log_path: Path) -> Path:
    work = tmp_path_factory.mktemp("artifact-ds")
    code, out, err = run(["export", "--log", str(log_path), "--out", str(work / "ds")])
    assert code == 0, out + err
    return work / "ds" / "dataset.jsonl"


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory, dataset_path: Path) -> Path:
    work = tmp_path_factory.mktemp("artifact-model")
    code, out, err = run(["train", "--data", str(dataset_path.parent), "--out", str(work / "v1")])
    assert code == 0, out + err
    return work / "v1"


@pytest.fixture(scope="module")
def packaged_dir(tmp_path_factory: pytest.TempPathFactory, model_dir: Path) -> Path:
    work = tmp_path_factory.mktemp("artifact-pkg")
    code, out, err = run(["package", "--artifact", str(model_dir), "--out", str(work / "v1-pkg")])
    assert code == 0, out + err
    return work / "v1-pkg"


class TestTheFileFormat:
    """What the directory is made of, and what it is not."""

    def test_the_packaged_directory_is_exactly_the_six_known_files(self, packaged_dir: Path) -> None:
        names = {p.name for p in packaged_dir.iterdir()}
        assert names == {"artifact.json", "W.npy", "b.npy", "vectorizer.json", "metrics.json",
                         "dataset.json"}

    def test_no_pickle_anywhere_in_the_artifact(self, packaged_dir: Path) -> None:
        for p in packaged_dir.iterdir():
            assert not p.name.endswith(".pkl"), p.name
            if p.suffix == ".json":
                assert b"\x80" not in p.read_bytes()  # pickle protocol 2 magic
        for name in ("W.npy", "b.npy"):
            assert packaged_dir.joinpath(name).read_bytes()[:6] == b"\x93NUMPY"

    def test_the_weights_load_with_allow_pickle_false(self, packaged_dir: Path) -> None:
        import numpy as np

        # load_artifact itself uses allow_pickle=False; this pins the files
        # directly so a regression shows up as a numpy error, not a checksum
        # mismatch on a file numpy refuses to open.
        for name in ("W.npy", "b.npy"):
            with open(packaged_dir / name, "rb") as fh:
                arr = np.load(fh, allow_pickle=False)
            assert arr.dtype == np.float32


class TestLoadingValidates:
    """A loaded artifact is a trusted input. Tampering must be loud."""

    @staticmethod
    def _copy(packaged_dir: Path, tmp_path: Path, tag: str) -> Path:
        dst = tmp_path / tag
        shutil.copytree(packaged_dir, dst)
        return dst

    def test_load_round_trips_the_head_layout_against_the_live_schema(
        self, packaged_dir: Path
    ) -> None:
        from jev_route.distill.artifact import load_artifact
        from jev_route.schema import COMPLEXITY_LEVELS, DOMAINS, SENSITIVITY_LEVELS

        art = load_artifact(packaged_dir)
        ladders = {
            "complexity": tuple(COMPLEXITY_LEVELS),
            "sensitivity": tuple(SENSITIVITY_LEVELS),
            "domain": tuple(DOMAINS),
        }
        for head in art.model.heads:
            if head.name in ladders:
                assert tuple(head.labels) == ladders[head.name], head.name

    def test_a_tampered_weight_fails_the_checksum_on_load(
        self, packaged_dir: Path, tmp_path: Path
    ) -> None:
        from jev_route.distill.artifact import ArtifactError, load_artifact

        dst = self._copy(packaged_dir, tmp_path, "tampered")
        w = dst / "W.npy"
        blob = bytearray(w.read_bytes())
        blob[-1] ^= 0xFF
        w.write_bytes(bytes(blob))
        with pytest.raises(ArtifactError) as excinfo:
            load_artifact(dst)
        assert "checksum mismatch for W.npy" in str(excinfo.value)

    def test_a_relabelled_head_fails_the_ladder_check(
        self, packaged_dir: Path, tmp_path: Path
    ) -> None:
        from jev_route.distill.artifact import ArtifactError, load_artifact

        dst = self._copy(packaged_dir, tmp_path, "relabelled")
        meta = json.loads((dst / "artifact.json").read_text(encoding="utf-8"))
        heads = meta["model"]["heads"]
        # sensitivity is a head the ladders check against; off-ladder labels are
        # exactly what a mis-mapped training run would produce.
        sens = next(h for h in heads if h["name"] == "sensitivity")
        sens["labels"] = [*list(sens["labels"])[:2], "not-a-real-level"]
        (dst / "artifact.json").write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ArtifactError) as excinfo:
            load_artifact(dst)
        assert "does not match jev_route.schema" in str(excinfo.value)

    def test_an_unsupported_schema_version_refuses_to_load(
        self, packaged_dir: Path, tmp_path: Path
    ) -> None:
        from jev_route.distill.artifact import ArtifactError, load_artifact

        dst = self._copy(packaged_dir, tmp_path, "wrong-version")
        meta = json.loads((dst / "artifact.json").read_text(encoding="utf-8"))
        meta["artifact_schema_version"] = "99"
        (dst / "artifact.json").write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ArtifactError) as excinfo:
            load_artifact(dst)
        assert "is not supported by this build" in str(excinfo.value)


class TestDeterminism:
    """Same data, same seed, same bytes. That is what makes the checksums
    meaningful and the artifact shareable."""

    def test_training_is_byte_reproducible(self, dataset_path: Path, tmp_path: Path) -> None:
        work = tmp_path / "repro"
        work.mkdir()
        for tag in ("a", "b"):
            code, out, err = run(
                ["train", "--data", str(dataset_path.parent), "--out", str(work / tag),
                 "--seed", "0"]
            )
            assert code == 0, out + err
        for name in ("W.npy", "b.npy", "vectorizer.json"):
            a = _sha256(work / "a" / name)
            b = _sha256(work / "b" / name)
            assert a == b, f"{name} differs between two runs: {a} != {b}"
        # artifact.json is compared by content, with the per-epoch wall-clock
        # ``seconds`` fields stripped: the losses and configurations must be
        # identical, the timing of this machine is not part of the model.
        meta_a = json.loads((work / "a" / "artifact.json").read_text(encoding="utf-8"))
        meta_b = json.loads((work / "b" / "artifact.json").read_text(encoding="utf-8"))
        assert _strip_seconds(meta_a) == _strip_seconds(meta_b)


class TestTheSplit:
    """The holdout must mean the same thing as the log grows, or the
    graduation's held-out agreement is measuring leakage."""

    @staticmethod
    def _sides(out: Path) -> dict[str, str]:
        with open(out, encoding="utf-8") as fh:
            return {json.loads(line)["request_id"]: json.loads(line)["split"] for line in fh}

    def test_the_split_is_stable_as_the_log_grows(
        self, log_path: Path, dataset_path: Path, tmp_path: Path
    ) -> None:
        grown = tmp_path / "grown.jsonl"
        grown.write_bytes(log_path.read_bytes())
        records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        with open(grown, "a", encoding="utf-8") as fh:
            for i in range(20):
                rec = dict(records[i % len(records)])
                rec["request_id"] = f"synthetic-grown-{i:03d}"
                rec["excerpt_hash"] = f"{i:016x}"
                fh.write(json.dumps(rec) + chr(10))

        out_a = tmp_path / "split-a"
        out_b = tmp_path / "split-b"
        code, out, err = run(["export", "--log", str(log_path), "--out", str(out_a)])
        assert code == 0, out + err
        code, out, err = run(["export", "--log", str(grown), "--out", str(out_b)])
        assert code == 0, out + err

        sides_a = self._sides(out_a / "dataset.jsonl")
        sides_b = self._sides(out_b / "dataset.jsonl")
        assert len(sides_b) == len(sides_a) + 20
        moved = [rid for rid in sides_a if rid in sides_b and sides_a[rid] != sides_b[rid]]
        assert moved == [], f"rows changed side as the log grew: {moved[:5]}"
        # and the holdout is actually a holdout, not an empty name
        assert "holdout" in sides_a.values()
        assert "train" in sides_a.values()


class TestGateBlockedRows:
    """The rows the gate never let anywhere carry the highest-confidence
    labels the pipeline has -- but only on the heads they answered."""

    def test_gate_blocked_rows_mask_the_unusable_heads(self, dataset_path: Path) -> None:
        rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]
        blocked = [r for r in rows if "gate-blocked" in r["tags"]]
        assert blocked, "the fixture corpus must include gate-blocked rows"
        for row in blocked:
            targets = row["targets"]
            # sensitivity and pii were set by the gate itself, at p=1.00
            assert targets["sensitivity"]["usable"] is True
            assert targets["pii"]["usable"] is True
            # the backend never answered this request
            assert targets["complexity"]["usable"] is False
            assert targets["domain"]["usable"] is False
        # the sidecar counts them
        sidecar = json.loads(dataset_path.with_name("dataset.stats.json").read_text(encoding="utf-8"))
        assert sidecar["stats"]["gate_blocked_rows"] == len(blocked)
        # every row the backend never answered (blocked, or forced local under
        # the default on_force_local) is masked the same way
        gate_answered = [r for r in rows if r["teacher"].get("backend") == "gate"]
        assert gate_answered, "the fixture must include gate-only rows"
        for row in gate_answered:
            assert row["targets"]["complexity"]["usable"] is False
            assert row["targets"]["domain"]["usable"] is False
            assert row["targets"]["sensitivity"]["usable"] is True
        # and a row the gate never touched: every head is usable, unless the
        # teacher answered uniform -- the one other shape of "no signal", which
        # the export marks unusable rather than training on noise
        clean = [r for r in rows if "gate-fired" not in r["tags"]]
        assert clean, "the fixture must also include unblocked rows"
        for row in clean:
            for head in ("complexity", "sensitivity", "domain", "pii"):
                target = row["targets"][head]
                if not target["usable"]:
                    probs = target["probs"]
                    even = 1.0 / len(probs)
                    assert all(abs(v - even) <= 1e-9 for v in probs.values()), row["request_id"]
                    assert target["confidence"] == 0.0, row["request_id"]
