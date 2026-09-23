"""The five graduation verbs: ``export``, ``train``, ``evaluate``, ``package``, ``graduate``.

Every function here is a *thin adapter*. The measurement, the training and the
artifact format all live in :mod:`jev_route.distill.export`,
:mod:`jev_route.distill.train`, :mod:`jev_route.distill.evaluate` and
:mod:`jev_route.distill.artifact`; this module parses and validates ``argparse``
output, calls those libraries, prints what they returned in a form an operator
can act on, writes the files the verb promises, and turns the result into a
process exit code. Nothing here reimplements a metric, and nothing here decides
whether a model is good -- it reports what the library measured.

Three rules this file follows, all of them load-bearing elsewhere in the
project:

* **No heavy import at module scope.** ``import jev_route`` must succeed in a
  container with no numpy, because the router runs there. Every import of the
  four library modules happens inside a function body, and
  :mod:`jev_route.distill`'s ``__init__`` re-exports these five names lazily.
* **No tracebacks.** A missing dataset, a diverged training run, a corrupt
  artifact and an absent extra are all *expected* operator states. Each one
  becomes one line on stderr naming the fix, plus a non-zero exit code. Set
  ``JEV_ROUTE_CLI_TRACEBACK=1`` to get the traceback back when debugging.
* **The numbers are the library's.** When a measurement could not be made -- no
  policy, so no tier agreement; no holdout rows; a head with nothing to grade --
  this module says so instead of substituting a friendlier number.

Exit codes: ``0`` the verb succeeded (for ``graduate``: the model is ready),
``1`` the verb ran and the answer is "not ready", ``2`` the verb could not run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

#: The verb succeeded. For ``graduate`` that means "ready to cut over".
EXIT_OK = 0
#: The verb ran and measured; the answer is "not ready". A report, not a crash.
EXIT_NOT_READY = 1
#: The verb could not run: bad path, unreadable artifact, missing extra.
EXIT_ERROR = 2

#: Mirrors :data:`jev_route.cli.DEFAULT_POLICY`. Duplicated rather than imported
#: because ``jev_route.cli`` imports *this* package, and a module-scope import
#: back the other way would make the cycle real instead of theoretical.
DEFAULT_POLICY = "policies/default.yaml"

#: ``graduate``'s sample-size floor, restated so the advice below can do the
#: arithmetic ("how much more traffic do I need?") without a second constant.
DEFAULT_MIN_SAMPLES = 500


class _CliError(Exception):
    """An operator-facing failure. Carries the message that gets printed."""

    def __init__(self, message: str, *, code: int = EXIT_ERROR) -> None:
        super().__init__(message)
        self.message = str(message)
        self.code = code


def _die(verb: str, message: str, *, code: int = EXIT_ERROR) -> int:
    """Print one actionable line on stderr and return the exit code."""
    print(f"jev-route {verb}: error: {str(message).strip()}", file=sys.stderr)
    return code


def _library_errors() -> tuple[type[Exception], ...]:
    """The four libraries' own exception types, imported only when one is raised.

    Called from an ``except`` clause, so a machine with no numpy never pays for
    the import unless something already went wrong far enough to need it.
    """
    from .artifact import ArtifactError, DistillDependencyError
    from .evaluate import EvaluateError
    from .export import ExportError
    from .train import TrainError

    return (ArtifactError, DistillDependencyError, EvaluateError, ExportError, TrainError)


#: The library error messages name the real verbs (``jev-route export`` /
#: ``train`` / ...); tests/test_distill_cli.py pins that no message names a verb
#: the CLI does not have.

def _guard(verb: str, body: Callable[[], int]) -> int:
    """Run a verb body, converting every expected failure into a printed reason.

    ``SystemExit`` is re-raised on purpose: ``_find_policy`` in
    :mod:`jev_route.cli` uses it, and it already carries an actionable message.
    """
    try:
        return body()
    except _CliError as exc:
        return _die(verb, exc.message, code=exc.code)
    except SystemExit:
        raise
    except _library_errors() as exc:
        return _die(verb, str(exc))
    except OSError as exc:
        where = getattr(exc, "filename", None) or "the requested path"
        return _die(verb, f"cannot use {where}: {getattr(exc, 'strerror', None) or exc}")
    except Exception as exc:
        # Deliberately broad: the CLI must never end in a traceback. The escape
        # hatch above keeps the real exception available when debugging.
        if os.environ.get("JEV_ROUTE_CLI_TRACEBACK"):
            raise
        return _die(
            verb,
            f"unexpected {type(exc).__name__}: {exc}\n"
            "  this is a bug in jev-route, not in your data. Re-run with "
            "JEV_ROUTE_CLI_TRACEBACK=1 for the traceback.",
        )


# --------------------------------------------------------------------------- #
# locating the log, the policy and the dataset
# --------------------------------------------------------------------------- #
def _policy_candidates(explicit: str | None) -> list[Path]:
    """The policy files to try, in order. Mirrors :func:`jev_route.cli._find_policy`."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("JEV_ROUTE_POLICY")
    if env:
        candidates.append(Path(env))
    candidates += [Path(DEFAULT_POLICY), Path(__file__).resolve().parents[3] / DEFAULT_POLICY]
    return candidates


def _find_policy_path(explicit: str | None) -> Path | None:
    """The first policy candidate that exists, or ``None``.

    ``None`` rather than an error because two verbs treat a missing policy as
    optional: ``evaluate`` still reports accuracy and calibration without one,
    and ``export`` only needs it to find the configured log path.
    """
    for candidate in _policy_candidates(explicit):
        if candidate.is_file():
            return candidate
    return None


def _load_policy(path: str | Path) -> Any:
    from ..policy import Policy

    return Policy.from_file(Path(path))


def _resolve_log_path(args: argparse.Namespace) -> Path:
    """The decision log to export: ``--log``, else the path the policy configures."""
    explicit = getattr(args, "log", None)
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise _CliError(
                f"decision log not found: {path}\n"
                "  Route some traffic first (`jev-route route ...`), or point --log at the "
                "JSONL file your deployment writes (policy key `logging.path`)."
            )
        return path
    policy_path = _find_policy_path(getattr(args, "policy", None))
    if policy_path is not None:
        configured = str((_load_policy(policy_path).logging or {}).get("path") or "")
        if configured:
            path = Path(configured)
            if path.exists():
                return path
            raise _CliError(
                f"policy {policy_path} sends its decision log to {path}, which does not exist.\n"
                "  Nothing has been routed under that policy yet, so there is nothing to export. "
                "Pass --log to name a different log."
            )
    raise _CliError(
        "no decision log to export. Pass --log <decisions.jsonl>, or route traffic under a policy "
        "whose `logging.path` exists (`jev-route route '...'` writes one)."
    )


def _dataset_out_path(raw: str) -> Path:
    """Where ``export --out`` writes.

    ``--out data/ds`` means the *directory* ``data/ds`` holding ``dataset.jsonl``,
    because that is what ``train --data data/ds`` and the documentation both say.
    The library only treats ``out`` as a directory when it already exists, so an
    extension-less first export would otherwise land in ``data/ds.jsonl`` and the
    next verb in the lifecycle would not find it. Naming a ``.jsonl`` file
    directly still means that file.
    """
    target = Path(raw)
    if target.suffix == ".jsonl":
        return target
    target.mkdir(parents=True, exist_ok=True)
    return target / "dataset.jsonl"


def _recorded_dataset(artifact: Any) -> str:
    """The dataset path an artifact says it was trained on, or ``""``."""
    block = artifact.dataset if isinstance(artifact.dataset, Mapping) else {}
    for key in ("source", "source_as_given"):
        value = str((block or {}).get(key) or "")
        if value:
            return value
    meta = artifact.metadata.get("dataset") if isinstance(artifact.metadata.get("dataset"), Mapping) else {}
    for key in ("source", "source_as_given"):
        value = str((meta or {}).get(key) or "")
        if value:
            return value
    return str(artifact.metadata.get("dataset_source") or "")


def _resolve_dataset_path(explicit: str | None, artifact: Any, *, verb: str) -> Path:
    """``--data`` if given, else the dataset the artifact records, if it is here.

    The recorded path is stored absolute at train time, with the relative form
    the operator typed beside it; both are tried, because an artifact copied to
    another checkout keeps only one of them valid.
    """
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise _CliError(
                f"--data {path} does not exist. Export a dataset first: "
                "`jev-route export --log <decisions.jsonl> --out <dir>`."
            )
        return path
    recorded = _recorded_dataset(artifact)
    if not recorded:
        raise _CliError(
            f"{artifact.source} records no dataset path, so {verb} cannot find the held-out rows "
            "on its own. Pass --data <dataset dir or .jsonl>."
        )
    block = artifact.dataset if isinstance(artifact.dataset, Mapping) else {}
    candidates = [Path(recorded)]
    fallback = str((block or {}).get("source_as_given") or "")
    if fallback:
        candidates.append(Path(fallback))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise _CliError(
        f"this artifact was trained on {recorded}, which is not on this machine.\n"
        f"  Pass --data <dataset dir or .jsonl> to point {verb} at a copy of it. "
        "Without the held-out rows there is nothing to measure against."
    )


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #
def _print_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _num(value: Any, spec: str = ".4f", dash: str = "n/a") -> str:
    """Format a measured number, or ``dash`` when it was not measured."""
    if value is None:
        return dash
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return dash
    return format(number, spec)


def _pct(value: Any, dash: str = "n/a") -> str:
    return _num(value, ".1%", dash)


def _human_bytes(count: int) -> str:
    size = float(max(0, int(count)))
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _next_step(text: str) -> None:
    print(f"\nnext: {text}")


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def _export(args: argparse.Namespace) -> int:
    """Decision log -> training dataset. All the deciding is ``export_dataset``'s."""
    from .export import export_dataset

    log = _resolve_log_path(args)
    holdout = float(args.holdout)
    if not 0.0 <= holdout < 0.9:
        raise _CliError(
            f"--holdout must be in [0, 0.9), got {holdout}. 0.2 holds out one request in five; "
            "0 exports everything as training rows and leaves `evaluate` and `graduate` nothing to measure."
        )
    result = export_dataset(log, _dataset_out_path(args.out), mode=args.mode, holdout_fraction=holdout)
    stats = result.stats

    if args.json:
        _print_json(
            {
                "log": str(log),
                "rows_path": str(result.rows_path),
                "stats_path": str(result.stats_path),
                "requested_seed": int(args.seed),
                "split_is_hash_of_request_id": True,
                "inspection": result.inspection.as_dict(),
                "stats": stats.as_dict(),
            }
        )
        return EXIT_OK

    print(f"decision log     : {log}")
    print(f"records read     : {stats.records_read}")
    print(f"mode             : {stats.requested_mode} -> {stats.resolved_mode}")
    print(f"holdout fraction : {stats.holdout_fraction:.2f}")
    print()
    print(stats.summary())
    print()
    print(f"dataset          : {result.rows_path}")
    print(f"stats sidecar    : {result.stats_path}")
    print(f"sha256           : {stats.dataset_sha256}")
    text_note = "YES -- treat this file as sensitive" if stats.contains_prompt_text else "no (features only)"
    print(f"prompt text      : {text_note}")
    # The split is a hash of request_id, not a shuffled index, so rows never move
    # between sides as the log grows. That is what makes the holdout trustworthy,
    # and it also means --seed cannot change it; saying so beats a silent no-op.
    print(
        f"split            : {stats.train_rows} train / {stats.holdout_rows} holdout, keyed on a hash of "
        "request_id (stable as the log grows; --seed does not affect it)"
    )
    if stats.teacher_model_versions:
        print(f"teacher versions : {', '.join(sorted(stats.teacher_model_versions))}")
    if stats.gate_rows:
        print(
            f"gate rows        : {stats.gate_rows} ({stats.gate_blocked_rows} never reached a backend); "
            "kept, with the heads no backend judged masked out"
        )
    if stats.holdout_rows == 0:
        print(
            "warning: the holdout is empty, so `evaluate` and `graduate` have nothing to measure "
            "against. Re-export with a larger --holdout, or collect more traffic."
        )
    _next_step(f"jev-route train --data {result.rows_path.parent} --out artifacts/v1")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def _resolve_trainer(requested: str) -> tuple[str, str]:
    """``--trainer`` -> a framework this machine can actually run, plus a note.

    ``auto`` means numpy, and that is a decision rather than a detection: the
    trainer's own docstring argues that a ~1k-parameter linear model on a sparse
    input does not justify a 2 GB deep-learning runtime in the request path. The
    torch path is opt-in, and asking for it without torch installed gets the pip
    incantation instead of an ``ImportError``.
    """
    from .artifact import require_numpy

    name = str(requested or "auto").lower()
    if name == "auto":
        require_numpy()
        return "numpy", "auto -> numpy (the default; torch is opt-in and buys nothing at this model size)"
    if name == "numpy":
        require_numpy()
        return "numpy", "numpy"
    if name == "torch":
        try:
            import torch  # noqa: F401 - presence is the only thing being checked
        except ImportError as exc:
            raise _CliError(
                "--trainer torch needs torch, which is not installed here:\n"
                "    pip install 'jev-route[distill-torch]'\n"
                f"  (import failed with: {exc})\n"
                "  The numpy trainer is the default and needs only numpy: drop --trainer, "
                "or pass --trainer numpy. It trains the same architecture and writes the same artifact."
            ) from exc
        return "torch", "torch"
    raise _CliError(f"unknown --trainer {requested!r}; expected auto, numpy or torch")


def _parse_head_weights(raw: str | None, defaults: Mapping[str, float]) -> tuple[dict[str, float], list[str]]:
    """``--head-weights '{"sensitivity":2.0}'`` merged over the defaults.

    Merged, not replaced: the documented override names the one head the operator
    cares about, and silently dropping the sensitivity=1.5 default for the other
    three would change the objective they did not ask to change.
    """
    from .artifact import ALL_HEADS

    if not raw:
        return dict(defaults), []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _CliError(
            f"--head-weights is not valid JSON: {exc}\n"
            "  Expected an object of head -> positive weight, e.g. '{\"sensitivity\":2.0}'."
        ) from exc
    if not isinstance(parsed, Mapping):
        raise _CliError(f"--head-weights must be a JSON object, got {type(parsed).__name__}")
    unknown = sorted(set(parsed) - set(ALL_HEADS))
    if unknown:
        raise _CliError(
            f"--head-weights names unknown head(s) {unknown}; the heads are {sorted(ALL_HEADS)}. "
            "A typo here would quietly train with the default weight."
        )
    weights = dict(defaults)
    notes: list[str] = []
    for head, value in parsed.items():
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise _CliError(f"--head-weights[{head!r}] must be a number, got {value!r}") from exc
        if not number > 0 or math.isnan(number):
            raise _CliError(
                f"--head-weights[{head!r}] must be a positive number, got {number}. "
                "Use a small positive weight to down-weight a head; 0 would remove it from the objective "
                "without removing it from the artifact."
            )
        weights[str(head)] = number
        notes.append(f"{head} {defaults.get(str(head), 1.0)} -> {number}")
    return weights, notes


def _validate_training_numbers(args: argparse.Namespace) -> None:
    """Reject knob values that would produce a meaningless run, before spending it."""
    checks = [
        ("--temperature", float(args.temperature), lambda v: v > 0, "must be > 0 (2.0 is the documented default)"),
        ("--epochs", float(args.epochs), lambda v: v >= 1, "must be at least 1"),
        ("--lr", float(args.lr), lambda v: 0 < v < 10, "must be in (0, 10); 0.05 is the CLI default"),
        ("--l2", float(args.l2), lambda v: v >= 0, "must be >= 0"),
    ]
    for flag, value, ok, why in checks:
        if math.isnan(value) or not ok(value):
            raise _CliError(f"{flag} {value:g} {why}")


def _model_version(mode: str, dataset_sha256: str, trained_at: str) -> str:
    """A version string an operator can read in a decision record.

    Derived from the dataset checksum and the training date rather than a clock,
    so re-running the same training on the same log produces the same version and
    the log's ``backend_model_version`` column stays interpretable.
    """
    day = (trained_at or "")[:10].replace("-", "") or "unknown"
    return f"distilled-{mode}-{(dataset_sha256 or 'nodata')[:8]}-{day}"


def _training_block(student: Any) -> dict[str, Any]:
    """The training section of the artifact envelope: the config plus what happened."""
    training: dict[str, Any] = dict(student.config.to_dict())
    for key in (
        "epochs_run",
        "best_epoch",
        "stopped_early",
        "final_train_loss",
        "best_val_loss",
        "training_seconds",
        "framework",
        "final_head_losses",
        "usable_rows_per_head",
    ):
        if key in student.stats:
            training[key] = student.stats[key]
    return training


def _dataset_block(student: Any, dataset: Any) -> dict[str, Any]:
    """``dataset.json``: the export sidecar, so the artifact knows what it was trained on.

    ``source`` is absolute and ``source_as_given`` is what the operator typed,
    because an artifact outlives the working directory it was built in and one of
    the two is usually still resolvable.
    """
    stats = dict(dataset.stats or {})
    given = str(dataset.source or "")
    source = str(Path(given).resolve()) if given else ""
    rows = list(dataset.rows)
    block: dict[str, Any] = {
        "schema": "jev_route.distill.dataset/1",
        "source": source,
        "source_as_given": given,
        "sha256": student.dataset_sha256 or str(stats.get("dataset_sha256", "")),
        "mode": dataset.mode,
        "rows": len(rows),
        "train_rows": int(stats.get("train_rows") or sum(1 for r in rows if r.split == "train")),
        "holdout_rows": int(stats.get("holdout_rows") or sum(1 for r in rows if r.split == "holdout")),
        "holdout_fraction": stats.get("holdout_fraction"),
        "usable_rows_per_head": dict(student.stats.get("usable_rows_per_head") or {}),
        "label_support": dict(student.stats.get("label_support") or {}),
        "teacher_model_versions": list(student.teacher_model_versions),
        "trainer_fit_rows": student.stats.get("train_rows"),
        "trainer_val_rows": student.stats.get("val_rows"),
        "export_stats": stats,
    }
    return block


def _artifact_metadata(student: Any, dataset: Any, dataset_block: Mapping[str, Any]) -> dict[str, Any]:
    """The ``artifact.json`` envelope. Everything a future reader needs to audit it."""
    from .artifact import ARTIFACT_KIND, ARTIFACT_SCHEMA_VERSION, environment_info

    version = _model_version(student.mode, str(dataset_block.get("sha256", "")), student.trained_at)
    backends = sorted({str(r.teacher.get("backend") or "") for r in dataset.rows} - {""})
    return {
        "kind": ARTIFACT_KIND,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_version": version,
        "created_at": student.trained_at,
        "mode": student.mode,
        "contains_prompt_text": student.mode == "text",
        "model": student.model.to_dict(),
        "training": _training_block(student),
        "teacher": {
            "name": "jev" if "jev" in backends else (backends[0] if backends else "unknown"),
            "backends": backends,
            "model_versions": list(student.teacher_model_versions),
        },
        "dataset": {
            "source": dataset_block.get("source"),
            "source_as_given": dataset_block.get("source_as_given"),
            "sha256": dataset_block.get("sha256"),
            "mode": dataset_block.get("mode"),
            "rows": dataset_block.get("rows"),
        },
        "environment": environment_info(),
        "history": {
            "epochs": len(student.history),
            "first": dict(student.history[0]) if student.history else {},
            "final": dict(student.history[-1]) if student.history else {},
        },
        "warnings": list(student.warnings),
    }


def _progress_printer(args: argparse.Namespace, epochs: int) -> Callable[[int, float, float], None] | None:
    """One line per tenth of the run, on stderr, and nothing at all under ``--json``."""
    if getattr(args, "json", False):
        return None
    every = max(1, int(epochs) // 10)

    def report(epoch: int, train_loss: float, val_loss: float) -> None:
        if epoch % every and epoch != int(epochs):
            return
        val = "" if math.isnan(val_loss) else f" val={val_loss:.5f}"
        print(f"  epoch {epoch:>4}/{int(epochs)}  train={train_loss:.5f}{val}", file=sys.stderr)

    return report


def _train(args: argparse.Namespace) -> int:
    """Distill a student from an exported dataset and write it as an artifact."""
    from .artifact import DistilledArtifact, load_artifact
    from .export import load_dataset
    from .train import DEFAULT_HEAD_WEIGHTS, TrainConfig, train

    _validate_training_numbers(args)
    dataset = load_dataset(args.data)
    framework, trainer_note = _resolve_trainer(args.trainer)
    head_weights, weight_notes = _parse_head_weights(args.head_weights, DEFAULT_HEAD_WEIGHTS)
    config = TrainConfig(
        framework=framework,
        mode="auto",
        temperature=float(args.temperature),
        epochs=int(args.epochs),
        learning_rate=float(args.lr),
        l2=float(args.l2),
        seed=int(args.seed),
        head_weights=head_weights,
    )

    student = train(dataset, config, progress=_progress_printer(args, config.epochs))
    dataset_block = _dataset_block(student, dataset)
    artifact = DistilledArtifact(
        model=student.model,
        vectorizer=student.vectorizer,
        metadata=_artifact_metadata(student, dataset, dataset_block),
        metrics=None,
        dataset=dict(dataset_block),
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    artifact.save(out)
    # Read it back before claiming success. An artifact that will not load is
    # worse than no artifact: `graduate` would refuse it later, far from here.
    reloaded = load_artifact(out)

    files = sorted(p for p in out.iterdir() if p.is_file())
    total_bytes = sum(p.stat().st_size for p in files)
    if args.json:
        _print_json(
            {
                "artifact": str(out),
                "files": {p.name: p.stat().st_size for p in files},
                "bytes": total_bytes,
                "model_version": reloaded.model_version,
                "config": config.to_dict(),
                "head_weight_overrides": weight_notes,
                "stats": student.stats,
                "history": list(student.history),
                "warnings": list(student.warnings),
                "summary": student.summary(),
            }
        )
        return EXIT_OK

    print(f"dataset          : {dataset.source}")
    print(
        f"rows             : {len(dataset.rows)} exported "
        f"({student.stats.get('train_rows')} fit / {student.stats.get('val_rows')} early-stopping val / "
        f"{student.stats.get('holdout_rows_excluded')} holdout, never seen)"
    )
    print(f"mode             : {student.mode}")
    print(f"trainer          : {trainer_note}")
    print(
        f"objective        : KL(teacher || student) on soft targets, T={config.temperature:g}, "
        f"head weights {dict(head_weights)}"
    )
    for note in weight_notes:
        print(f"  override       : {note}")
    print(f"student          : {student.summary()}")
    print(
        f"epochs           : {student.stats.get('epochs_run')}/{config.epochs} run, "
        f"best={student.stats.get('best_epoch')}, early stop={student.stats.get('stopped_early')}"
    )
    print(
        f"loss             : final train {_num(student.stats.get('final_train_loss'), '.5f')}, "
        f"best val {_num(student.stats.get('best_val_loss'), '.5f')} "
        f"({student.stats.get('training_seconds')}s, seed={config.seed})"
    )
    per_head = student.stats.get("final_head_losses") or {}
    if per_head:
        print("per-head KL      : " + "  ".join(f"{k}={_num(v, '.4f')}" for k, v in sorted(per_head.items())))
    for warning in student.warnings:
        print(f"warning          : {warning}")
    print()
    print(f"artifact         : {out}  ({len(files)} files, {_human_bytes(total_bytes)})")
    print(f"model_version    : {reloaded.model_version}")
    print(f"teacher          : {', '.join(reloaded.teacher_model_versions) or 'unknown'}")
    print(f"sha256 (dataset) : {dataset_block.get('sha256')}")
    _next_step(f"jev-route evaluate --artifact {out}")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
def _resolve_policy_arg(explicit: str | None, *, required: bool = False) -> Path | None:
    """The policy to measure tier agreement under, or ``None`` if there is none.

    An explicit ``--policy`` that does not exist is an error rather than a silent
    fallback: the operator named a file, and quietly grading the student under a
    different policy would produce a number that looks like theirs and is not.
    """
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise _CliError(
                f"--policy {path} does not exist. Pass the policy you actually run "
                f"(the shipped default is {DEFAULT_POLICY})."
            )
        return path
    found = _find_policy_path(None)
    if found is None and required:
        raise _CliError(
            f"no policy file found (tried: {', '.join(str(c) for c in _policy_candidates(None))}). "
            "Pass --policy; tier agreement is measured under the policy you run, so there is no default "
            "that would mean the same thing on another machine."
        )
    return found


def _split_counts(dataset: Any) -> dict[str, int]:
    counts = {"train": 0, "holdout": 0}
    for row in dataset.rows:
        counts[row.split if row.split in counts else "train"] += 1
    return counts


def _measure(artifact: Any, dataset_path: Path, policy_path: Path | None, bins: int) -> Any:
    """Load the dataset once and evaluate. Shared by ``evaluate``, ``package``, ``graduate``."""
    from .evaluate import evaluate
    from .export import load_dataset

    dataset = load_dataset(dataset_path)
    counts = _split_counts(dataset)
    if not counts["holdout"]:
        raise _CliError(
            f"{dataset_path} has no holdout rows ({counts['train']} train), so there is nothing to "
            "measure the student against that it was not trained on. Re-export with a non-zero "
            "--holdout (0.2 holds out one request in five)."
        )
    return evaluate(artifact, dataset, policy=policy_path, split="holdout", bins=bins)


def _evaluate(args: argparse.Namespace) -> int:
    """Accuracy, ECE, teacher agreement and latency, on the held-out rows."""
    from .artifact import load_artifact
    from .evaluate import render_report

    bins = int(args.bins)
    if bins < 2:
        raise _CliError(f"--bins must be at least 2, got {bins}. 10 is the documented default; 15 the library's.")
    artifact = load_artifact(args.artifact)
    dataset_path = _resolve_dataset_path(args.data, artifact, verb="evaluate")
    policy_path = _resolve_policy_arg(args.policy)
    report = _measure(artifact, dataset_path, policy_path, bins)

    if args.json:
        payload = report.to_dict()
        payload["artifact"] = str(artifact.source)
        payload["policy"] = str(policy_path) if policy_path else None
        _print_json(payload)
        return EXIT_OK

    print(f"artifact         : {artifact.source}")
    print(f"model_version    : {artifact.model_version}")
    text_note = "retains prompt text" if artifact.contains_prompt_text else "no prompt text"
    print(f"mode             : {artifact.mode} ({text_note})")
    print(f"dataset          : {dataset_path}")
    print(f"policy           : {policy_path or '(none -- tier agreement skipped)'}")
    print()
    print(render_report(report, verbose=bool(getattr(args, "verbose", False))))
    print()
    print("the four numbers `graduate` gates on")
    agreement = report.agreement
    tier_counts = agreement.tier_counts if agreement else None
    print(
        f"  tier agreement : {_pct(report.tier_agreement)}"
        + (f" ({tier_counts[0]}/{tier_counts[1]})" if tier_counts else "")
    )
    print(f"  worst ECE      : {_num(report.worst_ece)} (head: {report.worst_ece_head or 'n/a'})")
    print(
        f"  student p50    : {_num(report.latency.p50, '.3f')} ms "
        f"(teacher p50 {_num(report.latency.teacher.get('p50'), '.1f')} ms, "
        f"delta {_num(report.latency.delta_p50_ms, '+.3f')} ms)"
    )
    print(f"  held-out rows  : {report.n_rows}")
    if report.n_rows < DEFAULT_MIN_SAMPLES:
        print(
            f"  sample size    : {report.n_rows} is below graduate's default --min-samples "
            f"{DEFAULT_MIN_SAMPLES}. Every number above is noise at this size; it is a smoke test "
            "of the pipeline, not evidence for a cutover."
        )
    if agreement is not None and agreement.ceiling_tier_agreement is not None:
        print(
            f"  ceiling        : a *perfect* student scores {_pct(agreement.ceiling_tier_agreement)} tier "
            "agreement under this policy (the teacher's own distributions, replayed with the "
            "student's derived confidence). Tier agreement cannot beat it by much."
        )
    _next_step(f"jev-route package --model {artifact.source} --out artifacts/v1-packaged")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# package
# --------------------------------------------------------------------------- #
def _artifact_files(path: Path) -> dict[str, int]:
    """``{filename: bytes}`` for a written artifact, directory or zip.

    Read off what is actually on disk rather than off the envelope's manifest,
    because the manifest checksums the *other* files and so never lists
    ``artifact.json`` itself -- reporting from it would understate the contents.
    """
    import zipfile

    if path.is_dir():
        return {p.name: p.stat().st_size for p in sorted(path.iterdir()) if p.is_file()}
    with zipfile.ZipFile(path) as archive:
        return {info.filename: int(info.file_size) for info in sorted(archive.infolist(), key=lambda i: i.filename)}


def _package_source(args: argparse.Namespace) -> tuple[str, str | None]:
    """The model to package: ``--artifact``, or the deprecated ``--model`` alias.

    Returns ``(path, deprecation warning)``. Both spellings name the same object,
    which is exactly why ``--artifact`` is now the documented one: ``evaluate``
    and ``graduate`` already call it that, and a vocabulary where the same file is
    a "model" in one verb and an "artifact" in the next reads like two concepts.
    The alias is accepted for one release and warns, so a script written against
    0.1 keeps working but does not stay silent about the change.
    """
    artifact = getattr(args, "artifact", None)
    model = getattr(args, "model", None)
    if artifact and model and str(artifact) != str(model):
        raise _CliError(
            f"pass either --artifact or the deprecated --model, not both: they disagree "
            f"({artifact!r} vs {model!r}) and there is no rule for which one wins."
        )
    if artifact:
        return str(artifact), None
    if model:
        return str(model), (
            "--model is a deprecated alias for --artifact and will be removed in 0.3.0. "
            "Use: jev-route package --artifact <trained model> --out <artifact>"
        )
    raise _CliError(
        "package needs the trained model to package. Pass --artifact <artifact directory or zip> "
        "(the output of `jev-route train`), and --out <destination>."
    )


def _package(args: argparse.Namespace) -> int:
    """Write a loadable, self-describing artifact from a trained model."""
    from .artifact import DistilledArtifact, load_artifact, utcnow

    model_path, deprecation = _package_source(args)
    if deprecation:
        # stderr, so `--json` on stdout stays parseable by a script that is being warned.
        print(f"jev-route package: warning: {deprecation}", file=sys.stderr)
    source = load_artifact(model_path)
    notes: list[str] = []
    metrics = source.metrics
    if metrics is None:
        # The documented artifact layout carries metrics.json. Computing them here
        # needs the dataset the artifact was trained on; when that is not on this
        # machine the package still succeeds, because a valid servable artifact is
        # worth more than a bundled report -- but the omission is said out loud.
        try:
            dataset_path = _resolve_dataset_path(None, source, verb="package")
            policy_path = _resolve_policy_arg(None)
            report = _measure(source, dataset_path, policy_path, 10)
            metrics = report.to_dict()
            notes.append(
                f"evaluated the held-out rows to embed metrics.json "
                f"({report.n_rows} rows, policy {policy_path or 'none'})"
            )
        except _CliError as exc:
            notes.append(f"no metrics embedded: {exc.message.splitlines()[0]}")
    else:
        notes.append("carried over the metrics.json already in the model directory")

    metadata = {**source.metadata, "packaged_from": str(source.source or ""), "packaged_at": utcnow()}
    packaged = DistilledArtifact(
        model=source.model,
        vectorizer=source.vectorizer,
        metadata=metadata,
        metrics=metrics,
        dataset=source.dataset,
    )
    out = Path(args.out)
    as_zip = out.suffix == ".zip"
    if as_zip:
        target = packaged.save_zip(out)
    else:
        out.mkdir(parents=True, exist_ok=True)
        target = packaged.save(out)
    reloaded = load_artifact(target)
    files = _artifact_files(target)
    total_bytes = sum(files.values())

    if args.json:
        _print_json(
            {
                "artifact": str(target),
                "format": "zip" if as_zip else "directory",
                "packaged_from": str(source.source or ""),
                "model_version": reloaded.model_version,
                "files": files,
                "bytes": total_bytes,
                "metrics_embedded": metrics is not None,
                "notes": notes,
                "describe": reloaded.describe(),
            }
        )
        return EXIT_OK

    _print_package(source=source, target=target, reloaded=reloaded, files=files, notes=notes, as_zip=as_zip)
    return EXIT_OK


def _print_package(
    *,
    source: Any,
    target: Path,
    reloaded: Any,
    files: Mapping[str, int],
    notes: Sequence[str],
    as_zip: bool,
) -> None:
    """The human-readable half of ``package``: what was written, and how to serve it."""
    total_bytes = sum(files.values())
    print(f"packaged from    : {source.source}")
    print(f"artifact         : {target} ({'zip' if as_zip else 'directory'})")
    for note in notes:
        print(f"  note           : {note}")
    print()
    print(reloaded.describe())
    print()
    print(f"contents         : {len(files)} file(s), {_human_bytes(total_bytes)}, SHA-256 per file in artifact.json")
    for name, size in files.items():
        print(f"  {name:<18} {_human_bytes(size):>9}")
    # Named from what was actually written rather than from the documentation:
    # `StudentModel.weight_bytes()` emits one raw .npy per parameter array.
    weight_files = ", ".join(n for n in files if n.endswith(".npy")) or "none"
    print(f"no pickle anywhere: weights are raw arrays ({weight_files}), everything else is JSON")
    print()
    print("serve it with:")
    print("  backend:")
    print("    name: distilled")
    print(f"    artifact: {target}")
    if reloaded.mode == "features":
        print("    feature_mode: true        # this artifact reads the request's features, not its text")
    # `package` has no --policy flag (see build_parser in jev_route.cli), so the
    # suggestion names the policy this artifact was graded under, or the default.
    policy_hint = _find_policy_path(None) or DEFAULT_POLICY
    _next_step(f"jev-route graduate --artifact {target} --policy {policy_hint}")


# --------------------------------------------------------------------------- #
# graduate
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Check:
    """One of ``graduate``'s four thresholds, with the number it was judged on."""

    name: str
    requirement: str
    measured: str
    passed: bool
    detail: str = ""


def _checks_from_report(report: Any, args: argparse.Namespace) -> list[_Check]:
    """The four documented gates. Nothing here is tunable except by the flags."""
    agreement = report.agreement
    tier = report.tier_agreement
    counts = agreement.tier_counts if agreement else None
    if tier is None:
        tier_detail = (
            "no held-out row produced a comparable tier (every one was gate-blocked, or the policy never fired)"
        )
    else:
        tier_detail = f"{counts[0]}/{counts[1]} rows landed on the teacher's tier" if counts else ""
    ece = report.worst_ece
    ece_ok = not (ece is None or math.isnan(float(ece))) and float(ece) <= float(args.max_ece)
    p50 = float(report.latency.p50)
    latency_ok = not math.isnan(p50) and p50 <= float(args.max_latency_ms)
    teacher_p50 = report.latency.teacher.get("p50")
    return [
        _Check(
            "min_tier_agreement",
            f">= {float(args.min_tier_agreement):.2f}",
            _pct(tier),
            bool(tier is not None and tier >= float(args.min_tier_agreement)),
            tier_detail,
        ),
        _Check(
            "max_ece",
            f"<= {float(args.max_ece):.2f}",
            _num(ece),
            bool(ece_ok),
            f"worst head: {report.worst_ece_head or 'none produced a number'}",
        ),
        _Check(
            "max_latency_ms",
            f"<= {float(args.max_latency_ms):.1f}",
            f"{p50:.3f} ms",
            bool(latency_ok),
            f"teacher p50 {_num(teacher_p50, '.1f')} ms, so graduating changes the request path by "
            f"{_num(report.latency.delta_p50_ms, '+.1f')} ms",
        ),
        _Check(
            "min_samples",
            f">= {int(args.min_samples)}",
            f"{report.n_rows} rows",
            bool(report.n_rows >= int(args.min_samples)),
            "held-out rows the numbers above were computed on",
        ),
    ]


def _print_checks(checks: Sequence[_Check]) -> None:
    width = max(len(c.name) for c in checks)
    for check in checks:
        verdict = "pass" if check.passed else "FAIL"
        print(f"  {check.name.ljust(width)}   {check.requirement:<10} {check.measured:<14} [ {verdict} ]")
        if check.detail:
            print(f"  {''.ljust(width)}   {check.detail}")


def _samples_advice(report: Any, artifact: Any, min_samples: int) -> list[str]:
    """How much more traffic the sample-size floor actually needs. Arithmetic, not encouragement."""
    block = artifact.dataset if isinstance(artifact.dataset, Mapping) else {}
    fraction = float((block or {}).get("holdout_fraction") or 0.0)
    total = int((block or {}).get("rows") or 0)
    if not (0 < fraction < 1):
        return [
            f"only {report.n_rows} held-out rows. Collect more traffic and re-export; the artifact does not "
            "record the holdout fraction, so the exact target cannot be computed here."
        ]
    needed_total = math.ceil(min_samples / fraction)
    more = max(0, needed_total - total)
    return [
        f"only {report.n_rows} held-out rows, and every number above is noise at that size. "
        f"At --holdout {fraction:.2f} you need about {needed_total} exported rows "
        f"({more} more than the {total} in this dataset) before a cutover decision means anything.",
        "This is the one threshold the documentation tells you not to lower: a 0.97 tier agreement on "
        "80 samples is not a 0.97 tier agreement.",
    ]


def _tier_advice(report: Any, args: argparse.Namespace) -> list[str]:
    """Why tier agreement missed, and which of the three possible causes it is."""
    agreement = report.agreement
    if agreement is None:
        return ["tier agreement was not measured: no policy was available to replay the answers through."]
    out: list[str] = []
    ceiling = agreement.ceiling_tier_agreement
    if ceiling is not None and ceiling < float(args.min_tier_agreement):
        out.append(
            f"even a PERFECT student clears only {_pct(ceiling)} tier agreement under this policy, so the "
            f"{float(args.min_tier_agreement):.2f} floor is unreachable by training. The gap is the policy's "
            "confidence handling, not the model: a distilled answer's confidence is derived from its own "
            "distribution while the teacher reports its own, so the on_uncertain floors fire in different "
            "places. Options: re-tune on_uncertain against the student, lower --min-tier-agreement below the "
            "ceiling knowingly, or stay on the teacher."
        )
    missed = agreement.tier_counts[1] - agreement.tier_counts[0] if agreement.tier_counts else 0
    if missed:
        changes = ", ".join(f"{k} x{v}" for k, v in list(agreement.tier_confusion.items())[:5])
        out.append(
            f"the student would route {missed} of {agreement.tier_counts[1]} held-out decisions to a different "
            f"tier than the teacher{f' ({changes})' if changes else ''}. "
            f"{agreement.boundary_errors} of them crossed the sensitivity/confidential line, which is the "
            "difference between your own hardware and a third-party API."
        )
    if agreement.confidence_only_flips:
        out.append(
            f"{agreement.confidence_only_flips} tier flip(s) happened with every label agreeing -- a confidence "
            "floor firing on one backend's scale and not the other's."
        )
    if agreement.replay_fidelity is not None and agreement.replay_fidelity < 1.0:
        out.append(
            f"replaying the teacher's own logged answers reproduces the logged tier on only "
            f"{_pct(agreement.replay_fidelity)} of rows, so the policy used here is not quite the policy that "
            "produced the log. Read the tier numbers as approximate."
        )
    return out or ["tier agreement missed the floor; run `jev-route evaluate --artifact ...` for the per-head detail."]


def _ece_advice(report: Any) -> list[str]:
    head = report.worst_ece_head
    metrics = report.heads.get(head) if head else None
    if metrics is None:
        return ["no head produced a calibration number, so calibration could not be checked at all."]
    out = [
        f"head {head!r} has ECE {_num(metrics.ece)} against the teacher's probabilities "
        f"(macro-F1 {_num(metrics.macro_f1)}). An accurate but overconfident student silently disables every "
        "on_uncertain rule in the policy, which is the thing worth distilling."
    ]
    teacher_ece = getattr(metrics, "teacher_argmax_ece", None)
    if teacher_ece is not None and not math.isnan(float(teacher_ece)):
        out.append(
            f"for scale: the teacher graded the same textbook way scores {_num(teacher_ece)} on this head, and "
            f"the student's textbook number is {_num(metrics.argmax_ece)}."
        )
    return out


def _recommendation(report: Any, artifact: Any, checks: Sequence[_Check], args: argparse.Namespace) -> list[str]:
    """One actionable line per failed check. This is the part an operator acts on."""
    advice: list[str] = []
    failed = {c.name for c in checks if not c.passed}
    if "min_tier_agreement" in failed:
        advice += _tier_advice(report, args)
    if "max_ece" in failed:
        advice += _ece_advice(report)
    if "max_latency_ms" in failed:
        advice.append(
            f"the student's own inference p50 is {report.latency.p50:.3f} ms, above the "
            f"{float(args.max_latency_ms):.1f} ms ceiling. That is unusual for a linear model on a sparse "
            "vector; check whether the artifact was trained in text mode with a large vocabulary."
        )
    if "min_samples" in failed:
        advice += _samples_advice(report, artifact, int(args.min_samples))
    return advice


def _print_student_vs_teacher(report: Any) -> None:
    """Per head: what the student achieves next to what the teacher achieved.

    Printed by ``graduate`` rather than left to ``evaluate`` because "is my local
    model worse than the model it replaces?" is the question being answered here,
    and the teacher's own numbers are the only honest yardstick available offline.
    """
    print("student vs teacher, per head (the teacher's own numbers are the yardstick)")
    print("  head            n   student acc  macroF1  ECE vs teacher  teacher's own ECE")
    for name, head in report.heads.items():
        print(
            f"  {name:<14}{head.n:>4}   {_pct(head.accuracy):>11}  {_num(head.macro_f1, '.3f'):>7}"
            f"  {_num(head.ece, '.4f'):>14}  {_num(head.teacher_argmax_ece, '.4f'):>17}"
        )


def _shadow_config(policy: Any, artifact: Any) -> tuple[dict[str, Any], str]:
    """The teacher side of the graduated policy, and why that teacher was chosen."""
    previous = dict(policy.backend or {})
    name = str(previous.get("name", "")).lower()
    teachers = [str(t) for t in (artifact.teacher_model_versions or ())]
    trained_on_mock = bool(teachers) and all(t.startswith("mock") for t in teachers)
    if name == "jev":
        return previous, "the policy's own jev backend -- the teacher that produced the log"
    if trained_on_mock:
        kept = previous if name else {"name": "mock"}
        return kept, (
            f"the mock backend: this artifact's teacher versions are {teachers}, so the log was produced "
            "by MockBackend rather than by a cloud teacher"
        )
    jev = {
        "name": "jev",
        "api_key_env": str(previous.get("api_key_env", "TYPESAFE_API_KEY")),
        "model": str(previous.get("model", "jev-latest")),
        "timeout_seconds": 5.0,
        "max_retries": 0,
        "include_domain": bool(previous.get("include_domain", True)),
    }
    return jev, (
        f"the cloud teacher. The policy you graduated from used backend {name or 'unset'!r}, which is not "
        f"what this artifact was trained from ({', '.join(teachers) or 'unknown'}); the shadow is set to jev "
        "because that is the teacher the student imitates"
    )


def _graduated_backend(args: argparse.Namespace, artifact: Any, policy: Any) -> tuple[dict[str, Any], str]:
    """The ``backend:`` block ``graduate --write`` produces: serve local, shadow with the teacher."""
    shadow, note = _shadow_config(policy, artifact)
    return {
        "name": "shadow",
        "primary": {
            "name": "distilled",
            # Absolute, so the written policy works from any working directory. The
            # artifact is a directory of checksummed files; copying it is a file copy.
            "artifact": str(Path(args.artifact).resolve()),
            "feature_mode": artifact.mode == "features",
        },
        "shadow": shadow,
        "log_disagreements": True,
        "shadow_timeout_seconds": 5.0,
    }, note


def _destination(args: argparse.Namespace, policy_path: Path) -> Path:
    """Where the graduated policy goes. A NEW file unless ``--in-place`` says otherwise."""
    if getattr(args, "in_place", False):
        return Path(policy_path)
    if getattr(args, "out_policy", None):
        return Path(str(args.out_policy))
    return policy_path.with_name(f"{policy_path.stem}-graduated{policy_path.suffix or '.yaml'}")


def _try_build(cfg: Mapping[str, Any]) -> str:
    """Construct one backend config and report what it turned out to be.

    The config is passed as ``{"backend": cfg}``, the mapping form
    :func:`jev_route.backends.build_backend` documents. Handing it anything else
    is worse than an error: the factory treats an unrecognised argument as an
    empty config and defaults to ``MockBackend``, so a typo here would report
    "builds MockBackend" for a policy that actually names jev -- a verification
    step that always passes is worse than none.
    """
    from ..backends import build_backend

    try:
        backend = build_backend({"backend": dict(cfg)})
    except Exception as exc:  # a missing key or a missing extra is a report, not a crash
        return f"does not construct here: {type(exc).__name__}: {exc}"
    try:
        asyncio.run(backend.aclose())
    except Exception as exc:
        return f"builds {type(backend).__name__}, which failed to close cleanly: {exc}"
    return f"builds {type(backend).__name__}"


def _verify_written_policy(dest: Path) -> str:
    """Load the file we just wrote and build its backend, so the swap is proven, not assumed.

    A graduated policy serves local and shadows the teacher, and those two halves
    fail for unrelated reasons: the local half fails when the artifact is missing or
    the ``distill`` extra is not installed, the teacher half fails when there is no
    API key on this machine. Collapsing both into "the backend does not construct"
    sends an operator hunting for a problem with their model when the real problem
    is an environment variable, so each side is built and reported separately.
    """
    try:
        reloaded = _load_policy(dest)
    except Exception as exc:  # a policy that does not parse is the one thing we must not hand back
        return f"FAIL: {dest} does not parse as a policy: {type(exc).__name__}: {exc}"
    cfg = dict(reloaded.backend or {})
    if str(cfg.get("name", "")).lower() != "shadow":
        return f"verified: {dest} parses and {_try_build(cfg)}"
    sides = {side: _try_build(dict(cfg[side])) for side in ("primary", "shadow") if isinstance(cfg.get(side), dict)}
    report = "; ".join(f"{side} {result}" for side, result in sides.items())
    if any(not result.startswith("builds") for result in sides.values()):
        # Do not also build the combination. A shadow backend whose teacher half
        # cannot be constructed cannot be constructed either, and repeating the
        # same error twice buries the one clause that matters: which half failed.
        return f"{dest} parses; {report.rstrip('.')}. The combined shadow backend cannot build until every side can."
    whole = _try_build(cfg)
    combined = "build the shadow backend" if whole.startswith("builds") else whole
    return f"verified: {dest} parses and {report}; together they {combined}"


def _perform_swap(args: argparse.Namespace, artifact: Any, policy: Any, policy_path: Path) -> dict[str, Any]:
    """Write the graduated policy. Backs up anything it is about to overwrite."""
    backend_cfg, shadow_note = _graduated_backend(args, artifact, policy)
    graduated = policy.with_overrides(backend=backend_cfg)
    dest = _destination(args, policy_path)
    backup: Path | None = None
    if dest.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = dest.with_name(f"{dest.stem}.bak-{stamp}{dest.suffix}")
        shutil.copy2(dest, backup)
    dest.parent.mkdir(parents=True, exist_ok=True)
    graduated.write_yaml(dest)
    return {
        "dest": dest,
        "backup": backup,
        "backend": backend_cfg,
        "shadow_note": shadow_note,
        "verification": _verify_written_policy(dest),
        "in_place": bool(getattr(args, "in_place", False)),
    }


def _print_swap(result: Mapping[str, Any], policy_path: Path) -> None:
    """Show exactly what changed, so a cutover is reviewable in the terminal."""
    import yaml

    dest = result["dest"]
    print()
    print(f"policy written   : {dest}" + ("  (IN PLACE)" if result["in_place"] else "  (new file)"))
    print(f"graduated from   : {policy_path}")
    if result["backup"]:
        print(f"backup           : {result['backup']}")
    print(f"shadow backend   : {result['shadow_note']}")
    print()
    print("the backend block that was written:")
    print(yaml.safe_dump({"backend": result["backend"]}, sort_keys=False, width=100).rstrip())
    print(f"check            : {result['verification']}")
    print()
    print("your local model now serves traffic and the teacher runs alongside it, with every")
    print("disagreement written to the record's `shadow` field. Egress continues while the shadow")
    print("runs -- it stops when you replace this block with:")
    print("  backend:")
    print("    name: distilled")
    print(f"    artifact: {result['backend']['primary']['artifact']}")
    _next_step(f"jev-route route --policy {dest} 'does the cutover route correctly?'")


def _replay_error(dataset_mode: str) -> str:
    """Why ``--replay`` cannot run on a features-mode dataset, and what would make it possible."""
    return (
        "--replay re-runs the held-out PROMPTS through the live teacher, so it needs the prompt text.\n"
        f"  This dataset is {dataset_mode}-mode: it was exported from a log written under "
        "logging.excerpt_mode: hash,\n"
        "  which keeps a hash and the deterministic request features and no text. To make --replay "
        "available:\n"
        "    1. set logging.excerpt_mode: redacted in the policy (docs/privacy.md prices the retention "
        "tradeoff)\n"
        "    2. route traffic under it, then re-export (--mode text) and retrain\n"
        "  Without --replay, graduate compares the student to the teacher's LOGGED answers, which cannot "
        "detect\n"
        "  a teacher that has moved since the log was written. Everything else in this report still stands."
    )


async def _replay_rows(router: Any, rows: Sequence[Any], artifact: Any, policy: Any) -> list[dict[str, Any]]:
    """Route each held-out prompt through the live teacher and put the student next to it."""
    from .evaluate import replay_row

    out: list[dict[str, Any]] = []
    try:
        for row in rows:
            text = str(row.text or "")
            decision = await router.route_text(text)
            student = artifact.answers_for(text=text)
            out.append(
                {
                    "request_id": row.request_id,
                    "logged_tier": str((row.teacher or {}).get("tier") or ""),
                    "live_tier": decision.tier,
                    "student_tier": replay_row(policy, row, student).tier,
                    "live_backend": decision.backend,
                    "live_model_version": decision.backend_model_version,
                    "live_degraded": bool(decision.degraded),
                    "argmax": {
                        head: getattr(student, head).choice == getattr(decision.answers, head).choice
                        for head in ("complexity", "sensitivity", "domain")
                    },
                }
            )
    finally:
        await router.aclose()
    return out


def _replay_against_teacher(
    args: argparse.Namespace,
    *,
    load_router: Callable[..., Any],
    artifact: Any,
    dataset_path: Path,
) -> dict[str, Any]:
    """The stronger check: ask the CURRENT teacher, not the log, and measure drift."""
    from ..cache import NullCache
    from ..logging_sink import NullSink
    from .export import load_dataset

    dataset = load_dataset(dataset_path, split="holdout")
    if dataset.mode != "text":
        raise _CliError(_replay_error(dataset.mode))
    rows = [r for r in dataset.rows if str(r.text or "").strip() and not bool((r.gate or {}).get("blocks_backend"))]
    if not rows:
        raise _CliError(
            "no held-out row carries prompt text that reached a backend, so there is nothing to replay. "
            "Gate-blocked rows are never stored as text, by design."
        )
    router, policy, path = load_router(args)
    # A replay is a measurement of the teacher. It must not append rows to the log
    # that is the training dataset, and it must not be answered from a cache it is
    # trying to measure through -- so both are replaced for the duration.
    router.sink = NullSink()
    router.cache = NullCache()
    notes: list[str] = []
    backend_name = str((policy.backend or {}).get("name", "")).lower()
    if backend_name != "jev":
        notes.append(
            f"the live teacher here is backend {backend_name!r} (policy {path}), not jev: this measures drift "
            "against that backend, which is only a teacher-drift check if that backend is the teacher"
        )
    results = asyncio.run(_replay_rows(router, rows, artifact, policy))
    tier_pairs = [(r["student_tier"], r["live_tier"]) for r in results if not r["live_degraded"]]
    tier_matches = sum(1 for student, live in tier_pairs if student == live)
    drift = sum(1 for r in results if r["logged_tier"] and r["live_tier"] != r["logged_tier"])
    argmax: dict[str, list[int]] = {head: [0, 0] for head in ("complexity", "sensitivity", "domain")}
    for entry in results:
        for head, agreed in entry["argmax"].items():
            argmax[head][1] += 1
            argmax[head][0] += int(bool(agreed))
    versions = sorted({str(r["live_model_version"]) for r in results if r["live_model_version"]})
    return {
        "rows": len(results),
        "excluded_gate_blocked": len(dataset.rows) - len(rows),
        "tier_agreement": (tier_matches / len(tier_pairs)) if tier_pairs else None,
        "tier_counts": (tier_matches, len(tier_pairs)),
        "teacher_drift_rows": drift,
        "teacher_drift_rate": (drift / len(results)) if results else None,
        "argmax_agreement": {head: (hit / total if total else None) for head, (hit, total) in argmax.items()},
        "live_backend": backend_name,
        "live_model_versions": versions,
        "policy": str(path),
        "notes": notes,
    }


def _apply_replay(checks: Sequence[_Check], replay: Mapping[str, Any], args: argparse.Namespace) -> list[_Check]:
    """Swap the tier-agreement gate onto the live teacher's answer, which is the stricter one."""
    floor = float(args.min_tier_agreement)
    out: list[_Check] = []
    for check in checks:
        if check.name != "min_tier_agreement":
            out.append(check)
            continue
        tier = replay.get("tier_agreement")
        counts = replay.get("tier_counts") or (0, 0)
        out.append(
            _Check(
                check.name,
                check.requirement,
                _pct(tier),
                bool(tier is not None and float(tier) >= floor),
                f"measured against the LIVE teacher: {counts[0]}/{counts[1]} rows "
                f"(offline, against the log, it was {check.measured})",
            )
        )
    return out


def _print_replay(replay: Mapping[str, Any]) -> None:
    print()
    print(
        f"live replay      : {replay['rows']} held-out prompt(s) re-routed through "
        f"{replay['live_backend']} ({', '.join(replay['live_model_versions']) or 'version unknown'})"
    )
    print(
        f"  tier agreement : {_pct(replay['tier_agreement'])} against the live teacher "
        f"(offline, against the log: see the check above)"
    )
    print(
        f"  teacher drift  : {replay['teacher_drift_rows']}/{replay['rows']} rows now land on a different "
        f"tier than the log recorded ({_pct(replay['teacher_drift_rate'])})"
    )
    print("  argmax         : " + "  ".join(f"{h} {_pct(v)}" for h, v in sorted(replay["argmax_agreement"].items())))
    if replay["excluded_gate_blocked"]:
        print(
            f"  excluded       : {replay['excluded_gate_blocked']} gate-blocked row(s) -- the router never "
            "calls a backend for those"
        )
    for note in replay["notes"]:
        print(f"  note           : {note}")


def _teacher_name(artifact: Any) -> str:
    """How to name the teacher in a sentence. Read off the artifact, never guessed."""
    teacher = artifact.metadata.get("teacher") if isinstance(artifact.metadata.get("teacher"), Mapping) else {}
    name = str((teacher or {}).get("name") or "")
    versions = [str(v) for v in (artifact.teacher_model_versions or ())]
    if name:
        return "Jev" if name == "jev" else name
    return versions[0] if versions else "the teacher"


# The four gate defaults, copied from the parser in jev_route.cli and from the table
# in docs/graduation.md. They are repeated here rather than read out of the parser on
# purpose: the point of the comparison below is to notice when the command line moved
# away from the documented bar, so the bar has to be a constant this module states, not
# a value looked up from the thing being compared.
GATE_DEFAULTS: Final[tuple[tuple[str, float, str], ...]] = (
    ("min_tier_agreement", 0.95, "min"),
    ("max_ece", 0.10, "max"),
    ("max_latency_ms", 50.0, "max"),
    ("min_samples", float(DEFAULT_MIN_SAMPLES), "min"),
)


def _relaxed_gates(args: argparse.Namespace) -> list[str]:
    """Which gates this invocation relaxed away from the documented bar.

    ``graduate`` exists to be the one command whose verdict an operator can act on
    without reading the code, and every gate is a flag. That combination has one bad
    outcome: ``--min-tier-agreement 0.4`` prints ``VERDICT: READY`` in exactly the
    same type as the honest verdict, and the difference is only visible in the
    requirement column three lines up. So a relaxation is never silent -- it is
    reported next to the verdict, in both the human and the ``--json`` output.

    Relaxed means *more permissive*: a lower floor for a ``min`` gate, a higher
    ceiling for a ``max`` gate. Tightening a gate (``--max-ece 0.05``) is not
    reported, because it cannot manufacture a pass.
    """
    out: list[str] = []
    for name, default, direction in GATE_DEFAULTS:
        value = float(getattr(args, name, default))
        if direction == "min" and value < default:
            fmt = ".0f" if name == "min_samples" else ".2f"
            out.append(f"{name}: {value:{fmt}} on the command line, {default:{fmt}} in the docs (floor lowered)")
        elif direction == "max" and value > default:
            fmt = ".1f" if name == "max_latency_ms" else ".2f"
            out.append(f"{name}: {value:{fmt}} on the command line, {default:{fmt}} in the docs (ceiling raised)")
    return out


def _print_relaxed_gates(relaxed: Sequence[str]) -> None:
    """The warning that keeps a relaxed READY verdict from looking like a promotion."""
    if not relaxed:
        return
    print()
    print(f"WARNING: {len(relaxed)} of the 4 gates were relaxed on the command line, so the verdict above was")
    print("         measured against your flags, not against the documented graduation bar:")
    for line in relaxed:
        print(f"           - {line}")
    print("         A READY verdict under relaxed gates is NOT a promotion. Re-run with no threshold")
    print("         flags for the honest verdict; docs/graduation.md explains what each default buys.")
    if any(line.startswith("min_samples") for line in relaxed):
        print(
            "         min_samples in particular: a tier agreement measured on a few dozen rows is noise, "
            "and the docs say so explicitly."
        )


def _print_graduation(
    args: argparse.Namespace,
    *,
    artifact: Any,
    policy_path: Path,
    dataset_path: Path,
    report: Any,
    checks: Sequence[_Check],
    ready: bool,
    replay: Mapping[str, Any] | None,
) -> None:
    """The report ``graduate`` exists to print: the four gates, then what to do."""
    teacher = _teacher_name(artifact)
    print("graduation readiness")
    print("====================")
    print(f"artifact         : {artifact.source}")
    print(
        f"model_version    : {artifact.model_version}  ({artifact.mode} mode, {artifact.model.n_parameters} parameters)"
    )
    print(f"teacher          : {teacher} ({', '.join(artifact.teacher_model_versions) or 'version unknown'})")
    print(f"policy           : {policy_path}")
    print(f"dataset          : {dataset_path} (holdout split)")
    print()
    print(f"your local model agrees with {teacher} on {_pct(report.tier_agreement)} of held-out decisions,")
    print(
        f"ECE={_num(report.worst_ece)} (worst head: {report.worst_ece_head or 'n/a'}), "
        f"estimated added latency={_num(report.latency.p50, '.3f')}ms "
        f"(the teacher costs {_num(report.latency.teacher.get('p50'), '.1f')}ms)"
    )
    print()
    _print_checks(checks)
    print()
    _print_student_vs_teacher(report)
    tier = report.tier_agreement
    if tier is not None and tier < 0.9 and report.n_rows:
        print()
        print(
            f"plainly: on these {report.n_rows} held-out decisions the student would send "
            f"{1.0 - tier:.0%} of them somewhere other than where {teacher} sent them. "
            "That is not a cutover, and no threshold change makes it one."
        )
    if replay is not None:
        _print_replay(replay)
    failed = [c for c in checks if not c.passed]
    relaxed = _relaxed_gates(args)
    print()
    if ready:
        suffix = " under the gates you set" if relaxed else ""
        print(f"VERDICT: READY -- all four checks passed{suffix}.")
        if not args.write:
            destination = f" --out-policy {args.out_policy}" if getattr(args, "out_policy", None) else ""
            print("  Nothing was changed. Re-run with --write to perform the swap as a config change:")
            print(f"    jev-route graduate --artifact {args.artifact} --policy {policy_path}{destination} --write")
            print("  It writes a NEW policy file by default and backs up anything it overwrites.")
        _print_relaxed_gates(relaxed)
        return
    print(f"VERDICT: NOT READY -- {len(failed)} of {len(checks)} checks failed ({', '.join(c.name for c in failed)}).")
    _print_relaxed_gates(relaxed)
    for line in _recommendation(report, artifact, checks, args):
        print(f"  - {line}")
    print()
    print("recommendation: leave `backend.name` as it is and keep logging. The cloud phase is training-data")
    print("                collection; re-run this command when the holdout has grown. Nothing was written.")


def _graduate(args: argparse.Namespace, *, load_router: Callable[..., Any], find_policy: Callable[..., Any]) -> int:
    """Report readiness, and only then (with ``--write``) cut over.

    Router track by default: the distilled routing model against four thresholds,
    then the backend swap. Gate track (``--track gate``): the semantic layer
    against the promotion criteria, then the mode flip to enforce.
    """
    track = str(getattr(args, "track", "router") or "router")
    if track == "gate":
        return _graduate_gate(args)
    if str(getattr(args, "stage", None) or ""):
        raise _CliError("--stage is a gate-track option (the live semantic-layer gate); rerun with --track gate")

    from .artifact import load_artifact

    artifact = load_artifact(args.artifact)
    # `find_policy` is injected by jev_route.cli and, like every verb there, falls
    # back to the shipped default when the path it was handed does not exist. For
    # graduate that fallback is the wrong default: every gate below is measured
    # *under a policy*, so grading under a different policy than the operator named
    # yields a verdict that looks like the one they asked for and is not. An explicit
    # --policy is therefore resolved strictly here; the injected finder (which also
    # honours JEV_ROUTE_POLICY) is only used when nothing was named.
    policy_path = _resolve_policy_arg(args.policy) or Path(find_policy(None))
    policy = _load_policy(policy_path)
    dataset_path = _resolve_dataset_path(args.data, artifact, verb="graduate")
    # Always re-measured, even when the artifact carries a metrics.json from
    # `package`: a cutover decision should not rest on a number whose provenance
    # is a file that may have been copied from another machine or another dataset.
    report = _measure(artifact, dataset_path, policy_path, 10)
    replay = (
        _replay_against_teacher(args, load_router=load_router, artifact=artifact, dataset_path=dataset_path)
        if args.replay
        else None
    )
    checks = _checks_from_report(report, args)
    if replay is not None:
        checks = _apply_replay(checks, replay, args)
    ready = all(check.passed for check in checks)

    if args.json:
        _print_json(
            {
                "artifact": str(artifact.source),
                "model_version": artifact.model_version,
                "mode": artifact.mode,
                "teacher_model_versions": list(artifact.teacher_model_versions),
                "policy": str(policy_path),
                "dataset": str(dataset_path),
                "ready": ready,
                "checks": [
                    {
                        "name": c.name,
                        "requirement": c.requirement,
                        "measured": c.measured,
                        "passed": c.passed,
                        "detail": c.detail,
                    }
                    for c in checks
                ],
                "summary": {
                    "tier_agreement": report.tier_agreement,
                    "ceiling_tier_agreement": report.agreement.ceiling_tier_agreement if report.agreement else None,
                    "worst_ece": report.worst_ece,
                    "worst_ece_head": report.worst_ece_head,
                    "mean_macro_f1": sum(h.macro_f1 for h in report.heads.values() if h.n)
                    / max(1, sum(1 for h in report.heads.values() if h.n)),
                    "student_p50_ms": report.latency.p50,
                    "teacher_p50_ms": report.latency.teacher.get("p50"),
                    "held_out_rows": report.n_rows,
                },
                "gates_relaxed": _relaxed_gates(args),
                "replay": dict(replay) if replay else None,
                "report": report.to_dict(),
                "wrote": None,
            }
        )
        return EXIT_OK if ready else EXIT_NOT_READY

    _print_graduation(
        args,
        artifact=artifact,
        policy_path=policy_path,
        dataset_path=dataset_path,
        report=report,
        checks=checks,
        ready=ready,
        replay=replay,
    )
    if not ready:
        return EXIT_NOT_READY
    if not args.write:
        return EXIT_OK
    _print_swap(_perform_swap(args, artifact, policy, policy_path), policy_path)
    return EXIT_OK


def _gate_swap(
    args: argparse.Namespace,
    policy: Any,
    policy_path: Path,
    *,
    artifact_path: str,
    shadow_metrics: dict[str, Any] | None,
) -> tuple[Path, Path | None]:
    """Flip ``gate.semantic`` to enforce. A config change, not a code change.

    Same destination rules as the router swap (a new file unless ``--in-place``),
    same backup before overwrite, and the same reason: a cutover file that cannot
    be re-read is a cutover that happened to nobody.

    The flip carries its own evidence. The enforce layer re-checks promotion at
    construction, and the live criteria (shadow example counts, semantic-miss
    rate) only exist in the shadow observation -- so a policy that flips to
    enforce without naming that observation passes the CLI's check and then
    refuses to start, which is a cutover that looks done and is not. The shadow
    metrics are written next to the policy and pointed at, so the cutover file
    and its evidence move together.
    """
    raw_gate = dict((getattr(policy, "raw", None) or {}).get("gate") or {})
    semantic = dict(raw_gate.get("semantic") or {})
    semantic["mode"] = "enforce"
    # The policy may not name the artifact yet (shadow with no model trained is
    # a legal state); enforce without one refuses to construct, so pin the path
    # the operator just graduated.
    semantic.setdefault("artifact", artifact_path)
    if shadow_metrics is not None and "shadow_metrics" not in semantic:
        shadow_dest = _destination(args, policy_path).with_name(
            f"{_destination(args, policy_path).stem}-shadow-metrics.json"
        )
        shadow_dest.write_text(json.dumps(shadow_metrics, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        semantic["shadow_metrics"] = str(shadow_dest)
    raw_gate["semantic"] = semantic
    graduated = policy.with_overrides(gate=raw_gate)
    dest = _destination(args, policy_path)
    backup: Path | None = None
    if dest.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = dest.with_name(f"{dest.name}.bak-{stamp}")
        shutil.copy2(dest, backup)
    graduated.write_yaml(dest)
    return dest, backup


def _verify_gate_policy(dest: Path) -> str:
    """Prove the flip: reload the policy and construct the layer in enforce mode.

    Policy parsing only checks the shape; the substance -- the artifact loads and
    its metrics earn enforce -- is re-checked at layer construction, which is
    exactly the moment a router would hit it. Running it here means the cutover
    file cannot be one that refuses to start.
    """
    from ..gate_semantic import SemanticLayer

    try:
        reloaded = _load_policy(dest)
    except Exception as exc:  # a policy that does not parse is the one thing we must not hand back
        return f"FAIL: {dest} does not parse as a policy: {type(exc).__name__}: {exc}"
    if str(reloaded.semantic_gate.mode) != "enforce":
        return f"FAIL: {dest} parses but gate.semantic.mode is {reloaded.semantic_gate.mode!r}, not 'enforce'"
    try:
        layer = SemanticLayer(reloaded.semantic_gate)
    except Exception as exc:
        return f"FAIL: the enforce layer refuses to construct here: {exc}"
    source = layer.artifact.source if layer.artifact is not None else "(none)"
    return f"{dest} parses, mode is enforce, and the layer constructs (artifact {source} loaded)"


def _gate_evidence(args: argparse.Namespace, artifact: Any) -> tuple[dict[str, Any], dict[str, Any] | None, str]:
    """The artifact's holdout metrics, overlaid with live shadow numbers when supplied.

    The overlay is the library's (``merge_metrics``): shadow wins where both speak,
    and a model-version mismatch is an error, not an average. The raw shadow dict
    is returned as well, because a cutover policy has to carry that evidence with
    it (see :func:`_gate_swap`) or the enforce layer cannot re-derive it.
    """
    from ..gate_semantic import load_shadow_metrics, measure_shadow_log, merge_metrics

    metrics = dict(artifact.metrics)
    shadow_metrics: dict[str, Any] | None = None
    shadow_source = "none (offline holdout only)"
    if getattr(args, "shadow_metrics", None):
        if getattr(args, "log", None):
            raise _CliError("pass either --shadow-metrics or --log, not both")
        shadow_metrics = load_shadow_metrics(args.shadow_metrics)
        shadow_source = f"file {args.shadow_metrics}"
    elif getattr(args, "log", None):
        path = Path(args.log)
        if not path.is_file():
            raise _CliError(f"decision log not found: {path}")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        shadow_metrics = measure_shadow_log(rows)
        shadow_source = f"measured from {path} ({shadow_metrics.get('shadow_n_examples')} assessed records)"
    if shadow_metrics:
        metrics = merge_metrics(metrics, shadow_metrics, model_version=artifact.model_version)
    return metrics, shadow_metrics, shadow_source


def _print_gate_checks(report: Any) -> None:
    """The verdict. On failure it is the loud refusal: every number, what to do about it."""
    if report.ok:
        for check in report.checks:
            print(f"  ok   {check.line().strip()}")
        print("  READY: every promotion check passed.")
    else:
        print(report.message().strip())


def _resolve_live_window(args: argparse.Namespace, policy: Any) -> Path:
    """The live shadow window: ``--shadow-window``, else the policy's ``shadow_window`` key.

    Top-level policy key on purpose: the ``gate.semantic`` section is parsed
    strictly (an unknown key there fails the deploy), and this window is runtime
    data -- a rolling, gitignored file -- not part of the gate's config surface.
    """
    explicit = getattr(args, "shadow_window", None)
    if explicit:
        return Path(str(explicit))
    raw = getattr(policy, "raw", None) or {}
    configured = str(raw.get("shadow_window") or "").strip()
    if not configured:
        raise _CliError(
            "no live shadow window named. The live gate measures the rolling window the "
            "layer's ShadowMetrics collector appends to (one line per decided request); "
            "name it in the policy (top-level `shadow_window:` key) or pass "
            "--shadow-window <window.jsonl>."
        )
    return Path(configured)


def _resolve_injection_eval(args: argparse.Namespace, policy: Any) -> Path:
    """The injection-eval result file: ``--injection-eval-result``, else the policy key."""
    explicit = getattr(args, "injection_eval_result", None)
    if explicit:
        return Path(str(explicit))
    raw = getattr(policy, "raw", None) or {}
    configured = str(raw.get("injection_eval_result") or "").strip()
    if not configured:
        raise _CliError(
            "no injection-eval result named. The enforce stage will not promote on a stale or "
            "absent injection eval; regenerate it (evals/injection/run.py) and name the result "
            "file in the policy (top-level `injection_eval_result:` key) or pass "
            "--injection-eval-result <result.json>."
        )
    return Path(configured)


def _injection_eval_check(path: Path) -> tuple[bool, str, dict[str, Any]]:
    """Reduce the injection-eval result file to one machine-checkable criterion.

    The file is JSON with ``leaks`` and ``false_positives`` (an int, or a list of
    case records, either way). When the semantic-enforce columns were run, their
    ``sem_leaks`` / ``sem_false_positives`` are checked too: the layer being
    promoted is exactly what those columns measure, and a promotion that ignored
    them would be graded on a different system than the one being let through.
    """

    def fail(reason: str) -> tuple[bool, str, dict[str, Any]]:
        return False, f"{reason}: {path}", {"present": False, "path": str(path)}

    if not path.is_file():
        return fail("missing file")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return fail(f"unreadable ({exc})")
    if not isinstance(raw, Mapping):
        return fail("not a JSON object")

    def count(key: str) -> int | None:
        if key not in raw:
            return None
        value = raw[key]
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, (list, tuple)):
            return len(value)
        return None

    leaks = count("leaks")
    false_positives = count("false_positives")
    if leaks is None or leaks < 0 or false_positives is None or false_positives < 0:
        return fail("`leaks` and `false_positives` (int or list) are required")
    sem_leaks = count("sem_leaks") if "sem_leaks" in raw else None
    sem_false_positives = count("sem_false_positives") if "sem_false_positives" in raw else None
    total_leaks = leaks + (sem_leaks or 0)
    total_false_positives = false_positives + (sem_false_positives or 0)
    ok = total_leaks == 0 and total_false_positives == 0
    return (
        ok,
        f"{total_leaks} leaks / {total_false_positives} FP in {path}",
        {
            "present": True,
            "path": str(path),
            "leaks": leaks,
            "false_positives": false_positives,
            "sem_leaks": sem_leaks,
            "sem_false_positives": sem_false_positives,
        },
    )


def _window_evidence(artifact: Any, status: Any, window_path: Path) -> dict[str, Any]:
    """The live window reduced to the shape a cutover policy carries as shadow evidence.

    The enforce layer re-runs the promotion check at construction, overlaying this
    on the artifact's holdout metrics via ``merge_metrics`` -- so the model version
    must match, and the live disagreement becomes what the disagreement bar then
    measures. ``shadow_n_examples`` is the larger of the two counts: the window is
    an observation on top of the artifact's shadow period, not a replacement for
    it, and the cutover file must not make the re-check stricter than the check
    the gate just passed.
    """
    agreement = status.agreement_rate
    return {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "shadow_model_version": str(artifact.model_version),
        "shadow_n_examples": max(status.n_decisions, int(artifact.metrics.get("shadow_n_examples") or 0)),
        "disagreement_rate": round(1.0 - agreement, 6) if agreement is not None else None,
        "source": f"live shadow window {window_path} (jev_route.shadow_metrics)",
    }


def _live_window_checks(status: Any, eval_path: Path, *, policy_backend_model: str | None = None) -> tuple[list[Any], dict[str, Any]]:
    """The four live criteria, as machine-checkable lines, added on top of check_promotion."""
    from ..gate_semantic import PromotionCheck
    from ..shadow_metrics import DRIFT_FACTOR, MIN_AGREEMENT_RATE

    checks: list[Any] = [
        PromotionCheck(
            "window_complete",
            f">= {status.required_decisions} decisions and >= {status.required_days:g} days",
            f"{status.n_decisions} decisions / {status.days_covered:.1f} days",
            bool(status.window_complete),
        ),
        PromotionCheck(
            "agreement_rate",
            f">= {MIN_AGREEMENT_RATE:g}",
            _num(status.agreement_rate),
            isinstance(status.agreement_rate, (int, float)) and status.agreement_rate >= MIN_AGREEMENT_RATE,
        ),
        PromotionCheck(
            "drift_alarm",
            f"no drift (post-baseline disagreement <= {DRIFT_FACTOR:g}x baseline)",
            "alarm" if status.drift_alarm else "clear",
            not status.drift_alarm,
        ),
    ]
    ok, measured, payload = _injection_eval_check(eval_path)
    checks.append(PromotionCheck("injection_eval", "0 leaks / 0 FP", measured, ok))
    # v0.5 reproducibility bar: promotion is refused on a moving alias. Every
    # metric in the report was measured against one model version; an alias
    # makes them unverifiable the day the alias moves.
    if policy_backend_model is not None:
        pinned = not policy_backend_model.endswith("-latest") and policy_backend_model != ""
        checks.append(PromotionCheck(
            "model_pinned",
            "versioned model (not an alias)",
            policy_backend_model or "(unset)",
            pinned,
        ))
    return checks, payload


def _print_promotion_summary(status: Any, eval_payload: Mapping[str, Any]) -> None:
    """On a passing live gate: the numbers that earned the promotion, in one block."""
    from ..shadow_metrics import ECE_BINS, MIN_AGREEMENT_RATE

    print("PROMOTION SUMMARY: semantic layer shadow -> enforce")
    print(
        f"  live window:    {status.n_decisions} decisions over {status.days_covered:.1f} days "
        f"(required {status.required_decisions} / {status.required_days:g})"
    )
    print(f"  agreement:      {status.agreement_rate:.4f} (bar {MIN_AGREEMENT_RATE:g})")
    print(f"  ece:            {status.ece:.4f} ({ECE_BINS}-bin, shadow probability vs enforce outcome)")
    print(
        f"  drift:          clear (baseline {_num(status.baseline_disagreement)}, "
        f"post-baseline {_num(status.current_disagreement)})"
    )
    print(
        f"  injection eval: {eval_payload.get('leaks', 0)} leaks / {eval_payload.get('false_positives', 0)} FP "
        f"in {eval_payload.get('path')}"
    )


def _gate_header(stage: str, shadow_source: str, window: Path | None) -> tuple[str, str]:
    """The report head and the evidence line for a gate-track run."""
    base = "artifact holdout" + (f" + {shadow_source}" if shadow_source != "none (offline holdout only)" else "")
    if stage == "enforce":
        label = "graduation: gate track (semantic layer, live stage: enforce)"
        return label, base + f" + live shadow window ({window})"
    return "graduation: gate track (semantic layer, shadow -> enforce)", base


def _stage_enforce_report(args: argparse.Namespace, policy: Any, criteria: Any, metrics: dict[str, Any]):
    """The full live promotion gate: ``check_promotion`` plus the live-window requirements.

    Returns ``(report, window_path, window_status, injection_eval_payload)``. The
    criteria themselves stay in ``check_promotion``; this adds the live evidence
    (window complete, agreement, drift, injection eval) on top, one check each.
    The live window is the fresher evidence for the traffic-dependent numbers, so
    where both speak it wins: the disagreement bar then measures the window, and
    the shadow-example count is the sum of the two observations, not the smaller.
    """
    from ..gate_semantic import PromotionReport, check_promotion
    from ..shadow_metrics import ShadowMetrics

    if policy is None:
        raise _CliError(
            "--stage enforce needs a policy: the live window (`shadow_window`) and the "
            "injection-eval result (`injection_eval_result`) are named there. Pass --policy."
        )
    window = _resolve_live_window(args, policy)
    status = ShadowMetrics(window).window_status()
    if status.n_decisions and status.agreement_rate is not None:
        metrics["disagreement_rate"] = round(1.0 - status.agreement_rate, 6)
        metrics["shadow_n_examples"] = max(int(metrics.get("shadow_n_examples") or 0), status.n_decisions)
    report = check_promotion(criteria, metrics)
    eval_path = _resolve_injection_eval(args, policy)
    backend_model = ""
    if policy is not None:
        backend_cfg = policy.backend if isinstance(policy.backend, Mapping) else {}
        backend_model = str(backend_cfg.get("model", ""))
    live_checks, eval_payload = _live_window_checks(status, eval_path, policy_backend_model=backend_model)
    combined = PromotionReport(
        criteria=report.criteria,
        metrics={**metrics, "live_window": status.to_dict(), "injection_eval": eval_payload},
        checks=report.checks + tuple(live_checks),
    )
    return combined, window, status, eval_payload


def _graduate_gate_shadow(args: argparse.Namespace, policy: Any) -> int:
    """``--stage shadow``: how far the live window has come. A report, not a verdict."""
    from ..shadow_metrics import ECE_BINS, MIN_AGREEMENT_RATE, ShadowMetrics

    window = _resolve_live_window(args, policy)
    status = ShadowMetrics(window).window_status()
    if args.json:
        _print_json({"track": "gate", "stage": "shadow", "window": str(window), "status": status.to_dict()})
        return EXIT_OK
    print("graduation: gate track (semantic layer, live stage: shadow)")
    print(f"  window:       {window}")
    print(f"  n_decisions:  {status.n_decisions} (window completes at {status.required_decisions})")
    print(f"  days_covered: {status.days_covered:.1f} (window completes at {status.required_days:g})")
    print(f"  agreement:    {_num(status.agreement_rate)} (promotion bar {MIN_AGREEMENT_RATE:g})")
    print(f"  ece:          {_num(status.ece)} ({ECE_BINS}-bin, shadow probability vs enforce outcome)")
    print(f"  drift_alarm:  {status.drift_alarm}")
    if status.window_complete:
        print("  WINDOW COMPLETE: the live shadow period satisfies the window requirement.")
        _next_step("run with --stage enforce to grade it against the full live gate")
    else:
        missing: list[str] = []
        if status.n_decisions < status.required_decisions:
            missing.append(f"{status.required_decisions - status.n_decisions} more decisions")
        if status.days_covered < status.required_days:
            missing.append(f"{status.required_days - status.days_covered:.1f} more days")
        print(f"  window incomplete: needs {', '.join(missing)}")
    return EXIT_OK


def _cutover_evidence(stage: str, artifact: Any, status: Any, window: Path) -> dict[str, Any]:
    """The shadow evidence the cutover policy carries for an enforce flip.

    On the live enforce stage the window itself IS that evidence: the gate just
    graded it, so the cutover file carries the window's reduced numbers instead
    of requiring a second ``--shadow-metrics`` file. Any other stage still needs
    the explicitly supplied shadow observation.
    """
    if stage != "enforce":
        raise _CliError(
            "--write needs live shadow evidence for the cutover policy to carry; "
            "pass --shadow-metrics or --log. An enforce policy with no shadow observation "
            "refuses to construct, because its live criteria would be unmeasurable."
        )
    return _window_evidence(artifact, status, window)


def _maybe_print_promotion_summary(stage: str, status: Any, eval_payload: Mapping[str, Any]) -> None:
    if stage == "enforce":
        print()
        _print_promotion_summary(status, eval_payload)


def _graduate_gate(args: argparse.Namespace) -> int:  # noqa: PLR0912, PLR0915 -- a promotion gate enumerates criteria by design; splitting the checks would scatter the audit trail
    """Gate track: promote the semantic layer from shadow to enforce -- or refuse, with the numbers.

    The router track graduates the routing model and swaps the backend. The gate
    track graduates the gate itself: ``check_promotion`` over the artifact's
    measured holdout metrics, overlaid with live shadow numbers when supplied,
    against the policy's ``enforce_requires``. The cutover it writes is a config
    flip (``gate.semantic.mode: enforce``), and it is only written when every
    check passes -- a refusal here exits ``1`` with the measured numbers, exactly
    like the router track.

    ``--stage`` selects the live gate. Omitted (the default), this is today's
    behaviour: holdout plus optional shadow-log evidence. ``--stage shadow``
    reports the rolling live window -- how far the shadow period has come --
    without a promotion verdict. ``--stage enforce`` grades the full live
    promotion gate on top of the same criteria: the window must be complete
    (>= 500 decisions over >= 7 days), the agreement with the deterministic layer
    must be >= 0.95, the drift alarm must be clear, and a fresh injection-eval
    result file (path from the policy) must show 0 leaks / 0 FP. Every criterion
    gets one machine-checkable line.
    """
    from ..gate_semantic import EnforceCriteria, SemanticArtifact, check_promotion

    if getattr(args, "replay", False) or getattr(args, "data", None) is not None:
        raise _CliError("--replay and --data are router-track options; rerun with --track router")
    stage = str(getattr(args, "stage", None) or "")
    if stage not in ("", "shadow", "enforce"):
        raise _CliError(f"unknown --stage {stage!r}: use `shadow`, `enforce`, or omit it")

    try:
        artifact = SemanticArtifact.load(args.artifact)
    except Exception as exc:
        raise _CliError(f"semantic artifact {args.artifact}: {exc}") from exc
    metrics, shadow_metrics, shadow_source = _gate_evidence(args, artifact)

    # Criteria come from the policy's enforce_requires when a policy resolves,
    # because that is where the operator's bar lives; without one, the built-in
    # defaults, said out loud rather than implied.
    policy_path = _resolve_policy_arg(args.policy)
    criteria = EnforceCriteria()
    criteria_source = "built-in defaults (no policy named)"
    policy = None
    if policy_path is not None:
        policy = _load_policy(policy_path)
        criteria = policy.semantic_gate.enforce_requires
        criteria_source = f"enforce_requires in {policy_path}"

    if stage == "shadow":
        return _graduate_gate_shadow(args, policy)
    if stage == "enforce":
        report, window, status, eval_payload = _stage_enforce_report(args, policy, criteria, metrics)
    else:
        report, window, status, eval_payload = check_promotion(criteria, metrics), None, None, {}

    if args.json:
        payload: dict[str, Any] = {
            "track": "gate",
            "artifact": str(artifact.source),
            "model_version": artifact.model_version,
            "criteria": criteria_source,
            "shadow": shadow_source,
            "metrics": metrics,
            "report": report.to_dict(),
            "wrote": None,
        }
        if stage == "enforce":
            payload.update(
                {
                    "stage": "enforce",
                    "live_window": str(window),
                    "live_window_status": status.to_dict(),
                    "injection_eval": eval_payload,
                }
            )
        _print_json(payload)
        return EXIT_OK if report.ok else EXIT_NOT_READY

    header, evidence = _gate_header(stage, shadow_source, window)
    print(header)
    print(f"  artifact:   {artifact.source}  (model_version {artifact.model_version})")
    print(f"  criteria:   {criteria_source}")
    print(f"  evidence:   {evidence}")
    print()
    _print_gate_checks(report)
    if not report.ok:
        return EXIT_NOT_READY
    _maybe_print_promotion_summary(stage, status, eval_payload)
    if not args.write:
        _next_step("ready: rerun with --write to flip the policy to gate.semantic.mode: enforce")
        return EXIT_OK

    if policy is None:
        raise _CliError("--write on the gate track needs a policy to flip; pass --policy")
    if shadow_metrics is None:
        shadow_metrics = _cutover_evidence(stage, artifact, status, window)
    dest, backup = _gate_swap(
        args, policy, policy_path, artifact_path=str(args.artifact), shadow_metrics=shadow_metrics
    )
    verification = _verify_gate_policy(dest)
    print()
    print(f"wrote: {dest}" + (f"  (backup: {backup})" if backup else ""))
    print(f"verified: {verification}")
    if verification.startswith("FAIL"):
        # The file exists and the operator may want it -- but exit 0 here would
        # tell automation that a cutover happened when the router will refuse it.
        return EXIT_ERROR
    _next_step("the router re-checks promotion at construction; a stale artifact refuses to start in enforce mode.")
    return EXIT_OK


def _build_label_backend(backend: str) -> Any:
    """The teacher for label-sensitivity. mock is the offline default; jev is the real one."""
    if backend == "mock":
        from ..backends.mock import MockBackend

        return MockBackend()
    key = os.environ.get("TYPESAFE_API_KEY", "")
    if not key.strip():
        raise _CliError(
            "--backend jev needs TYPESAFE_API_KEY in the environment (never in this file). "
            "For an offline dry run that is clearly labelled as mock in the stats, rerun with --backend mock."
        )
    from ..backends.jev import JevBackend

    return JevBackend()


def _label_sensitivity(args: argparse.Namespace) -> int:
    """Build and label a sensitivity dataset: allowed sources only, every call audited.

    The library (``sensitivity_data``) is the privacy boundary -- the closed
    source list, the sendability checks, the canary-tested refusal of blocked
    content. This verb only parses arguments, picks the teacher, and prints what
    the library measured. Nothing here decides what may be sent; that decision
    has its own module, its own tests, and its own refusal reasons.
    """
    from .sensitivity_data import synthesize_sensitivity_dataset

    kinds = None
    raw_kinds = getattr(args, "kinds", None)
    if raw_kinds:
        kinds = tuple(k.strip() for k in str(raw_kinds).split(",") if k.strip())
    backend = _build_label_backend(str(args.backend))
    out = Path(args.out)
    audit_path = Path(args.audit) if getattr(args, "audit", None) else out / "labeling-audit.jsonl"
    result = asyncio.run(
        synthesize_sensitivity_dataset(
            out_path=out,
            backend=backend,
            seed=int(args.seed),
            per_kind=int(args.per_kind),
            kinds=kinds,
            contextual_per_category=int(args.contextual),
            production_log=Path(args.production_log) if getattr(args, "production_log", None) else None,
            production_limit=getattr(args, "production_limit", None),
            include_blocked_metadata=bool(getattr(args, "include_blocked_metadata", False)),
            mode=str(args.mode),
            audit_path=audit_path,
            exported_at=getattr(args, "exported_at", None),
        )
    )
    stats = result.stats
    if args.json:
        _print_json(
            {
                "rows": str(result.rows_path),
                "stats": str(result.stats_path),
                "audit": str(audit_path),
                **stats.as_dict(),
            }
        )
        return EXIT_OK

    print(f"sensitivity dataset: {result.rows_path}")
    print(
        f"  rows: {stats.rows_written} "
        f"(train {stats.train_rows} / holdout {stats.holdout_rows}, holdout_fraction {stats.holdout_fraction})"
    )
    print(f"  mode: {stats.mode}   contains_prompt_text: {stats.contains_prompt_text}")
    print("  sources:")
    for source, count in sorted(stats.source_breakdown.items()):
        print(f"    {source}: {count}")
    print("  labeling:")
    print(
        f"    backend: {stats.labeling_backend}   calls: {stats.labeling_calls}   "
        f"refusals: {stats.labeling_refusals}"
    )
    for reason, count in sorted(stats.refusal_reasons.items()):
        print(f"    refused {reason}: {count}")
    if getattr(args, "production_log", None):
        print(
            f"    production: read {stats.production_records_read}, "
            f"blocked-discarded {stats.production_blocked_discarded}, "
            f"blocked-metadata-only {stats.production_blocked_metadata_only}, "
            f"degraded-skipped {stats.production_degraded_skipped}"
        )
    print(f"  dataset_sha256: {stats.dataset_sha256}")
    print(f"  stats sidecar: {result.stats_path}")
    print(f"  labeling audit: {audit_path}  (every labeling call, with its source category)")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# the six verbs, as jev_route.cli calls them
# --------------------------------------------------------------------------- #
def export_cli(args: argparse.Namespace) -> int:
    """``jev-route export``: decision log -> training dataset."""
    return _guard("export", lambda: _export(args))


def train_cli(args: argparse.Namespace) -> int:
    """``jev-route train``: soft-target KL distillation -> an artifact directory."""
    return _guard("train", lambda: _train(args))


def evaluate_cli(args: argparse.Namespace) -> int:
    """``jev-route evaluate``: accuracy, ECE, teacher agreement and latency on the holdout."""
    return _guard("evaluate", lambda: _evaluate(args))


def package_cli(args: argparse.Namespace) -> int:
    """``jev-route package``: a trained model -> a loadable, self-describing artifact."""
    return _guard("package", lambda: _package(args))


def graduate_cli(
    args: argparse.Namespace,
    *,
    load_router: Callable[..., Any],
    find_policy: Callable[..., Any],
) -> int:
    """``jev-route graduate``: readiness, then the config-only cutover.

    Two tracks (``--track``): ``router`` (default) grades the distilled routing
    model against four thresholds and swaps the backend; ``gate`` grades the
    semantic layer against the promotion criteria and flips
    ``gate.semantic.mode`` to enforce. Both refuse with the measured numbers
    (exit 1) rather than writing a cutover that is not earned.

    ``load_router`` and ``find_policy`` are injected by :mod:`jev_route.cli` rather
    than imported, so this package never depends on the module that depends on it.
    ``find_policy`` resolves ``--policy`` (with the usual fallbacks) and ``load_router``
    builds a live Router for ``--replay``.
    """
    return _guard("graduate", lambda: _graduate(args, load_router=load_router, find_policy=find_policy))


def label_sensitivity_cli(args: argparse.Namespace) -> int:
    """``jev-route label-sensitivity``: build and label a sensitivity dataset."""
    return _guard("label-sensitivity", lambda: _label_sensitivity(args))


__all__ = ["evaluate_cli", "export_cli", "graduate_cli", "label_sensitivity_cli", "package_cli", "train_cli"]
