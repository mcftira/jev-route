"""The distilled-model artifact: its on-disk format, and the objects it carries.

This module owns three things that have to agree exactly, which is why they live
in one file:

1. **The vectorizer** that turns a request into numbers. Two kinds, matching the
   two export modes: ``tfidf`` (word/bigram counts over the redacted excerpt,
   hashed into a stored vocabulary) and ``features`` (a fixed vector built from
   :class:`~jev_route.schema.RequestFeatures`, for operators who never retain
   prompt text).
2. **The student** -- a small numpy network with one shared trunk and four output
   heads (complexity, sensitivity, domain, pii). Softmax heads plus one Bernoulli
   head, in a fixed order, because the order is part of the file format.
3. **The envelope**: ``artifact.json``, one ``<param>.npy`` per weight matrix
   (``W.npy``, ``b.npy`` for the linear student), ``vectorizer.json``,
   ``metrics.json`` and ``dataset.json`` in a directory, with SHA-256 checksums,
   validated on load.

Two deliberate choices a sceptical reader will look for:

* **No pickle, anywhere.** The ``.npy`` weight files are raw arrays and the rest
  is JSON. A pickle in an artifact directory is arbitrary code execution the moment
  someone loads a file they downloaded, and it silently rots across library versions.
  This format can be read by ``json.load`` and ``numpy.load`` and nothing else.
* **Loading validates.** The head label sets are checked against the live
  ``jev_route.schema`` ladders, the artifact schema version against the set this
  build understands, and the weight checksum against the envelope. An artifact
  trained against an older schema raises :class:`ArtifactError` naming the
  mismatch rather than mis-predicting quietly, which is the failure mode that
  would otherwise route regulated data to a cloud model.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from ..schema import (
    COMPLEXITY_LEVELS,
    DOMAINS,
    SCHEMA_VERSION,
    SENSITIVITY_LEVELS,
    ChoiceAnswer,
    DecisionAnswers,
    NoulAnswer,
    RequestFeatures,
    certainty_from_probabilities,
)

#: Bump when the artifact layout changes in a way an older build cannot read.
#: Additive metadata keys do not need a bump; moving or retyping weights does.
ARTIFACT_SCHEMA_VERSION = "1"
SUPPORTED_ARTIFACT_SCHEMA_VERSIONS: frozenset[str] = frozenset({"1"})

#: Marker written into ``artifact.json`` so a directory of JSON can be identified.
ARTIFACT_KIND = "jev_route.distill.artifact"

METADATA_FILE = "artifact.json"
VECTORIZER_FILE = "vectorizer.json"
METRICS_FILE = "metrics.json"
DATASET_FILE = "dataset.json"

#: numpy is a lazy import everywhere in this package. The routing hot path and the
#: core library must not require it; only distillation does.
_NUMPY: Any = None


class ArtifactError(Exception):
    """An artifact that cannot be trusted: missing, corrupt, or schema-mismatched.

    :class:`~jev_route.backends.distilled.DistilledBackend` catches this and
    degrades instead of raising, because a routing decision must never fail
    because a model file is bad -- it must fail *closed*.
    """


class DistillDependencyError(RuntimeError):
    """A required extra is not installed. Always carries the pip incantation."""


def require_numpy() -> Any:
    """Import numpy on first use, with an actionable error when it is missing.

    Lazy on purpose: ``jev_route`` imports cleanly (and the router runs on Jev or
    the mock) with no numerical stack installed at all.
    """
    global _NUMPY
    if _NUMPY is None:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise DistillDependencyError(
                "distillation needs numpy. Install it with:  pip install 'jev-route[distill]'"
            ) from exc
        _NUMPY = np
    return _NUMPY


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
# Sparse matrix
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SparseMatrix:
    """CSR-style sparse matrix over numpy arrays, without scipy.

    scipy would be the obvious choice, but it is a heavy dependency for something
    this small, and CSR is eleven lines. Rows are variable length; ``indptr`` has
    ``n_rows + 1`` entries. Keeping the dataset sparse is what lets a 50k-row
    text export sit in a few tens of megabytes instead of a dense
    ``n_rows x vocab`` block.
    """

    data: Any  # np.ndarray float32, shape (nnz,)
    indices: Any  # np.ndarray int32, shape (nnz,)
    indptr: Any  # np.ndarray int32, shape (n_rows + 1,)
    n_features: int

    @property
    def n_rows(self) -> int:
        return int(len(self.indptr) - 1)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_rows, self.n_features)

    @property
    def nnz(self) -> int:
        return len(self.data)

    def __len__(self) -> int:
        return self.n_rows

    def rows(self, start: int, stop: int) -> SparseMatrix:
        """A view of rows ``[start, stop)``. ``indptr`` is rebased to zero."""
        start = max(0, min(start, self.n_rows))
        stop = max(start, min(stop, self.n_rows))
        lo = int(self.indptr[start])
        hi = int(self.indptr[stop])
        return SparseMatrix(
            data=self.data[lo:hi],
            indices=self.indices[lo:hi],
            indptr=self.indptr[start : stop + 1] - lo,
            n_features=self.n_features,
        )

    def row(self, index: int) -> SparseMatrix:
        return self.rows(index, index + 1)

    def row_items(self, index: int) -> tuple[tuple[int, float], ...]:
        lo, hi = int(self.indptr[index]), int(self.indptr[index + 1])
        # CSR invariants: `indices` and `data` are parallel arrays, so the two
        # slices are the same length by construction. strict=True turns a broken
        # invariant into an error instead of silently dropping non-zero entries.
        return tuple((int(i), float(v)) for i, v in zip(self.indices[lo:hi], self.data[lo:hi], strict=True))

    def to_dense(self) -> Any:
        """Materialize as ``(n_rows, n_features)`` float32. Used per mini-batch."""
        np = require_numpy()
        out = np.zeros((self.n_rows, self.n_features), dtype=np.float32)
        if self.nnz:
            row_ids = np.repeat(np.arange(self.n_rows, dtype=np.int64), np.diff(self.indptr))
            out[row_ids, self.indices.astype(np.int64)] = self.data
        return out

    @classmethod
    def from_rows(cls, rows: Sequence[Sequence[tuple[int, float]]], n_features: int) -> SparseMatrix:
        np = require_numpy()
        data: list[float] = []
        indices: list[int] = []
        indptr: list[int] = [0]
        for row in rows:
            for index, value in row:
                if value == 0.0:
                    continue
                if not 0 <= index < n_features:
                    raise ArtifactError(f"feature index {index} outside 0..{n_features - 1}")
                indices.append(int(index))
                data.append(float(value))
            indptr.append(len(data))
        return cls(
            data=np.asarray(data, dtype=np.float32),
            indices=np.asarray(indices, dtype=np.int32),
            indptr=np.asarray(indptr, dtype=np.int32),
            n_features=int(n_features),
        )

    @classmethod
    def from_dense(cls, array: Any) -> SparseMatrix:
        np = require_numpy()
        arr = np.asarray(array, dtype=np.float32)
        if arr.ndim != 2:
            raise ArtifactError(f"expected a 2-D matrix, got shape {arr.shape}")
        rows = [
            tuple((int(j), float(v)) for j, v in enumerate(row) if v != 0.0) for row in arr
        ]
        return cls.from_rows(rows, arr.shape[1])


def _softmax_rows(logits: Any) -> Any:
    """Row-wise softmax, max-subtracted so large logits cannot overflow."""
    np = require_numpy()
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _sigmoid(x: Any) -> Any:
    np = require_numpy()
    # tanh form: numerically stable for large |x| in both directions.
    return 0.5 * (1.0 + np.tanh(np.asarray(x, dtype=np.float64) / 2.0))


# --------------------------------------------------------------------------- #
# Head layout
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HeadSpec:
    """One output head: its name, its label ladder, and its output kind."""

    name: str
    labels: tuple[str, ...]
    kind: Literal["softmax", "bernoulli"]

    @property
    def size(self) -> int:
        return 1 if self.kind == "bernoulli" else len(self.labels)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "labels": list(self.labels), "kind": self.kind}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HeadSpec:
        return cls(name=str(data["name"]), labels=tuple(str(x) for x in data["labels"]), kind=str(data["kind"]))  # type: ignore[arg-type]


def default_head_layout() -> tuple[HeadSpec, ...]:
    """The head layout, taken live from :mod:`jev_route.schema`.

    Read from the schema ladders rather than hardcoded here so that an artifact
    trained before a ladder changed is *detectably* stale instead of silently
    mis-aligned. Order is part of the format: complexity, sensitivity, domain, pii.
    """
    return (
        HeadSpec("complexity", tuple(COMPLEXITY_LEVELS), "softmax"),
        HeadSpec("sensitivity", tuple(SENSITIVITY_LEVELS), "softmax"),
        HeadSpec("domain", tuple(DOMAINS), "softmax"),
        # Bernoulli head. Labels are stated explicitly so the artifact is
        # self-describing even though only one logit is stored.
        HeadSpec("pii", ("false", "true"), "bernoulli"),
    )


CHOICE_HEADS: tuple[str, ...] = ("complexity", "sensitivity", "domain")
ALL_HEADS: tuple[str, ...] = ("complexity", "sensitivity", "domain", "pii")


def validate_head_layout(heads: Sequence[HeadSpec]) -> None:
    """Raise :class:`ArtifactError` unless ``heads`` matches the current schema."""
    expected = default_head_layout()
    if tuple(heads) != expected:
        got = [(h.name, h.labels, h.kind) for h in heads]
        want = [(h.name, h.labels, h.kind) for h in expected]
        raise ArtifactError(
            "artifact head layout does not match jev_route.schema; refusing to load. "
            f"expected {want!r}, got {got!r}. The label ladders changed after this artifact was "
            "trained (or the artifact is from another project), so its outputs would be mapped "
            "onto the wrong labels. Retrain with `jev-route train --data <dataset> --out <artifact>` "
            "against the current schema."
        )


# --------------------------------------------------------------------------- #
# Vectorizers
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_BIGRAM_JOIN = "_"


def tokenize(text: str, *, lower: bool = True, min_token_chars: int = 2, ngram_max: int = 2) -> list[str]:
    """Split into word unigrams plus bigrams.

    Word-level rather than character-level: the routing signal in a prompt is
    topical ("kubernetes", "hipaa", "prove"), and a bigram captures the cheap
    phrases ("stack trace", "pull request") without a real parser. Character
    n-grams would inflate the vocabulary an order of magnitude for little gain on
    a 4-5 way classification.
    """
    body = text.lower() if lower else text
    words = [t for t in _TOKEN_RE.findall(body) if len(t) >= min_token_chars or t.isdigit()]
    if ngram_max < 2:
        return words
    out = list(words)
    for a, b in itertools.pairwise(words):
        out.append(a + _BIGRAM_JOIN + b)
    return out


@runtime_checkable
class Vectorizer(Protocol):
    """Request -> numbers. The artifact stores enough to rebuild it exactly."""

    kind: str
    n_features: int

    def transform(self, items: Sequence[Any]) -> SparseMatrix: ...
    def to_config(self) -> dict[str, Any]: ...


class TextVectorizer:
    """TF-IDF over a stored vocabulary. ``kind == "tfidf"``.

    A stored vocabulary rather than feature hashing, for two reasons an operator
    can verify: the artifact lists the exact tokens the model keys on (readable
    with ``json.load``, auditable without running anything), and there are no
    hash collisions silently merging "hipaa" with "hippo". The cost is unseen
    tokens at serve time, which are dropped -- acceptable because the vocabulary
    is built from the operator's own traffic.
    """

    kind = "tfidf"
    #: Bump when the token math changes. Stored in the artifact and checked.
    config_version = 1

    def __init__(
        self,
        vocab: Mapping[str, int] | None = None,
        idf: Sequence[float] | None = None,
        *,
        max_features: int = 4096,
        ngram_max: int = 2,
        min_token_chars: int = 2,
        sublinear_tf: bool = True,
        l2_norm: bool = True,
        lower: bool = True,
    ) -> None:
        self.vocab: dict[str, int] = dict(vocab or {})
        self.idf: list[float] = [float(x) for x in (idf or [])]
        self.max_features = int(max_features)
        self.ngram_max = int(ngram_max)
        self.min_token_chars = int(min_token_chars)
        self.sublinear_tf = bool(sublinear_tf)
        self.l2_norm = bool(l2_norm)
        self.lower = bool(lower)
        if self.vocab and len(self.idf) != len(self.vocab):
            raise ArtifactError(
                f"tfidf vectorizer has {len(self.vocab)} terms but {len(self.idf)} idf values"
            )

    @property
    def n_features(self) -> int:
        return len(self.vocab)

    def fit(self, texts: Sequence[str]) -> TextVectorizer:
        docs = [
            tokenize(t or "", lower=self.lower, min_token_chars=self.min_token_chars, ngram_max=self.ngram_max)
            for t in texts
        ]
        df: dict[str, int] = {}
        for tokens in docs:
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1
        n_docs = max(1, len(docs))
        # Deterministic selection: most frequent first, ties broken alphabetically.
        # Sorting by document frequency keeps the informative tokens and the
        # alphabetical tiebreak makes the vocabulary independent of row order,
        # so re-running export+train on the same log gives the same model.
        ranked = sorted(df.items(), key=lambda kv: (-kv[1], kv[0]))
        chosen = ranked[: max(1, self.max_features)]
        self.vocab = {token: i for i, (token, _df) in enumerate(chosen)}
        self.idf = [math.log((1.0 + n_docs) / (1.0 + df[token])) + 1.0 for token, _df in chosen]
        return self

    def transform(self, texts: Sequence[str]) -> SparseMatrix:
        if not self.vocab:
            raise ArtifactError("tfidf vectorizer has not been fitted (empty vocabulary)")
        rows: list[list[tuple[int, float]]] = []
        for text in texts:
            counts: dict[int, float] = {}
            for token in tokenize(
                text or "", lower=self.lower, min_token_chars=self.min_token_chars, ngram_max=self.ngram_max
            ):
                index = self.vocab.get(token)
                if index is None:
                    continue  # unseen at training time; dropping it is the documented behaviour
                counts[index] = counts.get(index, 0.0) + 1.0
            row: list[tuple[int, float]] = []
            for index, tf in counts.items():
                weight = (1.0 + math.log(tf)) if (self.sublinear_tf and tf > 0.0) else tf
                row.append((index, weight * self.idf[index]))
            if self.l2_norm and row:
                norm = math.sqrt(sum(v * v for _i, v in row)) or 1.0
                row = [(i, v / norm) for i, v in row]
            row.sort()
            rows.append(row)
        return SparseMatrix.from_rows(rows, len(self.vocab))

    def fit_transform(self, texts: Sequence[str]) -> SparseMatrix:
        return self.fit(texts).transform(texts)

    def to_config(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "config_version": self.config_version,
            "n_features": len(self.vocab),
            "max_features": self.max_features,
            "ngram_max": self.ngram_max,
            "min_token_chars": self.min_token_chars,
            "sublinear_tf": self.sublinear_tf,
            "l2_norm": self.l2_norm,
            "lower": self.lower,
            "vocab": self.vocab,
            "idf": [round(v, 6) for v in self.idf],
        }

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> TextVectorizer:
        version = int(cfg.get("config_version", 1))
        if version != cls.config_version:
            raise ArtifactError(
                f"tfidf vectorizer config_version {version} is not supported by this build "
                f"(expects {cls.config_version}); retrain the artifact"
            )
        return cls(
            vocab={str(k): int(v) for k, v in (cfg.get("vocab") or {}).items()},
            idf=[float(v) for v in (cfg.get("idf") or [])],
            max_features=int(cfg.get("max_features", 4096)),
            ngram_max=int(cfg.get("ngram_max", 2)),
            min_token_chars=int(cfg.get("min_token_chars", 2)),
            sublinear_tf=bool(cfg.get("sublinear_tf", True)),
            l2_norm=bool(cfg.get("l2_norm", True)),
            lower=bool(cfg.get("lower", True)),
        )


#: Numeric block of the features vectorizer: ``(name, extractor)`` in fixed order.
#: The names are stored in the artifact and re-checked on load, so a reordered or
#: renamed :class:`~jev_route.schema.RequestFeatures` field invalidates old
#: artifacts instead of quietly feeding the wrong column to the wrong weight.
_NUMERIC_EXTRACTORS: tuple[tuple[str, Callable[[RequestFeatures], float]], ...] = (
    ("log_char_len", lambda f: math.log1p(max(0, f.char_len))),
    ("log_word_count", lambda f: math.log1p(max(0, f.word_count))),
    ("log_line_count", lambda f: math.log1p(max(0, f.line_count))),
    ("log_sentence_count", lambda f: math.log1p(max(0, f.sentence_count))),
    ("mean_word_len", lambda f: float(f.mean_word_len)),
    ("digit_ratio", lambda f: float(f.digit_ratio)),
    ("upper_ratio", lambda f: float(f.upper_ratio)),
    ("punct_ratio", lambda f: float(f.punct_ratio)),
    ("non_ascii_ratio", lambda f: float(f.non_ascii_ratio)),
    ("log_code_blocks", lambda f: math.log1p(max(0, f.code_blocks))),
    ("log_inline_code_spans", lambda f: math.log1p(max(0, f.inline_code_spans))),
    ("log_urls", lambda f: math.log1p(max(0, f.urls))),
    ("log_question_marks", lambda f: math.log1p(max(0, f.question_marks))),
    ("log_exclamations", lambda f: math.log1p(max(0, f.exclamations))),
    ("log_n_messages", lambda f: math.log1p(max(0, f.n_messages))),
    ("log_n_prior_turns", lambda f: math.log1p(max(0, f.n_prior_turns))),
    ("log_n_gate_findings", lambda f: math.log1p(max(0, f.n_gate_findings))),
    ("has_stack_trace", lambda f: 1.0 if f.has_stack_trace else 0.0),
    ("has_diff", lambda f: 1.0 if f.has_diff else 0.0),
    ("has_json", lambda f: 1.0 if f.has_json else 0.0),
    ("tool_output_present", lambda f: 1.0 if f.tool_output_present else 0.0),
    ("gate_force_local", lambda f: 1.0 if f.gate_force_local else 0.0),
    # Derived ratios. A raw count confounds prompt length with prompt shape; the
    # ratio is what actually distinguishes "a long chat" from "a wall of code".
    ("chars_per_word", lambda f: f.char_len / max(1, f.word_count)),
    ("words_per_line", lambda f: f.word_count / max(1, f.line_count)),
    ("words_per_sentence", lambda f: f.word_count / max(1, f.sentence_count)),
    ("code_density", lambda f: 100.0 * (f.code_blocks + f.inline_code_spans) / max(1, f.word_count)),
    ("question_density", lambda f: 100.0 * f.question_marks / max(1, f.word_count)),
    ("gate_findings_per_kchar", lambda f: 1000.0 * f.n_gate_findings / max(1, f.char_len)),
)


class FeatureVectorizer:
    """Fixed-width vector from :class:`~jev_route.schema.RequestFeatures`. ``kind == "features"``.

    This is the privacy-preserving mode: the inputs are counts, ratios, booleans,
    and detector names, never text. Its ceiling is lower than the text model's --
    it cannot read the prompt, only its shape -- and it cannot see the gate's
    advisory topic hints either, because those are not part of the logged
    ``RequestFeatures``. Both limits are documented rather than papered over: an
    operator who chose not to retain text bought zero text retention with
    accuracy, and should be able to see the size of that trade in
    ``evaluate``'s report.
    """

    kind = "features"
    config_version = 1
    #: Catch-all slot for a language / detector name not seen during training.
    OTHER = "__other__"

    def __init__(
        self,
        *,
        center: Sequence[float] | None = None,
        scale: Sequence[float] | None = None,
        lang_vocab: Sequence[str] = (),
        detector_vocab: Sequence[str] = (),
        numeric_names: Sequence[str] | None = None,
    ) -> None:
        self.numeric_names: tuple[str, ...] = tuple(numeric_names or (n for n, _ in _NUMERIC_EXTRACTORS))
        self.center: list[float] = [float(x) for x in (center or [])]
        self.scale: list[float] = [float(x) for x in (scale or [])]
        self.lang_vocab: tuple[str, ...] = tuple(lang_vocab)
        self.detector_vocab: tuple[str, ...] = tuple(detector_vocab)
        expected = len(self.numeric_names)
        if self.center and len(self.center) != expected:
            raise ArtifactError(f"features vectorizer center has {len(self.center)} values, expected {expected}")
        if self.scale and len(self.scale) != expected:
            raise ArtifactError(f"features vectorizer scale has {len(self.scale)} values, expected {expected}")

    @property
    def n_features(self) -> int:
        return len(self.numeric_names) + len(self.lang_vocab) + 1 + len(self.detector_vocab) + 1

    def _numeric(self, features: RequestFeatures) -> list[float]:
        by_name = dict(_NUMERIC_EXTRACTORS)
        out: list[float] = []
        for name in self.numeric_names:
            extractor = by_name.get(name)
            if extractor is None:
                raise ArtifactError(
                    f"features vectorizer references unknown numeric feature {name!r}; this build "
                    f"knows {sorted(by_name)}. The artifact predates a schema change -- retrain it."
                )
            out.append(float(extractor(features)))
        return out

    def fit(self, feature_rows: Sequence[RequestFeatures]) -> FeatureVectorizer:
        np = require_numpy()
        raw = np.asarray([self._numeric(f) for f in feature_rows], dtype=np.float64) if feature_rows else None
        names = tuple(n for n, _ in _NUMERIC_EXTRACTORS)
        self.numeric_names = names
        if raw is None or raw.shape[0] == 0:
            self.center = [0.0] * len(names)
            self.scale = [1.0] * len(names)
        else:
            self.center = [float(v) for v in raw.mean(axis=0)]
            std = raw.std(axis=0)
            # A constant column gets scale 1.0, not 0.0: dividing by zero would
            # produce NaN weights that only surface at inference time.
            self.scale = [float(s) if s > 1e-9 else 1.0 for s in std]
        self.lang_vocab = tuple(sorted({str(f.lang) for f in feature_rows}))
        detectors: set[str] = set()
        for f in feature_rows:
            detectors.update(str(k) for k in (f.gate_detectors or {}))
        self.detector_vocab = tuple(sorted(detectors))
        return self

    def transform(self, feature_rows: Sequence[RequestFeatures]) -> SparseMatrix:
        n_numeric = len(self.numeric_names)
        lang_index = {name: n_numeric + i for i, name in enumerate(self.lang_vocab)}
        lang_other = n_numeric + len(self.lang_vocab)
        det_base = lang_other + 1
        det_index = {name: det_base + i for i, name in enumerate(self.detector_vocab)}
        det_other = det_base + len(self.detector_vocab)

        rows: list[list[tuple[int, float]]] = []
        for features in feature_rows:
            values = self._numeric(features)
            row: list[tuple[int, float]] = []
            for i, value in enumerate(values):
                centered = value - (self.center[i] if i < len(self.center) else 0.0)
                scale = self.scale[i] if i < len(self.scale) and self.scale[i] else 1.0
                row.append((i, centered / scale))
            row.append((lang_index.get(str(features.lang), lang_other), 1.0))
            names = {str(k) for k, v in (features.gate_detectors or {}).items() if v}
            if names:
                seen: set[int] = set()
                for name in names:
                    slot = det_index.get(name, det_other)
                    if slot not in seen:
                        seen.add(slot)
                        row.append((slot, 1.0))
            row.sort()
            rows.append(row)
        return SparseMatrix.from_rows(rows, self.n_features)

    def fit_transform(self, feature_rows: Sequence[RequestFeatures]) -> SparseMatrix:
        return self.fit(feature_rows).transform(feature_rows)

    def to_config(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "config_version": self.config_version,
            "n_features": self.n_features,
            "numeric_names": list(self.numeric_names),
            "center": [round(v, 6) for v in self.center],
            "scale": [round(v, 6) for v in self.scale],
            "lang_vocab": list(self.lang_vocab),
            "detector_vocab": list(self.detector_vocab),
        }

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> FeatureVectorizer:
        version = int(cfg.get("config_version", 1))
        if version != cls.config_version:
            raise ArtifactError(
                f"features vectorizer config_version {version} is not supported by this build "
                f"(expects {cls.config_version}); retrain the artifact"
            )
        return cls(
            center=[float(v) for v in (cfg.get("center") or [])],
            scale=[float(v) for v in (cfg.get("scale") or [])],
            lang_vocab=tuple(str(v) for v in (cfg.get("lang_vocab") or ())),
            detector_vocab=tuple(str(v) for v in (cfg.get("detector_vocab") or ())),
            numeric_names=tuple(str(v) for v in (cfg.get("numeric_names") or ())) or None,
        )


def build_vectorizer(cfg: Mapping[str, Any]) -> Vectorizer:
    """Reconstruct the vectorizer named by an artifact config."""
    kind = str(cfg.get("kind", ""))
    if kind == TextVectorizer.kind:
        return TextVectorizer.from_config(cfg)
    if kind == FeatureVectorizer.kind:
        return FeatureVectorizer.from_config(cfg)
    raise ArtifactError(
        f"unknown vectorizer kind {kind!r}; this build supports "
        f"{TextVectorizer.kind!r} (text mode) and {FeatureVectorizer.kind!r} (features mode)"
    )


# --------------------------------------------------------------------------- #
# The student
# --------------------------------------------------------------------------- #
def _npy_bytes(array: Any) -> bytes:
    """Serialize one array to ``.npy`` bytes.

    ``.npy`` rather than ``.npz``: an npz is a zip, and zip entries carry the
    current timestamp, so the same weights would produce different bytes (and a
    different checksum) on every run. Raw npy headers are timestamp-free, which
    makes artifacts byte-reproducible and the checksum meaningful.
    """
    import io

    np = require_numpy()
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array, dtype=np.float32), allow_pickle=False)
    return buffer.getvalue()


@dataclass
class StudentModel:
    """A small multi-head network in numpy: optional shared trunk, four heads.

    Linear when ``hidden == 0`` (multinomial logistic regression, the right
    default for a few thousand rows), or a single hidden layer with ``tanh``
    otherwise. ``tanh`` rather than ReLU because these inputs are standardized
    TF-IDF/feature vectors, the network is tiny, and a dead ReLU unit in a 32-unit
    layer trained on 500 rows is a real way to lose capacity silently.

    All heads share the trunk. Sharing is not just parameter economy: complexity,
    sensitivity, and domain are read off the same prompt, so their features
    overlap heavily, and a shared representation trained on all three is better
    regularized than three independent ones on the same small dataset.
    """

    heads: tuple[HeadSpec, ...]
    n_features: int
    hidden: int = 0
    activation: str = "tanh"
    #: Parameter arrays keyed by name: ``W``/``b`` when linear, ``W1``/``b1``/``W2``/``b2`` otherwise.
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_head_layout(self.heads)
        self.hidden = max(0, int(self.hidden))

    # -- shape ------------------------------------------------------------ #
    @property
    def n_outputs(self) -> int:
        return sum(h.size for h in self.heads)

    @property
    def offsets(self) -> dict[str, int]:
        out: dict[str, int] = {}
        cursor = 0
        for head in self.heads:
            out[head.name] = cursor
            cursor += head.size
        return out

    @property
    def is_linear(self) -> bool:
        return self.hidden <= 0

    @property
    def n_parameters(self) -> int:
        return int(sum(p.size for p in self.params.values()))

    # -- forward ---------------------------------------------------------- #
    def _activate(self, h: Any) -> Any:
        np = require_numpy()
        if self.activation == "tanh":
            return np.tanh(h)
        if self.activation == "relu":
            return np.maximum(h, 0.0)
        return h

    def forward_logits(self, x: SparseMatrix | Any) -> Any:
        """Rows of ``x`` -> ``(n_rows, n_outputs)`` raw logits."""
        np = require_numpy()
        dense = x.to_dense() if isinstance(x, SparseMatrix) else np.asarray(x, dtype=np.float32)
        if dense.shape[1] != self.n_features:
            raise ArtifactError(
                f"vector of width {dense.shape[1]} fed to a model expecting {self.n_features}; "
                "the vectorizer and the weights disagree (mixed artifact files?)"
            )
        if self.is_linear:
            return dense @ self.params["W"] + self.params["b"]
        hidden = self._activate(dense @ self.params["W1"] + self.params["b1"])
        return hidden @ self.params["W2"] + self.params["b2"]

    def predict_distributions(self, x: SparseMatrix | Any) -> dict[str, Any]:
        """Per-head probabilities (``pii`` is a length-1 array of P(yes))."""
        logits = self.forward_logits(x)
        offsets = self.offsets
        out: dict[str, Any] = {}
        for head in self.heads:
            start = offsets[head.name]
            block = logits[:, start : start + head.size]
            out[head.name] = _sigmoid(block) if head.kind == "bernoulli" else _softmax_rows(block)
        return out

    def predict_answers(self, x: SparseMatrix | Any) -> list[DecisionAnswers]:
        """Full :class:`DecisionAnswers` per row -- distributions, not just argmax."""
        np = require_numpy()
        distributions = self.predict_distributions(x)
        answers: list[DecisionAnswers] = []
        for row in range(int(next(iter(distributions.values())).shape[0])):
            built: dict[str, ChoiceAnswer] = {}
            for head in self.heads:
                if head.kind == "bernoulli":
                    continue
                probs = [float(p) for p in distributions[head.name][row]]
                total = sum(probs) or 1.0
                probs = [p / total for p in probs]
                # labels and the head's probability vector index the same ladder;
                # a length mismatch would drop a label from the served answer.
                mapping = dict(zip(head.labels, probs, strict=True))
                best = max(range(len(probs)), key=lambda i: (probs[i], -i))
                built[head.name] = ChoiceAnswer(
                    choice=head.labels[best],
                    probabilities=mapping,
                    # Derived, never invented: certainty_from_probabilities is the
                    # same function the policy engine compares against for Jev
                    # answers, so `on_uncertain` floors mean the same thing for a
                    # distilled answer as for a cloud one.
                    confidence=certainty_from_probabilities(mapping),
                    confidence_reported=False,
                )
            pii_value = float(np.clip(distributions["pii"][row][0], 0.0, 1.0))
            answers.append(
                DecisionAnswers(
                    complexity=built["complexity"],
                    sensitivity=built["sensitivity"],
                    pii=NoulAnswer(value=round(pii_value, 6)),
                    domain=built["domain"],
                )
            )
        return answers

    # -- construction ----------------------------------------------------- #
    @classmethod
    def initialize(
        cls,
        n_features: int,
        *,
        hidden: int = 0,
        activation: str = "tanh",
        seed: int = 0,
        heads: Sequence[HeadSpec] | None = None,
    ) -> StudentModel:
        """Xavier-ish init from a seeded generator. Deterministic given ``seed``."""
        np = require_numpy()
        rng = np.random.default_rng(int(seed))
        layout = tuple(heads or default_head_layout())
        n_out = sum(h.size for h in layout)
        model = cls(heads=layout, n_features=int(n_features), hidden=int(hidden), activation=activation)
        if model.is_linear:
            bound = math.sqrt(6.0 / max(1, n_features + n_out))
            model.params = {
                "W": rng.uniform(-bound, bound, size=(n_features, n_out)).astype(np.float32),
                "b": np.zeros(n_out, dtype=np.float32),
            }
        else:
            bound1 = math.sqrt(6.0 / max(1, n_features + hidden))
            bound2 = math.sqrt(6.0 / max(1, hidden + n_out))
            model.params = {
                "W1": rng.uniform(-bound1, bound1, size=(n_features, hidden)).astype(np.float32),
                "b1": np.zeros(hidden, dtype=np.float32),
                "W2": rng.uniform(-bound2, bound2, size=(hidden, n_out)).astype(np.float32),
                # Zero output bias would be fine, but a tiny prior toward the
                # marginal class frequencies is set by the trainer instead.
                "b2": np.zeros(n_out, dtype=np.float32),
            }
        return model

    def clone_params(self) -> dict[str, Any]:
        np = require_numpy()
        return {k: np.array(v, copy=True) for k, v in self.params.items()}

    def restore_params(self, params: Mapping[str, Any]) -> None:
        for key, value in params.items():
            self.params[key] = value

    def weight_bytes(self) -> dict[str, bytes]:
        """Parameter arrays serialized in a stable, sorted order."""
        return {f"{name}.npy": _npy_bytes(self.params[name]) for name in sorted(self.params)}

    @classmethod
    def from_weight_bytes(
        cls,
        files: Mapping[str, bytes],
        *,
        heads: Sequence[HeadSpec],
        n_features: int,
        hidden: int,
        activation: str = "tanh",
    ) -> StudentModel:
        import io

        np = require_numpy()
        model = cls(
            heads=tuple(heads), n_features=int(n_features), hidden=int(hidden), activation=activation
        )
        expected = ("W", "b") if model.is_linear else ("W1", "b1", "W2", "b2")
        params: dict[str, Any] = {}
        for name in expected:
            payload = files.get(f"{name}.npy")
            if payload is None:
                raise ArtifactError(f"artifact is missing weights for {name!r} ({name}.npy)")
            params[name] = np.load(io.BytesIO(payload), allow_pickle=False).astype(np.float32)
        model.params = params
        model._check_shapes()
        return model

    def _check_shapes(self) -> None:
        np = require_numpy()
        if self.is_linear:
            want = {"W": (self.n_features, self.n_outputs), "b": (self.n_outputs,)}
        else:
            want = {
                "W1": (self.n_features, self.hidden),
                "b1": (self.hidden,),
                "W2": (self.hidden, self.n_outputs),
                "b2": (self.n_outputs,),
            }
        for name, shape in want.items():
            got = self.params.get(name)
            if got is None:
                raise ArtifactError(f"artifact weights are missing {name!r}")
            if tuple(got.shape) != shape:
                raise ArtifactError(
                    f"artifact weight {name} has shape {tuple(got.shape)}, expected {shape}; "
                    "the weights and the metadata describe different models"
                )
        if not all(bool(np.isfinite(p).all()) for p in self.params.values()):
            raise ArtifactError("artifact weights contain NaN/Inf; the file is corrupt or training diverged")

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": "linear" if self.is_linear else "mlp",
            "hidden": self.hidden,
            "activation": self.activation,
            "n_features": self.n_features,
            "n_outputs": self.n_outputs,
            "n_parameters": self.n_parameters,
            "dtype": "float32",
            "params": sorted(self.params),
            "heads": [h.to_dict() for h in self.heads],
        }


# --------------------------------------------------------------------------- #
# The artifact envelope
# --------------------------------------------------------------------------- #
def environment_info() -> dict[str, Any]:
    """Versions that a future reader needs to reproduce or distrust an artifact."""
    import sys

    np = require_numpy()
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "numpy": getattr(np, "__version__", "unknown"),
        "decision_schema_version": SCHEMA_VERSION,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
    }
    try:  # the package version is nice-to-have; a source checkout may lack metadata
        from .. import __version__ as _pkg_version

        info["jev_route"] = _pkg_version
    except Exception:  # pragma: no cover - never fail an artifact write over this
        info["jev_route"] = "unknown"
    return info


def _load_payload(path: str | Path) -> dict[str, bytes]:
    """Read an artifact directory (or a zip of one) into ``{filename: bytes}``."""
    p = Path(path)
    if p.is_dir():
        return {child.name: child.read_bytes() for child in sorted(p.iterdir()) if child.is_file()}
    if p.is_file() and zipfile.is_zipfile(p):
        with zipfile.ZipFile(p) as archive:
            return {name: archive.read(name) for name in sorted(archive.namelist()) if not name.endswith("/")}
    if not p.exists():
        raise ArtifactError(
            f"no distilled artifact at {p}. Export the log (`jev-route export --log <decisions.jsonl> "
            f"--out <dataset>`), train it (`jev-route train --data <dataset> --out {p}`), or point "
            f"backend.distilled.artifact at the directory you trained."
        )
    raise ArtifactError(f"{p} is neither an artifact directory nor a zip archive")


def _verify_envelope(
    path: str | Path,
    metadata: Mapping[str, Any],
    files: Mapping[str, bytes],
    *,
    verify_checksum: bool,
) -> None:
    """Prove a directory of JSON is *this* pipeline's artifact, and that it is intact.

    Split out of :meth:`DistilledArtifact.load` so that method reads as one
    load-and-build sequence. These four checks are the trust boundary: kind,
    schema version, a file manifest, and a SHA-256 per listed file. Every failure
    raises :class:`ArtifactError` with the reason, because an artifact that fails
    one of them would otherwise be served and mis-route.
    """
    kind = str(metadata.get("kind", ""))
    if kind != ARTIFACT_KIND:
        raise ArtifactError(f"{path} has kind {kind!r}, expected {ARTIFACT_KIND!r}")

    version = str(metadata.get("artifact_schema_version", ""))
    if version not in SUPPORTED_ARTIFACT_SCHEMA_VERSIONS:
        raise ArtifactError(
            f"artifact schema version {version or '(missing)'} is not supported by this build "
            f"(supports {sorted(SUPPORTED_ARTIFACT_SCHEMA_VERSIONS)}). "
            "Upgrade jev-route or retrain the artifact with the current version."
        )

    listed = metadata.get("files") or {}
    if not isinstance(listed, Mapping) or not listed:
        raise ArtifactError("artifact metadata lists no files; it was not written by this pipeline")
    if not verify_checksum:
        return
    for name, entry in listed.items():
        blob = files.get(str(name))
        if blob is None:
            raise ArtifactError(f"artifact is missing {name} listed in {METADATA_FILE}")
        expected = str((entry or {}).get("sha256", ""))
        actual = sha256_bytes(blob)
        if expected and expected != actual:
            raise ArtifactError(
                f"checksum mismatch for {name}: expected {expected[:16]}..., got {actual[:16]}.... "
                "The artifact was truncated, edited by hand, or copied incompletely. "
                "Refusing to serve it: retrain or restore from backup."
            )


@dataclass(frozen=True)
class DistilledArtifact:
    """A loaded, validated, ready-to-serve distilled model.

    Immutable and self-describing: everything needed to reproduce or audit the
    model -- head layout, vectorizer config, training config, dataset statistics,
    evaluation metrics, teacher provenance -- travels with the weights.
    """

    model: StudentModel
    vectorizer: Vectorizer
    metadata: dict[str, Any]
    metrics: dict[str, Any] | None = None
    dataset: dict[str, Any] | None = None
    source: str | None = None

    # -- provenance ------------------------------------------------------- #
    @property
    def mode(self) -> str:
        return str(self.metadata.get("mode", "features"))

    @property
    def model_version(self) -> str:
        return str(self.metadata.get("model_version", "distilled-unknown"))

    @property
    def created_at(self) -> str:
        return str(self.metadata.get("created_at", ""))

    @property
    def artifact_schema_version(self) -> str:
        return str(self.metadata.get("artifact_schema_version", ""))

    @property
    def teacher_model_versions(self) -> tuple[str, ...]:
        teacher = self.metadata.get("teacher") or {}
        return tuple(str(v) for v in (teacher.get("model_versions") or ()))

    @property
    def training_config(self) -> dict[str, Any]:
        return dict(self.metadata.get("training") or {})

    @property
    def contains_prompt_text(self) -> bool:
        """True when the artifact was trained from retained prompt text.

        Recorded explicitly so an operator can answer "does this file contain our
        prompts?" without reading the training code. The artifact itself never
        stores text: only the vocabulary derived from it.
        """
        return bool(self.metadata.get("contains_prompt_text", self.mode == "text"))

    @property
    def n_features(self) -> int:
        return int(self.model.n_features)

    @property
    def size_bytes(self) -> int:
        files = self.metadata.get("files") or {}
        return int(sum(int(entry.get("bytes", 0)) for entry in files.values() if isinstance(entry, Mapping)))

    # -- inference -------------------------------------------------------- #
    def answers_for(self, *, text: str | None = None, features: RequestFeatures | None = None) -> DecisionAnswers:
        """One decision's worth of answers. Raises :class:`ArtifactError` on a mode mismatch."""
        return self.predict_many(
            texts=[text] if text is not None else None,
            features=[features] if features is not None else None,
        )[0]

    def predict_many(
        self,
        *,
        texts: Sequence[str] | None = None,
        features: Sequence[RequestFeatures] | None = None,
    ) -> list[DecisionAnswers]:
        if self.mode == "text":
            if texts is None:
                raise ArtifactError(
                    "this artifact was trained in text mode and needs the redacted excerpt; "
                    "no text was supplied. Serve it with logging.excerpt_mode: redacted traffic, "
                    "or retrain in features mode (`jev-route export --mode features`, then `jev-route train`)."
                )
            matrix = self.vectorizer.transform(list(texts))
        else:
            if features is None:
                raise ArtifactError(
                    "this artifact was trained in features mode and needs RequestFeatures; none supplied."
                )
            matrix = self.vectorizer.transform(list(features))
        return self.model.predict_answers(matrix)

    # -- writing ---------------------------------------------------------- #
    def payload(self, *, write_metrics: bool = True, write_dataset: bool = True) -> dict[str, bytes]:
        """Every file of the artifact, in memory, metadata last (it checksums the rest)."""
        blobs: dict[str, bytes] = {
            VECTORIZER_FILE: _json_bytes(self.vectorizer.to_config()),
        }
        blobs.update(self.model.weight_bytes())
        if write_metrics and self.metrics:
            blobs[METRICS_FILE] = _json_bytes(self.metrics)
        if write_dataset and self.dataset:
            blobs[DATASET_FILE] = _json_bytes(self.dataset)

        metadata = {k: v for k, v in self.metadata.items() if k != "files"}
        metadata["files"] = {
            name: {"sha256": sha256_bytes(blob), "bytes": len(blob)} for name, blob in sorted(blobs.items())
        }
        metadata.setdefault("artifact_schema_version", ARTIFACT_SCHEMA_VERSION)
        metadata.setdefault("kind", ARTIFACT_KIND)
        blobs[METADATA_FILE] = _json_bytes(metadata, indent=2)
        return blobs

    def save(self, path: str | Path, **kwargs: Any) -> Path:
        """Write the artifact directory. Creates parents; existing files are overwritten."""
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        for name, blob in self.payload(**kwargs).items():
            (target / name).write_bytes(blob)
        return target

    def save_zip(self, path: str | Path, **kwargs: Any) -> Path:
        """Write a single-file artifact. Fixed timestamps, so the zip is reproducible."""
        target = Path(path)
        if target.parent != Path(""):
            target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, blob in self.payload(**kwargs).items():
                # A pinned date_time is the only way to make a zip checksum stable.
                archive.writestr(zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)), blob)
        return target

    # -- reading ---------------------------------------------------------- #
    @classmethod
    def load(cls, path: str | Path, *, verify_checksum: bool = True, validate: bool = True) -> DistilledArtifact:
        """Load and validate. Every failure mode raises :class:`ArtifactError` with a reason."""
        files = _load_payload(path)
        raw_meta = files.get(METADATA_FILE)
        if raw_meta is None:
            raise ArtifactError(
                f"{path} is not a jev-route artifact: no {METADATA_FILE}. Point backend.distilled.artifact "
                "at the directory written by `jev-route train`."
            )
        try:
            metadata = json.loads(raw_meta.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"{path}/{METADATA_FILE} is not valid JSON: {exc}") from exc
        if not isinstance(metadata, Mapping):
            raise ArtifactError(f"{METADATA_FILE} must be a JSON object")
        metadata = dict(metadata)

        _verify_envelope(path, metadata, files, verify_checksum=verify_checksum)

        vec_raw = files.get(VECTORIZER_FILE)
        if vec_raw is None:
            raise ArtifactError(f"artifact is missing {VECTORIZER_FILE}")
        vectorizer = build_vectorizer(json.loads(vec_raw.decode("utf-8")))

        model_meta = metadata.get("model") or {}
        if not isinstance(model_meta, Mapping):
            raise ArtifactError("artifact metadata has no `model` section")
        heads = tuple(HeadSpec.from_dict(h) for h in (model_meta.get("heads") or ()))
        if not heads:
            raise ArtifactError("artifact metadata lists no output heads")
        if validate:
            # The load-time check that matters most: an artifact whose ladders no
            # longer match the schema would map probabilities onto the wrong labels.
            validate_head_layout(heads)
        n_features = int(model_meta.get("n_features", vectorizer.n_features))
        if vectorizer.n_features != n_features:
            raise ArtifactError(
                f"vectorizer produces {vectorizer.n_features} features but the model expects {n_features}"
            )
        model = StudentModel.from_weight_bytes(
            files,
            heads=heads,
            n_features=n_features,
            hidden=int(model_meta.get("hidden", 0)),
            activation=str(model_meta.get("activation", "tanh")),
        )

        metrics = _optional_json(files.get(METRICS_FILE))
        dataset = _optional_json(files.get(DATASET_FILE))
        return cls(
            model=model,
            vectorizer=vectorizer,
            metadata=metadata,
            metrics=metrics,
            dataset=dataset,
            source=str(path),
        )

    def describe(self) -> str:
        """One-screen summary for the CLI."""
        training = self.training_config
        text_note = "retains prompt text in the dataset" if self.contains_prompt_text else "no prompt text"
        lines = [
            f"artifact        : {self.source}",
            f"model_version   : {self.model_version}",
            f"schema          : artifact v{self.artifact_schema_version}, decision schema v{SCHEMA_VERSION}",
            f"mode            : {self.mode} ({text_note})",
            f"created         : {self.created_at}",
            f"architecture    : {self.model.to_dict()['family']}, hidden={self.model.hidden}, "
            f"features={self.model.n_features}, params={self.model.n_parameters}",
            f"teacher         : {', '.join(self.teacher_model_versions) or 'unknown'}",
        ]
        if training:
            lines.append(
                f"trained with    : T={training.get('temperature')}, epochs={training.get('epochs_run')}"
                f"/{training.get('epochs')}, lr={training.get('learning_rate')}, seed={training.get('seed')}"
            )
        dataset = self.dataset or {}
        if dataset:
            lines.append(
                f"dataset         : {dataset.get('rows')} rows "
                f"({dataset.get('train_rows')} train / {dataset.get('holdout_rows')} holdout), "
                f"mode={dataset.get('mode')}"
            )
        return "\n".join(lines)


def _json_bytes(obj: Any, *, indent: int | None = None) -> bytes:
    return json.dumps(obj, sort_keys=True, indent=indent, default=str).encode("utf-8")


def _optional_json(blob: bytes | None) -> dict[str, Any] | None:
    if blob is None:
        return None
    try:
        data = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, Mapping) else {"value": data}


def save_artifact(artifact: DistilledArtifact, path: str | Path, **kwargs: Any) -> Path:
    """Write ``artifact`` to ``path``. Kept as a function for CLI symmetry with :func:`load_artifact`."""
    return artifact.save(path, **kwargs)


def load_artifact(path: str | Path, **kwargs: Any) -> DistilledArtifact:
    """Load and validate the artifact at ``path`` (a directory or a zip)."""
    return DistilledArtifact.load(path, **kwargs)


__all__ = [
    "ALL_HEADS",
    "ARTIFACT_KIND",
    "ARTIFACT_SCHEMA_VERSION",
    "CHOICE_HEADS",
    "DATASET_FILE",
    "METADATA_FILE",
    "METRICS_FILE",
    "SUPPORTED_ARTIFACT_SCHEMA_VERSIONS",
    "VECTORIZER_FILE",
    "ArtifactError",
    "DistillDependencyError",
    "DistilledArtifact",
    "FeatureVectorizer",
    "HeadSpec",
    "SparseMatrix",
    "StudentModel",
    "TextVectorizer",
    "Vectorizer",
    "build_vectorizer",
    "default_head_layout",
    "environment_info",
    "load_artifact",
    "require_numpy",
    "save_artifact",
    "tokenize",
    "validate_head_layout",
]
