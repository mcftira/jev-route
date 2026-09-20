"""Soft-target distillation: teach a small local model to imitate Jev.

The objective is KL(teacher || student) over the teacher's *probability
distributions*, not cross-entropy against argmax labels. That single choice is
why this pipeline exists. A hard label says "internal"; the teacher's
distribution says "internal 0.46, confidential 0.38, public 0.16" -- and the
second one is what lets the router's ``on_uncertain`` rules keep working after
the cloud dependency is gone. Distilling argmax labels would produce a confident
student that the policy engine can no longer reason about, which is a regression
dressed up as a cost saving.

**Temperature.** Teacher probabilities ``p`` are softened to
``normalize(p ** (1/T))`` and the student's logits are divided by the same ``T``,
with the loss scaled by ``T^2`` (Hinton's scaling, so gradient magnitudes do not
shrink as ``T`` grows). The algebra matters here, because it is what makes the
served model honest: matching ``softmax(z/T)`` to ``normalize(p^(1/T))`` implies
``z = log p + const``, so ``softmax(z) = p`` at ``T = 1``. Inference therefore
runs at temperature 1 and the student reproduces the teacher's calibration rather
than a flattened version of it. Higher ``T`` transfers more of the "dark
knowledge" in the tails -- the relative ordering of the labels the teacher
rejected -- at the cost of a noisier gradient.

**Default trainer: numpy, no torch.** A tiny shared-trunk network (linear by
default) trained with Adam on mini-batches. numpy is enough, and preferring it is
a deliberate call: the model has tens of thousands of parameters, the input is
sparse and small, and BLAS-backed matmuls finish in well under a second on a
laptop. torch would add a 2 GB dependency and a version-compatibility surface to
buy nothing at this size, and it would put a deep-learning runtime in the request
path of a router whose whole promise is that the local end state is boring and
auditable. A torch trainer for the *same* architecture is available behind
``framework="torch"`` for operators with very large logs; see :func:`train_torch`.

**Determinism.** Everything random comes from one seeded ``numpy`` generator:
weight init and epoch shuffling. No network, no GPU, no clock-derived state. The
same dataset, config, and numpy version produce the same weights.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .artifact import (
    ALL_HEADS,
    DistillDependencyError,
    FeatureVectorizer,
    HeadSpec,
    SparseMatrix,
    StudentModel,
    TextVectorizer,
    default_head_layout,
    require_numpy,
    utcnow,
)
from .export import Dataset, TrainingRow, load_dataset, split_bucket

#: Sensitivity is weighted highest on purpose: it is the head the data-egress rule
#: reads, so an error there is a privacy incident rather than a cost incident.
DEFAULT_HEAD_WEIGHTS: Mapping[str, float] = {
    "complexity": 1.0,
    "sensitivity": 1.5,
    "domain": 1.0,
    "pii": 1.0,
}

DEFAULT_SEED = 20260919


class TrainError(RuntimeError):
    """Training cannot proceed, or cannot proceed honestly. Always actionable."""


@dataclass(frozen=True)
class TrainConfig:
    """Every knob of the trainer, with defaults tuned for a laptop and a small log."""

    #: ``numpy`` (default) or ``torch``. Both produce the same artifact format.
    framework: str = "numpy"
    #: ``text``, ``features``, or ``auto`` (use the dataset's exported mode).
    mode: str = "auto"
    #: Distillation temperature. 1.0 = plain soft-target KL; 2-4 transfers more tail structure.
    temperature: float = 2.0
    head_weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_HEAD_WEIGHTS))
    learning_rate: float = 0.03
    epochs: int = 120
    batch_size: int = 64
    #: L2 on weights only (never biases). Reported in the artifact.
    l2: float = 1e-4
    #: 0 = linear multinomial logistic regression, which is the right default for
    #: a few thousand rows; >0 adds one shared hidden layer.
    hidden: int = 0
    activation: str = "tanh"
    seed: int = DEFAULT_SEED
    #: Text mode only: vocabulary cap. Big enough for topical signal, small enough
    #: that the artifact stays a few hundred kilobytes.
    max_features: int = 4096
    ngram_max: int = 2
    #: Carved out of the TRAIN split (never the holdout) for early stopping, by
    #: request_id hash, so it is stable across runs and cannot leak.
    val_fraction: float = 0.1
    early_stopping: bool = True
    patience: int = 15
    optimizer: Literal["adam", "sgd"] = "adam"
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    #: Down-weight rows the gate decided: their label is real but their heads are
    #: partly masked, and they should not dominate a small dataset.
    gate_row_weight: float = 1.0
    #: torch-only. A transformer encoder is deliberately NOT supported: see
    #: :func:`train_torch`'s docstring for why that is a decision and not an omission.
    encoder: str | None = None
    device: str = "cpu"

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            out[key] = dict(value) if isinstance(value, Mapping) else value
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrainConfig:
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - known)
        if unknown:
            raise TrainError(
                f"unknown training config key(s) {unknown}; known keys: {sorted(known)}. "
                "A typo here would silently train with defaults."
            )
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(frozen=True)
class TrainedStudent:
    """A trained model plus everything the artifact needs to describe itself."""

    model: StudentModel
    vectorizer: Any
    config: TrainConfig
    mode: str
    history: tuple[dict[str, float], ...]
    stats: dict[str, Any]
    warnings: tuple[str, ...] = ()
    trained_at: str = field(default_factory=utcnow)
    teacher_model_versions: tuple[str, ...] = ()
    dataset_sha256: str = ""
    dataset_source: str = ""
    #: Set only by the torch trainer when it also wrote a framework-specific bundle.
    torch_bundle: dict[str, Any] | None = None

    @property
    def n_parameters(self) -> int:
        return self.model.n_parameters

    def summary(self) -> str:
        final = self.history[-1] if self.history else {}
        return (
            f"trained {self.mode}-mode student: {self.model.to_dict()['family']}, "
            f"{self.model.n_features} features, {self.n_parameters} parameters, "
            f"{len(self.history)} epochs, final loss {final.get('train_loss', float('nan')):.4f}"
            + (f", best val {final.get('best_val_loss', float('nan')):.4f}" if "best_val_loss" in final else "")
        )


# --------------------------------------------------------------------------- #
# Vectorization
# --------------------------------------------------------------------------- #
def build_vectorizer(dataset: Dataset, config: TrainConfig) -> Any:
    """Fit the vectorizer implied by the dataset's mode."""
    mode = resolve_train_mode(dataset, config)
    if mode == "text":
        vectorizer = TextVectorizer(
            max_features=config.max_features, ngram_max=config.ngram_max
        ).fit(dataset.texts())
        if vectorizer.n_features == 0:
            raise TrainError(
                "text mode produced an empty vocabulary: every excerpt in the dataset is empty or "
                "shorter than the tokenizer's minimum. Re-export with --mode features, or check that "
                "logging.excerpt_mode: redacted was actually in effect."
            )
    else:
        vectorizer = FeatureVectorizer().fit(dataset.feature_rows())
    return vectorizer


def resolve_train_mode(dataset: Dataset, config: TrainConfig) -> str:
    wanted = str(config.mode or "auto").lower()
    if wanted == "auto":
        return dataset.mode
    if wanted not in ("text", "features"):
        raise TrainError(f"unknown training mode {config.mode!r}; expected text, features, or auto")
    if wanted == "text" and dataset.mode != "text":
        raise TrainError(
            f"cannot train a text model from a {dataset.mode!r}-mode dataset: it carries no prompt text. "
            "Re-export with --mode text (requires logging.excerpt_mode: redacted in the policy)."
        )
    return wanted


def vectorize(dataset: Dataset, vectorizer: Any) -> SparseMatrix:
    if isinstance(vectorizer, TextVectorizer):
        return vectorizer.transform(dataset.texts())
    return vectorizer.transform(dataset.feature_rows())


# --------------------------------------------------------------------------- #
# Target tensors
# --------------------------------------------------------------------------- #
def _softmax_np(logits: Any) -> Any:
    np = require_numpy()
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _sigmoid_np(x: Any) -> Any:
    np = require_numpy()
    return 0.5 * (1.0 + np.tanh(np.asarray(x, dtype=np.float64) / 2.0))


def soften_distribution(probs: Sequence[float], temperature: float) -> list[float]:
    """``normalize(p ** (1/T))``: the teacher side of temperature scaling.

    ``T = 1`` returns ``p`` unchanged; larger ``T`` flattens toward uniform.
    Implemented in log space so a teacher probability of 0.0 stays finite.
    """
    if temperature <= 1.0 + 1e-9:
        return [float(p) for p in probs]
    floor = 1e-12
    logs = [math.log(max(float(p), floor)) / temperature for p in probs]
    top = max(logs)
    exps = [math.exp(v - top) for v in logs]
    total = sum(exps) or 1.0
    return [v / total for v in exps]


def soften_scalar(value: float, temperature: float) -> float:
    """Temperature scaling for the Bernoulli head, in logit space."""
    if temperature <= 1.0 + 1e-9:
        return float(value)
    v = min(1.0 - 1e-9, max(1e-9, float(value)))
    logit = math.log(v / (1.0 - v))
    scaled = logit / temperature
    return 1.0 / (1.0 + math.exp(-scaled))


@dataclass(frozen=True)
class TrainingTensors:
    """Everything the optimizer needs, precomputed once.

    Targets are softened *here*, not inside the loss: the softening is a pure
    function of the dataset and the temperature, so doing it once per row instead
    of once per epoch per batch is both faster and easier to reason about.
    """

    x: SparseMatrix
    targets: Any  # (n_rows, n_outputs) float32, temperature-softened -- what the loss uses
    #: The same targets at T=1, i.e. exactly what the teacher said. Kept so the
    #: per-head diagnostics measure the *served* model against the *real* teacher
    #: rather than against a softened version of it.
    raw: Any
    mask: Any  # (n_rows, n_heads) float32, 1.0 where the head has signal
    row_weight: Any  # (n_rows,) float32
    heads: tuple[HeadSpec, ...]
    offsets: dict[str, int]

    @property
    def n_rows(self) -> int:
        return int(self.mask.shape[0])


def build_tensors(dataset: Dataset, vectorizer: Any, config: TrainConfig) -> TrainingTensors:
    """Rows -> (sparse features, softened targets, per-head usability mask)."""
    np = require_numpy()
    heads = default_head_layout()
    offsets: dict[str, int] = {}
    cursor = 0
    for head in heads:
        offsets[head.name] = cursor
        cursor += head.size
    n_out = cursor

    rows = dataset.rows
    temperature = max(1e-6, float(config.temperature))
    targets = np.zeros((len(rows), n_out), dtype=np.float32)
    raw = np.zeros((len(rows), n_out), dtype=np.float32)
    mask = np.zeros((len(rows), len(heads)), dtype=np.float32)
    row_weight = np.ones(len(rows), dtype=np.float32)

    for i, row in enumerate(rows):
        for h, head in enumerate(heads):
            start = offsets[head.name]
            target = row.targets.get(head.name)
            usable = bool(target.get("usable", False))
            mask[i, h] = 1.0 if usable else 0.0
            if not usable:
                # Fill with something finite; the mask keeps it out of the loss.
                if head.kind == "bernoulli":
                    targets[i, start] = raw[i, start] = 0.5
                else:
                    targets[i, start : start + head.size] = 1.0 / head.size
                    raw[i, start : start + head.size] = 1.0 / head.size
                continue
            if head.kind == "bernoulli":
                value = float(target["value"])
                raw[i, start] = value
                targets[i, start] = soften_scalar(value, temperature)
            else:
                probs = [float(target["probs"][label]) for label in head.labels]
                for k, prob in enumerate(probs):
                    raw[i, start + k] = prob
                for k, value in enumerate(soften_distribution(probs, temperature)):
                    targets[i, start + k] = value
        if row.is_gate_row and config.gate_row_weight != 1.0:
            row_weight[i] = float(config.gate_row_weight)

    matrix = vectorize(dataset, vectorizer)
    return TrainingTensors(
        x=matrix, targets=targets, raw=raw, mask=mask, row_weight=row_weight, heads=heads, offsets=offsets
    )


def check_degenerate(dataset: Dataset, tensors: TrainingTensors, config: TrainConfig) -> list[str]:
    """Refuse to train on data that cannot teach anything, with a reason a human can act on.

    The failure this guards against is silent: a log of gate-blocked requests only
    produces uniform teacher distributions for three of the four heads and a
    constant for the fourth. Gradient descent happily converges on "always answer
    the base rate", the metrics look non-NaN, and the operator ships a model that
    never actually reads a prompt. Failing loudly here is cheaper.
    """
    np = require_numpy()
    warnings: list[str] = []
    if tensors.n_rows == 0:
        raise TrainError(
            "the training split is empty. Export produced no usable rows -- check the export summary "
            "for skip reasons (degraded records, missing excerpts) and collect more traffic."
        )
    if tensors.x.n_features == 0:
        raise TrainError(
            "the vectorizer produced zero features; nothing can be learned. In text mode this means every "
            "excerpt was empty; in features mode it means the dataset has no rows."
        )

    usable = tensors.mask.sum(axis=0)
    heads = tensors.heads
    for index, head in enumerate(heads):
        if usable[index] == 0:
            warnings.append(
                f"head {head.name!r} has no usable rows (every teacher distribution was uniform); "
                f"it will stay at its prior and contribute nothing. "
                + (
                    "This is normal for gate-blocked traffic, where no backend classified complexity or domain."
                    if dataset.rows and all(r.is_gate_row for r in dataset.rows)
                    else "Collect traffic where the backend actually answered."
                )
            )

    # Signal test: does any head's teacher distribution vary across rows? A
    # constant target is not learnable signal, it is a base rate.
    varying = 0.0
    for index, head in enumerate(heads):
        if usable[index] < 2:
            continue
        start = tensors.offsets[head.name]
        block = tensors.targets[:, start : start + head.size][tensors.mask[:, index] > 0]
        deviation = float(np.abs(block - block.mean(axis=0)).mean())
        varying = max(varying, deviation)
        if deviation < 1e-9:
            warnings.append(
                f"head {head.name!r}: all {int(usable[index])} teacher distributions are identical. "
                "The student can only learn a constant here."
            )
    if varying < 1e-9:
        raise TrainError(
            "every teacher distribution in this dataset is identical (uniform or constant): there is no "
            "signal to distill. This happens when the log contains only gate-blocked or degraded requests, "
            "because nothing was ever classified by a backend. Fix it by routing real traffic while the "
            "backend is healthy, and by setting gate.on_force_local: still_classify if you want labels for "
            "gate-blocked prompts too."
        )

    temperature = float(config.temperature)
    if not 1.0 <= temperature <= 20.0:
        raise TrainError(
            f"temperature {temperature} is outside the sane range [1, 20]. T=1 is plain soft-target KL; "
            "T=2-4 is the usual distillation range; above that the targets are nearly uniform and the "
            "student learns nothing."
        )
    if config.learning_rate <= 0:
        raise TrainError(f"learning_rate must be positive, got {config.learning_rate}")
    if config.epochs <= 0:
        raise TrainError(f"epochs must be positive, got {config.epochs}")
    unknown_heads = sorted(set(config.head_weights) - set(ALL_HEADS))
    if unknown_heads:
        raise TrainError(
            f"head_weights names unknown head(s) {unknown_heads}; valid heads are {list(ALL_HEADS)}"
        )
    return warnings


# --------------------------------------------------------------------------- #
# Loss, gradients, optimizer
# --------------------------------------------------------------------------- #
def _kl_loss_and_grad(
    logits: Any,
    targets: Any,
    mask: Any,
    row_weight: Any,
    *,
    heads: Sequence[HeadSpec],
    offsets: Mapping[str, int],
    head_weights: Mapping[str, float],
    temperature: float,
) -> tuple[float, Any]:
    """Weighted, masked KL(teacher || student) and its exact gradient w.r.t. logits.

    The gradient is analytic, not autodiff: with ``q = softmax(z/T)`` and the loss
    scaled by ``T^2``, ``d/dz`` collapses to ``T * (q - p)`` per row, and the
    Bernoulli head has exactly the same form. Writing it down explicitly keeps the
    trainer dependency-free and makes the temperature scaling auditable in one
    line instead of hidden in a graph.

    Each head is normalized by its own usable mass (``sum(mask * row_weight)``),
    so a head that only has signal on 5% of the rows still contributes at full
    strength rather than being drowned out by the heads that are masked in.
    """
    np = require_numpy()
    grad = np.zeros_like(logits)
    total = 0.0
    t = max(1e-6, float(temperature))
    for index, head in enumerate(heads):
        start = offsets[head.name]
        stop = start + head.size
        p = targets[:, start:stop]
        m = mask[:, index] * row_weight
        denom = float(m.sum())
        if denom <= 0.0:
            continue  # no signal for this head in this batch
        weight = float(head_weights.get(head.name, 1.0))
        if head.kind == "bernoulli":
            q = np.clip(_sigmoid_np(logits[:, start:stop] / t), 1e-12, 1.0 - 1e-12)
            kl = -(p * np.log(q) + (1.0 - p) * np.log(1.0 - q)).sum(axis=1)
        else:
            q = np.clip(_softmax_np(logits[:, start:stop] / t), 1e-12, 1.0)
            kl = (p * (np.log(np.clip(p, 1e-12, 1.0)) - np.log(q))).sum(axis=1)
        total += weight * (t * t) * float((kl * m).sum()) / denom
        # T^2 from the loss times 1/T from the softmax derivative = T.
        grad[:, start:stop] = (q - p) * (weight * t / denom) * m[:, None]
    return total, grad


def _regularized(params: Mapping[str, Any], l2: float) -> tuple[float, dict[str, Any]]:
    """L2 penalty and its gradient over weight matrices only.

    Biases are excluded: shrinking a bias toward zero pulls the model away from
    the empirical class prior, which is exactly the calibration we are trying to
    preserve.
    """
    # Called for the actionable error it raises when numpy is missing, not for its
    # return value: the penalty below is pure Python floats times numpy arrays that
    # the caller already produced, so there is no `np` to use here.
    require_numpy()
    if l2 <= 0.0:
        return 0.0, {}
    penalty = 0.0
    grads: dict[str, Any] = {}
    for name, array in params.items():
        if name.startswith("b"):
            continue
        penalty += float(l2) * 0.5 * float((array * array).sum())
        grads[name] = float(l2) * array
    return penalty, grads


def _forward(model: StudentModel, dense: Any) -> dict[str, Any]:
    if model.is_linear:
        return {"logits": dense @ model.params["W"] + model.params["b"], "hidden": None, "dense": dense}
    pre = dense @ model.params["W1"] + model.params["b1"]
    hidden = _activate(pre, model.activation)
    return {"logits": hidden @ model.params["W2"] + model.params["b2"], "hidden": hidden, "dense": dense}


def _activate(pre: Any, activation: str) -> Any:
    """Hidden-layer activation. ``tanh`` is the default; see the module docstring."""
    np = require_numpy()
    if activation == "tanh":
        return np.tanh(pre)
    if activation == "relu":
        return np.maximum(pre, 0.0)
    if activation in ("none", "identity", "linear"):
        return pre
    raise TrainError(f"unknown activation {activation!r}; expected tanh, relu, or none")


def _backward(model: StudentModel, cache: Mapping[str, Any], grad_logits: Any) -> dict[str, Any]:
    dense = cache["dense"]
    if model.is_linear:
        return {"W": dense.T @ grad_logits, "b": grad_logits.sum(axis=0)}
    hidden = cache["hidden"]
    grads = {"W2": hidden.T @ grad_logits, "b2": grad_logits.sum(axis=0)}
    d_hidden = grad_logits @ model.params["W2"].T
    if model.activation == "tanh":
        d_pre = d_hidden * (1.0 - hidden * hidden)
    elif model.activation == "relu":
        d_pre = d_hidden * (hidden > 0.0)
    else:
        d_pre = d_hidden  # identity activation: the gradient passes straight through
    grads["W1"] = dense.T @ d_pre
    grads["b1"] = d_pre.sum(axis=0)
    return grads


class _Optimizer:
    """Adam (default) or SGD+momentum over a flat parameter dict.

    Written out rather than borrowed from a library so the trainer has no
    dependency beyond numpy and so the update rule is visible in the repo: for a
    model this small, an optimizer you can read is worth more than an optimized one.
    """

    def __init__(self, params: Mapping[str, Any], config: TrainConfig) -> None:
        np = require_numpy()
        self.config = config
        self.state = {name: [np.zeros_like(array), np.zeros_like(array)] for name, array in params.items()}
        self.step_count = 0

    def apply(self, params: dict[str, Any], grads: Mapping[str, Any]) -> None:
        np = require_numpy()
        self.step_count += 1
        lr = float(self.config.learning_rate)
        if self.config.optimizer == "sgd":
            momentum = float(self.config.momentum)
            for name, array in params.items():
                grad = grads.get(name)
                if grad is None:
                    continue
                velocity = self.state[name][0]
                velocity *= momentum
                velocity -= lr * grad
                array += velocity
            return
        beta1, beta2, eps = float(self.config.beta1), float(self.config.beta2), float(self.config.epsilon)
        correction1 = 1.0 - beta1**self.step_count
        correction2 = 1.0 - beta2**self.step_count
        for name, array in params.items():
            grad = grads.get(name)
            if grad is None:
                continue
            first, second = self.state[name]
            first *= beta1
            first += (1.0 - beta1) * grad
            second *= beta2
            second += (1.0 - beta2) * (grad * grad)
            array -= lr * (first / correction1) / (np.sqrt(second / correction2) + eps)


def _marginal_bias(dataset: Dataset, heads: Sequence[HeadSpec], offsets: Mapping[str, int]) -> Any:
    """Output bias at the empirical class prior.

    Starting from the prior rather than zeros means epoch 0 already predicts base
    rates, which matters for heads like ``domain`` where one class holds most of
    the mass: without it, Adam spends the first epochs just finding the intercept.
    Computed from *unsoftened* targets, because the bias belongs to the T=1
    distribution the model serves.
    """
    np = require_numpy()
    n_out = sum(h.size for h in heads)
    bias = np.zeros(n_out, dtype=np.float32)
    for head in heads:
        start = offsets[head.name]
        if head.kind == "bernoulli":
            values = [
                float(r.targets.pii["value"]) for r in dataset.rows if r.targets.usable("pii")
            ]
            if not values:
                continue
            mean = min(1.0 - 1e-6, max(1e-6, sum(values) / len(values)))
            bias[start] = math.log(mean / (1.0 - mean))
            continue
        sums = np.zeros(head.size, dtype=np.float64)
        count = 0
        for row in dataset.rows:
            target = row.targets.get(head.name)
            if not target.get("usable"):
                continue
            count += 1
            for k, label in enumerate(head.labels):
                sums[k] += float(target["probs"][label])
        if count == 0:
            continue
        prior = np.maximum(sums / count, 1e-6)
        prior = prior / prior.sum()
        bias[start : start + head.size] = np.log(prior)
    return bias


def _dense_batch(matrix: SparseMatrix, indices: Any) -> Any:
    """Densify an arbitrary subset of rows, fully vectorized.

    Batches are small (tens to hundreds of rows), so materializing them is both
    cheaper and simpler than writing a sparse matmul: the heavy lifting goes to
    BLAS on a dense ``(batch, features)`` block, and the dataset itself stays
    sparse in memory, which is what keeps a 50k-row text export affordable.
    """
    np = require_numpy()
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return np.zeros((0, matrix.n_features), dtype=np.float32)
    lo = matrix.indptr.astype(np.int64)[idx]
    counts = matrix.indptr.astype(np.int64)[idx + 1] - lo
    out = np.zeros((idx.size, matrix.n_features), dtype=np.float32)
    nnz = int(counts.sum())
    if nnz:
        row_ids = np.repeat(np.arange(idx.size, dtype=np.int64), counts)
        cum = np.cumsum(counts) - counts
        within = np.arange(nnz, dtype=np.int64) - np.repeat(cum, counts)
        src = np.repeat(lo, counts) + within
        out[row_ids, matrix.indices.astype(np.int64)[src]] = matrix.data[src]
    return out


def _loss_over(model: StudentModel, tensors: TrainingTensors, indices: Any, config: TrainConfig) -> float:
    """Mean regularized KL over an arbitrary row subset (the validation curve)."""
    np = require_numpy()
    idx = np.asarray(indices)
    if idx.size == 0:
        return float("nan")
    chunk = 1024
    total = 0.0
    blocks = 0
    penalty, _ = _regularized(model.params, config.l2)
    for start in range(0, idx.size, chunk):
        block = idx[start : start + chunk]
        logits = _forward(model, _dense_batch(tensors.x, block))["logits"]
        loss, _grad = _kl_loss_and_grad(
            logits,
            tensors.targets[block],
            tensors.mask[block],
            tensors.row_weight[block],
            heads=tensors.heads,
            offsets=tensors.offsets,
            head_weights=config.head_weights,
            temperature=config.temperature,
        )
        total += loss
        blocks += 1
    return total / max(1, blocks) + penalty


# --------------------------------------------------------------------------- #
# The numpy trainer
# --------------------------------------------------------------------------- #
def _per_head_losses(model: StudentModel, tensors: TrainingTensors, indices: Any) -> dict[str, float]:
    """Unweighted, unsoftened-scale KL per head. Diagnostics, not the objective."""
    np = require_numpy()
    idx = np.asarray(indices)
    if idx.size == 0:
        return {head.name: float("nan") for head in tensors.heads}
    logits = _forward(model, _dense_batch(tensors.x, idx))["logits"]
    out: dict[str, float] = {}
    for index, head in enumerate(tensors.heads):
        start = tensors.offsets[head.name]
        stop = start + head.size
        mask = tensors.mask[idx, index] > 0
        if not bool(mask.any()):
            out[head.name] = float("nan")
            continue
        p = tensors.raw[idx][mask][:, start:stop]
        z = logits[mask][:, start:stop]
        if head.kind == "bernoulli":
            q = np.clip(_sigmoid_np(z), 1e-12, 1.0 - 1e-12)
            kl = -(p * np.log(q) + (1.0 - p) * np.log(1.0 - q)).sum(axis=1)
        else:
            q = np.clip(_softmax_np(z), 1e-12, 1.0)
            kl = (p * (np.log(np.clip(p, 1e-12, 1.0)) - np.log(q))).sum(axis=1)
        out[head.name] = round(float(kl.mean()), 6)
    return out


@dataclass(frozen=True)
class _NumpySplit:
    """Which rows the numpy trainer fits on, which it validates on, and what it excluded.

    The holdout exclusion and the validation carve-out live together because both
    are "which rows may this trainer see", and getting either wrong silently
    invalidates every number the pipeline reports afterwards.
    """

    fit_rows: list[TrainingRow]
    val_rows: list[TrainingRow]
    #: How many rows of the dataset were held back and never touched by the trainer.
    holdout_excluded: int
    warnings: list[str]


def _numpy_training_split(dataset: Dataset, config: TrainConfig) -> _NumpySplit:
    """Exclude the holdout, then carve an early-stopping split out of what is left.

    Split out of :func:`_train_numpy` so that function reads as *prepare, fit,
    package*. The two properties that matter are enforced here:

    * the holdout is removed **first**, so the validation split is carved out of
      train rows only and can never reintroduce a held-out row;
    * the validation split is keyed on a differently salted hash of ``request_id``
      (:func:`~jev_route.distill.export.split_bucket` with a ``"val\0"`` prefix), so
      it is stable across runs and uncorrelated with the train/holdout split.
    """
    train_rows = tuple(row for row in dataset.rows if row.split != "holdout")
    holdout_excluded = len(dataset.rows) - len(train_rows)
    if not train_rows:
        raise TrainError(
            "the dataset contains only holdout rows; there is nothing to train on. "
            "Re-export with a smaller --holdout-fraction, or collect more traffic."
        )

    fit_rows, val_rows = list(train_rows), []
    if config.early_stopping and config.val_fraction > 0:
        val_rows = [
            row for row in train_rows if split_bucket("val\x00" + row.request_id) < float(config.val_fraction)
        ]
        if val_rows and len(val_rows) < len(train_rows):
            val_ids = {row.request_id for row in val_rows}
            fit_rows = [row for row in train_rows if row.request_id not in val_ids]
        else:
            val_rows = []
    warnings: list[str] = []
    if config.early_stopping and not val_rows:
        warnings.append(
            "no validation split could be carved (the training set is too small or val_fraction is 0), "
            "so early stopping is disabled and the final epoch's weights are kept"
        )
    return _NumpySplit(fit_rows=fit_rows, val_rows=val_rows, holdout_excluded=holdout_excluded, warnings=warnings)


@dataclass(frozen=True)
class _EpochOutcome:
    """What one run of the epoch loop produced.

    ``last_epoch`` is the epoch the loop stopped on, which :func:`_train_numpy`
    compares against ``best_epoch`` to decide whether the final weights are the
    overfit ones. It is 0 when no epoch ran at all -- the same value the leaked
    loop variable would have had to be ignored by, so the "restore best weights"
    guard below behaves exactly as it did when the loop lived inline.
    """

    history: list[dict[str, float]]
    best_loss: float
    best_params: dict[str, Any]
    best_epoch: int
    stopped_early: bool
    last_epoch: int


def _run_numpy_epochs(
    model: StudentModel,
    tensors: TrainingTensors,
    val_tensors: TrainingTensors | None,
    config: TrainConfig,
    *,
    fit_indices: Any,
    progress: Callable[[int, float, float], None] | None,
    started: float,
) -> _EpochOutcome:
    """The mini-batch loop: Adam over shuffled batches, with early stopping.

    Mutates ``model.params`` in place and returns the history plus the best weights
    seen. Everything random is re-seeded from ``config.seed`` here, so the shuffle
    order is identical to what an inline loop produced: one generator, created once,
    consumed in epoch order.
    """
    import time

    np = require_numpy()
    heads = tensors.heads
    offsets = tensors.offsets
    optimizer = _Optimizer(model.params, config)
    n = tensors.n_rows
    batch = max(1, min(int(config.batch_size), n))
    rng = np.random.default_rng(int(config.seed))
    val_indices = np.arange(val_tensors.n_rows) if val_tensors is not None else np.array([], dtype=np.int64)

    history: list[dict[str, float]] = []
    best_loss = math.inf
    best_params = model.clone_params()
    best_epoch = 0
    last_epoch = 0
    stale = 0
    stopped_early = False

    for epoch in range(1, int(config.epochs) + 1):
        last_epoch = epoch
        order = rng.permutation(fit_indices)
        epoch_loss = 0.0
        steps = 0
        for start in range(0, n, batch):
            idx = order[start : start + batch]
            dense = _dense_batch(tensors.x, idx)
            cache = _forward(model, dense)
            loss, grad_logits = _kl_loss_and_grad(
                cache["logits"],
                tensors.targets[idx],
                tensors.mask[idx],
                tensors.row_weight[idx],
                heads=heads,
                offsets=offsets,
                head_weights=config.head_weights,
                temperature=config.temperature,
            )
            penalty, reg_grads = _regularized(model.params, config.l2)
            grads = _backward(model, cache, grad_logits.astype(np.float32))
            for name, value in reg_grads.items():
                cast = value.astype(np.float32)
                grads[name] = grads[name] + cast if name in grads else cast
            optimizer.apply(model.params, grads)
            epoch_loss += loss + penalty
            steps += 1

        train_loss = epoch_loss / max(1, steps)
        val_loss = _loss_over(model, val_tensors, val_indices, config) if val_tensors is not None else float("nan")
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": round(float(train_loss), 6),
                # round(float("nan"), 6) is nan, so no NaN guard is needed here.
                "val_loss": round(float(val_loss), 6),
                "seconds": round(time.perf_counter() - started, 3),
            }
        )
        if progress is not None:
            progress(epoch, float(train_loss), float(val_loss))

        # val_loss is nan when there is no validation split: fall back to the train
        # curve so early stopping still has something to monitor.
        monitored = float(train_loss) if math.isnan(val_loss) else val_loss
        if monitored < best_loss - 1e-7:
            best_loss = monitored
            best_params = model.clone_params()
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if val_tensors is not None and stale >= max(1, int(config.patience)):
                stopped_early = True
                break

    return _EpochOutcome(
        history=history,
        best_loss=best_loss,
        best_params=best_params,
        best_epoch=best_epoch,
        stopped_early=stopped_early,
        last_epoch=last_epoch,
    )


def _train_numpy(
    dataset: Dataset, config: TrainConfig, progress: Callable[[int, float, float], None] | None
) -> TrainedStudent:
    import time

    np = require_numpy()
    started = time.perf_counter()

    splits = _numpy_training_split(dataset, config)
    fit_rows, val_rows = splits.fit_rows, splits.val_rows
    warnings = list(splits.warnings)

    fit_set = Dataset(rows=tuple(fit_rows), mode=dataset.mode, stats=dict(dataset.stats), source=dataset.source)
    vectorizer = build_vectorizer(fit_set, config)
    tensors = build_tensors(fit_set, vectorizer, config)
    val_tensors = None
    if val_rows:
        val_set = Dataset(rows=tuple(val_rows), mode=dataset.mode, stats=dict(dataset.stats), source=dataset.source)
        val_tensors = build_tensors(val_set, vectorizer, config)
    warnings.extend(check_degenerate(fit_set, tensors, config))

    heads = tensors.heads
    offsets = tensors.offsets
    model = StudentModel.initialize(
        tensors.x.n_features, hidden=config.hidden, activation=config.activation, seed=config.seed, heads=heads
    )
    bias_key = "b" if model.is_linear else "b2"
    model.params[bias_key] = _marginal_bias(fit_set, heads, offsets).astype(np.float32)

    fit_indices = np.arange(tensors.n_rows)
    outcome = _run_numpy_epochs(
        model,
        tensors,
        val_tensors,
        config,
        fit_indices=fit_indices,
        progress=progress,
        started=started,
    )

    if outcome.best_epoch and (
        outcome.stopped_early or (val_tensors is not None and outcome.best_epoch != outcome.last_epoch)
    ):
        # Restore the best-so-far weights. With a validation split the last epoch is
        # usually already past the minimum, and shipping it would mean shipping the
        # overfit one.
        model.restore_params(outcome.best_params)

    for array in model.params.values():
        if not bool(np.isfinite(array).all()):
            raise TrainError(
                "training produced non-finite weights (NaN/Inf). This is divergence, not a data problem: "
                f"lower learning_rate (currently {config.learning_rate}), raise l2, or reduce temperature "
                f"(currently {config.temperature})."
            )

    stats = {
        "rows": len(dataset.rows),
        "train_rows": len(fit_rows),
        "val_rows": len(val_rows),
        "holdout_rows_excluded": splits.holdout_excluded,
        "mode": resolve_train_mode(dataset, config),
        "n_features": int(tensors.x.n_features),
        "usable_rows_per_head": {head.name: int(tensors.mask[:, i].sum()) for i, head in enumerate(heads)},
        "label_support": fit_set.label_support(),
        "epochs_configured": int(config.epochs),
        "epochs_run": len(outcome.history),
        "stopped_early": outcome.stopped_early,
        "best_epoch": outcome.best_epoch,
        "final_train_loss": round(float(outcome.history[-1]["train_loss"]), 6) if outcome.history else None,
        "best_val_loss": round(float(outcome.best_loss), 6) if outcome.best_loss != math.inf else None,
        "final_head_losses": _per_head_losses(model, tensors, fit_indices),
        "training_seconds": round(time.perf_counter() - started, 3),
        "framework": "numpy",
    }
    return TrainedStudent(
        model=model,
        vectorizer=vectorizer,
        config=config,
        mode=stats["mode"],
        history=tuple(outcome.history),
        stats=stats,
        warnings=tuple(warnings),
        teacher_model_versions=dataset.teacher_model_versions(),
        dataset_sha256=str((dataset.stats or {}).get("dataset_sha256", "")),
        dataset_source=dataset.source,
    )


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def train(
    dataset: Dataset | TrainingRow | str | Any,
    config: TrainConfig | Mapping[str, Any] | None = None,
    *,
    progress: Callable[[int, float, float], None] | None = None,
) -> TrainedStudent:
    """Distill a student from an exported dataset.

    Accepts a :class:`~jev_route.distill.export.Dataset`, a path to one, or a
    :class:`~jev_route.distill.artifact.SparseMatrix`-free row iterable. Holdout
    rows are always excluded here, defensively: a trainer that can accidentally see
    its own evaluation set makes every downstream number meaningless, and the
    graduation decision is built on those numbers.
    """
    from pathlib import Path

    cfg = config if isinstance(config, TrainConfig) else TrainConfig(**dict(config or {}))
    if isinstance(dataset, (str, Path)):
        dataset = load_dataset(dataset)
    elif isinstance(dataset, TrainingRow):
        dataset = Dataset(rows=(dataset,), mode="features")
    elif not isinstance(dataset, Dataset):
        rows = tuple(dataset)
        if rows and not isinstance(rows[0], TrainingRow):
            raise TrainError(
                f"train() expects a Dataset, a path, or TrainingRow objects; got {type(rows[0]).__name__}"
            )
        mode = "text" if any(getattr(r, "text", None) for r in rows) else "features"
        dataset = Dataset(rows=rows, mode=mode)

    framework = str(cfg.framework or "numpy").lower()
    if framework == "torch":
        return train_torch(dataset, cfg, progress=progress)
    if framework != "numpy":
        raise TrainError(
            f"unknown training framework {cfg.framework!r}; this build supports 'numpy' (default) and 'torch'"
        )
    if cfg.encoder:
        # The numpy trainer has nowhere to put encoder weights, so the alternative to this
        # refusal is a silent no-op -- and `config.to_dict()` is sealed into the artifact's
        # metadata["training"], which would make a hash-verified artifact claim a transformer
        # student it does not contain. :func:`_require_torch` refuses the same request on the
        # torch path with different wording, because the reason is different: there it is what
        # serving a transformer would cost, here it is a provenance lie in the artifact.
        raise TrainError(
            f"encoder={cfg.encoder!r} cannot be trained by the numpy trainer, which fits a small "
            "student over the exported features and has nowhere to put transformer weights. "
            "Refusing rather than ignoring it: the training config is sealed into the artifact's "
            "metadata['training'], so a silently dropped encoder would ship an artifact claiming "
            "a transformer student it does not contain. Unset `encoder` to train the numpy "
            "student, or serve a transformer student behind your own DecisionBackend."
        )
    return _train_numpy(dataset, cfg, progress)


def _require_torch(cfg: TrainConfig) -> Any:
    """Refuse an encoder student, then return the torch module or an actionable error.

    Split out of :func:`train_torch` because both halves are about *whether this
    trainer may run at all*, and the encoder refusal has to happen before the
    import: an operator without torch should still be told that a transformer
    student is not a format this project ships, rather than being told to install
    a 2 GB dependency for something that would be rejected anyway.
    """
    if cfg.encoder:
        raise TrainError(
            f"encoder={cfg.encoder!r} is not supported by the distillation artifact format. "
            "A fine-tuned transformer needs a tokenizer and hundreds of megabytes of weights, and serving it "
            "would put torch in the request path -- which is exactly what graduating is supposed to remove. "
            "Serve a transformer student behind your own DecisionBackend instead, or train the small "
            "numpy-compatible model (unset `encoder`)."
        )
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise DistillDependencyError(
            "framework='torch' needs PyTorch. Install it with:  pip install 'jev-route[distill]'  "
            "-- or use the default numpy trainer (framework='numpy'), which needs nothing extra."
        ) from exc
    return torch


def _run_torch_epochs(
    torch: Any,
    net: Any,
    tensors: TrainingTensors,
    cfg: TrainConfig,
    *,
    device: Any,
    optimizer: Any,
    head_weight_vector: Any,
    progress: Callable[[int, float, float], None] | None,
    started: float,
) -> list[dict[str, float]]:
    """The torch mini-batch loop. Same objective as :func:`_kl_loss_and_grad`, autodiff'd.

    There is no validation split here and therefore no early stopping: ``val_loss``
    is nan on every epoch, which is what tells :func:`_train_numpy`'s counterpart
    numbers apart in the artifact. The shuffle generator is seeded from
    ``cfg.seed`` inside, so a re-run reproduces the same epoch order.
    """
    import time

    heads = tensors.heads
    offsets = tensors.offsets
    targets = torch.as_tensor(tensors.targets, dtype=torch.float32, device=device)
    mask = torch.as_tensor(tensors.mask, dtype=torch.float32, device=device)
    row_weight = torch.as_tensor(tensors.row_weight, dtype=torch.float32, device=device)

    n = tensors.n_rows
    batch = max(1, min(int(cfg.batch_size), n))
    generator = torch.Generator().manual_seed(int(cfg.seed))
    history: list[dict[str, float]] = []
    temperature = max(1e-6, float(cfg.temperature))

    for epoch in range(1, int(cfg.epochs) + 1):
        order = torch.randperm(n, generator=generator).numpy()
        running = 0.0
        steps = 0
        for start in range(0, n, batch):
            idx = order[start : start + batch]
            dense = torch.as_tensor(_dense_batch(tensors.x, idx), dtype=torch.float32, device=device)
            logits = net(dense)
            loss = torch.zeros((), device=device)
            for h, head in enumerate(heads):
                lo, hi = offsets[head.name], offsets[head.name] + head.size
                m = mask[idx, h] * row_weight[idx]
                denom = m.sum().clamp_min(1e-9)
                p = targets[idx][:, lo:hi]
                if head.kind == "bernoulli":
                    log_q = torch.logsigmoid(logits[:, lo:hi] / temperature)
                    kl = -(p * log_q + (1.0 - p) * (log_q - logits[:, lo:hi] / temperature)).sum(dim=1)
                else:
                    log_q = torch.log_softmax(logits[:, lo:hi] / temperature, dim=-1)
                    kl = (p * (torch.log(p.clamp_min(1e-12)) - log_q)).sum(dim=-1)
                loss = loss + float(head_weight_vector[h]) * (temperature**2) * (kl * m).sum() / denom
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
            steps += 1
        train_loss = running / max(1, steps)
        history.append({"epoch": float(epoch), "train_loss": round(train_loss, 6), "val_loss": float("nan"),
                        "seconds": round(time.perf_counter() - started, 3)})
        if progress is not None:
            progress(epoch, train_loss, float("nan"))
    return history


def _student_from_torch(net: Any, *, heads: Sequence[HeadSpec], n_features: int, hidden: int) -> StudentModel:
    """Copy trained torch weights back into the numpy :class:`StudentModel`.

    This is the step that makes "trained with torch" not mean "requires torch to
    route": the served artifact is the same format either trainer produces. torch
    stores ``Linear`` weights transposed relative to this format, hence the ``.T``;
    ``Sequential`` indexes the layers positionally, hence ``"0.*"``/``"2.*"`` (index
    1 is the activation, which has no weights).
    """
    np = require_numpy()
    params = {name: value.detach().cpu().numpy().astype(np.float32) for name, value in net.state_dict().items()}
    if hidden > 0:
        model = StudentModel(
            heads=heads,
            n_features=n_features,
            hidden=hidden,
            activation="tanh",
            params={"W1": params["0.weight"].T.copy(), "b1": params["0.bias"].copy(),
                    "W2": params["2.weight"].T.copy(), "b2": params["2.bias"].copy()},
        )
    else:
        model = StudentModel(
            heads=heads,
            n_features=n_features,
            hidden=0,
            params={"W": params["0.weight"].T.copy(), "b": params["0.bias"].copy()},
        )
    model._check_shapes()
    return model


def train_torch(
    dataset: Dataset,
    config: TrainConfig | None = None,
    *,
    progress: Callable[[int, float, float], None] | None = None,
) -> TrainedStudent:
    """The same model and the same objective, with torch doing the autodiff.

    Exists for operators whose decision log is large enough that the numpy
    mini-batch loop becomes the bottleneck (a GPU or a many-core box then buys real
    time). It produces the *identical* artifact: the trained weights are copied back
    into numpy arrays, so :mod:`jev_route.backends.distilled` serves a torch-trained
    model without torch being installed at serve time. That property is the reason
    this exists at all -- "trained with torch" must not become "requires torch to
    route".

    ``config.encoder`` (a ModernBERT-class transformer) is deliberately rejected
    rather than half-supported. A fine-tuned encoder cannot be stored in this
    artifact format: it is hundreds of megabytes, it needs a tokenizer, and serving
    it would put a deep-learning runtime on the request path of every route. The
    project's promise is that the local end state is small, auditable, and boring,
    so if you want a transformer student, serve it behind your own
    :class:`~jev_route.backends.base.DecisionBackend` and point the policy at that.
    Failing loudly here beats shipping a format we would have to support forever.
    """
    cfg = config or TrainConfig()
    torch = _require_torch(cfg)

    import time

    np = require_numpy()
    started = time.perf_counter()
    warnings: list[str] = []

    train_rows = tuple(row for row in dataset.rows if row.split != "holdout")
    if not train_rows:
        raise TrainError("the dataset contains only holdout rows; there is nothing to train on")
    train_set = Dataset(rows=train_rows, mode=dataset.mode, stats=dict(dataset.stats), source=dataset.source)
    vectorizer = build_vectorizer(train_set, cfg)
    tensors = build_tensors(train_set, vectorizer, cfg)
    warnings.extend(check_degenerate(train_set, tensors, cfg))

    device = torch.device(cfg.device or "cpu")
    n_out = sum(head.size for head in tensors.heads)
    layers: list[Any] = []
    if cfg.hidden > 0:
        layers += [torch.nn.Linear(tensors.x.n_features, cfg.hidden), torch.nn.Tanh()]
    layers += [torch.nn.Linear(tensors.x.n_features if cfg.hidden <= 0 else cfg.hidden, n_out)]
    net = torch.nn.Sequential(*layers).to(device)

    heads = tensors.heads
    offsets = tensors.offsets
    head_weight_vector = np.asarray(
        [float(cfg.head_weights.get(head.name, 1.0)) for head in heads], dtype=np.float32
    )
    with torch.no_grad():  # start from the empirical prior, as the numpy trainer does
        prior = _marginal_bias(train_set, heads, offsets)
        layers[-1].bias.copy_(torch.as_tensor(prior, dtype=torch.float32, device=device))

    optimizer = torch.optim.Adam(net.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.l2))
    n = tensors.n_rows
    history = _run_torch_epochs(
        torch,
        net,
        tensors,
        cfg,
        device=device,
        optimizer=optimizer,
        head_weight_vector=head_weight_vector,
        progress=progress,
        started=started,
    )

    model = _student_from_torch(net, heads=heads, n_features=tensors.x.n_features, hidden=cfg.hidden)

    return TrainedStudent(
        model=model,
        vectorizer=vectorizer,
        config=cfg,
        mode=resolve_train_mode(dataset, cfg),
        history=tuple(history),
        stats={
            "rows": len(dataset.rows),
            "train_rows": n,
            "val_rows": 0,
            "holdout_rows_excluded": len(dataset.rows) - n,
            "mode": resolve_train_mode(dataset, cfg),
            "n_features": int(tensors.x.n_features),
            "usable_rows_per_head": {head.name: int(tensors.mask[:, i].sum()) for i, head in enumerate(heads)},
            "label_support": train_set.label_support(),
            "epochs_configured": int(cfg.epochs),
            "epochs_run": len(history),
            "stopped_early": False,
            "best_epoch": len(history),
            "final_train_loss": round(history[-1]["train_loss"], 6) if history else None,
            "best_val_loss": None,
            "final_head_losses": _per_head_losses(model, tensors, np.arange(n)),
            "training_seconds": round(time.perf_counter() - started, 3),
            "framework": f"torch/{torch.__version__}",
            "device": str(device),
        },
        warnings=tuple(warnings),
        teacher_model_versions=dataset.teacher_model_versions(),
        dataset_sha256=str((dataset.stats or {}).get("dataset_sha256", "")),
        dataset_source=dataset.source,
    )


__all__ = [
    "DEFAULT_HEAD_WEIGHTS",
    "DEFAULT_SEED",
    "TrainConfig",
    "TrainError",
    "TrainedStudent",
    "TrainingTensors",
    "build_tensors",
    "build_vectorizer",
    "check_degenerate",
    "resolve_train_mode",
    "soften_distribution",
    "soften_scalar",
    "train",
    "train_torch",
    "vectorize",
]
