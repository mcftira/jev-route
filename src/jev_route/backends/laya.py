"""LayaBackend: the fully local, air-gapped decision backend.

This is the local edition of the backend seam. Where :mod:`.jev` sends the
redacted excerpt to TypeSafe's cloud and :mod:`.mock` answers from a
deterministic table, :class:`LayaBackend` runs the open-source
Laya System One decision model (``convaiinnovations/laya`` on Hugging Face,
arXiv:2503.23303, Apache-2.0) on local hardware. No API key, no cloud, no
egress: the model is a local checkpoint and the only "call" is a forward pass
on the machine that hosts the router.

Why a third backend, not a fork
-------------------------------
The :class:`~jev_route.backends.base.DecisionBackend` protocol is the seam that
makes this possible. The router, the policy engine, the gate, the cache and the
decision log all speak to the protocol and never learn which backend answered.
Adding Laya is therefore an *addition* to the core, not a divergent copy: the
same policy YAML, the same log schema, the same evals, and a one-line
``backend.name`` swap. That is the graduation story in its local flavour --
instead of distilling a small model from your own log, you start from a
pretrained System One student and *calibrate* it on your traffic.

The question mapping is native, not an adapter
----------------------------------------------
Laya scores ``[MASK]``-per-option questions of three types (``choice`` /
``score`` / ``noul``). The exact ``{"type", "instructions", "criteria"}``
mapping that :func:`.jev.build_questions` emits for our four routing questions
is the exact shape Laya's ``Agent.system_one`` consumes. So all four questions
go into **one** forward pass per decision and come back as per-question
probability distributions -- the same full-soft-distribution contract every
other backend honours, which is what keeps the decision log comparable across
backends and keeps distillation working.

The two invariants from :mod:`.base` hold exactly: the schema is total (an
unanswerable question comes back at maximum uncertainty, never omitted) and an
outage degrades to a :class:`BackendResult` rather than raising.

Calibration is a hard precondition of ``enforce``
-------------------------------------------------
Laya ships overconfident: its raw probabilities are peaked and its built-in
temperature is not fit on our routing task. The policy engine thresholds on
confidence floors that only mean something for a *calibrated* distribution, so
an uncalibrated Laya in ``enforce`` mode would route on numbers that are lying.
``LayaBackend`` therefore refuses to construct in ``enforce`` mode without a
fitted :class:`~jev_route.backends.laya_calibration.CalibrationArtifact`
(fail-closed, exactly like :class:`.jev` refusing without an API key).
``shadow`` mode -- where Laya's answers are logged and compared but do not
drive the decision -- is always allowed, uncalibrated, because there the honest
answer is "we are measuring how bad the raw numbers are."
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..schema import (
    COMPLEXITY_LEVELS,
    DOMAINS,
    SENSITIVITY_LEVELS,
    DecisionAnswers,
)
from .base import BackendError, BackendResult, CircuitBreaker, DecisionRequest
from .jev import _normalize_choice, _normalize_noul, build_questions

#: Checkpoints Laya ships, and the (hub repo, subfolder) that resolves each.
#: ``english`` is the default; ``multilingual`` is selected automatically when
#: the excerpt is non-English; ``typed-decisions`` is an explicit opt-in only.
CHECKPOINTS: dict[str, tuple[str, str | None]] = {
    "english": ("convaiinnovations/laya", None),
    "multilingual": ("convaiinnovations/laya", "multilingual"),
    "typed-decisions": ("convaiinnovations/laya", "typed-decisions"),
}

#: The state budget default. The English checkpoint's sequence budget is 512
#: tokens; 448 leaves headroom for the question head (type + instructions + the
#: rendered options) so the *excerpt* is what gets truncated, never the question.
DEFAULT_TOKEN_BUDGET = 448
DEFAULT_MAX_LEN = 512
DEFAULT_HEAD_MAX_LEN = 192

#: The explicit marker inserted when the excerpt is truncated. It is logged, not
#: implied: a routing input that was cut is visible in the decision record.
TRUNCATION_MARKER = " ... "


class ExcerptBudget:
    """The deterministic, logged excerpt strategy against a Laya question head.

    Laya's encoder has a hard sequence length. The question head (type,
    instructions, and every rendered option) is fixed per question; the state --
    our redacted excerpt -- is the only part we may shrink. We shrink it
    deterministically and *never silently*: the caller records the strategy and
    how many tokens were dropped on the decision, so a truncated routing input
    is visible in the log rather than a quiet accuracy cliff.

    The strategy is head+tail: keep the first half of the budget from the start
    of the excerpt and the second half from the end, with the explicit
    truncation marker between them. Head+tail beats head-only because a routing
    prompt's intent usually lives at the start ("do X") and its payload at the
    end ("... with this data"); dropping the middle loses the least signal.
    """

    def __init__(self, tokenizer: Any, *, max_len: int, head_max_len: int) -> None:
        self._tok = tokenizer
        self._max_len = int(max_len)
        self._head_max_len = int(head_max_len)

    def overhead_tokens(self, question: Mapping[str, Any]) -> int:
        """Tokens the question head consumes with an *empty* state.

        Mirrors ``laya.common.build_sequence``: ``[CLS] head [SEP]`` followed by
        each ``[MASK] option`` (options capped at 48 tokens each, the whole
        options block capped by ``head_max_len``), then a closing ``[SEP]``. We
        recompute it against our own tokenizer so the budget is exact for the
        checkpoint actually loaded, rather than trusting a constant that could
        drift from the weights.
        """
        tok = self._tok
        qtype = str(question.get("type", "choice"))
        instructions = question.get("instructions", "")
        if not isinstance(instructions, str):
            instructions = json.dumps(instructions, ensure_ascii=False)
        head_ids = list(tok(f"{qtype} question: {instructions}", add_special_tokens=False)["input_ids"])

        opt_ids: list[list[int]] = []
        for opt in _render_option_texts(question):
            ids = list(tok(" " + opt, add_special_tokens=False)["input_ids"])[:48]
            opt_ids.append([tok.mask_token_id, *ids])
        opt_budget = self._head_max_len - sum(len(o) for o in opt_ids)
        if opt_budget < 16:
            per = max(4, (self._head_max_len - 16) // max(1, len(opt_ids)))
            opt_ids = [o[:per] for o in opt_ids]
            opt_budget = self._head_max_len - sum(len(o) for o in opt_ids)
        head_ids = head_ids[: max(8, opt_budget)]

        n = 1 + len(head_ids) + 1  # [CLS] head [SEP]
        for o in opt_ids:
            n += len(o)
        n += 1  # closing [SEP] before the state
        return n

    def budget(self, text: str, question: Mapping[str, Any], *, token_budget: int) -> tuple[str, dict[str, Any]]:
        """Fit ``text`` into the state room, returning ``(state, report)``.

        ``report`` carries the strategy and token counts for the decision log.
        When the excerpt already fits, the strategy is ``"none"`` and the state
        is byte-identical to the input -- that is the common case.
        """
        overhead = self.overhead_tokens(question)
        # One trailing [SEP] after the state; stay at or under the encoder limit.
        room = max(1, self._max_len - overhead - 1)
        effective_budget = max(1, min(int(token_budget), room))

        ids = list(self._tok(text, add_special_tokens=False)["input_ids"])
        if len(ids) <= effective_budget:
            return text, {
                "strategy": "none",
                "tokens_total": len(ids),
                "tokens_kept": len(ids),
                "tokens_dropped": 0,
                "token_budget": effective_budget,
            }

        marker_ids = list(self._tok(TRUNCATION_MARKER, add_special_tokens=False)["input_ids"]) or [0]
        keep_total = max(2, effective_budget - len(marker_ids))
        head_keep = keep_total // 2
        tail_keep = keep_total - head_keep
        final_ids = ids[:head_keep] + marker_ids + ids[-tail_keep:]
        return self._tok.decode(final_ids), {
            "strategy": "head+tail",
            "tokens_total": len(ids),
            "tokens_kept": keep_total,
            "tokens_dropped": max(0, len(ids) - keep_total),
            "token_budget": effective_budget,
        }


def _render_option_texts(question: Mapping[str, Any]) -> list[str]:
    """Render the option texts for a question, matching Laya's ``render_options``."""
    qtype = str(question.get("type", "choice"))
    crit = question.get("criteria", {})
    if qtype == "choice":
        if isinstance(crit, (list, tuple)):
            return [str(c) for c in crit]
        if isinstance(crit, Mapping):
            return [(k if (v is None or v == "") else f"{k}: {_render_criterion(v)}") for k, v in crit.items()]
        return []
    if qtype == "score":
        seq = crit if isinstance(crit, (list, tuple)) else (list(crit.values()) if isinstance(crit, Mapping) else [])
        return [f"level {i}: {_render_criterion(c)}" for i, c in enumerate(seq)]
    crit = crit if isinstance(crit, Mapping) else {}
    false_crit, true_crit = crit.get("false"), crit.get("true")
    return [
        "false: "
        + (_render_criterion(false_crit) if false_crit not in (None, "") else "no, the statement does not hold"),
        "true: " + (_render_criterion(true_crit) if true_crit not in (None, "") else "yes, the statement holds"),
    ]


def _render_criterion(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


class LayaBackend:
    """A :class:`DecisionBackend` served by a local Laya checkpoint.

    Args:
        model: checkpoint name (``english`` / ``multilingual`` /
            ``typed-decisions``) or ``auto`` to pick by detected excerpt language.
        model_path: a local checkpoint directory. When set, loading is fully
            offline; when ``None`` the hub id is resolved (first use downloads).
        device: ``auto`` / ``cpu`` / ``mps`` / ``cuda``. ``auto`` prefers cuda,
            then mps, then cpu, and falls back to cpu with a warning on failure.
        token_budget: max tokens the excerpt may occupy (default 448).
        max_len / head_max_len: the checkpoint's sequence and head budgets.
        role: ``enforce`` (Laya's calibrated answers drive the decision) or
            ``shadow`` (answers are logged/compared, not authoritative).
        calibration: a loaded :class:`CalibrationArtifact` or a path to one.
            Required when ``role == "enforce"``; optional in ``shadow``.
        include_domain: mirror of the Jev option; drop the domain question when
            no policy rule reads it.
        agent_factory: injectable per-checkpoint loader, ``factory(checkpoint) ->
            agent``. Tests pass a mock so the core suite never needs
            torch/transformers/laya installed.
        tokenizer: injectable per-checkpoint tokenizer provider,
            ``provider(checkpoint) -> tokenizer``. Defaults to the checkpoint's own.
        language_fn: injectable ``text -> language_name`` for ``auto`` routing.
            Defaults to Laya's own script/language detection.
    """

    name = "laya"

    def __init__(
        self,
        *,
        model: str = "english",
        model_path: str | None = None,
        device: str = "auto",
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        max_len: int = DEFAULT_MAX_LEN,
        head_max_len: int = DEFAULT_HEAD_MAX_LEN,
        role: str = "enforce",
        calibration: Any = None,
        include_domain: bool = True,
        breaker: CircuitBreaker | None = None,
        agent_factory: Callable[[str], Any] | None = None,
        tokenizer: Callable[[str], Any] | None = None,
        language_fn: Callable[[str], str] | None = None,
    ) -> None:
        self.role = str(role).lower()
        if self.role not in ("enforce", "shadow"):
            raise BackendError(f"LayaBackend role must be 'enforce' or 'shadow', got {role!r}")

        self.checkpoint = self._resolve_checkpoint(model)
        self.model_path = model_path
        self.device = device
        self.token_budget = int(token_budget)
        self.max_len = int(max_len)
        self.head_max_len = int(head_max_len)
        self.include_domain = bool(include_domain)
        self.breaker = breaker or CircuitBreaker()
        self._agent_factory = agent_factory
        self._tokenizer_provider = tokenizer
        self._language_fn = language_fn
        self.calibration = calibration
        self._agents: dict[str, Any] = {}
        self._toks: dict[str, Any] = {}
        self._owns_agents = agent_factory is None
        self.model_version = self._initial_model_version()
        self.load_latency_ms: dict[str, float] = {}
        # Inference serializes on ONE dedicated worker thread. MPS (Apple GPU)
        # aborts the whole process when two threads encode command buffers at
        # once -- verified on M5 Pro / torch 2.14: concurrent asyncio.to_thread
        # workers SIGABRT mid-batch. One worker keeps the event loop free while
        # making the encode single-threaded; a 421M-parameter model does not
        # benefit from overlapping forward passes on one accelerator anyway.
        self._infer_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="laya-infer"
        )

        # Fail-closed: an uncalibrated Laya must not drive decisions. This is the
        # local-edition analogue of JevBackend refusing to start without a key.
        if self.role == "enforce" and calibration is None:
            raise BackendError(
                "LayaBackend in role 'enforce' requires a fitted calibration artifact; "
                "role 'shadow' is allowed without one. Fit one with the laya-route "
                "`calibrate` command, or run in shadow while you measure."
            )

    # -- checkpoint resolution -------------------------------------------- #
    @staticmethod
    def _resolve_checkpoint(model: str) -> str:
        key = str(model).strip().lower()
        if key in ("auto", "default", ""):
            return "auto"
        aliases = {
            "en": "english", "laya": "english",
            "multi": "multilingual", "ml": "multilingual",
            "typed": "typed-decisions", "typed_decisions": "typed-decisions",
        }
        key = aliases.get(key, key)
        if key not in CHECKPOINTS:
            raise BackendError(
                f"unknown Laya checkpoint {model!r}; choose one of {sorted(CHECKPOINTS)} or 'auto'"
            )
        return key

    def _initial_model_version(self) -> str:
        if self.checkpoint == "auto":
            return "laya/auto"
        repo, sub = CHECKPOINTS[self.checkpoint]
        return f"laya/{repo.split('/')[-1]}" + (f":{sub}" if sub else "")

    def _effective_checkpoint(self, excerpt: str) -> str:
        if self.checkpoint != "auto":
            return self.checkpoint
        lang = (self._language_fn or _default_language)(excerpt)
        return "multilingual" if _is_non_english(lang) else "english"

    # -- model loading (lazy, per checkpoint) ----------------------------- #
    def _load(self, checkpoint: str) -> tuple[Any, Any]:
        if checkpoint in self._agents:
            return self._agents[checkpoint], self._toks[checkpoint]

        if self._agent_factory is not None:
            agent = self._agent_factory(checkpoint)
            tok = (self._tokenizer_provider or (lambda _cp: getattr(agent, "tok", None)))(checkpoint)
        else:
            from laya import load as laya_load  # lazy: torch/transformers/laya are heavy

            repo, sub = CHECKPOINTS[checkpoint]
            started = time.perf_counter()
            agent = laya_load(
                self.model_path or repo,
                device=None if self.device == "auto" else self.device,
                subfolder=None if self.model_path else sub,
            )
            self.load_latency_ms[checkpoint] = (time.perf_counter() - started) * 1000.0
            tok = self._tokenizer_provider(checkpoint) if self._tokenizer_provider else getattr(agent, "tok", None)

        if tok is None:
            raise BackendError(
                f"Laya checkpoint {checkpoint!r} has no usable tokenizer; the excerpt "
                "budget cannot be computed. Pass a tokenizer or a real checkpoint."
            )
        self._agents[checkpoint] = agent
        self._toks[checkpoint] = tok
        return agent, tok

    def preload(self, checkpoints: list[str] | None = None) -> None:
        """Load the checkpoint(s) now so no download happens during ``decide``.

        Call this at startup to make the no-network-at-inference guarantee
        explicit: after ``preload`` returns, ``decide`` is a pure forward pass.
        """
        names = checkpoints or ([self.checkpoint] if self.checkpoint != "auto" else list(CHECKPOINTS))
        for c in names:
            self._load(c)

    def budgeter(self, checkpoint: str) -> ExcerptBudget:
        _, tok = self._load(checkpoint)
        return ExcerptBudget(tok, max_len=self.max_len, head_max_len=self.head_max_len)

    # -- public API ------------------------------------------------------- #
    async def decide(self, request: DecisionRequest) -> BackendResult:
        """Answer the routing questions with one local forward pass.

        Never raises for an outage; degrades to maximum uncertainty instead,
        exactly like :meth:`.jev.JevBackend.decide`.
        """
        questions = build_questions(include_domain=self.include_domain)
        started = time.perf_counter()

        if not self.breaker.allow():
            return self._degraded(questions, started, f"circuit open (state={self.breaker.state})")

        try:
            excerpt = str(request.state().get("prompt_excerpt", ""))
            checkpoint = self._effective_checkpoint(excerpt)
            agent, _ = self._load(checkpoint)
            state, budget_report = self._build_state(request, questions, checkpoint)
            loop = asyncio.get_running_loop()
            raw = await loop.run_in_executor(
                self._infer_executor, agent.system_one, state, questions
            )
        except Exception as exc:  # a local crash is an outage, not a programmer error
            self.breaker.on_failure()
            return self._degraded(questions, started, f"{type(exc).__name__}: {str(exc)[:160]}")

        self.breaker.on_success()
        elapsed = (time.perf_counter() - started) * 1000.0

        answers = _parse_laya_answers(raw)
        if self.calibration is not None:
            answers = self.calibration.apply_to(answers)

        questions_sent = {
            **questions,
            "_laya": {
                "checkpoint": checkpoint,
                "role": self.role,
                "calibrated": self.calibration is not None,
                "excerpt": budget_report,
            },
        }
        return BackendResult(
            answers=answers,
            model_version=self.model_version,
            questions_sent=questions_sent,
            latency_ms=round(elapsed, 3),
        )

    def _build_state(self, request: DecisionRequest, questions: Mapping[str, Any], checkpoint: str):
        """Token-budget the redacted excerpt and build the state payload.

        The state carries the same fields Jev receives (so the two backends see
        an identical, comparable input), but the excerpt is pre-fitted to Laya's
        sequence budget with the strategy recorded in the returned report.
        """
        budgeter = self.budgeter(checkpoint)
        base = request.state()
        excerpt = str(base.get("prompt_excerpt", ""))
        # Budget against the *largest* question head so every question fits; the
        # excerpt text is identical across questions, so budget once on the
        # worst-case head and reuse the resulting state for all four.
        worst = max(questions.values(), key=lambda q: budgeter.overhead_tokens(q))
        state_text, report = budgeter.budget(excerpt, worst, token_budget=self.token_budget)
        state = dict(base)
        state["prompt_excerpt"] = state_text
        return state, report

    def _degraded(self, questions: Mapping[str, Any], started: float, reason: str) -> BackendResult:
        return BackendResult(
            answers=DecisionAnswers.unknown(),
            model_version=self.model_version,
            questions_sent=dict(questions),
            latency_ms=round((time.perf_counter() - started) * 1000.0),
            degraded=True,
            degrade_reason=reason,
        )

    async def aclose(self) -> None:
        self._agents.clear()
        self._toks.clear()
        self._infer_executor.shutdown(wait=False, cancel_futures=True)


def _default_language(text: str) -> str:
    from laya.lang import analyse  # lazy: only needed for `auto` routing

    return str(analyse(text).get("language", "en"))


def _is_non_english(lang: str) -> bool:
    return str(lang).strip().lower()[:2] not in ("en", "")


def _parse_laya_answers(raw: Any) -> DecisionAnswers:
    """Map a Laya ``system_one`` result onto :class:`DecisionAnswers`.

    Total function: any malformed body degrades per-question to maximum
    uncertainty rather than raising, so the schema stays total.
    """
    if not isinstance(raw, Mapping):
        return DecisionAnswers.unknown()
    answers = raw.get("answers")
    if not isinstance(answers, Mapping):
        return DecisionAnswers.unknown()

    def pick(qid: str) -> Mapping[str, Any]:
        v = answers.get(qid)
        return v if isinstance(v, Mapping) else {}

    return DecisionAnswers(
        complexity=_normalize_choice(pick("complexity"), COMPLEXITY_LEVELS, "standard"),
        sensitivity=_normalize_choice(pick("sensitivity"), SENSITIVITY_LEVELS, "internal"),
        pii=_normalize_noul(pick("pii_present")),
        domain=_normalize_choice(pick("domain"), DOMAINS, "chat"),
    )


__all__ = [
    "CHECKPOINTS",
    "DEFAULT_HEAD_MAX_LEN",
    "DEFAULT_MAX_LEN",
    "DEFAULT_TOKEN_BUDGET",
    "TRUNCATION_MARKER",
    "ExcerptBudget",
    "LayaBackend",
]
