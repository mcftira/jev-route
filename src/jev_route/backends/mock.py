"""MockBackend: a deterministic, offline decision backend.

Exists so that jev-route is runnable with no API key, no network, and no account
-- which is what makes the test suite trustworthy, the CI meaningful, and the
README's quickstart actually copy-pasteable. It is not a stub that returns
constants: it produces real probability distributions from real request features,
so every downstream consumer (policy thresholds, confidence floors, decision
logging, distillation, shadow mode) is exercised against plausible output rather
than a hardcoded argmax.

It is *deliberately* less accurate than Jev. That is the honest trade: zero cost
and zero data egress in exchange for a coarse read. If you need accuracy on day
one, use :class:`~jev_route.backends.jev.JevBackend`; if you need to develop
against the pipeline without spending money or sending text anywhere, use this.

Determinism contract: the same input always yields the same output, in any
process, on any machine. No randomness, no clock, no counter.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..schema import (
    COMPLEXITY_LEVELS,
    DOMAINS,
    SENSITIVITY_LEVELS,
    ChoiceAnswer,
    DecisionAnswers,
    NoulAnswer,
)
from .base import BackendResult, DecisionRequest

MODEL_VERSION = "mock-1.0.0"

#: Deliberately small so the mock's distributions are peaked enough to be useful
#: but never degenerate -- a backend that always answers 1.0 would make the
#: confidence-floor code paths untestable.
TEMPERATURE = 0.55

_COMPLEXITY_MARKERS: tuple[tuple[str, float], ...] = (
    (r"\b(?:prove|proof|derive|show that)\b", 1.5),
    (r"\b(?:design|architect|architecture)\b", 1.2),
    (r"\b(?:optimi[sz]e|performance|bottleneck|p9[59]|latency)\b", 1.0),
    (r"\b(?:trade[- ]?offs?|compare and contrast|evaluate the)\b", 1.0),
    (r"\b(?:debug|root cause|why is|intermittent|race condition|deadlock)\b", 1.1),
    (r"\b(?:migrat|refactor|distributed|consensus|sharding|concurrency)\b", 1.2),
    (r"\b(?:research|survey|literature|state of the art)\b", 1.3),
    (r"\b(?:translate|rewrite|rephrase|fix the typo|summariz)\b", -1.2),
    (r"\b(?:hello|hi|hey|thanks|thank you|ok|okay)\b", -1.8),
    (r"\b(?:what is|define|list|name)\b", -0.7),
)

_SENSITIVITY_MARKERS: tuple[tuple[str, float], ...] = (
    (r"\b(?:patient|health|medical|clinical|diagnos|prescription)\b", 1.1),
    (r"\b(?:gdpr|hipaa|pci|dsgvo|compliance|regulat)\b", 0.8),
    (r"\b(?:salary|compensation|performance review|layoff|hr case)\b", 1.0),
    (r"\b(?:secret|credential|password|api key|token|vulnerability|cve)\b", 1.1),
    (r"\b(?:nda|confidential|trade secret|unreleased|internal only)\b", 0.9),
    (r"\b(?:bank|account number|invoice|payment|billing)\b", 0.5),
    (r"\b(?:public blog|documentation|readme|open source|announce)\b", -1.0),
    (r"\b(?:weather|joke|recipe|poem|trivia)\b", -1.3),
)

#: Asking ABOUT a regulated topic is not the same as carrying regulated data.
#: These patterns are the reason a mock can be crude without being wrong on the
#: single most important distinction in the project: "explain how HIPAA works"
#: contains no patient data and must not be air-gapped. Without them the mock
#: reproduces exactly the keyword-router failure jev-route exists to fix, and the
#: offline demo -- the first thing anyone runs -- would contradict the README.
_ABOUT_NOT_DATA: tuple[tuple[str, float], ...] = (
    (r"\bexplain (?:how|what|why)\b", -1.6),
    (r"\bwhat (?:is|are|does|do)\b.{0,60}\b(?:require|mean|apply|cover|count)\b", -1.5),
    (r"\bhow does .{0,40} work\b", -1.5),
    (r"\bwho it applies to\b|\bapplies to whom\b", -1.4),
    (r"\b(?:primer|intro|introduction|overview|guide|tutorial|cheat ?sheet|faq)\b", -1.3),
    (r"\bwhat everyone gets wrong\b|\bcommon misconceptions?\b", -1.2),
    (r"\bdo i need to comply\b|\bare we compliant\b", -0.9),
    (r"\bwhat should\b.{0,50}\b(?:say|read|look like|be|include)\b", -1.2),
    (r"\bwhat copy should\b|\bwhat wording\b|\bwhat should the (?:copy|text|banner)\b", -1.2),
)

_DOMAIN_MARKERS: tuple[tuple[str, str, float], ...] = (
    (r"\b(?:code|function|class|bug|test|refactor|compile|stack trace|regex|api)\b", "code", 1.4),
    (r"\b(?:write|draft|blog|email to|copy|essay|rephrase|tone|headline)\b", "writing", 1.3),
    (r"\b(?:analy[sz]e|compare|calculate|estimate|forecast|why|evaluate|metrics)\b", "analysis", 1.2),
    (r"\b(?:extract|parse|json|csv|table|scrape|convert)\b", "data-extraction", 1.3),
    (r"\b(?:hello|hi|hey|how are you|thanks|chat|talk)\b", "chat", 1.2),
)


def _softmax(scores: Mapping[str, float], temperature: float = TEMPERATURE) -> dict[str, float]:
    """Softmax over named scores. ``temperature`` near 0 sharpens, large flattens."""
    if not scores:
        return {}
    t = max(1e-6, temperature)
    scaled = {k: v / t for k, v in scores.items()}
    top = max(scaled.values())
    exps = {k: math.exp(v - top) for k, v in scaled.items()}
    total = sum(exps.values()) or 1.0
    return {k: round(v / total, 6) for k, v in exps.items()}


def _argmax(distribution: Mapping[str, float]) -> str:
    return max(distribution.items(), key=lambda kv: kv[1])[0]


def _confidence(distribution: Mapping[str, float]) -> float:
    """Top-1 minus top-2: simple, monotone, and comparable across backends."""
    values = sorted(distribution.values(), reverse=True)
    if len(values) < 2:
        return 1.0
    return round(max(0.0, min(1.0, values[0] - values[1])), 6)


def _marker_score(text: str, markers: Sequence[tuple[str, float]]) -> float:
    return sum(weight for pattern, weight in markers if re.search(pattern, text, re.IGNORECASE))


class MockBackend:
    """Deterministic offline backend. Implements :class:`~.base.DecisionBackend`."""

    name = "mock"
    model_version = MODEL_VERSION

    def __init__(self, *, temperature: float = TEMPERATURE) -> None:
        self.temperature = temperature

    def noul(self, state: Any, instructions: str) -> float:
        """The outcome-verification seam (optional protocol method). Mock rule:
        a non-empty response excerpt completes the task. Deterministic, so CI
        exercises the verifier without keys."""
        if isinstance(state, dict):
            excerpt = str(state.get("response_excerpt", ""))
        else:
            excerpt = str(state)
        return 0.9 if excerpt.strip() else 0.1

    async def decide(self, request: DecisionRequest) -> BackendResult:
        answers = self.decide_sync(request)
        return BackendResult(answers=answers, model_version=self.model_version, questions_sent={}, latency_ms=0.0)

    def decide_sync(self, request: DecisionRequest) -> DecisionAnswers:
        """The synchronous core. Public so tests and the CLI can call it directly."""
        text = request.redacted_excerpt or ""
        features = request.features
        lowered = text.lower()

        # A salt of the text keeps answers stable per prompt while letting two
        # otherwise-identical feature vectors differ slightly, which is what
        # produces non-degenerate distributions for the confidence-floor tests.
        jitter = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        nudge = (jitter - 0.5) * 0.6

        complexity = self._complexity(text, features, nudge, lowered)
        sensitivity = self._sensitivity(text, request, nudge)
        pii = self._pii(request, sensitivity)
        domain = self._domain(text, features)

        return DecisionAnswers(
            complexity=complexity,
            sensitivity=sensitivity,
            pii=pii,
            domain=domain,
        )

    def _complexity(self, text: str, features: Any, nudge: float, lowered: str) -> ChoiceAnswer:
        score = _marker_score(lowered, _COMPLEXITY_MARKERS)
        # Shape features push complexity up: long prompts, code, stack traces,
        # multi-turn context and questions all correlate with harder tasks.
        score += min(2.0, math.log1p(features.char_len) / 4.0)
        score += 0.8 * min(features.code_blocks, 3)
        score += 0.6 if features.has_stack_trace else 0.0
        score += 0.4 if features.has_diff else 0.0
        score += 0.3 * min(features.question_marks, 4)
        score += 0.2 * min(features.n_prior_turns, 4)
        score -= 1.2 if features.char_len < 40 else 0.0
        score += nudge

        band = {
            "trivial": -score * 1.1,
            "standard": 1.0 - abs(score - 1.0) * 0.75,
            "hard": score - 1.4,
            "frontier": score - 3.1,
        }
        distribution = _softmax({k: band[k] for k in COMPLEXITY_LEVELS}, self.temperature)
        return ChoiceAnswer(
            choice=_argmax(distribution),
            probabilities=distribution,
            confidence=_confidence(distribution),
            confidence_reported=False,
        )

    def _sensitivity(self, text: str, request: DecisionRequest, nudge: float) -> ChoiceAnswer:
        lowered = text.lower()
        score = _marker_score(lowered, _SENSITIVITY_MARKERS)
        # Talking about a regulated topic pushes the other way. Weighted to roughly
        # cancel a single topic marker, which is what makes "explain how HIPAA
        # works" resolve to public instead of regulated.
        score += _marker_score(lowered, _ABOUT_NOT_DATA)
        # The gate's advisory topic hints are weak evidence on purpose: they come
        # from curated regulated-domain vocabulary and mean "look harder here",
        # never "this is regulated". The hard gate already handled the identifiers
        # it could verify; what is left genuinely needs judgement.
        score += 0.3 * len(request.advisory_topics)
        score += nudge * 0.5

        band = {
            "public": -score * 1.0 + 0.9,
            "internal": 1.1 - abs(score - 1.1) * 0.7,
            "confidential": score - 1.2,
            "regulated": score - 2.6,
        }
        distribution = _softmax({k: band[k] for k in SENSITIVITY_LEVELS}, self.temperature)
        return ChoiceAnswer(
            choice=_argmax(distribution),
            probabilities=distribution,
            confidence=_confidence(distribution),
            confidence_reported=False,
        )

    def _pii(self, request: DecisionRequest, sensitivity: ChoiceAnswer) -> NoulAnswer:
        """PII likelihood, derived from the sensitivity distribution.

        Correlated on purpose: a mock that answered sensitivity=regulated and
        pii=0.01 would be internally inconsistent and would hide real bugs in the
        policy engine's merging logic.
        """
        mass = sensitivity.probability_of("confidential") + sensitivity.probability_of("regulated")
        # Advisory topics do NOT raise the PII estimate. A topic keyword is not a
        # person, and letting it move this number is how "what does PCI-DSS
        # require?" ends up scored as containing cardholder data.
        return NoulAnswer(value=round(max(0.01, min(0.99, mass * 0.9)), 6))

    def _domain(self, text: str, features: Any) -> ChoiceAnswer:
        scores = dict.fromkeys(DOMAINS, 0.35)
        for pattern, domain, weight in _DOMAIN_MARKERS:
            if re.search(pattern, text, re.IGNORECASE):
                scores[domain] += weight
        if features.code_blocks or features.inline_code_spans:
            scores["code"] += 1.2
        if features.has_json or features.urls:
            scores["data-extraction"] += 0.8
        if features.has_stack_trace:
            scores["code"] += 0.8
        if features.char_len < 60 and features.question_marks == 0:
            scores["chat"] += 0.9
        distribution = _softmax(scores, self.temperature)
        return ChoiceAnswer(
            choice=_argmax(distribution),
            probabilities=distribution,
            confidence=_confidence(distribution),
            confidence_reported=False,
        )

    async def aclose(self) -> None:
        return None


#: Module-level instance, mirroring how LiteLLM resolves dotted-path plugins.
backend = MockBackend()

__all__ = ["MODEL_VERSION", "MockBackend", "backend"]
