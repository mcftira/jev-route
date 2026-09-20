"""Shared fixtures for the jev-route test suite.

The suite has one non-negotiable property: **it runs offline, with no API key.**
That is not a stylistic preference. jev-route's whole pitch is "run it, log it,
distill it, own it", and the first step has to work on a laptop with no account,
no egress, and no money spent. A test suite that needs the cloud to pass cannot
prove that claim -- it can only assume it.

So this file does three things beyond the usual fixture plumbing:

* It **blocks outbound sockets** for every test (loopback stays open, so a test
  may still serve a local HTTP stub). A test that reaches api.typesafe.ai raises
  ``NetworkBlockedError`` with the offending host in the message.
* It **removes ``TYPESAFE_API_KEY``** from the environment, so a developer's real
  key can never make a test pass locally and fail in CI.
* It provides **programmable doubles** -- ``FakeBackend`` and ``StubTransport`` --
  so backend behaviour (distributions, latency, degradation, HTTP status) is set
  by the test rather than discovered by it.

Both guards step aside for tests marked ``@pytest.mark.network``, which are
skipped unless ``RUN_NETWORK_TESTS=1``.
"""

from __future__ import annotations

import json
import os
import socket
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from jev_route.backends.base import BackendResult, DecisionRequest
from jev_route.cache import InMemoryTTLCache, NullCache
from jev_route.gate import HardGate
from jev_route.policy import Policy
from jev_route.router import Router
from jev_route.schema import (
    COMPLEXITY_LEVELS,
    DOMAINS,
    SENSITIVITY_LEVELS,
    ChoiceAnswer,
    DecisionAnswers,
    DecisionRecord,
    NoulAnswer,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EVALS_PATH = REPO_ROOT / "evals" / "data" / "labeled_prompts.jsonl"
DEFAULT_POLICY_PATH = REPO_ROOT / "policies" / "default.yaml"

RUN_NETWORK = os.environ.get("RUN_NETWORK_TESTS", "") == "1"


# --------------------------------------------------------------------------- #
# Offline enforcement
# --------------------------------------------------------------------------- #
class NetworkBlockedError(RuntimeError):
    """Raised when a test tries to open a non-loopback connection.

    A distinct type rather than a bare ``AssertionError`` so a failure reads as
    "this test needs the network" and not as "an assertion about routing failed".
    """


#: Loopback is deliberately allowed: integrations tests stand up a local server
#: and talk to it, and that is still "offline". Everything else is refused.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0", ""})


def _assert_loopback(address: Any) -> None:
    """Raise unless ``address`` is a loopback TCP endpoint or a unix socket."""
    if isinstance(address, (bytes, bytearray, os.PathLike, str)):
        return  # AF_UNIX / AF_VSOCK style address: local by definition
    if not isinstance(address, tuple) or not address:
        return
    host = str(address[0])
    if host in _LOOPBACK_HOSTS or host.startswith("127.") or host.startswith("::1"):
        return
    raise NetworkBlockedError(
        f"test attempted a network connection to {host!r}. The jev-route suite is "
        f"offline by contract: use MockBackend, FakeBackend, or StubTransport. "
        f"If this test genuinely needs the cloud, mark it @pytest.mark.network."
    )


@pytest.fixture(autouse=True)
def _offline(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Block outbound sockets and drop the API key, unless the test opts in."""
    if RUN_NETWORK or request.node.get_closest_marker("network") is not None:
        yield
        return

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create_connection = socket.create_connection

    def connect(self: socket.socket, address: Any) -> Any:
        _assert_loopback(address)
        return real_connect(self, address)

    def connect_ex(self: socket.socket, address: Any) -> int:
        _assert_loopback(address)
        return real_connect_ex(self, address)

    def create_connection(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
        _assert_loopback(address)
        return real_create_connection(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "create_connection", create_connection)

    # A real key on the developer's machine must not be what makes a test pass.
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    yield


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``@pytest.mark.network`` tests unless RUN_NETWORK_TESTS=1."""
    if RUN_NETWORK:
        return
    skip = pytest.mark.skip(reason="needs the network; set RUN_NETWORK_TESTS=1 to run")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #
def make_policy_doc(
    *,
    backend: Mapping[str, Any] | None = None,
    gate: Mapping[str, Any] | None = None,
    tiers: Mapping[str, Any] | None = None,
    rules: Sequence[Mapping[str, Any]] | None = None,
    logging: Mapping[str, Any] | None = None,
    cache: Mapping[str, Any] | None = None,
    on_backend_down: Mapping[str, Any] | None = None,
    on_uncertain: Mapping[str, Any] | None = None,
    tier_order: Sequence[str] | None = None,
    pii_threshold: float | None = None,
    log_path: str | Path | None = None,
    excerpt_mode: str = "hash",
) -> dict[str, Any]:
    """Build a minimal but complete policy document.

    Kept as a function rather than a module constant so a test can vary one knob
    and still see the whole document. ``local``/``cheap``/``strong`` are all
    defined because ``FailurePolicy`` validates both of its tiers at load time --
    a policy that omits ``strong`` cannot load even when it never fails open.
    """
    doc: dict[str, Any] = {
        "version": 1,
        "backend": dict(backend or {"name": "mock"}),
        "gate": {
            "on_force_local": "skip_backend",
            "disabled_detectors": [],
            "placeholder_domains_as_pii": False,
            **(gate or {}),
        },
        "tiers": dict(tiers or {"local": ["local-model"], "cheap": ["cheap-model"], "strong": ["strong-model"]}),
        "tier_order": list(tier_order or ["cheap", "strong", "local"]),
        "pii_threshold": 0.5 if pii_threshold is None else pii_threshold,
        "rules": list(
            rules
            or [
                {
                    "id": "gate.force-local",
                    "if": "gate_force_local",
                    "then": {"tier": "local"},
                    "reason": "local hard gate matched a structured identifier or credential",
                },
                {
                    "id": "data.sensitive",
                    "if": 'sensitivity in ["confidential", "regulated"] or pii_present',
                    "then": {"tier": "local"},
                    "reason": "sensitive or personal data must not leave",
                },
                {
                    "id": "complexity.frontier",
                    "if": 'complexity == "frontier"',
                    "then": {"tier": "strong"},
                    "reason": "task needs frontier-class reasoning",
                },
                {
                    "id": "complexity.hard",
                    "if": 'complexity == "hard"',
                    "then": {"tier": "strong"},
                    "reason": "task needs a strong model",
                },
                {"id": "default", "then": {"tier": "cheap"}, "reason": "routine task, no sensitivity signal"},
            ]
        ),
        "on_uncertain": {
            "sensitivity_confidence_below": 0.8,
            "sensitivity_bump_levels": 1,
            "complexity_confidence_below": 0.7,
            "complexity_bump_levels": 1,
            "pii_uncertain_threshold": 0.35,
            "pii_uncertain_counts_as_present": True,
            **(on_uncertain or {}),
        },
        "on_backend_down": {
            "mode": "fail_closed",
            "fail_closed_tier": "local",
            "fail_open_tier": "strong",
            **(on_backend_down or {}),
        },
        "cache": dict(cache or {"enabled": False}),
        "logging": {
            "enabled": True,
            "path": str(log_path or "./unused-decision-log.jsonl"),
            "excerpt_mode": excerpt_mode,
            "hash_salt": "",
            **(logging or {}),
        },
    }
    return doc


@pytest.fixture
def policy_doc(tmp_path: Path) -> dict[str, Any]:
    """A valid policy document whose log path lives in ``tmp_path``."""
    return make_policy_doc(log_path=tmp_path / "decisions.jsonl")


@pytest.fixture
def tmp_policy_path(tmp_path: Path, policy_doc: dict[str, Any]) -> Path:
    """The same document, written to a temp YAML file (exercises the file path)."""
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(policy_doc, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def policy(policy_doc: dict[str, Any]) -> Policy:
    return Policy.from_dict(policy_doc)


@pytest.fixture
def default_policy() -> Policy:
    """The shipped ``policies/default.yaml``. Read-only; tests must not mutate it."""
    if not DEFAULT_POLICY_PATH.exists():  # pragma: no cover - repo layout guard
        pytest.skip("policies/default.yaml not present")
    return Policy.from_file(DEFAULT_POLICY_PATH)


# --------------------------------------------------------------------------- #
# Backend doubles
# --------------------------------------------------------------------------- #
def peaked(level: str, ladder: Sequence[str], confidence: float) -> ChoiceAnswer:
    """A distribution with ``level`` at ``confidence`` and the rest sharing the remainder.

    Tests need to set a *confidence*, and confidence in this codebase is a real
    number derived from a real distribution. Building the distribution instead of
    faking the field keeps the doubles honest: anything the router computes from
    ``probabilities`` agrees with anything it reads from ``confidence``.
    """
    if level not in ladder:
        raise ValueError(f"{level!r} is not in {tuple(ladder)}")
    others = [x for x in ladder if x != level]
    share = round((1.0 - confidence) / len(others), 6) if others else 0.0
    probabilities = dict.fromkeys(others, share)
    probabilities[level] = round(1.0 - share * len(others), 6)
    return ChoiceAnswer(
        choice=level,
        probabilities={k: probabilities[k] for k in ladder},
        confidence=confidence,
        confidence_reported=True,
    )


@dataclass
class FakeBackend:
    """A programmable :class:`~jev_route.backends.base.DecisionBackend`.

    Every knob the router reacts to is a field: the four answers, the reported
    confidences, latency, and the two failure shapes (``degraded`` for an honest
    outage, ``exc`` for a backend that breaks its contract by raising). Requests
    are recorded verbatim so a test can assert on what the backend was *shown*,
    which is how the redaction and gate-hint promises get checked.
    """

    name: str = "fake"
    model_version: str = "fake-1.0.0"
    complexity: str = "standard"
    complexity_confidence: float = 0.95
    sensitivity: str = "internal"
    sensitivity_confidence: float = 0.95
    domain: str = "chat"
    domain_confidence: float = 0.95
    pii: float = 0.05
    degraded: bool = False
    degrade_reason: str = "fake backend unavailable"
    latency_ms: float = 0.0
    exc: BaseException | None = None
    #: Answer sequence, when a test needs different answers per call.
    script: list[DecisionAnswers] = field(default_factory=list)
    requests: list[DecisionRequest] = field(default_factory=list)

    @property
    def calls(self) -> int:
        return len(self.requests)

    def answers_for(self, index: int) -> DecisionAnswers:
        if self.script:
            return self.script[min(index, len(self.script) - 1)]
        return DecisionAnswers(
            complexity=peaked(self.complexity, COMPLEXITY_LEVELS, self.complexity_confidence),
            sensitivity=peaked(self.sensitivity, SENSITIVITY_LEVELS, self.sensitivity_confidence),
            pii=NoulAnswer(value=self.pii),
            domain=peaked(self.domain, DOMAINS, self.domain_confidence),
        )

    async def decide(self, request: DecisionRequest) -> BackendResult:
        self.requests.append(request)
        if self.latency_ms:
            # Real sleeps, not a mocked clock: the router measures wall time and a
            # test asserting latency ordering should see a real ordering.
            time.sleep(self.latency_ms / 1000.0)
        if self.exc is not None:
            raise self.exc
        if self.degraded:
            return BackendResult(
                answers=DecisionAnswers.unknown(),
                model_version=self.model_version,
                questions_sent={"complexity": {"type": "choice"}},
                latency_ms=self.latency_ms,
                degraded=True,
                degrade_reason=self.degrade_reason,
            )
        return BackendResult(
            answers=self.answers_for(self.calls - 1),
            model_version=self.model_version,
            questions_sent={"complexity": {"type": "choice"}, "sensitivity": {"type": "choice"}},
            latency_ms=self.latency_ms,
        )

    async def aclose(self) -> None:
        return None


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


class StubResponse:
    """Minimal ``httpx.Response`` shape: status, ``.text``, and ``.json()``."""

    def __init__(self, status_code: int = 200, body: Mapping[str, Any] | None = None, text: str | None = None) -> None:
        self.status_code = status_code
        self._body = dict(body or {})
        self.text = text if text is not None else json.dumps(self._body)

    def json(self) -> dict[str, Any]:
        return self._body


def jev_response(
    *,
    model: str = "jev-test-1",
    complexity: Mapping[str, float] | None = None,
    complexity_choice: str | None = None,
    sensitivity: Mapping[str, float] | None = None,
    sensitivity_choice: str | None = None,
    pii: float | None = None,
    domain: Mapping[str, float] | None = None,
    domain_choice: str | None = None,
    raw: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A well-formed System One response body, with only the parts you care about set.

    Anything left ``None`` is simply absent from the payload, which is the shape
    the parser has to tolerate in production. Pass ``raw`` to override wholesale
    when the test *is* about a malformed body.
    """
    if raw is not None:
        return dict(raw)
    answers: dict[str, Any] = {}
    if complexity is not None or complexity_choice is not None:
        entry: dict[str, Any] = {}
        if complexity is not None:
            entry["probabilities"] = dict(complexity)
        if complexity_choice is not None:
            entry["choice"] = complexity_choice
        answers["complexity"] = entry
    if sensitivity is not None or sensitivity_choice is not None:
        entry = {}
        if sensitivity is not None:
            entry["probabilities"] = dict(sensitivity)
        if sensitivity_choice is not None:
            entry["choice"] = sensitivity_choice
        answers["sensitivity"] = entry
    if pii is not None:
        answers["pii_present"] = {"noul": pii}
    if domain is not None or domain_choice is not None:
        entry = {}
        if domain is not None:
            entry["probabilities"] = dict(domain)
        if domain_choice is not None:
            entry["choice"] = domain_choice
        answers["domain"] = entry
    return {"model": model, "answers": answers}


class StubTransport:
    """An ``httpx.AsyncClient`` stand-in with an ``async post()``.

    The script is a list of outcomes consumed in order; the last entry repeats, so
    ``StubTransport([500])`` means "always 500" and ``StubTransport([429, ok])``
    means "rate limited once, then fine". An outcome is a :class:`StubResponse`,
    an ``int`` status shorthand, an exception instance to raise, or a callable.
    Every call is recorded so a test can assert on the payload that *would* have
    left the process.
    """

    def __init__(self, script: Sequence[Any] = (), *, sleep: float = 0.0) -> None:
        self.script: list[Any] = list(script) or [StubResponse(200, jev_response())]
        self.sleep = sleep
        self.calls: list[dict[str, Any]] = []
        self.closed = 0

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def payloads(self) -> list[dict[str, Any]]:
        return [c["json"] for c in self.calls]

    @property
    def statuses(self) -> list[int]:
        return [c["status"] for c in self.calls]

    def outcome_at(self, index: int) -> Any:
        """The scripted outcome for call ``index``; the last entry repeats forever."""
        return self.script[min(index, len(self.script) - 1)]

    async def post(self, url: str, json: Any = None, headers: Any = None) -> StubResponse:
        import asyncio as _asyncio

        index = len(self.calls)
        self.calls.append({"url": url, "json": json, "headers": dict(headers or {}), "status": None})
        if self.sleep:
            await _asyncio.sleep(self.sleep)
        outcome = self.outcome_at(index)
        if callable(outcome):
            outcome = outcome(self)
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, int):
            return StubResponse(outcome, {}, text=f"HTTP {outcome}")
        self.calls[-1]["status"] = getattr(outcome, "status_code", None)
        return outcome

    async def aclose(self) -> None:
        self.closed += 1


@pytest.fixture
def stub_transport() -> StubTransport:
    return StubTransport()


class FakeClock:
    """Injectable monotonic clock for :class:`~jev_route.backends.base.CircuitBreaker`."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


# --------------------------------------------------------------------------- #
# Router plumbing
# --------------------------------------------------------------------------- #
class RecordingSink:
    """A sink that keeps every record, so tests can read the training data back."""

    def __init__(self) -> None:
        self.records: list[DecisionRecord] = []
        self.closed = 0

    def write(self, record: DecisionRecord) -> None:
        self.records.append(record)

    def close(self) -> None:
        self.closed += 1

    def stats(self) -> dict[str, Any]:
        return {"backend": "recording", "written": len(self.records)}

    # -- conveniences ---------------------------------------------------- #
    @property
    def count(self) -> int:
        return len(self.records)

    @property
    def last(self) -> DecisionRecord:
        assert self.records, "the sink received no records"
        return self.records[-1]

    def json_lines(self) -> list[str]:
        return [r.to_json() for r in self.records]


@pytest.fixture
def sink() -> RecordingSink:
    return RecordingSink()


@pytest.fixture
def router_factory(
    policy: Policy,
    sink: RecordingSink,
) -> Callable[..., Router]:
    """Build routers that share one recording sink and one non-caching default.

    Caching is off unless a test asks for it, because "the backend was called
    twice" is a different assertion from "the cache missed twice" and mixing them
    makes a failure ambiguous.
    """
    created: list[Router] = []

    def make(
        backend: Any = None,
        *,
        pol: Policy | None = None,
        gate: HardGate | None = None,
        cache: Any = None,
        own_sink: Any = None,
        shadow_backend: Any = None,
        excerpt_mode: str | None = None,
        hash_salt: str | None = None,
        **kwargs: Any,
    ) -> Router:
        from jev_route.backends.mock import MockBackend

        router = Router(
            pol if pol is not None else policy,
            backend if backend is not None else MockBackend(),
            gate=gate,
            cache=cache if cache is not None else NullCache(),
            sink=own_sink if own_sink is not None else sink,
            shadow_backend=shadow_backend,
            excerpt_mode=excerpt_mode,
            hash_salt=hash_salt,
            **kwargs,
        )
        created.append(router)
        return router

    make.created = created  # type: ignore[attr-defined]
    return make


@pytest.fixture
def memory_cache() -> InMemoryTTLCache:
    return InMemoryTTLCache(ttl_seconds=60.0, max_entries=64)


# --------------------------------------------------------------------------- #
# Sample prompts from the eval dataset
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SamplePrompt:
    """One row of ``evals/data/labeled_prompts.jsonl``."""

    id: str
    text: str
    labels: Mapping[str, Any]
    expected_tier: str
    difficulty: str
    notes: str


def load_sample_prompts() -> list[SamplePrompt]:
    if not EVALS_PATH.exists():
        return []
    out: list[SamplePrompt] = []
    for line in EVALS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out.append(
            SamplePrompt(
                id=str(row["id"]),
                text=str(row["text"]),
                labels=dict(row.get("labels") or {}),
                expected_tier=str(row.get("expected_tier", "")),
                difficulty=str(row.get("difficulty", "")),
                notes=str(row.get("notes", "")),
            )
        )
    return out


@pytest.fixture(scope="session")
def sample_prompts() -> list[SamplePrompt]:
    """The labeled eval prompts, or a skip when the dataset is not checked out.

    A missing dataset is not a test failure: the evals file is maintained
    separately and a fresh clone of the core package should still test green.
    """
    prompts = load_sample_prompts()
    if not prompts:
        pytest.skip(f"eval dataset not present at {EVALS_PATH}")
    return prompts


@pytest.fixture(scope="session")
def sample_prompts_by_tier(sample_prompts: list[SamplePrompt]) -> dict[str, list[SamplePrompt]]:
    grouped: dict[str, list[SamplePrompt]] = {}
    for prompt in sample_prompts:
        grouped.setdefault(prompt.expected_tier, []).append(prompt)
    return grouped


# --------------------------------------------------------------------------- #
# Text samples. Synthetic and RFC 2606 / invented domains only -- never real PII.
# --------------------------------------------------------------------------- #
CARD_LUHN_VALID = "4111 1111 1111 1111"
CARD_LUHN_VALID_MC = "5555 5555 5555 4444"
CARD_LUHN_INVALID_SPACED = "4111 1111 1111 1112"
CARD_LUHN_INVALID_FLAT = "4111111111111112"
IBAN_VALID = "GB33 BUKB 2020 1555 5555 55"
IBAN_INVALID = "GB33 BUKB 2020 1555 5555 56"
NHS_VALID = "943 476 5919"
NHS_INVALID = "123 456 7890"
NINO_VALID = "AB123456C"
SSN_RESERVED = ("000-12-3456", "666-45-6789", "900-45-6789")
#: Realistic-looking but invented. Never use a real person's address in a test.
EMAIL_INVENTED = "dana.kovacs@northside-health.org"
EMAIL_PLACEHOLDER = "user@example.com"
#: Provider-shaped but fake keys. The dotted one is a regression fixture: an
#: earlier ``sk-`` pattern stopped at the first dot and missed Alibaba-style keys.
API_KEY_DOTTED = "sk-demo-D.OTTED.abcdefghijklmnopqrstuvwx"
API_KEY_ANTHROPIC = "sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWX"
API_KEY_AWS = "AKIAIOSFODNN7EXAMPLE"
API_KEY_GITHUB = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"
API_KEY_SLACK = "xoxb-123456789012-abcdefghijklmnop"
API_KEY_GOOGLE = "AIzaSyA1234567890abcdefghijklmnopqrstuv"
API_KEY_HF = "hf_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh"
PRIVATE_KEY_HEADER = "-----BEGIN RSA PRIVATE KEY-----"
INLINE_PASSWORD = "config password=Sup3rSecret!value"
BASIC_AUTH_URL = "postgres://router:hunter2@db.internal.northside.example"
TOPIC_HIPAA = "Explain how HIPAA actually works and who it applies to."
TOPIC_CVV = "What is a credit card CVV?"
TOPIC_PCI = "What does PCI-DSS require of merchants who store card data?"
CLEAN_PROSE = "Summarize the three main arguments of this essay about urban cycling infrastructure."


@pytest.fixture
def samples() -> dict[str, Any]:
    """Named text samples, so a test reads as intent rather than as a regex."""
    return {
        "card_valid": CARD_LUHN_VALID,
        "card_valid_mc": CARD_LUHN_VALID_MC,
        "card_invalid_spaced": CARD_LUHN_INVALID_SPACED,
        "card_invalid_flat": CARD_LUHN_INVALID_FLAT,
        "iban_valid": IBAN_VALID,
        "iban_invalid": IBAN_INVALID,
        "nhs_valid": NHS_VALID,
        "nhs_invalid": NHS_INVALID,
        "nino_valid": NINO_VALID,
        "ssn_reserved": SSN_RESERVED,
        "email_invented": EMAIL_INVENTED,
        "email_placeholder": EMAIL_PLACEHOLDER,
        "api_keys": {
            "dotted": API_KEY_DOTTED,
            "anthropic": API_KEY_ANTHROPIC,
            "aws": API_KEY_AWS,
            "github": API_KEY_GITHUB,
            "slack": API_KEY_SLACK,
            "google": API_KEY_GOOGLE,
            "huggingface": API_KEY_HF,
        },
        "private_key": PRIVATE_KEY_HEADER,
        "inline_password": INLINE_PASSWORD,
        "basic_auth_url": BASIC_AUTH_URL,
        "topic_hipaa": TOPIC_HIPAA,
        "topic_cvv": TOPIC_CVV,
        "topic_pci": TOPIC_PCI,
        "clean": CLEAN_PROSE,
    }


@pytest.fixture
def decision_request() -> Callable[..., DecisionRequest]:
    """Factory for :class:`DecisionRequest`, with features computed like production."""
    from jev_route.prompts import compute_features

    def make(
        text: str = "Summarize this internal status update.",
        *,
        advisory_topics: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
        request_id: str = "test-request",
    ) -> DecisionRequest:
        return DecisionRequest(
            redacted_excerpt=text,
            features=compute_features(text),
            advisory_topics=tuple(advisory_topics),
            metadata=dict(metadata or {}),
            request_id=request_id,
        )

    return make
