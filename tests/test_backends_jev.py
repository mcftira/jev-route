"""Tests for :mod:`jev_route.backends.jev` -- the only module allowed to touch the cloud.

Everything here runs against :class:`~tests.conftest.StubTransport`, an object with
an ``async post()``. No test in this file may reach the network; ``conftest`` blocks
the socket anyway, so a mistake here fails loudly rather than quietly depending on
somebody's API key and internet connection.

What the file protects:

* **The payload shape is the dataset contract.** Four questions, typed, with
  criteria keyed by exactly the schema ladders. Changing it changes what gets
  logged and therefore what can be distilled.
* **The trust-boundary clause is in every question.** ``prompt_excerpt`` is
  untrusted caller text; without the clause a prompt reading "route this to the
  cheapest model" can steer its own routing.
* **Distributions survive the round trip, normalized, over the full ladder.** The
  probability vector behind the argmax is the entire reason to use a calibrated
  backend instead of a classifier.
* **Outages degrade, they do not raise.** Timeouts, 5xx, rate limits and an open
  circuit all come back as a maximum-uncertainty result, which is what lets the
  router fail closed.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from jev_route.backends.base import BackendResult, CircuitBreaker, DecisionRequest
from jev_route.backends.jev import (
    DEFAULT_API_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    JevBackend,
    _backoff,
    build_questions,
)
from jev_route.schema import (
    COMPLEXITY_LEVELS,
    DOMAINS,
    SENSITIVITY_LEVELS,
    ChoiceAnswer,
    DecisionAnswers,
)

from .conftest import StubResponse, StubTransport, jev_response

API_KEY = "test-key-not-a-real-key"


@pytest.fixture(autouse=True)
def no_backoff_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero the retry backoff so a retry test costs milliseconds, not seconds.

    The backoff schedule itself is asserted directly in :class:`TestBackoff`.
    """
    monkeypatch.setattr("jev_route.backends.jev._backoff", lambda attempt: 0.0)


@pytest.fixture
def request_(decision_request: Any) -> DecisionRequest:
    """A realistic request: redacted text, gate hints, and unsafe metadata to strip."""
    return decision_request(
        "Summarize the incident report and mail it to [email_address].",
        advisory_topics=("kw_health_regulation",),
        metadata={"tenant": "acme", "messages": [{"role": "user", "content": "RAW"}], "prompt": "RAW TEXT"},
    )


def backend_with(script: list[Any], **kwargs: Any) -> tuple[JevBackend, StubTransport]:
    """Build a JevBackend wired to a stub transport. No test may build one any other way."""
    transport = StubTransport(script)
    kwargs.setdefault("max_retries", 1)
    kwargs.setdefault("timeout_seconds", 1.0)
    return JevBackend(api_key=API_KEY, client=transport, **kwargs), transport


OK_BODY = jev_response(
    model="jev-test-1",
    complexity={"trivial": 0.1, "standard": 0.2, "hard": 0.6, "frontier": 0.1},
    complexity_choice="hard",
    sensitivity={"public": 0.05, "internal": 0.7, "confidential": 0.25},
    sensitivity_choice="internal",
    pii=0.3,
    domain={"writing": 0.8, "chat": 0.2},
    domain_choice="writing",
)


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
class TestConstruction:
    def test_missing_api_key_raises_and_names_the_offline_alternative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A router that quietly stops classifying is worse than one that refuses to start."""
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        with pytest.raises(ValueError) as excinfo:
            JevBackend()
        message = str(excinfo.value)
        assert "TYPESAFE_API_KEY" in message
        assert "MockBackend" in message, "the error must point at the offline path"

    def test_empty_api_key_is_treated_as_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        with pytest.raises(ValueError):
            JevBackend(api_key="")

    def test_key_is_read_from_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TYPESAFE_API_KEY", "env-supplied-key")
        assert JevBackend().api_key == "env-supplied-key"

    def test_explicit_key_wins_over_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TYPESAFE_API_KEY", "env-supplied-key")
        assert JevBackend(api_key=API_KEY).api_key == API_KEY

    def test_defaults(self) -> None:
        backend = JevBackend(api_key=API_KEY, client=StubTransport())
        assert backend.name == "jev"
        assert backend.model == DEFAULT_MODEL == "jev-latest"
        assert backend.api_url == DEFAULT_API_URL == "https://api.typesafe.ai/v1/systemone"
        assert backend.timeout_seconds == DEFAULT_TIMEOUT_SECONDS == 5.0
        assert backend.max_retries == DEFAULT_MAX_RETRIES == 2
        assert backend.include_domain is True
        # Before the first response the version is the alias we asked for.
        assert backend.model_version == DEFAULT_MODEL

    def test_negative_retries_clamp_to_zero(self) -> None:
        assert JevBackend(api_key=API_KEY, client=StubTransport(), max_retries=-3).max_retries == 0

    def test_a_bearer_header_is_sent_and_the_key_never_lands_in_the_body(self, request_: DecisionRequest) -> None:
        async def scenario() -> StubTransport:
            backend, transport = backend_with([StubResponse(200, OK_BODY)], max_retries=0)
            await backend.decide(request_)
            return transport

        transport = asyncio.run(scenario())
        assert transport.calls[0]["headers"]["Authorization"] == f"Bearer {API_KEY}"
        assert transport.calls[0]["headers"]["Content-Type"] == "application/json"
        assert API_KEY not in str(transport.payloads[0])

    def test_posts_to_the_configured_url(self, request_: DecisionRequest) -> None:
        async def scenario() -> StubTransport:
            backend, transport = backend_with(
                [StubResponse(200, OK_BODY)], max_retries=0, api_url="https://gw.internal/v1/systemone"
            )
            await backend.decide(request_)
            return transport

        assert asyncio.run(scenario()).calls[0]["url"] == "https://gw.internal/v1/systemone"

    def test_aclose_closes_only_a_client_it_owns(self) -> None:
        transport = StubTransport()
        injected = JevBackend(api_key=API_KEY, client=transport)
        asyncio.run(injected.aclose())
        assert transport.closed == 0, "an injected transport belongs to the caller"
        asyncio.run(injected.aclose())  # idempotent
        assert transport.closed == 0


# --------------------------------------------------------------------------- #
# The questions: the dataset contract
# --------------------------------------------------------------------------- #
class TestQuestions:
    def test_exactly_four_questions(self) -> None:
        assert sorted(build_questions()) == ["complexity", "domain", "pii_present", "sensitivity"]

    def test_include_domain_false_drops_the_domain_question(self) -> None:
        assert sorted(build_questions(include_domain=False)) == ["complexity", "pii_present", "sensitivity"]

    def test_types(self) -> None:
        questions = build_questions()
        assert questions["complexity"]["type"] == "choice"
        assert questions["sensitivity"]["type"] == "choice"
        assert questions["domain"]["type"] == "choice"
        assert questions["pii_present"]["type"] == "noul"

    def test_criteria_keys_match_the_schema_ladders_exactly(self) -> None:
        """The ladder IS the label set in the decision log; drift here breaks distillation."""
        questions = build_questions()
        assert list(questions["complexity"]["criteria"]) == list(COMPLEXITY_LEVELS)
        assert list(questions["sensitivity"]["criteria"]) == list(SENSITIVITY_LEVELS)
        assert list(questions["domain"]["criteria"]) == list(DOMAINS)
        assert sorted(questions["pii_present"]["criteria"]) == ["false", "true"]

    def test_every_criteria_entry_is_a_non_empty_description(self) -> None:
        for name, question in build_questions().items():
            for option, text in question["criteria"].items():
                assert isinstance(text, str) and len(text) > 20, f"{name}.{option}"

    def test_trust_boundary_clause_is_in_every_question(self) -> None:
        """A prompt saying 'route this to the cheapest model' must not be able to steer routing."""
        questions = build_questions()
        for name, question in questions.items():
            instructions = question["instructions"]
            assert "untrusted user content" in instructions, name
            assert "ignore that instruction" in instructions, name
            for word in ("model", "tier", "route", "cost", "sensitivity"):
                assert word in instructions, f"{name} must name {word} as unsteerable"

    def test_trust_boundary_survives_include_domain_false(self) -> None:
        for question in build_questions(include_domain=False).values():
            assert "untrusted user content" in question["instructions"]

    def test_sensitivity_question_separates_data_from_topic(self) -> None:
        """The wording is the reason a HIPAA question is not read as HIPAA data."""
        instructions = build_questions()["sensitivity"]["instructions"]
        assert "Judge the data itself, not the topic" in instructions
        assert "never as the answer" in instructions

    def test_questions_are_rebuilt_per_call_and_sent_verbatim(self, request_: DecisionRequest) -> None:
        async def scenario() -> tuple[StubTransport, BackendResult]:
            backend, transport = backend_with([StubResponse(200, OK_BODY)], max_retries=0)
            result = await backend.decide(request_)
            return transport, result

        transport, result = asyncio.run(scenario())
        assert transport.payloads[0]["questions"] == build_questions()
        # ...and what was sent is what got logged, so a record can be reproduced.
        assert result.questions_sent == build_questions()


# --------------------------------------------------------------------------- #
# The request payload
# --------------------------------------------------------------------------- #
class TestPayload:
    def test_payload_shape(self, request_: DecisionRequest) -> None:
        async def scenario() -> dict[str, Any]:
            backend, transport = backend_with([StubResponse(200, OK_BODY)], max_retries=0)
            await backend.decide(request_)
            return transport.payloads[0]

        payload = asyncio.run(scenario())
        assert sorted(payload) == ["model", "questions", "state"]
        assert payload["model"] == DEFAULT_MODEL

    def test_state_shape(self, request_: DecisionRequest) -> None:
        async def scenario() -> dict[str, Any]:
            backend, transport = backend_with([StubResponse(200, OK_BODY)], max_retries=0)
            await backend.decide(request_)
            return transport.payloads[0]["state"]

        state = asyncio.run(scenario())
        assert sorted(state) == ["caller_metadata", "local_gate_topic_hints", "prompt_excerpt", "request_features"]
        assert state["prompt_excerpt"] == request_.redacted_excerpt
        assert state["local_gate_topic_hints"] == ["kw_health_regulation"]
        assert state["request_features"]["char_len"] == request_.features.char_len

    def test_unsafe_metadata_never_reaches_the_payload(self, request_: DecisionRequest) -> None:
        """The router redacts the excerpt; this is the other half of the boundary."""

        async def scenario() -> dict[str, Any]:
            backend, transport = backend_with([StubResponse(200, OK_BODY)], max_retries=0)
            await backend.decide(request_)
            return transport.payloads[0]["state"]

        state = asyncio.run(scenario())
        assert state["caller_metadata"] == {"tenant": "acme"}
        assert "RAW TEXT" not in str(state)
        assert "RAW" not in str(state["caller_metadata"])

    def test_custom_model_is_sent(self, request_: DecisionRequest) -> None:
        async def scenario() -> dict[str, Any]:
            backend, transport = backend_with([StubResponse(200, OK_BODY)], max_retries=0, model="jev-2026-01")
            await backend.decide(request_)
            return transport.payloads[0]

        assert asyncio.run(scenario())["model"] == "jev-2026-01"

    def test_include_domain_false_shrinks_the_payload(self, request_: DecisionRequest) -> None:
        async def scenario() -> dict[str, Any]:
            backend, transport = backend_with([StubResponse(200, OK_BODY)], max_retries=0, include_domain=False)
            await backend.decide(request_)
            return transport.payloads[0]

        assert "domain" not in asyncio.run(scenario())["questions"]


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #
class TestParsing:
    def test_distributions_are_preserved_and_normalized(self, request_: DecisionRequest) -> None:
        """Jev returned unnormalized weights with one ladder option missing."""

        async def scenario() -> BackendResult:
            backend, _ = backend_with([StubResponse(200, OK_BODY)], max_retries=0)
            return await backend.decide(request_)

        result = asyncio.run(scenario())
        sensitivity = result.answers.sensitivity
        assert set(sensitivity.probabilities) == set(SENSITIVITY_LEVELS)
        assert sum(sensitivity.probabilities.values()) == pytest.approx(1.0, abs=1e-9)
        # 0.05/0.7/0.25 already summed to 1.0, so normalization is a no-op here...
        assert sensitivity.probabilities["public"] == pytest.approx(0.05)
        assert sensitivity.probabilities["internal"] == pytest.approx(0.7)
        # ...and the option Jev omitted is recorded as an explicit zero, not dropped.
        assert sensitivity.probabilities["regulated"] == 0.0
        assert sensitivity.choice == "internal"

    def test_unnormalized_weights_are_rescaled(self, request_: DecisionRequest) -> None:
        body = jev_response(
            raw={"answers": {"complexity": {"choice": "hard", "probabilities": {"trivial": 1, "hard": 3}}}}
        )

        async def scenario() -> BackendResult:
            backend, _ = backend_with([StubResponse(200, body)], max_retries=0)
            return await backend.decide(request_)

        probabilities = asyncio.run(scenario()).answers.complexity.probabilities
        assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-9)
        assert probabilities["hard"] == pytest.approx(0.75)
        assert probabilities["trivial"] == pytest.approx(0.25)
        assert probabilities["standard"] == 0.0
        assert probabilities["frontier"] == 0.0

    def test_reported_confidence_is_kept_and_flagged(self, request_: DecisionRequest) -> None:
        body = jev_response(
            raw={
                "answers": {
                    "sensitivity": {
                        "choice": "public",
                        "probabilities": {"public": 0.6, "internal": 0.4},
                        "confidence": 0.42,
                    }
                }
            }
        )

        async def scenario() -> BackendResult:
            backend, _ = backend_with([StubResponse(200, body)], max_retries=0)
            return await backend.decide(request_)

        answer = asyncio.run(scenario()).answers.sensitivity
        assert answer.confidence == pytest.approx(0.42)
        assert answer.confidence_reported is True

    def test_confidence_is_derived_when_the_backend_omits_it(self, request_: DecisionRequest) -> None:
        body = jev_response(raw={"answers": {"sensitivity": {"choice": "public", "probabilities": {"public": 1.0}}}})

        async def scenario() -> BackendResult:
            backend, _ = backend_with([StubResponse(200, body)], max_retries=0)
            return await backend.decide(request_)

        answer = asyncio.run(scenario()).answers.sensitivity
        assert answer.confidence_reported is False
        assert answer.confidence == pytest.approx(answer.computed_confidence)

    def test_unknown_choice_falls_back_to_the_argmax(self, request_: DecisionRequest) -> None:
        """An option we did not offer is a schema violation; crashing over it is worse."""
        body = jev_response(
            raw={
                "answers": {"sensitivity": {"choice": "TOP_SECRET", "probabilities": {"public": 0.1, "regulated": 0.9}}}
            }
        )

        async def scenario() -> BackendResult:
            backend, _ = backend_with([StubResponse(200, body)], max_retries=0)
            return await backend.decide(request_)

        answer = asyncio.run(scenario()).answers.sensitivity
        assert answer.choice == "regulated"
        assert answer.probabilities["regulated"] == pytest.approx(0.9)

    def test_missing_choice_uses_the_documented_fallback(self, request_: DecisionRequest) -> None:
        body = jev_response(raw={"answers": {"sensitivity": {"probabilities": {"public": 1.0}}}})

        async def scenario() -> BackendResult:
            backend, _ = backend_with([StubResponse(200, body)], max_retries=0)
            return await backend.decide(request_)

        assert asyncio.run(scenario()).answers.sensitivity.choice == "internal"

    def test_off_ladder_probability_keys_are_dropped(self, request_: DecisionRequest) -> None:
        body = jev_response(
            raw={"answers": {"complexity": {"choice": "hard", "probabilities": {"hard": 0.5, "impossible": 0.5}}}}
        )

        async def scenario() -> BackendResult:
            backend, _ = backend_with([StubResponse(200, body)], max_retries=0)
            return await backend.decide(request_)

        probabilities = asyncio.run(scenario()).answers.complexity.probabilities
        assert "impossible" not in probabilities
        assert probabilities["hard"] == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "body",
        [
            {},  # no answers key at all
            {"answers": {}},  # empty answers
            {"answers": None},
            {"answers": "nonsense"},  # wrong type
            {"answers": {"complexity": "hard"}},  # answer is a string, not a mapping
            {"answers": {"complexity": {"probabilities": {}}}},  # empty distribution
            {"answers": {"complexity": {"probabilities": {"nope": 1.0}}}},  # nothing on the ladder
        ],
    )
    def test_malformed_answers_become_maximum_uncertainty_instead_of_raising(
        self, request_: DecisionRequest, body: dict[str, Any]
    ) -> None:
        async def scenario() -> BackendResult:
            backend, _ = backend_with([StubResponse(200, body)], max_retries=0)
            return await backend.decide(request_)

        result = asyncio.run(scenario())
        unknown = DecisionAnswers.unknown()
        assert result.answers.complexity.probabilities == unknown.complexity.probabilities
        assert result.answers.sensitivity.probabilities == unknown.sensitivity.probabilities
        assert result.answers.domain.probabilities == unknown.domain.probabilities
        assert result.answers.pii.value == 0.5
        for answer in (result.answers.complexity, result.answers.sensitivity, result.answers.domain):
            assert answer.confidence == 0.0

    def test_noul_is_clamped_into_range(self, request_: DecisionRequest) -> None:
        async def scenario(float_value: float) -> float:
            body = jev_response(raw={"answers": {"pii_present": {"noul": float_value}}})
            backend, _ = backend_with([StubResponse(200, body)], max_retries=0)
            return (await backend.decide(request_)).answers.pii.value

        assert asyncio.run(scenario(1.7)) == 1.0
        assert asyncio.run(scenario(-0.4)) == 0.0
        assert asyncio.run(scenario(0.25)) == pytest.approx(0.25)

    @pytest.mark.parametrize("value", ["yes", None, {"noul": 0.5}, [0.5], "0.9"])
    def test_non_numeric_noul_becomes_unknown(self, request_: DecisionRequest, value: Any) -> None:
        """A noul the router cannot read must land on 0.5, the honest coin flip."""
        body = jev_response(raw={"answers": {"pii_present": {"noul": value}}})

        async def scenario() -> float:
            backend, _ = backend_with([StubResponse(200, body)], max_retries=0)
            return (await backend.decide(request_)).answers.pii.value

        assert asyncio.run(scenario()) == 0.5

    @pytest.mark.parametrize(("value", "expected"), [(True, 1.0), (False, 0.0)])
    def test_boolean_noul_is_coerced_numerically(self, request_: DecisionRequest, value: bool, expected: float) -> None:
        """``bool`` is an ``int`` subclass, so a JSON true is read as 1.0 rather than rejected.

        Pinned deliberately: it is a defensible reading of "yes, certainly", and a
        future refactor that tightens the isinstance check should fail here on
        purpose so the change is a decision rather than an accident.
        """
        body = jev_response(raw={"answers": {"pii_present": {"noul": value}}})

        async def scenario() -> float:
            backend, _ = backend_with([StubResponse(200, body)], max_retries=0)
            return (await backend.decide(request_)).answers.pii.value

        assert asyncio.run(scenario()) == expected

    def test_answer_aliases_are_accepted(self, request_: DecisionRequest) -> None:
        """Response keys have varied across System One versions; the parser tolerates them."""
        body = {
            "answers": {
                "data_sensitivity": {"choice": "regulated", "probabilities": {"regulated": 1.0}},
                "has_pii": {"noul": 0.9},
                "task_domain": {"choice": "code", "probabilities": {"code": 1.0}},
            }
        }

        async def scenario() -> BackendResult:
            backend, _ = backend_with([StubResponse(200, body)], max_retries=0)
            return await backend.decide(request_)

        answers = asyncio.run(scenario()).answers
        assert answers.sensitivity.choice == "regulated"
        assert answers.pii.value == pytest.approx(0.9)
        assert answers.domain.choice == "code"

    def test_include_domain_false_yields_a_uniform_domain(self, request_: DecisionRequest) -> None:
        """We did not ask, so we must not invent: uniform, not a guess."""

        async def scenario() -> ChoiceAnswer:
            backend, _ = backend_with([StubResponse(200, OK_BODY)], max_retries=0, include_domain=False)
            return (await backend.decide(request_)).answers.domain

        domain = asyncio.run(scenario())
        assert set(domain.probabilities) == set(DOMAINS)
        assert all(p == pytest.approx(1 / len(DOMAINS)) for p in domain.probabilities.values())

    def test_model_version_comes_from_the_response(self, request_: DecisionRequest) -> None:
        async def scenario() -> tuple[BackendResult, JevBackend]:
            backend, _ = backend_with([StubResponse(200, OK_BODY)], max_retries=0)
            return await backend.decide(request_), backend

        result, backend = asyncio.run(scenario())
        assert result.model_version == "jev-test-1"
        assert backend.model_version == "jev-test-1", "recorded for every later decision"

    def test_model_version_falls_back_to_the_requested_alias(self, request_: DecisionRequest) -> None:
        async def scenario() -> BackendResult:
            backend, _ = backend_with([StubResponse(200, {"answers": {}})], max_retries=0, model="jev-pinned")
            return await backend.decide(request_)

        assert asyncio.run(scenario()).model_version == "jev-pinned"

    def test_a_successful_call_is_not_degraded(self, request_: DecisionRequest) -> None:
        async def scenario() -> BackendResult:
            backend, _ = backend_with([StubResponse(200, OK_BODY)], max_retries=0)
            return await backend.decide(request_)

        result = asyncio.run(scenario())
        assert result.degraded is False
        assert result.degrade_reason is None
        assert result.latency_ms >= 0.0

    def test_answers_are_always_total(self, request_: DecisionRequest) -> None:
        """Every backend must return all four answers, or the policy engine cannot be total."""

        async def scenario() -> DecisionAnswers:
            backend, _ = backend_with([StubResponse(200, OK_BODY)], max_retries=0)
            return (await backend.decide(request_)).answers

        answers = asyncio.run(scenario())
        assert set(answers.complexity.probabilities) == set(COMPLEXITY_LEVELS)
        assert set(answers.sensitivity.probabilities) == set(SENSITIVITY_LEVELS)
        assert set(answers.domain.probabilities) == set(DOMAINS)
        assert 0.0 <= answers.pii.value <= 1.0


# --------------------------------------------------------------------------- #
# HTTP failure handling
# --------------------------------------------------------------------------- #
class TestHttpFailures:
    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 529])
    def test_retryable_statuses_are_retried(self, request_: DecisionRequest, status: int) -> None:
        async def scenario() -> tuple[int, BackendResult]:
            backend, transport = backend_with([status, StubResponse(200, OK_BODY)], max_retries=1)
            result = await backend.decide(request_)
            return transport.call_count, result

        calls, result = asyncio.run(scenario())
        assert calls == 2
        assert result.degraded is False

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_client_errors_are_not_retried(self, request_: DecisionRequest, status: int) -> None:
        """Retrying a 401 three times is how a bad key becomes an account lockout."""

        async def scenario() -> tuple[int, BackendResult]:
            backend, transport = backend_with([status], max_retries=3)
            result = await backend.decide(request_)
            return transport.call_count, result

        calls, result = asyncio.run(scenario())
        assert calls == 1
        assert result.degraded is True
        assert f"http {status}" in (result.degrade_reason or "")

    def test_retries_stop_at_max_retries(self, request_: DecisionRequest) -> None:
        async def scenario() -> tuple[int, BackendResult]:
            backend, transport = backend_with([500], max_retries=2)
            result = await backend.decide(request_)
            return transport.call_count, result

        calls, result = asyncio.run(scenario())
        assert calls == 3  # one attempt plus two retries
        assert result.degraded is True

    def test_zero_retries_means_one_attempt(self, request_: DecisionRequest) -> None:
        async def scenario() -> int:
            backend, transport = backend_with([503], max_retries=0)
            await backend.decide(request_)
            return transport.call_count

        assert asyncio.run(scenario()) == 1

    def test_transport_exception_degrades(self, request_: DecisionRequest) -> None:
        """DNS, TLS and connection-reset all arrive as exceptions, not statuses."""

        async def scenario() -> BackendResult:
            backend, _ = backend_with([RuntimeError("name resolution failed")], max_retries=0)
            return await backend.decide(request_)

        result = asyncio.run(scenario())
        assert result.degraded is True
        assert "RuntimeError" in (result.degrade_reason or "")
        assert "name resolution failed" in (result.degrade_reason or "")

    def test_transport_exception_is_retried(self, request_: DecisionRequest) -> None:
        async def scenario() -> int:
            backend, transport = backend_with([ConnectionError("reset"), StubResponse(200, OK_BODY)], max_retries=1)
            await backend.decide(request_)
            return transport.call_count

        assert asyncio.run(scenario()) == 2

    @pytest.mark.parametrize("answers_shape", [[], [["complexity", {}]], "nope", 7])
    def test_a_non_mapping_answers_block_is_tolerated(self, request_: DecisionRequest, answers_shape: Any) -> None:
        """HTTP succeeded but carried nothing readable: maximum uncertainty, no crash."""

        async def scenario() -> BackendResult:
            backend, _ = backend_with([StubResponse(200, {"answers": answers_shape})], max_retries=0)
            return await backend.decide(request_)

        result = asyncio.run(scenario())
        assert result.degraded is False
        assert result.answers.sensitivity.probabilities == DecisionAnswers.unknown().sensitivity.probabilities

    @pytest.mark.parametrize(
        "body",
        [
            [],  # a JSON array at the top level (a proxy returning a list of errors)
            "not json at all",  # a JSON string body
            5,  # a JSON number body
            {"answers": {"complexity": {"choice": "hard", "probabilities": {"hard": "very"}}}},
        ],
    )
    def test_malformed_body_never_escapes_decide(self, request_: DecisionRequest, body: Any) -> None:
        async def scenario() -> BackendResult:
            response = StubResponse(200, {}, text=repr(body))
            response._body = body  # type: ignore[assignment]
            backend, _ = backend_with([response], max_retries=0)
            return await backend.decide(request_)

        result = asyncio.run(scenario())
        assert result.answers.sensitivity.probabilities == DecisionAnswers.unknown().sensitivity.probabilities

    def test_timeout_degrades_rather_than_raising(self, request_: DecisionRequest) -> None:
        async def scenario() -> BackendResult:
            slow = StubTransport([StubResponse(200, OK_BODY)], sleep=5.0)
            backend = JevBackend(api_key=API_KEY, client=slow, timeout_seconds=0.05, max_retries=0)
            return await backend.decide(request_)

        result = asyncio.run(scenario())
        assert result.degraded is True
        assert "timeout" in (result.degrade_reason or "")

    def test_timeout_is_retried_then_degrades(self, request_: DecisionRequest) -> None:
        async def scenario() -> tuple[int, BackendResult]:
            slow = StubTransport([StubResponse(200, OK_BODY)], sleep=5.0)
            backend = JevBackend(api_key=API_KEY, client=slow, timeout_seconds=0.05, max_retries=1)
            result = await backend.decide(request_)
            return slow.call_count, result

        calls, result = asyncio.run(scenario())
        assert calls == 2
        assert result.degraded is True

    def test_degraded_result_carries_unknown_answers(self, request_: DecisionRequest) -> None:
        """Uniform distributions and zero confidence: the fail-closed shape."""

        async def scenario() -> BackendResult:
            backend, _ = backend_with([400], max_retries=0)
            return await backend.decide(request_)

        result = asyncio.run(scenario())
        unknown = DecisionAnswers.unknown()
        assert result.degraded is True
        assert result.answers.complexity.probabilities == unknown.complexity.probabilities
        assert result.answers.sensitivity.probabilities == unknown.sensitivity.probabilities
        assert result.answers.domain.probabilities == unknown.domain.probabilities
        assert result.answers.pii.value == unknown.pii.value == 0.5
        for answer in (result.answers.complexity, result.answers.sensitivity, result.answers.domain):
            assert answer.confidence == 0.0
            assert answer.confidence_reported is False

    def test_degraded_result_still_records_the_questions_it_would_have_sent(self, request_: DecisionRequest) -> None:
        """An unanswered request is still a training row: the log must show what was asked."""

        async def scenario() -> BackendResult:
            backend, _ = backend_with([400], max_retries=0)
            return await backend.decide(request_)

        assert sorted(asyncio.run(scenario()).questions_sent) == sorted(build_questions())


class TestBackoff:
    def test_exponential_with_a_ceiling_and_no_jitter(self) -> None:
        """No jitter on purpose: a deterministic schedule keeps retry tests exact."""
        assert _backoff(0) == pytest.approx(0.2)
        assert _backoff(1) == pytest.approx(0.4)
        assert _backoff(2) == pytest.approx(0.8)
        assert _backoff(3) == pytest.approx(1.6)
        assert _backoff(4) == 2.0
        assert _backoff(20) == 2.0


# --------------------------------------------------------------------------- #
# Circuit breaker
# --------------------------------------------------------------------------- #
class TestCircuitBreaker:
    def test_starts_closed_and_allows_calls(self, fake_clock: Any) -> None:
        breaker = CircuitBreaker(failure_threshold=3, recovery_seconds=10.0, clock=fake_clock)
        assert breaker.state == "closed"
        assert breaker.allow() is True

    def test_opens_after_n_failures(self, fake_clock: Any) -> None:
        breaker = CircuitBreaker(failure_threshold=3, recovery_seconds=10.0, clock=fake_clock)
        for _ in range(2):
            breaker.on_start()
            breaker.on_failure()
            assert breaker.state == "closed"
        breaker.on_start()
        breaker.on_failure()
        assert breaker.state == "open"
        assert breaker.allow() is False

    def test_a_success_resets_the_failure_count(self, fake_clock: Any) -> None:
        breaker = CircuitBreaker(failure_threshold=3, recovery_seconds=10.0, clock=fake_clock)
        breaker.on_start()
        breaker.on_failure()
        breaker.on_start()
        breaker.on_success()
        breaker.on_start()
        breaker.on_failure()
        breaker.on_start()
        breaker.on_failure()
        assert breaker.state == "closed", "failures must be consecutive"

    def test_stays_open_until_recovery_seconds_elapse(self, fake_clock: Any) -> None:
        breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=10.0, clock=fake_clock)
        breaker.on_start()
        breaker.on_failure()
        assert breaker.state == "open"
        fake_clock.advance(9.99)
        assert breaker.state == "open"
        assert breaker.allow() is False
        fake_clock.advance(0.02)
        assert breaker.state == "half-open"
        assert breaker.allow() is True

    def test_half_open_admits_one_probe_call(self, fake_clock: Any) -> None:
        breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=5.0, half_open_max_calls=1, clock=fake_clock)
        breaker.on_start()
        breaker.on_failure()
        fake_clock.advance(6.0)
        assert breaker.allow() is True
        breaker.on_start()
        assert breaker.allow() is False, "one probe at a time, or an outage gets stampeded"

    def test_one_success_closes_a_half_open_breaker(self, fake_clock: Any) -> None:
        breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=5.0, clock=fake_clock)
        breaker.on_start()
        breaker.on_failure()
        fake_clock.advance(6.0)
        breaker.on_start()
        breaker.on_success()
        assert breaker.state == "closed"
        assert breaker.allow() is True

    def test_one_failure_reopens_a_half_open_breaker(self, fake_clock: Any) -> None:
        breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=5.0, clock=fake_clock)
        breaker.on_start()
        breaker.on_failure()
        fake_clock.advance(6.0)
        assert breaker.state == "half-open"
        breaker.on_start()
        breaker.on_failure()
        assert breaker.state == "open"
        # ...and the recovery window restarts from the re-open, not the first trip.
        fake_clock.advance(4.9)
        assert breaker.state == "open"
        fake_clock.advance(0.2)
        assert breaker.state == "half-open"

    def test_reset_returns_to_closed(self, fake_clock: Any) -> None:
        breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=1000.0, clock=fake_clock)
        breaker.on_start()
        breaker.on_failure()
        assert breaker.state == "open"
        breaker.reset()
        assert breaker.state == "closed"
        assert breaker.allow() is True

    def test_threshold_and_recovery_are_clamped_to_sane_values(self) -> None:
        breaker = CircuitBreaker(failure_threshold=0, recovery_seconds=-5.0, half_open_max_calls=0)
        assert breaker.failure_threshold == 1
        assert breaker.recovery_seconds == 0.0
        assert breaker.half_open_max_calls == 1


class TestBreakerIntegration:
    def test_an_open_breaker_short_circuits_without_calling_the_transport(
        self, request_: DecisionRequest, fake_clock: Any
    ) -> None:
        """The point of the breaker: a dead backend must not add a timeout to every request."""

        async def scenario() -> tuple[int, BackendResult]:
            breaker = CircuitBreaker(failure_threshold=2, recovery_seconds=30.0, clock=fake_clock)
            backend, transport = backend_with([500], max_retries=0, breaker=breaker)
            await backend.decide(request_)
            await backend.decide(request_)
            assert breaker.state == "open"
            third = await backend.decide(request_)
            return transport.call_count, third

        calls, result = asyncio.run(scenario())
        assert calls == 2, "the third decision never reached the transport"
        assert result.degraded is True
        assert "circuit open" in (result.degrade_reason or "")

    def test_breaker_recovers_and_serves_real_answers_again(self, request_: DecisionRequest, fake_clock: Any) -> None:
        async def scenario() -> tuple[str, BackendResult]:
            breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=30.0, clock=fake_clock)
            backend, _transport = backend_with([500, StubResponse(200, OK_BODY)], max_retries=0, breaker=breaker)
            await backend.decide(request_)
            assert breaker.state == "open"
            fake_clock.advance(31.0)
            result = await backend.decide(request_)
            return breaker.state, result

        state, result = asyncio.run(scenario())
        assert state == "closed"
        assert result.degraded is False
        assert result.answers.sensitivity.choice == "internal"

    def test_a_degraded_call_still_counts_as_a_breaker_failure(
        self, request_: DecisionRequest, fake_clock: Any
    ) -> None:
        async def scenario() -> str:
            breaker = CircuitBreaker(failure_threshold=2, recovery_seconds=30.0, clock=fake_clock)
            backend, _ = backend_with([400], max_retries=0, breaker=breaker)
            await backend.decide(request_)
            await backend.decide(request_)
            return breaker.state

        assert asyncio.run(scenario()) == "open"

    def test_backend_uses_its_own_breaker_by_default(self) -> None:
        assert isinstance(JevBackend(api_key=API_KEY, client=StubTransport()).breaker, CircuitBreaker)
