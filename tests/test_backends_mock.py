"""Tests for :mod:`jev_route.backends.mock` -- the reason "no API key" is a promise.

MockBackend is not a test double. It ships, it is the default backend in
``policies/default.yaml``, and it is what a sceptical engineer runs in the first
thirty seconds after cloning the repo. So its contract is unusually strict, and
each group of tests below protects one clause of it:

* **Offline and keyless.** It must construct and answer with no ``TYPESAFE_API_KEY``
  in the environment and no route to the internet. ``conftest`` blocks outbound
  sockets for every test here, which is what turns "these tests pass" into
  evidence rather than hope.
* **Total.** Every ladder option is present in every distribution, every value is
  a probability, and every distribution sums to one. The policy engine indexes
  ``probabilities[level]`` directly; a missing key is a crash in production, not a
  softer answer.
* **Deterministic.** Same input, same output, in any process on any machine. The
  module binds no clock, no RNG and no counter, and the ``PYTHONHASHSEED``
  subprocess test below is what proves the last of those rather than asserting it.
* **Never degraded.** ``degraded`` is the router's fail-closed signal. A mock that
  reported degradation would make every offline demo route to ``local`` and hide
  the policy paths the demo exists to show.
* **Non-degenerate.** Peaked, but never one-hot. A backend that answered ``1.0``
  every time would make the confidence-floor and escalation code paths
  untestable -- and untestable code paths in the part of the system that decides
  what may leave the building is exactly the wrong place to be blind.
* **Not a constant.** It has to carry real signal: a proof is harder than "hey",
  a fenced block is code, an invoice extraction is data extraction, and asking
  *about* a regulated topic is not the same as carrying regulated data.

Where an assertion is a ranking rather than an exact argmax, that is deliberate:
rankings survive a re-tuned marker table, argmaxes do not. The few exact argmaxes
kept here are the ones the README states in words.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from jev_route.backends.base import BackendResult, DecisionBackend, DecisionRequest
from jev_route.backends.mock import MODEL_VERSION, TEMPERATURE, MockBackend
from jev_route.backends.mock import backend as module_backend
from jev_route.prompts import MAX_EXCERPT_CHARS, compute_features, excerpt_from_text
from jev_route.schema import (
    COMPLEXITY_LEVELS,
    DEFAULT_NOUL_THRESHOLD,
    DOMAINS,
    SENSITIVITY_LEVELS,
    ChoiceAnswer,
    DecisionAnswers,
    NoulAnswer,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Probability sums are asserted with this tolerance, never with ``==``.
#: ``_softmax`` rounds each value to 6 decimals independently, so a distribution
#: can legitimately sum to 0.999999 or 1.000002. Verified across the whole labelled
#: dataset: the worst observed deviation is 2e-6.
SUM_TOLERANCE = 1e-5

# --------------------------------------------------------------------------- #
# Prompts. Each one is chosen to isolate a single signal.
# --------------------------------------------------------------------------- #
GREETING = "hey"
PROOF = "Prove that every graph with n vertices and more than n-1 edges contains a cycle."
CODE_BLOCK = (
    "```python\n"
    "async def gather(shards):\n"
    "    return await asyncio.gather(*[fetch(s) for s in shards])\n"
    "```\n"
    "Refactor this to respect a concurrency limit and debug the intermittent deadlock."
)
#: A fence and nothing else that reads as engineering: isolates the feature-driven
#: ``code`` bonus from the keyword-driven one.
CODE_FENCE_ONLY = "```python\ndef f(x):\n    return x * 2\n```\nPlease look over this snippet."
INVOICE = "Extract the invoice totals into JSON"
MID_DIFFICULTY = "Summarize this internal status update for the platform team."
#: Matches no domain marker at all, so the domain answer is an honest "no signal".
NO_DOMAIN_SIGNAL = (
    "Please summarize the quarterly internal status update and circulate it to the platform engineering group."
)
CRASH_PLAIN = "Our nightly import job failed again and we need the root cause before the next release window."
CRASH_WITH_TRACE = (
    CRASH_PLAIN + "\nTraceback (most recent call last):\n"
    '  File "/app/worker.py", line 88, in run\n'
    "    payload = parse(raw)\n"
    "ValueError: malformed record"
)
CONFIDENTIAL = (
    "Confidential and internal only: the patient diagnosis, the prescription list and the salary review are under NDA."
)
PUBLIC = "Write a public blog post about the weather, a joke and a soup recipe for the open source readme."
#: The distinction the whole project turns on: asking ABOUT a regulated topic.
HIPAA_ABOUT = "Explain how HIPAA actually works and who it applies to."
PCI_ABOUT = "What does PCI-DSS require of merchants who store card data?"

#: A spread wide enough that "total" and "non-degenerate" mean something.
PROMPT_SPREAD: tuple[str, ...] = (
    GREETING,
    PROOF,
    CODE_BLOCK,
    CODE_FENCE_ONLY,
    INVOICE,
    MID_DIFFICULTY,
    NO_DOMAIN_SIGNAL,
    CRASH_PLAIN,
    CRASH_WITH_TRACE,
    CONFIDENTIAL,
    PUBLIC,
    HIPAA_ABOUT,
    PCI_ABOUT,
    "",
)

LADDER_ANSWERS: tuple[tuple[str, Callable[[DecisionAnswers], ChoiceAnswer], tuple[str, ...]], ...] = (
    ("complexity", lambda a: a.complexity, COMPLEXITY_LEVELS),
    ("sensitivity", lambda a: a.sensitivity, SENSITIVITY_LEVELS),
    ("domain", lambda a: a.domain, DOMAINS),
)


def make_request(
    text: str,
    *,
    advisory_topics: Sequence[str] = (),
    request_id: str = "mock-test",
    metadata: Mapping[str, Any] | None = None,
    features: Any = None,
) -> DecisionRequest:
    """A :class:`DecisionRequest` with features computed exactly as production does."""
    return DecisionRequest(
        redacted_excerpt=text,
        features=features if features is not None else compute_features(text),
        advisory_topics=tuple(advisory_topics),
        metadata=dict(metadata or {}),
        request_id=request_id,
    )


@pytest.fixture
def mock() -> MockBackend:
    return MockBackend()


def answer_for(mock: MockBackend, text: str, **kwargs: Any) -> DecisionAnswers:
    return mock.decide_sync(make_request(text, **kwargs))


def hard_mass(answers: DecisionAnswers) -> float:
    """P(hard) + P(frontier): the "this needs a big model" mass."""
    return answers.complexity.probability_of("hard") + answers.complexity.probability_of("frontier")


def sensitive_mass(answers: DecisionAnswers) -> float:
    """P(confidential) + P(regulated): the "this must not leave" mass."""
    return answers.sensitivity.probability_of("confidential") + answers.sensitivity.probability_of("regulated")


def runner_up(probabilities: Mapping[str, float], choice: str) -> float:
    """The largest probability that is not the winner's."""
    return max(v for k, v in probabilities.items() if k != choice)


# --------------------------------------------------------------------------- #
# Identity and lifecycle
# --------------------------------------------------------------------------- #
class TestIdentity:
    """What the config layer and the decision log see."""

    def test_name_is_mock(self) -> None:
        assert MockBackend().name == "mock"

    def test_model_version_matches_the_module_constant(self) -> None:
        # The log records backend_model_version for reproducibility. If the instance
        # and the constant ever drift, two runs of the same code would look like two
        # different models in the dataset.
        assert MockBackend().model_version == MODEL_VERSION
        assert MODEL_VERSION == "mock-1.0.0"

    def test_identity_is_readable_without_instantiating(self) -> None:
        # Backend resolution reads the class attributes off a dotted path; a rename
        # that moved them onto __init__ would break config loading, not just this test.
        assert MockBackend.name == "mock"
        assert MockBackend.model_version == MODEL_VERSION

    def test_it_satisfies_the_decision_backend_protocol(self, mock: MockBackend) -> None:
        assert isinstance(mock, DecisionBackend)

    def test_the_module_level_singleton_is_a_mock_backend(self) -> None:
        # Mirrors how LiteLLM resolves a dotted-path plugin: importing the module
        # must hand back a usable backend, not a factory.
        assert isinstance(module_backend, MockBackend)
        assert module_backend.name == "mock"

    async def test_aclose_returns_none_and_is_safe_to_call_twice(self, mock: MockBackend) -> None:
        # Router.aclose() closes every backend during teardown and is documented as
        # never raising, so a backend that minded being closed twice would break it.
        assert await mock.aclose() is None
        assert await mock.aclose() is None

    def test_the_default_temperature_is_the_documented_one(self, mock: MockBackend) -> None:
        assert mock.temperature == TEMPERATURE
        assert TEMPERATURE == 0.55

    def test_no_api_key_is_needed(self, mock: MockBackend) -> None:
        # conftest deletes TYPESAFE_API_KEY for every test, so this is the real
        # assertion: the headline "runs with no key" claim, checked where it lives.
        assert "TYPESAFE_API_KEY" not in os.environ
        assert answer_for(mock, MID_DIFFICULTY).complexity.choice in COMPLEXITY_LEVELS


# --------------------------------------------------------------------------- #
# The shape of a result
# --------------------------------------------------------------------------- #
class TestResultShape:
    """``decide()`` returns a complete, honest, non-degraded result."""

    async def test_decide_returns_a_backend_result(self, mock: MockBackend) -> None:
        # Nothing in this test opens a socket; conftest would have raised
        # NetworkBlockedError if it had, which is what makes "offline" evidence here.
        result = await mock.decide(make_request(MID_DIFFICULTY))
        assert isinstance(result, BackendResult)

    async def test_the_mock_is_never_degraded(self, mock: MockBackend) -> None:
        result = await mock.decide(make_request(MID_DIFFICULTY))
        # degraded is the router's fail-closed trigger. A mock that reported it would
        # send every offline demo to the local tier and hide the policy paths.
        assert result.degraded is False
        assert result.degrade_reason is None

    async def test_latency_is_zero(self, mock: MockBackend) -> None:
        result = await mock.decide(make_request(MID_DIFFICULTY))
        # Exactly zero, not merely small: the log stores this number, and a mock that
        # measured its own wall time would put a nondeterministic value in the dataset.
        assert result.latency_ms == 0.0

    async def test_no_questions_are_sent_anywhere(self, mock: MockBackend) -> None:
        result = await mock.decide(make_request(MID_DIFFICULTY))
        assert dict(result.questions_sent) == {}
        # Not laziness. questions_sent is recorded verbatim on every decision, so an
        # empty mapping is what makes "no data egress" auditable from the log itself
        # rather than something you have to take the README's word for.

    async def test_the_result_carries_the_model_version(self, mock: MockBackend) -> None:
        result = await mock.decide(make_request(MID_DIFFICULTY))
        assert result.model_version == MODEL_VERSION

    async def test_decide_and_decide_sync_agree(self, mock: MockBackend) -> None:
        request = make_request(PROOF)
        result = await mock.decide(request)
        assert result.answers == mock.decide_sync(request)

    def test_decide_sync_is_public_and_synchronous(self, mock: MockBackend) -> None:
        # Public on purpose: the CLI and this suite call it without an event loop.
        # If it returned a coroutine the assertion below would compare a coroutine
        # object and quietly pass nothing.
        assert not mock.decide_sync.__name__.startswith("_")
        answers = mock.decide_sync(make_request(GREETING))
        assert isinstance(answers, DecisionAnswers)

    def test_answers_are_total_in_shape(self, mock: MockBackend) -> None:
        answers = answer_for(mock, MID_DIFFICULTY)
        assert isinstance(answers.complexity, ChoiceAnswer)
        assert isinstance(answers.sensitivity, ChoiceAnswer)
        assert isinstance(answers.domain, ChoiceAnswer)
        assert isinstance(answers.pii, NoulAnswer)

    async def test_the_result_answers_are_total_in_shape(self, mock: MockBackend) -> None:
        answers = (await mock.decide(make_request(MID_DIFFICULTY))).answers
        # The schema promises four answers, always. A backend that omitted one would
        # turn every ``answers.pii.value`` in the policy engine into an AttributeError.
        assert set(answers.to_dict()) == {"complexity", "sensitivity", "pii", "domain"}


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
#: Printed by a fresh interpreter so the distribution can be compared across
#: PYTHONHASHSEED values. Deliberately self-contained: importing this test module
#: would drag pytest into the child process and muddy what is being measured.
_HASHSEED_PROBE = """
import json
from jev_route.backends.base import DecisionRequest
from jev_route.backends.mock import MockBackend
from jev_route.prompts import compute_features

texts = %r
out = {}
for text in texts:
    answers = MockBackend().decide_sync(
        DecisionRequest(redacted_excerpt=text, features=compute_features(text))
    )
    out[text] = {
        "complexity": dict(answers.complexity.probabilities),
        "sensitivity": dict(answers.sensitivity.probabilities),
        "domain": dict(answers.domain.probabilities),
        "pii": answers.pii.value,
    }
print(json.dumps(out, sort_keys=True))
"""


def probe_payload() -> dict[str, Any]:
    """The same computation as ``_HASHSEED_PROBE``, run in this process."""
    out: dict[str, Any] = {}
    for text in (GREETING, PROOF, INVOICE, CONFIDENTIAL):
        answers = MockBackend().decide_sync(make_request(text))
        out[text] = {
            "complexity": dict(answers.complexity.probabilities),
            "sensitivity": dict(answers.sensitivity.probabilities),
            "domain": dict(answers.domain.probabilities),
            "pii": answers.pii.value,
        }
    return out


def run_probe(seed: str) -> dict[str, Any]:
    """Run ``_HASHSEED_PROBE`` in a child interpreter with a pinned hash seed."""
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    env.pop("TYPESAFE_API_KEY", None)
    script = _HASHSEED_PROBE % ((GREETING, PROOF, INVOICE, CONFIDENTIAL),)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload: dict[str, Any] = json.loads(proc.stdout.strip().splitlines()[-1])
    return payload


class TestDeterminism:
    """Same input, same output -- in this process, and in any other."""

    def test_calling_twice_on_one_instance_is_equal(self, mock: MockBackend) -> None:
        request = make_request(PROOF)
        assert mock.decide_sync(request) == mock.decide_sync(request)

    def test_two_instances_agree(self, mock: MockBackend) -> None:
        request = make_request(PROOF)
        # No per-instance counter or seeded state: two backends must be
        # interchangeable, or a router that rebuilds its backend would change answers.
        assert MockBackend().decide_sync(request) == mock.decide_sync(request)

    def test_answers_do_not_depend_on_the_request_id(self, mock: MockBackend) -> None:
        a = answer_for(mock, PROOF, request_id="aaa")
        b = answer_for(mock, PROOF, request_id="zzz")
        assert a == b

    def test_answers_do_not_depend_on_caller_metadata(self, mock: MockBackend) -> None:
        a = answer_for(mock, PROOF, metadata={"tenant": "acme"})
        b = answer_for(mock, PROOF, metadata={"tenant": "other", "api_key_alias": "prod"})
        # Metadata is provenance for the log, not evidence about the prompt. A mock
        # that read it would route two tenants differently on identical text.
        assert a == b

    def test_answers_do_not_depend_on_feature_mapping_order(self, mock: MockBackend) -> None:
        features = compute_features(INVOICE)
        one = dataclasses.replace(features, gate_detectors={"payment_card": 1, "iban": 2})
        two = dataclasses.replace(features, gate_detectors={"iban": 2, "payment_card": 1})
        request_one = make_request(INVOICE, features=one)
        request_two = make_request(INVOICE, features=two)
        # Dict ordering is the classic source of "works on my machine" nondeterminism.
        assert one == two
        assert mock.decide_sync(request_one) == mock.decide_sync(request_two)

    def test_no_clock_or_rng_is_consulted(self, mock: MockBackend, monkeypatch: pytest.MonkeyPatch) -> None:
        def forbid(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("MockBackend must not read the clock or an RNG")

        for target in (
            "time.time",
            "time.monotonic",
            "random.random",
            "random.uniform",
            "random.choice",
            "os.urandom",
        ):
            monkeypatch.setattr(target, forbid)
        # Behavioural proof of the module docstring's "no randomness, no clock, no
        # counter". Answering at all with every source of nondeterminism poisoned is
        # stronger than asserting the module happens not to import them.
        first = mock.decide_sync(make_request(CONFIDENTIAL))
        second = mock.decide_sync(make_request(CONFIDENTIAL))
        assert first == second

    def test_the_module_binds_no_time_random_or_os_names(self) -> None:
        import jev_route.backends.mock as mock_module

        # The structural companion to the poisoned-clock test above: if a future
        # change reaches for one of these, this fails before it can reach production.
        for banned in ("time", "random", "os", "uuid"):
            assert not hasattr(mock_module, banned), banned

    @pytest.mark.parametrize("seed", ["0", "12345", "424242"])
    def test_answers_do_not_depend_on_pythonhashseed(self, seed: str) -> None:
        # String hashing is randomised per process, and the mock's distributions come
        # from dict comprehensions over fixed ladders. A dependency on iteration order
        # would show up only in a fresh interpreter, never in a single pytest run --
        # so the check has to leave this process.
        assert run_probe(seed) == probe_payload()

    def test_every_prompt_in_the_spread_is_stable_under_repetition(self, mock: MockBackend) -> None:
        for text in PROMPT_SPREAD:
            first = answer_for(mock, text)
            for _ in range(3):
                assert answer_for(mock, text) == first, repr(text)


# --------------------------------------------------------------------------- #
# Totality
# --------------------------------------------------------------------------- #
class TestTotality:
    """A backend that cannot answer must still answer, at full width."""

    @pytest.mark.parametrize("text", PROMPT_SPREAD)
    def test_every_ladder_option_is_present(self, mock: MockBackend, text: str) -> None:
        answers = answer_for(mock, text)
        for name, getter, ladder in LADDER_ANSWERS:
            probabilities = getter(answers).probabilities
            # Exact key set and exact order. The policy engine and the log format both
            # assume the ladder is total; a dropped option is a KeyError in production.
            assert tuple(probabilities) == tuple(ladder), name

    @pytest.mark.parametrize("text", PROMPT_SPREAD)
    def test_every_probability_is_a_probability(self, mock: MockBackend, text: str) -> None:
        answers = answer_for(mock, text)
        for name, getter, ladder in LADDER_ANSWERS:
            for option in ladder:
                value = getter(answers).probability_of(option)
                assert 0.0 <= value <= 1.0, f"{name}.{option}={value}"

    @pytest.mark.parametrize("text", PROMPT_SPREAD)
    def test_every_distribution_sums_to_one(self, mock: MockBackend, text: str) -> None:
        answers = answer_for(mock, text)
        for name, getter, _ladder in LADDER_ANSWERS:
            total = sum(getter(answers).probabilities.values())
            # approx, never ==: _softmax rounds each value to 6 decimals on its own,
            # so a distribution can legitimately sum to 0.999999. Verified worst case
            # across the 223-prompt labelled dataset: 2e-6.
            assert total == pytest.approx(1.0, abs=SUM_TOLERANCE), f"{name} sums to {total!r}"

    @pytest.mark.parametrize("text", PROMPT_SPREAD)
    def test_the_choice_is_a_member_of_its_ladder(self, mock: MockBackend, text: str) -> None:
        answers = answer_for(mock, text)
        for name, getter, ladder in LADDER_ANSWERS:
            assert getter(answers).choice in ladder, name

    @pytest.mark.parametrize("text", PROMPT_SPREAD)
    def test_the_choice_carries_the_highest_probability(self, mock: MockBackend, text: str) -> None:
        answers = answer_for(mock, text)
        for name, getter, _ladder in LADDER_ANSWERS:
            answer = getter(answers)
            # Stated as "the winner has the max" rather than recomputing an argmax, so
            # the assertion stays true on ties (where _argmax keeps insertion order).
            assert answer.probabilities[answer.choice] == max(answer.probabilities.values()), name

    @pytest.mark.parametrize("text", PROMPT_SPREAD)
    def test_the_noul_answer_is_total(self, mock: MockBackend, text: str) -> None:
        pii = answer_for(mock, text).pii
        assert set(pii.probabilities) == {"yes", "no"}
        assert sum(pii.probabilities.values()) == pytest.approx(1.0, abs=SUM_TOLERANCE)

    def test_an_empty_excerpt_does_not_raise_and_is_still_total(self, mock: MockBackend) -> None:
        answers = answer_for(mock, "")
        assert tuple(answers.complexity.probabilities) == COMPLEXITY_LEVELS
        assert sum(answers.complexity.probabilities.values()) == pytest.approx(1.0, abs=SUM_TOLERANCE)
        assert answers.pii.value >= 0.01

    def test_a_whitespace_only_excerpt_is_still_total(self, mock: MockBackend) -> None:
        # A truncated or fully redacted prompt arrives as blanks in real traffic;
        # "no text" is a routing input, not an error condition.
        answers = answer_for(mock, "   \n\t ")
        assert tuple(answers.sensitivity.probabilities) == SENSITIVITY_LEVELS
        assert answers.complexity.choice in COMPLEXITY_LEVELS

    def test_the_whole_labelled_dataset_gets_total_answers(self, mock: MockBackend, sample_prompts: list[Any]) -> None:
        for prompt in sample_prompts:
            answers = answer_for(mock, prompt.text)
            for name, getter, ladder in LADDER_ANSWERS:
                probabilities = getter(answers).probabilities
                assert tuple(probabilities) == tuple(ladder), f"{prompt.id}:{name}"
                assert sum(probabilities.values()) == pytest.approx(1.0, abs=SUM_TOLERANCE), prompt.id


# --------------------------------------------------------------------------- #
# Confidence
# --------------------------------------------------------------------------- #
class TestConfidence:
    """The documented formula, and honest provenance for the number."""

    @pytest.mark.parametrize("text", PROMPT_SPREAD)
    def test_confidence_is_top_one_minus_top_two(self, mock: MockBackend, text: str) -> None:
        answers = answer_for(mock, text)
        for name, getter, _ladder in LADDER_ANSWERS:
            answer = getter(answers)
            top_two = sorted(answer.probabilities.values(), reverse=True)[:2]
            expected = round(max(0.0, min(1.0, top_two[0] - top_two[1])), 6)
            assert answer.confidence == expected, name

    @pytest.mark.parametrize("text", PROMPT_SPREAD)
    def test_confidence_is_rounded_to_six_decimals(self, mock: MockBackend, text: str) -> None:
        answers = answer_for(mock, text)
        for name, getter, _ladder in LADDER_ANSWERS:
            confidence = getter(answers).confidence
            assert confidence == round(confidence, 6), name

    @pytest.mark.parametrize("text", PROMPT_SPREAD)
    def test_confidence_is_never_reported(self, mock: MockBackend, text: str) -> None:
        answers = answer_for(mock, text)
        for name, getter, _ladder in LADDER_ANSWERS:
            # The mock derives confidence from its own distribution; it is not a
            # calibrated measurement. Claiming confidence_reported=True would let a
            # derived number be compared against itself in distill.evaluate.
            assert getter(answers).confidence_reported is False, name

    @pytest.mark.parametrize("text", PROMPT_SPREAD)
    def test_confidence_is_within_the_unit_interval(self, mock: MockBackend, text: str) -> None:
        answers = answer_for(mock, text)
        for name, getter, _ladder in LADDER_ANSWERS:
            assert 0.0 <= getter(answers).confidence <= 1.0, name

    def test_the_derived_confidence_is_not_the_entropy_based_one(self, mock: MockBackend) -> None:
        answer = answer_for(mock, MID_DIFFICULTY).complexity
        # ChoiceAnswer keeps both on purpose: top1-top2 is comparable across
        # backends, normalized entropy is reproducible offline. They are different
        # numbers, and a change that collapsed them would lose one of the two views.
        assert answer.confidence != pytest.approx(answer.computed_confidence)
        assert 0.0 <= answer.computed_confidence <= 1.0


# --------------------------------------------------------------------------- #
# Non-degeneracy: the reason the confidence-floor paths are testable at all
# --------------------------------------------------------------------------- #
class TestNonDegenerate:
    """Peaked, but never one-hot."""

    def test_runner_up_mass_is_real_for_at_least_one_answer(self, mock: MockBackend) -> None:
        found = [
            (text, name, runner_up(getter(answer_for(mock, text)).probabilities, getter(answer_for(mock, text)).choice))
            for text in PROMPT_SPREAD
            for name, getter, _ladder in LADDER_ANSWERS
        ]
        assert any(mass > 0.01 for _text, _name, mass in found), found

    def test_a_mid_difficulty_prompt_is_genuinely_uncertain(self, mock: MockBackend) -> None:
        answers = answer_for(mock, MID_DIFFICULTY)
        # Strictly inside (0, 1) for both judgement calls. This is the property the
        # module docstring names: with confidence pinned at 1.0 the on_uncertain
        # bump and the escalation paths could never fire, and the mock exists to
        # make them fire offline.
        assert 0.0 < answers.complexity.confidence < 1.0
        assert 0.0 < answers.sensitivity.confidence < 1.0

    def test_no_prompt_in_the_dataset_gets_a_one_hot_answer(self, mock: MockBackend, sample_prompts: list[Any]) -> None:
        for prompt in sample_prompts:
            answers = answer_for(mock, prompt.text)
            for name, getter, _ladder in LADDER_ANSWERS:
                answer = getter(answers)
                assert runner_up(answer.probabilities, answer.choice) > 0.0, f"{prompt.id}:{name}"

    def test_complexity_confidence_never_saturates_on_the_dataset(
        self, mock: MockBackend, sample_prompts: list[Any]
    ) -> None:
        for prompt in sample_prompts:
            confidence = answer_for(mock, prompt.text).complexity.confidence
            # Both ends matter: 1.0 would disable the confidence floor, 0.0 would
            # escalate everything to the strongest tier and make the mock useless.
            assert 0.0 < confidence < 1.0, prompt.id

    def test_the_confidence_floor_paths_are_reachable(self, mock: MockBackend, sample_prompts: list[Any]) -> None:
        # 0.7 and 0.8 are the on_uncertain defaults a policy ships with. If no prompt
        # ever fell below them, the bump logic would be dead code in every offline run.
        complexity_low = [p.id for p in sample_prompts if answer_for(mock, p.text).complexity.confidence < 0.7]
        sensitivity_low = [p.id for p in sample_prompts if answer_for(mock, p.text).sensitivity.confidence < 0.8]
        assert complexity_low, "no prompt exercises the complexity confidence floor"
        assert sensitivity_low, "no prompt exercises the sensitivity confidence floor"

    def test_a_prompt_with_no_domain_signal_answers_uniformly(self, mock: MockBackend) -> None:
        domain = answer_for(mock, NO_DOMAIN_SIGNAL).domain
        # Verified reality, and the honest one: when nothing matches, all five
        # domains get the same 0.35 prior and the softmax returns a flat 0.2 each.
        assert domain.confidence == 0.0
        # Compared element-wise, not as a set: pytest.approx() returns an
        # unhashable ApproxScalar, so `{pytest.approx(0.2)}` raises TypeError
        # before the assertion is ever evaluated.
        assert all(p == pytest.approx(0.2) for p in domain.probabilities.values())
        assert len(domain.probabilities) == 5
        # A flat distribution resolves to the first ladder entry, so an unqualified
        # "domain == code" assertion can pass on a tie. Pinned here so that cannot
        # be mistaken for signal.
        assert domain.choice == DOMAINS[0]

    def test_the_dataset_contains_prompts_with_no_domain_signal(
        self, mock: MockBackend, sample_prompts: list[Any]
    ) -> None:
        # The uniform answer must be reachable from real traffic, not just from a
        # prompt constructed to produce it.
        assert any(answer_for(mock, p.text).domain.confidence == 0.0 for p in sample_prompts)


# --------------------------------------------------------------------------- #
# Signal: a crude read, but a real one
# --------------------------------------------------------------------------- #
class TestSignal:
    """The mock is deliberately less accurate than Jev. It is not arbitrary."""

    def test_a_greeting_reads_as_trivial(self, mock: MockBackend) -> None:
        # One of the two argmaxes the README states in words, so it is asserted exactly.
        assert answer_for(mock, GREETING).complexity.choice == "trivial"

    def test_a_greeting_reads_as_chat(self, mock: MockBackend) -> None:
        domain = answer_for(mock, GREETING).domain
        assert domain.choice == "chat"
        assert domain.confidence > 0.0

    def test_a_mathematical_proof_reads_as_hard(self, mock: MockBackend) -> None:
        # The other README-stated argmax.
        assert answer_for(mock, PROOF).complexity.choice == "hard"

    @pytest.mark.parametrize(
        "text",
        [PROOF, CODE_BLOCK, CRASH_WITH_TRACE],
        ids=["proof", "code-block", "stack-trace"],
    )
    def test_harder_prompts_outscore_a_greeting_on_complexity(self, mock: MockBackend, text: str) -> None:
        # Ranking, not argmax: a re-tuned marker table should not break this, and
        # "more demanding text scores higher" is the actual claim being made.
        assert hard_mass(answer_for(mock, text)) > hard_mass(answer_for(mock, GREETING))

    def test_a_fenced_code_block_reads_as_hard_and_as_code(self, mock: MockBackend) -> None:
        answers = answer_for(mock, CODE_BLOCK)
        assert answers.complexity.choice == "hard"
        assert answers.domain.choice == "code"
        # Confidence > 0 proves this is signal and not a tie resolved by ladder order.
        assert answers.domain.confidence > 0.5

    def test_a_code_fence_alone_is_enough_to_read_as_code(self, mock: MockBackend) -> None:
        domain = answer_for(mock, CODE_FENCE_ONLY).domain
        # No domain keyword here at all: the shape feature (features.code_blocks) is
        # doing the work, which is what lets the mock read prompts it has no
        # vocabulary for.
        assert domain.choice == "code"
        assert domain.probability_of("code") > domain.probability_of("writing")

    def test_an_extraction_request_reads_as_data_extraction(self, mock: MockBackend) -> None:
        domain = answer_for(mock, INVOICE).domain
        assert domain.choice == "data-extraction"
        assert domain.confidence > 0.0

    def test_a_stack_trace_pushes_complexity_up(self, mock: MockBackend) -> None:
        with_trace = hard_mass(answer_for(mock, CRASH_WITH_TRACE))
        without_trace = hard_mass(answer_for(mock, CRASH_PLAIN))
        # Same sentence, plus the traceback. features.has_stack_trace is the only
        # thing that changed, so the delta is attributable.
        assert compute_features(CRASH_WITH_TRACE).has_stack_trace is True
        assert compute_features(CRASH_PLAIN).has_stack_trace is False
        assert with_trace > without_trace

    def test_longer_prompts_outscore_shorter_ones_on_complexity(self, mock: MockBackend) -> None:
        short = "Summarize the report."
        medium = short + (" The report covers the quarterly platform numbers and the follow up work." * 2)
        long = short + (" The report covers the quarterly platform numbers and the follow up work in detail." * 20)
        scores = [hard_mass(answer_for(mock, text)) for text in (short, medium, long)]
        # Monotone in length: char_len feeds the score through log1p, so a 50 KB
        # incident dump must not read as trivially as the sentence that opened it.
        assert scores == sorted(scores), scores
        assert scores[0] < scores[-1]

    def test_more_questions_push_complexity_up(self, mock: MockBackend) -> None:
        base = NO_DOMAIN_SIGNAL
        with_questions = base + " Why did the numbers move? What should we change? Who owns the follow up?"
        assert compute_features(with_questions).question_marks > compute_features(base).question_marks
        assert hard_mass(answer_for(mock, with_questions)) > hard_mass(answer_for(mock, base))

    def test_a_diff_pushes_complexity_up(self, mock: MockBackend) -> None:
        base = CRASH_PLAIN
        with_diff = base + "\n--- a/worker.py\n+++ b/worker.py\n@@ -1,3 +1,3 @@\n-old = parse(raw)\n+new = parse(raw)\n"
        assert compute_features(with_diff).has_diff is True
        assert hard_mass(answer_for(mock, with_diff)) > hard_mass(answer_for(mock, base))


# --------------------------------------------------------------------------- #
# Sensitivity, advisory topics and PII
# --------------------------------------------------------------------------- #
class TestSensitivityAndPii:
    """The distinction the project exists to get right."""

    @pytest.mark.parametrize(
        "text",
        [GREETING, MID_DIFFICULTY, HIPAA_ABOUT, PUBLIC, INVOICE],
        ids=["greeting", "mid", "hipaa-about", "public", "invoice"],
    )
    def test_advisory_topics_raise_the_sensitive_mass(self, mock: MockBackend, text: str) -> None:
        plain = sensitive_mass(answer_for(mock, text))
        hinted = sensitive_mass(answer_for(mock, text, advisory_topics=("kw_health_regulation",)))
        # Strictly greater, on every text tried. The gate's topic hints are weak
        # evidence and must move the answer -- a hint that could not move anything
        # would be decoration, and the "look harder here" contract would be a lie.
        assert hinted > plain, f"{text!r}: {plain} -> {hinted}"

    def test_more_advisory_topics_raise_it_further(self, mock: MockBackend) -> None:
        none = sensitive_mass(answer_for(mock, MID_DIFFICULTY))
        one = sensitive_mass(answer_for(mock, MID_DIFFICULTY, advisory_topics=("kw_health_regulation",)))
        two = sensitive_mass(
            answer_for(mock, MID_DIFFICULTY, advisory_topics=("kw_health_regulation", "kw_financial_regulation"))
        )
        assert none < one < two

    def test_advisory_topics_raise_the_sensitive_mass_across_the_dataset(
        self, mock: MockBackend, sample_prompts: list[Any]
    ) -> None:
        for prompt in sample_prompts:
            plain = sensitive_mass(answer_for(mock, prompt.text))
            hinted = sensitive_mass(answer_for(mock, prompt.text, advisory_topics=("kw_health_regulation",)))
            assert hinted > plain, prompt.id

    def test_pii_is_a_pure_function_of_the_sensitivity_distribution(self, mock: MockBackend) -> None:
        for text in PROMPT_SPREAD:
            for topics in ((), ("kw_health_regulation",), ("kw_financial_regulation", "kw_health_regulation")):
                answers = answer_for(mock, text, advisory_topics=topics)
                expected = round(max(0.01, min(0.99, sensitive_mass(answers) * 0.9)), 6)
                # The formula _pii documents, checked exactly. It also pins the
                # negative half of that docstring: there is no separate advisory term,
                # so a topic keyword can only move PII by way of sensitivity.
                assert answers.pii.value == expected, f"{text!r} {topics}"

    def test_a_confidential_prompt_scores_higher_pii_than_a_public_one(self, mock: MockBackend) -> None:
        confidential = answer_for(mock, CONFIDENTIAL)
        public = answer_for(mock, PUBLIC)
        assert confidential.sensitivity.choice == "confidential"
        assert public.sensitivity.choice == "public"
        # Correlated on purpose: a mock answering sensitivity=confidential and
        # pii=0.01 would be internally inconsistent and would hide real bugs in the
        # policy engine's merging logic.
        assert confidential.pii.value > public.pii.value

    def test_pii_stays_inside_its_open_bounds_on_the_dataset(
        self, mock: MockBackend, sample_prompts: list[Any]
    ) -> None:
        for prompt in sample_prompts:
            value = answer_for(mock, prompt.text).pii.value
            # Clamped to [0.01, 0.99] so a mock answer can never be an absolute:
            # 0.0 and 1.0 would let the mock assert certainty it does not have.
            assert 0.01 <= value <= 0.99, f"{prompt.id}: {value}"

    def test_asking_about_a_regulated_topic_is_not_carrying_regulated_data(self, mock: MockBackend) -> None:
        for text in (HIPAA_ABOUT, PCI_ABOUT):
            answers = answer_for(mock, text)
            # This is the single most important read in the offline demo. Get it
            # wrong and "explain how HIPAA works" is air-gapped, which makes the
            # mock indistinguishable from the keyword router jev-route exists to fix.
            assert answers.sensitivity.choice == "public", text
            assert answers.sensitivity.probability_of("public") > 0.9
            assert answers.pii.value < DEFAULT_NOUL_THRESHOLD

    def test_an_advisory_hint_cannot_make_a_topic_question_look_like_pii(self, mock: MockBackend) -> None:
        hinted = answer_for(mock, PCI_ABOUT, advisory_topics=("kw_financial_regulation",))
        # The gate really does flag this text, so the hint is present; the promise is
        # that it still does not amount to "cardholder data is in this prompt".
        assert hinted.sensitivity.choice == "public"
        assert hinted.pii.value < DEFAULT_NOUL_THRESHOLD

    def test_pii_confidence_is_distance_from_a_coin_flip(self, mock: MockBackend) -> None:
        pii = answer_for(mock, CONFIDENTIAL).pii
        assert pii.confidence == pytest.approx(abs(2.0 * pii.value - 1.0))
        assert pii.is_true() is True
        assert answer_for(mock, PUBLIC).pii.is_true() is False


# --------------------------------------------------------------------------- #
# The temperature knob
# --------------------------------------------------------------------------- #
class TestTemperature:
    """One knob, and it has to do the one thing it says."""

    def test_a_low_temperature_sharpens(self) -> None:
        sharp = MockBackend(temperature=0.01).decide_sync(make_request(MID_DIFFICULTY))
        flat = MockBackend(temperature=5.0).decide_sync(make_request(MID_DIFFICULTY))
        assert sharp.complexity.confidence > flat.complexity.confidence
        assert sharp.sensitivity.confidence > flat.sensitivity.confidence
        assert max(sharp.complexity.probabilities.values()) > max(flat.complexity.probabilities.values())

    def test_a_high_temperature_flattens_but_stays_total(self) -> None:
        flat = MockBackend(temperature=5.0).decide_sync(make_request(PROOF))
        # Flattening must not become truncation: the argmax survives a hot softmax.
        assert flat.complexity.choice == "hard"
        assert sum(flat.complexity.probabilities.values()) == pytest.approx(1.0, abs=SUM_TOLERANCE)
        assert flat.complexity.confidence < MockBackend().decide_sync(make_request(PROOF)).complexity.confidence

    @pytest.mark.parametrize("temperature", [0.0, -1.0, 1e-9, 0.01, 1.0, 5.0, 100.0])
    def test_extreme_temperatures_never_raise_or_de_normalise(self, temperature: float) -> None:
        # _softmax clamps at 1e-6, so temperature=0 is a sharp answer rather than a
        # ZeroDivisionError on the request path.
        answers = MockBackend(temperature=temperature).decide_sync(make_request(CONFIDENTIAL))
        for name, getter, ladder in LADDER_ANSWERS:
            probabilities = getter(answers).probabilities
            assert tuple(probabilities) == tuple(ladder), name
            assert sum(probabilities.values()) == pytest.approx(1.0, abs=SUM_TOLERANCE), name
        assert 0.01 <= answers.pii.value <= 0.99

    def test_the_temperature_is_stored_on_the_instance(self) -> None:
        assert MockBackend(temperature=1.5).temperature == 1.5


# --------------------------------------------------------------------------- #
# Scale
# --------------------------------------------------------------------------- #
class TestScale:
    """Big inputs must be cheap, bounded and still deterministic."""

    @pytest.fixture
    def huge_prompt(self) -> str:
        return "Analyze the p99 latency bottleneck and optimize the sharding layer. " * 800

    def test_a_50_kb_prompt_completes_quickly(self, mock: MockBackend, huge_prompt: str) -> None:
        assert len(huge_prompt) > 50_000
        started = time.perf_counter()
        answers = answer_for(mock, huge_prompt)
        elapsed = time.perf_counter() - started
        # The router runs inline on the request path. A backend that took seconds on
        # a long prompt would be slower than the model call it is routing.
        assert elapsed < 1.0, f"took {elapsed:.3f}s"
        assert answers.complexity.choice in COMPLEXITY_LEVELS

    def test_a_50_kb_prompt_is_still_deterministic_and_total(self, mock: MockBackend, huge_prompt: str) -> None:
        first = answer_for(mock, huge_prompt)
        assert answer_for(mock, huge_prompt) == first
        for name, getter, ladder in LADDER_ANSWERS:
            assert tuple(getter(first).probabilities) == tuple(ladder), name

    def test_the_production_excerpt_bound_caps_what_the_mock_sees(self, huge_prompt: str) -> None:
        excerpt = excerpt_from_text(huge_prompt)
        assert len(excerpt) <= MAX_EXCERPT_CHARS
        assert len(excerpt) < len(huge_prompt)
        # The bound lives in prompts.excerpt_from_text, not in the mock: the backend
        # is handed an already-trimmed excerpt. Asserting the cap here keeps the two
        # modules honest about which one owns the limit.
        answers = MockBackend().decide_sync(make_request(excerpt))
        assert answers.complexity.choice in COMPLEXITY_LEVELS

    def test_a_long_prompt_still_reads_as_hard(self, mock: MockBackend, huge_prompt: str) -> None:
        # Length and engineering vocabulary should agree, not cancel out.
        assert hard_mass(answer_for(mock, huge_prompt)) > hard_mass(answer_for(mock, GREETING))


# --------------------------------------------------------------------------- #
# Agreement with the shared fixtures
# --------------------------------------------------------------------------- #
class TestFixtureAgreement:
    """This module's helper must build requests the way the rest of the suite does."""

    def test_the_local_helper_matches_the_conftest_factory(
        self, decision_request: Callable[..., DecisionRequest]
    ) -> None:
        from_fixture = decision_request(PROOF, advisory_topics=("kw_health_regulation",), request_id="mock-test")
        from_helper = make_request(PROOF, advisory_topics=("kw_health_regulation",), request_id="mock-test")
        # Two suites building DecisionRequest differently would test two different
        # backends. If this fails, one of the two helpers has drifted from production.
        assert from_fixture == from_helper

    def test_the_mock_answers_a_conftest_built_request(self, mock: MockBackend, decision_request: Any) -> None:
        answers = mock.decide_sync(decision_request(CONFIDENTIAL))
        assert answers.sensitivity.choice == "confidential"
        assert answers.pii.value > DEFAULT_NOUL_THRESHOLD
