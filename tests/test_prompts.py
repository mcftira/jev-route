"""Tests for :mod:`jev_route.prompts` -- excerpting, redaction, and features.

This module sits on the trust boundary. Everything a remote decision backend is
allowed to see is produced here, so the tests are written as privacy assertions
first and correctness assertions second:

* **Redaction must actually remove the secret.** Not "replace most of it" -- the
  card number must be absent from the returned string. That is checked literally.
* **Features must not reproduce the text.** ``RequestFeatures`` is what gets
  logged when an operator has chosen ``excerpt_mode: hash``, so no field may
  contain the prompt, a substring of it, or anything that reverses to it.
* **Topic words must survive redaction.** Redacting "HIPAA" out of "explain how
  HIPAA works" hands a classifier an empty prompt and an operator a router that
  cannot read the question it was asked.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

import pytest

from jev_route.gate import HardGate
from jev_route.prompts import (
    MAX_EXCERPT_CHARS,
    ROUTING_RELEVANT_ROLES,
    build_backend_state,
    compute_features,
    detect_language,
    excerpt_from_text,
    excerpt_messages,
    flatten_content,
    hash_text,
    redact,
)
from jev_route.schema import RequestFeatures

CARD = "4111 1111 1111 1111"
EMAIL = "dana.kovacs@northside-health.org"
KEY = "sk-demo-D.OTTED.abcdefghijklmnopqrstuvwx"


@pytest.fixture
def gate() -> HardGate:
    return HardGate()


def user(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def assistant(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


# --------------------------------------------------------------------------- #
# flatten_content
# --------------------------------------------------------------------------- #
class TestFlattenContent:
    def test_string_passes_through_unchanged(self) -> None:
        assert flatten_content("hello world") == "hello world"

    def test_none_becomes_empty_string(self) -> None:
        # A tool call with no content is legal on the wire; it is not an error.
        assert flatten_content(None) == ""

    def test_empty_string_stays_empty(self) -> None:
        assert flatten_content("") == ""

    def test_list_of_text_and_image_parts(self) -> None:
        content = [
            {"type": "text", "text": "what does this diagram show?"},
            {"type": "image_url", "image_url": {"url": "https://cdn.example/a.png"}},
            {"type": "text", "text": "and this table?"},
        ]
        out = flatten_content(content)
        assert out == "what does this diagram show?\n[image_url]\nand this table?"

    def test_non_text_parts_collapse_to_a_marker_not_their_payload(self) -> None:
        """Routing on an image is out of scope, and the marker says so in the log."""
        out = flatten_content([{"type": "input_audio", "data": "AAAA"}])
        assert out == "[input_audio]"
        assert "AAAA" not in out

    def test_bare_dict_text_part(self) -> None:
        assert flatten_content({"type": "text", "text": "inline"}) == "inline"

    def test_bare_dict_other_type(self) -> None:
        assert flatten_content({"type": "file", "file": {"id": "f_1"}}) == "[file]"

    def test_dict_without_type(self) -> None:
        assert flatten_content({"text": "x"}) == "[non-text]"

    def test_empty_list(self) -> None:
        assert flatten_content([]) == ""

    def test_parts_with_no_text_are_dropped_not_joined_as_blanks(self) -> None:
        assert flatten_content([{"type": "image_url"}, {"type": "text", "text": "real"}]) == "[image_url]\nreal"

    def test_tuple_of_parts_behaves_like_a_list(self) -> None:
        assert flatten_content(({"type": "text", "text": "a"}, {"type": "text", "text": "b"})) == "a\nb"

    def test_scalar_falls_back_to_str(self) -> None:
        assert flatten_content(12345) == "12345"

    def test_text_part_with_non_string_text_is_not_leaked(self) -> None:
        assert flatten_content({"type": "text", "text": {"nested": 1}}) == "[text]"


# --------------------------------------------------------------------------- #
# excerpt_messages
# --------------------------------------------------------------------------- #
CONVERSATION: list[dict[str, Any]] = [
    {"role": "system", "content": "You are a helpful assistant."},
    user("U1 " + "a" * 30),
    assistant("A1 " + "b" * 30),
    user("U2 " + "c" * 30),
    assistant("A2 " + "d" * 30),
    {"role": "tool", "content": "TOOL_OUTPUT " + "t" * 200},
    {"role": "function", "content": "FUNCTION_OUTPUT"},
    user("U3 " + "e" * 30),
]


def turns(excerpt: str) -> list[str]:
    return [chunk.split(" ", 1)[0] for chunk in excerpt.split("\n\n")]


class TestExcerptMessages:
    def test_empty_and_none_inputs(self) -> None:
        assert excerpt_messages([]) == ""
        assert excerpt_messages(None) == ""

    def test_zero_budget_yields_nothing(self) -> None:
        assert excerpt_messages(CONVERSATION, max_chars=0) == ""
        assert excerpt_messages(CONVERSATION, max_chars=-5) == ""

    def test_result_never_exceeds_max_chars(self) -> None:
        for budget in (1, 5, 25, 60, 200, MAX_EXCERPT_CHARS):
            assert len(excerpt_messages(CONVERSATION, max_chars=budget)) <= budget

    def test_newest_user_turn_wins_the_budget(self) -> None:
        """With a budget of one turn, the turn you get is the most recent one."""
        excerpt = excerpt_messages(CONVERSATION, max_chars=25)
        assert excerpt.startswith("U3 ")
        assert "U1" not in excerpt and "U2" not in excerpt

    def test_older_turns_are_prepended_in_order(self) -> None:
        excerpt = excerpt_messages(CONVERSATION, include_prior_turns=5)
        assert turns(excerpt) == ["U1", "A1", "U2", "A2", "U3"]

    @pytest.mark.parametrize(
        ("include_prior_turns", "expected"),
        [
            (0, ["A2", "U3"]),
            (1, ["A1", "U2", "A2", "U3"]),
            (2, ["U1", "A1", "U2", "A2", "U3"]),
            (5, ["U1", "A1", "U2", "A2", "U3"]),
        ],
    )
    def test_include_prior_turns_counts_user_turns_not_messages(
        self, include_prior_turns: int, expected: list[str]
    ) -> None:
        """The knob counts *user* turns: an assistant reply is not a turn.

        Note the documented consequence at 0 and 1: the assistant message that
        precedes the newest included user turn still comes along, because the walk
        is newest-first and only stops when it hits one user turn too many.
        """
        assert turns(excerpt_messages(CONVERSATION, include_prior_turns=include_prior_turns)) == expected

    def test_tool_and_function_roles_are_skipped(self) -> None:
        excerpt = excerpt_messages(CONVERSATION, include_prior_turns=10)
        assert "TOOL_OUTPUT" not in excerpt
        assert "FUNCTION_OUTPUT" not in excerpt

    def test_tool_output_cannot_drown_out_the_request(self) -> None:
        """A 200-char tool dump must not consume the budget of a 30-char question."""
        messages = [{"role": "tool", "content": "t" * 5000}, user("what does this error mean?")]
        assert excerpt_messages(messages, max_chars=100) == "what does this error mean?"

    def test_system_excluded_by_default_and_included_on_request(self) -> None:
        messages = [{"role": "system", "content": "SYS"}, user("U")]
        assert excerpt_messages(messages) == "U"
        assert excerpt_messages(messages, include_system=True) == "SYS\n\nU"

    def test_routing_relevant_roles_constant(self) -> None:
        assert ROUTING_RELEVANT_ROLES == ("user", "assistant")

    def test_content_parts_are_flattened(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": "x"}},
                ],
            }
        ]
        assert excerpt_messages(messages) == "describe this\n[image_url]"

    def test_non_mapping_entries_are_ignored(self) -> None:
        assert excerpt_messages(["junk", None, user("real")]) == "real"

    def test_blank_messages_are_skipped(self) -> None:
        messages = [user("   "), user(""), user("U3 real")]
        assert excerpt_messages(messages) == "U3 real"

    def test_whitespace_is_stripped_from_each_turn(self) -> None:
        assert excerpt_messages([user("  padded  ")]) == "padded"

    def test_default_ceiling_is_the_module_constant(self) -> None:
        long_text = "x" * (MAX_EXCERPT_CHARS + 5000)
        assert len(excerpt_messages([user(long_text)])) == MAX_EXCERPT_CHARS
        assert MAX_EXCERPT_CHARS == 4000


class TestExcerptFromText:
    def test_strips_and_bounds(self) -> None:
        assert excerpt_from_text("   hello   ") == "hello"
        assert excerpt_from_text("x" * 5000, max_chars=10) == "x" * 10
        assert excerpt_from_text("") == ""
        assert excerpt_from_text(None or "") == ""

    def test_agrees_with_single_message_excerpt(self) -> None:
        """route_text and route_messages must see the same bytes for the same input."""
        text = "Summarize the incident report from last night."
        assert excerpt_from_text(text) == excerpt_messages([user(text)])


# --------------------------------------------------------------------------- #
# redact
# --------------------------------------------------------------------------- #
class TestRedact:
    def test_card_number_is_replaced_and_gone(self, gate: HardGate) -> None:
        out, count = redact(f"charge {CARD} please", gate)
        assert out == "charge [payment_card+phone_number] please"
        assert count == 1
        # The promise, stated literally: the secret is not in the output.
        assert CARD not in out
        assert CARD.replace(" ", "") not in out
        assert "4111" not in out

    def test_placeholder_names_the_detector(self, gate: HardGate) -> None:
        out, _ = redact(f"mail {EMAIL}", gate)
        assert out == "mail [email_address]"
        assert EMAIL not in out
        assert "dana.kovacs" not in out
        assert "northside-health.org" not in out

    def test_custom_placeholder_template(self, gate: HardGate) -> None:
        out, _ = redact(f"mail {EMAIL}", gate, placeholder="<{name}>")
        assert out == "mail <email_address>"

    def test_count_is_one_per_redacted_span(self, gate: HardGate) -> None:
        out, count = redact(f"cards {CARD} and 5555 5555 5555 4444 plus {EMAIL}", gate)
        assert count == 3
        assert out.count("[payment_card") == 2
        assert out.count("[email_address]") == 1

    def test_overlapping_spans_are_collapsed_into_one_placeholder(self, gate: HardGate) -> None:
        """A basic-auth URL also matches the email rule; emit one placeholder, not two."""
        out, count = redact("postgres://router:hunter2@db.internal.northside.example", gate)
        assert count == 1
        assert out == "[basic_auth_url+email_address]"
        assert "hunter2" not in out

    def test_nested_spans_do_not_double_redact(self, gate: HardGate) -> None:
        out, count = redact(CARD, gate)
        assert count == 1
        assert "1111" not in out

    def test_clean_text_is_returned_unchanged(self, gate: HardGate) -> None:
        text = "Summarize the three main arguments of this cycling essay."
        assert redact(text, gate) == (text, 0)

    def test_empty_text(self, gate: HardGate) -> None:
        assert redact("", gate) == ("", 0)

    def test_api_key_is_gone_after_redaction(self, gate: HardGate) -> None:
        out, count = redact(f"export the key {KEY} now", gate)
        assert count == 1
        assert KEY not in out
        assert "IPIPE" not in out

    def test_advisory_topic_words_survive_redaction(self, gate: HardGate) -> None:
        """Redacting the topic would leave the classifier with nothing to read."""
        text = "Explain how HIPAA actually works and who it applies to."
        out, count = redact(text, gate)
        assert out == text
        assert count == 0
        assert "HIPAA" in out

    def test_topic_word_survives_next_to_a_real_identifier(self, gate: HardGate) -> None:
        text = f"Under HIPAA, forward the record to {EMAIL}."
        out, count = redact(text, gate)
        assert "HIPAA" in out
        assert EMAIL not in out
        assert count == 1

    def test_redaction_is_idempotent_on_the_output(self, gate: HardGate) -> None:
        once, _ = redact(f"charge {CARD} and mail {EMAIL}", gate)
        twice, count = redact(once, gate)
        assert twice == once
        assert count == 0

    def test_offsets_stay_valid_with_many_spans(self, gate: HardGate) -> None:
        """Spans are applied right-to-left; a bug here shows up as scrambled text."""
        text = f"{EMAIL} and {CARD} and {EMAIL} again"
        out, count = redact(text, gate)
        assert count == 3
        assert out == "[email_address] and [payment_card+phone_number] and [email_address] again"

    def test_default_gate_is_used_when_none_passed(self) -> None:
        out, count = redact(f"charge {CARD}")
        assert count == 1
        assert CARD not in out

    def test_disabled_detector_is_not_redacted(self) -> None:
        quiet = HardGate(disabled_detectors=["email_address"])
        out, count = redact(f"mail {EMAIL}", quiet)
        assert count == 0
        assert out == f"mail {EMAIL}"


# --------------------------------------------------------------------------- #
# hash_text
# --------------------------------------------------------------------------- #
class TestHashText:
    def test_stable_across_calls(self) -> None:
        assert hash_text("same input") == hash_text("same input")

    def test_length_is_16_hex_chars(self) -> None:
        digest = hash_text("anything")
        assert len(digest) == 16
        assert re.fullmatch(r"[0-9a-f]{16}", digest)

    def test_salt_changes_the_hash(self) -> None:
        assert hash_text("same input") != hash_text("same input", salt="deployment-a")
        assert hash_text("x", salt="a") != hash_text("x", salt="b")

    def test_empty_salt_is_the_default(self) -> None:
        assert hash_text("x") == hash_text("x", salt="")

    def test_different_text_different_hash(self) -> None:
        assert hash_text("prompt one") != hash_text("prompt two")

    def test_hash_does_not_leak_the_text(self) -> None:
        digest = hash_text(CARD)
        assert CARD not in digest
        assert "4111" not in digest

    def test_separator_prevents_boundary_ambiguity(self) -> None:
        """salt="a"+text="bc" must differ from salt="ab"+text="c"."""
        assert hash_text("bc", salt="a") != hash_text("c", salt="ab")

    def test_unicode_is_handled(self) -> None:
        assert len(hash_text("árvíztűrő tükörfúrógép")) == 16


# --------------------------------------------------------------------------- #
# detect_language
# --------------------------------------------------------------------------- #
class TestDetectLanguage:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("The quick brown fox jumps over the lazy dog and the cat", "en"),
            ("Der Hund ist nicht schnell und die Katze ist müde", "de"),
            ("A gyors barna róka átugorja a lusta kutyát és a macskát", "hu"),
            ("Le chat est sur la table et la chaise est dans la salle", "fr"),
            ("El gato está en la mesa y el perro no está en la casa", "es"),
        ],
    )
    def test_detects_common_languages(self, text: str, expected: str) -> None:
        assert detect_language(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "hi",  # fewer than four words: not enough signal to guess
            "xyzzy plugh qwerty uiop",  # no stopwords at all
            "",
            "1234 5678 9012 3456",  # digits are not words
        ],
    )
    def test_returns_und_when_unsure(self, text: str) -> None:
        assert detect_language(text) == "und"

    def test_is_case_insensitive(self) -> None:
        assert detect_language("THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG") == "en"

    def test_never_returns_a_language_outside_the_table(self) -> None:
        for text in ("hello there friend", "bonjour mon ami", "!! ?? .."):
            assert detect_language(text) in {"en", "de", "fr", "es", "it", "pt", "hu", "nl", "und"}


# --------------------------------------------------------------------------- #
# compute_features
# --------------------------------------------------------------------------- #
class TestComputeFeatures:
    def test_empty_text_is_all_zeroes(self) -> None:
        f = compute_features("")
        assert f.char_len == 0
        assert f.word_count == 0
        assert f.line_count == 0
        assert f.sentence_count == 0
        assert f.mean_word_len == 0.0
        assert f.digit_ratio == 0.0
        assert f.lang == "und"
        assert f.n_messages == 0

    def test_digit_ratio(self) -> None:
        assert compute_features("abc123").digit_ratio == 0.5
        assert compute_features("no digits here").digit_ratio == 0.0
        assert compute_features("9999").digit_ratio == 1.0

    def test_upper_ratio_is_measured_over_letters_only(self) -> None:
        assert compute_features("aA").upper_ratio == 0.5
        assert compute_features("SHOUTING").upper_ratio == 1.0
        assert compute_features("1234").upper_ratio == 0.0

    def test_punct_and_non_ascii_ratios(self) -> None:
        assert compute_features("a!").punct_ratio == 0.5
        assert compute_features("árvíztűrő tükörfúrógép").non_ascii_ratio > 0.0
        assert compute_features("plain ascii text").non_ascii_ratio == 0.0

    def test_code_blocks_count_pairs_of_fences(self) -> None:
        assert compute_features("```python\nx = 1\n```\n```js\ny = 2\n```").code_blocks == 2
        assert compute_features("a single ``` fence").code_blocks == 0
        assert compute_features("no code at all").code_blocks == 0

    def test_inline_code_spans(self) -> None:
        assert compute_features("call `foo()` then `bar()`").inline_code_spans == 2
        assert compute_features("no backticks").inline_code_spans == 0

    def test_urls(self) -> None:
        assert compute_features("see https://a.example/x and http://b.example").urls == 2
        assert compute_features("no links here").urls == 0

    def test_question_marks_and_exclamations(self) -> None:
        f = compute_features("really? yes! and again??")
        assert f.question_marks == 3
        assert f.exclamations == 1

    def test_line_and_sentence_counts(self) -> None:
        f = compute_features("First sentence. Second one! Third? Fourth")
        assert f.line_count == 1
        assert f.sentence_count == 4
        assert compute_features("a\nb\nc").line_count == 3

    def test_mean_word_len(self) -> None:
        assert compute_features("aa bbbb").mean_word_len == 3.0

    @pytest.mark.parametrize(
        "trace",
        [
            'Traceback (most recent call last):\n  File "x.py", line 3, in <module>',
            'Exception in thread "main" java.lang.NullPointerException\n\tat com.example.Foo.bar(Foo.java:42)',
            "#0  0x00007fff5fbff8a0 in foo ()",
        ],
    )
    def test_has_stack_trace(self, trace: str) -> None:
        assert compute_features(trace).has_stack_trace is True

    def test_has_stack_trace_false_for_prose(self) -> None:
        assert compute_features("The stack trace was long and confusing.").has_stack_trace is False

    def test_has_stack_trace_detects_nodejs_frames(self) -> None:
        node = (
            "TypeError: Cannot read properties of undefined (reading 'id')\n"
            "    at Object.<anonymous> (/app/src/index.js:12:34)\n"
            "    at Module._compile (node:internal/modules/cjs/loader:1105:14)"
        )
        assert compute_features(node).has_stack_trace is True

    def test_has_diff(self) -> None:
        diff = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old line\n+new line\n"
        assert compute_features(diff).has_diff is True
        assert compute_features("nothing about diffs here").has_diff is False

    @pytest.mark.parametrize("text", ['{"a": 1}', "[1, 2, 3]", '{\n  "nested": true\n}'])
    def test_has_json(self, text: str) -> None:
        assert compute_features(text).has_json is True

    @pytest.mark.parametrize("text", ["It is {not} json.", "explain this {code} please"])
    def test_has_json_false_for_prose_with_braces(self, text: str) -> None:
        assert compute_features(text).has_json is False

    def test_message_derived_fields(self) -> None:
        messages = [
            {"role": "system", "content": "s"},
            user("u1"),
            assistant("a"),
            {"role": "tool", "content": "t"},
            user("u2"),
        ]
        f = compute_features("u2", messages=messages)
        assert f.n_messages == 5
        assert f.n_prior_turns == 1  # two user turns minus the current one
        assert f.tool_output_present is True

    def test_no_messages_means_one_message(self) -> None:
        assert compute_features("text only").n_messages == 1
        assert compute_features("text only").n_prior_turns == 0
        assert compute_features("text only").tool_output_present is False

    def test_function_role_counts_as_tool_output(self) -> None:
        assert compute_features("x", messages=[{"role": "function", "content": "y"}]).tool_output_present is True

    def test_gate_fields_are_passed_through(self) -> None:
        f = compute_features("x", gate_detectors={"payment_card": 2}, n_gate_findings=1, gate_force_local=True)
        assert f.gate_detectors == {"payment_card": 2}
        assert f.n_gate_findings == 1
        assert f.gate_force_local is True
        assert f.to_dict()["gate_detectors"] == {"payment_card": 2}

    def test_gate_detectors_default_to_an_empty_plain_dict(self) -> None:
        assert compute_features("x").gate_detectors == {}

    def test_deterministic(self) -> None:
        text = f"Fix this bug in {CARD}: ```python\nx=1\n```"
        assert compute_features(text).to_dict() == compute_features(text).to_dict()

    def test_no_field_reproduces_the_input_text(self) -> None:
        """The privacy contract of the feature vector, asserted field by field.

        ``excerpt_mode: hash`` means these features are all an operator keeps. If
        any field carried the prompt, that setting would be a lie.
        """
        secret = f"Patient {CARD} belongs to {EMAIL} with key {KEY}"
        features = compute_features(
            secret,
            gate_detectors={"payment_card": 1},
            n_gate_findings=1,
            gate_force_local=True,
        )
        forbidden = {secret, CARD, CARD.replace(" ", ""), EMAIL, KEY, "dana.kovacs", "northside-health.org", "IPIPE"}
        for field in dataclasses.fields(RequestFeatures):
            value = getattr(features, field.name)
            rendered = value if isinstance(value, str) else repr(value)
            for needle in forbidden:
                assert needle not in rendered, f"{field.name} leaked {needle!r}"
            assert rendered != secret

    def test_every_ratio_is_between_zero_and_one(self) -> None:
        f = compute_features("MiXeD 123 !? árvíztűrő https://x.example `code`")
        for name in ("digit_ratio", "upper_ratio", "punct_ratio", "non_ascii_ratio"):
            value = getattr(f, name)
            assert 0.0 <= value <= 1.0, name


# --------------------------------------------------------------------------- #
# build_backend_state
# --------------------------------------------------------------------------- #
class TestBuildBackendState:
    def test_shape(self) -> None:
        features = compute_features("hello there")
        state = build_backend_state("redacted text", features=features, metadata={"tenant": "acme"})
        assert set(state) == {"prompt_excerpt", "request_features", "caller_metadata"}
        assert state["prompt_excerpt"] == "redacted text"
        assert state["request_features"] == features.to_dict()
        assert state["caller_metadata"] == {"tenant": "acme"}

    @pytest.mark.parametrize("unsafe", ["messages", "prompt", "content", "input", "raw", "body"])
    def test_unsafe_metadata_keys_are_stripped(self, unsafe: str) -> None:
        """Callers stuff request bodies into metadata; that must not reach a backend."""
        state = build_backend_state(
            "redacted",
            features=compute_features("redacted"),
            metadata={unsafe: "RAW PROMPT TEXT", "tenant": "acme"},
        )
        assert unsafe not in state["caller_metadata"]
        assert state["caller_metadata"] == {"tenant": "acme"}
        assert "RAW PROMPT TEXT" not in repr(state)

    def test_all_unsafe_keys_stripped_at_once(self) -> None:
        metadata = {k: f"value-{k}" for k in ("messages", "prompt", "content", "input", "raw", "body")}
        metadata["keep"] = 1
        state = build_backend_state("r", features=compute_features("r"), metadata=metadata)
        assert state["caller_metadata"] == {"keep": 1}

    def test_none_metadata_becomes_an_empty_dict(self) -> None:
        state = build_backend_state("r", features=compute_features("r"))
        assert state["caller_metadata"] == {}

    def test_caller_metadata_is_copied_not_aliased(self) -> None:
        metadata = {"tenant": "acme"}
        state = build_backend_state("r", features=compute_features("r"), metadata=metadata)
        state["caller_metadata"]["tenant"] = "changed"
        assert metadata == {"tenant": "acme"}

    def test_features_block_contains_no_text(self) -> None:
        state = build_backend_state("redacted only", features=compute_features(f"raw {CARD} raw"), metadata=None)
        assert CARD not in repr(state["request_features"])
