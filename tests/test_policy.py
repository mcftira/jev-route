"""Tests for :mod:`jev_route.policy` -- the operator's surface and a security boundary.

Two very different things live in this module and the test file mirrors that split:

``compile_expression`` is a **sandbox**. A policy YAML is config an operator loads
from wherever they keep config, and its ``if:`` strings become executable code.
The whitelist is the only thing between a policy file and ``os.system``, so the
tests below assert rejection of every escape that comes to mind -- calls,
attributes, lambdas, comprehensions, imports, builtins, nested subscripts -- and
do it by *compiling*, not by evaluating, because the promise is "raises at load
time, naming the offending node".

``Policy`` is a **contract with the operator**. The tier table it resolves is the
product: data protection before cost, first match wins, gate before complexity.
Those rows are asserted against the shipped ``policies/default.yaml`` rather than
a synthetic document, so a change to the reference policy that weakens the
ordering fails here.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from jev_route.policy import (
    EXPRESSION_VARIABLES,
    POLICY_VERSION,
    Expression,
    FailurePolicy,
    Policy,
    PolicyError,
    Rule,
    UncertaintyPolicy,
    compile_expression,
)
from jev_route.schema import COMPLEXITY_LEVELS, SENSITIVITY_LEVELS, TIERS

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = REPO_ROOT / "policies" / "default.yaml"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def namespace(**overrides: Any) -> dict[str, Any]:
    """A complete rule namespace: every name in EXPRESSION_VARIABLES, benign values."""
    base: dict[str, Any] = {
        "complexity": "standard",
        "sensitivity": "public",
        "domain": "chat",
        "complexity_confidence": 0.95,
        "sensitivity_confidence": 0.95,
        "domain_confidence": 0.95,
        "pii": 0.02,
        "pii_present": False,
        "gate_force_local": False,
        "gate_blocks_backend": False,
        "gate_sensitivity_floor": None,
        "gate_detectors": set(),
        "advisory_topics": set(),
        # Layer 2's actionable projection. Benign values are the shadow-shaped
        # ones: a rule must not be able to act on a layer that is not enforcing.
        "semantic_fired": False,
        "semantic_score": 0.0,
        "semantic_force_local": False,
        "semantic_mode": "shadow",
        "degraded": False,
        "cached": False,
        "tier": "",
        "model": "",
        "char_len": 120,
        "word_count": 20,
        "line_count": 3,
        "code_blocks": 0,
        "question_marks": 1,
        "has_stack_trace": False,
        "lang": "en",
        "n_prior_turns": 0,
        "requested_model": "",
        "metadata": {"tenant": "acme"},
    }
    base.update(overrides)
    return base


def minimal_doc(**overrides: Any) -> dict[str, Any]:
    """The smallest document that loads. ``strong`` must exist even when unused,
    because ``FailurePolicy`` validates both of its tiers at load time."""
    doc: dict[str, Any] = {
        "version": 1,
        "tiers": {"local": ["m-local"], "cheap": ["m-cheap"], "strong": ["m-strong"]},
        "tier_order": ["cheap", "strong", "local"],
        "rules": [
            {"id": "gate", "if": "gate_force_local", "then": {"tier": "local"}},
            {"id": "default", "then": {"tier": "cheap"}},
        ],
    }
    doc.update(copy.deepcopy(overrides))
    return doc


@pytest.fixture(scope="module")
def shipped() -> Policy:
    """The reference policy. Module-scoped: it is immutable and loading is not free."""
    return Policy.from_file(DEFAULT_POLICY)


# --------------------------------------------------------------------------- #
# The sandbox
# --------------------------------------------------------------------------- #
class TestSafeAstRejects:
    """Every one of these must raise PolicyError **at compile time**."""

    @pytest.mark.parametrize(
        ("source", "why"),
        [
            ("__import__('os').system('x')", "import + call"),
            ("open('/etc/passwd').read()", "file access via a call"),
            ('eval("1")', "eval"),
            ('exec("x=1")', "exec"),
            ("len(metadata) > 2", "any call at all"),
            ("print('hi')", "a builtin name used as a callable"),
            ("complexity.upper() == 'HARD'", "method call on a value"),
            ("lambda: 1", "lambda"),
            ("[x for x in gate_detectors]", "list comprehension"),
            ("{x for x in advisory_topics}", "set comprehension"),
            ("{k: v for k, v in metadata}", "dict comprehension"),
            ("(x for x in gate_detectors)", "generator expression"),
            ("(1).__class__", "attribute access -- the classic sandbox escape"),
            ("metadata.__class__.__mro__", "attribute chain"),
            ("complexity.__class__.__bases__[0].__subclasses__()", "subclass walk"),
            ("a.b.c", "dotted names"),
            ("undefined_variable == 1", "a name outside the whitelist"),
            ("print", "a bare builtin name"),
            ("os", "a bare module name"),
            ("self", "an implicit name"),
            ("1 if pii else 2", "conditional expression"),
            ("{**metadata}", "dict literal / splat"),
            ("{'a': 1}", "dict literal"),
            ("complexity == 'hard' or (yield 1)", "yield"),
            ("await complexity", "await"),
            ("f'{metadata}'", "f-string with a replacement field"),
            ("char_len // 2 > 1", "floor division is outside + - *"),
            ("char_len % 2 == 0", "modulo is outside + - *"),
            ("char_len ** 2 > 4", "exponentiation is outside + - *"),
            ("complexity == 'hard'; pii_present", "statement chaining"),
            ("import os", "import statement"),
            ("metadata['a'] = 1", "assignment"),
            ("", "empty expression"),
            ("   ", "whitespace-only expression"),
            ("complexity ==", "syntax error"),
        ],
    )
    def test_rejected_at_compile_time(self, source: str, why: str) -> None:
        with pytest.raises(PolicyError):
            compile_expression(source)

    def test_error_names_the_offending_node(self) -> None:
        """The message has to be actionable: an operator sees it at startup."""
        with pytest.raises(PolicyError, match="disallowed syntax 'Call'"):
            compile_expression("len(metadata) > 2")

    def test_error_names_the_unknown_variable(self) -> None:
        with pytest.raises(PolicyError, match="unknown variable 'tenant'"):
            compile_expression("tenant == 'acme'")

    def test_error_lists_the_available_variables(self) -> None:
        with pytest.raises(PolicyError) as excinfo:
            compile_expression("tenant == 'acme'")
        for name in ("complexity", "gate_force_local", "advisory_topics"):
            assert name in str(excinfo.value)

    def test_nested_subscript_is_rejected(self) -> None:
        """``metadata['a']['b']`` would let a policy walk arbitrary object graphs."""
        with pytest.raises(PolicyError, match="only simple names may be subscripted"):
            compile_expression("metadata['a']['b'] == 1")

    def test_subscript_of_a_literal_is_rejected(self) -> None:
        with pytest.raises(PolicyError, match="only simple names may be subscripted"):
            compile_expression("[1, 2][0] == 1")

    def test_syntax_error_is_a_policy_error_not_a_syntax_error(self) -> None:
        with pytest.raises(PolicyError, match="not valid syntax"):
            compile_expression("complexity === 'hard'")

    def test_rejected_expression_is_not_cached(self) -> None:
        with pytest.raises(PolicyError):
            compile_expression("__import__('os')")
        # Compiling a *legal* expression afterwards must still work.
        assert compile_expression("gate_force_local").evaluate(namespace(gate_force_local=True)) is True

    def test_evaluation_errors_surface_as_policy_errors(self) -> None:
        """A missing key or a bad comparison is a policy bug, not a 500 at runtime."""
        expr = compile_expression("metadata['absent'] == 1")
        with pytest.raises(PolicyError, match="failed to evaluate"):
            expr.evaluate(namespace())

    def test_type_mismatch_surfaces_as_a_policy_error(self) -> None:
        with pytest.raises(PolicyError, match="failed to evaluate"):
            compile_expression("char_len > 'x'").evaluate(namespace())

    def test_missing_variable_surfaces_as_a_policy_error(self) -> None:
        expr = compile_expression("complexity == 'hard'")
        with pytest.raises(PolicyError, match="failed to evaluate"):
            expr.evaluate({})

    def test_builtins_are_not_available_during_evaluation(self) -> None:
        """``True``/``False``/``None`` are AST constants, not builtins lookups."""
        assert compile_expression("True").evaluate({}) is True
        assert compile_expression("None == None").evaluate({}) is True


class TestSafeAstAccepts:
    """The documented grammar. A rejection here is as much a bug as an acceptance above."""

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            # comparisons
            ("complexity == 'standard'", True),
            ("complexity == 'hard'", False),
            ("complexity == 'frontier'", False),
            ("sensitivity != 'public'", False),
            ("char_len < 200", True),
            ("char_len > 200", False),
            ("word_count <= 20", True),
            ("question_marks >= 1", True),
            # membership on a list, a set and a tuple
            ("sensitivity in ['confidential', 'regulated']", False),
            ("complexity in ('hard', 'frontier')", False),
            ("domain in {'chat', 'code'}", True),
            ("domain not in {'code'}", True),
            ("'payment_card' in gate_detectors", False),
            # boolean operators
            ("complexity == 'hard' and sensitivity == 'public'", False),
            ("complexity == 'hard' or sensitivity == 'public'", True),
            ("not gate_force_local", True),
            ("not (complexity == 'hard' and pii_present)", True),
            # literals
            ("[1, 2] == [1, 2]", True),
            ("(1, 2) == (1, 2)", True),
            ("{1, 2} == {2, 1}", True),
            # subscript of a simple name
            ("metadata['tenant'] == 'acme'", True),
            ("metadata['tenant'] != 'other'", True),
            # arithmetic
            ("char_len + word_count > 100", True),
            ("char_len - word_count > 0", True),
            ("word_count * 2 == 40", True),
            ("char_len + 1 - 1 == char_len", True),
            # unary minus
            ("-char_len < 0", True),
            ("-1 < 0", True),
            # is / is not
            ("gate_sensitivity_floor is None", True),
            ("complexity is not None", True),
            # chained comparison
            ("0 < pii < 1", True),
        ],
    )
    def test_allowed_grammar(self, source: str, expected: bool) -> None:
        assert compile_expression(source).evaluate(namespace()) is expected

    def test_set_membership_against_gate_detectors(self) -> None:
        ns = namespace(gate_detectors={"payment_card", "email_address"})
        assert compile_expression("'payment_card' in gate_detectors").evaluate(ns) is True
        assert compile_expression("'us_ssn' in gate_detectors").evaluate(ns) is False

    def test_set_membership_against_advisory_topics(self) -> None:
        ns = namespace(advisory_topics={"kw_health_regulation"})
        assert compile_expression("'kw_health_regulation' in advisory_topics").evaluate(ns) is True
        assert compile_expression("'kw_minors' in advisory_topics").evaluate(ns) is False

    def test_result_is_coerced_to_bool(self) -> None:
        assert compile_expression("char_len").evaluate(namespace()) is True
        assert compile_expression("0").evaluate(namespace()) is False

    def test_compile_is_memoized(self) -> None:
        """Policies are parsed once and evaluated per request; identity is the proof."""
        assert compile_expression("gate_force_local") is compile_expression("gate_force_local")

    def test_expression_keeps_its_source(self) -> None:
        expr = compile_expression("  complexity == 'hard'  ")
        assert expr.source == "complexity == 'hard'"

    def test_expression_is_directly_constructible_via_compile(self) -> None:
        assert isinstance(Expression.compile("pii_present"), Expression)

    def test_documented_variable_list_matches_the_namespace(self) -> None:
        """A policy author can only introspect the docs, so the docs must be the truth."""
        assert set(EXPRESSION_VARIABLES) == set(namespace())


# --------------------------------------------------------------------------- #
# The shipped default policy: the tier resolution table
# --------------------------------------------------------------------------- #
class TestDefaultPolicyResolution:
    """These rows ARE the product: data protection first, then capability, then cost."""

    @pytest.mark.parametrize(
        ("label", "overrides", "expected_tier", "expected_rule"),
        [
            ("trivial/public", {"complexity": "trivial", "sensitivity": "public"}, "cheap", "default"),
            ("standard/public", {"complexity": "standard", "sensitivity": "public"}, "cheap", "default"),
            ("hard/public", {"complexity": "hard", "sensitivity": "public"}, "strong", "complexity.hard"),
            ("frontier/public", {"complexity": "frontier", "sensitivity": "public"}, "strong", "complexity.frontier"),
            ("trivial/internal", {"complexity": "trivial", "sensitivity": "internal"}, "cheap", "default"),
            ("confidential", {"sensitivity": "confidential"}, "local", "data.sensitive"),
            ("regulated", {"sensitivity": "regulated"}, "local", "data.sensitive"),
            ("pii_present", {"pii": 0.9, "pii_present": True}, "local", "data.sensitive"),
            (
                "gate_force_local",
                {"gate_force_local": True, "gate_sensitivity_floor": "regulated"},
                "local",
                "gate.force-local",
            ),
            (
                "gate_force_local beats frontier",
                {"gate_force_local": True, "complexity": "frontier", "sensitivity": "public"},
                "local",
                "gate.force-local",
            ),
            (
                "gate blocks_backend beats frontier",
                {"gate_blocks_backend": True, "gate_force_local": True, "complexity": "frontier"},
                "local",
                "gate.force-local",
            ),
            (
                "frontier + confidential",
                {"complexity": "frontier", "sensitivity": "confidential"},
                "local",
                "data.sensitive",
            ),
            ("advisory topic alone", {"advisory_topics": {"kw_health_regulation"}}, "cheap", "default"),
        ],
    )
    def test_tier_table(
        self, shipped: Policy, label: str, overrides: dict[str, Any], expected_tier: str, expected_rule: str
    ) -> None:
        rule, tier, model = shipped.evaluate(namespace(**overrides))
        assert tier == expected_tier, label
        assert rule.rule_id == expected_rule, label
        assert model == shipped.pick_model(tier)

    def test_gate_force_local_wins_over_frontier_complexity(self, shipped: Policy) -> None:
        """Ordering is the whole game: a hard task on regulated data is still regulated."""
        rule_ids = [r.rule_id for r in shipped.rules]
        assert rule_ids.index("gate.force-local") < rule_ids.index("complexity.frontier")
        rule, tier, _ = shipped.evaluate(namespace(gate_force_local=True, complexity="frontier"))
        assert (rule.rule_id, tier) == ("gate.force-local", "local")

    def test_advisory_topic_does_not_route_local(self, shipped: Policy) -> None:
        """A HIPAA *question* contains no HIPAA data and must not be treated as if it did."""
        rule, tier, _ = shipped.evaluate(namespace(advisory_topics={"kw_health_regulation"}))
        assert tier == "cheap"
        assert rule.rule_id == "default"

    def test_first_match_wins(self, shipped: Policy) -> None:
        # Satisfies data.sensitive AND complexity.frontier; the earlier rule answers.
        rule, tier, _ = shipped.evaluate(namespace(sensitivity="regulated", complexity="frontier"))
        assert rule.rule_id == "data.sensitive"
        assert tier == "local"

    def test_rule_order_is_the_documented_order(self, shipped: Policy) -> None:
        assert [r.rule_id for r in shipped.rules] == [
            "gate.force-local",
            "data.sensitive",
            "complexity.frontier",
            "complexity.hard",
            "default",
        ]

    def test_exactly_one_default_rule_and_it_is_last(self, shipped: Policy) -> None:
        defaults = [r for r in shipped.rules if r.is_default]
        assert len(defaults) == 1
        assert shipped.rules[-1].is_default

    def test_tiers_and_models(self, shipped: Policy) -> None:
        assert set(shipped.tiers) == {"local", "cheap", "strong"}
        assert all(models for models in shipped.tiers.values())

    def test_failure_and_uncertainty_defaults(self, shipped: Policy) -> None:
        assert shipped.failure == FailurePolicy(mode="fail_closed", fail_closed_tier="local", fail_open_tier="strong")
        assert shipped.uncertainty.sensitivity_confidence_below == 0.8
        assert shipped.uncertainty.complexity_confidence_below == 0.7
        assert shipped.uncertainty.pii_uncertain_threshold == 0.35
        assert shipped.uncertainty.pii_uncertain_counts_as_present is True
        assert shipped.pii_threshold == 0.5

    def test_backend_defaults_to_mock_so_the_policy_runs_with_no_key(self, shipped: Policy) -> None:
        assert shipped.backend["name"] == "mock"

    def test_version_and_source(self, shipped: Policy) -> None:
        assert shipped.version == POLICY_VERSION
        assert shipped.source is not None and shipped.source.endswith("default.yaml")


# --------------------------------------------------------------------------- #
# Loading and validation
# --------------------------------------------------------------------------- #
class TestValidation:
    @pytest.mark.parametrize(
        ("label", "overrides"),
        [
            ("no rules", {"rules": []}),
            ("rules key absent", {"rules": None}),
            (
                "default rule not last",
                {
                    "rules": [
                        {"id": "default", "then": {"tier": "cheap"}},
                        {"id": "g", "if": "gate_force_local", "then": {"tier": "local"}},
                    ]
                },
            ),
            (
                "two default rules",
                {"rules": [{"id": "a", "then": {"tier": "cheap"}}, {"id": "b", "then": {"tier": "local"}}]},
            ),
            ("no default rule", {"rules": [{"id": "g", "if": "gate_force_local", "then": {"tier": "local"}}]}),
            (
                "rule tier not defined",
                {
                    "rules": [
                        {"id": "g", "if": "gate_force_local", "then": {"tier": "gpu"}},
                        {"id": "d", "then": {"tier": "cheap"}},
                    ]
                },
            ),
            ("empty tier model list", {"tiers": {"local": [], "cheap": ["a"], "strong": ["b"]}}),
            ("tier_order names an undefined tier", {"tier_order": ["cheap", "turbo"]}),
            ("duplicate tier_order entries", {"tier_order": ["cheap", "cheap"]}),
            (
                "bad on_backend_down.mode",
                {"on_backend_down": {"mode": "yolo", "fail_closed_tier": "local", "fail_open_tier": "strong"}},
            ),
            (
                "fail_closed_tier undefined",
                {"on_backend_down": {"mode": "fail_closed", "fail_closed_tier": "gpu", "fail_open_tier": "strong"}},
            ),
            (
                "fail_open_tier undefined",
                {"on_backend_down": {"mode": "fail_closed", "fail_closed_tier": "local", "fail_open_tier": "gpu"}},
            ),
            ("unsupported version", {"version": 99}),
            ("no tiers at all", {"tiers": {}}),
            ("tiers not a mapping", {"tiers": ["local", "cheap"]}),
            ("rule is not a mapping", {"rules": ["cheap", {"id": "d", "then": {"tier": "cheap"}}]}),
            (
                "then sets neither tier nor model",
                {"rules": [{"id": "g", "if": "gate_force_local", "then": {}}, {"id": "d", "then": {"tier": "cheap"}}]},
            ),
            (
                "then.model is empty",
                {
                    "rules": [
                        {"id": "g", "if": "gate_force_local", "then": {"tier": "local", "model": "  "}},
                        {"id": "d", "then": {"tier": "cheap"}},
                    ]
                },
            ),
            (
                "rule expression is illegal",
                {
                    "rules": [
                        {"id": "g", "if": "__import__('os')", "then": {"tier": "local"}},
                        {"id": "d", "then": {"tier": "cheap"}},
                    ]
                },
            ),
            ("tier name is blank", {"tiers": {"  ": ["m"], "cheap": ["c"], "strong": ["s"]}}),
            ("tier models are not strings or a list", {"tiers": {"local": 5, "cheap": ["c"], "strong": ["s"]}}),
        ],
    )
    def test_invalid_documents_raise_policy_error(self, label: str, overrides: dict[str, Any]) -> None:
        with pytest.raises(PolicyError):
            Policy.from_dict(minimal_doc(**overrides))

    @pytest.mark.parametrize(
        ("label", "text"),
        [
            ("unparseable yaml", "tiers: [a\n  bad: : :"),
            ("top level is a list", "- a\n- b\n"),
            ("top level is a scalar", "just a string\n"),
            ("empty document", ""),
        ],
    )
    def test_invalid_yaml_raises_policy_error(self, label: str, text: str) -> None:
        with pytest.raises(PolicyError):
            Policy.from_yaml(text)

    def test_yaml_error_is_wrapped_not_leaked(self) -> None:
        with pytest.raises(PolicyError, match="policy YAML failed to parse"):
            Policy.from_yaml("tiers: [a\n  bad: : :")
        with pytest.raises(PolicyError, match="must be a mapping at the top level"):
            Policy.from_yaml("- a\n")

    def test_missing_file_raises_policy_error(self, tmp_path: Path) -> None:
        with pytest.raises(PolicyError, match="policy file not found"):
            Policy.from_file(tmp_path / "nope.yaml")

    def test_from_file_records_its_source(self, tmp_path: Path) -> None:
        path = tmp_path / "p.yaml"
        path.write_text(yaml.safe_dump(minimal_doc()), encoding="utf-8")
        assert Policy.from_file(path).source == str(path)

    def test_valid_minimal_document_loads(self) -> None:
        policy = Policy.from_dict(minimal_doc())
        assert policy.version == POLICY_VERSION
        assert len(policy.rules) == 2

    def test_version_defaults_when_absent(self) -> None:
        doc = minimal_doc()
        del doc["version"]
        assert Policy.from_dict(doc).version == POLICY_VERSION

    def test_tier_order_defaults_to_the_schema_order(self) -> None:
        doc = minimal_doc()
        del doc["tier_order"]
        assert Policy.from_dict(doc).tier_order == TIERS

    def test_string_tier_value_is_accepted_as_a_single_model(self) -> None:
        doc = minimal_doc(tiers={"local": "one-model", "cheap": "two", "strong": "three"})
        assert Policy.from_dict(doc).tiers["local"] == ("one-model",)

    def test_then_as_a_bare_string_is_a_tier(self) -> None:
        doc = minimal_doc(rules=[{"id": "g", "if": "gate_force_local", "then": "local"}, {"id": "d", "then": "cheap"}])
        policy = Policy.from_dict(doc)
        assert policy.rules[0].tier == "local"
        assert policy.rules[0].model is None

    def test_on_backend_down_as_a_bare_string_is_a_mode(self) -> None:
        """The string shorthand the loader explicitly codes for must work -- or must raise PolicyError."""
        policy = Policy.from_dict(minimal_doc(on_backend_down="fail_open"))
        assert policy.failure.mode == "fail_open"

    def test_on_backend_down_as_a_mapping_works(self) -> None:
        policy = Policy.from_dict(minimal_doc(on_backend_down={"mode": "fail_open"}))
        assert policy.failure.mode == "fail_open"
        assert policy.failure.fail_closed_tier == "local"

    def test_unknown_on_uncertain_keys_are_ignored_not_fatal(self) -> None:
        policy = Policy.from_dict(minimal_doc(on_uncertain={"sensitivity_bump_levels": 2, "made_up_knob": 7}))
        assert policy.uncertainty.sensitivity_bump_levels == 2

    @pytest.mark.parametrize("bump_key", ["sensitivity_bump_levels", "complexity_bump_levels"])
    def test_a_negative_bump_is_rejected_at_load(self, bump_key: str) -> None:
        # A negative bump is not a milder setting: it de-escalates an uncertain
        # judgement, which is the inversion of the knob's purpose ("bumps only
        # ever go stricter"). A policy that names one must fail the deploy
        # instead of turning "confidential, not sure" into "internal, send it".
        with pytest.raises(PolicyError, match="bumps only ever go stricter"):
            Policy.from_dict(minimal_doc(on_uncertain={bump_key: -2}))

    def test_a_zero_bump_loads_and_does_not_move_the_level(self) -> None:
        # Zero is the legal "disable this bump", and it must not be confused
        # with the negative case above: it loads, and it moves nothing.
        policy = Policy.from_dict(minimal_doc(on_uncertain={"sensitivity_bump_levels": 0}))
        assert policy.uncertainty.sensitivity_bump_levels == 0
        assert policy.escalate_sensitivity("confidential", 0) == "confidential"

    def test_rule_ids_and_reasons_are_derived_when_absent(self) -> None:
        policy = Policy.from_dict(
            minimal_doc(rules=[{"if": "gate_force_local", "then": {"tier": "local"}}, {"then": {"tier": "cheap"}}])
        )
        assert policy.rules[0].rule_id == "rule-0"
        assert policy.rules[1].rule_id == "rule-1-default"
        assert "default rule" in policy.rules[1].reason

    def test_explicit_reason_is_preserved(self) -> None:
        policy = Policy.from_dict(minimal_doc())
        doc = minimal_doc()
        doc["rules"][0]["reason"] = "because I said so"
        assert Policy.from_dict(doc).rules[0].reason == "because I said so"
        assert policy.rules[0].reason  # and a derived one is non-empty

    def test_models_for_tier_rejects_an_unknown_tier(self) -> None:
        with pytest.raises(PolicyError, match="undefined tier"):
            Policy.from_dict(minimal_doc()).models_for_tier("gpu")


# --------------------------------------------------------------------------- #
# Round-tripping
# --------------------------------------------------------------------------- #
class TestRoundTrip:
    def test_to_yaml_then_from_yaml_is_equivalent(self, shipped: Policy) -> None:
        restored = Policy.from_yaml(shipped.to_yaml(), source=shipped.source)
        assert restored == shipped
        assert dict(restored.tiers) == dict(shipped.tiers)
        assert [r.rule_id for r in restored.rules] == [r.rule_id for r in shipped.rules]
        assert [r.expression.source if r.expression else None for r in restored.rules] == [
            r.expression.source if r.expression else None for r in shipped.rules
        ]
        assert restored.failure == shipped.failure
        assert restored.uncertainty == shipped.uncertainty
        assert restored.tier_order == shipped.tier_order
        assert restored.pii_threshold == shipped.pii_threshold
        assert restored.version == shipped.version

    def test_round_trip_preserves_comments_it_never_understood(self, shipped: Policy) -> None:
        """Unknown top-level keys must survive, or `graduate` would silently drop config."""
        doc = minimal_doc(something_new={"nested": [1, 2]})
        policy = Policy.from_dict(doc)
        assert Policy.from_yaml(policy.to_yaml()).raw["something_new"] == {"nested": [1, 2]}

    def test_round_tripped_policy_resolves_the_same_tiers(self, shipped: Policy) -> None:
        restored = Policy.from_yaml(shipped.to_yaml())
        for ns in (
            namespace(),
            namespace(sensitivity="regulated"),
            namespace(complexity="frontier"),
            namespace(gate_force_local=True),
        ):
            assert restored.evaluate(ns)[1] == shipped.evaluate(ns)[1]

    def test_to_yaml_is_valid_yaml(self, shipped: Policy) -> None:
        assert isinstance(yaml.safe_load(shipped.to_yaml()), dict)

    def test_with_overrides_changes_one_knob_and_leaves_the_rest(self, shipped: Policy) -> None:
        tightened = shipped.with_overrides(pii_threshold=0.9)
        assert tightened.pii_threshold == 0.9
        assert shipped.pii_threshold == 0.5
        assert tightened.rules == shipped.rules
        assert dict(tightened.tiers) == dict(shipped.tiers)

    def test_with_overrides_does_not_mutate_the_original(self, shipped: Policy) -> None:
        before = shipped.to_yaml()
        shipped.with_overrides(pii_threshold=0.1, tiers={"local": ["x"], "cheap": ["y"], "strong": ["z"]})
        assert shipped.to_yaml() == before

    def test_with_backend_is_the_graduation_swap(self, shipped: Policy) -> None:
        graduated = shipped.with_backend({"name": "distilled", "artifact": "./artifacts/local"})
        assert graduated.backend == {"name": "distilled", "artifact": "./artifacts/local"}
        assert shipped.backend["name"] == "mock"
        # ...and it survives a write/read cycle, which is how `graduate` persists it.
        assert Policy.from_yaml(graduated.to_yaml()).backend == graduated.backend

    def test_with_overrides_replaces_whole_sections(self, shipped: Policy) -> None:
        changed = shipped.with_overrides(cache={"enabled": True, "ttl_seconds": 5})
        assert changed.cache == {"enabled": True, "ttl_seconds": 5}
        assert changed.logging == shipped.logging

    def test_write_yaml_creates_the_file_and_parent_dirs(self, tmp_path: Path, shipped: Policy) -> None:
        target = tmp_path / "nested" / "dir" / "policy.yaml"
        returned = shipped.write_yaml(target)
        assert returned == target
        assert target.exists()
        assert Policy.from_file(target).rules[0].rule_id == shipped.rules[0].rule_id

    def test_overrides_are_revalidated(self, shipped: Policy) -> None:
        """An override is not a backdoor past validation."""
        with pytest.raises(PolicyError):
            shipped.with_overrides(rules=[])
        with pytest.raises(PolicyError):
            shipped.with_overrides(tiers={"local": []})

    def test_source_is_carried_through_overrides(self, shipped: Policy) -> None:
        assert shipped.with_overrides(pii_threshold=0.7).source == shipped.source


# --------------------------------------------------------------------------- #
# Model selection and escalation ladders
# --------------------------------------------------------------------------- #
class TestModelSelection:
    @pytest.fixture
    def multi(self) -> Policy:
        return Policy.from_dict(
            minimal_doc(
                tiers={"local": ["l1", "l2", "l3"], "cheap": ["c1"], "strong": ["s1", "s2"]},
            )
        )

    def test_round_robin_across_a_multi_model_tier(self, multi: Policy) -> None:
        assert [multi.pick_model("local") for _ in range(7)] == ["l1", "l2", "l3", "l1", "l2", "l3", "l1"]

    def test_round_robin_is_independent_per_tier(self, multi: Policy) -> None:
        multi.pick_model("local")
        assert multi.pick_model("strong") == "s1"

    def test_single_model_tier_is_stable(self, multi: Policy) -> None:
        assert {multi.pick_model("cheap") for _ in range(5)} == {"c1"}

    def test_rule_that_pins_a_model_bypasses_round_robin(self) -> None:
        policy = Policy.from_dict(
            minimal_doc(
                tiers={"local": ["l1", "l2"], "cheap": ["c1"], "strong": ["s1"]},
                rules=[
                    {"id": "pin", "if": "gate_force_local", "then": {"tier": "local", "model": "tenant-pinned"}},
                    {"id": "d", "then": {"tier": "cheap"}},
                ],
            )
        )
        for _ in range(3):
            rule, tier, model = policy.evaluate(namespace(gate_force_local=True))
            assert (rule.rule_id, tier, model) == ("pin", "local", "tenant-pinned")

    def test_undefined_tier_at_evaluation_time_is_a_policy_error(self) -> None:
        policy = Policy.from_dict(minimal_doc())
        with pytest.raises(PolicyError):
            policy.pick_model("gpu")


class TestEscalationLadders:
    def test_bump_tier_moves_up_the_declared_order(self) -> None:
        policy = Policy.from_dict(minimal_doc(tier_order=["cheap", "strong", "local"]))
        assert policy.bump_tier("cheap") == "strong"
        assert policy.bump_tier("strong") == "local"
        assert policy.bump_tier("cheap", 2) == "local"

    def test_bump_tier_clamps_at_the_top(self) -> None:
        policy = Policy.from_dict(minimal_doc(tier_order=["cheap", "strong", "local"]))
        assert policy.bump_tier("local", 1) == "local"
        assert policy.bump_tier("local", 99) == "local"

    def test_bump_tier_clamps_at_the_bottom(self) -> None:
        policy = Policy.from_dict(minimal_doc(tier_order=["cheap", "strong", "local"]))
        assert policy.bump_tier("cheap", -1) == "cheap"
        assert policy.bump_tier("strong", -99) == "cheap"

    def test_bump_tier_leaves_an_unlisted_tier_alone(self) -> None:
        """The shipped policy's tier_order omits `local` on purpose; bumping it is a no-op."""
        policy = Policy.from_file(DEFAULT_POLICY)
        assert policy.tier_order == ("cheap", "strong")
        assert policy.bump_tier("local", 1) == "local"
        assert policy.bump_tier("local", -1) == "local"

    @pytest.mark.parametrize(
        ("level", "steps", "expected"),
        [
            ("public", 1, "internal"),
            ("internal", 1, "confidential"),
            ("confidential", 1, "regulated"),
            ("regulated", 1, "regulated"),
            ("regulated", 9, "regulated"),
            ("public", -1, "public"),
            ("regulated", -1, "confidential"),
            ("nonsense", 1, "nonsense"),
        ],
    )
    def test_escalate_sensitivity(self, level: str, steps: int, expected: str) -> None:
        assert Policy.from_dict(minimal_doc()).escalate_sensitivity(level, steps) == expected

    @pytest.mark.parametrize(
        ("level", "steps", "expected"),
        [
            ("trivial", 1, "standard"),
            ("standard", 1, "hard"),
            ("hard", 1, "frontier"),
            ("frontier", 1, "frontier"),
            ("frontier", 9, "frontier"),
            ("trivial", -1, "trivial"),
            ("frontier", -1, "hard"),
            ("nonsense", 1, "nonsense"),
        ],
    )
    def test_escalate_complexity(self, level: str, steps: int, expected: str) -> None:
        assert Policy.from_dict(minimal_doc()).escalate_complexity(level, steps) == expected

    def test_escalation_never_leaves_the_ladder(self) -> None:
        policy = Policy.from_dict(minimal_doc())
        for level in SENSITIVITY_LEVELS:
            for steps in range(-5, 6):
                assert policy.escalate_sensitivity(level, steps) in SENSITIVITY_LEVELS
        for level in COMPLEXITY_LEVELS:
            for steps in range(-5, 6):
                assert policy.escalate_complexity(level, steps) in COMPLEXITY_LEVELS

    def test_escalation_is_monotone(self) -> None:
        """Every bump moves stricter or stays put -- never looser."""
        policy = Policy.from_dict(minimal_doc())
        for level in SENSITIVITY_LEVELS:
            assert SENSITIVITY_LEVELS.index(policy.escalate_sensitivity(level)) >= SENSITIVITY_LEVELS.index(level)


# --------------------------------------------------------------------------- #
# Rule objects
# --------------------------------------------------------------------------- #
class TestRuleObjects:
    def test_is_default_is_true_only_without_an_expression(self) -> None:
        policy = Policy.from_dict(minimal_doc())
        assert policy.rules[0].is_default is False
        assert policy.rules[-1].is_default is True

    def test_rule_fields(self) -> None:
        policy = Policy.from_dict(minimal_doc())
        rule = policy.rules[0]
        assert isinstance(rule, Rule)
        assert rule.rule_id == "gate"
        assert rule.tier == "local"
        assert rule.model is None
        assert rule.expression is not None and rule.expression.source == "gate_force_local"

    def test_dataclass_defaults_are_safe(self) -> None:
        assert UncertaintyPolicy().pii_uncertain_counts_as_present is True
        assert FailurePolicy().mode == "fail_closed"
        assert FailurePolicy().fail_closed_tier == "local"

    def test_policy_equality_ignores_runtime_state(self) -> None:
        """Cursors and locks are per-process; two loads of one document are equal."""
        a = Policy.from_dict(minimal_doc())
        b = Policy.from_dict(minimal_doc())
        a.pick_model("local")
        assert a == b
        assert a.raw == b.raw
