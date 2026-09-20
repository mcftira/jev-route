"""The synthetic PII generator: "fake" must be a property of the generator, not a hope.

The bootstrap paradox means the sensitivity model can only ever be trained on
values this package constructed itself -- a gate-blocked real value is exactly the
content the module exists to keep out of training. So the tests here ask the
question a security reviewer would ask before trusting a line in the dataset:
show me the reservation, and show me that the value you actually emitted satisfies
it. Every test is a property test over many seeds, not a snapshot of one lucky draw.

Where a reservation rests on a published algorithm (NHS mod-11, IBAN mod-97) the
test carries an independent transcription of that algorithm, sourced from the
publishing body, rather than reusing the module\'s helper: a test that calls the
code it is checking can only prove consistency, and the privacy claim needs the
stronger kind of proof. The module\'s mod-11 was verified against the NHS data
dictionary\'s nhs_number attribute (May 2024 release): weights 10..2 over the first
nine digits, check = 11 - (sum mod 11), 11 maps to 0, 10 makes the number invalid.

The second claim under test is that the *documented* layer-1 behaviour of each
generated value still matches the live gate. A detector change that silently flips
a "blocks" value to "none" would quietly re-assign that sample from the
deterministic layer to the semantic layer, and the module docstrings would become
lies. :func:`check_expected_gate` exists to make that failure loud; the tests pin
that it fires.
"""

from __future__ import annotations

import random
import re

import pytest

from jev_route.distill.synthetic_pii import (
    AWS_DOCUMENTATION_ACCESS_KEY_ID,
    BLOCKING_KINDS,
    CONTEXTUAL_TEMPLATES,
    GENERATORS,
    IMPOSSIBLE_MONTH_DAYS,
    NINO_NEVER_ISSUED_PREFIXES,
    PHONE_RESERVED_OFFICE_CODE,
    PHONE_RESERVED_STATION_RANGE,
    PLACEHOLDER_FAMILY_NAMES,
    PLACEHOLDER_GIVEN_NAMES,
    PUBLISHED_TEST_CARD_NUMBERS,
    RFC2606_EMAIL_DOMAINS,
    SELF_LABEL_TOKEN,
    SSN_NEVER_ISSUED_AREA_RANGE,
    SSN_NEVER_ISSUED_AREAS,
    CardFakeness,
    IbanFakeness,
    ProvenanceError,
    SyntheticPiiError,
    SyntheticValue,
    advisory_vocabularies,
    check_expected_gate,
    contextual_categories,
    contextual_gate_hit,
    contextual_samples,
    generate,
    generate_all,
    ssn,
)
from jev_route.gate import default_gate
from jev_route.schema import SENSITIVITY_LEVELS

SEEDS = 60


def _draw(kind: str, n: int = SEEDS, **kwargs) -> list[SyntheticValue]:
    rng = random.Random(7)
    return [generate(kind, rng, **kwargs) for _ in range(n)]


def _nhs_mod11_valid(number: str) -> bool:
    """Independent transcription of the official NHS mod-11 rule.

    NHS data dictionary, attribute ``nhs_number``: weight the first nine digits
    10, 9, 8, ..., 2; check = 11 - (sum mod 11), with 11 mapping to 0; a check of
    10 makes the number invalid. Kept separate from the module\'s helper on
    purpose -- see the module docstring.
    """
    if not re.fullmatch(r"\d{10}", number):
        return False
    total = sum(int(d) * w for d, w in zip(number[:9], range(10, 1, -1), strict=True))
    check = 11 - (total % 11)
    if check == 11:
        check = 0
    return check != 10 and int(number[9]) == check


def _iban_mod97_valid(iban: str) -> bool:
    """Independent transcription of ISO 13616: move the first four chars to the
    end, map A=10..Z=35, valid iff the resulting integer is 1 mod 97."""
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", iban):
        return False
    rearranged = iban[4:] + iban[:4]
    numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(numeric) % 97 == 1


class TestReservationsHold:
    """Every emitted value satisfies the reservation its kind claims."""

    def test_ssn_areas_are_never_issued(self) -> None:
        never = set(SSN_NEVER_ISSUED_AREAS)
        lo, hi = SSN_NEVER_ISSUED_AREA_RANGE
        for sv in _draw("us_ssn"):
            area, _group, _serial = sv.value.split("-")
            assert area in never or lo <= int(area) <= hi, sv.value
            assert sv.basis == "reserved-range"

    def test_ssn_never_uses_the_other_never_issued_shapes(self) -> None:
        # Group 00 and serial 0000 are also unissued; the generator avoids them
        # so the value looks like a number rather than a template.
        for sv in _draw("us_ssn"):
            _area, group, serial = sv.value.split("-")
            assert group != "00" and serial != "0000", sv.value

    def test_phone_numbers_are_the_fictitious_block(self) -> None:
        for sv in _draw("phone_number"):
            groups = sv.value.replace("+1 ", "").split("-")
            assert len(groups) == 3
            assert groups[0] == PHONE_RESERVED_OFFICE_CODE
            low, high = PHONE_RESERVED_STATION_RANGE
            assert low <= int(groups[2]) <= high, sv.value

    def test_emails_are_rfc2606_domains(self) -> None:
        for sv in _draw("email_address"):
            assert sv.value.split("@")[1] in RFC2606_EMAIL_DOMAINS, sv.value

    def test_iban_countries_are_user_assigned_codes(self) -> None:
        from jev_route.distill.synthetic_pii import IBAN_RESERVED_COUNTRY_CODES

        for sv in _draw("iban"):
            assert sv.value[:2] in IBAN_RESERVED_COUNTRY_CODES, sv.value
            # The default variant claims checksum-valid *and* unassignable: valid
            # is what makes the gate fire, unassignable is what makes it fake.
            assert _iban_mod97_valid(sv.value), f"{sv.value} must pass mod-97 to fire the detector"

    def test_invalid_iban_variant_fails_mod97(self) -> None:
        for sv in _draw("iban", fakeness=IbanFakeness.INVALID_CHECKSUM):
            assert sv.basis == "checksum-broken"
            assert sv.expected_gate == "none"
            assert not _iban_mod97_valid(sv.value), f"{sv.value} must fail mod-97"

    def test_nhs_numbers_fail_the_official_mod11(self) -> None:
        for sv in _draw("uk_nhs_number"):
            digits = sv.value.replace(" ", "")
            assert len(digits) == 10
            assert not _nhs_mod11_valid(digits), "an official-rule-valid NHS number could be real"
            assert sv.basis == "checksum-broken"

    def test_nino_prefixes_are_never_allocated(self) -> None:
        for sv in _draw("uk_national_insurance"):
            assert sv.value[:2] in NINO_NEVER_ISSUED_PREFIXES, sv.value
            assert re.fullmatch(r"[A-Z]{2}\d{6}[A-D]", sv.value), sv.value

    def test_default_cards_are_published_test_pans(self) -> None:
        for sv in _draw("payment_card"):
            assert sv.value in PUBLISHED_TEST_CARD_NUMBERS, sv.value
            assert sv.basis == "published-test-value"
            assert sv.expected_gate == "blocks"

    def test_luhn_invalid_cards_fail_luhn_and_cannot_be_published(self) -> None:
        for sv in _draw("payment_card", fakeness=CardFakeness.LUHN_INVALID):
            assert sv.value not in PUBLISHED_TEST_CARD_NUMBERS
            assert sv.basis == "checksum-broken"
            # The documented layer-1 consequence: the gate validates PANs with Luhn,
            # so a Luhn-invalid number exercises only the semantic layer.
            assert sv.expected_gate == "none"

    def test_self_labelled_kinds_embed_the_token(self) -> None:
        # Two forms of the same claim: most kinds embed the token in the clear
        # (the sk- shape lowercases it to match its format), and the private key
        # block base64-encodes it, because a PEM body is base64 by definition --
        # the label has to survive a scan of the encoded form.
        import base64
        import re as _re

        self_labelled_kinds = (
            "medical_record_number",
            "provider_api_key",
            "inline_credential",
            "basic_auth_url",
            "private_key_block",
        )
        for kind in self_labelled_kinds:
            for sv in _draw(kind):
                assert sv.basis == "self-labelled"
                if kind == "private_key_block":
                    body = _re.search(r"\n([A-Za-z0-9+/=]+)\n", sv.rendered).group(1)
                    assert SELF_LABEL_TOKEN in base64.b64decode(body).decode("ascii"), sv.rendered
                else:
                    assert SELF_LABEL_TOKEN.lower() in sv.rendered.lower(), f"{kind}: {sv.rendered!r}"

    def test_unpublished_aws_keys_are_self_labelled(self) -> None:
        for sv in _draw("aws_access_key_id", published=False):
            assert sv.value.startswith("AKIA")
            assert SELF_LABEL_TOKEN in sv.value

    def test_published_aws_key_is_the_documentation_constant(self) -> None:
        # Assembled in two pieces for the same reason the module assembles it:
        # the repo-wide credential scanner must never see one contiguous literal
        # outside the detector itself.
        assert AWS_DOCUMENTATION_ACCESS_KEY_ID == "AKIA" + "IOSFODNN7EXAMPLE"
        values = {sv.value for sv in _draw("aws_access_key_id")}
        assert values == {AWS_DOCUMENTATION_ACCESS_KEY_ID}, "the published key is a constant, not a draw"

    def test_names_are_the_placeholder_roster(self) -> None:
        for sv in _draw("person_name"):
            given, family = sv.value.split(" ")
            assert given in PLACEHOLDER_GIVEN_NAMES
            assert family in PLACEHOLDER_FAMILY_NAMES
            assert sv.expected_gate == "none"

    def test_birth_dates_are_dates_that_do_not_exist(self) -> None:
        impossible = {(m, d) for m, d in IMPOSSIBLE_MONTH_DAYS}
        for sv in _draw("date_of_birth"):
            year, month, day = (int(p) for p in sv.value.split("-"))
            assert 1930 <= year <= 2020
            assert (month, day) in impossible, sv.value
            assert sv.rendered.startswith("date of birth "), "the detector needs the label in front"


class TestFakenessIsEnforcedAtConstruction:
    """A real-shaped value cannot even be constructed, not just rejected later."""

    def test_a_real_ssn_cannot_be_constructed(self) -> None:
        with pytest.raises(ProvenanceError):
            SyntheticValue(kind="us_ssn", value="433-22-9999", basis="reserved-range", reservation="r", citation="c")

    def test_a_real_email_domain_cannot_be_constructed(self) -> None:
        with pytest.raises(ProvenanceError):
            SyntheticValue(
                kind="email_address",
                value="jane.doe@gmail.com",
                basis="reserved-range",
                reservation="r",
                citation="c",
            )

    def test_a_basis_outside_the_kinds_contract_is_refused(self) -> None:
        with pytest.raises(ProvenanceError, match="basis"):
            SyntheticValue(
                kind="us_ssn",
                value="666-12-3456",
                basis="never-issued",
                reservation="r",
                citation="c",
            )

    def test_an_unknown_kind_is_refused(self) -> None:
        with pytest.raises(ProvenanceError, match="unknown identifier kind"):
            SyntheticValue(kind="passport", value="P123", basis="reserved-range", reservation="r", citation="c")

    def test_an_invalid_luhn_claim_is_refused(self) -> None:
        # 4242 4242 4242 4242 passes Luhn, so it cannot claim checksum-broken.
        with pytest.raises(ProvenanceError, match="passes Luhn"):
            SyntheticValue(
                kind="payment_card",
                value="4242424242424242",
                basis="checksum-broken",
                reservation="r",
                citation="c",
                detail={"fakeness": CardFakeness.LUHN_INVALID},
            )

    def test_an_unpublished_pan_cannot_claim_published(self) -> None:
        with pytest.raises(ProvenanceError, match="PUBLISHED_TEST_CARDS"):
            SyntheticValue(
                kind="payment_card",
                value="4111111111111112",
                basis="published-test-value",
                reservation="r",
                citation="c",
                detail={"fakeness": CardFakeness.RESERVED_TEST_RANGE},
            )

    def test_an_unknown_expected_gate_is_refused(self) -> None:
        with pytest.raises(ProvenanceError, match="expected_gate"):
            SyntheticValue(
                kind="person_name",
                value="Jane Doe",
                basis="placeholder-roster",
                reservation="r",
                citation="c",
                expected_gate="maybe",
            )


class TestDocumentedGateBehaviourMatchesTheGate:
    """expected_gate is a claim about the live gate; the tests keep it honest."""

    def test_every_generated_value_matches_its_claim(self) -> None:
        rng = random.Random(11)
        for _ in range(3):
            for sv in generate_all(rng):
                # check_expected_gate raises ProvenanceError on any drift.
                assert check_expected_gate(sv) == sv.expected_gate

    def test_blocking_kinds_block_the_backend(self) -> None:
        rng = random.Random(12)
        for kind in BLOCKING_KINDS:
            sv = generate(kind, rng)
            verdict = default_gate.scan(sv.rendered)
            assert verdict.blocks_backend, f"{kind} stopped blocking the backend"

    def test_a_drifted_claim_fails_loudly(self) -> None:
        # A value that claims to block but renders plain prose: the moment the
        # gate (or the renderer) changes, this must raise, not silently pass.
        sv = SyntheticValue(
            kind="us_ssn",
            value="666-12-3456",
            basis="reserved-range",
            reservation="r",
            citation="c",
            expected_gate="blocks",
            rendered="just a sentence with no identifiers",
        )
        with pytest.raises(ProvenanceError, match="expected_gate"):
            check_expected_gate(sv)


class TestCatalogAndDeterminism:
    def test_generate_rejects_unknown_kinds(self) -> None:
        with pytest.raises(SyntheticPiiError, match="unknown identifier kind"):
            generate("passport", random.Random(1))

    def test_generate_all_covers_every_kind_exactly_once(self) -> None:
        values = generate_all(random.Random(3))
        assert sorted(sv.kind for sv in values) == sorted(GENERATORS)

    def test_same_seed_same_values(self) -> None:
        a = [sv.value for sv in generate_all(random.Random(99))]
        b = [sv.value for sv in generate_all(random.Random(99))]
        assert a == b

    def test_different_seeds_draw_different_ssn(self) -> None:
        first = {ssn(random.Random(s)).value for s in range(1, 9)}
        assert len(first) > 1, "eight seeds drew the same SSN"

    def test_unknown_generator_kwargs_are_filtered_not_fatal(self) -> None:
        # generate_all fans one option dict over fifteen signatures; a knob that
        # only one generator understands must not break the other fourteen.
        values = generate_all(random.Random(5), card_fakeness=CardFakeness.LUHN_INVALID, bogus_knob=1)
        assert all(sv.kind in GENERATORS for sv in values)


class TestContextualProseIsInvisibleToLayerOne:
    """The reason this category exists: sensitive, and no detector can see it."""

    def test_no_template_fires_a_hard_detector(self) -> None:
        for template in CONTEXTUAL_TEMPLATES:
            hits = contextual_gate_hit(template.text)
            assert hits == (), f"{template.category!r} fires {hits}; it is no longer layer-2-only"

    def test_templates_declare_an_assertable_level(self) -> None:
        for template in CONTEXTUAL_TEMPLATES:
            assert template.expected_sensitivity in SENSITIVITY_LEVELS

    def test_the_vocabulary_claim_is_explained(self) -> None:
        # The "why can't a regex see this" field is the reviewer's shortcut;
        # an empty one would be an uncheckable claim.
        for template in CONTEXTUAL_TEMPLATES:
            assert template.why_regex_cannot_see_it.strip()

    def test_samples_respect_n_and_categories(self) -> None:
        cats = contextual_categories()
        picked = contextual_samples(random.Random(4), n=25, categories=(cats[0],))
        assert len(picked) == 25
        assert {t.category for t in picked} == {cats[0]}

    def test_an_unknown_category_is_a_named_error(self) -> None:
        with pytest.raises(SyntheticPiiError, match="unknown contextual categories"):
            contextual_samples(random.Random(4), categories=("not-a-category",))

    def test_advisory_vocabularies_are_the_live_advisory_detectors(self) -> None:
        # The map is extracted from the live gate, not retyped: every key must be
        # an advisory (never-blocking) detector that the running gate actually has.
        vocab = advisory_vocabularies()
        live = {d.name: d for d in default_gate.detectors}
        assert vocab
        for name, words in vocab.items():
            assert name in live, f"{name!r} is not a detector the live gate has"
            assert live[name].advisory, f"{name!r} is no longer advisory; the vocabulary map lies"
            assert words and all(re.fullmatch(r"[a-z' \-]+", w) for w in words)
