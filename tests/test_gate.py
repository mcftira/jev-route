"""Tests for :mod:`jev_route.gate` -- the local hard gate.

The gate is the one component that is never model-decided and never bypassed, so
its test suite is written adversarially: for every "this is caught" there is a
"this lookalike is not", because a gate that fires on everything gets disabled by
an operator within a week, and a disabled gate is worse than no gate.

Three properties dominate the file and are worth naming:

* **Checksums are the difference between a gate and a nuisance.** A Luhn-invalid
  digit run of exactly the same shape as a card number must not be reported as a
  card. Same for IBAN mod-97, NHS mod-11, and NINO letter exclusions.
* **Advisory is not hard.** A topic keyword ("HIPAA", "CVV", "PCI-DSS") is a hint
  to the backend and a line in the training log. It must never become a
  sensitivity floor, never force a tier, and never block the cloud call. If it
  did, every compliance question in the corpus would route to an air-gapped model
  and the calibrated router would be indistinguishable from a keyword filter.
* **Nothing matched is retained.** A finding carries a hash of the span, not the
  span. The log has to prove a card number was present without keeping it.
"""

from __future__ import annotations

import re
from dataclasses import fields

import pytest

from jev_route.gate import (
    DEFAULT_DETECTORS,
    Detector,
    HardGate,
    default_gate,
    iban_ok,
    luhn_ok,
    scan,
)
from jev_route.schema import SENSITIVITY_LEVELS, GateFinding, GateVerdict

CARD_VALID = "4111 1111 1111 1111"
CARD_VALID_MC = "5555 5555 5555 4444"
CARD_INVALID_SPACED = "4111 1111 1111 1112"
CARD_INVALID_FLAT = "4111111111111112"
IBAN_VALID = "GB33 BUKB 2020 1555 5555 55"
IBAN_VALID_DE = "DE89 3704 0044 0532 0130 00"
IBAN_INVALID = "GB33 BUKB 2020 1555 5555 56"
NHS_VALID = "943 476 5919"
NHS_BAD_CHECK = "943 476 5918"
NINO_VALID = "AB123456C"
EMAIL_INVENTED = "dana.kovacs@northside-health.org"
KEY_DOTTED = "sk-demo-D.OTTED.abcdefghijklmnopqrstuvwx"


@pytest.fixture
def gate() -> HardGate:
    """A default gate, freshly built: no test may leak state into another."""
    return HardGate()


def names(verdict: GateVerdict) -> set[str]:
    """Detector names that fired. Most assertions care about names, not order."""
    return {f.detector for f in verdict.findings}


# --------------------------------------------------------------------------- #
# Checksum validators, exercised directly
# --------------------------------------------------------------------------- #
class TestChecksumValidators:
    """The validators are public so a test can pin them without going through regexes.

    Testing them directly matters: if the *pattern* stops matching, the finding
    disappears silently, and only a direct checksum test tells you whether the
    arithmetic still works.
    """

    @pytest.mark.parametrize("candidate", ["4111111111111111", "5555555555554444", "4111 1111 1111 1111"])
    def test_luhn_accepts_valid_cards(self, candidate: str) -> None:
        assert luhn_ok(candidate) is True

    @pytest.mark.parametrize("candidate", ["4111111111111112", "1234567890123456", "0000000000000001"])
    def test_luhn_rejects_invalid_cards(self, candidate: str) -> None:
        assert luhn_ok(candidate) is False

    def test_luhn_rejects_too_short_and_too_long(self) -> None:
        # 12 and 20 digits are outside the ISO/IEC 7812 range the detector claims.
        assert luhn_ok("411111111111") is False
        assert luhn_ok("41111111111111111111") is False

    @pytest.mark.parametrize("candidate", [IBAN_VALID, IBAN_VALID_DE, "GB33BUKB20201555555555"])
    def test_iban_accepts_mod97_valid(self, candidate: str) -> None:
        assert iban_ok(candidate) is True

    @pytest.mark.parametrize("candidate", [IBAN_INVALID, "DE89 3704 0044 0532 0130 02", "XX12", "not an iban"])
    def test_iban_rejects_mod97_invalid(self, candidate: str) -> None:
        assert iban_ok(candidate) is False


# --------------------------------------------------------------------------- #
# Payment cards: the checksum is the whole point
# --------------------------------------------------------------------------- #
class TestPaymentCard:
    @pytest.mark.parametrize("card", [CARD_VALID, CARD_VALID_MC])
    def test_luhn_valid_card_is_caught(self, gate: HardGate, card: str) -> None:
        verdict = gate.scan(f"Please charge the card {card} for order 88.")
        assert "payment_card" in names(verdict)
        # Regulated data blocks the cloud entirely: redaction is not trusted to be
        # the only thing between a card number and a third-party API.
        assert verdict.sensitivity_floor == "regulated"
        assert verdict.blocks_backend is True
        assert verdict.force_local is True
        assert verdict.pii_floor == 1.0

    def test_luhn_valid_card_caught_contiguous(self, gate: HardGate) -> None:
        assert "payment_card" in names(gate.scan("card 4111111111111111 on file"))

    @pytest.mark.parametrize("card", [CARD_INVALID_SPACED, CARD_INVALID_FLAT])
    def test_luhn_invalid_card_is_not_reported_as_a_card(self, gate: HardGate, card: str) -> None:
        """A digit run one checksum away from a card must not be called a card.

        Note what is *not* asserted: the flat 16-digit run produces no finding at
        all, while the spaced one still trips the shape-only ``phone_number``
        detector on an inner group. That is deliberate and documented here so a
        future reader does not mistake it for a leak -- the promise is that the
        Luhn-validated detector does not fire, not that the gate is blind.
        """
        verdict = gate.scan(f"Please charge the card {card} for order 88.")
        assert "payment_card" not in names(verdict)
        assert verdict.blocks_backend is False

    def test_luhn_invalid_flat_run_produces_no_finding(self, gate: HardGate) -> None:
        assert gate.scan(CARD_INVALID_FLAT).fired is False

    def test_short_digit_runs_are_ignored(self, gate: HardGate) -> None:
        # Order numbers, timestamps and versions are the classic false positives.
        for text in ("order 12345", "build 20240117", "version 1.101.0", "invoice 99887766"):
            assert "payment_card" not in names(gate.scan(text)), text


# --------------------------------------------------------------------------- #
# US SSN
# --------------------------------------------------------------------------- #
class TestUsSsn:
    @pytest.mark.parametrize("ssn", ["000-12-3456", "666-45-6789", "900-45-6789", "078-05-1120"])
    def test_ssn_shape_is_caught_including_reserved_ranges(self, gate: HardGate, ssn: str) -> None:
        """Reserved ranges (000-, 666-, 9xx-) are the SSA's own test numbers.

        They are caught on purpose: the detector's job is "a 3-2-4 government
        identifier shape is present", and the only safe reading of a reserved-range
        SSN in a prompt is that somebody pasted a test fixture from real data.
        """
        verdict = gate.scan(f"employee ssn {ssn} for the payroll export")
        assert "us_ssn" in names(verdict)
        assert verdict.sensitivity_floor == "regulated"
        assert verdict.blocks_backend is True

    @pytest.mark.parametrize(
        "text",
        [
            "ref 123-45-67890",  # 5 digits in the last group
            "ref 12-345-6789",  # 2-3-4 instead of 3-2-4
            "ref 123456789",  # contiguous, no separators
            "ref 123-456-7890",  # phone-shaped 3-3-4
            "case 1234-56-7890",  # 4 digits in the first group
        ],
    )
    def test_non_ssn_groupings_are_not_caught(self, gate: HardGate, text: str) -> None:
        assert "us_ssn" not in names(gate.scan(text))


# --------------------------------------------------------------------------- #
# IBAN, NHS number, UK National Insurance number
# --------------------------------------------------------------------------- #
class TestRegulatedIdentifiers:
    @pytest.mark.parametrize("iban", [IBAN_VALID, IBAN_VALID_DE])
    def test_valid_iban_is_caught(self, gate: HardGate, iban: str) -> None:
        verdict = gate.scan(f"Please wire the invoice to IBAN {iban} today.")
        assert "iban" in names(verdict)
        assert verdict.sensitivity_floor == "regulated"
        assert verdict.blocks_backend is True

    def test_invalid_iban_is_not_caught_as_an_iban(self, gate: HardGate) -> None:
        """mod-97 failure means "not an account number", so the iban detector stays quiet.

        The verdict may still be non-clean: a long digit run inside the IBAN can
        satisfy the Luhn check and be reported as a ``payment_card``. That is the
        conservative direction -- a false positive costs a tier, a false negative
        leaks an account -- and it is asserted here so nobody "fixes" it by
        loosening the checksum.
        """
        verdict = gate.scan(f"Please wire the invoice to IBAN {IBAN_INVALID} today.")
        assert "iban" not in names(verdict)

    def test_valid_nhs_number_is_caught(self, gate: HardGate) -> None:
        verdict = gate.scan(f"Patient NHS number {NHS_VALID} was admitted yesterday.")
        assert "uk_nhs_number" in names(verdict)
        assert verdict.sensitivity_floor == "regulated"
        assert verdict.blocks_backend is True

    def test_nhs_number_with_bad_check_digit_is_not_caught(self, gate: HardGate) -> None:
        """mod-11 is what separates an NHS number from a formatted phone number."""
        assert "uk_nhs_number" not in names(gate.scan(f"Patient NHS number {NHS_BAD_CHECK} was admitted."))

    def test_valid_nino_is_caught(self, gate: HardGate) -> None:
        verdict = gate.scan(f"Employee NI number {NINO_VALID} for payroll.")
        assert "uk_national_insurance" in names(verdict)
        assert verdict.blocks_backend is True

    @pytest.mark.parametrize(
        "nino",
        [
            "GB123456C",  # reserved prefix
            "NK123456C",  # reserved prefix
            "TN123456C",  # reserved prefix
            "ZZ123456C",  # reserved prefix
            "QQ123456A",  # Q is not a legal prefix letter
            "AB123456E",  # suffix must be A-D
            "AB12345C",  # too few digits
        ],
    )
    def test_nino_exclusions_are_honoured(self, gate: HardGate, nino: str) -> None:
        assert "uk_national_insurance" not in names(gate.scan(f"Employee NI number {nino} for payroll."))

    @pytest.mark.parametrize("text", ["patient id MRN-88431", "MRN: 88431X", "chart no. 4471"])
    def test_medical_record_number_is_caught(self, gate: HardGate, text: str) -> None:
        verdict = gate.scan(text)
        assert "medical_record_number" in names(verdict)
        assert verdict.sensitivity_floor == "regulated"


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #
class TestSecrets:
    @pytest.mark.parametrize(
        "key",
        [
            "sk-proj-abcdefghijklmnopqrstuvwx",
            KEY_DOTTED,
            "AKIAIOSFODNN7EXAMPLE",
            "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij",
            "github_pat_11ABCDEFGHIJKLMNOPQRSTUV_abcdefghijklmnopqrstuvwxyz01",
            "xoxb-123456789012-abcdefghijklmnop",
            "xoxp-123456789012-abcdefghijklmnop",
            "AIzaSyA1234567890abcdefghijklmnopqrstuv",
            "sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWX",
            "hf_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh",
        ],
    )
    def test_provider_api_keys_are_caught(self, gate: HardGate, key: str) -> None:
        # Deliberately not "API_KEY=<key>": that form also matches inline_credential,
        # whose span contains this one, and containment resolves in favour of the
        # containing detector. See test_key_in_assignment_form_is_reported_as_inline.
        verdict = gate.scan(f"rotate the credential {key} before Friday")
        assert "provider_api_key" in names(verdict), key
        assert verdict.blocks_backend is True
        assert verdict.sensitivity_floor == "confidential"

    def test_key_containing_dots_is_caught(self, gate: HardGate) -> None:
        """Regression: provider keys can embed dots (``sk-demo-D.OTTED.abc...``).

        An earlier ``sk-[A-Za-z0-9]{20,}`` pattern stopped at the first dot, so the
        one credential class the project ships with support for was the one class the
        gate missed. The dot is in the character class now and this test keeps it there.
        """
        assert "." in KEY_DOTTED
        verdict = gate.scan(f"ALIBABA_API_KEY={KEY_DOTTED}")
        assert "provider_api_key" in names(verdict)

    def test_key_in_assignment_form_is_reported_as_inline(self, gate: HardGate) -> None:
        """``api_key=<provider key>`` reports BOTH the container and the contained.

        Both detectors are non-advisory secrets with the same floor and the same
        ``blocks_backend``, so the *verdict* is identical whichever names appear --
        the detector names only change what the decision log says. Reporting both
        is deliberate and is the reason ``Detector.specificity`` exists: the
        generic ``key = value`` shape spans the whole assignment, and letting the
        wider match swallow the narrower one would discard which provider's
        credential was in the prompt, which is the first thing an operator asks
        when a request gets air-gapped. The sibling test
        ``test_key_containing_dots_is_caught`` asserts the other half of the same
        guarantee, so an exact-set assertion here would contradict it.
        """
        verdict = gate.scan("API_KEY=sk-proj-abcdefghijklmnopqrstuvwx")
        assert names(verdict) == {"inline_credential", "provider_api_key"}
        assert verdict.blocks_backend is True
        assert verdict.sensitivity_floor == "confidential"

    def test_short_sk_prefix_is_not_a_key(self, gate: HardGate) -> None:
        # "sk-" plus a handful of characters is prose in half of all ML docs.
        assert "provider_api_key" not in names(gate.scan("use the sk-latest alias here"))

    @pytest.mark.parametrize(
        "text",
        [
            "-----BEGIN RSA PRIVATE KEY-----",
            "-----BEGIN OPENSSH PRIVATE KEY-----",
            "-----BEGIN PRIVATE KEY-----",
            "-----BEGIN PGP PRIVATE KEY BLOCK-----",
        ],
    )
    def test_private_key_block_is_caught(self, gate: HardGate, text: str) -> None:
        verdict = gate.scan(f"{text}\nMIIEowIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----")
        assert "private_key_block" in names(verdict)
        assert verdict.blocks_backend is True

    @pytest.mark.parametrize(
        "text",
        [
            "config password=Sup3rSecret!value",
            "password: 'correct horse battery staple'",
            "api_key=abcdef0123456789",
            "access_token: ya29.a0AfH6SMBx",
        ],
    )
    def test_inline_credential_is_caught(self, gate: HardGate, text: str) -> None:
        assert "inline_credential" in names(gate.scan(text))

    def test_short_inline_value_is_not_a_credential(self, gate: HardGate) -> None:
        # Six characters is the floor: "password=123" in a doc example is not a leak.
        assert "inline_credential" not in names(gate.scan("password=abc"))

    def test_basic_auth_url_is_caught(self, gate: HardGate) -> None:
        verdict = gate.scan("DATABASE_URL=postgres://router:hunter2@db.internal.northside.example:5432/app")
        assert "basic_auth_url" in names(verdict)
        assert verdict.blocks_backend is True

    # -- Known gap, reported not patched. See the module-level BUG note. -------- #
    @pytest.mark.parametrize(
        "text",
        [
            "DB_PASSWORD=hunter2hunter2",
            "db_password=hunter2hunter2",
            "client_secret=abc123456",
            "secret_token = 9f8e7d6c5b4a",
        ],
    )
    def test_prefixed_credential_names_are_caught(self, gate: HardGate, text: str) -> None:
        """A credential detector that misses ``PREFIX_PASSWORD=`` misses most .env files.

        Suggested fix: replace the leading ``\\b`` with ``(?<![A-Za-z0-9])`` and allow
        an optional ``[\\w.\\-]*`` prefix plus a ``secret_?\\w*`` alternative, e.g.
        ``(?<![A-Za-z0-9])[\\w.\\-]*(?:password|passwd|pwd|secret|api_?key|token)``.
        """
        assert "inline_credential" in names(gate.scan(text))

    def test_plain_url_has_no_credentials(self, gate: HardGate) -> None:
        assert "basic_auth_url" not in names(gate.scan("see https://example.com/docs/routing for details"))


# --------------------------------------------------------------------------- #
# Emails and the RFC 2606 question
# --------------------------------------------------------------------------- #
class TestEmailAddresses:
    @pytest.mark.parametrize(
        "address",
        [
            "user@example.com",
            "someone@example.org",
            "a@sub.domain.example.org",
            "docs@example.net",
            "x@example.invalid",
        ],
    )
    def test_rfc2606_addresses_are_not_pii_by_default(self, gate: HardGate, address: str) -> None:
        """Reserved documentation domains hold nobody's address.

        Firing on ``sales@example.com`` is how a gate earns a reputation for crying
        wolf, and a gate that cries wolf gets disabled.
        """
        verdict = gate.scan(f"Send the report to {address} when it is ready.")
        assert "email_address" not in names(verdict)
        assert verdict.fired is False

    @pytest.mark.parametrize("address", ["user@example.com", "a@sub.domain.example.org"])
    def test_rfc2606_addresses_are_pii_when_configured(self, address: str) -> None:
        strict = HardGate(placeholder_domains_as_pii=True)
        verdict = strict.scan(f"Send the report to {address} when it is ready.")
        assert "email_address" in names(verdict)
        assert verdict.force_local is True
        # Personal identifiers are redactable, so they force local without blocking
        # the classification call outright -- unlike credential material.
        assert verdict.blocks_backend is False

    def test_invented_realistic_domain_is_pii_by_default(self, gate: HardGate) -> None:
        verdict = gate.scan(f"Please forward the results to {EMAIL_INVENTED} today.")
        assert "email_address" in names(verdict)
        assert verdict.sensitivity_floor == "confidential"
        assert verdict.force_local is True
        assert verdict.pii_floor == 1.0

    def test_subdomain_of_a_reserved_domain_is_still_reserved(self, gate: HardGate) -> None:
        assert "email_address" not in names(gate.scan("mail me at team@mail.example.com"))

    def test_lookalike_domain_is_treated_as_real(self, gate: HardGate) -> None:
        # "notexample.com" only *contains* a reserved name; it is not one.
        assert "email_address" in names(gate.scan("mail me at someone@notexample.com"))


# --------------------------------------------------------------------------- #
# Advisory vs hard -- the most important behaviour in the module
# --------------------------------------------------------------------------- #
class TestAdvisoryIsNotHard:
    """A topic mention is a hint. A hard finding is a floor. Collapsing them is the bug."""

    @pytest.mark.parametrize(
        ("text", "expected_topic"),
        [
            ("Explain how HIPAA actually works and who it applies to.", "kw_health_regulation"),
            ("What is a credit card CVV?", "kw_financial_regulation"),
            ("What does PCI-DSS require of merchants who store card data?", "kw_financial_regulation"),
            ("Summarize the attorney-client privilege rules in Texas.", "kw_legal_privilege"),
            ("What does COPPA say about collecting data from children?", "kw_minors"),
            ("How should we handle a zero-day disclosure?", "kw_security_vulnerability"),
            ("Define 'trade secret' for a new hire.", "kw_confidential_business"),
        ],
    )
    def test_topic_questions_are_advisory_only(self, gate: HardGate, text: str, expected_topic: str) -> None:
        verdict = gate.scan(text)
        assert expected_topic in verdict.advisory_topics
        # The four assertions that matter. A topic must never move any of these.
        assert verdict.sensitivity_floor is None
        assert verdict.force_local is False
        assert verdict.blocks_backend is False
        assert verdict.pii_floor is None

    def test_advisory_findings_do_not_contribute_a_floor(self, gate: HardGate) -> None:
        verdict = gate.scan("Explain how HIPAA actually works and who it applies to.")
        assert verdict.fired is True, "the hit is recorded, it is simply not a floor"
        for finding in verdict.findings:
            assert finding.sensitivity_floor is None or finding.force_local is False
        assert all(f.detector.startswith("kw_") for f in verdict.findings)

    def test_hard_finding_and_topic_together_take_the_hard_floor(self, gate: HardGate) -> None:
        """Mixing the two must produce the hard finding's floor, not a sum of both."""
        verdict = gate.scan(f"Explain HIPAA, then email the record to {EMAIL_INVENTED}.")
        assert "kw_health_regulation" in verdict.advisory_topics
        assert verdict.sensitivity_floor == "confidential"  # from email_address
        assert verdict.force_local is True
        assert verdict.blocks_backend is False

    def test_every_advisory_detector_has_pii_false(self) -> None:
        """An advisory hit must not be able to set the pii floor either."""
        for detector in DEFAULT_DETECTORS:
            if detector.advisory:
                assert detector.pii is False, detector.name
                assert detector.force_local is False
                assert detector.blocks_backend is False

    def test_no_advisory_detector_can_set_a_floor_in_a_verdict(self, gate: HardGate) -> None:
        # A text that hits only advisory detectors must leave every floor unset.
        text = (
            "HIPAA, GDPR, PCI-DSS, COPPA, attorney-client privilege, trade secret, "
            "zero-day, internal only, performance review."
        )
        verdict = gate.scan(text)
        assert verdict.advisory_topics
        assert verdict.sensitivity_floor is None
        assert verdict.pii_floor is None
        assert verdict.force_local is False
        assert verdict.blocks_backend is False


# --------------------------------------------------------------------------- #
# Detector construction invariants
# --------------------------------------------------------------------------- #
class TestDetectorInvariants:
    def _detector(self, **overrides: object) -> Detector:
        base: dict[str, object] = {
            "name": "unit",
            "category": "secret",
            "pattern": re.compile(r"xyz"),
            "sensitivity_floor": "confidential",
        }
        base.update(overrides)
        return Detector(**base)  # type: ignore[arg-type]

    def test_blocks_backend_implies_force_local(self) -> None:
        """__post_init__ invariant: refusing to show text to a classifier implies
        refusing to route it to a model. Anything else would be a hole."""
        detector = self._detector(blocks_backend=True, force_local=False)
        assert detector.blocks_backend is True
        assert detector.force_local is True

    @pytest.mark.parametrize(
        "kwargs", [{"force_local": True}, {"blocks_backend": True}, {"force_local": True, "blocks_backend": True}]
    )
    def test_advisory_detector_cannot_force_or_block(self, kwargs: dict[str, bool]) -> None:
        with pytest.raises(ValueError, match="advisory detector cannot force_local or block"):
            self._detector(category="keyword", sensitivity_floor="internal", advisory=True, **kwargs)

    def test_unknown_sensitivity_floor_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown sensitivity floor"):
            self._detector(sensitivity_floor="top-secret")

    @pytest.mark.parametrize("floor", SENSITIVITY_LEVELS)
    def test_every_ladder_floor_is_accepted(self, floor: str) -> None:
        assert self._detector(sensitivity_floor=floor).sensitivity_floor == floor

    def test_default_detectors_all_have_valid_floors(self) -> None:
        for detector in DEFAULT_DETECTORS:
            assert detector.sensitivity_floor in SENSITIVITY_LEVELS
            assert detector.category in {"pii", "secret", "regulated", "keyword"}

    def test_default_detector_names_are_unique(self) -> None:
        detector_names = [d.name for d in DEFAULT_DETECTORS]
        assert len(detector_names) == len(set(detector_names))


# --------------------------------------------------------------------------- #
# Containment suppression
# --------------------------------------------------------------------------- #
class TestContainmentSuppression:
    def test_one_identifier_one_finding(self, gate: HardGate) -> None:
        """A card number contains several phone-shaped digit runs. Report the card.

        "payment_card + phone_number" for one sixteen-digit number is noise that
        double-counts features and makes the decision log harder to read.
        """
        verdict = gate.scan(CARD_VALID)
        assert names(verdict) == {"payment_card"}
        assert sum(f.count for f in verdict.findings) == 1

    def test_counts_are_per_detector(self, gate: HardGate) -> None:
        verdict = gate.scan(f"cards {CARD_VALID} and {CARD_VALID_MC} on file")
        assert verdict.detectors() == {"payment_card": 2}

    def test_advisory_hits_are_never_suppressed_by_containment(self, gate: HardGate) -> None:
        """A keyword inside a matched identifier span is still a topic signal."""
        verdict = gate.scan("HIPAA applies to the medical record attached to 4111 1111 1111 1111")
        assert "kw_health_regulation" in verdict.advisory_topics
        assert "payment_card" in names(verdict)

    def test_multiple_distinct_identifiers_all_reported(self, gate: HardGate) -> None:
        verdict = gate.scan(f"card {CARD_VALID}, email {EMAIL_INVENTED}, ssn 000-12-3456")
        assert {"payment_card", "email_address", "us_ssn"} <= names(verdict)


# --------------------------------------------------------------------------- #
# Disabling detectors
# --------------------------------------------------------------------------- #
class TestDisabledDetectors:
    def test_disabling_one_detector_silences_only_that_one(self) -> None:
        text = f"call 020 7946 0958 or email {EMAIL_INVENTED}"
        assert "phone_number" in names(default_gate.scan(text))
        quiet = HardGate(disabled_detectors=["phone_number"])
        verdict = quiet.scan(text)
        assert "phone_number" not in names(verdict)
        assert "email_address" in names(verdict)
        assert quiet.disabled == frozenset({"phone_number"})

    def test_disabling_a_blocking_detector_removes_the_block(self) -> None:
        quiet = HardGate(disabled_detectors=["payment_card"])
        assert quiet.scan(CARD_VALID).blocks_backend is False

    def test_unknown_detector_name_raises(self) -> None:
        """A typo in config must not silently disable nothing at all."""
        with pytest.raises(ValueError, match="unknown detector names"):
            HardGate(disabled_detectors=["payment_cards"])

    def test_gate_still_runs_when_every_detector_is_disabled(self) -> None:
        empty = HardGate(disabled_detectors=[d.name for d in DEFAULT_DETECTORS])
        verdict = empty.scan(f"{CARD_VALID} {EMAIL_INVENTED} HIPAA")
        assert isinstance(verdict, GateVerdict)
        assert verdict.fired is False
        assert verdict.findings == ()


# --------------------------------------------------------------------------- #
# Clean inputs
# --------------------------------------------------------------------------- #
class TestCleanInputs:
    def test_empty_string_returns_a_clean_verdict(self, gate: HardGate) -> None:
        verdict = gate.scan("")
        assert verdict == GateVerdict.clean()
        assert verdict.fired is False
        assert verdict.findings == ()
        assert verdict.sensitivity_floor is None
        assert verdict.pii_floor is None
        assert verdict.force_local is False
        assert verdict.blocks_backend is False
        assert verdict.advisory_topics == ()

    @pytest.mark.parametrize(
        "text",
        [
            "Summarize the three main arguments of this essay about urban cycling.",
            "def add(a, b):\n    return a + b\n",
            "The quarterly revenue grew 12% and the churn rate fell to 3%.",
            "Meet me at 5 Main Street at 3pm to discuss the roadmap.",
        ],
    )
    def test_clean_prose_returns_a_clean_verdict(self, gate: HardGate, text: str) -> None:
        verdict = gate.scan(text)
        assert verdict.fired is False, verdict.findings
        assert verdict == GateVerdict.clean()

    def test_module_level_convenience_wrapper(self) -> None:
        assert isinstance(default_gate, HardGate)
        assert scan(CARD_VALID).blocks_backend is True
        assert scan("").fired is False


# --------------------------------------------------------------------------- #
# Privacy: nothing matched is retained
# --------------------------------------------------------------------------- #
class TestNoRetention:
    HEX16 = re.compile(r"\A[0-9a-f]{16}\Z")

    @pytest.mark.parametrize(
        "secret",
        [CARD_VALID, CARD_VALID_MC, IBAN_VALID, NHS_VALID, EMAIL_INVENTED, KEY_DOTTED, "000-12-3456"],
    )
    def test_span_hash_is_16_hex_chars(self, gate: HardGate, secret: str) -> None:
        verdict = gate.scan(f"here it is: {secret} -- handle with care")
        assert verdict.findings, f"expected a finding for {secret!r}"
        for finding in verdict.findings:
            assert self.HEX16.match(finding.span_hash), finding

    @pytest.mark.parametrize(
        "secret",
        [CARD_VALID, CARD_VALID_MC, IBAN_VALID, NHS_VALID, EMAIL_INVENTED, KEY_DOTTED, "000-12-3456"],
    )
    def test_no_finding_field_contains_the_matched_text(self, gate: HardGate, secret: str) -> None:
        """The log must prove a detector fired without keeping what it matched."""
        text = f"here it is: {secret} -- handle with care"
        verdict = gate.scan(text)
        # The bare secret, the compact form, and any distinctive inner fragment.
        fragments = {secret, secret.replace(" ", ""), secret.replace("-", "")}
        fragments |= {t for t in re.split(r"[\s@.\-]+", secret) if len(t) >= 4}
        for finding in verdict.findings:
            for value in (getattr(finding, f.name) for f in fields(GateFinding)):
                rendered = str(value)
                for fragment in fragments:
                    assert fragment not in rendered, (finding, fragment)

    def test_verdict_serialization_contains_no_matched_text(self, gate: HardGate) -> None:
        text = f"card {CARD_VALID} email {EMAIL_INVENTED} key {KEY_DOTTED}"
        blob = str(gate.scan(text).to_dict())
        for secret in (CARD_VALID, CARD_VALID.replace(" ", ""), EMAIL_INVENTED, KEY_DOTTED):
            assert secret not in blob

    def test_span_hash_differs_for_different_values(self, gate: HardGate) -> None:
        a = gate.scan(f"card {CARD_VALID}").findings[0].span_hash
        b = gate.scan(f"card {CARD_VALID_MC}").findings[0].span_hash
        assert a != b

    def test_span_hash_is_stable_for_the_same_value(self, gate: HardGate) -> None:
        a = gate.scan(f"first {CARD_VALID} here").findings[0].span_hash
        b = gate.scan(f"second {CARD_VALID} there").findings[0].span_hash
        assert a == b, "the hash is of the matched span, not of the surrounding text"

    def test_redaction_map_excludes_advisory_detectors(self, gate: HardGate) -> None:
        """Redacting "HIPAA" out of a HIPAA question leaves a classifier nothing to read."""
        mapping = gate.redaction_map(f"Explain HIPAA for {EMAIL_INVENTED}")
        assert "email_address" in mapping
        assert "kw_health_regulation" not in mapping

    def test_redaction_map_spans_point_at_the_matched_text(self, gate: HardGate) -> None:
        text = f"mail {EMAIL_INVENTED} now"
        mapping = gate.redaction_map(text)
        for detector_name, spans in mapping.items():
            for start, end in spans:
                assert "@" in text[start:end], detector_name


# --------------------------------------------------------------------------- #
# Determinism and verdict algebra
# --------------------------------------------------------------------------- #
class TestDeterminism:
    MIXED = (
        f"Explain HIPAA, then charge {CARD_VALID}, wire {IBAN_VALID}, "
        f"notify {EMAIL_INVENTED}, call 020 7946 0958. Key: {KEY_DOTTED}"
    )

    def test_same_text_twice_gives_identical_verdicts(self, gate: HardGate) -> None:
        first = gate.scan(self.MIXED)
        second = gate.scan(self.MIXED)
        assert first == second
        assert first.to_dict() == second.to_dict()
        assert [f.span_hash for f in first.findings] == [f.span_hash for f in second.findings]

    def test_separate_gate_instances_agree(self) -> None:
        assert HardGate().scan(self.MIXED) == HardGate().scan(self.MIXED)

    def test_detectors_counts_sum_to_findings(self, gate: HardGate) -> None:
        verdict = gate.scan(self.MIXED)
        counts = verdict.detectors()
        assert sum(counts.values()) == sum(f.count for f in verdict.findings)
        assert set(counts) == names(verdict)

    def test_floors_take_the_strictest_finding(self, gate: HardGate) -> None:
        """Email is confidential, a card is regulated: the verdict must say regulated."""
        verdict = gate.scan(f"{EMAIL_INVENTED} and {CARD_VALID}")
        assert verdict.sensitivity_floor == "regulated"
        assert verdict.blocks_backend is True
        assert verdict.force_local is True

    def test_finding_fields_round_trip(self, gate: HardGate) -> None:
        verdict = gate.scan(CARD_VALID)
        restored = GateVerdict.from_dict(verdict.to_dict())
        assert restored == verdict
        assert restored.findings[0] == verdict.findings[0]

    def test_gate_finding_ignores_unknown_keys_on_restore(self) -> None:
        """Forward compatibility: a record written by a newer build must still load."""
        payload = {
            "detector": "payment_card",
            "category": "regulated",
            "sensitivity_floor": "regulated",
            "force_local": True,
            "span_hash": "0" * 16,
            "count": 1,
            "added_in_version_2": "ignored",
        }
        finding = GateFinding.from_dict(payload)
        assert finding.detector == "payment_card"
        assert not hasattr(finding, "added_in_version_2")

    def test_gate_is_reusable_across_many_scans(self, gate: HardGate) -> None:
        # Construction is the expensive part; a gate is built once per process.
        for i in range(50):
            text = CARD_VALID if i % 2 == 0 else "nothing to see here"
            verdict = gate.scan(f"request {i}: {text}")
            assert verdict.blocks_backend is (i % 2 == 0)
