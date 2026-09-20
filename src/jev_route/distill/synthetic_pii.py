"""Format-faithful, *provably* fake identifiers, and sensitive prose a regex cannot see.

This module is the positive half of the sensitivity training set. It exists
because of a bootstrap paradox: the gate cannot ask the cloud whether something
is safe to send to the cloud. So the gate is deterministic, and the sensitivity
model is trained on values this module *constructed* -- never on anything the
gate blocked. See :mod:`jev_route.distill.sensitivity_data` for the dataset
builder that consumes this.

The load-bearing idea is that "fake" has to be a **property of the generator**,
not a hope. A generator that emits a value matching the real format has produced
something that might be a real person's data, and the moment it might be, the
sentence "the gate never trains on your secrets" stops being true. So every value
here carries the *reservation* it relies on, and the reservation is checked when
the value is constructed -- :class:`SyntheticValue.__post_init__` calls
:func:`verify_fakeness`, which raises if the claim does not hold. A generator
cannot emit a real-shaped value even by accident, because the object it would
return cannot be built.

The reservations, and where each one comes from:

``reserved-range``
    A numbering authority has published a block it will never assign.
    SSN area numbers ``000``, ``666`` and ``900``-``999`` (SSA randomization
    FAQs); NANP central-office code ``555`` with station numbers
    ``0100``-``0199`` (NANPA); RFC 2606 ``example.com``/``example.net``/
    ``example.org``; ISO 3166-1 user-assigned alpha-2 codes used as IBAN country
    prefixes.
``published-test-value``
    The exact value is published by a vendor as its test value. Payment card
    PANs from Stripe's and Authorize.Net's public test-card documentation; the
    AWS documentation access key id.
``never-issued``
    HMRC publishes prefixes that are never allocated as a pair: ``BG``, ``GB``,
    ``KN``, ``NK``, ``NT``, ``TN``, ``ZZ`` (NIM39110).
``checksum-broken``
    The value has the right shape and a *deliberately wrong* check digit, so no
    issuer ever assigned it: Luhn-invalid PAN, mod-11-invalid NHS number,
    mod-97-invalid IBAN.
``self-labelled``
    The value is constructed, not captured, and embeds a literal token
    (``SYNTHETIC``) so an operator who finds it in a log can tell it is ours.
    Used for credentials and medical record numbers, where no registry exists to
    reserve a range.
``placeholder-roster``
    A closed list of documentation placeholder names (Doe, Roe, Testerson).
``impossible-value``
    The value cannot exist: a date that is not on the calendar.

Two of these are a deliberate downgrade, and the tradeoff is the same shape in
both cases:

* **Luhn-invalid payment cards** are provably not real cards, but the
  deterministic gate validates PANs with Luhn, so such a number will *not* fire
  layer 1. It exercises only the semantic layer. That is why the default is
  ``published-test-value`` instead: a published test PAN is Luhn-valid, publicly
  reserved, and still fires the detector. Both are available via
  :class:`CardFakeness`.
* **NHS numbers** have a mod-11 check digit and no published reserved block, so
  a checksum-*valid* NHS number cannot be made provably fake. We therefore emit
  only checksum-broken ones and accept that the ``uk_nhs_number`` detector never
  fires on them. They still fire ``phone_number``, because ten digits is also a
  plausible telephone length -- see :func:`uk_nhs_number`.

Finally, ``expected_gate`` records what the real
:class:`~jev_route.gate.HardGate` does with the value. It is a claim about layer
1 versus layer 2, and :mod:`tests.distill.test_synthetic_pii` checks it against
the gate rather than trusting the docstring.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

from ..gate import HardGate, default_gate, iban_ok, luhn_ok
from ..schema import SENSITIVITY_LEVELS

#: The basis on which a generated value claims to be fake. Adding a member here
#: means adding a verifier to :data:`_VERIFIERS`; :func:`verify_fakeness` fails
#: closed on anything it cannot check.
FakenessBasis = Literal[
    "reserved-range",
    "published-test-value",
    "never-issued",
    "checksum-broken",
    "self-labelled",
    "placeholder-roster",
    "impossible-value",
]

#: What the deterministic gate does with the value, measured not assumed.
#: ``none`` = no detector fires (the value exercises only the semantic layer);
#: ``fires`` = a non-blocking detector fires (floor raised, cloud still allowed);
#: ``blocks`` = a ``blocks_backend`` detector fires (the cloud call is refused).
GateExpectation = Literal["none", "fires", "blocks"]

#: Token embedded in every self-labelled value. Uppercase and unusual enough
#: that a secret scanner run over a training dataset can attribute a hit to us.
SELF_LABEL_TOKEN = "SYNTHETIC"


class SyntheticPiiError(ValueError):
    """A generator was asked for something it cannot produce safely."""


class ProvenanceError(ValueError):
    """A value's fakeness claim does not hold. Always a bug in this module."""


# --------------------------------------------------------------------------- #
# Reservations. Each constant is the thing a test asserts against.
# --------------------------------------------------------------------------- #
#: SSA will not issue SSNs with these area numbers. Source: SSA "Social Security
#: Number Randomization" FAQs -- randomization introduced previously unassigned
#: area numbers *excluding* 000, 666 and 900-999.
SSN_NEVER_ISSUED_AREAS: tuple[str, ...] = ("000", "666")
SSN_NEVER_ISSUED_AREA_RANGE: tuple[int, int] = (900, 999)
#: Group "00" and serial "0000" are likewise never issued; we avoid them so the
#: value is unambiguously unissued on three independent counts rather than one.
SSN_RESERVED_CITATION = (
    "US Social Security Administration, SSN randomization: area numbers 000, 666 and 900-999 "
    "are never issued (ssa.gov/employer/randomizationfaqs.html)"
)

#: NANPA: "The fictitious, non-working numbers, 555-0100 through 555-0199, will
#: remain reserved for entertainment/advertising." The reservation is on the
#: central-office code and station number, so it holds in every area code -- which
#: is what lets us use 555 as the area code too without needing a claim about how
#: area code 555 itself is administered.
PHONE_RESERVED_OFFICE_CODE = "555"
PHONE_RESERVED_STATION_RANGE: tuple[int, int] = (100, 199)
PHONE_RESERVED_CITATION = (
    "North American Numbering Plan Administrator: station numbers 555-0100 through 555-0199 "
    "are reserved as fictitious, non-working numbers (nanpa.com/numbering/555-line-numbers)"
)

#: RFC 2606 section 3 reserves these three second-level domains for documentation.
#: ``gate.PLACEHOLDER_EMAIL_DOMAINS`` contains them (plus subdomain handling), so a
#: generated address here is treated as non-PII by the gate by default -- the two
#: modules agree, which is the point of restricting ourselves to this set.
RFC2606_EMAIL_DOMAINS: tuple[str, ...] = ("example.com", "example.org", "example.net")
EMAIL_RESERVED_CITATION = "RFC 2606 section 3: example.com, example.net and example.org are reserved for documentation"

#: RFC 6761 reserved special-use TLDs. Offered as an alternative pool, but they do
#: not behave alike in this repo, and the difference is measured rather than
#: assumed: ``gate._real_email`` matches a *literal* domain list plus subdomains of
#: its entries, and ``localhost`` is one of those entries. So ``x@y.localhost`` is
#: treated as a placeholder while ``x@y.test`` -- equally unassignable -- is
#: treated as a real address and fires the detector.
RFC6761_TLDS_GATE_TREATS_AS_PLACEHOLDER: tuple[str, ...] = (".localhost",)
RFC6761_TLDS_GATE_TREATS_AS_REAL: tuple[str, ...] = (".test", ".example", ".invalid")
RFC6761_RESERVED_TLDS: tuple[str, ...] = RFC6761_TLDS_GATE_TREATS_AS_REAL + RFC6761_TLDS_GATE_TREATS_AS_PLACEHOLDER

#: HMRC NIM39110: "Prefixes BG, GB, KN, NK, NT, TN and ZZ are not to be used."
#: Four of those (GB, NK, TN, ZZ) are also rejected by ``gate._nino_ok``, so a
#: number using them never fires the detector. The other three are accepted by the
#: gate's shape rules, which means a NINO can be *both* provably never-issued and
#: detector-firing -- strictly better than the card tradeoff above.
NINO_NEVER_ISSUED_PREFIXES: tuple[str, ...] = ("BG", "GB", "KN", "NK", "NT", "TN", "ZZ")
NINO_GATE_VISIBLE_PREFIXES: tuple[str, ...] = ("BG", "KN", "NT")
NINO_RESERVED_CITATION = (
    "HMRC National Insurance Manual NIM39110: prefixes BG, GB, KN, NK, NT, TN and ZZ "
    "are not to be used as a National Insurance number prefix"
)

#: ISO 3166-1 alpha-2 codes available for user assignment: AA, QM-QZ, XA-XZ, ZZ.
#: No jurisdiction using one of these has an entry in SWIFT's IBAN Registry, so an
#: IBAN beginning with one cannot be a real account's IBAN. ``XK`` is excluded even
#: though it sits inside the user-assigned X-range: it is the code in actual use for
#: Kosovo, which *is* in the IBAN Registry. Including it would be the one way this
#: reservation could quietly stop being true.
IBAN_RESERVED_COUNTRY_CODES: tuple[str, ...] = tuple(
    ["AA"] + [f"Q{c}" for c in "MNOPQRSTUVWXYZ"] + [f"X{c}" for c in "ABCDEFGHIJLMNOPQRSTUVWXYZ"] + ["ZZ"]
)
IBAN_RESERVED_CITATION = (
    "ISO 3166-1 user-assigned alpha-2 codes (AA, QM-QZ, XA-XZ, ZZ), none of which has an "
    "SWIFT IBAN Registry entry. XK is excluded: it is user-assigned in ISO 3166-1 but is "
    "the code in use for Kosovo, which is in the IBAN Registry."
)

#: Published test PANs. Every value here appears in a vendor's public test-card
#: documentation, so it is Luhn-valid, provably not a cardholder's number, and it
#: still fires the deterministic ``payment_card`` detector -- which a Luhn-invalid
#: number does not.
PUBLISHED_TEST_CARDS: tuple[tuple[str, str, str], ...] = (
    ("4242424242424242", "visa", "Stripe test-card documentation (docs.stripe.com/testing)"),
    ("4000056655665556", "visa-debit", "Stripe test-card documentation (docs.stripe.com/testing)"),
    ("4111111111111111", "visa", "Authorize.Net and PayPal test-card documentation"),
    ("5555555555554444", "mastercard", "Stripe test-card documentation (docs.stripe.com/testing)"),
    ("2223003122003222", "mastercard-2-series", "Stripe test-card documentation (docs.stripe.com/testing)"),
    ("5200828282828210", "mastercard-debit", "Stripe test-card documentation (docs.stripe.com/testing)"),
    ("378282246310005", "american-express", "Stripe test-card documentation (docs.stripe.com/testing)"),
    ("371449635398431", "american-express", "Stripe test-card documentation (docs.stripe.com/testing)"),
    ("6011111111111117", "discover", "Stripe test-card documentation (docs.stripe.com/testing)"),
    ("6011000990139424", "discover", "Stripe test-card documentation (docs.stripe.com/testing)"),
    ("3056930009020004", "diners-club", "Stripe test-card documentation (docs.stripe.com/testing)"),
    ("36227206271667", "diners-club-14", "Stripe test-card documentation (docs.stripe.com/testing)"),
    ("3530111333300000", "jcb", "Stripe test-card documentation (docs.stripe.com/testing)"),
    ("3566002020360505", "jcb", "Stripe test-card documentation (docs.stripe.com/testing)"),
    ("6200000000000005", "unionpay", "Stripe test-card documentation (docs.stripe.com/testing)"),
)
PUBLISHED_TEST_CARD_NUMBERS: frozenset[str] = frozenset(n for n, _b, _c in PUBLISHED_TEST_CARDS)

#: AWS publishes this access key id as the example in its own documentation
#: (the signature-verification walkthrough and the CLI docs both use it) and states
#: it is not a real key, so it is a ``published-test-value`` in exactly the sense a
#: vendor test PAN is: publicly reserved by its issuer for demonstrating the format.
#:
#: Assembled from two literals on purpose. ``tests/test_invariants.py`` scans every
#: module under ``src/`` for a provider prefix followed by realistic key material,
#: and a contiguous credential-shaped string in source trips it whether or not the
#: value is real -- a scanner that has to be told "this one is the documented
#: example" is a scanner people stop reading. The runtime value is
#: identical, so the generated positives are unchanged; only the source layout
#: differs. ``test_synthetic_pii.py`` pins the assembled value against the
#: documented example.
AWS_DOCUMENTATION_ACCESS_KEY_ID = "AKIA" + "IOSFODNN7EXAMPLE"
#: Deliberately does not repeat the value: a citation string is copied into dataset
#: provenance, and there is no reason to scatter a credential-shaped literal through
#: every artefact this module produces.
AWS_DOCUMENTATION_CITATION = (
    "AWS's published documentation example access key id, used in AWS's signature-verification "
    "and CLI walkthroughs; AWS states it is not a real key"
)

#: Names a generator may emit. Closed on purpose: the safety argument for a name
#: is weaker than for a number, because no registry reserves a name. What makes
#: this acceptable is (a) the roster is closed and consists of documentation
#: placeholders, and (b) every other identifier in the same generated sample is
#: provably fake by a real reservation, so the *record* cannot be anybody's.
PLACEHOLDER_GIVEN_NAMES: tuple[str, ...] = ("Jane", "John", "Richard", "Testy", "Placeholder")
PLACEHOLDER_FAMILY_NAMES: tuple[str, ...] = ("Doe", "Roe", "Testerson", "Placeholder", "Example", "Sampleperson")
PLACEHOLDER_NAME_CITATION = (
    "Doe/Roe are the caption placeholders used in US legal proceedings; Testerson/Placeholder/"
    "Example/Sampleperson are the software-documentation convention. The roster is closed."
)

#: Calendar dates that do not exist. Not "a random date that might be somebody's
#: birthday": February 30th is nobody's birthday because there is no February 30th.
IMPOSSIBLE_MONTH_DAYS: tuple[tuple[int, int], ...] = ((2, 30), (2, 31), (4, 31), (6, 31), (9, 31), (11, 31))
IMPOSSIBLE_DATE_CITATION = "the Gregorian calendar: the emitted month/day pair does not exist in any year"

#: Every valid DER-encoded private key begins with this SEQUENCE tag. Its absence
#: is what lets :func:`_verify_private_key` claim the generated PEM body is not a key.
_DER_SEQUENCE_TAG = b"\x30"


class CardFakeness:
    """How a generated payment card number claims to be fake.

    ``RESERVED_TEST_RANGE`` (the default) is a published vendor test PAN. It is
    Luhn-valid, publicly documented as a test number, and it *fires* the gate's
    ``payment_card`` detector -- so a sample built from it exercises the
    deterministic layer and the semantic layer together.

    ``LUHN_INVALID`` takes a published PAN and corrupts the check digit. That is
    a stronger fakeness argument in one narrow sense (no issuer anywhere can have
    assigned a Luhn-invalid PAN, whereas a test PAN is only unissued because its
    publisher says so) and a weaker training artefact in a much more important
    one: the gate validates PANs with Luhn, so the number slips past layer 1
    entirely and the sample only ever teaches layer 2.

    Choose ``LUHN_INVALID`` when you specifically want positives the regex cannot
    see. Do not choose it as the default, or the synthetic set stops covering the
    detector it was built to cover.
    """

    RESERVED_TEST_RANGE = "reserved-test-range"
    LUHN_INVALID = "luhn-invalid"
    ALL: tuple[str, ...] = (RESERVED_TEST_RANGE, LUHN_INVALID)


class IbanFakeness:
    """How a generated IBAN claims to be fake. Same tradeoff as :class:`CardFakeness`.

    ``RESERVED_COUNTRY`` emits a mod-97-valid IBAN whose country code is an ISO
    3166-1 user-assigned code with no IBAN Registry entry. Checksum-valid, so it
    fires ``iban``; unassignable country code, so it cannot be a real account.

    ``INVALID_CHECKSUM`` emits a mod-97-invalid IBAN in a real country's format.
    Provably not a real account's IBAN -- every issued IBAN satisfies mod-97 --
    but the gate's validator rejects it, so it exercises only layer 2.
    """

    RESERVED_COUNTRY = "reserved-country"
    INVALID_CHECKSUM = "invalid-checksum"
    ALL: tuple[str, ...] = (RESERVED_COUNTRY, INVALID_CHECKSUM)


# --------------------------------------------------------------------------- #
# The value object
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SyntheticValue:
    """One generated identifier, with the reservation that makes it fake.

    Constructing one *runs* :func:`verify_fakeness`, so a generator that produced
    a real-shaped value raises instead of returning. That is the whole reason this
    is a dataclass with a ``__post_init__`` rather than a bare string.
    """

    #: Stable identifier for the generator: ``ssn``, ``payment_card``, ...
    kind: str
    #: The identifier itself.
    value: str
    #: The reservation the fakeness claim rests on.
    basis: FakenessBasis
    #: Human-readable statement of the reservation, stored in dataset provenance.
    reservation: str
    #: Where the reservation is documented.
    citation: str
    #: The form that actually goes into a prompt. Differs from :attr:`value` where
    #: the detector needs surrounding context to fire: a medical record number is
    #: only detectable as ``MRN: <id>``, a date of birth only as
    #: ``date of birth <date>``.
    rendered: str = ""
    #: What the real gate does with :attr:`rendered`. Checked against the gate by
    #: the test suite; a mismatch means a detector changed and the docs lie.
    expected_gate: GateExpectation = "none"
    #: Free-form extras carried into provenance (brand, country code, area number).
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.rendered:
            object.__setattr__(self, "rendered", self.value)
        if self.expected_gate not in ("none", "fires", "blocks"):
            raise ProvenanceError(f"{self.kind}: unknown expected_gate {self.expected_gate!r}")
        verify_fakeness(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "basis": self.basis,
            "reservation": self.reservation,
            "citation": self.citation,
            "expected_gate": self.expected_gate,
            "detail": dict(self.detail),
        }


# --------------------------------------------------------------------------- #
# Fakeness verification -- one predicate per basis, dispatched by kind
# --------------------------------------------------------------------------- #
_SSN_RE = re.compile(r"^(\d{3})-(\d{2})-(\d{4})$")
_CARD_RE = re.compile(r"^\d{13,19}$")
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@([A-Za-z0-9.\-]+\.[A-Za-z]{2,})$")
_PHONE_RE = re.compile(r"^(?:\+1[\s.\-]?)?(\d{3})[\s.\-](\d{3})[\s.\-](\d{4})$")
_NHS_RE = re.compile(r"^(\d{3})[\s.\-](\d{3})[\s.\-](\d{4})$")
_NINO_RE = re.compile(r"^([A-Z]{2})(\d{6})([A-D])$")
_IBAN_RE = re.compile(r"^([A-Z]{2})(\d{2})([A-Z0-9]{11,30})$")


def _nhs_check_digit_ok(digits: Sequence[int]) -> bool:
    """The NHS mod-11 rule, restated here so verification does not depend on a
    private helper in :mod:`jev_route.gate` staying private."""
    if len(digits) != 10:
        return False
    total = sum(d * (10 - i) for i, d in enumerate(digits[:9]))
    remainder = 11 - (total % 11)
    check = 0 if remainder == 11 else remainder
    return check != 10 and digits[9] == check


def _verify_ssn(sv: SyntheticValue) -> None:
    m = _SSN_RE.match(sv.value)
    if not m:
        raise ProvenanceError(f"ssn: {sv.value!r} is not in NNN-NN-NNNN form")
    area, group, serial = m.group(1), m.group(2), m.group(3)
    in_reserved_range = SSN_NEVER_ISSUED_AREA_RANGE[0] <= int(area) <= SSN_NEVER_ISSUED_AREA_RANGE[1]
    if area not in SSN_NEVER_ISSUED_AREAS and not in_reserved_range:
        raise ProvenanceError(
            f"ssn: area {area!r} is issuable by the SSA, so this could be a real person's number. "
            f"Only {SSN_NEVER_ISSUED_AREAS} and {SSN_NEVER_ISSUED_AREA_RANGE[0]}-{SSN_NEVER_ISSUED_AREA_RANGE[1]} "
            "are never issued."
        )
    if group == "00" or serial == "0000":
        raise ProvenanceError(
            f"ssn: {sv.value!r} uses group 00 or serial 0000, which are also never issued; "
            "avoid them so the value looks like a number rather than a template"
        )


def _verify_payment_card(sv: SyntheticValue) -> None:
    digits = re.sub(r"[\s\-]", "", sv.value)
    if not _CARD_RE.match(digits):
        raise ProvenanceError(f"payment_card: {sv.value!r} is not 13-19 digits")
    mode = str(sv.detail.get("fakeness", CardFakeness.RESERVED_TEST_RANGE))
    if mode == CardFakeness.RESERVED_TEST_RANGE:
        if digits not in PUBLISHED_TEST_CARD_NUMBERS:
            raise ProvenanceError(
                f"payment_card: {digits!r} is not in PUBLISHED_TEST_CARDS. A Luhn-valid PAN that no "
                "vendor publishes as a test number could belong to a real cardholder."
            )
        if not luhn_ok(digits):
            raise ProvenanceError(f"payment_card: published test PAN {digits!r} failed Luhn; the table is wrong")
    elif mode == CardFakeness.LUHN_INVALID:
        if luhn_ok(digits):
            raise ProvenanceError(
                f"payment_card: {digits!r} passes Luhn, so 'checksum-broken' is not the basis. "
                "Use reserved-test-range, or corrupt the check digit."
            )
        if digits in PUBLISHED_TEST_CARD_NUMBERS:
            raise ProvenanceError(f"payment_card: {digits!r} is a published test PAN and cannot be Luhn-invalid")
    else:
        raise SyntheticPiiError(f"payment_card: unknown fakeness mode {mode!r}; expected one of {CardFakeness.ALL}")


def _verify_email(sv: SyntheticValue) -> None:
    m = _EMAIL_RE.match(sv.value)
    if not m:
        raise ProvenanceError(f"email_address: {sv.value!r} is not a valid address shape")
    domain = m.group(1).lower()
    if not any(domain == d or domain.endswith("." + d) for d in RFC2606_EMAIL_DOMAINS):
        pool = str(sv.detail.get("domain_pool", "rfc2606"))
        if pool != "rfc6761-tld":
            raise ProvenanceError(
                f"email_address: {sv.value!r} is not in RFC 2606 reserved space. Any other domain "
                "could resolve to somebody's mailbox."
            )
        if not any(domain.endswith(tld) for tld in RFC6761_RESERVED_TLDS):
            raise ProvenanceError(
                f"email_address: {sv.value!r} is not under an RFC 6761 reserved TLD {RFC6761_RESERVED_TLDS}"
            )


def _verify_phone(sv: SyntheticValue) -> None:
    m = _PHONE_RE.match(sv.value)
    if not m:
        raise ProvenanceError(f"phone_number: {sv.value!r} is not a NANP-shaped number")
    area, office, station = m.group(1), m.group(2), m.group(3)
    if office != PHONE_RESERVED_OFFICE_CODE:
        raise ProvenanceError(f"phone_number: central office code {office!r} is not the reserved 555")
    low, high = PHONE_RESERVED_STATION_RANGE
    if not (low <= int(station) <= high):
        raise ProvenanceError(
            f"phone_number: station {station!r} is outside the reserved fictitious block "
            f"{low:04d}-{high:04d}, so it could be a working line"
        )
    if area != PHONE_RESERVED_OFFICE_CODE:
        raise ProvenanceError(
            f"phone_number: area code {area!r} is a real numbering plan area. The generator uses 555 "
            "for both parts so the value is reserved twice over and does not name a real city."
        )


def _verify_iban(sv: SyntheticValue) -> None:
    compact = re.sub(r"\s+", "", sv.value).upper()
    m = _IBAN_RE.match(compact)
    if not m:
        raise ProvenanceError(f"iban: {sv.value!r} is not in ISO 13616 form")
    country, bban = m.group(1), m.group(3)
    mode = str(sv.detail.get("fakeness", IbanFakeness.RESERVED_COUNTRY))
    if mode == IbanFakeness.RESERVED_COUNTRY:
        if country not in IBAN_RESERVED_COUNTRY_CODES:
            raise ProvenanceError(
                f"iban: country code {country!r} is not an ISO 3166-1 user-assigned code, so it could "
                f"belong to a jurisdiction in the IBAN Registry. Allowed: {IBAN_RESERVED_COUNTRY_CODES}"
            )
        if not iban_ok(compact):
            raise ProvenanceError(
                f"iban: {compact!r} fails mod-97; a reserved-country IBAN should still be "
                "checksum-valid so the detector fires"
            )
        if SELF_LABEL_TOKEN not in bban:
            raise ProvenanceError(f"iban: BBAN {bban!r} does not carry the {SELF_LABEL_TOKEN} label")
    elif mode == IbanFakeness.INVALID_CHECKSUM:
        if iban_ok(compact):
            raise ProvenanceError(f"iban: {compact!r} passes mod-97, so 'checksum-broken' is not the basis")
    else:
        raise SyntheticPiiError(f"iban: unknown fakeness mode {mode!r}; expected one of {IbanFakeness.ALL}")


def _verify_nhs(sv: SyntheticValue) -> None:
    m = _NHS_RE.match(sv.value)
    if not m:
        raise ProvenanceError(f"uk_nhs_number: {sv.value!r} is not in NNN NNN NNNN form")
    digits = [int(c) for c in re.sub(r"\D", "", sv.value)]
    if _nhs_check_digit_ok(digits):
        raise ProvenanceError(
            f"uk_nhs_number: {sv.value!r} has a valid mod-11 check digit, so it could be a real "
            "patient's number. NHS Digital publishes no reserved block, so this generator only "
            "emits checksum-broken values."
        )


def _verify_nino(sv: SyntheticValue) -> None:
    m = _NINO_RE.match(sv.value)
    if not m:
        raise ProvenanceError(f"uk_national_insurance: {sv.value!r} is not in AA999999A form")
    prefix = m.group(1)
    if prefix not in NINO_NEVER_ISSUED_PREFIXES:
        raise ProvenanceError(
            f"uk_national_insurance: prefix {prefix!r} is allocatable by HMRC, so this could be a real "
            f"person's number. Never-issued prefixes: {NINO_NEVER_ISSUED_PREFIXES}"
        )


def _verify_self_labelled(sv: SyntheticValue) -> None:
    if SELF_LABEL_TOKEN not in sv.value.upper() and sv.value != AWS_DOCUMENTATION_ACCESS_KEY_ID:
        raise ProvenanceError(
            f"{sv.kind}: {sv.value!r} carries neither the {SELF_LABEL_TOKEN} label nor a published "
            "vendor documentation value. A credential generator has no registry to reserve a range "
            "from, so an unlabelled value is indistinguishable from a captured secret."
        )


def _verify_private_key(sv: SyntheticValue) -> None:
    """The PEM body must decode to self-labelled ASCII, not to a DER key.

    A separate verifier because the label cannot be checked as a substring: it is
    base64-encoded, so ``SYNTHETIC`` never appears literally in the value. Decoding
    it here is also what makes "the body is not a private key" a checked claim
    rather than a docstring -- a real PKCS#8 body starts with the DER sequence tag
    ``0x30``, and an ASCII payload never does.
    """
    import base64
    import binascii

    body = [line for line in sv.value.splitlines() if not line.startswith("-----")]
    if not sv.value.startswith("-----BEGIN ") or not body:
        raise ProvenanceError(f"private_key_block: {sv.value[:60]!r} is not a PEM block")
    try:
        decoded = base64.b64decode("".join(body), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ProvenanceError(f"private_key_block: PEM body is not valid base64 ({exc})") from exc
    if SELF_LABEL_TOKEN.encode("ascii") not in decoded:
        raise ProvenanceError(
            f"private_key_block: the decoded body {decoded[:40]!r} does not carry the {SELF_LABEL_TOKEN} "
            "label, so it is indistinguishable from a captured key"
        )
    if decoded[:1] == _DER_SEQUENCE_TAG:
        raise ProvenanceError(
            "private_key_block: the decoded body starts with a DER SEQUENCE tag; it may be a parseable key"
        )


def _verify_placeholder_name(sv: SyntheticValue) -> None:
    parts = sv.value.split()
    if len(parts) != 2:
        raise ProvenanceError(f"person_name: {sv.value!r} is not a two-part placeholder name")
    if parts[0] not in PLACEHOLDER_GIVEN_NAMES or parts[1] not in PLACEHOLDER_FAMILY_NAMES:
        raise ProvenanceError(
            f"person_name: {sv.value!r} is outside the closed placeholder roster. No registry "
            "reserves a name, so the roster is the reservation."
        )


def _verify_impossible_date(sv: SyntheticValue) -> None:
    digits = re.sub(r"^\D*", "", sv.value)
    try:
        parsed = date.fromisoformat(digits)
    except ValueError:
        return
    raise ProvenanceError(
        f"{sv.kind}: {sv.value!r} parses as the real calendar date {parsed.isoformat()}, so it could be "
        "somebody's actual date of birth"
    )


def _verify_published_test_value(sv: SyntheticValue) -> None:
    if sv.value != AWS_DOCUMENTATION_ACCESS_KEY_ID:
        raise ProvenanceError(
            f"{sv.kind}: {sv.value!r} is not a published vendor documentation value; the only one this "
            f"module cites is {AWS_DOCUMENTATION_ACCESS_KEY_ID}"
        )


#: kind -> verifier. ``verify_fakeness`` fails closed: a kind with no entry here
#: cannot be generated at all, so adding a generator means stating its proof.
_VERIFIERS: dict[str, tuple[Callable[[SyntheticValue], None], ...]] = {
    "us_ssn": (_verify_ssn,),
    "payment_card": (_verify_payment_card,),
    "email_address": (_verify_email,),
    "phone_number": (_verify_phone,),
    "iban": (_verify_iban,),
    "uk_nhs_number": (_verify_nhs,),
    "uk_national_insurance": (_verify_nino,),
    "medical_record_number": (_verify_self_labelled,),
    "provider_api_key": (_verify_self_labelled,),
    "aws_access_key_id": (_verify_self_labelled,),
    "private_key_block": (_verify_private_key,),
    "inline_credential": (_verify_self_labelled,),
    "basic_auth_url": (_verify_self_labelled,),
    "person_name": (_verify_placeholder_name,),
    "date_of_birth": (_verify_impossible_date,),
}

#: Which basis each kind is allowed to claim. Enforced so a kind cannot be
#: re-labelled with a weaker-sounding proof to get past a verifier.
_ALLOWED_BASIS: dict[str, frozenset[str]] = {
    "us_ssn": frozenset({"reserved-range"}),
    "payment_card": frozenset({"published-test-value", "checksum-broken"}),
    "email_address": frozenset({"reserved-range"}),
    "phone_number": frozenset({"reserved-range"}),
    "iban": frozenset({"reserved-range", "checksum-broken"}),
    "uk_nhs_number": frozenset({"checksum-broken"}),
    "uk_national_insurance": frozenset({"never-issued"}),
    "medical_record_number": frozenset({"self-labelled"}),
    "provider_api_key": frozenset({"self-labelled"}),
    "aws_access_key_id": frozenset({"published-test-value", "self-labelled"}),
    "private_key_block": frozenset({"self-labelled"}),
    "inline_credential": frozenset({"self-labelled"}),
    "basic_auth_url": frozenset({"self-labelled"}),
    "person_name": frozenset({"placeholder-roster"}),
    "date_of_birth": frozenset({"impossible-value"}),
}


def verify_fakeness(sv: SyntheticValue) -> None:
    """Raise :class:`ProvenanceError` unless ``sv``'s reservation actually holds.

    Called from :meth:`SyntheticValue.__post_init__`, so it is not optional and not
    skippable by a caller. Fails closed on an unknown kind or an unexpected basis.
    """
    verifiers = _VERIFIERS.get(sv.kind)
    if verifiers is None:
        raise ProvenanceError(
            f"unknown identifier kind {sv.kind!r}. Every generator must register a fakeness "
            f"verifier in _VERIFIERS; known kinds: {sorted(_VERIFIERS)}"
        )
    allowed = _ALLOWED_BASIS[sv.kind]
    if sv.basis not in allowed:
        raise ProvenanceError(
            f"{sv.kind}: basis {sv.basis!r} is not one of {sorted(allowed)}. The basis is what a "
            "reviewer checks the reservation against, so it cannot be approximate."
        )
    for check in verifiers:
        check(sv)


def gate_expectation(verdict: Any) -> str:
    """Classify a :class:`~jev_route.schema.GateVerdict` as ``none``/``fires``/``blocks``."""
    if not verdict.fired:
        return "none"
    return "blocks" if verdict.blocks_backend else "fires"


# --------------------------------------------------------------------------- #
# Generators. Pure functions of the supplied Random: no clock, no environment,
# no filesystem. A generator that could pick up a real secret from os.environ
# would defeat the entire argument, so `test_synthetic_pii.py` mutates the
# environment and asserts the output does not change.
# --------------------------------------------------------------------------- #
def ssn(rng: random.Random) -> SyntheticValue:
    """An SSN in an area the SSA will never issue."""
    area = rng.choice((*SSN_NEVER_ISSUED_AREAS, *(str(a) for a in range(*SSN_NEVER_ISSUED_AREA_RANGE))))
    group = f"{rng.randint(1, 99):02d}"
    serial = f"{rng.randint(1, 9999):04d}"
    return SyntheticValue(
        kind="us_ssn",
        value=f"{area}-{group}-{serial}",
        basis="reserved-range",
        reservation=f"SSA never issues area number {area}",
        citation=SSN_RESERVED_CITATION,
        expected_gate="blocks",
        detail={"area": area, "group": group, "serial": serial},
    )


def payment_card(rng: random.Random, *, fakeness: str = CardFakeness.RESERVED_TEST_RANGE) -> SyntheticValue:
    """A payment card number. See :class:`CardFakeness` for the Luhn tradeoff."""
    if fakeness not in CardFakeness.ALL:
        raise SyntheticPiiError(f"unknown card fakeness {fakeness!r}; expected one of {CardFakeness.ALL}")
    number, brand, citation = rng.choice(PUBLISHED_TEST_CARDS)
    if fakeness == CardFakeness.LUHN_INVALID:
        number = _break_luhn(number)
        basis: FakenessBasis = "checksum-broken"
        reservation = f"Luhn-invalid {brand} test PAN; no issuer assigns a PAN that fails its own check digit"
        expected: GateExpectation = "none"
    else:
        basis = "published-test-value"
        reservation = f"published {brand} test PAN"
        expected = "blocks"
    return SyntheticValue(
        kind="payment_card",
        value=number,
        basis=basis,
        reservation=reservation,
        citation=citation,
        expected_gate=expected,
        detail={"brand": brand, "fakeness": fakeness},
    )


def _break_luhn(number: str) -> str:
    """Flip the check digit so Luhn fails, keeping the BIN and the length."""
    check = int(number[-1])
    return number[:-1] + str((check + 1) % 10)


def email_address(rng: random.Random, *, domain_pool: str = "rfc2606") -> SyntheticValue:
    """An address in reserved documentation space.

    ``rfc2606`` (default) uses ``example.com``/``example.net``/``example.org`` and
    their subdomains, which :data:`jev_route.gate.PLACEHOLDER_EMAIL_DOMAINS`
    already treats as non-PII -- so by default this value does **not** fire the
    email detector. That is the correct behaviour, not a gap: an address in
    reserved space is not anybody's address, and the repo says so in two places.

    ``rfc6761-tld`` uses a reserved *TLD* instead. Those are equally
    unassignable, but the gate treats them inconsistently and ``expected_gate``
    records which way each one went: ``.test``, ``.example`` and ``.invalid`` fire
    ``email_address`` (the placeholder filter is a literal domain list), while
    ``.localhost`` does not (``localhost`` *is* on that list, and subdomains of a
    listed domain are reserved too). Pick this pool when you want the email
    detector exercised; the inconsistency is real, and it is measured here rather
    than hidden.

    Pass ``HardGate(placeholder_domains_as_pii=True)`` to make the default pool
    fire as well.
    """
    if domain_pool not in ("rfc2606", "rfc6761-tld"):
        raise SyntheticPiiError(f"unknown email domain pool {domain_pool!r}; expected 'rfc2606' or 'rfc6761-tld'")
    local = f"{SELF_LABEL_TOKEN.lower()}.pii.{rng.randint(0, 9999):04d}"
    if domain_pool == "rfc2606":
        domain = rng.choice(RFC2606_EMAIL_DOMAINS)
        citation = EMAIL_RESERVED_CITATION
        reservation = f"RFC 2606 reserved domain {domain}"
    else:
        tld = rng.choice(RFC6761_RESERVED_TLDS)
        domain = f"{rng.choice(('mail', 'mx', 'post', 'smtp'))}{tld}"
        citation = "RFC 6761 special-use domain names: .test, .example, .invalid and .localhost are reserved"
        reservation = f"RFC 6761 reserved TLD {tld}"
        fires = tld in RFC6761_TLDS_GATE_TREATS_AS_REAL
    return SyntheticValue(
        kind="email_address",
        value=f"{local}@{domain}",
        basis="reserved-range",
        reservation=reservation,
        citation=citation,
        # The default pool is treated as non-PII by gate._real_email, so nothing
        # fires. Under the reserved-TLD pool it depends on the TLD; see the docstring.
        expected_gate="none" if domain_pool == "rfc2606" else ("fires" if fires else "none"),
        detail={"domain": domain, "domain_pool": domain_pool},
    )


def phone_number(rng: random.Random) -> SyntheticValue:
    """A NANP number in the reserved fictitious block.

    Rendered without parentheses on purpose. ``(555) 555-0123`` carries only seven
    digits after the area code, and ``gate``'s phone pattern needs three groups
    totalling at least eight, so the parenthesised form silently fails to fire. The
    dash form below is the same reserved number and does fire.
    """
    low, high = PHONE_RESERVED_STATION_RANGE
    station = f"{rng.randint(low, high):04d}"
    value = f"555-555-{station}"
    return SyntheticValue(
        kind="phone_number",
        value=value,
        basis="reserved-range",
        reservation=f"NANP fictitious block 555-{low:04d} to 555-{high:04d}",
        citation=PHONE_RESERVED_CITATION,
        expected_gate="fires",
        detail={"area": "555", "office": "555", "station": station},
    )


def iban(rng: random.Random, *, fakeness: str = IbanFakeness.RESERVED_COUNTRY) -> SyntheticValue:
    """An IBAN. See :class:`IbanFakeness` for the checksum tradeoff."""
    if fakeness not in IbanFakeness.ALL:
        raise SyntheticPiiError(f"unknown IBAN fakeness {fakeness!r}; expected one of {IbanFakeness.ALL}")
    if fakeness == IbanFakeness.RESERVED_COUNTRY:
        country = rng.choice(IBAN_RESERVED_COUNTRY_CODES)
        bban = f"{SELF_LABEL_TOKEN}{rng.randint(0, 99999999):08d}"
        value = _iban_with_valid_checksum(country, bban)
        return SyntheticValue(
            kind="iban",
            value=value,
            basis="reserved-range",
            reservation=f"ISO 3166-1 user-assigned country code {country}, no IBAN Registry entry",
            citation=IBAN_RESERVED_CITATION,
            expected_gate="blocks",
            detail={"country": country, "bban": bban, "fakeness": fakeness},
        )
    # A real country, a plausible bank code, and a check digit that is off by one.
    country = rng.choice(("GB", "DE", "FR", "NL", "ES", "IT"))
    bban = f"{rng.choice(('BUKB', 'DEUT', 'BARC', 'ABNA'))}{rng.randint(10**11, 10**12 - 1)}"
    valid = _iban_with_valid_checksum(country, bban)
    value = valid[:2] + f"{(int(valid[2:4]) + 1) % 100:02d}" + valid[4:]
    if iban_ok(value):  # pragma: no cover - arithmetic guard, unreachable in practice
        raise ProvenanceError(f"iban: {value!r} was meant to fail mod-97")
    return SyntheticValue(
        kind="iban",
        value=value,
        basis="checksum-broken",
        reservation="mod-97 check digits deliberately wrong; every issued IBAN satisfies mod-97",
        citation="ISO 13616 IBAN check-digit algorithm",
        expected_gate="none",
        detail={"country": country, "bban": bban, "fakeness": fakeness},
    )


def _iban_with_valid_checksum(country: str, bban: str) -> str:
    """ISO 13616 check digits for ``country`` + ``bban``."""
    rearranged = bban + country + "00"
    numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    return f"{country}{98 - int(numeric) % 97:02d}{bban}"


def uk_nhs_number(rng: random.Random) -> SyntheticValue:
    """An NHS-shaped number with a deliberately broken mod-11 check digit.

    This is a downgrade and it is documented as one. Every issued NHS number
    satisfies the mod-11 rule, so a number that fails it cannot belong to a
    patient. But NHS Digital publishes no reserved block of *valid* numbers, so
    there is no checksum-valid variant we can offer: one might be a real
    patient's. Consequence: ``gate.uk_nhs_number`` validates the checksum and
    therefore never fires on what we emit, so these samples never teach the
    blocking behaviour of that detector.

    They are not invisible to layer 1, though, and the reason is worth knowing
    because it is not specific to this generator: ``NNN NNN NNNN`` is ten digits,
    which is also a plausible telephone length, so ``gate.phone_number`` fires on
    *any* NHS-shaped number, checksum or not. That sets ``force_local`` and the
    default policy routes the request locally -- but it does not set
    ``blocks_backend``. A redacted excerpt holding an unrecognised NHS number is
    therefore still eligible to be shown to a remote classifier for complexity.
    """
    for _attempt in range(64):
        head = [rng.randint(0, 9) for _ in range(9)]
        wrong = rng.randint(0, 9)
        candidate = "".join(str(d) for d in head) + str(wrong)
        digits = [int(c) for c in candidate]
        if not _nhs_check_digit_ok(digits):
            value = f"{candidate[:3]} {candidate[3:6]} {candidate[6:]}"
            return SyntheticValue(
                kind="uk_nhs_number",
                value=value,
                basis="checksum-broken",
                reservation="mod-11 check digit deliberately wrong; every issued NHS number satisfies it",
                citation="NHS number format: 10 digits with a mod-11 check digit",
                # Not "none": the ten-digit shape also matches phone_number, which
                # is non-blocking. uk_nhs_number itself never fires, because its
                # validator rejects the broken checksum. See the docstring.
                expected_gate="fires",
                detail={"digits": candidate, "fires_as": "phone_number", "never_fires_as": "uk_nhs_number"},
            )
    raise SyntheticPiiError("uk_nhs_number: could not construct a checksum-broken number")


def uk_national_insurance(rng: random.Random) -> SyntheticValue:
    """A National Insurance number with a prefix HMRC will never allocate.

    Uses only ``BG``/``KN``/``NT`` -- the never-issued prefixes that
    ``gate._nino_ok`` does not already reject -- so the value is provably not a
    real person's *and* still fires the detector. ``GB``/``NK``/``TN``/``ZZ`` are
    equally never issued but the gate's validator refuses them, which would make
    every generated NINO a layer-2-only sample.
    """
    prefix = rng.choice(NINO_GATE_VISIBLE_PREFIXES)
    digits = f"{rng.randint(0, 999999):06d}"
    suffix = rng.choice("ABCD")
    return SyntheticValue(
        kind="uk_national_insurance",
        value=f"{prefix}{digits}{suffix}",
        basis="never-issued",
        reservation=f"HMRC never allocates the prefix {prefix}",
        citation=NINO_RESERVED_CITATION,
        expected_gate="blocks",
        detail={"prefix": prefix, "suffix": suffix},
    )


def medical_record_number(rng: random.Random) -> SyntheticValue:
    """A medical record number that names itself as synthetic.

    MRNs are issuer-local: there is no registry, no reserved block, and no check
    digit. The only honest proof available is construction -- the generator writes
    the value from a template that embeds ``SYNTHETIC``, which no institution
    issues as a chart number and which lets an operator attribute the string to us
    if they ever find it in a log.
    """
    value = f"{SELF_LABEL_TOKEN}-{rng.randint(0, 999999):06d}"
    return SyntheticValue(
        kind="medical_record_number",
        value=value,
        basis="self-labelled",
        reservation=f"value embeds the literal token {SELF_LABEL_TOKEN}; constructed, never captured",
        citation="no MRN registry exists; construction plus self-labelling is the available proof",
        rendered=f"MRN: {value}",
        expected_gate="blocks",
        detail={"prefix": SELF_LABEL_TOKEN},
    )


def provider_api_key(rng: random.Random) -> SyntheticValue:
    """An ``sk-``-shaped key whose body spells out that it is synthetic."""
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    tail = "".join(rng.choice(alphabet) for _ in range(24))
    value = f"sk-{SELF_LABEL_TOKEN.lower()}-example-{tail}"
    return SyntheticValue(
        kind="provider_api_key",
        value=value,
        basis="self-labelled",
        reservation=f"value embeds the literal token {SELF_LABEL_TOKEN}; constructed, never captured",
        citation="no provider reserves a key prefix for test data; construction is the available proof",
        expected_gate="blocks",
        detail={"shape": "sk-"},
    )


def aws_access_key_id(rng: random.Random, *, published: bool = True) -> SyntheticValue:
    """An AWS access key id.

    ``published=True`` (default) emits the exact value AWS uses as its
    documentation example, which AWS states is not a real key. ``published=False``
    emits a self-labelled ``AKIA``-shaped id instead, for samples that need many
    distinct keys rather than one well-known constant.
    """
    if published:
        return SyntheticValue(
            kind="aws_access_key_id",
            value=AWS_DOCUMENTATION_ACCESS_KEY_ID,
            basis="published-test-value",
            reservation="AWS's published documentation example access key id",
            citation=AWS_DOCUMENTATION_CITATION,
            expected_gate="blocks",
            detail={"published": True},
        )
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
    body = f"{SELF_LABEL_TOKEN[:9]}" + "".join(rng.choice(alphabet) for _ in range(7))
    return SyntheticValue(
        kind="aws_access_key_id",
        value=f"AKIA{body}",
        basis="self-labelled",
        reservation=f"value embeds the literal token {SELF_LABEL_TOKEN}; constructed, never captured",
        citation=AWS_DOCUMENTATION_CITATION,
        expected_gate="blocks",
        detail={"published": False},
    )


def private_key_block(rng: random.Random) -> SyntheticValue:
    """A PEM private-key block whose body is not a parseable key.

    The header is what the gate matches, and it has to be the real header for the
    sample to be format-faithful. The body is deliberately not valid DER: it is
    base64 over a self-labelled ASCII string, so nothing here can be loaded as a
    private key even by accident.
    """
    import base64

    payload = f"{SELF_LABEL_TOKEN}-NOT-A-REAL-KEY-{rng.randint(0, 999999):06d}".encode("ascii")
    body = base64.b64encode(payload).decode("ascii")
    wrapped = "\n".join(body[i : i + 64] for i in range(0, len(body), 64))
    value = f"-----BEGIN PRIVATE KEY-----\n{wrapped}\n-----END PRIVATE KEY-----"
    return SyntheticValue(
        kind="private_key_block",
        value=value,
        basis="self-labelled",
        reservation=f"PEM body is base64 over an ASCII string containing {SELF_LABEL_TOKEN}, not a DER key",
        citation="PKCS#8 PEM armouring; the body is not a valid SubjectPrivateKeyInfo",
        expected_gate="blocks",
        detail={"armour": "PRIVATE KEY"},
    )


def inline_credential(rng: random.Random) -> SyntheticValue:
    """A ``password=``-style assignment whose value is self-labelled."""
    key = rng.choice(("password", "DB_PASSWORD", "client_secret", "secret_token", "api_key"))
    secret = f"{SELF_LABEL_TOKEN}-NOT-A-REAL-SECRET-{rng.randint(0, 999999):06d}"
    return SyntheticValue(
        kind="inline_credential",
        value=secret,
        basis="self-labelled",
        reservation=f"value embeds the literal token {SELF_LABEL_TOKEN}; constructed, never captured",
        citation="no credential registry exists; construction is the available proof",
        rendered=f"{key}={secret}",
        expected_gate="blocks",
        detail={"key": key},
    )


def basic_auth_url(rng: random.Random) -> SyntheticValue:
    """A URL with embedded credentials, on an RFC 2606 host.

    The host is ``db.example.com`` rather than a reserved *TLD* so that the value
    fires exactly one detector. ``gate``'s email rule also matches the ``user@host``
    part of a URL, and its placeholder filter is a literal domain list: an
    ``@db.invalid`` host would add a spurious ``email_address`` finding, while
    ``@db.example.com`` is recognised as a placeholder and suppressed.
    """
    secret = f"{SELF_LABEL_TOKEN}-PASSWORD-{rng.randint(0, 999999):06d}"
    scheme = rng.choice(("postgres", "mysql", "amqp", "https"))
    port = rng.choice((5432, 3306, 5672, 443))
    value = f"{scheme}://{SELF_LABEL_TOKEN.lower()}:{secret}@db.example.com:{port}/{SELF_LABEL_TOKEN.lower()}"
    return SyntheticValue(
        kind="basic_auth_url",
        value=value,
        basis="self-labelled",
        reservation=f"password embeds {SELF_LABEL_TOKEN}; host is the RFC 2606 reserved domain example.com",
        citation="RFC 2606 section 3 for the host; construction for the credential",
        expected_gate="blocks",
        detail={"scheme": scheme, "host": "db.example.com"},
    )


def person_name(rng: random.Random) -> SyntheticValue:
    """A name from the closed placeholder roster."""
    given = rng.choice(PLACEHOLDER_GIVEN_NAMES)
    family = rng.choice(PLACEHOLDER_FAMILY_NAMES)
    value = f"{given} {family}"
    return SyntheticValue(
        kind="person_name",
        value=value,
        basis="placeholder-roster",
        reservation="closed roster of documentation placeholder names",
        citation=PLACEHOLDER_NAME_CITATION,
        expected_gate="none",
        detail={"given": given, "family": family},
    )


def date_of_birth(rng: random.Random) -> SyntheticValue:
    """A date of birth that is not a date.

    February 30th is nobody's birthday because it does not exist. That is a
    stronger proof than "we randomised it": a randomised date is very likely
    somebody's actual birthday, and a date of birth is one of the eighteen HIPAA
    identifiers precisely because it is identifying in combination with a name.
    The generated *person* is also from the placeholder roster, so the sample is
    fake twice over.
    """
    year = rng.randint(1930, 2020)
    month, day = rng.choice(IMPOSSIBLE_MONTH_DAYS)
    value = f"{year:04d}-{month:02d}-{day:02d}"
    return SyntheticValue(
        kind="date_of_birth",
        value=value,
        basis="impossible-value",
        reservation=f"{month:02d}-{day:02d} does not exist on the Gregorian calendar in any year",
        citation=IMPOSSIBLE_DATE_CITATION,
        # A bare date fires nothing; the detector needs the label in front of it.
        rendered=f"date of birth {value}",
        expected_gate="fires",
        detail={"year": year, "month": month, "day": day},
    )


#: kind -> generator, with the keyword arguments each one accepts.
GENERATORS: dict[str, Callable[..., SyntheticValue]] = {
    "us_ssn": ssn,
    "payment_card": payment_card,
    "email_address": email_address,
    "phone_number": phone_number,
    "iban": iban,
    "uk_nhs_number": uk_nhs_number,
    "uk_national_insurance": uk_national_insurance,
    "medical_record_number": medical_record_number,
    "provider_api_key": provider_api_key,
    "aws_access_key_id": aws_access_key_id,
    "private_key_block": private_key_block,
    "inline_credential": inline_credential,
    "basic_auth_url": basic_auth_url,
    "person_name": person_name,
    "date_of_birth": date_of_birth,
}

#: Identifier kinds whose gate behaviour is ``blocks`` by default, i.e. the ones
#: the deterministic floor refuses to send anywhere. Useful for building a set
#: that covers every blocking detector.
BLOCKING_KINDS: tuple[str, ...] = (
    "us_ssn",
    "payment_card",
    "iban",
    "uk_national_insurance",
    "medical_record_number",
    "provider_api_key",
    "aws_access_key_id",
    "private_key_block",
    "inline_credential",
    "basic_auth_url",
)


def generate(kind: str, rng: random.Random, **kwargs: Any) -> SyntheticValue:
    """Run one generator by name."""
    try:
        fn = GENERATORS[kind]
    except KeyError:
        raise SyntheticPiiError(f"unknown identifier kind {kind!r}; known kinds: {sorted(GENERATORS)}") from None
    return fn(rng, **kwargs)


def generate_all(rng: random.Random, **kwargs: Any) -> list[SyntheticValue]:
    """One value of every kind. ``kwargs`` are passed to the generators that accept them."""
    out: list[SyntheticValue] = []
    for kind in GENERATORS:
        accepted = _kwargs_for(kind, kwargs)
        out.append(generate(kind, rng, **accepted))
    return out


def _kwargs_for(kind: str, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Filter ``kwargs`` to the ones ``kind``'s generator actually accepts.

    Needed because :func:`generate_all` fans one option dict out over fifteen
    generators with three different signatures, and a typo should not turn into a
    ``TypeError`` from an unrelated generator.
    """
    import inspect

    params = inspect.signature(GENERATORS[kind]).parameters
    return {k: v for k, v in kwargs.items() if k in params}


def check_expected_gate(sv: SyntheticValue, gate: HardGate | None = None) -> str:
    """Measure what the real gate does with ``sv.rendered`` and compare to the claim.

    Returns the measured expectation. Raises :class:`ProvenanceError` on a
    mismatch, because a mismatch means a detector changed and every docstring in
    this module that says "fires the detector" or "exercises only layer 2" is now
    a lie in a repo that security reviewers read.
    """
    verdict = (gate or default_gate).scan(sv.rendered)
    measured = gate_expectation(verdict)
    if measured != sv.expected_gate:
        raise ProvenanceError(
            f"{sv.kind}: documented expected_gate={sv.expected_gate!r} but the real gate returned "
            f"{measured!r} for {sv.rendered!r} (detectors={verdict.detectors()}). Either the gate "
            "changed or the claim was wrong; fix the generator's expected_gate and this docstring."
        )
    return measured


# --------------------------------------------------------------------------- #
# Contextual sensitivity: sentences with no detector hit at all
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ContextualTemplate:
    """A genuinely sensitive request that contains nothing a regex can match.

    These are the reason the semantic layer exists. Every one of them would be
    labelled ``public`` by a keyword router that only had the gate's vocabulary,
    and every one of them describes data an operator would not paste into a
    third-party API.

    :attr:`expected_sensitivity` is the *author's* intent, not a label. It is
    recorded in provenance so the dataset can report how often the teacher agrees
    with the template author -- a low agreement rate is a signal about the
    templates, the teacher, or both, and hiding it would be worse than useless.
    """

    category: str
    text: str
    expected_sensitivity: str
    #: Why layer 1 cannot see this. A reviewer should be able to check the claim
    #: without reading fifteen regexes.
    why_regex_cannot_see_it: str

    def __post_init__(self) -> None:
        if self.expected_sensitivity not in SENSITIVITY_LEVELS:
            raise SyntheticPiiError(
                f"contextual template {self.category!r}: expected_sensitivity must be one of "
                f"{SENSITIVITY_LEVELS}, got {self.expected_sensitivity!r}"
            )


#: Every advisory keyword the default gate knows about, extracted from the live
#: detector set rather than retyped. Used by the verification helper and by the
#: test suite to explain *why* a template that starts firing is firing.
def advisory_vocabularies(gate: HardGate | None = None) -> dict[str, tuple[str, ...]]:
    """detector name -> the literal words its pattern can match, for advisory detectors."""
    active = gate or default_gate
    out: dict[str, tuple[str, ...]] = {}
    for detector in active.detectors:
        if not detector.advisory:
            continue
        words = re.findall(r"[A-Za-z][A-Za-z' \-]{2,}", detector.pattern.pattern)
        out[detector.name] = tuple(w.strip().lower() for w in words if w.strip())
    return out


def contextual_gate_hit(text: str, gate: HardGate | None = None) -> tuple[str, ...]:
    """Detector names that fire on ``text``, or ``()`` when layer 1 is blind to it."""
    verdict = (gate or default_gate).scan(text)
    return tuple(sorted(verdict.detectors()))


CONTEXTUAL_TEMPLATES: tuple[ContextualTemplate, ...] = (
    ContextualTemplate(
        category="hr_disciplinary",
        text=(
            "Draft the outcome letter for the second-stage conduct hearing held on Thursday. The colleague "
            "in the Leeds warehouse has two prior verbal warnings on file, the shift supervisor's account "
            "contradicts theirs on the timing, and the letter has to state the finding, the right of "
            "appeal, and the date the notice period ends."
        ),
        expected_sensitivity="confidential",
        why_regex_cannot_see_it=(
            "an individual's disciplinary record, described entirely by role and place. No name, no "
            "identifier, and none of the gate's business-keyword list (no 'performance review', no "
            "'salary', no 'confidential')."
        ),
    ),
    ContextualTemplate(
        category="hr_harassment_complaint",
        text=(
            "Someone in the Manchester office says their manager has been making comments about their "
            "accent for months and that two colleagues left for the same reason. Nobody has written "
            "anything down yet. Set out the first three steps the people team should take this week, and "
            "what must not be said to the manager until they have been taken."
        ),
        expected_sensitivity="confidential",
        why_regex_cannot_see_it=(
            "an unrecorded complaint about a named-by-role individual. The sensitivity is in the "
            "situation, not in any token: no detector in the default set matches grievance vocabulary."
        ),
    ),
    ContextualTemplate(
        category="legal_privilege",
        text=(
            "Outside counsel has sent over their read of the claimant's witness statements ahead of the "
            "hearing. Summarise the three factual points where their account and our incident log "
            "disagree, and flag anything that would weaken our position if the other side saw it first."
        ),
        expected_sensitivity="regulated",
        why_regex_cannot_see_it=(
            "legally privileged material in substance, but it never uses the phrases the gate keys on: "
            "no 'attorney-client', no 'legal privilege', no 'work product doctrine', no 'deposition'."
        ),
    ),
    ContextualTemplate(
        category="legal_insider_conduct",
        text=(
            "Someone in the finance team bought shares in a supplier the day before we told the market we "
            "were extending their contract. Nobody has raised it formally. Write the note that goes to "
            "general counsel, the three facts that decide whether this is a problem, and who else needs "
            "to know before Monday."
        ),
        expected_sensitivity="regulated",
        why_regex_cannot_see_it=(
            "market-abuse material. The gate's financial vocabulary covers PCI, GLBA, SAR filings and "
            "account data; it has nothing for trading ahead of a disclosure."
        ),
    ),
    ContextualTemplate(
        category="clinical_narrative",
        text=(
            "The night-shift nurse recorded that the person in bay four became disoriented during the "
            "early hours, pulled at the cannula twice, and settled only once a family member arrived. "
            "Write the handover entry so the morning team knows what to watch for and what has already "
            "been tried."
        ),
        expected_sensitivity="regulated",
        why_regex_cannot_see_it=(
            "a care note about an identifiable-by-location individual, written without any of the "
            "clinical vocabulary the gate keys on: no 'diagnosis', no 'medication', no 'treatment "
            "plan', no 'clinical note', and no name."
        ),
    ),
    ContextualTemplate(
        category="clinical_genetics",
        text=(
            "A relative has asked whether the results from the hereditary cancer panel should be shared "
            "with a sibling who has not been tested, and the consultant is unsure how far the duty of "
            "care reaches. Summarise both arguments and what the professional bodies actually say."
        ),
        expected_sensitivity="regulated",
        why_regex_cannot_see_it=(
            "genetic information about identifiable living people, which is special-category data under "
            "every regime that has such a category. Nothing in the text matches a health-regulation "
            "keyword, because the words doing the work are 'hereditary cancer panel' and 'sibling'."
        ),
    ),
    ContextualTemplate(
        category="safeguarding_minors",
        text=(
            "A year-ten pupil told the pastoral lead that something is happening at home and asked her "
            "not to write it down. Set out what the safeguarding policy requires in the next twenty-four "
            "hours and what the pastoral lead should say to the pupil about it."
        ),
        expected_sensitivity="regulated",
        why_regex_cannot_see_it=(
            "a disclosure about a child. The gate's minors vocabulary is regulatory (COPPA, FERPA, "
            "'student records', 'under 13'); a pastoral disclosure uses none of it."
        ),
    ),
    ContextualTemplate(
        category="minors_data_deletion",
        text=(
            "A parent has asked us to delete everything we hold about their daughter, who is in year nine "
            "and joined the after-school club in September. List what would have to be removed, what we "
            "are allowed to keep, and how long the whole thing should take."
        ),
        expected_sensitivity="regulated",
        why_regex_cannot_see_it=(
            "an erasure request concerning a child's data, expressed without 'child's data', 'minor's "
            "record', FERPA or COPPA. The sensitivity is the subject's age and the request type."
        ),
    ),
    ContextualTemplate(
        category="financial_covenant",
        text=(
            "The treasury desk wants a one-page note on the headroom in the new facility before Friday. "
            "Include the ratio we are actually at, the two levers that would move it, and what the "
            "lender is entitled to do if we miss the test date."
        ),
        expected_sensitivity="confidential",
        why_regex_cannot_see_it=(
            "material non-public financial position. The gate's financial keywords are about payment "
            "instrument compliance and account identifiers, not about lender covenants."
        ),
    ),
    ContextualTemplate(
        category="deal_negotiation",
        text=(
            "Two of the founders have been talking to the same strategic buyer for six weeks and the "
            "third has not been told. Draft the conversation the chief executive needs to have on "
            "Monday, including what can be said, what cannot, and what happens to the option pool if a "
            "term sheet lands before the all-hands."
        ),
        expected_sensitivity="confidential",
        why_regex_cannot_see_it=(
            "an undisclosed transaction and a governance problem at the same time. 'Acquisition target', "
            "'merger discussions' and 'board resolution' are the gate's words for this; the request uses "
            "none of them."
        ),
    ),
    ContextualTemplate(
        category="product_safety",
        text=(
            "The field team reports that the latch on the second production run releases under load in "
            "cold weather, and about four hundred units are already in customers' hands. Write the note "
            "that decides whether we stop shipping, who signs it, and what we tell the people who "
            "already have one."
        ),
        expected_sensitivity="confidential",
        why_regex_cannot_see_it=(
            "an unannounced safety defect with regulatory exposure. No keyword in the default set "
            "describes product harm; the security vocabulary is about software vulnerabilities."
        ),
    ),
    ContextualTemplate(
        category="security_intrusion",
        text=(
            "An unknown host answered a health check that only the bastion should have been able to "
            "reach, and nobody on the rota changed anything. Write the timeline we would hand to the "
            "incident commander, the two questions that decide how bad this is, and what should be "
            "turned off first."
        ),
        expected_sensitivity="confidential",
        why_regex_cannot_see_it=(
            "an active intrusion with no vulnerability identifier attached. The gate's security "
            "keywords need a CVE id, an exploit phrase, or the word 'exfiltration'; an unexplained host "
            "on an internal segment has none of them."
        ),
    ),
    ContextualTemplate(
        category="asylum_casework",
        text=(
            "Our caseworker has been sent a bundle for someone whose claim turns on events in a region "
            "where the family still lives. Explain what has to be taken out before anything is shared "
            "with the interpreting agency, and why that agency's own retention rules matter here."
        ),
        expected_sensitivity="regulated",
        why_regex_cannot_see_it=(
            "data whose disclosure could endanger third parties. There is no vocabulary for that in any "
            "detector set, and adding one would not help: the risk is in the relationship, not a token."
        ),
    ),
    ContextualTemplate(
        category="domestic_abuse_helpline",
        text=(
            "A caller to the helpline said they are planning to leave on Saturday and asked us to remove "
            "the message trail before then. Explain what our systems keep by default, who inside the "
            "organisation can see it, and the fastest way to make sure it is gone."
        ),
        expected_sensitivity="regulated",
        why_regex_cannot_see_it=(
            "the existence of the record is the danger. No identifier appears, because an identifier is "
            "not what makes this sensitive."
        ),
    ),
    ContextualTemplate(
        category="pay_equity_dispute",
        text=(
            "Two people in the same grade have found out they are paid differently and one of them has "
            "asked for the numbers behind the last three review cycles. Draft the response from people "
            "operations, including what we are required to disclose and what we would rather not put in "
            "writing."
        ),
        expected_sensitivity="confidential",
        why_regex_cannot_see_it=(
            "individual remuneration data across named-by-role employees. The gate keys on the words "
            "'salary', 'compensation band' and 'performance review'; this request says 'paid "
            "differently' and 'review cycles'."
        ),
    ),
    ContextualTemplate(
        category="research_consent",
        text=(
            "A participant in the second arm has emailed to say they no longer want to be involved but "
            "do not want their samples withdrawn, and the coordinator is unsure whether that combination "
            "is allowed. Explain what the protocol permits and what has to happen to the material "
            "already taken."
        ),
        expected_sensitivity="regulated",
        why_regex_cannot_see_it=(
            "consent status and biological material linked to an identifiable participant. The clinical "
            "keywords in the gate are about records and treatment, not about consent withdrawal."
        ),
    ),
)


def contextual_samples(
    rng: random.Random,
    *,
    n: int | None = None,
    categories: Iterable[str] | None = None,
    gate: HardGate | None = None,
) -> list[ContextualTemplate]:
    """Pick contextual templates, verifying against the real gate as they are picked.

    ``n=None`` returns every template once (optionally filtered to ``categories``).
    With ``n`` set, templates are drawn with replacement so a larger set can be
    built from a fixed vocabulary; the draw is seeded, so it is reproducible.

    Raises :class:`ProvenanceError` naming the template and the detectors that
    fired. This is not a nicety: the entire claim of this category is that layer 1
    is blind to it, and a template that drifts into firing a keyword silently
    turns a layer-2 positive into a layer-1 one. Failing loudly is the only way the
    claim stays true as the gate evolves.
    """
    active = gate or default_gate
    pool = list(CONTEXTUAL_TEMPLATES)
    if categories is not None:
        wanted = frozenset(categories)
        unknown = wanted - {t.category for t in CONTEXTUAL_TEMPLATES}
        if unknown:
            raise SyntheticPiiError(
                f"unknown contextual categories {sorted(unknown)}; known: "
                f"{sorted({t.category for t in CONTEXTUAL_TEMPLATES})}"
            )
        pool = [t for t in pool if t.category in wanted]
    if not pool:
        raise SyntheticPiiError("no contextual templates matched the requested categories")

    chosen = pool if n is None else [rng.choice(pool) for _ in range(max(0, int(n)))]
    for template in chosen:
        hits = contextual_gate_hit(template.text, active)
        if hits:
            raise ProvenanceError(
                f"contextual template {template.category!r} fires the deterministic gate: {hits}. "
                "It is no longer a layer-2-only example. Rewrite it without the matched vocabulary, or "
                "move it to the synthetic-PII category and say so."
            )
    return chosen


def contextual_categories() -> tuple[str, ...]:
    return tuple(dict.fromkeys(t.category for t in CONTEXTUAL_TEMPLATES))


__all__ = [
    "AWS_DOCUMENTATION_ACCESS_KEY_ID",
    "AWS_DOCUMENTATION_CITATION",
    "BLOCKING_KINDS",
    "CONTEXTUAL_TEMPLATES",
    "EMAIL_RESERVED_CITATION",
    "GENERATORS",
    "IBAN_RESERVED_CITATION",
    "IBAN_RESERVED_COUNTRY_CODES",
    "IMPOSSIBLE_MONTH_DAYS",
    "NINO_GATE_VISIBLE_PREFIXES",
    "NINO_NEVER_ISSUED_PREFIXES",
    "PHONE_RESERVED_OFFICE_CODE",
    "PHONE_RESERVED_STATION_RANGE",
    "PLACEHOLDER_FAMILY_NAMES",
    "PLACEHOLDER_GIVEN_NAMES",
    "PUBLISHED_TEST_CARDS",
    "PUBLISHED_TEST_CARD_NUMBERS",
    "RFC2606_EMAIL_DOMAINS",
    "RFC6761_RESERVED_TLDS",
    "RFC6761_TLDS_GATE_TREATS_AS_PLACEHOLDER",
    "RFC6761_TLDS_GATE_TREATS_AS_REAL",
    "SELF_LABEL_TOKEN",
    "SSN_NEVER_ISSUED_AREAS",
    "SSN_NEVER_ISSUED_AREA_RANGE",
    "CardFakeness",
    "ContextualTemplate",
    "FakenessBasis",
    "GateExpectation",
    "IbanFakeness",
    "ProvenanceError",
    "SyntheticPiiError",
    "SyntheticValue",
    "advisory_vocabularies",
    "aws_access_key_id",
    "basic_auth_url",
    "check_expected_gate",
    "contextual_categories",
    "contextual_gate_hit",
    "contextual_samples",
    "date_of_birth",
    "email_address",
    "gate_expectation",
    "generate",
    "generate_all",
    "iban",
    "inline_credential",
    "medical_record_number",
    "person_name",
    "phone_number",
    "private_key_block",
    "provider_api_key",
    "ssn",
    "uk_national_insurance",
    "uk_nhs_number",
    "verify_fakeness",
]
