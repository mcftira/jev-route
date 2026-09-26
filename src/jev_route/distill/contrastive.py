"""Contrastive pair curation for the v0.5 held-out flip eval.

This is the data half of v0.5's contrastive track. A *contrastive pair* is two
texts that differ by exactly one mutation -- and that mutation is precisely the
fact a scorer has to read: a check digit that flips Luhn validity, a TAJ number
that is present instead of redacted, an SSN area number that is never issued
instead of issuable, or a task scope that moves from a one-line fix to a
multi-file refactor or a schema migration. For every pair the correct decision
is known by construction, and the held-out flip eval
(``evals/contrastive/run.py``) scores one metric: when the key fact changes,
does the decision change? A scorer that cannot see the mutation cannot be
graduated, no matter how accurate it is on single-label data.

Two rules shape this module, and both come from the provenance/PII-fake
conventions already in force in the repo:

* **Fakeness is a property of the generator, not a hope.** The Luhn-pass card
  member is a *published vendor test PAN* taken from
  :data:`jev_route.distill.synthetic_pii.PUBLISHED_TEST_CARDS` (Luhn-valid and
  publicly reserved, so it is not anybody's card and it still fires the gate's
  ``payment_card`` detector). The Luhn-fail member is the same PAN with the
  check digit broken, which no issuer can have assigned. SSN bases use the SSA
  never-issued area numbers 000/666/900-999; SSN variants use area 219, which is
  structurally issuable and therefore labelled sensitive, and the value is a
  constructed placeholder in a controlled corpus, never captured -- the same
  convention ``evals/injection/gen_cases.py`` already ships. TAJ numbers are the
  repo's 9-digit, 000-led fixture convention (never captured). Nothing here is
  a real person's data, and nothing here is a real PAN.
* **The label is checked, not assumed.** Every pair is verified at generation
  time against the real :class:`~jev_route.gate.HardGate` and the keyless
  scorers below: a card pair whose Luhn-pass member fails Luhn, an SSN pair
  whose base is not in a never-issued area, or a routing pair whose variant
  does not land on the labelled tier raises :class:`ContrastiveDataError`
  instead of being emitted. A pair that would misdescribe the fact it is meant
  to isolate cannot be built.

The determinism contract matches :mod:`jev_route.distill.synthetic_pii`:
:func:`generate_pairs` is a pure function of its arguments. No clock, no
environment, no filesystem -- so the same seed produces the same pairs in any
process on any machine, and the train and eval splits (different seeds) are
disjoint by construction and verified to be disjoint where they are written.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..backends.base import DecisionRequest
from ..backends.mock import MODEL_VERSION as MOCK_MODEL_VERSION
from ..backends.mock import MockBackend
from ..gate import HardGate, luhn_ok
from ..prompts import compute_features
from .synthetic_pii import (
    PUBLISHED_TEST_CARDS,
    SSN_NEVER_ISSUED_AREA_RANGE,
    SSN_NEVER_ISSUED_AREAS,
    SSN_RESERVED_CITATION,
)

#: The two domains a pair can live in. Sensitivity pairs flip a 0/1 PII label;
#: routing pairs flip the tier string. The flip eval reports per domain and per
#: pair kind, which is why both are recorded on every pair.
SENSITIVITY_DOMAIN = "sensitivity"
ROUTING_DOMAIN = "routing"
PAIR_DOMAINS: tuple[str, ...] = (SENSITIVITY_DOMAIN, ROUTING_DOMAIN)

#: Pair kinds, in the fixed order the generator walks them. The order is part of
#: the determinism contract: the RNG stream is consumed in this order, so a
#: reorder changes every pair that follows.
SENSITIVITY_KINDS: tuple[str, ...] = ("card_luhn", "taj", "ssn")
ROUTING_KINDS: tuple[str, ...] = ("one_liner_vs_refactor", "one_liner_vs_migration")
KINDS_BY_DOMAIN: dict[str, tuple[str, ...]] = {
    SENSITIVITY_DOMAIN: SENSITIVITY_KINDS,
    ROUTING_DOMAIN: ROUTING_KINDS,
}

#: The held-out eval split: ~200 pairs per domain. Eval numbers must reproduce,
#: so this table -- not a free-form count -- is what ``gen_pairs.py`` writes.
EVAL_SPLIT: dict[str, int] = {
    "card_luhn": 80,
    "taj": 60,
    "ssn": 60,
    "one_liner_vs_refactor": 100,
    "one_liner_vs_migration": 100,
}
#: The contrastive training split: double the eval, same kind mix.
TRAIN_SPLIT: dict[str, int] = {
    "card_luhn": 160,
    "taj": 120,
    "ssn": 120,
    "one_liner_vs_refactor": 200,
    "one_liner_vs_migration": 200,
}


class ContrastiveDataError(ValueError):
    """A pair (or split) violates a label, fakeness or determinism invariant."""


# --------------------------------------------------------------------------- #
# The pair
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ContrastivePair:
    """One contrastive pair: two texts, one mutation, a known correct decision on each.

    ``label_base``/``label_variant`` are the correct decisions: 0/1 for the
    sensitivity domain (0 = not sensitive, 1 = sensitive), the tier string
    (``local``/``cheap``/``strong``) for the routing domain. By construction the
    two labels differ -- a pair whose labels agree isolates nothing and cannot
    be emitted (see the ``__post_init__`` check).
    """

    id: str
    domain: str
    kind: str
    base: str
    variant: str
    label_base: int | str
    label_variant: int | str
    #: Provenance: the mutation, the fakeness reservations, the citations, and
    #: (for identifiers) the exact values used. Stored on every pair so the
    #: dataset can be audited without re-deriving anything.
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.domain not in PAIR_DOMAINS:
            raise ContrastiveDataError(f"pair {self.id}: unknown domain {self.domain!r}")
        if self.kind not in KINDS_BY_DOMAIN.get(self.domain, ()):
            raise ContrastiveDataError(f"pair {self.id}: kind {self.kind!r} is not a {self.domain} kind")
        if not self.base.strip() or not self.variant.strip():
            raise ContrastiveDataError(f"pair {self.id}: empty text carries no signal")
        if self.label_base == self.label_variant:
            raise ContrastiveDataError(
                f"pair {self.id}: both members carry label {self.label_base!r}; a pair whose labels "
                "agree isolates no fact and cannot be emitted"
            )

    def to_dict(self) -> dict[str, Any]:
        """The JSONL row. The contract keys first, kind and provenance after."""
        return {
            "id": self.id,
            "domain": self.domain,
            "kind": self.kind,
            "base": self.base,
            "variant": self.variant,
            "label_base": self.label_base,
            "label_variant": self.label_variant,
            "provenance": dict(self.provenance),
        }


# --------------------------------------------------------------------------- #
# Keyless scorers: the CI decision layer, no API keys, no model
# --------------------------------------------------------------------------- #
class GateSensitivityScorer:
    """The deterministic gate as a pair scorer.

    The decision is the gate's *canonical verdict*: ``clean``, ``fires:<dets>``,
    ``local:<dets>`` (a force_local finding) or ``blocks:<dets>``. A pair flips
    when the gate treats the two members differently. Note the verdict, not
    "blocked or not", is the decision: a Luhn-fail 16-digit run still fires
    ``phone_number`` (the gate keeps 16-digit runs local by design), while the
    Luhn-pass twin fires ``payment_card`` and blocks -- two different treatments
    of one fact (the check digit) the gate can read.
    """

    name = "deterministic-gate"
    model_version = "hard-gate-default"

    def __init__(self, gate: HardGate | None = None) -> None:
        self.gate = gate or HardGate()

    def score(self, text: str, domain: str = SENSITIVITY_DOMAIN) -> str:
        if domain != SENSITIVITY_DOMAIN:
            raise ContrastiveDataError(
                f"the deterministic gate scores the {SENSITIVITY_DOMAIN!r} domain, not {domain!r}"
            )
        verdict = self.gate.scan(text)
        if not verdict.fired:
            return "clean"
        dets = "+".join(sorted({f.detector for f in verdict.findings}))
        if verdict.blocks_backend:
            return f"blocks:{dets}"
        if verdict.force_local:
            return f"local:{dets}"
        return f"fires:{dets}"


class MockTierScorer:
    """MockBackend-shaped tier rules, keyless.

    The same rule order the default policy runs: the hard gate first (blocks or
    force_local -> ``local``), then the MockBackend's complexity read mapped the
    way ``policies/default.yaml`` maps it (``hard``/``frontier`` -> ``strong``,
    everything else -> ``cheap``). No routing pair in this set is sensitive, but
    the gate step is kept so a template regression that leaks PII shows up as
    ``local`` -- a visible failure, not a silent one.
    """

    name = "mock-tier-rules"
    model_version = MOCK_MODEL_VERSION

    def __init__(self, gate: HardGate | None = None, backend: MockBackend | None = None) -> None:
        self.gate = gate or HardGate()
        self.backend = backend or MockBackend()

    def score(self, text: str, domain: str = ROUTING_DOMAIN) -> str:
        if domain != ROUTING_DOMAIN:
            raise ContrastiveDataError(
                f"the mock tier rules score the {ROUTING_DOMAIN!r} domain, not {domain!r}"
            )
        verdict = self.gate.scan(text)
        if verdict.blocks_backend or verdict.force_local:
            return "local"
        answers = self.backend.decide_sync(DecisionRequest(redacted_excerpt=text, features=compute_features(text)))
        return "strong" if answers.complexity.choice in ("hard", "frontier") else "cheap"


class KeylessFlipScorer:
    """The combined keyless scorer the flip eval runs in CI: gate, then tier rule, by domain."""

    name = "keyless"

    def __init__(self, gate: HardGate | None = None, backend: MockBackend | None = None) -> None:
        self.sensitivity = GateSensitivityScorer(gate)
        self.routing = MockTierScorer(gate, backend)

    def score(self, text: str, domain: str) -> str:
        if domain == SENSITIVITY_DOMAIN:
            return self.sensitivity.score(text)
        if domain == ROUTING_DOMAIN:
            return self.routing.score(text)
        raise ContrastiveDataError(f"unknown domain {domain!r}; known: {PAIR_DOMAINS}")


# --------------------------------------------------------------------------- #
# Material: frames and pools. Every identifier in here is constructed or a
# published test value; the per-pair provenance says which one and why it is fake.
# --------------------------------------------------------------------------- #
#: Placeholder names from the closed roster in :mod:`jev_route.distill.synthetic_pii`.
#: Frames avoid the "customer named X Y" construction on purpose: that shape is
#: what the gate's ``named_individual`` detector keys on, and a sensitivity pair
#: should isolate exactly one fact, not two.
PLACEHOLDER_NAMES: tuple[str, ...] = (
    "Jane Doe",
    "John Roe",
    "Richard Testerson",
    "Testy Sampleperson",
    "Jane Roe",
    "John Doe",
    # Widened for the TAJ kind: its redacted variant text contains no TAJ value,
    # so the pool of distinct variant texts is frames x names -- six names gave
    # 36 slots against a 60-pair eval split, and the draw loop hit the ceiling.
    "Alice Example",
    "Bob Placeholder",
    "Carol Fictional",
    "Dave Mockdata",
    "Eve Synthetic",
    "Frank Dummy",
    "Grace Fixture",
    "Henry Specimen",
    "Ivy Sample",
    "Jack Standin",
    "Karen Proforma",
    "Leo Pseudonym",
    "Maria Token",
    "Nina Anon",
    "Oscar Fillin",
    "Pat Stunt",
    "Quinn Template",
    "Rosa Stockname",
    "Sam Alias",
    "Tina Covername",
    "Uli Subname",
    "Vera Maskname",
)

#: Order/invoice ids. Five-digit bodies: below the gate's phone length floor, so
#: they cannot add a second detector hit to a pair that isolates one fact.
ORDER_IDS: tuple[str, ...] = ("ORD-48217", "ORD-10923", "ORD-55001", "INV-77420", "ORD-30314")

#: Sentence frames for card pairs. The number sits where such a value actually
#: arrives, and every frame keeps the number's neighbourhood free of a ``.`` or a
#: digit right after the last digit: the card detector's trailing lookahead
#: refuses both, so a frame that put a period on the number would silently kill
#: the ``payment_card`` finding.
CARD_FRAMES: tuple[str, ...] = (
    "The refund for order {order} is on card {value} and the finance team needs the receipt.",
    "Customer {name} disputes a charge on card {value} and wants it reversed today.",
    "Update the saved card for {name} to {value} before the Friday billing run.",
    "The failed charge on card {value} needs a retry once the invoice is reissued.",
    "Charge the annual plan upgrade to card {value} for {name}.",
    "The card {value} was declined twice; check the 3DS log and the issuer response.",
    "Replace the stored card {value} in the billing export for the auditor.",
    "The chargeback filed against card {value} cites unauthorized use.",
    "Export the billing row for card {value} into the finance sheet.",
    "The duplicate charge on card {value} must be refunded before month end.",
    "Add card {value} to the escrow wallet for order {order}.",
    "The subscription renewal on card {value} keeps bouncing; find the reason.",
)

#: TAJ frames. ``{taj}`` is a 9-digit, 000-led placeholder (the repo fixture
#: convention); the redacted member is the same sentence with the slot replaced
#: by ``[REDACTED]``. The 9-digit run is what the gate's ``phone_number``
#: detector fires on (it cannot know Hungarian number shapes), which is why the
#: gate flips on these pairs: present fires, redacted is clean.
TAJ_FRAMES: tuple[str, ...] = (
    "A TAJ szamom: {taj}, es a nevem {name}. Kuldje a jelentest a hivatalnak.",
    "Fill in the tax form: TAJ {taj}, name {name}, tax year 2024.",
    "The tax office needs my TAJ {taj} and the full name {name}.",
    "Adozo igelet: TAJ szam {taj} (nevem {name}), illeto: adozasi bejelentkezes.",
    "Send the 2025 tax summary to my address, TAJ {taj}, in the name of {name}.",
    "The accounting firm asked for TAJ {taj} to file the return on time.",
)

#: SSN frames. ``{ssn}`` is the mutated slot: the base is a never-issued area
#: (label 0, provably not a person's number), the variant is the same group and
#: serial under area 219 (label 1, structurally indistinguishable from a real
#: SSN). The gate blocks both on purpose -- it reads the shape, not the area --
#: which is exactly the reservation knowledge this pair set is built to measure.
SSN_FRAMES: tuple[str, ...] = (
    "Process the payroll change for employee {name}. Their SSN is {ssn}.",
    "The background check came back for applicant {name}, SSN {ssn}. Summarise it.",
    "Update the tax form for {name}: SSN {ssn}, effective next pay period.",
    "The onboarding file for {name} lists SSN {ssn}; verify the format.",
    "Payroll rejected the new starter {name}, SSN {ssn}. Check the format.",
)

#: The variant SSN area. Structurally issuable (post-randomization), which is
#: what makes the variant sensitive; ``evals/injection/gen_cases.py`` already
#: ships 219-xx-xxxx as SSN-shaped synthetic PII.
SSN_VARIANT_AREA = "219"

#: Base tasks for routing pairs: subject + one code identifier + the module fan-out.
ROUTING_TASKS: tuple[tuple[str, str, str], ...] = (
    ("the pagination cursor", "pagination.py", "api/, workers/, cli/ and the test suite"),
    ("the session token store", "sessions.py", "gateway/, auth/, cache/ and the expiry worker"),
    ("the rate limiter state", "ratelimit.py", "ingest/, api/ and the metrics exporter"),
    ("the user preferences blob", "prefs.py", "web/, mobile-api/ and the sync worker"),
    ("the invoice line items", "invoices.py", "billing/, reporting/ and the dunning job"),
    ("the audit log writer", "audit.py", "core/, api/ and the archive job"),
    ("the webhook delivery queue", "webhooks.py", "delivery/, retry/ and the dead-letter job"),
    ("the feature flag cache", "flags.py", "edge/, api/ and the admin console"),
    ("the export job scheduler", "exports.py", "scheduler/, storage/ and the notification worker"),
    ("the address normaliser", "address.py", "checkout/, shipping/ and the import script"),
    ("the idempotency key map", "idem.py", "payments/, retry/ and the cleanup job"),
    ("the locale catalogue", "locales.py", "web/, cli/ and the translation pipeline"),
    ("the attachment metadata", "attach.py", "storage/, upload/ and the virus-scan worker"),
    ("the price tier lookup", "pricing.py", "checkout/, admin/ and the forecast report"),
    # Widened for the routing kinds: base/variant text entropy is frames x tasks
    # x n -- 14 tasks exhausted under the eval split's 100-pair demand.
    ("the token bucket refill", "buckets.py", "ingest/, api/ and the quota worker"),
    ("the session heartbeat map", "heartbeat.py", "gateway/, health/ and the reaper job"),
    ("the search index cursor", "index_cursor.py", "search/, indexer/ and the rebuild task"),
    ("the shipment tracking cache", "tracking.py", "fulfillment/, web/ and the poll worker"),
    ("the discount code ledger", "discounts.py", "checkout/, billing/ and the cleanup job"),
    ("the rollout percentage store", "rollouts.py", "flags/, api/ and the metrics exporter"),
    ("the draft autosave buffer", "drafts.py", "editor/, api/ and the cleanup worker"),
    ("the notification throttle", "throttle.py", "notify/, workers/ and the rate auditor"),
    ("the translation fallback map", "fallback.py", "i18n/, web/ and the prefetch job"),
    ("the payment retry plan", "retry_plan.py", "payments/, dunning/ and the scheduler"),
    ("the import dedupe set", "dedupe.py", "import/, storage/ and the audit job"),
    ("the device trust registry", "devices.py", "auth/, api/ and the review console"),
    ("the calendar sync token", "calendarsync.py", "calendar/, sync/ and the webhook worker"),
    ("the inventory reservation map", "reserve.py", "warehouse/, checkout/ and the expiry job"),
    ("the content moderation queue", "moderation.py", "review/, api/ and the escalate worker"),
    ("the firmware rollout rings", "rings.py", "ota/, devices/ and the metrics job"),
    ("the subscription proration calc", "prorate.py", "billing/, checkout/ and the forecast job"),
    ("the mailbox threading index", "threading.py", "mail/, search/ and the archive job"),
    ("the analytics sampling map", "sampling.py", "analytics/, api/ and the rollup worker"),
    ("the license seat counter", "seats.py", "admin/, billing/ and the audit job"),
    ("the edge cache invalidation log", "invalidation.py", "edge/, origin/ and the purge worker"),
    ("the workflow retry registry", "wfretry.py", "orchestrator/, workers/ and the monitor job"),
)

#: One-liner scope. Deliberately free of the MockBackend's hard-task vocabulary
#: (no "refactor", no "migrat*", no "performance"): the base member of a routing
#: pair must stay on the cheap tier, and a stray marker in the base would erode
#: the flip the pair exists to measure.
ONE_LINER_FRAMES: tuple[str, ...] = (
    "Fix the bug in {subject}: in {code} the off-by-one on the empty page returns {bug} instead of the first item.",
    # Every frame must format {bug}: a slot that only one frame uses collapses
    # the text space and starves the draw loop (measured: 129/300 collisions).
    "In {code}, {subject} counts the last row twice when the list is empty; {bug} slipped through the boundary check.",
    "One-line fix for {subject}: {code} uses the wrong variable in the zero-item branch ({bug}).",
    "Patch {code} so {subject} stops crashing on the empty case; the guard is missing one clause for {bug}.",
)
ONE_LINER_BUGS: tuple[str, ...] = (
    "None", "an empty list", "-1", "the stale cursor",
    # Widened for the same reason: shared one-liner bases are drawn by BOTH
    # routing kinds, so the bug pool is the bottleneck for both.
    "a zero count", "the wrong key", "an off-by-one bound", "the missing default",
    "a double count", "the unsorted input", "a swallowed exception", "the inverted flag",
)

#: Multi-file refactor scope. Every frame carries at least one of the
#: MockBackend's hard-complexity markers ("refactor", "performance"/"bottleneck",
#: "concurrency") in addition to the scope itself, so the strong tier is earned
#: by the task, not by luck of the draw.
REFACTOR_FRAMES: tuple[str, ...] = (
    "Refactor {subject} across {modules}: the same boundary logic is copy-pasted in {n} places and "
    "they disagree on the empty case. Consolidate it into one module, update every call site, and add "
    "regression tests for the edge case.",
    "Redesign {subject}. The boundary logic is duplicated across {modules} and has become a "
    "performance bottleneck: the empty case is handled differently in each copy and the paths race "
    "under concurrency. Unify it into one module, keep the public API stable, and add regression "
    "tests for the edge case.",
    "Clean up {subject} in {modules}: {n} modules each handle the boundary differently, the "
    "duplication shows up as a performance bottleneck, and the empty case is inconsistent. Move the "
    "shared logic into one helper, migrate all call sites in the same change, and cover the empty "
    "case with tests.",
)

#: Schema-migration scope. Same story: "migrat*" plus a second hard marker
#: (latency, concurrency) in every frame.
MIGRATION_FRAMES: tuple[str, ...] = (
    "Write the schema migration for {subject}: move the state out of the legacy column into a "
    "dedicated table, backfill {rows} rows without blocking writes, add the new indexes, provide a "
    "rollback script, and plan the cutover so p99 read latency stays under 50ms.",
    "Plan and write the migration that reshapes {subject}: the current layout needs a new table plus "
    "a two-week dual-write phase. Cover the backfill of {rows} rows, the index build, the rollback "
    "path, and the replica cutover with a latency budget.",
    "Create the data migration for {subject}: split the legacy field into a normalised table, "
    "backfill {rows} rows in batches without blocking writes, verify checksums, add indexes, and "
    "document the rollback, the concurrency risk during the dual write, and the cutover window for "
    "the on-call rotation.",
)
MIGRATION_ROW_COUNTS: tuple[str, ...] = (
    "40 million", "12 million", "200 million", "5 million",
    # Widened: the split draws train and eval from one space, and 4 row counts
    # were the bottleneck that exhausted it.
    "80 million", "900 thousand", "150 million", "25 million",
    "60 million", "3 million", "400 million", "18 million",
)


def _never_issued_area(area: str) -> bool:
    """True for the SSA never-issued areas (000, 666 and 900-999)."""
    return area in SSN_NEVER_ISSUED_AREAS or (
        SSN_NEVER_ISSUED_AREA_RANGE[0] <= int(area) <= SSN_NEVER_ISSUED_AREA_RANGE[1]
    )


def _break_luhn(number: str) -> str:
    """Flip the check digit so Luhn fails, keeping the BIN and the length (synthetic_pii's rule)."""
    return number[:-1] + str((int(number[-1]) + 1) % 10)


def _spaced_card(number: str) -> str:
    """Group the PAN the way a card is typed: 4-6-5 for Amex, 4-4-4-4 otherwise."""
    if len(number) == 15:
        return f"{number[:4]} {number[4:10]} {number[10:]}"
    return f"{number[:4]} {number[4:8]} {number[8:12]} {number[12:]}"


def _is_single_replacement(base: str, old: str, variant: str, new: str) -> bool:
    """True iff ``variant`` is ``base`` with exactly one occurrence of ``old`` replaced by ``new``.

    This is the "identical except one mutation" property of a pair, checked the
    way it is meant to hold: same text, one slot changed. (Position diffs are the
    wrong test: the SSN area mutation changes one field whose characters partly
    coincide, e.g. 919 -> 219 differs in one character, not three.)
    """
    return base.count(old) == 1 and base.replace(old, new, 1) == variant


@dataclass(frozen=True)
class _Ctx:
    """Shared, stateless context the builders verify against (one gate, one mock)."""

    gate: HardGate
    mock: MockBackend


# --------------------------------------------------------------------------- #
# Pair builders: one per kind. Each one is a pure function of the RNG stream and
# each one ends in the check that makes its label a measured property.
# --------------------------------------------------------------------------- #
def _pair_id(id_prefix: str, domain: str, kind: str, index: int) -> str:
    return f"{id_prefix}-{domain}-{kind}-{index:03d}"


def _build_card_pair(rng: random.Random, index: int, id_prefix: str, ctx: _Ctx) -> ContrastivePair:  # noqa: ARG001 -- builder signature is uniform across kinds
    """Luhn-fail (base, label 0) vs published Luhn-pass test PAN (variant, label 1)."""
    number, brand, citation = rng.choice(PUBLISHED_TEST_CARDS)
    broken = _break_luhn(number)
    if luhn_ok(broken):  # pragma: no cover - arithmetic guard, unreachable in practice
        raise ContrastiveDataError(f"card pair {index}: _break_luhn produced a Luhn-valid number")
    if broken in {n for n, _b, _c in PUBLISHED_TEST_CARDS}:
        raise ContrastiveDataError(f"card pair {index}: broken PAN collides with a published test PAN")
    frame = rng.choice(CARD_FRAMES)
    slots = {
        "name": rng.choice(PLACEHOLDER_NAMES),
        "order": rng.choice(ORDER_IDS),
    }
    base = frame.format(value=_spaced_card(broken), **slots)
    variant = frame.format(value=_spaced_card(number), **slots)
    if not _is_single_replacement(base, _spaced_card(broken), variant, _spaced_card(number)):
        raise ContrastiveDataError(f"card pair {index}: base and variant differ outside the check digit")
    return ContrastivePair(
        id=_pair_id(id_prefix, SENSITIVITY_DOMAIN, "card_luhn", index),
        domain=SENSITIVITY_DOMAIN,
        kind="card_luhn",
        base=base,
        variant=variant,
        label_base=0,
        label_variant=1,
        provenance={
            "mutation": "the card check digit (exactly one character)",
            "base_value": _spaced_card(broken),
            "variant_value": _spaced_card(number),
            "base_fakeness": "checksum-broken: Luhn-invalid; no issuer assigns a PAN that fails its own check digit",
            "variant_fakeness": f"published {brand} test PAN (Luhn-valid, publicly reserved)",
            "citation": citation,
            "gate": (
                "payment_card fires only on the variant; the base fires phone_number "
                "(16-digit run) and stays local"
            ),
        },
    )


def _build_taj_pair(rng: random.Random, index: int, id_prefix: str, ctx: _Ctx) -> ContrastivePair:
    """TAJ-shaped number present (base, label 1) vs the same sentence redacted (variant, label 0)."""
    frame = rng.choice(TAJ_FRAMES)
    name = rng.choice(PLACEHOLDER_NAMES)
    taj = f"000 {rng.randint(100, 999)} {rng.randint(100, 999)}"
    # The redacted variant contains no TAJ value, so variant-text capacity was
    # frames x names and both splits exhausted it. A per-draw case-number
    # suffix on BOTH members keeps the content realistic, the space unbounded,
    # and the single-replacement check satisfied (the only difference is the
    # TAJ slot).
    case_id = f" (ugyszam: {rng.randint(100000, 999999)})"
    base = frame.format(taj=taj, name=name) + case_id
    variant = frame.format(taj="[REDACTED]", name=name) + case_id
    if not _is_single_replacement(base, taj, variant, "[REDACTED]"):
        raise ContrastiveDataError(f"taj pair {index}: base and variant differ outside the TAJ slot")
    gate = GateSensitivityScorer(ctx.gate)
    if gate.score(base) == "clean" or gate.score(variant) != "clean":
        raise ContrastiveDataError(
            f"taj pair {index}: the gate does not flip on this frame (base={gate.score(base)!r}, "
            f"variant={gate.score(variant)!r}); a TAJ pair must isolate presence vs redaction"
        )
    return ContrastivePair(
        id=_pair_id(id_prefix, SENSITIVITY_DOMAIN, "taj", index),
        domain=SENSITIVITY_DOMAIN,
        kind="taj",
        base=base,
        variant=variant,
        label_base=1,
        label_variant=0,
        provenance={
            "mutation": "the TAJ number is present vs [REDACTED]",
            "base_value": taj,
            "base_fakeness": (
                "9-digit TAJ placeholder with the 000 lead group: the repo fixture convention "
                "(evals/injection/gen_cases.py HU_PII), constructed, never captured"
            ),
            "gate": "phone_number (9-digit run) fires on the base and is clean on the redacted variant",
        },
    )


def _build_ssn_pair(rng: random.Random, index: int, id_prefix: str, ctx: _Ctx) -> ContrastivePair:  # noqa: ARG001 -- builder signature is uniform across kinds  # noqa: ARG001 -- builder signature is uniform across kinds
    """Never-issued area (base, label 0) vs structurally-issuable area 219 (variant, label 1)."""
    frame = rng.choice(SSN_FRAMES)
    name = rng.choice(PLACEHOLDER_NAMES)
    area = rng.choice((*SSN_NEVER_ISSUED_AREAS, *(str(a) for a in range(*SSN_NEVER_ISSUED_AREA_RANGE))))
    group = f"{rng.randint(1, 99):02d}"
    serial = f"{rng.randint(1, 9999):04d}"
    base_ssn = f"{area}-{group}-{serial}"
    variant_ssn = f"{SSN_VARIANT_AREA}-{group}-{serial}"
    if not _never_issued_area(area):  # pragma: no cover - construction guard
        raise ContrastiveDataError(f"ssn pair {index}: base area {area!r} is issuable")
    if _never_issued_area(SSN_VARIANT_AREA):  # pragma: no cover - constant guard
        raise ContrastiveDataError(f"ssn pair {index}: variant area {SSN_VARIANT_AREA!r} is never issued")
    base = frame.format(name=name, ssn=base_ssn)
    variant = frame.format(name=name, ssn=variant_ssn)
    if not _is_single_replacement(base, base_ssn, variant, variant_ssn):
        raise ContrastiveDataError(f"ssn pair {index}: base and variant differ outside the area number")
    return ContrastivePair(
        id=_pair_id(id_prefix, SENSITIVITY_DOMAIN, "ssn", index),
        domain=SENSITIVITY_DOMAIN,
        kind="ssn",
        base=base,
        variant=variant,
        label_base=0,
        label_variant=1,
        provenance={
            "mutation": "the SSN area number (one field)",
            "base_value": base_ssn,
            "variant_value": variant_ssn,
            "base_fakeness": f"SSA never issues area number {area}; {SSN_RESERVED_CITATION}",
            "variant_fakeness": (
                f"area {SSN_VARIANT_AREA} is structurally issuable, so the shape is indistinguishable from a "
                "real SSN; the value is a constructed placeholder in a controlled corpus "
                "(evals/injection/gen_cases.py convention), never captured"
            ),
            "gate": "us_ssn blocks both members by design (shape, no area knowledge); the flip needs layer 2",
        },
    )


def _build_routing_pair(kind: str, rng: random.Random, index: int, id_prefix: str, ctx: _Ctx) -> ContrastivePair:
    """Same base task, one scope mutation: one-liner (cheap) vs refactor/migration (strong)."""
    subject, code, modules = rng.choice(ROUTING_TASKS)
    one_liner = rng.choice(ONE_LINER_FRAMES).format(subject=subject, code=code, bug=rng.choice(ONE_LINER_BUGS))
    if kind == "one_liner_vs_refactor":
        variant = rng.choice(REFACTOR_FRAMES).format(subject=subject, modules=modules, n=rng.randint(3, 14))
        mutation = f"task scope: one-line fix -> multi-file refactor across {modules}"
    else:
        variant = rng.choice(MIGRATION_FRAMES).format(subject=subject, rows=rng.choice(MIGRATION_ROW_COUNTS))
        mutation = f"task scope: one-line fix -> schema migration ({subject})"
    scorer = MockTierScorer(ctx.gate, ctx.mock)
    base_tier, variant_tier = scorer.score(one_liner), scorer.score(variant)
    if base_tier != "cheap" or variant_tier != "strong":
        raise ContrastiveDataError(
            f"routing pair {index} ({kind}): the keyless tier rules do not flip "
            f"(base={base_tier!r}, variant={variant_tier!r}); the pair would measure nothing"
        )
    return ContrastivePair(
        id=_pair_id(id_prefix, ROUTING_DOMAIN, kind, index),
        domain=ROUTING_DOMAIN,
        kind=kind,
        base=one_liner,
        variant=variant,
        label_base="cheap",
        label_variant="strong",
        provenance={
            "mutation": mutation,
            "base_task": subject,
            "tier_rule": "gate first, then MockBackend complexity: hard|frontier -> strong else cheap",
        },
    )


def _build_one_liner_vs_refactor(rng: random.Random, index: int, id_prefix: str, ctx: _Ctx) -> ContrastivePair:
    return _build_routing_pair("one_liner_vs_refactor", rng, index, id_prefix, ctx)


def _build_one_liner_vs_migration(rng: random.Random, index: int, id_prefix: str, ctx: _Ctx) -> ContrastivePair:
    return _build_routing_pair("one_liner_vs_migration", rng, index, id_prefix, ctx)


_BUILDERS: dict[str, Callable[[random.Random, int, str, _Ctx], ContrastivePair]] = {
    "card_luhn": _build_card_pair,
    "taj": _build_taj_pair,
    "ssn": _build_ssn_pair,
    "one_liner_vs_refactor": _build_one_liner_vs_refactor,
    "one_liner_vs_migration": _build_one_liner_vs_migration,
}


def _check_text_label(text_labels: dict[str, str], pair: ContrastivePair) -> None:
    """One text has exactly one correct decision, in or out of the pair role.

    The flip eval's oracle scorer is a text->label map; a split in which the
    same text carried two labels would make that map ambiguous (and the split
    untrainable on top of that). Checked pair by pair so the offender is named,
    not discovered as a flaky test later.
    """
    for text, label in ((pair.base, pair.label_base), (pair.variant, pair.label_variant)):
        prior = text_labels.get(text)
        if prior is not None and prior != str(label):
            raise ContrastiveDataError(
                f"pair {pair.id}: text already seen with label {prior!r}, now {str(label)!r}; "
                "one text must carry one decision"
            )
        text_labels[text] = str(label)


#: Retries when a generated pair collides with an excluded or already-seen text.
#: The pools are large enough that this is a guard against an exhausted template
#: table, not an expected path; hitting it names the kind so the table can be
#: widened instead of the split quietly shrinking.
_MAX_TEXT_RETRIES = 1000


def generate_pairs(
    seed: str,
    split: Mapping[str, int] | None = None,
    *,
    id_prefix: str = "ce",
    exclude_texts: Iterable[str] | None = None,
) -> list[ContrastivePair]:
    """Generate one contrastive split. Pure function of its arguments.

    ``seed`` is the only source of randomness: the same seed, split and excluded
    texts produce the same pairs, in the same order, in any process on any
    machine. Different seeds are meant to be used for different splits (train vs
    eval); to make the two splits *absolutely* disjoint, generate one first and
    pass its texts (both slots of every pair) as ``exclude_texts`` when
    generating the other. A colliding pair is redrawn from the same RNG stream,
    so exclusion stays deterministic: the exclusion set is an input, not luck.

    Every emitted pair has been checked: label properties (Luhn, SSN area, TAJ
    gate flip) and -- for routing -- the tier the keyless rules actually assign.
    A pair that fails a check raises :class:`ContrastiveDataError` instead of
    being emitted, the same fail-closed posture as
    :func:`jev_route.distill.synthetic_pii.verify_fakeness`.
    """
    counts = dict(split if split is not None else EVAL_SPLIT)
    known = set(KINDS_BY_DOMAIN[SENSITIVITY_DOMAIN]) | set(KINDS_BY_DOMAIN[ROUTING_DOMAIN])
    unknown = sorted(set(counts) - known)
    if unknown:
        raise ContrastiveDataError(f"unknown pair kinds {unknown}; known: {sorted(known)}")
    if any(n < 0 for n in counts.values()):
        raise ContrastiveDataError(f"split counts must be >= 0, got {counts}")
    excluded: set[str] = set(exclude_texts or ())
    ctx = _Ctx(gate=HardGate(), mock=MockBackend())
    rng = random.Random(f"contrastive-{seed}")
    pairs: list[ContrastivePair] = []
    text_labels: dict[str, str] = {}
    for domain in PAIR_DOMAINS:
        for kind in KINDS_BY_DOMAIN[domain]:
            builder = _BUILDERS[kind]
            for index in range(counts.get(kind, 0)):
                pair = _draw_until_fresh(rng, builder, index, id_prefix, ctx, excluded, text_labels, kind)
                _check_text_label(text_labels, pair)
                pairs.append(pair)
    return pairs


def _draw_until_fresh(  # noqa: PLR0917 -- the draw needs the full context tuple; grouping it would add a class to hide one
    rng: random.Random,
    builder: Callable[[random.Random, int, str, _Ctx], ContrastivePair],
    index: int,
    id_prefix: str,
    ctx: _Ctx,
    excluded: set[str],
    text_labels: dict[str, str],
    kind: str,
) -> ContrastivePair:
    """Draw pairs until none of its texts was excluded or already used in the split."""
    for _attempt in range(_MAX_TEXT_RETRIES):
        pair = builder(rng, index, id_prefix, ctx)
        fresh = (
            pair.base not in excluded and pair.variant not in excluded
            and pair.base not in text_labels and pair.variant not in text_labels
        )
        if fresh:
            return pair
            return pair
    raise ContrastiveDataError(
        f"kind {kind!r}: {index} pairs requested but {_MAX_TEXT_RETRIES} draws all collided with excluded "
        "or already-used texts; widen the template pools"
    )


def write_pairs_jsonl(pairs: Sequence[ContrastivePair], path: str) -> None:
    """Write pairs as deterministic JSONL: one pair per line, stable key order, UTF-8."""
    lines = [json.dumps(pair.to_dict(), ensure_ascii=False, sort_keys=False) for pair in pairs]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


__all__ = [
    "EVAL_SPLIT",
    "KINDS_BY_DOMAIN",
    "ROUTING_DOMAIN",
    "SENSITIVITY_DOMAIN",
    "TRAIN_SPLIT",
    "ContrastiveDataError",
    "ContrastivePair",
    "GateSensitivityScorer",
    "KeylessFlipScorer",
    "MockTierScorer",
    "generate_pairs",
    "write_pairs_jsonl",
]
