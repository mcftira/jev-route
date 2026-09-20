"""Synthetic sensitivity training data: the gate, made learnable without ever seeing your secrets.

THE BOOTSTRAP PARADOX
---------------------
The gate cannot ask the cloud whether something is safe to send to the cloud. If
it could, the cloud would already have seen the thing the gate exists to keep
out. So the deterministic floor never asks a model -- it is regexes, checksums
and fixed keyword lists -- and the sensitivity model is trained only on
synthetic and public positives plus real production negatives, never on real
blocked content.

**The gate never trains on your secrets.**

That sentence is the reason this module exists, and every rule below is a
consequence of it rather than an independent design choice.

The three (and only three) allowed positive sources, plus one negative source:

===========================================  =====================  ==========================
Source                                         Label role             May go to the cloud?
===========================================  =====================  ==========================
Synthetic PII generator (format-faithful,      positive               yes
  provably fake)
Contextual-sensitivity templates (sensitive    positive               yes
  with no regex hit at all)
Public annotated PII corpora, documented       positive               yes, and only if the local
  licences only                                                       gate does not block the row
Real production traffic that PASSED the gate   negative/borderline    yes (already clean)
===========================================  =====================  ==========================

:-data:`ALLOWED_SOURCES` is that table, and :class:`SensitivitySample` will not
construct with any other ``source``. "Anything else must be impossible to
express" is enforced by a ``Literal`` plus a ``__post_init__`` check, not by a
docstring.

THE HARD PROHIBITION
--------------------
No code path here reads blocked-request content. A gate-blocked decision record
contributes at most its metadata, and only when the operator explicitly asks for
it with ``include_blocked_metadata=True`` in ``features`` mode.

This is enforced three ways, because a comment is not a control:

1. **By construction.** :func:`_metadata_only_projection` rebuilds a blocked
   record field by field, naming only the text-free ones, and hands
   :func:`jev_route.distill.export.build_row` an object from which the stored
   text is not reachable at all.
2. **By mode.** A blocked record is only ever passed to ``build_row`` with
   ``mode="features"``, which cannot carry text into a row.
3. **By test.** ``tests/distill/test_sensitivity_data.py`` writes a decision log
   containing a gate-blocked record whose stored text is a recognisable canary,
   builds a dataset from that log, and asserts the canary appears nowhere in the
   output -- not in a row, not in the serialized JSONL, not in the stats sidecar,
   not in the audit trail. A second test walks this module's AST and asserts no
   function reads an ``excerpt`` attribute or subscript, and that the only
   ``excerpt=`` keyword in the file is the erasure in the projection.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not invent labels. The teacher's *full soft distribution* over all four
heads is the label, recorded exactly as the backend returned it, because the
project's whole discipline is that distillation runs on distributions and not on
argmax. A generator's ``expected_sensitivity`` is the author's intent; it is
carried in provenance and reported as an agreement rate, never used as a target.

It does not bundle or download any corpus. :func:`load_public_corpus` reads a
local path the operator already has, refuses a URL, and refuses a corpus whose
licence has not been stated.

It does not wire itself into the CLI. :func:`label_sensitivity` and
:func:`synthesize_sensitivity_dataset` are library entry points.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from ..backends.base import DecisionBackend, DecisionRequest
from ..backends.mock import MockBackend
from ..gate import HardGate, default_gate
from ..logging_sink import DecisionSink
from ..prompts import compute_features, hash_text, safe_caller_metadata
from ..schema import (
    SCHEMA_VERSION,
    SENSITIVITY_LEVELS,
    DecisionAnswers,
    DecisionRecord,
    GateVerdict,
    RequestFeatures,
    RoutingDecision,
    level_index,
)
from .export import CHOICE_HEADS, ROW_SCHEMA, Dataset, TrainingRow, build_row, iter_source_records
from .synthetic_pii import (
    GENERATORS,
    CardFakeness,
    ContextualTemplate,
    IbanFakeness,
    SyntheticPiiError,
    SyntheticValue,
    check_expected_gate,
    contextual_gate_hit,
    contextual_samples,
    gate_expectation,
    generate,
    person_name,
)

#: Bump when the audit record layout changes. Stamped on every audit line so a
#: file written by an older build is detectably older rather than silently
#: reinterpreted -- the same rule the decision log follows.
AUDIT_RECORD_KIND = "jev_route.sensitivity_labeling"
AUDIT_SCHEMA_VERSION = "1"

#: A synthetic sample never went through the router, so no tier exists for it.
#: ``teacher.tier`` gets this sentinel rather than a guess, because
#: :mod:`jev_route.distill.evaluate` compares that field against a policy replay:
#: a fabricated tier would not merely be wrong, it would quietly drag the tier
#: agreement number down and make a good student look bad.
UNROUTED_TIER = "unrouted"
SYNTHETIC_RULE_ID = "synthetic.sensitivity-data"

#: Deterministic epoch for synthetic timestamps. Wall-clock timestamps would make
#: two runs over the same seed produce different files, and the split is keyed on
#: ``request_id`` while the dataset hash covers every byte -- reproducibility is
#: the property that lets an operator diff two builds of the same dataset.
_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)

#: Refusal reasons recorded in the audit trail. Strings, not an enum, so they
#: survive a round trip through JSONL unchanged and can be counted by a shell.
REFUSAL_GATE_BLOCKED_CORPUS = "gate-blocked-public-corpus-row"
REFUSAL_GATE_BLOCKED_PRODUCTION = "gate-blocked-production-record"
REFUSAL_DEGRADED_TEACHER = "degraded-teacher-answer"


class SensitivityDataError(ValueError):
    """The dataset cannot be built as asked. Always actionable."""


class UnknownSourceError(SensitivityDataError):
    """A sample named a source outside :data:`ALLOWED_SOURCES`."""


class CorpusLicenceError(SensitivityDataError):
    """A public corpus was offered without a stated licence."""


class CloudSendRefusedError(SensitivityDataError):
    """A sample is not provably safe to show a remote labeler."""


# --------------------------------------------------------------------------- #
# The allowed sources -- the table from the module docstring, as data
# --------------------------------------------------------------------------- #
SourceCategory = Literal["synthetic-pii", "contextual-template", "public-corpus", "production-negative"]


@dataclass(frozen=True)
class SourcePolicy:
    """One row of the allowed-sources table.

    :attr:`cloud_condition` is a predicate over ``(sample, verdict)`` rather than
    a flag, because the answer genuinely differs per row: two categories are
    unconditionally sendable *because their fakeness is established by
    construction*, one is sendable only when the local gate does not block it, and
    one is sendable because it already passed the gate in production.
    """

    category: SourceCategory
    label_role: Literal["positive", "negative/borderline"]
    may_go_to_cloud: str
    cloud_condition: Callable[[SensitivitySample, GateVerdict], bool]
    #: Why the row above is allowed to leave the process. Printed by the CLI and
    #: stored in the stats sidecar, so an operator reading a dataset can see the
    #: argument rather than having to find it in the source.
    rationale: str


def _always_sendable(_sample: SensitivitySample, _verdict: GateVerdict) -> bool:
    return True


def _sendable_if_gate_allows(_sample: SensitivitySample, verdict: GateVerdict) -> bool:
    return not verdict.blocks_backend


ALLOWED_SOURCES: tuple[SourcePolicy, ...] = (
    SourcePolicy(
        category="synthetic-pii",
        label_role="positive",
        may_go_to_cloud="yes",
        cloud_condition=_always_sendable,
        rationale=(
            "every identifier is format-faithful and provably fake by a published reservation "
            "(SSA never-issued area numbers, NANP fictitious block, RFC 2606 domains, ISO 3166-1 "
            "user-assigned IBAN country codes, vendor-published test PANs, self-labelled credentials). "
            "The reservation is what earns the right to send it."
        ),
    ),
    SourcePolicy(
        category="contextual-template",
        label_role="positive",
        may_go_to_cloud="yes",
        cloud_condition=_always_sendable,
        rationale=(
            "hand-written prose that is sensitive while containing no identifier and no detector hit. "
            "Verified against the real HardGate at generation time, so 'layer 1 cannot see this' is a "
            "measured property, not an intention."
        ),
    ),
    SourcePolicy(
        category="public-corpus",
        label_role="positive",
        may_go_to_cloud="yes, if the local gate does not block the row",
        cloud_condition=_sendable_if_gate_allows,
        rationale=(
            "third-party annotated data with a stated licence. We cannot prove a stranger's corpus is "
            "fake the way we can prove our own generator is, so a row that trips a blocking detector is "
            "kept out of the cloud call and recorded as a refusal."
        ),
    ),
    SourcePolicy(
        category="production-negative",
        label_role="negative/borderline",
        may_go_to_cloud="yes (already clean)",
        cloud_condition=_sendable_if_gate_allows,
        rationale=(
            "real traffic the local gate already let through, so it is clean by the same judgement that "
            "would have stopped it. Gate-blocked records are excluded outright: their content is exactly "
            "what this module exists never to touch."
        ),
    ),
)

SOURCE_CATEGORIES: tuple[str, ...] = tuple(s.category for s in ALLOWED_SOURCES)
_SOURCE_POLICIES: dict[str, SourcePolicy] = {s.category: s for s in ALLOWED_SOURCES}


def source_policy(category: str) -> SourcePolicy:
    """The policy row for ``category``, or an error naming the allowed set."""
    try:
        return _SOURCE_POLICIES[category]
    except KeyError:
        raise UnknownSourceError(
            f"unknown source category {category!r}. This module can only express {SOURCE_CATEGORIES}; "
            "the list is closed on purpose, because the bootstrap paradox has no fourth answer."
        ) from None


# --------------------------------------------------------------------------- #
# The sample
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SensitivitySample:
    """One training example, with the provenance that makes it safe to label.

    The constructor is the enforcement point for the closed source list: there is
    no way to build a sample whose ``source`` is not one of
    :data:`SOURCE_CATEGORIES`, and the per-source requirements (a licence for a
    corpus, a fakeness basis for a synthetic identifier) are checked here rather
    than left to whichever builder remembered to check them.
    """

    sample_id: str
    text: str
    source: SourceCategory
    #: Which generator or template produced this. Empty for production traffic.
    generator: str = ""
    #: The reservation the fakeness claim rests on, and where it is documented.
    fakeness_basis: str = ""
    reservation: str = ""
    citation: str = ""
    #: The author's intent. Provenance, never a target: the teacher's soft
    #: distribution is the label. See the module docstring.
    expected_sensitivity: str = ""
    expected_pii: float | None = None
    #: Required for ``public-corpus``. Not invented: whatever the operator states.
    licence: str = ""
    licence_url: str = ""
    corpus: str = ""
    #: Detector names that fired, measured by the real gate. Stored so the
    #: dataset can report layer-1 coverage without re-scanning.
    gate_detectors: Mapping[str, int] = field(default_factory=dict)
    gate_expectation: str = "none"
    detail: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source_policy(self.source)  # raises UnknownSourceError on anything else
        if not self.text.strip():
            raise SensitivityDataError(f"{self.sample_id}: a sample with empty text carries no signal")
        if self.expected_sensitivity and self.expected_sensitivity not in SENSITIVITY_LEVELS:
            raise SensitivityDataError(
                f"{self.sample_id}: expected_sensitivity must be one of {SENSITIVITY_LEVELS} or empty, "
                f"got {self.expected_sensitivity!r}"
            )
        if self.source == "public-corpus":
            if not self.licence.strip():
                raise CorpusLicenceError(
                    f"{self.sample_id}: a public-corpus sample requires the 'licence' field to be filled in"
                )
            if not self.corpus.strip():
                raise SensitivityDataError(
                    f"{self.sample_id}: a public-corpus sample requires 'corpus' to name the corpus"
                )
        if self.source == "synthetic-pii" and not self.fakeness_basis:
            raise SensitivityDataError(
                f"{self.sample_id}: a synthetic-pii sample must carry the fakeness_basis its generator claimed"
            )

    @property
    def label_role(self) -> str:
        return source_policy(self.source).label_role

    def provenance(self) -> dict[str, Any]:
        """The source-attribution block written onto every exported row."""
        return {
            "source_category": self.source,
            "label_role": self.label_role,
            "generator": self.generator,
            "fakeness_basis": self.fakeness_basis,
            "reservation": self.reservation,
            "citation": self.citation,
            "expected_sensitivity": self.expected_sensitivity,
            "licence": self.licence,
            "licence_url": self.licence_url,
            "corpus": self.corpus,
            "detail": dict(self.detail),
        }


# --------------------------------------------------------------------------- #
# Synthetic-PII samples: identifier + a sentence it could plausibly sit in
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RecordSentence:
    """A sentence frame for one identifier kind.

    The frame matters because a bare ``666-45-6789`` is not a routing request; it
    is a string. The frames below are written the way such a value actually arrives
    -- inside a task -- so the sample teaches sensitivity in context and the
    complexity and domain heads get something real to read.
    """

    kind: str
    template: str
    expected_sensitivity: str


#: One or more frames per identifier kind, as plain tuples so the table stays
#: readable inside the line limit. Order is not significant.
_SENTENCE_TABLE: tuple[tuple[str, str, str], ...] = (
    ("us_ssn", "Process the payroll change for employee named {name}. Their SSN is {value}.", "regulated"),
    ("us_ssn", "The background check came back for applicant {name}, SSN {value}. Summarise it.", "regulated"),
    ("payment_card", "Customer {name} disputes a charge on card {value}; it cleared Tuesday.", "regulated"),
    ("payment_card", "Refund {value} and explain why the customer has to wait five working days.", "regulated"),
    ("email_address", "Send the onboarding pack to {value} and copy the hiring manager.", "confidential"),
    ("phone_number", "Call {value} before five and confirm the appointment for {name}.", "confidential"),
    ("iban", "Set up the standing order to {value} for {name}, starting on the first.", "regulated"),
    ("uk_nhs_number", "The referral form lists NHS number {value} for {name}; the surname differs.", "regulated"),
    ("uk_national_insurance", "Payroll rejected {value} for the new starter {name}. Check the format.", "regulated"),
    ("medical_record_number", "Pull the notes for {value} and check whether the follow-up was booked.", "regulated"),
    ("date_of_birth", "Update the record for {name}, {value}, before the audit on Friday.", "regulated"),
    ("person_name", "The applicant, {value}, asked us to correct the spelling on their file.", "confidential"),
    ("provider_api_key", "The staging build stopped working after we rotated {value}. Why?", "confidential"),
    ("aws_access_key_id", "Put {value} into the deployment secrets and redeploy the worker.", "confidential"),
    (
        "private_key_block",
        "Here is the signing key we lost, please tell me if it is recoverable:\n{value}",
        "confidential",
    ),
    ("inline_credential", "The compose file still has {value} in it and the review failed.", "confidential"),
    ("basic_auth_url", "The connection string is {value}; it works locally, not from the runner.", "confidential"),
)

RECORD_SENTENCES: tuple[RecordSentence, ...] = tuple(
    RecordSentence(kind=kind, template=template, expected_sensitivity=expected)
    for kind, template, expected in _SENTENCE_TABLE
)

_SENTENCES_BY_KIND: dict[str, tuple[RecordSentence, ...]] = {}
for _sentence in RECORD_SENTENCES:
    _SENTENCES_BY_KIND[_sentence.kind] = (*_SENTENCES_BY_KIND.get(_sentence.kind, ()), _sentence)

#: Used for a kind with no frame of its own. Unreachable with the shipped
#: generator set, but a missing frame should produce a usable sample rather than a
#: KeyError from ``str.format``.
_FALLBACK_SENTENCE_TEMPLATE = "Please look at {value} and tell me what to do next."


def _sentence_for(kind: str, rng: random.Random) -> RecordSentence:
    options = _SENTENCES_BY_KIND.get(kind)
    if not options:  # pragma: no cover - every generated kind has a frame
        return RecordSentence(kind, _FALLBACK_SENTENCE_TEMPLATE, "confidential")
    return rng.choice(options)


def synthetic_pii_samples(
    rng: random.Random,
    *,
    per_kind: int = 6,
    kinds: Iterable[str] | None = None,
    card_fakeness: str = CardFakeness.RESERVED_TEST_RANGE,
    iban_fakeness: str = IbanFakeness.RESERVED_COUNTRY,
    email_domain_pool: str = "rfc2606",
    gate: HardGate | None = None,
    id_prefix: str = "",
) -> list[SensitivitySample]:
    """Generate format-faithful, provably fake identifier samples.

    Every value is built by :mod:`jev_route.distill.synthetic_pii`, whose
    constructor verifies the fakeness reservation before returning, and every
    value's documented gate behaviour is re-checked here against the real
    :class:`~jev_route.gate.HardGate` with :func:`check_expected_gate`. A sample
    that would misdescribe layer 1 raises instead of being emitted.

    ``kinds=None`` covers every generator, which is what you want at least once:
    it guarantees the synthetic set touches every blocking detector the gate has.
    """
    active = gate or default_gate
    wanted = tuple(kinds) if kinds is not None else tuple(sorted(GENERATORS))
    unknown = set(wanted) - set(GENERATORS)
    if unknown:
        raise SyntheticPiiError(f"unknown identifier kinds {sorted(unknown)}; known: {sorted(GENERATORS)}")
    if per_kind < 0:
        raise SensitivityDataError(f"per_kind must be >= 0, got {per_kind}")

    out: list[SensitivitySample] = []
    seen: set[str] = set()
    counter = 0
    for kind in wanted:
        for _ in range(per_kind):
            value = generate(kind, rng, **_generator_options(kind, card_fakeness, iban_fakeness, email_domain_pool))
            # Re-check the documented layer-1 behaviour against the live gate. This
            # is the line that stops this module's docstrings from going stale.
            check_expected_gate(value, active)
            sentence = _sentence_for(kind, rng)
            name = person_name(rng) if "{name}" in sentence.template else None
            text = sentence.template.format(value=value.rendered, name=name.rendered if name else "")
            if text in seen:
                # Two generators return a constant by design (the published AWS
                # documentation key), so per_kind>1 would emit byte-identical rows.
                # Identical text under different request_ids is worse than fewer
                # rows: the split is keyed on request_id, so the same example can
                # land on both sides and make every holdout number optimistic.
                continue
            seen.add(text)
            counter += 1
            out.append(
                _sample_from_value(
                    value=value, sentence=sentence, text=text, counter=counter, id_prefix=id_prefix, gate=active
                )
            )
    return out


def _generator_options(kind: str, card_fakeness: str, iban_fakeness: str, email_domain_pool: str) -> dict[str, Any]:
    """Which knobs apply to ``kind``, so one option set can fan out over fifteen generators."""
    if kind == "payment_card":
        return {"fakeness": card_fakeness}
    if kind == "iban":
        return {"fakeness": iban_fakeness}
    if kind == "email_address":
        return {"domain_pool": email_domain_pool}
    return {}


def _sample_from_value(
    *,
    value: SyntheticValue,
    sentence: RecordSentence,
    text: str,
    counter: int,
    id_prefix: str,
    gate: HardGate,
) -> SensitivitySample:
    verdict = gate.scan(text)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return SensitivitySample(
        sample_id=f"{id_prefix or 'spii'}-{counter:06d}-{digest}",
        text=text,
        source="synthetic-pii",
        generator=value.kind,
        fakeness_basis=value.basis,
        reservation=value.reservation,
        citation=value.citation,
        expected_sensitivity=sentence.expected_sensitivity,
        expected_pii=0.9 if value.kind == "person_name" else 1.0,
        gate_detectors=verdict.detectors(),
        gate_expectation=gate_expectation(verdict),
        detail={**value.detail, "sentence_kind": sentence.kind, "value_expected_gate": value.expected_gate},
    )


def contextual_positive_samples(
    rng: random.Random,
    *,
    n: int | None = None,
    categories: Iterable[str] | None = None,
    gate: HardGate | None = None,
    id_prefix: str = "",
) -> list[SensitivitySample]:
    """Sensitive prose with no detector hit, verified against the real gate.

    :func:`~jev_route.distill.synthetic_pii.contextual_samples` raises if any
    chosen template fires the gate, so every sample returned here is provably a
    layer-2-only example. That is the whole reason this category exists: without
    it the model would learn "sensitive == some regex matched", which is the
    keyword-router failure this project was written to fix.
    """
    templates = contextual_samples(rng, n=n, categories=categories, gate=gate)
    out: list[SensitivitySample] = []
    for index, template in enumerate(templates, start=1):
        hits = contextual_gate_hit(template.text, gate or default_gate)
        if hits:  # pragma: no cover - contextual_samples already checked this
            raise SensitivityDataError(f"contextual template {template.category!r} fires {hits}")
        out.append(contextual_sample_from_template(template, index, id_prefix))
    return out


def contextual_sample_from_template(template: ContextualTemplate, index: int, id_prefix: str = "") -> SensitivitySample:
    """One :class:`ContextualTemplate` as a sample. Public so a caller can hand-write templates."""
    text = template.text
    return SensitivitySample(
        sample_id=f"{id_prefix or 'ctx'}-{index:06d}-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]}",
        text=text,
        source="contextual-template",
        generator=template.category,
        fakeness_basis="",
        reservation="hand-written prose; contains no identifier, so there is nothing to reserve",
        citation=template.why_regex_cannot_see_it,
        expected_sensitivity=template.expected_sensitivity,
        expected_pii=0.0,
        gate_detectors={},
        gate_expectation="none",
        detail={"category": template.category},
    )


# --------------------------------------------------------------------------- #
# Public corpora: a licence is not optional
# --------------------------------------------------------------------------- #
#: Values that look like a licence field but state nothing. Compared
#: case-insensitively after stripping, so ``"Unknown"`` and ``"  TBD "`` are both
#: refusals. A real licence name is never refused: this module does not maintain a
#: list of acceptable licences, because inventing one would mean inventing legal
#: judgements about somebody else's data.
LICENCE_FIELD = "licence"
LICENCE_NOT_STATED: frozenset[str] = frozenset(
    {
        "",
        "unknown",
        "unspecified",
        "none",
        "null",
        "n/a",
        "na",
        "tbd",
        "tbc",
        "todo",
        "pending",
        "?",
        "-",
        "see licence",
    }
)


@dataclass(frozen=True)
class CorpusRow:
    """One annotated row of a public corpus."""

    text: str
    row_id: str = ""
    #: The corpus's own label, if it has one. Carried into provenance for
    #: comparison against the teacher; never used as the target.
    sensitivity: str = ""
    pii: float | None = None
    annotation: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PublicCorpus:
    """A public annotated PII corpus with its licence *stated*.

    ``licence`` is required at construction time, which is the point: the refusal
    happens when the corpus is described, not later when somebody tries to ship a
    dataset built from data nobody checked the terms of. Nothing here is bundled
    and nothing is downloaded -- :func:`load_public_corpus` reads a path the
    operator already has on disk.
    """

    name: str
    licence: str
    rows: tuple[CorpusRow, ...] = ()
    licence_url: str = ""
    citation: str = ""
    homepage: str = ""
    retrieved_at: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise CorpusLicenceError("a public corpus must have a 'name'")
        stated = self.licence.strip()
        if stated.lower() in LICENCE_NOT_STATED:
            raise CorpusLicenceError(
                f"corpus {self.name!r} has no stated licence. Fill in the `{LICENCE_FIELD}` field with the "
                "exact licence the corpus is published under (an SPDX identifier such as 'CC-BY-4.0', or "
                "the verbatim licence name), and `licence_url` if it has one. jev-route refuses to build "
                "training data from a corpus whose redistribution terms nobody has checked: the dataset "
                "is written to disk and its rows may be sent to a remote labeler."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "licence": self.licence,
            "licence_url": self.licence_url,
            "citation": self.citation,
            "homepage": self.homepage,
            "retrieved_at": self.retrieved_at,
            "rows": len(self.rows),
        }

    def samples(self, *, limit: int | None = None, id_prefix: str = "") -> list[SensitivitySample]:
        """This corpus as samples. Licence and citation travel on every one."""
        out: list[SensitivitySample] = []
        for index, row in enumerate(self.rows[:limit] if limit is not None else self.rows, start=1):
            if not row.text.strip():
                continue
            verdict = default_gate.scan(row.text)
            out.append(
                SensitivitySample(
                    sample_id=row.row_id or f"{id_prefix or 'cor'}-{index:06d}",
                    text=row.text,
                    source="public-corpus",
                    generator=f"corpus:{self.name}",
                    fakeness_basis="",
                    reservation=f"third-party corpus under a stated licence: {self.licence}",
                    citation=self.citation or self.homepage or self.name,
                    expected_sensitivity=row.sensitivity if row.sensitivity in SENSITIVITY_LEVELS else "",
                    expected_pii=row.pii,
                    licence=self.licence,
                    licence_url=self.licence_url,
                    corpus=self.name,
                    gate_detectors=verdict.detectors(),
                    gate_expectation=gate_expectation(verdict),
                    detail={"annotation": dict(row.annotation)},
                )
            )
        return out


def load_public_corpus(
    path: str | Path,
    *,
    name: str,
    licence: str,
    licence_url: str = "",
    citation: str = "",
    homepage: str = "",
    retrieved_at: str = "",
    text_field: str = "text",
    sensitivity_field: str = "sensitivity",
    pii_field: str = "pii",
    id_field: str = "id",
    limit: int | None = None,
) -> PublicCorpus:
    """Read a JSONL corpus the operator already has. Never fetches anything.

    A ``path`` that looks like a URL is refused rather than downloaded. That is not
    caution about the network: it is because a corpus this module pulled would have
    no ``retrieved_at``, no provenance, and no way for a reviewer to check the
    licence claim against the thing that was actually retrieved.

    Args:
        licence: required. See :class:`PublicCorpus`; an unstated licence raises
            :class:`CorpusLicenceError` naming this field.
        text_field/sensitivity_field/pii_field/id_field: the corpus's own column
            names, because there is no standard and guessing one silently produces
            empty text.
    """
    raw = str(path)
    if "://" in raw:
        raise SensitivityDataError(
            f"refusing to fetch a corpus from {raw!r}. Download it yourself, check its licence, then pass "
            "the local path. This module never performs network I/O."
        )
    source = Path(raw)
    if not source.exists():
        raise SensitivityDataError(f"corpus file not found: {source}. Pass a local JSONL path you already have.")

    rows: list[CorpusRow] = []
    with open(source, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SensitivityDataError(f"{source}: line {len(rows) + 1} is not valid JSON ({exc})") from exc
            if not isinstance(payload, Mapping):
                raise SensitivityDataError(f"{source}: line {len(rows) + 1} is not a JSON object")
            text = str(payload.get(text_field) or "")
            if not text.strip():
                continue
            pii_raw = payload.get(pii_field)
            annotation = {k: v for k, v in payload.items() if k not in (text_field, id_field)}
            rows.append(
                CorpusRow(
                    text=text,
                    row_id=str(payload.get(id_field) or ""),
                    sensitivity=str(payload.get(sensitivity_field) or ""),
                    pii=float(pii_raw) if isinstance(pii_raw, (int, float)) and not isinstance(pii_raw, bool) else None,
                    annotation=annotation,
                )
            )
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise SensitivityDataError(
            f"{source} produced no rows. Is the text in a field called {text_field!r}? Pass text_field= "
            "with the corpus's own column name."
        )
    return PublicCorpus(
        name=name,
        licence=licence,
        rows=tuple(rows),
        licence_url=licence_url,
        citation=citation,
        homepage=homepage,
        retrieved_at=retrieved_at,
    )


# --------------------------------------------------------------------------- #
# Production negatives: metadata only, and never from a blocked request
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProductionIntake:
    """What a decision log contributed, and what it was refused from contributing.

    The counts are the audit: ``blocked_discarded`` is the number of records whose
    content this module declined to read, and it should appear in the dataset
    stats next to the rows that were written.
    """

    records: tuple[DecisionRecord, ...] = ()
    blocked_discarded: int = 0
    blocked_metadata_only: int = 0
    degraded_skipped: int = 0
    unsupported_schema_skipped: int = 0
    read: int = 0
    notes: tuple[str, ...] = ()


def _metadata_only_projection(record: DecisionRecord) -> DecisionRecord:
    """A blocked record reduced to its text-free fields.

    Built by naming the safe fields one at a time instead of copying the record
    and blanking a field, so the stored text is not reachable from the returned
    object even by accident. ``questions_sent`` is emptied for the same reason: it
    would quote the prompt back, and a blocked record has no questions anyway
    because the router never called a backend. ``excerpt_hash`` is kept -- it is a
    digest, and it is how an operator deduplicates against their own log.

    This is the mechanical form of "a blocked record contributes at most its
    metadata".
    """
    return DecisionRecord(
        request_id=record.request_id,
        timestamp=record.timestamp,
        decision=record.decision,
        features=record.features,
        excerpt_hash=record.excerpt_hash,
        backend_latency_ms=record.backend_latency_ms,
        total_latency_ms=record.total_latency_ms,
        questions_sent={},
        excerpt=None,
        metadata=safe_caller_metadata(record.metadata),
        requested_model=record.requested_model,
        shadow=None,
    )


def production_negative_samples(
    source: str | Path | Iterable[DecisionRecord],
    *,
    limit: int | None = None,
    include_blocked_metadata: bool = False,
    pattern: str = "decisions*.jsonl",
) -> ProductionIntake:
    """Harvest real traffic that passed the gate, as negatives and borderlines.

    This is the only source here that is not synthetic, and it is allowed exactly
    because the gate already cleared it: the same deterministic judgement that
    would have stopped a card number is what makes the rest safe to reuse.

    Gate-blocked records are discarded. With ``include_blocked_metadata=True`` they
    contribute a metadata-only projection instead, and nothing else: no text, no
    questions, no matched spans. Even that projection is only usable in
    ``features`` mode, because a text-mode dataset is a promise that every row has
    text to train on.
    """
    notes: list[str] = []
    kept: list[DecisionRecord] = []
    blocked_discarded = 0
    blocked_metadata_only = 0
    degraded = 0
    bad_schema = 0
    read = 0
    for record in iter_source_records(source, pattern=pattern):
        read += 1
        if limit is not None and len(kept) >= limit:
            notes.append(f"stopped at --limit {limit}; {read} records read in total")
            break
        if record.schema_version != SCHEMA_VERSION:
            bad_schema += 1
            continue
        if record.decision.gate.blocks_backend or record.decision.backend == "gate":
            if include_blocked_metadata:
                kept.append(_metadata_only_projection(record))
                blocked_metadata_only += 1
            else:
                blocked_discarded += 1
            continue
        if record.decision.degraded:
            # A degraded decision is a uniform distribution: the shape of "the
            # backend was down". Training on it teaches the student to be unsure
            # about everything, so it is dropped here and counted.
            degraded += 1
            continue
        kept.append(replace(record, metadata=safe_caller_metadata(record.metadata)))
    if blocked_discarded:
        notes.append(
            f"{blocked_discarded} gate-blocked record(s) were discarded without their content being read. "
            "Pass include_blocked_metadata=True to keep a metadata-only projection instead."
        )
    if blocked_metadata_only:
        notes.append(
            f"{blocked_metadata_only} gate-blocked record(s) contributed metadata only: no text, no questions, "
            "no matched spans. Requires features mode."
        )
    if degraded:
        notes.append(f"{degraded} degraded record(s) were skipped: a failed backend logged uniform distributions")
    if bad_schema:
        notes.append(f"{bad_schema} record(s) had an unsupported schema_version and were skipped")
    return ProductionIntake(
        records=tuple(kept),
        blocked_discarded=blocked_discarded,
        blocked_metadata_only=blocked_metadata_only,
        degraded_skipped=degraded,
        unsupported_schema_skipped=bad_schema,
        read=read,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------- #
# Labeling: the only path that talks to a backend
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LabeledSample:
    """A sample plus the teacher's full soft answer.

    :attr:`answers` keeps all four heads and every probability the backend
    returned. Storing the argmax instead would throw away the calibration that is
    the entire reason to have paid for a cloud teacher, so there is no accessor
    here that returns only a label.
    """

    sample: SensitivitySample
    answers: DecisionAnswers
    gate: GateVerdict
    backend: str
    model_version: str
    latency_ms: float = 0.0
    degraded: bool = False
    degrade_reason: str | None = None
    #: The teacher's own report of what it was asked. Kept so a labeling run is
    #: reproducible from the audit trail alone.
    questions_sent: Mapping[str, Any] = field(default_factory=dict)

    @property
    def sensitivity_label(self) -> str:
        return self.answers.sensitivity.choice

    @property
    def sensitivity_probabilities(self) -> Mapping[str, float]:
        return self.answers.sensitivity.probabilities


@dataclass(frozen=True)
class LabelingAuditEntry:
    """One line of the labeling audit trail.

    Every call is recorded, including the ones that were refused -- a refusal that
    is not logged is indistinguishable from a sample that was never offered, and
    "we did not send your blocked content anywhere" is a claim an operator should
    be able to check against a file.
    """

    sample_id: str
    source_category: str
    label_role: str
    generator: str
    fakeness_basis: str
    licence: str
    excerpt_hash: str
    sent_to_backend: bool
    backend: str
    model_version: str
    latency_ms: float
    degraded: bool
    refusal_reason: str | None
    #: Human-readable form of the refusal. Separate from :attr:`refusal_reason`,
    #: which is a stable key meant to be counted; this one names the detectors that
    #: fired so an operator can see what was refused and why.
    refusal_detail: str
    gate_detectors: Mapping[str, int]
    gate_blocks_backend: bool
    sensitivity_probabilities: Mapping[str, float]
    sensitivity_label: str
    pii: float
    expected_sensitivity: str
    teacher_agrees_with_expected: bool | None
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": AUDIT_RECORD_KIND,
            "audit_schema_version": AUDIT_SCHEMA_VERSION,
            "timestamp": self.timestamp,
            "sample_id": self.sample_id,
            "source_category": self.source_category,
            "label_role": self.label_role,
            "generator": self.generator,
            "fakeness_basis": self.fakeness_basis,
            "licence": self.licence,
            "excerpt_hash": self.excerpt_hash,
            "sent_to_backend": self.sent_to_backend,
            "backend": self.backend,
            "model_version": self.model_version,
            "latency_ms": self.latency_ms,
            "degraded": self.degraded,
            "refusal_reason": self.refusal_reason,
            "refusal_detail": self.refusal_detail,
            "gate_detectors": dict(self.gate_detectors),
            "gate_blocks_backend": self.gate_blocks_backend,
            "sensitivity_probabilities": dict(self.sensitivity_probabilities),
            "sensitivity_label": self.sensitivity_label,
            "pii": self.pii,
            "expected_sensitivity": self.expected_sensitivity,
            "teacher_agrees_with_expected": self.teacher_agrees_with_expected,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LabelingAuditEntry:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class LabelingAudit:
    """The audit trail for one labeling run, in call order."""

    entries: list[LabelingAuditEntry] = field(default_factory=list)

    def add(self, entry: LabelingAuditEntry) -> None:
        self.entries.append(entry)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[LabelingAuditEntry]:
        return iter(self.entries)

    def sent(self) -> list[LabelingAuditEntry]:
        return [e for e in self.entries if e.sent_to_backend]

    def refused(self) -> list[LabelingAuditEntry]:
        return [e for e in self.entries if not e.sent_to_backend]

    def source_breakdown(self) -> dict[str, int]:
        """Calls per source category, so an audit can be read as "what did we send, and from where"."""
        out: dict[str, int] = {}
        for entry in self.entries:
            out[entry.source_category] = out.get(entry.source_category, 0) + 1
        return out

    def refusal_reasons(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for entry in self.refused():
            reason = entry.refusal_reason or "unspecified"
            out[reason] = out.get(reason, 0) + 1
        return out

    def to_jsonl(self) -> str:
        return "".join(e.to_json() + "\n" for e in self.entries)

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_jsonl(), encoding="utf-8")
        return target


@dataclass(frozen=True)
class LabelingResult:
    """What a labeling run produced."""

    labeled: tuple[LabeledSample, ...]
    refused: tuple[SensitivitySample, ...]
    audit: LabelingAudit

    @property
    def n_sent(self) -> int:
        return len(self.labeled)

    @property
    def n_refused(self) -> int:
        return len(self.refused)

    def source_breakdown(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for item in self.labeled:
            out[item.sample.source] = out.get(item.sample.source, 0) + 1
        return out

    def summary(self) -> str:
        lines = [
            f"labeled {self.n_sent} samples with "
            f"{sorted({item.backend for item in self.labeled}) or ['(none)']}; refused {self.n_refused}"
        ]
        for category, count in sorted(self.source_breakdown().items()):
            lines.append(f"  {category:<22} {count}")
        for reason, count in sorted(self.audit.refusal_reasons().items()):
            lines.append(f"  refused: {reason} x{count}")
        return "\n".join(lines)


def may_go_to_cloud(sample: SensitivitySample, verdict: GateVerdict) -> tuple[bool, str | None]:
    """Whether ``sample`` may be shown to a remote labeler, and why not if it may not.

    The answer comes from :data:`ALLOWED_SOURCES`, so the rule and the table in the
    module docstring cannot drift apart. Note that a *blocking* gate verdict does
    not by itself refuse a synthetic sample: the identifiers in it are reserved
    values, and a deterministic hit on ``666-45-6789`` is a hit on a number the SSA
    will never issue. Refusing those would leave the synthetic positives with no
    blocking-detector coverage at all, which is the coverage that matters most.
    """
    policy = source_policy(sample.source)
    if policy.cloud_condition(sample, verdict):
        return True, None
    detectors = sorted(verdict.detectors())
    return False, (
        f"a blocking detector fired ({detectors}), and {policy.category} rows are not provably fake "
        "by construction the way this module's own generators are"
    )


def refusal_reason(sample: SensitivitySample, verdict: GateVerdict) -> str:
    """The short, stable refusal key recorded in the audit trail.

    Kept separate from :func:`may_go_to_cloud`'s message because the audit file is
    meant to be counted with a shell pipeline: these values are a closed set, the
    human-readable message is not.
    """
    del verdict  # the reason depends only on which source table row refused
    if sample.source == "public-corpus":
        return REFUSAL_GATE_BLOCKED_CORPUS
    return REFUSAL_GATE_BLOCKED_PRODUCTION


def assert_sendable(sample: SensitivitySample, verdict: GateVerdict) -> None:
    """Raise :class:`CloudSendRefusedError` unless ``sample`` may be shown to a remote labeler.

    The raising form of :func:`may_go_to_cloud`, for a caller that would rather fail
    than skip. :func:`label_sensitivity` uses the tuple form, because refusing one
    corpus row must not abort a labeling run: it has to be recorded, and the run
    has to continue.
    """
    sendable, why = may_go_to_cloud(sample, verdict)
    if not sendable:
        raise CloudSendRefusedError(f"{sample.sample_id}: not sendable to a remote labeler. {why}")


async def label_sensitivity(
    samples: Iterable[SensitivitySample],
    *,
    backend: DecisionBackend | None = None,
    gate: HardGate | None = None,
    max_concurrency: int = 4,
    metadata: Mapping[str, Any] | None = None,
    audit: LabelingAudit | None = None,
    audit_sink: DecisionSink | None = None,
) -> LabelingResult:
    """Send the allowed sources to a decision backend and keep the full soft answer.

    This is the only function in the module that talks to a backend, and it is the
    only place a sample's text leaves the process. Three properties are load
    bearing:

    * **Source-gated egress.** :func:`may_go_to_cloud` is consulted per sample
      against the live gate verdict. A refusal is recorded in the audit trail with
      its reason and never reaches the backend.
    * **Full distributions.** :attr:`LabeledSample.answers` is exactly what the
      backend returned -- four heads, every option's probability. Nothing here
      reduces it to an argmax.
    * **Auditable per call.** One :class:`LabelingAuditEntry` per sample, sent or
      refused, carrying ``source_category``, the generator, the fakeness basis, the
      licence, the backend and its model version, and the excerpt *hash*.

    ``backend=None`` builds a :class:`~jev_route.backends.mock.MockBackend`, so the
    whole path runs offline with no API key and no network. Pass a
    :class:`~jev_route.backends.jev.JevBackend` to label for real; this module
    never constructs one itself, because :mod:`jev_route.backends.jev` is the only
    module in jev-route allowed to name the cloud endpoint.

    The excerpt is sent **unredacted**, which is only sound because of the source
    table: redacting a synthetic identifier would erase the thing the sample exists
    to teach.
    """
    active_gate = gate or default_gate
    active_backend: DecisionBackend = backend if backend is not None else MockBackend()
    trail = audit if audit is not None else LabelingAudit()
    items = list(samples)
    if not items:
        return LabelingResult(labeled=(), refused=(), audit=trail)
    if max_concurrency < 1:
        raise SensitivityDataError(f"max_concurrency must be >= 1, got {max_concurrency}")

    caller_metadata = safe_caller_metadata(metadata)
    semaphore = asyncio.Semaphore(max_concurrency)
    # The index is captured before the gather so the audit trail stays in input
    # order even though the calls complete out of order.
    timestamps = [_timestamp_at(index) for index in range(len(items))]

    async def one(index: int, sample: SensitivitySample) -> tuple[LabeledSample | None, LabelingAuditEntry]:
        verdict = active_gate.scan(sample.text)
        sendable, why = may_go_to_cloud(sample, verdict)
        refusal = None if sendable else refusal_reason(sample, verdict)
        digest = hash_text(sample.text)
        base: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "source_category": sample.source,
            "label_role": sample.label_role,
            "generator": sample.generator,
            "fakeness_basis": sample.fakeness_basis,
            "licence": sample.licence,
            "excerpt_hash": digest,
            "gate_detectors": verdict.detectors(),
            "gate_blocks_backend": verdict.blocks_backend,
            "expected_sensitivity": sample.expected_sensitivity,
            "timestamp": timestamps[index],
        }
        if not sendable:
            return None, LabelingAuditEntry(
                **base,
                sent_to_backend=False,
                backend=active_backend.name,
                model_version="",
                latency_ms=0.0,
                degraded=False,
                refusal_reason=refusal,
                refusal_detail=why or "",
                sensitivity_probabilities={},
                sensitivity_label="",
                pii=0.0,
                teacher_agrees_with_expected=None,
            )

        request = DecisionRequest(
            redacted_excerpt=sample.text,
            features=compute_features(
                sample.text,
                gate_detectors=verdict.detectors(),
                n_gate_findings=len(verdict.findings),
                gate_force_local=verdict.force_local,
            ),
            advisory_topics=verdict.advisory_topics,
            metadata={**caller_metadata, **safe_caller_metadata(sample.metadata)},
            request_id=sample.sample_id,
        )
        async with semaphore:
            result = await active_backend.decide(request)
        sensitivity = result.answers.sensitivity
        agrees = sensitivity.choice == sample.expected_sensitivity if sample.expected_sensitivity else None
        entry = LabelingAuditEntry(
            **base,
            sent_to_backend=True,
            backend=active_backend.name,
            model_version=result.model_version,
            latency_ms=round(result.latency_ms, 3),
            degraded=result.degraded,
            refusal_reason=REFUSAL_DEGRADED_TEACHER if result.degraded else None,
            refusal_detail="" if not result.degraded else str(result.degrade_reason or "the teacher degraded"),
            sensitivity_probabilities=dict(sensitivity.probabilities),
            sensitivity_label=sensitivity.choice,
            pii=round(result.answers.pii.value, 6),
            teacher_agrees_with_expected=agrees,
        )
        if result.degraded:
            # A uniform distribution is not a label. Counting it as one would put
            # maximum-uncertainty rows into a dataset whose purpose is calibration.
            return None, entry
        labeled = LabeledSample(
            sample=sample,
            answers=result.answers,
            gate=verdict,
            backend=active_backend.name,
            model_version=result.model_version,
            latency_ms=result.latency_ms,
            degraded=result.degraded,
            degrade_reason=result.degrade_reason,
            questions_sent=dict(result.questions_sent or {}),
        )
        return labeled, entry

    outcomes = await asyncio.gather(*(one(index, sample) for index, sample in enumerate(items)))
    labeled: list[LabeledSample] = []
    refused: list[SensitivitySample] = []
    for (item, entry), sample in zip(outcomes, items, strict=True):
        trail.add(entry)
        if item is None:
            refused.append(sample)
        else:
            labeled.append(item)
        if audit_sink is not None:
            # An operator who already ships decision records can ship labeling
            # audit lines through the same plumbing. Failures are swallowed by the
            # sink contract, so a broken warehouse cannot fail a labeling run.
            _emit_to_sink(audit_sink, entry)
    return LabelingResult(labeled=tuple(labeled), refused=tuple(refused), audit=trail)


def label_sensitivity_sync(samples: Iterable[SensitivitySample], **kwargs: Any) -> LabelingResult:
    """Blocking wrapper, for scripts and for a CLI verb that has no event loop."""
    return asyncio.run(label_sensitivity(samples, **kwargs))


def _emit_to_sink(sink: DecisionSink, entry: LabelingAuditEntry) -> None:
    """Hand an audit entry to a :class:`~jev_route.logging_sink.DecisionSink`.

    The sink protocol takes a :class:`DecisionRecord`, so the entry is carried as
    one with ``kind`` set to the audit kind and the payload in ``metadata``. A
    consumer filtering on ``kind == RECORD_KIND`` ignores these lines, which is the
    behaviour :func:`jev_route.logging_sink.iter_records` already implements.
    """
    record = DecisionRecord(
        request_id=entry.sample_id,
        timestamp=entry.timestamp,
        decision=RoutingDecision(
            tier=UNROUTED_TIER,
            model="",
            rule_id="labeling.audit",
            reason=f"sensitivity labeling audit: {entry.source_category}",
            answers=DecisionAnswers.unknown(),
            gate=GateVerdict.clean(),
            backend=entry.backend,
            backend_model_version=entry.model_version,
            effective_sensitivity=entry.sensitivity_label or "public",
            effective_complexity="standard",
            latency_ms=entry.latency_ms,
            degraded=entry.degraded,
        ),
        features=RequestFeatures(),
        excerpt_hash=entry.excerpt_hash,
        backend_latency_ms=entry.latency_ms,
        total_latency_ms=entry.latency_ms,
        excerpt=None,
        kind=AUDIT_RECORD_KIND,
        metadata={
            "source_category": entry.source_category,
            "sent_to_backend": entry.sent_to_backend,
            "refusal_reason": entry.refusal_reason or "",
            "generator": entry.generator,
            "sensitivity_label": entry.sensitivity_label,
        },
    )
    sink.write(record)


def _timestamp_at(index: int) -> str:
    """A deterministic timestamp: the epoch plus one second per sample."""
    return (_EPOCH + timedelta(seconds=index)).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Dataset building
# --------------------------------------------------------------------------- #
def _synthetic_record(labeled: LabeledSample, timestamp: str) -> DecisionRecord:
    """A :class:`DecisionRecord` for a labeled synthetic sample.

    Built so that :func:`jev_route.distill.export.build_row` can do the row
    construction -- targets, splits, teacher block, provenance -- with exactly the
    conventions the production log produces, which is what lets ``train`` consume
    this dataset without a special case.

    Two deliberate choices. ``excerpt`` is always ``None`` here and the text is put
    onto the row afterwards from the sample, so this module never writes text into
    a record field it is forbidden to read. And ``tier`` is
    :data:`UNROUTED_TIER`: no router ever saw this request, so inventing a tier
    would corrupt :mod:`jev_route.distill.evaluate`'s tier-agreement numbers.
    """
    sample = labeled.sample
    answers = labeled.answers
    floor = labeled.gate.sensitivity_floor
    effective = answers.sensitivity.choice
    if floor is not None and level_index(floor, SENSITIVITY_LEVELS) > level_index(effective, SENSITIVITY_LEVELS):
        effective = floor
    pii = answers.pii.value
    if labeled.gate.pii_floor is not None and labeled.gate.pii_floor > pii:
        pii = labeled.gate.pii_floor
    return DecisionRecord(
        request_id=sample.sample_id,
        timestamp=timestamp,
        decision=RoutingDecision(
            tier=UNROUTED_TIER,
            model="",
            rule_id=SYNTHETIC_RULE_ID,
            reason=f"synthetic sensitivity sample ({sample.source}/{sample.generator or 'n-a'})",
            answers=answers,
            gate=labeled.gate,
            backend=labeled.backend,
            backend_model_version=labeled.model_version,
            effective_sensitivity=effective,
            effective_complexity=answers.complexity.choice,
            escalated=(),
            degraded=labeled.degraded,
            degrade_reason=labeled.degrade_reason,
            latency_ms=labeled.latency_ms,
        ),
        features=compute_features(
            sample.text,
            gate_detectors=labeled.gate.detectors(),
            n_gate_findings=len(labeled.gate.findings),
            gate_force_local=labeled.gate.force_local,
        ),
        excerpt_hash=hash_text(sample.text),
        backend_latency_ms=round(labeled.latency_ms, 3),
        total_latency_ms=round(labeled.latency_ms, 3),
        questions_sent={},
        excerpt=None,
        metadata=safe_caller_metadata(sample.metadata),
    )


def _row_from_labeled(
    labeled: LabeledSample,
    *,
    mode: str,
    holdout_fraction: float,
    exported_at: str,
    timestamp: str,
) -> TrainingRow:
    """One labeled sample as a :class:`TrainingRow`, in export's exact layout."""
    sample = labeled.sample
    record = _synthetic_record(labeled, timestamp)
    # features mode on purpose: build_row must not be the thing that decides
    # whether this row carries text. The text comes from our own sample below.
    row = build_row(record, mode="features", holdout_fraction=holdout_fraction, exported_at=exported_at)
    text = sample.text if mode == "text" else None
    provenance = {
        **row.provenance,
        "export_mode": mode,
        "excerpt_retained": text is not None,
        "source_module": "jev_route.distill.sensitivity_data",
        **sample.provenance(),
        "teacher_agrees_with_expected": (
            labeled.answers.sensitivity.choice == sample.expected_sensitivity if sample.expected_sensitivity else None
        ),
    }
    tags = (*row.tags, f"source:{sample.source}", "synthetic-dataset")
    return replace(row, text=text, provenance=provenance, tags=tags)


def _row_from_record(
    record: DecisionRecord,
    *,
    mode: str,
    holdout_fraction: float,
    exported_at: str,
) -> TrainingRow:
    """One production record as a row.

    ``mode`` here is decided by the caller from whether the record was blocked:
    a blocked record arrives already reduced by :func:`_metadata_only_projection`
    and is always built in ``features`` mode, so no text can enter the row even if
    a future ``build_row`` started reading more of the record than it does today.
    """
    row = build_row(record, mode=mode, holdout_fraction=holdout_fraction, exported_at=exported_at)
    blocked = record.decision.gate.blocks_backend or record.decision.backend == "gate"
    provenance = {
        **row.provenance,
        "source_module": "jev_route.distill.sensitivity_data",
        "source_category": "production-negative",
        "label_role": "negative/borderline",
        "generator": "",
        "fakeness_basis": "",
        "reservation": "real traffic the local hard gate already cleared",
        "citation": "",
        "expected_sensitivity": "",
        "licence": "",
        "licence_url": "",
        "corpus": "",
        "gate_blocked_metadata_only": blocked,
    }
    tags = (*row.tags, "source:production-negative", "synthetic-dataset")
    if blocked:
        tags += ("blocked-metadata-only",)
    return replace(row, provenance=provenance, tags=tags)


@dataclass(frozen=True)
class SynthesisStats:
    """Everything an operator needs to judge the dataset without re-reading it."""

    source: str
    mode: str
    rows_written: int = 0
    train_rows: int = 0
    holdout_rows: int = 0
    holdout_fraction: float = 0.2
    #: Rows per source category. The headline number for "where did this come from".
    source_breakdown: dict[str, int] = field(default_factory=dict)
    role_breakdown: dict[str, int] = field(default_factory=dict)
    generator_counts: dict[str, int] = field(default_factory=dict)
    fakeness_basis_counts: dict[str, int] = field(default_factory=dict)
    #: Measured on the row text with the real gate, not copied from a generator's
    #: claim: how many rows fire a blocking detector, a non-blocking one, or neither.
    gate_expectation_counts: dict[str, int] = field(default_factory=dict)
    detector_counts: dict[str, int] = field(default_factory=dict)
    contextual_categories: dict[str, int] = field(default_factory=dict)
    labeling_backend: str = ""
    labeling_model_versions: dict[str, int] = field(default_factory=dict)
    labeling_calls: int = 0
    labeling_refusals: int = 0
    refusal_reasons: dict[str, int] = field(default_factory=dict)
    #: Teacher argmax vs the generator's intent. A low agreement rate on
    #: contextual positives is the interesting number in the whole file.
    teacher_expected_agreement: dict[str, dict[str, int]] = field(default_factory=dict)
    corpora: dict[str, dict[str, Any]] = field(default_factory=dict)
    production_records_read: int = 0
    production_blocked_discarded: int = 0
    production_blocked_metadata_only: int = 0
    production_degraded_skipped: int = 0
    usable_rows_per_head: dict[str, int] = field(default_factory=dict)
    label_support: dict[str, dict[str, int]] = field(default_factory=dict)
    allowed_sources: tuple[dict[str, str], ...] = ()
    contains_prompt_text: bool = False
    dataset_sha256: str = ""
    row_schema: str = ROW_SCHEMA
    exported_at: str = ""
    warnings: tuple[str, ...] = ()

    @property
    def skipped_total(self) -> int:
        return self.labeling_refusals + self.production_degraded_skipped

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            out[key] = list(value) if isinstance(value, tuple) else value
        return out

    def summary(self) -> str:
        lines = [
            f"built {self.rows_written} sensitivity rows ({self.mode} mode): "
            f"{self.train_rows} train / {self.holdout_rows} holdout",
        ]
        for category, count in sorted(self.source_breakdown.items()):
            role = source_policy(category).label_role if category in _SOURCE_POLICIES else "?"
            lines.append(f"  {category:<22} {count:>5}  ({role})")
        coverage = " ".join(f"{k}={v}" for k, v in sorted(self.gate_expectation_counts.items()))
        lines.append(f"  gate coverage: {coverage}")
        if self.fakeness_basis_counts:
            lines.append(
                "  fakeness basis: " + " ".join(f"{k}={v}" for k, v in sorted(self.fakeness_basis_counts.items()))
            )
        lines.append(f"  teacher: {self.labeling_backend or '(none)'} {dict(self.labeling_model_versions)}")
        lines.append(
            f"  labeling calls={self.labeling_calls} refusals={self.labeling_refusals} {dict(self.refusal_reasons)}"
        )
        if self.production_records_read:
            lines.append(
                f"  production log: {self.production_records_read} read, "
                f"{self.production_blocked_discarded} blocked-and-discarded (content never read), "
                f"{self.production_blocked_metadata_only} metadata-only, "
                f"{self.production_degraded_skipped} degraded-skipped"
            )
        for head in (*CHOICE_HEADS, "pii"):
            support = self.label_support.get(head) or {}
            lines.append(
                f"  {head:<12} usable={self.usable_rows_per_head.get(head, 0):<6} "
                + " ".join(f"{k}={v}" for k, v in sorted(support.items()))
            )
        for category, tally in sorted(self.teacher_expected_agreement.items()):
            total = sum(tally.values())
            lines.append(
                f"  teacher vs generator intent [{category}]: "
                + " ".join(f"{k}={v}" for k, v in sorted(tally.items()))
                + f" (n={total})"
            )
        lines.extend(f"  warning: {w}" for w in self.warnings)
        return "\n".join(lines)


@dataclass(frozen=True)
class SynthesisResult:
    """A built dataset, its stats, the audit trail, and where both were written."""

    dataset: Dataset
    stats: SynthesisStats
    audit: LabelingAudit
    rows_path: Path | None = None
    stats_path: Path | None = None


def _dataset_paths(out_path: str | Path) -> tuple[Path, Path]:
    """Rows file and stats sidecar, named the way :func:`export.load_dataset` expects.

    Mirrors ``export._dataset_paths`` rather than calling it: that helper is
    private to another module, and the naming convention (``<dir>/dataset.jsonl``
    plus ``<stem>.stats.json``) is the actual contract ``load_dataset`` reads.
    """
    target = Path(out_path)
    if target.is_dir() or not target.suffix:
        target = target / "dataset.jsonl"
    if target.suffix != ".jsonl":
        target = target.with_suffix(target.suffix + ".jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target, target.with_name(target.stem + ".stats.json")


def _tally_row(stats: dict[str, Any], row: TrainingRow, sample: SensitivitySample | None) -> None:
    """Fold one written row into the running counters behind :class:`SynthesisStats`."""
    stats["rows_written"] += 1
    if row.split == "holdout":
        stats["holdout_rows"] += 1
    else:
        stats["train_rows"] += 1
    category = str(row.provenance.get("source_category", ""))
    stats["source_breakdown"][category] = stats["source_breakdown"].get(category, 0) + 1
    role = source_policy(category).label_role if category in _SOURCE_POLICIES else "unknown"
    stats["role_breakdown"][role] = stats["role_breakdown"].get(role, 0) + 1
    generator = str(row.provenance.get("generator") or "")
    if generator:
        stats["generator_counts"][generator] = stats["generator_counts"].get(generator, 0) + 1
    basis = str(row.provenance.get("fakeness_basis") or "")
    if basis:
        stats["fakeness_basis_counts"][basis] = stats["fakeness_basis_counts"].get(basis, 0) + 1
    for detector, count in (row.features.gate_detectors or {}).items():
        stats["detector_counts"][detector] = stats["detector_counts"].get(detector, 0) + int(count)
    expectation = "blocks" if row.gate.get("blocks_backend") else ("fires" if row.gate.get("fired") else "none")
    stats["gate_expectation_counts"][expectation] = stats["gate_expectation_counts"].get(expectation, 0) + 1
    if sample is not None and sample.source == "contextual-template":
        name = str(sample.detail.get("category", ""))
        stats["contextual_categories"][name] = stats["contextual_categories"].get(name, 0) + 1
    agreement = row.provenance.get("teacher_agrees_with_expected")
    if agreement is not None:
        bucket = stats["teacher_expected_agreement"].setdefault(category, {})
        key = "agree" if agreement else "disagree"
        bucket[key] = bucket.get(key, 0) + 1
    for head in (*CHOICE_HEADS, "pii"):
        if row.targets.usable(head):
            stats["usable_rows_per_head"][head] = stats["usable_rows_per_head"].get(head, 0) + 1
            support = stats["label_support"].setdefault(head, {})
            if head == "pii":
                label = "true" if row.targets.pii["label"] else "false"
            else:
                label = str(row.targets.get(head)["label"])
            support[label] = support.get(label, 0) + 1


def _stats_warnings(stats: Mapping[str, Any], mode: str, backend_name: str) -> list[str]:
    warnings: list[str] = []
    if mode == "text":
        warnings.append(
            "this dataset contains prompt text. Every row is either provably synthetic, from a corpus "
            "with a stated licence, or real traffic the local gate already cleared -- but treat the file "
            "as a dataset, not as a log, and delete it after training."
        )
    if backend_name == "mock":
        warnings.append(
            "the teacher was MockBackend, which keys on roughly the same vocabulary the deterministic gate "
            "does. Its labels for contextual positives are therefore weak by construction: those samples "
            "were written to be invisible to keyword matching. Relabel with JevBackend before training on "
            "the contextual rows."
        )
    if stats.get("gate_expectation_counts", {}).get("blocks", 0) == 0:
        warnings.append(
            "no row fires a blocking detector, so the dataset does not cover the identifiers the gate "
            "refuses to send anywhere. Generate synthetic-pii samples (the default) to cover them."
        )
    if not stats.get("contextual_categories"):
        warnings.append(
            "no contextual positives in this dataset: every positive is one a regex can already see. "
            "Layer 2 will have nothing to learn from."
        )
    agreement = stats.get("teacher_expected_agreement", {})
    for category, tally in sorted(agreement.items()):
        total = sum(tally.values())
        if total and tally.get("agree", 0) / total < 0.5:
            warnings.append(
                f"the teacher disagreed with the generator's intent on {tally.get('disagree', 0)}/{total} "
                f"{category} rows. The teacher's distribution is still the label; read the disagreement as "
                "a signal about the templates or the teacher, not as a reason to override either."
            )
    warnings.append(
        f"synthetic rows carry teacher.tier={UNROUTED_TIER!r} because no router ever saw them. Run "
        "`distill evaluate` without --policy, or filter the 'synthetic-dataset' tag, or the tier "
        "agreement numbers will be meaningless."
    )
    return warnings


def build_sensitivity_dataset(
    *,
    labeled: Iterable[LabeledSample] = (),
    records: Iterable[DecisionRecord] = (),
    mode: str = "text",
    holdout_fraction: float = 0.2,
    exported_at: str | None = None,
    out_path: str | Path | None = None,
    audit: LabelingAudit | None = None,
    intake: ProductionIntake | None = None,
    corpora: Iterable[PublicCorpus] = (),
    sample_of: Mapping[str, SensitivitySample] | None = None,
) -> SynthesisResult:
    """Assemble rows into the shape :mod:`jev_route.distill.train` already consumes.

    Rows are produced by :func:`jev_route.distill.export.build_row`, so the layout,
    the soft targets, the split assignment and the provenance keys are identical to
    a dataset exported from a production log. ``train`` needs no special case.

    ``mode="features"`` drops every row's text and keeps only the deterministic
    feature vector: the zero-text-retention build. ``include_blocked_metadata``
    rows are only representable there, which is why
    :func:`synthesize_sensitivity_dataset` rejects the combination.
    """
    if mode not in ("text", "features"):
        raise SensitivityDataError(f"unknown mode {mode!r}; expected 'text' or 'features'")
    if not 0.0 <= float(holdout_fraction) < 0.9:
        raise SensitivityDataError(f"holdout_fraction must be in [0, 0.9), got {holdout_fraction}")

    labeled_items = list(labeled)
    production = list(records)
    if not labeled_items and not production:
        raise SensitivityDataError(
            "nothing to build a dataset from: pass labeled= samples, records= from a production log, or both"
        )

    stamp = exported_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    trail = audit if audit is not None else LabelingAudit()
    counters: dict[str, Any] = {
        "rows_written": 0,
        "train_rows": 0,
        "holdout_rows": 0,
        "source_breakdown": {},
        "role_breakdown": {},
        "generator_counts": {},
        "fakeness_basis_counts": {},
        "gate_expectation_counts": {},
        "detector_counts": {},
        "contextual_categories": {},
        "teacher_expected_agreement": {},
        "usable_rows_per_head": {},
        "label_support": {},
    }
    rows: list[TrainingRow] = []
    digest = hashlib.sha256()
    by_id = dict(sample_of or {})

    for index, item in enumerate(labeled_items):
        row = _row_from_labeled(
            item, mode=mode, holdout_fraction=holdout_fraction, exported_at=stamp, timestamp=_timestamp_at(index)
        )
        rows.append(row)
        _tally_row(counters, row, item.sample)
        by_id.setdefault(item.sample.sample_id, item.sample)

    metadata_only = 0
    for record in production:
        blocked = record.decision.gate.blocks_backend or record.decision.backend == "gate"
        row = _row_from_record(
            record,
            mode="features" if blocked else mode,
            holdout_fraction=holdout_fraction,
            exported_at=stamp,
        )
        if blocked:
            metadata_only += 1
        rows.append(row)
        _tally_row(counters, row, None)

    for row in rows:
        digest.update((row.to_json() + "\n").encode("utf-8"))

    backends = {item.backend for item in labeled_items}
    versions: dict[str, int] = {}
    for item in labeled_items:
        versions[item.model_version or "(unknown)"] = versions.get(item.model_version or "(unknown)", 0) + 1
    refusals = trail.refusal_reasons()
    intake_stats = intake or ProductionIntake()

    stats = SynthesisStats(
        source=str(out_path) if out_path else "(in-memory)",
        mode=mode,
        rows_written=counters["rows_written"],
        train_rows=counters["train_rows"],
        holdout_rows=counters["holdout_rows"],
        holdout_fraction=float(holdout_fraction),
        source_breakdown=counters["source_breakdown"],
        role_breakdown=counters["role_breakdown"],
        generator_counts=counters["generator_counts"],
        fakeness_basis_counts=counters["fakeness_basis_counts"],
        gate_expectation_counts=counters["gate_expectation_counts"],
        detector_counts=counters["detector_counts"],
        contextual_categories=counters["contextual_categories"],
        labeling_backend=",".join(sorted(backends)),
        labeling_model_versions=versions,
        labeling_calls=len(trail.sent()),
        labeling_refusals=len(trail.refused()),
        refusal_reasons=refusals,
        teacher_expected_agreement=counters["teacher_expected_agreement"],
        corpora={c.name: c.as_dict() for c in corpora},
        production_records_read=intake_stats.read,
        production_blocked_discarded=intake_stats.blocked_discarded,
        production_blocked_metadata_only=intake_stats.blocked_metadata_only or metadata_only,
        production_degraded_skipped=intake_stats.degraded_skipped,
        usable_rows_per_head=counters["usable_rows_per_head"],
        label_support=counters["label_support"],
        allowed_sources=tuple(
            {
                "category": s.category,
                "label_role": s.label_role,
                "may_go_to_cloud": s.may_go_to_cloud,
                "rationale": s.rationale,
            }
            for s in ALLOWED_SOURCES
        ),
        contains_prompt_text=mode == "text",
        dataset_sha256=digest.hexdigest(),
        exported_at=stamp,
        warnings=tuple(_stats_warnings(counters, mode, ",".join(sorted(backends)))),
    )
    dataset = Dataset(
        rows=tuple(rows),
        mode=mode,
        stats=stats.as_dict(),
        source=stats.source,
    )
    rows_path: Path | None = None
    stats_path: Path | None = None
    if out_path is not None:
        rows_path, stats_path = write_sensitivity_dataset(dataset, stats, out_path)
    return SynthesisResult(dataset=dataset, stats=stats, audit=trail, rows_path=rows_path, stats_path=stats_path)


def write_sensitivity_dataset(dataset: Dataset, stats: SynthesisStats, out_path: str | Path) -> tuple[Path, Path]:
    """Write rows and the stats sidecar so :func:`export.load_dataset` reads them back."""
    rows_path, stats_path = _dataset_paths(out_path)
    written = dataset.save(rows_path)
    stats_path.write_text(
        json.dumps({"synthesis": stats.as_dict(), "stats": stats.as_dict()}, sort_keys=True, indent=2, default=str),
        encoding="utf-8",
    )
    return written, stats_path


def mask_secrets(text: str) -> str:
    """Replace every gate-matched span with ``[...]`` for printing examples.

    Not a privacy control -- the text is already provably synthetic -- but the
    README and the CLI both show example rows, and a page of test PANs and
    ``sk-`` strings trips every secret scanner in CI and every reviewer's reflex.
    """
    from ..prompts import redact

    masked, _count = redact(text, default_gate, placeholder="[masked-{name}]")
    return masked


async def synthesize_sensitivity_dataset(
    *,
    out_path: str | Path | None = None,
    backend: DecisionBackend | None = None,
    gate: HardGate | None = None,
    seed: int = 20260919,
    per_kind: int = 6,
    kinds: Iterable[str] | None = None,
    contextual_per_category: int = 2,
    contextual_categories: Iterable[str] | None = None,
    card_fakeness: str = CardFakeness.RESERVED_TEST_RANGE,
    iban_fakeness: str = IbanFakeness.RESERVED_COUNTRY,
    email_domain_pool: str = "rfc2606",
    production_log: str | Path | Iterable[DecisionRecord] | None = None,
    production_limit: int | None = None,
    include_blocked_metadata: bool = False,
    corpora: Iterable[PublicCorpus] = (),
    corpus_row_limit: int | None = None,
    mode: str = "text",
    holdout_fraction: float = 0.2,
    audit_path: str | Path | None = None,
    audit_sink: DecisionSink | None = None,
    exported_at: str | None = None,
    max_concurrency: int = 4,
    metadata: Mapping[str, Any] | None = None,
) -> SynthesisResult:
    """Generate, label and write a sensitivity dataset in one call.

    The whole pipeline: synthetic identifiers and contextual prose from
    :mod:`jev_route.distill.synthetic_pii`, optional licensed corpus rows, optional
    production negatives from a decision log, all labeled through ``backend`` (a
    :class:`~jev_route.backends.mock.MockBackend` by default, so this runs offline
    with no API key), then assembled into the row layout
    :mod:`jev_route.distill.train` already reads.

    Reproducible: the same ``seed`` and inputs produce the same rows and the same
    ``dataset_sha256``, because sample ids, timestamps and the generator draws are
    all derived from the seed rather than from a clock or ``uuid4``.
    """
    if mode not in ("text", "features"):
        raise SensitivityDataError(f"unknown mode {mode!r}; expected 'text' or 'features'")
    if include_blocked_metadata and mode == "text":
        raise SensitivityDataError(
            "include_blocked_metadata=True needs mode='features'. A gate-blocked record contributes "
            "metadata only, so its row has no text; putting that row in a text dataset would leave the "
            "trainer an empty string and make 'this dataset contains text' mean something untrue."
        )

    rng = random.Random(seed)
    active_gate = gate or default_gate

    samples: list[SensitivitySample] = []
    samples.extend(
        synthetic_pii_samples(
            rng,
            per_kind=per_kind,
            kinds=kinds,
            card_fakeness=card_fakeness,
            iban_fakeness=iban_fakeness,
            email_domain_pool=email_domain_pool,
            gate=active_gate,
            id_prefix=f"spii{seed:06d}",
        )
    )
    if contextual_per_category > 0:
        chosen = contextual_samples(rng, categories=contextual_categories, gate=active_gate)
        repeated = list(chosen) * max(1, contextual_per_category)
        rng.shuffle(repeated)
        for index, template in enumerate(repeated, start=1):
            samples.append(contextual_sample_from_template(template, index, f"ctx{seed:06d}"))
    corpus_list = list(corpora)
    for corpus in corpus_list:
        samples.extend(corpus.samples(limit=corpus_row_limit))

    intake = ProductionIntake()
    if production_log is not None:
        intake = production_negative_samples(
            production_log, limit=production_limit, include_blocked_metadata=include_blocked_metadata
        )

    result = await label_sensitivity(
        samples,
        backend=backend,
        gate=active_gate,
        max_concurrency=max_concurrency,
        metadata=metadata,
        audit_sink=audit_sink,
    )
    if audit_path is not None:
        result.audit.write(audit_path)

    built = build_sensitivity_dataset(
        labeled=result.labeled,
        records=intake.records,
        mode=mode,
        holdout_fraction=holdout_fraction,
        exported_at=exported_at,
        out_path=out_path,
        audit=result.audit,
        intake=intake,
        corpora=corpus_list,
        sample_of={s.sample.sample_id: s.sample for s in result.labeled},
    )
    stats = built.stats
    notes = list(intake.notes)
    if result.refused:
        notes.append(f"{len(result.refused)} sample(s) were refused before any backend call")
    if notes:
        # Rebuild is not needed: warnings are informational and SynthesisStats is
        # frozen, so a note that arrived after the tally is reported alongside it.
        object.__setattr__(stats, "warnings", (*stats.warnings, *(f"note: {n}" for n in notes)))
    return built


def example_rows(result: SynthesisResult, n: int = 3, *, mask: bool = True) -> str:
    """A printable view of ``n`` rows, with gate-matched spans masked.

    For the CLI and for the README. Masking is on by default because example rows
    are the ones most likely to be pasted into an issue, and a paste of raw test
    PANs and ``sk-`` strings is a paste that gets flagged by every scanner between
    here and there.
    """
    lines: list[str] = []
    for row in result.dataset.rows[:n]:
        text = row.text or "(no text: features mode)"
        if mask:
            text = mask_secrets(text)
        probs = {k: round(float(v), 4) for k, v in row.targets.sensitivity["probs"].items()}
        blocks = bool(row.gate.get("blocks_backend"))
        fired = bool(row.gate.get("fired"))
        lines.append(
            f"- [{row.split}] {row.request_id}\n"
            f"    source     : {row.provenance.get('source_category')}/{row.provenance.get('generator') or '-'}\n"
            f"    fakeness   : {row.provenance.get('fakeness_basis') or '-'} "
            f"({row.provenance.get('reservation') or '-'})\n"
            f"    gate       : fired={fired} blocks_backend={blocks} "
            f"detectors={dict(row.features.gate_detectors)}\n"
            f"    sensitivity: {row.targets.sensitivity['label']} probs={probs} "
            f"pii={round(float(row.targets.pii['value']), 4)}\n"
            f"    text       : {text}"
        )
    return "\n".join(lines)


__all__ = [
    "ALLOWED_SOURCES",
    "AUDIT_RECORD_KIND",
    "AUDIT_SCHEMA_VERSION",
    "RECORD_SENTENCES",
    "SOURCE_CATEGORIES",
    "SYNTHETIC_RULE_ID",
    "UNROUTED_TIER",
    "CloudSendRefusedError",
    "CorpusLicenceError",
    "CorpusRow",
    "LabeledSample",
    "LabelingAudit",
    "LabelingAuditEntry",
    "LabelingResult",
    "ProductionIntake",
    "PublicCorpus",
    "RecordSentence",
    "SensitivityDataError",
    "SensitivitySample",
    "SourceCategory",
    "SourcePolicy",
    "SynthesisResult",
    "SynthesisStats",
    "UnknownSourceError",
    "assert_sendable",
    "build_sensitivity_dataset",
    "contextual_positive_samples",
    "contextual_sample_from_template",
    "example_rows",
    "label_sensitivity",
    "label_sensitivity_sync",
    "load_public_corpus",
    "mask_secrets",
    "may_go_to_cloud",
    "production_negative_samples",
    "refusal_reason",
    "source_policy",
    "synthesize_sensitivity_dataset",
    "synthetic_pii_samples",
    "write_sensitivity_dataset",
]
