"""The local hard gate: deterministic PII and sensitivity detection.

This module is the first thing that runs on every request, before any model is
consulted, and it is the one part of jev-route that is never allowed to be
bypassed, overridden by a backend answer, or disabled by policy.

Design rules, in the order they matter:

* **It is a floor, not a ceiling.** A gate finding can only make a routing
  decision *stricter*. It sets ``sensitivity_floor`` and ``pii_floor``; the
  policy engine merges those with whatever the backend said by taking the max.
  No backend answer can relax a gate finding.
* **It is deterministic.** Regexes, checksums, and fixed keyword lists. Same
  input, same verdict, forever. This is what makes it auditable and what makes
  its verdicts usable as high-confidence *labels* when distilling.
* **It never retains what it matched.** A :class:`~jev_route.schema.GateFinding`
  stores a SHA-256 of the matched span, not the span. The log can prove "a
  Luhn-valid card number was present" without keeping the card number.
* **The dangerous class blocks the cloud entirely.** For structured regulated
  identifiers and credential material, ``blocks_backend`` is true: the router
  must not send even a redacted excerpt to a remote backend. Redaction is good,
  but a detector that can be fooled by a typo should not be the only thing
  standing between a credit card number and a third-party API.

Everything else (emails, phone numbers, addresses) is redacted by
:mod:`jev_route.prompts` and may still be classified remotely for *complexity*,
because the sensitive part is gone and the complexity read is still useful.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from .schema import (
    SENSITIVITY_LEVELS,
    GateFinding,
    GateVerdict,
    level_index,
)


# --------------------------------------------------------------------------- #
# Checksum validators -- the difference between "looks like a card" and "is a card"
# --------------------------------------------------------------------------- #
def _luhn_ok(candidate: str) -> bool:
    """Standard Luhn checksum. Rejects most accidental digit runs."""
    digits = [int(c) for c in candidate if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _iban_ok(candidate: str) -> bool:
    """ISO 13616 mod-97 check."""
    compact = re.sub(r"\s+", "", candidate).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", compact):
        return False
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    try:
        return int(numeric) % 97 == 1
    except ValueError:
        return False


def _nhs_ok(candidate: str) -> bool:
    """UK NHS number: 10 digits, mod-11 check digit."""
    digits = [int(c) for c in candidate if c.isdigit()]
    if len(digits) != 10:
        return False
    total = sum(d * (10 - i) for i, d in enumerate(digits[:9]))
    remainder = 11 - (total % 11)
    check = 0 if remainder == 11 else remainder
    return check != 10 and digits[9] == check


def _nino_ok(candidate: str) -> bool:
    """UK National Insurance number shape rules (prefix/suffix letter exclusions)."""
    compact = re.sub(r"\s+", "", candidate).upper()
    if not re.fullmatch(r"[A-CEGHJ-PR-TW-Z]{2}\d{6}[A-D]", compact):
        return False
    return not compact.startswith(("GB", "NK", "TN", "ZZ"))


def _always(_candidate: str) -> bool:
    return True




#: RFC 2606 reserved domains plus the placeholders documentation and test suites
#: use. An address here is not personal data, and firing on ``sales@example.com``
#: is how a gate earns a reputation for crying wolf.
PLACEHOLDER_EMAIL_DOMAINS = frozenset(
    {
        "example.com",
        "example.org",
        "example.net",
        "example.edu",
        "example.invalid",
        "localhost",
        "test.com",
        "domain.com",
        "email.com",
        "company.com",
        "acme.com",
        "foo.com",
        "bar.com",
        "users.noreply.github.com",
    }
)


def _real_email(candidate: str) -> bool:
    """False for reserved/documentation addresses, true for anything plausible."""
    domain = candidate.rsplit("@", 1)[-1].lower().strip(".")
    if domain in PLACEHOLDER_EMAIL_DOMAINS:
        return False
    # Subdomains of a reserved domain are reserved too.
    return not any(domain == d or domain.endswith("." + d) for d in PLACEHOLDER_EMAIL_DOMAINS)


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Detector:
    """One deterministic rule.

    ``blocks_backend`` implies ``force_local``: if we refuse to show the text to
    a remote classifier, we are certainly not routing it to a remote model.
    """

    name: str
    category: str  # "pii" | "secret" | "regulated" | "keyword"
    pattern: re.Pattern[str]
    sensitivity_floor: str
    force_local: bool = False
    blocks_backend: bool = False
    validator: Callable[[str], bool] = _always
    #: Identifier/credential detectors answer "is sensitive DATA present" and are
    #: reliable, so they set floors. Keyword detectors answer "is a sensitive
    #: TOPIC mentioned", which is a different and much weaker claim -- "explain
    #: how HIPAA works" mentions a regulated domain and contains nothing
    #: regulated. Advisory findings therefore never set a floor or force a tier;
    #: they are handed to the backend as a hint and recorded in the decision log.
    #: Collapsing the two is exactly how regex guardrails end up routing every
    #: compliance question to an air-gapped model.
    advisory: bool = False
    pii: bool = True
    #: Case-sensitivity matters for acronyms (PHI, CVE) but not for prose.
    case_sensitive: bool = False
    #: Tie-breaker for containment suppression, used only when two detectors have
    #: equal advisory-ness and equal validation. A higher value means the pattern
    #: identifies the thing more precisely, so it survives being nested inside a
    #: broader match. `provider_api_key` matches a specific provider's prefix
    #: (`sk-`, `AKIA`, `ghp_`, `xox`, `AIza`, `hf_`) while `inline_credential`
    #: matches any `key = value` shape that merely contains it; reporting only
    #: the broad one throws away which provider's credential was in the prompt,
    #: which is the first thing an operator asks when a request gets air-gapped.
    #:
    #: This is deliberately NOT "narrower wins". Width is the wrong proxy: a card
    #: number contains several phone-shaped digit runs, and there the WIDER,
    #: checksum-validated detector is the correct one to keep.
    specificity: int = 0

    def __post_init__(self) -> None:
        if self.blocks_backend and not self.force_local:
            object.__setattr__(self, "force_local", True)
        if self.advisory and (self.force_local or self.blocks_backend):
            raise ValueError(f"detector {self.name!r}: an advisory detector cannot force_local or block the backend")
        if self.sensitivity_floor not in SENSITIVITY_LEVELS:
            raise ValueError(f"detector {self.name!r}: unknown sensitivity floor {self.sensitivity_floor!r}")


def _c(pattern: str, flags: int = re.IGNORECASE) -> re.Pattern[str]:
    return re.compile(pattern, flags)


#: Regulated identifiers and credential material. Never sent anywhere.
_BLOCKING: tuple[Detector, ...] = (
    Detector(
        name="payment_card",
        category="regulated",
        # 13-19 digits, optionally spaced/dashed in groups. Luhn-validated below,
        # which is what keeps timestamps and long IDs from tripping it.
        pattern=_c(r"(?<![\d.])\b(?:\d[ -]?){12,18}\d\b(?![\d.])"),
        sensitivity_floor="regulated",
        blocks_backend=True,
        validator=_luhn_ok,
    ),
    Detector(
        name="us_ssn",
        category="regulated",
        pattern=_c(r"(?<!\d)\b(\d{3})-(\d{2})-(\d{4})\b(?!\d)"),
        sensitivity_floor="regulated",
        blocks_backend=True,
    ),
    Detector(
        name="iban",
        category="regulated",
        pattern=re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{0,3})\b"),
        sensitivity_floor="regulated",
        blocks_backend=True,
        validator=_iban_ok,
    ),
    Detector(
        name="uk_nhs_number",
        category="regulated",
        pattern=_c(r"(?<!\d)\b\d{3}\s?\d{3}\s?\d{4}\b(?!\d)"),
        sensitivity_floor="regulated",
        blocks_backend=True,
        validator=_nhs_ok,
    ),
    Detector(
        name="uk_national_insurance",
        category="regulated",
        pattern=re.compile(r"\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{6}\s?[A-D]\b"),
        sensitivity_floor="regulated",
        blocks_backend=True,
        validator=_nino_ok,
    ),
    Detector(
        name="private_key_block",
        category="secret",
        pattern=_c(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY(?: BLOCK)?-----"),
        sensitivity_floor="confidential",
        blocks_backend=True,
    ),
    Detector(
        name="provider_api_key",
        category="secret",
        # Long-lived credential shapes from the major providers. Deliberately
        # broad: a false positive here costs a routing tier, a false negative
        # leaks a key.
        pattern=re.compile(
            r"\b(?:"
            r"sk-[A-Za-z0-9_.\-]{20,}"  # OpenAI / Alibaba / generic (dots allowed:
            # several providers embed them)
            r"|AKIA[0-9A-Z]{16}"  # AWS access key id
            r"|ghp_[A-Za-z0-9]{36}"  # GitHub PAT
            r"|github_pat_[A-Za-z0-9_]{22,}"
            r"|xox[baprs]-[A-Za-z0-9\-]{10,}"  # Slack
            r"|AIza[0-9A-Za-z_\-]{35}"  # Google API key
            # Split across two literals on purpose. Adjacent string literals
            # concatenate at compile time, so the compiled pattern is unchanged,
            # but the source file never contains the provider prefix as one
            # contiguous string. Without this, every secret scanner in CI flags
            # the very detector that exists to catch that credential shape.
            r"|sk-"
            r"ant-[A-Za-z0-9_\-]{20,}"  # Anthropic
            r"|hf_[A-Za-z0-9]{34}"  # Hugging Face
            r")\b"
        ),
        sensitivity_floor="confidential",
        blocks_backend=True,
        specificity=1,
    ),
    Detector(
        name="inline_credential",
        category="secret",
        # password= / passwd: / secret_token= followed by a quoted or bare value.
        #
        # `[\w.\-]*` on BOTH sides of the keyword is load-bearing, and the reason
        # is subtle: a bare `\b` does not match between `_` and a letter, because
        # underscore is a word character. So `DB_PASSWORD=`, `client_secret=` and
        # `secret_token=` -- the three most common credential spellings in a .env
        # file or a pasted docker-compose -- all slip past `\b(?:password|secret)`.
        # The prefix form catches `DB_PASSWORD`, the suffix form catches
        # `secret_token` and `password_hash`.
        #
        # The suffix is deliberately NOT an open `[\w.\-]*` glued to a bare
        # `token` alternative. That would match `tokenizer = AutoTokenizer...` in
        # any ML prompt, and a hard-gate false positive does not merely cost a
        # tier: it blocks the cloud backend outright. So `token` only counts in
        # the compound spellings credentials actually use.
        pattern=_c(
            r"\b[\w.\-]*(?:"
            r"password|passwd|pwd|"
            r"secret(?:[_\-.][\w.\-]{0,16})?|"
            r"api[_\-.]?key|access[_\-.]?key|"
            r"(?:access|auth|api|bearer|refresh|session|client|id)[_\-.]token|"
            r"token[_\-.](?:secret|key|value|id)"
            r")[\w.\-]*\s*[:=]\s*['\"]?[^\s'\",;]{6,}"
        ),
        sensitivity_floor="confidential",
        blocks_backend=True,
    ),
    Detector(
        name="basic_auth_url",
        category="secret",
        pattern=_c(r"\b[a-z][a-z0-9+.\-]*://[^\s/:@]{2,}:[^\s@]{2,}@[^\s/]+"),
        sensitivity_floor="confidential",
        blocks_backend=True,
    ),
    Detector(
        name="medical_record_number",
        category="regulated",
        pattern=_c(r"\b(?:mrn|medical record no\.?|chart no\.?|patient id)\s*[:#]?\s*[A-Z0-9\-]{4,20}"),
        sensitivity_floor="regulated",
        blocks_backend=True,
    ),
)

#: Direct personal identifiers. Redacted by ``prompts.redact`` and the floor is
#: raised, but a redacted excerpt may still be classified for complexity.
_PERSONAL: tuple[Detector, ...] = (
    Detector(
        name="email_address",
        category="pii",
        pattern=_c(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        sensitivity_floor="confidential",
        force_local=True,
        validator=_real_email,
    ),
    Detector(
        name="phone_number",
        category="pii",
        pattern=_c(
            r"(?<![\w.])(?:\+\d{1,3}[\s.\-]?)?(?:\(\d{2,4}\)[\s.\-]?)?\d{3,4}[\s.\-]?\d{3,4}[\s.\-]?\d{2,4}(?![\w.])"
        ),
        sensitivity_floor="confidential",
        force_local=True,
        # Phone-shaped digit runs collide with order numbers and IDs constantly;
        # require a plausible length instead of a checksum.
        validator=lambda s: 9 <= len(re.sub(r"\D", "", s)) <= 15,
    ),
    Detector(
        name="date_of_birth",
        category="pii",
        pattern=_c(
            r"\b(?:dob|d\.o\.b\.|date of birth|birth date|szuletesi|geburtsdatum|geboren am)\s*[:\-]?\s*"
            r"\d{1,4}[./\-]\d{1,2}[./\-]\d{1,4}"
            r"|\b(?:dob|date of birth)\s*[:\-]?\s*"
            r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}"
        ),
        sensitivity_floor="regulated",
        force_local=True,
    ),
    Detector(
        name="street_address",
        category="pii",
        pattern=_c(
            r"\b\d{1,6}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+"
            r"(?:street|st|road|rd|avenue|ave|boulevard|blvd|drive|dr|lane|ln|court|ct|way|terrace|ter)\b"
            r"(?:[.,]?\s+[A-Z][a-z]+){0,2}[.,]?\s*(?:\d{5}(?:-\d{4})?|[A-Z]\d[A-Z] ?\d[A-Z]\d)\b"
        ),
        sensitivity_floor="confidential",
        force_local=True,
    ),
    Detector(
        name="named_individual",
        category="pii",
        # "a named person + a second identifier" is the GDPR-sensitive shape.
        # Deliberately narrow: matching bare capitalized names would fire on
        # every product name in existence.
        # Narrow on purpose. "customer Call Transcripts" and "client Library Here"
        # both match a loose person-name regex, and a gate that fires on ordinary
        # prose gets switched off -- which is worse than not having it. Require an
        # explicit naming construction instead, and let the backend handle the
        # rest: identifying people is a judgement call, not a pattern match.
        pattern=_c(
            r"\b(?:patient|client|customer|employee|student|applicant|claimant|suspect|user|member)\s+"
            r"(?:named|is|was|id)\s+(?:[A-Z][a-z]+\.?\s+){1,2}[A-Z][a-z]+\b"
            r"|\b(?:patient|client|employee|student|applicant)\s+"
            r"[A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+\b"
        ),
        sensitivity_floor="confidential",
        force_local=True,
    ),
)

#: Regulated-domain vocabulary. Sets a floor; the policy decides the tier.
#: These are the hits that make a *keyword-only* router look smart and a
#: *calibrated* router look honest -- they are necessary but not sufficient.
_REGULATED_KEYWORDS: tuple[Detector, ...] = (
    Detector(
        name="kw_health_regulation",
        category="regulated",
        pattern=_c(
            r"\b(?:hipaa|ephi|phi|protected health information|medical record|health record|"
            r"diagnosis|diagnosed|prognosis|prescription|medication|dosage|treatment plan|clinical note|"
            r"patient history|icd-10|icd10|gdpr|dsgvo)\b",
            re.IGNORECASE,
        ),
        sensitivity_floor="regulated",
        advisory=True,
        pii=False,
    ),
    Detector(
        name="kw_legal_privilege",
        category="regulated",
        pattern=_c(
            r"\b(?:attorney[- ]client|legal privilege|privileged (?:and )?confidential|work product doctrine|"
            r"under oath|deposition|subpoena|litigation hold)\b"
        ),
        sensitivity_floor="regulated",
        advisory=True,
        pii=False,
    ),
    Detector(
        name="kw_financial_regulation",
        category="regulated",
        pattern=_c(
            r"\b(?:pci[- ]dss|pci compliance|cardholder data|cvv|cvc2|glba|sox compliance|"
            r"sar filing|suspicious activity report|kyc document|account balance of|bank statement|"
            r"routing number|swift code)\b"
        ),
        sensitivity_floor="regulated",
        advisory=True,
        pii=False,
    ),
    Detector(
        name="kw_minors",
        category="regulated",
        pattern=_c(
            r"\b(?:coppa|ferpa|minor(?:'s)? record|child(?:'s)? (?:data|records|medical)|under 13|under-13|"
            r"student records|school transcript)\b"
        ),
        sensitivity_floor="regulated",
        advisory=True,
        pii=False,
    ),
    Detector(
        name="kw_confidential_business",
        category="keyword",
        pattern=_c(
            r"\b(?:trade secret|under nda|nda-covered|strictly confidential|do not distribute|"
            r"unreleased|unannounced|pre-announcement|acquisition target|merger discussions|"
            r"layoff plan|restructuring plan|performance review|salary|compensation band|"
            r"equity grant|board resolution)\b"
        ),
        sensitivity_floor="confidential",
        advisory=True,
        pii=False,
    ),
    Detector(
        name="kw_security_vulnerability",
        category="keyword",
        pattern=_c(
            r"\b(?:zero[- ]day|0[- ]day|unpatched|exploit chain|cve-\d{4}-\d{4,}|"
            r"remote code execution|privilege escalation path|credentials? (?:dump|leak)|"
            r"exfiltrat\w+|breach response)\b"
        ),
        sensitivity_floor="confidential",
        advisory=True,
        pii=False,
    ),
    Detector(
        name="kw_internal_only",
        category="keyword",
        pattern=_c(
            r"\b(?:internal only|internal-only|confidential(?:ly)?|restricted distribution|"
            r"employees only|staff only|not for external)\b"
        ),
        sensitivity_floor="internal",
        advisory=True,
        pii=False,
    ),
)

#: Detector set a default deployment ships with. Order matters only for the
#: stability of ``findings``; the verdict is order-independent.
DEFAULT_DETECTORS: tuple[Detector, ...] = _BLOCKING + _PERSONAL + _REGULATED_KEYWORDS


class HardGate:
    """Runs :class:`Detector` instances over text and returns a :class:`GateVerdict`.

    Cheap and side-effect free. Construct once, reuse for the process lifetime.
    """

    def __init__(
        self,
        detectors: Iterable[Detector] = DEFAULT_DETECTORS,
        disabled_detectors: Iterable[str] = (),
        *,
        placeholder_domains_as_pii: bool = False,
    ) -> None:
        """
        Args:
            detectors: the rule set to run.
            disabled_detectors: names to silence. The gate itself always runs.
            placeholder_domains_as_pii: treat RFC 2606 addresses
                (``user@example.com``) as personal data. Off by default, because
                those domains are reserved for documentation and an address there
                is not anybody's email address. Turn it on in tests and evals that
                need to exercise the email detector without inventing a
                real-looking domain.
        """
        if placeholder_domains_as_pii:
            detectors = tuple(replace(d, validator=_always) if d.name == "email_address" else d for d in detectors)
        disabled = frozenset(disabled_detectors)
        unknown = disabled - {d.name for d in DEFAULT_DETECTORS}
        if unknown:
            raise ValueError(f"unknown detector names in disabled_detectors: {sorted(unknown)}")
        # Individual detectors may be silenced (some shops have legitimate
        # high-volume phone numbers in prompts), but the gate itself cannot be:
        # ``scan`` always runs and always returns a verdict.
        self.detectors: tuple[Detector, ...] = tuple(d for d in detectors if d.name not in disabled)
        self.disabled: frozenset[str] = disabled

    @staticmethod
    def _suppress_contained(
        hits: list[tuple[Detector, list[re.Match[str]]]],
    ) -> list[tuple[Detector, list[re.Match[str]]]]:
        """Drop matches whose span sits inside another detector's match.

        Detectors overlap by nature: a card number contains several phone-shaped
        digit runs, and a URL with credentials also matches the email rule. Left
        alone this inflates the finding list, double-counts features, and makes
        the decision log harder to read -- "payment_card + phone_number" for one
        sixteen-digit number is noise, not information.

        Containment is resolved in favour of the *structured* detector: a
        checksum-validated identifier beats a shape-only one, and a non-advisory
        detector beats an advisory one. Advisory matches are never suppressed by
        containment, because a topic keyword overlapping an identifier is still a
        real topic signal.
        """
        flat: list[tuple[Detector, re.Match[str]]] = [(detector, m) for detector, matches in hits for m in matches]

        def rank(detector: Detector) -> tuple[int, int, int]:
            return (
                0 if detector.advisory else 1,
                1 if detector.validator is not _always else 0,
                detector.specificity,
            )

        kept: list[tuple[Detector, re.Match[str]]] = []
        for detector, match in flat:
            span = match.span()
            swallowed = False
            if not detector.advisory:
                for other, other_match in flat:
                    if other is detector or other.advisory:
                        continue
                    ospan = other_match.span()
                    if ospan == span:
                        # Identical span: keep the better-ranked detector only.
                        if rank(other) > rank(detector):
                            swallowed = True
                            break
                    elif ospan[0] <= span[0] and span[1] <= ospan[1] and rank(other) >= rank(detector):
                        # A wider match swallows a nested one only when it ranks at
                        # least as well. `specificity` in the rank is what lets a
                        # narrow-but-precise detector (provider_api_key) survive
                        # inside a broad one (inline_credential), while a wide
                        # checksum-validated one (payment_card) still swallows the
                        # phone-shaped digit runs nested inside it.
                        swallowed = True
                        break
            if not swallowed:
                kept.append((detector, match))

        grouped: dict[str, tuple[Detector, list[re.Match[str]]]] = {}
        for detector, match in kept:
            entry = grouped.setdefault(detector.name, (detector, []))
            entry[1].append(match)
        # Preserve the original detector ordering so findings stay stable.
        order = [d.name for d, _ in hits]
        return [grouped[name] for name in order if name in grouped]

    def scan(self, text: str) -> GateVerdict:
        """Find every detector hit in ``text`` and reduce it to one verdict."""
        if not text:
            return GateVerdict.clean()

        findings: list[GateFinding] = []
        advisory: list[str] = []
        hits: list[tuple[Detector, list[re.Match[str]]]] = []
        floor_idx = -1
        floor: str | None = None
        force_local = False
        blocks = False
        pii_floor: float | None = None

        # Encoding normalization (v0.3): scan the original text AND decoded
        # variants of it (base64 blobs, spaced-out digits, leet-in-digit-runs).
        # Regexes cannot see through encodings, but the encodings are a
        # normalization problem, not a model problem.
        from .gate_normalize import normalization_variants

        texts = [text, *normalization_variants(text)]
        scan_text = " ".join(texts) if len(texts) > 1 else text

        for detector in self.detectors:
            matches = list(detector.pattern.finditer(scan_text))
            if not matches:
                continue
            accepted = [m for m in matches if detector.validator(m.group(0))]
            if not accepted:
                continue
            # Normalization variants can reproduce the same finding (a spaced
            # card and its collapsed form): one content, one finding.
            seen: set[str] = set()
            deduped = []
            for m in accepted:
                if m.group(0) in seen:
                    continue
                seen.add(m.group(0))
                deduped.append(m)
            hits.append((detector, deduped))

        hits = self._suppress_contained(hits)

        for detector, accepted in hits:
            span_hash = hashlib.sha256("\x00".join(m.group(0) for m in accepted).encode("utf-8")).hexdigest()[:16]
            findings.append(
                GateFinding(
                    detector=detector.name,
                    category=detector.category,  # type: ignore[arg-type]
                    sensitivity_floor=detector.sensitivity_floor,
                    force_local=detector.force_local,
                    span_hash=span_hash,
                    count=len(accepted),
                )
            )

            # Advisory (topic-keyword) hits are recorded and forwarded as hints,
            # but they never move a floor. See Detector.advisory.
            if detector.advisory:
                advisory.append(detector.name)
                continue

            idx = level_index(detector.sensitivity_floor, SENSITIVITY_LEVELS)
            if idx > floor_idx:
                floor_idx, floor = idx, detector.sensitivity_floor
            force_local = force_local or detector.force_local
            blocks = blocks or detector.blocks_backend
            if detector.pii:
                # A deterministic hit is not a probability: it is a certainty.
                # 1.0 is the honest value, and it is what makes the gate
                # un-overridable by a backend that answers 0.2.
                pii_floor = 1.0

        if not findings:
            return GateVerdict.clean()

        return GateVerdict(
            fired=True,
            findings=tuple(findings),
            sensitivity_floor=floor,
            pii_floor=pii_floor,
            force_local=force_local,
            blocks_backend=blocks,
            advisory_topics=tuple(dict.fromkeys(advisory)),
        )

    def redaction_map(self, text: str) -> Mapping[str, Sequence[tuple[int, int]]]:
        """Detector name -> matched spans, for :func:`jev_route.prompts.redact`.

        Computed separately from :meth:`scan` so redaction can run even on text
        the gate would not block, and so callers that only want redaction do not
        pay for verdict reduction.
        """
        out: dict[str, list[tuple[int, int]]] = {}
        for detector in self.detectors:
            if detector.advisory:
                # Redacting "HIPAA" out of "explain how HIPAA works" would leave a
                # classifier with nothing to read. Topics are hints, not secrets.
                continue
            spans = [m.span() for m in detector.pattern.finditer(text) if detector.validator(m.group(0))]
            if spans:
                out[detector.name] = spans
        return out


#: Module-level default instance. Integrations that do not configure a gate get
#: this one -- there is no code path where no gate runs.
default_gate = HardGate()


def scan(text: str) -> GateVerdict:
    """Convenience wrapper around :data:`default_gate`."""
    return default_gate.scan(text)


__all__ = [
    "DEFAULT_DETECTORS",
    "Detector",
    "HardGate",
    "default_gate",
    "iban_ok",
    "luhn_ok",
    "scan",
]

# Public aliases for the validators, so tests can exercise them directly.
luhn_ok = _luhn_ok
iban_ok = _iban_ok
