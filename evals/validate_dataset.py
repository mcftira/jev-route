#!/usr/bin/env python3
"""Strict validator for the jev-route labeled prompt evaluation dataset.

Usage:
    python3 evals/validate_dataset.py            # validate evals/data/labeled_prompts.jsonl
    python3 evals/validate_dataset.py PATH.jsonl # validate an explicit file
    python3 evals/validate_dataset.py --quiet    # only print the coverage table / failures

Exits 0 when every check passes, 1 otherwise. Stdlib only, Python 3.9+.

The validator is deliberately strict: it is the guard rail that keeps the
dataset honest. It re-derives expected_tier from the labels using the default
policy, enforces enum values, coverage minimums, and a synthetic-PII safety
policy (no real-looking SSNs, card numbers, emails, phone numbers, IBANs or
credential-shaped tokens).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "labeled_prompts.jsonl"

# ---------------------------------------------------------------------------
# Schema / rubric constants
# ---------------------------------------------------------------------------
COMPLEXITIES = ("trivial", "standard", "hard", "frontier")
SENSITIVITIES = ("public", "internal", "confidential", "regulated")
DOMAINS = ("code", "writing", "analysis", "chat", "data-extraction")
DIFFICULTIES = ("clear", "ambiguous")
TIERS = ("local", "cheap", "strong")

REQUIRED_KEYS = ("id", "text", "labels", "expected_tier", "difficulty", "notes")
LABEL_KEYS = ("complexity", "sensitivity", "pii", "domain")

ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

# ---------------------------------------------------------------------------
# Coverage minimums
# ---------------------------------------------------------------------------
MIN_TOTAL = 180
MIN_PER_COMPLEXITY = 25
MIN_PER_SENSITIVITY = 25
MIN_PER_DOMAIN = 25
MIN_PII_TRUE = 40
MIN_PII_FALSE_CONF_REG = 40  # pii false AND sensitivity in {confidential, regulated}
MIN_NEGATIVE_CONTROLS = 15  # ids starting with "neg-"
MIN_INJECTION = 8  # ids starting with "inj-"
MIN_HUNGARIAN = 10  # ids starting with "hu-"
MIN_GERMAN = 5  # ids starting with "de-"
MIN_LONG = 5  # prompts of at least LONG_CHARS characters
LONG_CHARS = 800
MIN_TEXT_CHARS = 8
MAX_TEXT_CHARS = 6000
MIN_NOTES_CHARS = 20
AMBIGUOUS_BAND = (0.10, 0.40)

# ---------------------------------------------------------------------------
# Synthetic-PII safety policy
# ---------------------------------------------------------------------------
# RFC 2606 reserved documentation domains. An address here is NOT personal data,
# so it is allowed anywhere, and it may never be the only reason a row says pii=true.
RESERVED_EMAIL_DOMAINS = {"example.com", "example.org", "example.edu", "example.net"}

# Invented organisation and consumer-ISP domains used to make PII rows realistic.
# Every one of these is fictional; no domain belonging to a real company, mail
# provider, hospital or bank may be added to this set.
FICTIONAL_ORG_DOMAINS = {
    "northside-health.org",
    "hartmann-klinik.de",
    "drkovacs-praxis.hu",
    "northwind-logistics.io",
    "acme-logistics.io",
    "nordkapp-logistics.fi",
    "vireobank.com",
    "falconridge-capital.com",
    "kestrel-analytics.com",
    "brightpath-schools.org",
    "keller-legal.com",
    "trevallyn-press.co.uk",
    "lindqvist-bygg.se",
    "volanynet.hu",
    "postboxmail.com",
    "schnellpost.de",
    "levelezo.hu",
    "postafiok.hu",
    "pureunmail.kr",
}

ALLOWED_EMAIL_DOMAINS = RESERVED_EMAIL_DOMAINS | FICTIONAL_ORG_DOMAINS
ALLOWED_CARD_NUMBERS = {
    "4111111111111111",  # Visa test
    "4012888888881881",  # Visa test
    "4242424242424242",  # Stripe Visa test
    "5555555555554444",  # Mastercard test
    "5105105105105100",  # Mastercard test
    "378282246310005",  # Amex test
    "371449635398431",  # Amex test
    "6011111111111117",  # Discover test
}
ALLOWED_IBANS = {
    "GB33BUKB20201555555555",
    "GB29NWBK60161331926819",
    "DE89370400440532013000",
    "FR1420041010050500013M02606",
    "CH9300762011623852957",
}
ALLOWED_SECRET_TOKENS = {
    "AKIAIOSFODNN7EXAMPLE",  # AWS documentation example
    "sk-test-00000000000000000000000000000000",  # obviously inert placeholder
}

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?:[A-Z0-9]{8,32})\b")
CARD_RE = re.compile(r"\b(?:\d[ \-]?){12,18}\d\b")
SSN_RE = re.compile(r"(?<!\d)(\d{3})-(\d{2})-(\d{4})(?!\d)")
PHONE_PATTERNS = (
    re.compile(r"(?<!\d)(?:\(\d{3}\)[\s.\-]?|\d{3}[\s.\-])\d{3}[\s.\-]\d{4}(?!\d)"),
    re.compile(r"(?<!\d)\d{3}[\s.\-]\d{4}(?!\d)"),
    re.compile(r"\+\d[\d\s().\-]{7,}\d"),
)
# Credential-shaped tokens. Each pattern is written tightly (explicit separator,
# fixed-width body) so ordinary words such as "skip-level" cannot match.
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bpk_(?:live|test)_[A-Za-z0-9]{12,}"),
    re.compile(r"\brk_(?:live|test)_[A-Za-z0-9]{12,}"),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{25,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"\bgsk_[A-Za-z0-9]{20,}"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{10,}"),
)


def domain_is_reserved(domain: str) -> bool:
    """True for RFC 2606 documentation domains and their subdomains."""
    return any(domain == d or domain.endswith("." + d) for d in RESERVED_EMAIL_DOMAINS)


def domain_is_allowed(domain: str) -> bool:
    """True for reserved documentation domains and the invented dataset domains."""
    return any(domain == d or domain.endswith("." + d) for d in ALLOWED_EMAIL_DOMAINS)


# Signals that a prompt really does carry personal data. A row labelled pii=true
# must contain at least one of these; an RFC 2606 address alone never counts,
# because an address at example.com identifies nobody.
# Note: patterns are written as complete-phrase alternations. A trailing \b on a
# group that ends in a single digit or in an accented Hungarian stem silently
# fails to match, so boundaries are placed per alternative instead.
PII_SIGNAL_PATTERNS = (
    (
        "national-id",
        re.compile(
            r"(?i)\b(?:mrn|medical record number|passport(?:\s+number)?|social security"
            r"(?:\s+number)?|ssn|taj\s+sz[aá]ma|tajszám|ad[oó]azonos[ií]t[oó]|national id"
            r"|national insurance number)"
        ),
    ),
    (
        "birth-date",
        re.compile(
            r"(?i)\b(?:dob|date of birth|born|birthday|sz[uú]let[eé]si|geburtsdatum|geboren"
            r"|n[aé]e)\b"
        ),
    ),
    (
        "address",
        re.compile(
            r"(?i)\b(?:home address|address is|address:|shipping to|lakc[ií]me|adresse"
            r"|last known location|precise location|geolocation)\b"
        ),
    ),
    (
        "account",
        re.compile(
            r"(?i)\b(?:account\s+(?:number|no|#|\d)|customer id|iban\b|sort code|routing number"
            r"|banksz[aá]mlasz[aá]m)"
        ),
    ),
    ("compensation", re.compile(r"(?i)\b(?:salary|compensation|gross annual|payroll|fizet[eé]s)\b")),
    ("record-id", re.compile(r"(?i)\b(?:employee id|patient id|pupil record|staff number|device tag)\b")),
    (
        "health-record",
        re.compile(
            r"(?i)\b(?:diagnosis|diagn[oó]zis|discharge summary|lab results|clinical note"
            r"|lelet|hba1c)\b"
        ),
    ),
)


def default_policy(labels: dict) -> str:
    """Mirror of the default routing policy in the task specification."""
    if labels["sensitivity"] in ("confidential", "regulated") or labels["pii"] is True:
        return "local"
    if labels["complexity"] == "frontier":
        return "strong"
    if labels["complexity"] == "hard":
        return "strong"
    return "cheap"


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def fail(self, msg: str) -> None:
        self.failures.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def check_line_syntax(raw_lines: list[str], rep: Report) -> list[dict]:
    rows: list[dict] = []
    for lineno, line in enumerate(raw_lines, start=1):
        if not line.strip():
            rep.fail(f"line {lineno}: blank line (JSONL must have no empty lines)")
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            rep.fail(f"line {lineno}: malformed JSON ({exc})")
            continue
        if not isinstance(obj, dict):
            rep.fail(f"line {lineno}: top-level value is {type(obj).__name__}, expected object")
            continue
        rows.append(obj)
    return rows


def check_schema(row: dict, lineno: int, rep: Report) -> None:
    keys = list(row.keys())
    missing = [k for k in REQUIRED_KEYS if k not in keys]
    extra = [k for k in keys if k not in REQUIRED_KEYS]
    if missing:
        rep.fail(f"line {lineno}: missing keys {missing}")
    if extra:
        rep.fail(f"line {lineno}: extra keys {extra}")
    if missing or extra:
        return

    pid = row["id"]
    if not isinstance(pid, str) or not ID_RE.fullmatch(pid):
        rep.fail(f"line {lineno}: id {pid!r} is not a lowercase hyphenated slug")
    if not isinstance(row["text"], str):
        rep.fail(f"line {lineno} ({pid}): text is not a string")
    if not isinstance(row["notes"], str) or len(row["notes"].strip()) < MIN_NOTES_CHARS:
        rep.fail(f"line {lineno} ({pid}): notes must be a sentence of >= {MIN_NOTES_CHARS} chars")
    if row["difficulty"] not in DIFFICULTIES:
        rep.fail(f"line {lineno} ({pid}): difficulty {row['difficulty']!r} not in {DIFFICULTIES}")
    if row["expected_tier"] not in TIERS:
        rep.fail(f"line {lineno} ({pid}): expected_tier {row['expected_tier']!r} not in {TIERS}")

    labels = row["labels"]
    if not isinstance(labels, dict):
        rep.fail(f"line {lineno} ({pid}): labels must be an object")
        return
    lmissing = [k for k in LABEL_KEYS if k not in labels]
    lextra = [k for k in labels if k not in LABEL_KEYS]
    if lmissing:
        rep.fail(f"line {lineno} ({pid}): labels missing {lmissing}")
    if lextra:
        rep.fail(f"line {lineno} ({pid}): labels has extra keys {lextra}")
    if lmissing or lextra:
        return
    if labels["complexity"] not in COMPLEXITIES:
        rep.fail(f"line {lineno} ({pid}): complexity {labels['complexity']!r} not in {COMPLEXITIES}")
    if labels["sensitivity"] not in SENSITIVITIES:
        rep.fail(f"line {lineno} ({pid}): sensitivity {labels['sensitivity']!r} not in {SENSITIVITIES}")
    if labels["domain"] not in DOMAINS:
        rep.fail(f"line {lineno} ({pid}): domain {labels['domain']!r} not in {DOMAINS}")
    if not isinstance(labels["pii"], bool):
        rep.fail(f"line {lineno} ({pid}): pii must be a JSON boolean, got {labels['pii']!r}")


def check_text_and_policy(row: dict, lineno: int, rep: Report) -> None:
    pid = row.get("id", f"line{lineno}")
    text = row["text"]
    labels = row["labels"]
    if not isinstance(text, str) or not all(k in labels for k in LABEL_KEYS):
        return
    if not isinstance(labels["pii"], bool):
        return
    if len(text.strip()) < MIN_TEXT_CHARS:
        rep.fail(f"{pid}: text shorter than {MIN_TEXT_CHARS} characters")
    if len(text) > MAX_TEXT_CHARS:
        rep.fail(f"{pid}: text longer than {MAX_TEXT_CHARS} characters")
    if "\r" in text:
        rep.fail(f"{pid}: text contains a carriage return")

    want = default_policy(labels)
    if row["expected_tier"] != want:
        rep.fail(
            f"{pid}: expected_tier {row['expected_tier']!r} disagrees with the default policy "
            f"({want!r}) for labels {labels}"
        )

    if pid.startswith("hu-") and "Hungarian" not in row["notes"]:
        rep.fail(f"{pid}: notes must mention 'Hungarian' for hu- prefixed prompts")
    if pid.startswith("de-") and "German" not in row["notes"]:
        rep.fail(f"{pid}: notes must mention 'German' for de- prefixed prompts")


def check_pii_safety(row: dict, lineno: int, rep: Report) -> None:
    """Enforce the synthetic-PII policy on the prompt text."""
    pid = row.get("id", f"line{lineno}")
    text = row["text"]
    if not isinstance(text, str):
        return

    work = text

    for iban in IBAN_RE.findall(work):
        if iban not in ALLOWED_IBANS:
            rep.fail(f"{pid}: IBAN-shaped token {iban!r} is not a documented test IBAN")
    work = IBAN_RE.sub(" <IBAN> ", work)

    for card_match in CARD_RE.finditer(work):
        digits = re.sub(r"[\s\-]", "", card_match.group(0))
        if len(digits) >= 13 and digits not in ALLOWED_CARD_NUMBERS:
            rep.fail(
                f"{pid}: card-shaped number {card_match.group(0).strip()!r} is not a known "
                f"test number; only public test ranges are allowed"
            )
    work = CARD_RE.sub(" <CARD> ", work)

    realistic_emails: list[str] = []
    reserved_emails: list[str] = []
    for email in EMAIL_RE.findall(work):
        domain = email.rsplit("@", 1)[1].lower()
        if not domain_is_allowed(domain):
            rep.fail(
                f"{pid}: email {email!r} uses domain {domain!r}, which is neither an RFC 2606 "
                f"reserved domain nor one of the invented dataset domains"
            )
        elif domain_is_reserved(domain):
            reserved_emails.append(email)
        else:
            realistic_emails.append(email)

    for m in SSN_RE.finditer(work):
        area = m.group(1)
        if not (area in ("000", "666") or area.startswith("9")):
            rep.fail(f"{pid}: SSN-shaped value {m.group(0)!r} uses area {area}; only 000/666/9xx are safe")

    for pat in PHONE_PATTERNS:
        for m in pat.finditer(work):
            digits = re.sub(r"\D", "", m.group(0))
            if "555" not in digits or "01" not in digits:
                rep.fail(f"{pid}: phone-shaped value {m.group(0)!r} is not in the fictional 555-01xx range")

    for pat in SECRET_PATTERNS:
        for token in pat.finditer(work):
            if token.group(0) not in ALLOWED_SECRET_TOKENS:
                rep.fail(f"{pid}: credential-shaped token {token.group(0)!r} is not in the allowed synthetic-key list")

    check_pii_grounding(
        row,
        lineno,
        rep,
        text=text,
        masked=work,
        realistic_emails=realistic_emails,
        reserved_emails=reserved_emails,
    )


def check_pii_grounding(
    row: dict,
    lineno: int,
    rep: Report,
    *,
    text: str,
    masked: str,
    realistic_emails: list[str],
    reserved_emails: list[str],
) -> None:
    """Keep the pii label honest in both directions.

    * pii=true must be grounded in the text. An RFC 2606 address (example.com and
      friends) identifies nobody, so a row whose only "PII" is such an address is
      a labelling bug: the local PII gate will never fire on it and the row would
      silently claim coverage the dataset does not have.
    * pii=false must not contain a realistic address on an invented org domain,
      because that is exactly the case where the gate should fire.
    """
    labels = row.get("labels")
    pid = row.get("id", f"line{lineno}")
    if not isinstance(labels, dict) or not isinstance(labels.get("pii"), bool):
        return

    signals: set[str] = set()
    if realistic_emails:
        signals.add("email")
    if SSN_RE.search(masked):
        signals.add("national-id")
    if IBAN_RE.search(text):
        signals.add("account")
    for pat in PHONE_PATTERNS:
        if pat.search(masked):
            signals.add("phone")
            break
    for m in CARD_RE.finditer(text):
        if len(re.sub(r"[\s\-]", "", m.group(0))) >= 13:
            signals.add("card")
            break
    for _name, pat in PII_SIGNAL_PATTERNS:
        if pat.search(text):
            signals.add(_name)

    if labels["pii"] is True and not signals:
        extra = ""
        if reserved_emails:
            extra = f"; the only addresses present are RFC 2606 reserved ({sorted(set(reserved_emails))[:2]})"
        rep.fail(
            f"{pid}: pii=true but the text carries no personal-data signal{extra}. "
            f"Either ground the label with real synthetic PII or set pii=false."
        )
    if labels["pii"] is False and realistic_emails:
        rep.fail(
            f"{pid}: pii=false but the text contains {realistic_emails[0]!r} on an invented "
            f"organisation domain; that address is personal data, so pii must be true "
            f"(or the address must be an RFC 2606 reserved one)"
        )


def coverage_stats(rows: list[dict]) -> dict:
    ok = [r for r in rows if isinstance(r.get("labels"), dict) and "complexity" in r["labels"]]
    comp = Counter(r["labels"]["complexity"] for r in ok)
    sens = Counter(r["labels"]["sensitivity"] for r in ok)
    dom = Counter(r["labels"]["domain"] for r in ok)
    tier = Counter(r["expected_tier"] for r in ok)
    diff = Counter(r["difficulty"] for r in ok)
    # The refactor that lifted the minimum checks into check_minimums() dropped this
    # function's return value, so main() handed None to both check_minimums and
    # print_table and the coverage gate died with a TypeError instead of ever judging
    # the dataset. A validator that crashes is worse than one that is lenient: it
    # reports nothing at all, which reads as silence rather than as failure.
    return {
        "total": len(rows),
        "complexity": comp,
        "sensitivity": sens,
        "domain": dom,
        "tier": tier,
        "difficulty": diff,
        "pii_true": sum(1 for r in ok if r["labels"]["pii"] is True),
        "pii_false_conf_reg": sum(
            1 for r in ok if r["labels"]["pii"] is False and r["labels"]["sensitivity"] in ("confidential", "regulated")
        ),
        "negative_controls": sum(1 for r in ok if str(r.get("id", "")).startswith("neg-")),
        "injection": sum(1 for r in ok if str(r.get("id", "")).startswith("inj-")),
        "hungarian": sum(1 for r in ok if str(r.get("id", "")).startswith("hu-")),
        "german": sum(1 for r in ok if str(r.get("id", "")).startswith("de-")),
        "long": sum(1 for r in ok if len(str(r.get("text", ""))) >= LONG_CHARS),
    }


def check_minimums(stats: dict, rep: Report) -> None:
    def need(label: str, got: int, want: int) -> None:
        if got < want:
            rep.fail(f"coverage: {label} = {got}, minimum {want}")

    need("total prompts", stats["total"], MIN_TOTAL)
    for level in COMPLEXITIES:
        need(f"complexity[{level}]", stats["complexity"].get(level, 0), MIN_PER_COMPLEXITY)
    for level in SENSITIVITIES:
        need(f"sensitivity[{level}]", stats["sensitivity"].get(level, 0), MIN_PER_SENSITIVITY)
    for level in DOMAINS:
        need(f"domain[{level}]", stats["domain"].get(level, 0), MIN_PER_DOMAIN)
    need("pii == true", stats["pii_true"], MIN_PII_TRUE)
    need("pii == false with sensitivity confidential/regulated", stats["pii_false_conf_reg"], MIN_PII_FALSE_CONF_REG)
    need("negative controls (neg-*)", stats["negative_controls"], MIN_NEGATIVE_CONTROLS)
    need("injection/adversarial (inj-*)", stats["injection"], MIN_INJECTION)
    need("Hungarian prompts (hu-*)", stats["hungarian"], MIN_HUNGARIAN)
    need("German prompts (de-*)", stats["german"], MIN_GERMAN)
    need(f"long prompts (>= {LONG_CHARS} chars)", stats["long"], MIN_LONG)

    total = stats["total"] or 1
    frac = stats["difficulty"].get("ambiguous", 0) / total
    lo, hi = AMBIGUOUS_BAND
    if not (lo <= frac <= hi):
        rep.fail(f"coverage: ambiguous fraction {frac:.1%} outside the target band {lo:.0%}-{hi:.0%}")


def print_table(stats: dict) -> None:
    def block(title: str, counter: Counter, order: tuple[str, ...], minimum: int) -> None:
        print(f"\n{title}  (minimum {minimum} each)")
        print(f"  {'value':<16}{'count':>7}   share")
        total = sum(counter.values()) or 1
        for key in order:
            n = counter.get(key, 0)
            flag = "OK " if n >= minimum else "LOW"
            print(f"  {key:<16}{n:>7}   {n / total:6.1%}  {flag}")

    print(f"jev-route dataset coverage  ({stats['total']} prompts)")
    block("complexity", stats["complexity"], COMPLEXITIES, MIN_PER_COMPLEXITY)
    block("sensitivity", stats["sensitivity"], SENSITIVITIES, MIN_PER_SENSITIVITY)
    block("domain", stats["domain"], DOMAINS, MIN_PER_DOMAIN)
    block("expected_tier", stats["tier"], TIERS, 0)
    block("difficulty", stats["difficulty"], DIFFICULTIES, 0)

    print("\nspecial groups")
    rows = [
        ("pii == true", stats["pii_true"], MIN_PII_TRUE),
        ("pii false + confidential/regulated", stats["pii_false_conf_reg"], MIN_PII_FALSE_CONF_REG),
        ("negative controls (neg-*)", stats["negative_controls"], MIN_NEGATIVE_CONTROLS),
        ("injection / adversarial (inj-*)", stats["injection"], MIN_INJECTION),
        ("Hungarian (hu-*)", stats["hungarian"], MIN_HUNGARIAN),
        ("German (de-*)", stats["german"], MIN_GERMAN),
        (f"long prompts (>= {LONG_CHARS} chars)", stats["long"], MIN_LONG),
    ]
    for label, got, want in rows:
        flag = "OK " if got >= want else "LOW"
        print(f"  {label:<38}{got:>6}   min {want:<4} {flag}")
    frac = stats["difficulty"].get("ambiguous", 0) / (stats["total"] or 1)
    print(f"  {'ambiguous share':<38}{frac:>6.1%}   target {AMBIGUOUS_BAND[0]:.0%}-{AMBIGUOUS_BAND[1]:.0%}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the jev-route labeled prompt dataset.")
    parser.add_argument("path", nargs="?", default=str(DEFAULT_PATH), help="path to the JSONL dataset")
    parser.add_argument("--quiet", action="store_true", help="suppress the coverage table on success")
    args = parser.parse_args(argv)

    path = Path(args.path)
    rep = Report()

    if not path.exists():
        print(f"FAIL: dataset not found at {path}", file=sys.stderr)
        return 1

    raw_bytes = path.read_bytes()
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        rep.fail("file starts with a UTF-8 BOM")
    try:
        content = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        print(f"FAIL: file is not valid UTF-8 ({exc})", file=sys.stderr)
        return 1
    if content and not content.endswith("\n"):
        rep.fail("file does not end with a newline")
    if "\r\n" in content:
        rep.fail("file uses CRLF line endings; JSONL must use LF")

    raw_lines = content.split("\n")
    if raw_lines and raw_lines[-1] == "":
        raw_lines = raw_lines[:-1]

    rows = check_line_syntax(raw_lines, rep)

    seen_ids: dict[str, int] = {}
    seen_texts: dict[str, str] = {}
    for lineno, row in enumerate(rows, start=1):
        pid = row.get("id")
        text = row.get("text")
        if isinstance(pid, str):
            if pid in seen_ids:
                rep.fail(f"duplicate id {pid!r} on lines {seen_ids[pid]} and {lineno}")
            else:
                seen_ids[pid] = lineno
        if isinstance(text, str):
            key = " ".join(text.split())
            if key in seen_texts:
                rep.fail(f"duplicate text: {pid!r} repeats the prompt of {seen_texts[key]!r}")
            else:
                seen_texts[key] = pid if isinstance(pid, str) else f"line{lineno}"

        check_schema(row, lineno, rep)
        check_text_and_policy(row, lineno, rep)
        check_pii_safety(row, lineno, rep)

    stats = coverage_stats(rows)
    if not rep.failures:
        check_minimums(stats, rep)

    if not args.quiet or rep.failures:
        print_table(stats)

    for w in rep.warnings:
        print(f"WARN: {w}")
    if rep.failures:
        print(f"\nFAIL: {len(rep.failures)} problem(s) found in {path}", file=sys.stderr)
        for f in rep.failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(f"\nPASS: {len(rows)} prompts, all checks green ({path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
