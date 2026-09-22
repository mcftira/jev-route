"""Encoding normalization for the hard gate: decode-then-rescan.

The injection eval's category 3 (encoding tricks) exposed the deterministic
layer's blind spot: regexes cannot see through base64 or spaced-out digits.
But those encodings are NOT a model problem -- they are a normalization
problem. This module produces decoded variants of the text; the gate scans
the original AND every variant, so a base64 blob containing an SSN fires the
SSN detector on the decoded content.

Depth is one: variants are never re-normalized (a blob inside a blob is not
our threat model). Fail-closed per variant: a blob that does not decode is
skipped, never an error.
"""

from __future__ import annotations

import base64
import re

#: Base64-shaped runs: long enough to carry a payload, OR short but padded
#: (the "=" terminator is the tell). False positives are harmless by design:
#: a variant only matters if its decoded content trips a detector.
_B64_RE = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}|[A-Za-z0-9+/]{12,}={1,2}")

#: Digit runs separated by single spaces ("4 1 1 1 1 1 1 1"): 8+ digits.
_SPACED_DIGITS_RE = re.compile(r"(?:\d\s){7,}\d")

#: Mostly-digit tokens with letter confusables ("000-l2-3456" where the second
#: char is a lowercase L): >=6 chars, at least half digits.
_MIXED_RE = re.compile(r"[0-9][0-9A-Za-z\-\s]{4,}[0-9A-Za-z]")

_LEET = str.maketrans({"l": "1", "I": "1", "O": "0", "o": "0"})


def normalization_variants(text: str) -> list[str]:
    """Decoded/normalized variants of ``text`` worth rescanning."""
    out: list[str] = []
    for m in _B64_RE.finditer(text):
        try:
            decoded = base64.b64decode(m.group(0), validate=True).decode("utf-8", errors="strict")
        except Exception:
            continue
        if decoded.strip():
            out.append(decoded)
    for m in _SPACED_DIGITS_RE.finditer(text):
        collapsed = m.group(0).replace(" ", "")
        if collapsed != m.group(0):
            out.append(collapsed)
    for m in _MIXED_RE.finditer(text):
        tok = m.group(0)
        digits = sum(ch.isdigit() for ch in tok)
        if digits * 2 < len(tok):
            continue
        fixed = tok.translate(_LEET)
        if fixed != tok:
            out.append(fixed)
    return out
