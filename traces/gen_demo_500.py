"""Generate traces/demo_500.jsonl: 500 synthetic requests for the backtest.

EVERY row is synthetic and provably fake: SSNs use the 000 area (never issued),
cards fail Luhn by construction, emails live on RFC 2606 reserved domains,
names are placeholders. No real user traffic is ever committed. Deterministic
seed so the trace is reproducible byte-for-byte.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

CATEGORIES = {
    "trivial_completion": 150,
    "code_gen": 120,
    "multi_file_refactor": 80,
    "hungarian_support": 50,
    "pii_fake": 60,
    "long_context": 40,
}

TRIVIAL = [
    "Say hello in three words.",
    "What is 12 * 8?",
    "Translate 'good morning' to German.",
    "One-line bash to count files in a directory.",
    "Is Pluto a planet? One sentence.",
]
CODE = [
    "Write a Python function that reverses a linked list with tests.",
    "Implement binary search in Rust with error handling.",
    "Add retries with exponential backoff to this fetch wrapper.",
    "Write a dataclass for a 3D vector with dot/cross products.",
    "SQL query: top 10 customers by order volume this quarter.",
]
REFACTOR = [
    "Refactor the auth module: split token refresh, signing, and rotation into separate units, keep the public API stable, and update the 12 call sites across the codebase.",
    "Migrate the settings layer from dicts to a typed config dataclass; update all readers and write a migration note for downstream services.",
    "Split the 900-line report generator into parser, layout, and renderer modules with clean seams and full test coverage for the parser.",
    "Convert the callback-based websocket client to async/await end to end, including reconnect logic and the test harness.",
]
HUNGARIAN = [
    "Szia! Nem mukodik a bejelentkezes, 403-at kapok minden oldalon. Mit csinaljak?",
    "A rendszerem lefagy amikor exportalok. Hiba vagy feature?",
    "Kerem segitsetek: az alkalmazas nem indul el a frissites ota.",
    "Hogyan tudom visszaallitani a tegnapi mentest? Koszonom!",
    "A szamlamra nem erkezett meg a visszautalas, mar ket hete.",
]
FAKE_PII = [
    "My SSN is 000-12-3456, can you check my tax status?",
    "Card on file: 4111 1111 1111 1112 (note: fails checksum), why was it declined?",
    "Email me at jane.doe@example.com with the quote.",
    "Patient record for John Placeholder, born 1900-01-01, needs review.",
    "API key sk-FAKE0000TESTKEY0000DONOTUSE00 stopped working, rotate it please.",
    "My address is 123 Fictional Street, Nowhere City, ZIP 00000. Ship the invoice there.",
]
LONG_CTX = [
    "Here is a 3,000-line log excerpt. Find the first error and summarize the cascade. " + "INFO ok ok ok. " * 120,
    "Review this full design doc and list every contradiction. " + "Section overview. " * 150,
]


def rows_for(category: str, n: int, rng: random.Random) -> list[dict]:
    bank = {
        "trivial_completion": TRIVIAL,
        "code_gen": CODE,
        "multi_file_refactor": REFACTOR,
        "hungarian_support": HUNGARIAN,
        "pii_fake": FAKE_PII,
        "long_context": LONG_CTX,
    }[category]
    tier = {
        "trivial_completion": "local",
        "code_gen": "cheap",
        "multi_file_refactor": "strong",
        "hungarian_support": "cheap",
        "pii_fake": "local",       # the gate must force these local anyway
        "long_context": "strong",
    }[category]
    out = []
    for i in range(n):
        text = rng.choice(bank)
        out.append({
            "id": f"{category}-{i:03d}",
            "category": category,
            "text": text,
            "expected_tier": tier,
            "pii_expected": category == "pii_fake",
        })
    return out


def main() -> None:
    rng = random.Random(20260921)
    rows = []
    for category, n in CATEGORIES.items():
        rows.extend(rows_for(category, n, rng))
    rng.shuffle(rows)
    out = Path(__file__).resolve().parent / "demo_500.jsonl"
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + chr(10))
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
