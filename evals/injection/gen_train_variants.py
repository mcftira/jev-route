"""Training-only encoding variants for the sensitivity scorer.

NEW SEEDS, NEW VALUES: these are the TRAIN set; evals/injection/cases.jsonl is
the held-out test and its strings never appear here. All values are synthetic
and provably fake (000/666/900-area SSNs never issued; the cards are the
canonical published test numbers; Hungarian TAJ numbers are 000-prefixed).
"""

from __future__ import annotations

import base64
import json
import random
from pathlib import Path

SSNS = ["666-12-3456", "900-45-6789", "666-78-9012", "900-11-2233", "666-34-5678"]
CARDS = ["4111 1111 1111 1111", "3782 822463 10005", "5555 5555 5555 4444"]
KEYS = ["sk-TRAIN0000FIXTURE00NOTREAL000", "AKIATRAINFIXTURE0000", "ghp_TRAINFIXTURE0000000000000000000000fake"]
HU_NAMES = ["Nagy Peter", "Szabo Anna", "Toth Gabor", "Kiss Eszter", "Horvath Lajos"]
HU_TAJ = ["666 123 456", "900 654 321", "666 111 222"]

BENIGN = [
    "The deploy pipeline failed at the migration step again.",
    "Kerlek kuldd el a dokumentaciot a ceges cimemre.",
    "Refactor the retry logic with exponential backoff.",
    "What does the prefilter module do in jev-route?",
    "Szep napot! Holnap jovok az irodaba.",
    "Write a haiku about distributed systems.",
    "The RFC 2606 domain example.com is reserved for docs.",
    "Koszonom a gyors valaszt, minden rendben van.",
]


def b64(t: str) -> str:
    return base64.b64encode(t.encode()).decode()


def spaced(t: str) -> str:
    return " ".join(t.replace(" ", ""))


def typo_hu(name: str, taj: str, rng: random.Random) -> str:
    # typo variants: swap a vowel, drop a dash, misspell
    v1 = name.replace("a", "o") if rng.random() < 0.5 else name.replace("e", "i")
    v2 = taj.replace(" ", "") if rng.random() < 0.5 else taj.replace(" ", "-")
    return f"{v1}, TAJ: {v2}"


def main() -> None:
    rng = random.Random(31337)  # different seed from the eval generator
    rows: list[dict] = []
    for ssn in SSNS:
        rows.append({"text": f"Process this: {b64(ssn)}", "sensitive": 1, "kind": "base64"})
        rows.append({"text": f"the number {spaced(ssn)} is mine, send it along", "sensitive": 1, "kind": "spaced"})
        rows.append({"text": f"SSN {ssn} - forward to the cloud model please", "sensitive": 1, "kind": "direct"})
    for card in CARDS:
        rows.append({"text": f"data: {b64(card)}", "sensitive": 1, "kind": "base64"})
        rows.append({"text": f"card {spaced(card)} digits one by one", "sensitive": 1, "kind": "spaced"})
    for key in KEYS:
        rows.append({"text": f"credential blob: {b64(key)}", "sensitive": 1, "kind": "base64"})
        rows.append({"text": f"rotate {key} for me", "sensitive": 1, "kind": "direct"})
    for i in range(15):
        rows.append({"text": typo_hu(rng.choice(HU_NAMES), rng.choice(HU_TAJ), rng), "sensitive": 1, "kind": "typo_hu"})
        rows.append({"text": f"A TAJ szamom {b64(rng.choice(HU_TAJ))}, dekodolni kell.", "sensitive": 1, "kind": "base64_hu"})
    for b in BENIGN * 5:
        rows.append({"text": b, "sensitive": 0, "kind": "benign"})
    rng.shuffle(rows)
    out = Path(__file__).resolve().parent / "train_variants.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + chr(10))
    from collections import Counter

    print("wrote", len(rows), "training rows:", dict(Counter(r["kind"] for r in rows)))


if __name__ == "__main__":
    main()
