"""Generate the v0.5 contrastive split: ``pairs_train.jsonl`` + ``pairs_eval.jsonl``.

Contrastive data curation for the held-out flip eval (Nimble's method): every
row is a pair of texts that differ by exactly one mutation that flips the
correct decision. Sensitivity pairs isolate one PII fact (a card check digit,
a TAJ number's presence vs redaction, an SSN area's reservation); routing pairs
isolate one scope fact (one-line fix vs multi-file refactor vs schema migration
of the same base task).

Conventions, inherited from ``evals/injection/gen_cases.py`` and
``src/jev_route/distill/synthetic_pii.py``:

* ALL identifiers are synthetic and provably fake or publicly reserved:
  published vendor test PANs (Luhn-pass member) and the same PANs with the
  check digit broken (Luhn-fail member); SSA never-issued SSN areas for the
  non-sensitive member; the repo's 000-led 9-digit TAJ fixture convention.
  Nothing is a real person's data and nothing is a real PAN.
* Every pair carries its provenance on the row (the mutation, the fakeness
  reservations, the citations), so the dataset is auditable without
  re-deriving anything.

Determinism: the split is a pure function of the seeds (no clock, no
environment, no network). The train and eval seeds are different, and this
script verifies the two splits share no id and no text -- "train != eval"
is checked absolutely, not by convention. Rerun to regenerate byte-identical
files:

    python evals/contrastive/gen_pairs.py            # writes into evals/contrastive/
    python evals/contrastive/gen_pairs.py --out-dir /tmp/x   # redirect
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from jev_route.distill.contrastive import (
    ROUTING_DOMAIN,
    SENSITIVITY_DOMAIN,
    TRAIN_SPLIT,
    EVAL_SPLIT,
    ContrastivePair,
    generate_pairs,
    write_pairs_jsonl,
)

#: New seeds, different by construction: the eval split must never see a pair
#: the model trained on.
TRAIN_SEED = "train-v0.5"
EVAL_SEED = "eval-v0.5"

HERE = Path(__file__).resolve().parent


def assert_disjoint(train: list[ContrastivePair], eval_pairs: list[ContrastivePair]) -> None:
    """The splits share no id and no text, absolutely.

    "Different seeds" is a means, not a guarantee: two seeds can still draw the
    same template and the same fillers, and a pair in both splits would put the
    model's training answer under the model's held-out grade. So the disjointness
    is verified on the actual outputs -- ids and both text slots -- and the
    generator refuses to run past a violation.
    """
    shared_ids = sorted({p.id for p in train} & {p.id for p in eval_pairs})
    if shared_ids:
        raise AssertionError(f"train and eval share pair ids: {shared_ids[:5]}")
    train_texts = {p.base for p in train} | {p.variant for p in train}
    eval_texts = {p.base for p in eval_pairs} | {p.variant for p in eval_pairs}
    shared_texts = train_texts & eval_texts
    if shared_texts:
        raise AssertionError(f"train and eval share {len(shared_texts)} texts; the splits must be disjoint")


def provenance(train: list[ContrastivePair], eval_pairs: list[ContrastivePair]) -> dict[str, Any]:
    """The deterministic sidecar: seeds, counts, fakeness. No timestamp, so the
    files this script writes are byte-identical across runs."""

    def by_domain(pairs: list[ContrastivePair]) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for domain in (SENSITIVITY_DOMAIN, ROUTING_DOMAIN):
            out[domain] = dict(Counter(p.kind for p in pairs if p.domain == domain))
        return out

    return {
        "generator": "jev_route.distill.contrastive.generate_pairs",
        "train_seed": TRAIN_SEED,
        "eval_seed": EVAL_SEED,
        "train_pairs": by_domain(train),
        "eval_pairs": by_domain(eval_pairs),
        "disjointness": "verified at generation time: no shared pair id, no shared text (base or variant)",
        "fakeness": {
            "card_luhn": (
                "variant: published vendor test PAN (Luhn-valid, publicly reserved, citation on the row); "
                "base: the same PAN with the check digit broken (Luhn-invalid, no issuer assigns it)"
            ),
            "taj": "9-digit TAJ placeholders with the 000 lead group (repo fixture convention), constructed, never captured",
            "ssn": (
                "base: SSA never-issued area numbers 000/666/900-999 (provably not a person's number); "
                "variant: structurally-issuable area 219, a constructed placeholder in a controlled corpus, never captured"
            ),
            "routing": "no PII; fictional module, table and job names only",
        },
        "note": "no real PANs, SSNs, TAJ numbers, names or credentials; every identifier is constructed",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(HERE), help="directory to write the split into")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = generate_pairs(TRAIN_SEED, TRAIN_SPLIT, id_prefix="ct")
    # Eval is generated against the train texts: disjointness by construction,
    # and assert_disjoint below stays as the check on the actual outputs.
    train_texts = {p.base for p in train} | {p.variant for p in train}
    eval_pairs = generate_pairs(EVAL_SEED, EVAL_SPLIT, id_prefix="ce", exclude_texts=train_texts)
    assert_disjoint(train, eval_pairs)

    write_pairs_jsonl(train, str(out_dir / "pairs_train.jsonl"))
    write_pairs_jsonl(eval_pairs, str(out_dir / "pairs_eval.jsonl"))
    (out_dir / "pairs_provenance.json").write_text(
        json.dumps(provenance(train, eval_pairs), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    train_counts = Counter(p.domain for p in train)
    eval_counts = Counter(p.domain for p in eval_pairs)
    print(
        f"wrote {out_dir / 'pairs_train.jsonl'} ({len(train)} pairs: "
        f"{train_counts[SENSITIVITY_DOMAIN]} sensitivity, {train_counts[ROUTING_DOMAIN]} routing); "
        f"{out_dir / 'pairs_eval.jsonl'} ({len(eval_pairs)} pairs: "
        f"{eval_counts[SENSITIVITY_DOMAIN]} sensitivity, {eval_counts[ROUTING_DOMAIN]} routing); "
        "splits disjoint (no shared id, no shared text)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
