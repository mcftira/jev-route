"""The contrastive FLIP eval: the decision must change when the key fact changes.

Reads evals/contrastive/pairs_eval.jsonl (regenerate with gen_pairs.py;
deterministic seeds, disjoint from the train split). Keyless in CI: the
deterministic gate scores sensitivity pairs, the MockBackend-shaped tier rules
score routing pairs. A model run against this eval is a manual local step whose
result file is the committed artifact.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from jev_route.distill.contrastive import KeylessFlipScorer, ROUTING_DOMAIN, SENSITIVITY_DOMAIN


def evaluate(pairs_path: Path) -> dict:
    scorer = KeylessFlipScorer()
    rows = [json.loads(line) for line in pairs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    per_domain: dict[str, Counter] = {}
    per_kind: dict[str, Counter] = {}
    misses: list[dict] = []
    for row in rows:
        base = scorer.score(row["base"], row["domain"])
        variant = scorer.score(row["variant"], row["domain"])
        flipped = base != variant
        per_domain.setdefault(row["domain"], Counter())
        per_kind.setdefault(row["kind"], Counter())
        per_domain[row["domain"]]["flip" if flipped else "no-flip"] += 1
        per_kind[row["kind"]]["flip" if flipped else "no-flip"] += 1
        if not flipped:
            misses.append({"id": row["id"], "kind": row["kind"], "decision": base})
    def rate(c: Counter) -> float:
        total = c["flip"] + c["no-flip"]
        return c["flip"] / total if total else 0.0
    return {
        "n_pairs": len(rows),
        "flip_accuracy": rate(Counter({"flip": sum(c["flip"] for c in per_domain.values()),
                                       "no-flip": sum(c["no-flip"] for c in per_domain.values())})),
        "per_domain": {d: rate(c) for d, c in per_domain.items()},
        "per_kind": {k: rate(c) for k, c in per_kind.items()},
        "misses": misses,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(Path(__file__).resolve().parent / "pairs_eval.jsonl"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "result.json"))
    args = ap.parse_args()
    result = evaluate(Path(args.pairs))
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"flip accuracy: {result['flip_accuracy']:.3f} on {result['n_pairs']} pairs")
    for k, v in result["per_kind"].items():
        print(f"  {k}: {v:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
