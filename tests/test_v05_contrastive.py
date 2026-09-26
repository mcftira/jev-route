"""v0.5 contrastive: pair generation + flip eval + the promotion criterion."""

from __future__ import annotations

import json
from pathlib import Path

from jev_route.distill.contrastive import generate_pairs, luhn_ok

CT = Path(__file__).resolve().parents[1] / "evals" / "contrastive"


def test_generator_is_deterministic() -> None:
    a = generate_pairs("s1", split={"card_luhn": 5, "taj": 3, "ssn": 3,
                                    "one_liner_vs_refactor": 4, "one_liner_vs_migration": 4})
    b = generate_pairs("s1", split={"card_luhn": 5, "taj": 3, "ssn": 3,
                                    "one_liner_vs_refactor": 4, "one_liner_vs_migration": 4})
    assert [p.to_dict() for p in a] == [p.to_dict() for p in b]
    c = generate_pairs("s2", split={"card_luhn": 5})
    assert [p.to_dict() for p in c] != [p.to_dict() for p in a][:5]


def test_luhn_labels_are_correct() -> None:
    pairs = generate_pairs("lu", split={"card_luhn": 8})
    assert len(pairs) == 8
    for p in pairs:
        base_digits = "".join(ch for ch in p.provenance["base_value"] if ch.isdigit())
        variant_digits = "".join(ch for ch in p.provenance["variant_value"] if ch.isdigit())
        assert not luhn_ok(base_digits)   # base is checksum-broken by construction
        assert luhn_ok(variant_digits)    # variant is a published test PAN (Luhn-valid)


def test_ssn_pairs_document_the_layer2_need() -> None:
    pairs = generate_pairs("sn", split={"ssn": 5})
    for p in pairs:
        assert p.label_base == 0 and p.label_variant == 1
        assert "gate" in p.provenance and "layer 2" in p.provenance["gate"]


def test_flip_eval_catches_a_never_flipper_and_passes_a_perfect_scorer() -> None:
    from evals.contrastive.run import evaluate

    # the keyless scorer must flip everywhere except the ssn kind (by design)
    result = evaluate(CT / "pairs_eval.jsonl")
    assert 0.0 < result["flip_accuracy"] <= 1.0
    assert result["per_kind"]["card_luhn"] == 1.0
    assert result["per_kind"]["taj"] == 1.0
    assert result["per_kind"]["ssn"] == 0.0  # documented: layer 2 owns this flip


def test_pairs_eval_file_exists_and_is_disjoint_from_train() -> None:
    train_texts = set()
    for line in (CT / "pairs_train.jsonl").read_text().splitlines():
        row = json.loads(line)
        train_texts.add(row["base"])
        train_texts.add(row["variant"])
    for line in (CT / "pairs_eval.jsonl").read_text().splitlines():
        row = json.loads(line)
        assert row["base"] not in train_texts
        assert row["variant"] not in train_texts


def test_contrastive_flip_check(tmp_path: Path) -> None:
    from jev_route.distill.cli import _contrastive_flip_check
    from jev_route.gate_semantic import EnforceCriteria

    criteria = EnforceCriteria()
    ok, measured = _contrastive_flip_check(tmp_path / "missing.json", criteria)
    assert ok is False and "missing file" in measured
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"flip_accuracy": 0.99}))
    ok, measured = _contrastive_flip_check(good, criteria)
    assert ok is True
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"flip_accuracy": 0.85}))
    ok, measured = _contrastive_flip_check(bad, criteria)
    assert ok is False and "0.850" in measured
