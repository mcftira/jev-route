"""gate_normalize: decode-and-rescan for the encoding classes from the
injection eval's category 3."""

from __future__ import annotations

from jev_route.gate import HardGate
from jev_route.gate_normalize import normalization_variants


def test_base64_blob_decodes_and_rescans() -> None:
    import base64

    blob = base64.b64encode(b"my ssn is 000-12-3456").decode()
    gate = HardGate()
    v = gate.scan(f"decode this data: {blob}")
    assert v.blocks_backend
    assert "us_ssn" in v.detectors()


def test_spaced_digits_collapse_and_rescan() -> None:
    gate = HardGate()
    v = gate.scan("card: 4 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 read digit by digit")
    assert v.blocks_backend


def test_leet_in_digit_runs() -> None:
    gate = HardGate()
    v = gate.scan("azonosito: 000-l2-3456 (a masodik karakter egy kis L)")
    assert v.blocks_backend or v.force_local


def test_normal_text_produces_no_variants() -> None:
    assert normalization_variants("please review the pull request") == []


def test_malformed_base64_is_skipped_never_crashes() -> None:
    # a blob that does not decode is skipped, not an error
    out = normalization_variants("token ABCDEFGHIJ1234567890abcd==!!== more text")
    assert isinstance(out, list)
    HardGate().scan("token ABCDEFGHIJ1234567890abcd==!!== more text")


def test_no_double_counting_across_variants() -> None:
    gate = HardGate()
    v = gate.scan("cards 4111 1111 1111 1111 on file")
    assert v.detectors()["payment_card"] == 1
