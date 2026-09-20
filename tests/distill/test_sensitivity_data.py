"""The sensitivity training-data module: the privacy boundary, pinned by tests.

This module exists for exactly one sentence: *the gate never trains on your
secrets.* Everything in it -- the closed source list, the sendability checks, the
metadata-only projection of blocked records -- is a consequence of that sentence,
and a consequence that has drifted is a leak. So the tests here are written the
way a security reviewer would write them: with a canary that is recognisable if
it ever appears anywhere it should not, and with an AST walk of the module itself,
because a comment saying "blocked content is never read" is not a control.

Two of these tests are specified in the module docstring ("This is enforced three
ways"): the canary test and the AST test. They are not optional decorations; the
docstring names this file as the pin.
"""

from __future__ import annotations

import ast
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

import jev_route.distill.sensitivity_data as sd
from jev_route.distill.export import load_dataset
from jev_route.gate import default_gate
from jev_route.logging_sink import CallbackSink
from jev_route.schema import (
    COMPLEXITY_LEVELS,
    DOMAINS,
    SENSITIVITY_LEVELS,
    DecisionAnswers,
    DecisionRecord,
    GateVerdict,
    RequestFeatures,
    RoutingDecision,
)

#: A string that is impossible to mistake for anything a generator could produce.
CANARY = "CANARY-BLOCKED-SSN-DO-NOT-LEAK-666456789"

#: A published test PAN: Luhn-valid, so the real gate fires on it, and
#: blocks_backend, so it exercises the refusal path with the production code's
#: own verdict rather than a hand-rolled one.
BLOCKING_TEXT = "our card on file is 4242 4242 4242 4242"


def _blocked_verdict() -> GateVerdict:
    return default_gate.scan(BLOCKING_TEXT)


def _record(
    request_id: str,
    *,
    excerpt: str | None,
    blocked: bool = False,
    degraded: bool = False,
    metadata: dict | None = None,
) -> DecisionRecord:
    if blocked:
        verdict = _blocked_verdict()
        backend, tier = "gate", "local"
    else:
        verdict = GateVerdict.clean()
        backend, tier = "mock", "cheap"
    return DecisionRecord(
        request_id=request_id,
        timestamp="2026-01-01T00:00:00+00:00",
        decision=RoutingDecision(
            tier=tier,
            model="mock-1.0.0" if backend == "mock" else "",
            rule_id="gate.force-local" if blocked else "default",
            reason="blocked by the local gate" if blocked else "no rule matched",
            answers=DecisionAnswers.unknown(),
            gate=verdict,
            backend=backend,
            backend_model_version="" if blocked else "mock-1.0.0",
            effective_sensitivity="regulated" if blocked else "public",
            effective_complexity="standard",
            latency_ms=1.0,
            degraded=degraded,
        ),
        features=RequestFeatures(),
        excerpt_hash="sha256:" + ("0" * 16),
        backend_latency_ms=1.0,
        total_latency_ms=1.0,
        excerpt=excerpt,
        metadata=metadata or {},
    )


def _write_log(path: Path, records: list[DecisionRecord]) -> Path:
    path.write_text("\n".join(json.dumps(r.to_dict(), sort_keys=True) for r in records), encoding="utf-8")
    return path


def _sample(source: str, text: str, **kwargs) -> sd.SensitivitySample:
    base: dict = {"sample_id": f"t-{source}-{abs(hash(text)) % 100000:05d}", "text": text, "source": source}
    if source == "synthetic-pii":
        base["fakeness_basis"] = "reserved-range"
    if source == "public-corpus":
        base.update(licence="CC BY 4.0", corpus="test-corpus")
    base.update(kwargs)
    return sd.SensitivitySample(**base)


class TestSourceClosure:
    """The four allowed sources are the whole language; anything else is a type error."""

    def test_an_unknown_source_cannot_be_constructed(self) -> None:
        with pytest.raises(sd.UnknownSourceError, match="closed on purpose"):
            _sample("customer-complaint", "the user pasted their card number")

    def test_public_corpus_requires_a_stated_licence(self) -> None:
        with pytest.raises(sd.CorpusLicenceError, match="licence"):
            sd.SensitivitySample(sample_id="x", text="t", source="public-corpus", corpus="c")
        with pytest.raises(sd.SensitivityDataError, match="corpus"):
            sd.SensitivitySample(sample_id="x", text="t", source="public-corpus", licence="CC BY 4.0")

    def test_synthetic_requires_the_fakeness_claim(self) -> None:
        with pytest.raises(sd.SensitivityDataError, match="fakeness_basis"):
            sd.SensitivitySample(sample_id="x", text="ssn 666-45-6789", source="synthetic-pii")

    def test_empty_text_is_refused(self) -> None:
        with pytest.raises(sd.SensitivityDataError, match="empty text"):
            _sample("contextual-template", "   ")

    def test_an_invalid_expected_sensitivity_is_refused(self) -> None:
        with pytest.raises(sd.SensitivityDataError, match="expected_sensitivity"):
            _sample("contextual-template", "t", expected_sensitivity="apocalyptic")

    def test_source_categoris_are_the_documented_table(self) -> None:
        assert sd.SOURCE_CATEGORIES == (
            "synthetic-pii",
            "contextual-template",
            "public-corpus",
            "production-negative",
        )
        for policy in sd.ALLOWED_SOURCES:
            assert policy.rationale.strip(), "the stats sidecar prints the rationale; an empty one is a dead claim"


class TestSendabilityIsTheSourceTable:
    """may_go_to_cloud must agree with the table, row by row, including the exception."""

    def test_synthetic_is_sendable_even_under_a_blocking_verdict(self) -> None:
        # The documented exception: the identifiers are reserved values, so a
        # blocking verdict on "666-45-6789" is a verdict about a number the SSA
        # will never issue, not about a real person.
        verdict = _blocked_verdict()
        sendable, why = sd.may_go_to_cloud(_sample("synthetic-pii", "ssn 666-45-6789"), verdict)
        assert sendable and why is None

    def test_contextual_is_sendable_under_a_blocking_verdict(self) -> None:
        sendable, _ = sd.may_go_to_cloud(_sample("contextual-template", "hr case notes"), _blocked_verdict())
        assert sendable

    def test_production_is_refused_under_a_blocking_verdict(self) -> None:
        sample = _sample("production-negative", BLOCKING_TEXT)
        verdict = _blocked_verdict()
        sendable, why = sd.may_go_to_cloud(sample, verdict)
        assert not sendable and "blocking detector" in (why or "")
        assert sd.refusal_reason(sample, verdict) == sd.REFUSAL_GATE_BLOCKED_PRODUCTION

    def test_corpus_is_refused_under_a_blocking_verdict(self) -> None:
        sample = _sample("public-corpus", BLOCKING_TEXT)
        verdict = _blocked_verdict()
        assert not sd.may_go_to_cloud(sample, verdict)[0]
        assert sd.refusal_reason(sample, verdict) == sd.REFUSAL_GATE_BLOCKED_CORPUS

    def test_clean_verdicts_are_sendable_for_every_category(self) -> None:
        clean = GateVerdict.clean()
        for source in sd.SOURCE_CATEGORIES:
            assert sd.may_go_to_cloud(_sample(source, "a quiet sentence"), clean)[0], source

    def test_assert_sendable_raises_with_the_reason(self) -> None:
        sample = _sample("production-negative", BLOCKING_TEXT)
        with pytest.raises(sd.CloudSendRefusedError, match="not sendable"):
            sd.assert_sendable(sample, _blocked_verdict())


class TestBlockedContentNeverTrains:
    """The docstring-specified pins: the canary test and the AST walk."""

    def test_canary_never_reaches_the_dataset(self, tmp_path: Path) -> None:
        log = _write_log(
            tmp_path / "decisions.jsonl",
            [
                _record("b-1", excerpt=CANARY, blocked=True),
                _record("c-1", excerpt="a perfectly ordinary question about the weather"),
                _record("c-2", excerpt="how do I centre a div"),
            ],
        )
        # Default: the blocked record is discarded without its content being read.
        intake = sd.production_negative_samples(log)
        assert intake.blocked_discarded == 1
        assert all(CANARY not in (r.excerpt or "") for r in intake.records)

        # Explicit metadata-only projection: the record contributes, its text does not.
        result = asyncio.run(
            sd.synthesize_sensitivity_dataset(
                out_path=tmp_path / "ds",
                production_log=log,
                include_blocked_metadata=True,
                mode="features",
                per_kind=1,
                contextual_per_category=0,
                seed=42,
                exported_at="2026-01-01T00:00:00+00:00",
            )
        )
        rows_json = "\n".join(row.to_json() for row in result.dataset.rows)
        on_disk = result.rows_path.read_text(encoding="utf-8")
        stats_on_disk = result.stats_path.read_text(encoding="utf-8")
        audit_json = json.dumps([entry.to_dict() for entry in result.audit.entries])
        for where, blob in (
            ("rows", rows_json),
            ("rows file", on_disk),
            ("stats sidecar", stats_on_disk),
            ("audit", audit_json),
        ):
            assert CANARY not in blob, f"the canary leaked into the {where}"
        # features mode means no row has text at all.
        assert all(row.text is None for row in result.dataset.rows)
        # and the projection row itself is the one with the blocked gate.
        blocked_rows = [row for row in result.dataset.rows if row.gate.get("blocks_backend")]
        assert blocked_rows, "the metadata-only projection row is missing"
        assert all(r.text is None for r in blocked_rows)

    def test_blocked_record_in_text_mode_still_cannot_carry_text(self, tmp_path: Path) -> None:
        log = _write_log(tmp_path / "decisions.jsonl", [_record("b-1", excerpt=CANARY, blocked=True)])
        intake = sd.production_negative_samples(log, include_blocked_metadata=True)
        assert intake.blocked_metadata_only == 1
        assert intake.records[0].excerpt is None
        assert intake.records[0].questions_sent == {}
        result = sd.build_sensitivity_dataset(
            records=intake.records, mode="text", holdout_fraction=0.2, exported_at="2026-01-01T00:00:00+00:00"
        )
        assert all(row.text is None for row in result.dataset.rows)
        assert CANARY not in "\n".join(row.to_json() for row in result.dataset.rows)

    def test_no_code_path_reads_excerpt(self) -> None:
        # The docstring's second pin: walk the module's AST. A reader of
        # ``.excerpt`` or ``["excerpt"]`` would be the leak, whatever it is
        # wrapped in. The only sanctioned use of the name is the erasure
        # (``excerpt=None``) in the metadata-only projection.
        source_path = Path(sd.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "excerpt":
                pytest.fail(f"{source_path.name}:{node.lineno} reads an .excerpt attribute")
            if isinstance(node, ast.Subscript):
                sl = node.slice
                if isinstance(sl, ast.Constant) and sl.value == "excerpt":
                    pytest.fail(f"{source_path.name}:{node.lineno} subscripts ['excerpt']")
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "excerpt":
                        assert isinstance(kw.value, ast.Constant) and kw.value.value is None, (
                            f"{source_path.name}:{node.lineno} passes a non-None excerpt="
                        )


class TestProductionIntake:
    def test_degraded_records_are_dropped_and_counted(self, tmp_path: Path) -> None:
        log = _write_log(
            tmp_path / "decisions.jsonl",
            [
                _record("ok-1", excerpt="fine"),
                _record("deg-1", excerpt="uniform distribution from a dead backend", degraded=True),
            ],
        )
        intake = sd.production_negative_samples(log)
        assert intake.degraded_skipped == 1
        assert [r.request_id for r in intake.records] == ["ok-1"]

    def test_limit_stops_the_reads(self, tmp_path: Path) -> None:
        log = _write_log(
            tmp_path / "decisions.jsonl", [_record(f"r-{i}", excerpt=f"row {i}") for i in range(10)]
        )
        intake = sd.production_negative_samples(log, limit=3)
        assert len(intake.records) == 3
        # Four reads, three kept: the fourth read is what makes the limit
        # reachable, and the note says so instead of the count lying about it.
        assert intake.read == 4
        assert any("stopped at" in note and "--limit 3" in note for note in intake.notes)

    def test_unsafe_caller_metadata_is_sanitized(self, tmp_path: Path) -> None:
        # The contract is key-based: prompt-bearing keys are the leak path, so
        # they are the ones stripped. A credential-looking value passes through
        # untouched -- structural filtering is the caller's job, by design.
        log = _write_log(
            tmp_path / "decisions.jsonl",
            [_record(
                "m-1",
                excerpt="fine",
                metadata={
                    "tenant": "acme",
                    "prompt": "the raw user prompt",
                    "messages": [{"role": "user", "content": "x"}],
                },
            )],
        )
        intake = sd.production_negative_samples(log)
        assert "prompt" not in intake.records[0].metadata
        assert "messages" not in intake.records[0].metadata
        assert intake.records[0].metadata.get("tenant") == "acme"


class TestLabelingRunsOffline:
    def test_mock_backend_returns_full_distributions(self) -> None:
        samples = [
            _sample("synthetic-pii", "the ssn is 666-45-6789", fakeness_basis="reserved-range"),
            _sample("contextual-template", "the performance review cites the incident from march"),
        ]
        result = asyncio.run(sd.label_sensitivity(samples))
        assert len(result.labeled) == 2
        for item in result.labeled:
            answers = item.answers
            assert set(answers.sensitivity.probabilities) == set(SENSITIVITY_LEVELS)
            assert set(answers.complexity.probabilities) == set(COMPLEXITY_LEVELS)
            assert set(answers.domain.probabilities) == set(DOMAINS)
            for head in (answers.sensitivity, answers.complexity, answers.domain):
                total = sum(float(p) for p in head.probabilities.values())
                # 1e-4, not 1e-6: the backend rounds each probability to six
                # decimal places, so a four-value head can legally sum 2e-6 off.
                assert abs(total - 1.0) < 1e-4, head.choice
            assert 0.0 <= answers.pii.value <= 1.0
            assert item.backend == "mock"
            assert item.model_version

    def test_a_blocked_production_sample_is_refused_and_never_sent(self) -> None:
        sample = _sample("production-negative", BLOCKING_TEXT)
        sink_records: list = []
        result = asyncio.run(sd.label_sensitivity([sample], audit_sink=CallbackSink(sink_records.append)))
        assert result.n_refused == 1
        assert result.refused[0] is sample
        entry = result.audit.entries[0]
        assert entry.sent_to_backend is False
        assert entry.refusal_reason == sd.REFUSAL_GATE_BLOCKED_PRODUCTION
        # The refusal was still audited -- and through the same sink a live run uses.
        assert sink_records and entry.sample_id in str(sink_records)
        assert BLOCKING_TEXT not in str([e.to_dict() for e in result.audit.entries])

    def test_labeling_is_deterministic_offline(self) -> None:
        samples = [_sample("synthetic-pii", "card 4242 4242 4242 4242 on file")]
        first = asyncio.run(sd.label_sensitivity(samples))
        second = asyncio.run(sd.label_sensitivity(samples))
        assert [a.answers.to_dict() for a in first.labeled] == [a.answers.to_dict() for a in second.labeled]

    def test_empty_input_is_an_empty_result(self) -> None:
        result = asyncio.run(sd.label_sensitivity([]))
        assert result.n_sent == 0 and result.n_refused == 0

    def test_max_concurrency_below_one_is_refused(self) -> None:
        with pytest.raises(sd.SensitivityDataError, match="max_concurrency"):
            asyncio.run(sd.label_sensitivity([_sample("contextual-template", "t")], max_concurrency=0))


class TestDatasetBuild:
    def test_text_mode_keeps_text_and_features_mode_drops_it(self) -> None:
        samples = [_sample("contextual-template", "the hr case file remains sealed")]
        labeled = asyncio.run(sd.label_sensitivity(samples)).labeled
        exported_at = "2026-01-01T00:00:00+00:00"
        text_result = sd.build_sensitivity_dataset(labeled=labeled, mode="text", exported_at=exported_at)
        assert all(row.text is not None for row in text_result.dataset.rows)
        features_result = sd.build_sensitivity_dataset(labeled=labeled, mode="features", exported_at=exported_at)
        assert all(row.text is None for row in features_result.dataset.rows)
        # The feature vector survives the projection: the trainer still gets the shape.
        assert features_result.dataset.rows[0].features.char_len == len("the hr case file remains sealed")

    def test_something_must_be_passed(self) -> None:
        with pytest.raises(sd.SensitivityDataError, match="nothing to build"):
            sd.build_sensitivity_dataset(mode="text")
        with pytest.raises(sd.SensitivityDataError, match="mode"):
            sd.build_sensitivity_dataset(records=[], labeled=(), mode="yolo")
        with pytest.raises(sd.SensitivityDataError, match="holdout_fraction"):
            sd.build_sensitivity_dataset(
                labeled=[_sample("contextual-template", "x")], mode="text", holdout_fraction=0.95
            )

    def test_synthesize_is_reproducible(self, tmp_path: Path) -> None:
        common = {"per_kind": 2, "contextual_per_category": 1, "seed": 7, "exported_at": "2026-01-01T00:00:00+00:00"}
        first = asyncio.run(sd.synthesize_sensitivity_dataset(out_path=tmp_path / "a", **common))
        second = asyncio.run(sd.synthesize_sensitivity_dataset(out_path=tmp_path / "b", **common))
        assert first.rows_path.read_bytes() == second.rows_path.read_bytes()
        assert first.stats.dataset_sha256 == second.stats.dataset_sha256

    def test_synthesize_refuses_blocked_metadata_in_text_mode(self) -> None:
        with pytest.raises(sd.SensitivityDataError, match="mode='features'"):
            asyncio.run(
                sd.synthesize_sensitivity_dataset(
                    per_kind=1, contextual_per_category=0, include_blocked_metadata=True, mode="text"
                )
            )

    def test_the_published_aws_key_is_deduplicated(self, tmp_path: Path) -> None:
        # per_kind>1 on a constant-value generator used to emit byte-identical
        # rows; the split is keyed on request_id, so identical text could land on
        # both sides of it. Dedup at generation is what keeps the holdout honest.
        result = asyncio.run(
            sd.synthesize_sensitivity_dataset(
                out_path=tmp_path / "ds",
                per_kind=4,
                kinds=("aws_access_key_id",),
                contextual_per_category=0,
                seed=3,
                exported_at="2026-01-01T00:00:00+00:00",
            )
        )
        aws_rows = [r for r in result.dataset.rows if r.provenance.get("generator") == "aws_access_key_id"]
        assert len(aws_rows) == 1
        texts = [r.text for r in result.dataset.rows if r.text]
        assert len(texts) == len(set(texts)), "duplicate rows: the holdout would double-count an example"

    def test_stats_sidecar_round_trips(self, tmp_path: Path) -> None:
        result = asyncio.run(
            sd.synthesize_sensitivity_dataset(
                out_path=tmp_path / "ds",
                per_kind=1,
                contextual_per_category=0,
                seed=5,
                exported_at="2026-01-01T00:00:00+00:00",
            )
        )
        reloaded = load_dataset(result.rows_path)
        assert len(reloaded.rows) == len(result.dataset.rows)
        assert reloaded.usable_counts(), "no head had usable rows; the teacher produced nothing"

    def test_example_rows_mask_the_secrets(self, tmp_path: Path) -> None:
        result = asyncio.run(
            sd.synthesize_sensitivity_dataset(
                out_path=tmp_path / "ds",
                per_kind=1,
                kinds=("payment_card",),
                contextual_per_category=0,
                seed=9,
                exported_at="2026-01-01T00:00:00+00:00",
            )
        )
        view = sd.example_rows(result, n=3)
        assert "[masked-" in view, "the example view should show masked spans, not raw PANs"
        from jev_route.distill.synthetic_pii import PUBLISHED_TEST_CARD_NUMBERS

        for number in PUBLISHED_TEST_CARD_NUMBERS:
            assert number not in view


class TestLazyImports:
    def test_import_pulls_no_heavy_stack(self) -> None:
        # The router container has no numpy and no torch; a module-scope import
        # here would crash it. Run in a subprocess so the parent process's own
        # sys.modules cannot mask a regression.
        code = (
            "import sys, jev_route.distill.sensitivity_data"
            " ; assert 'numpy' not in sys.modules, 'numpy pulled in at import'"
            " ; assert 'torch' not in sys.modules, 'torch pulled in at import'"
            " ; print('ok')"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
        assert proc.returncode == 0, proc.stderr
        assert "ok" in proc.stdout
