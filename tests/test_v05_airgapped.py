"""v0.5 air-gapped package: the same router, pointed at a local server, with zero cloud credentials.

The air-gapped swap is config, not code: ``backend: {name: jev, airgapped: true,
api_url: http://<server>/v1}``. This file proves the swap end to end:

* a plain stdlib HTTP server that serves ``POST /v1/systemone`` is a complete
  decision backend -- the wire contract did not change;
* with no ``TYPESAFE_API_KEY`` anywhere, ``Router.from_policy_file`` routes
  against it, and the Authorization header carries the placeholder, not a
  cloud credential;
* a URL that already ends in the wire-contract path is used verbatim; any
  other base URL gets the path appended;
* a per-backend calibration profile (``backend.profile.confidence_adjust``)
  scales the three choice confidences only -- never the PII probability --
  clamps to [0, 1], never mutates the cached result, and demonstrably flips
  a confidence-floor escalation;
* the shadow window partitions per backend, and untagged (pre-v0.5) records
  read back under the window's configured label.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import yaml

from jev_route.backends import build_backend
from jev_route.backends.jev import DEFAULT_API_URL, JevBackend, systemone_endpoint
from jev_route.cache import InMemoryTTLCache, NullCache
from jev_route.logging_sink import NullSink
from jev_route.policy import Policy
from jev_route.router import Router
from jev_route.shadow_metrics import ShadowMetrics
from tests.conftest import CLEAN_PROSE, FakeBackend, make_policy_doc

LOCAL_HOST = "127.0.0.1"



# --------------------------------------------------------------------------- #
# Local System One server (stdlib only: this IS the air-gapped deployment)
# --------------------------------------------------------------------------- #
class _SystemoneHandler(BaseHTTPRequestHandler):
    """Records every POST verbatim and replies with a canned System One body."""

    def do_POST(self) -> None:
        server = self.server  # type: ignore[attr-defined]
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"__unparsable__": raw[:200].decode("utf-8", "replace")}
        server.requests.append(  # type: ignore[attr-defined]
            {"path": self.path, "headers": dict(self.headers), "json": payload}
        )
        body = json.dumps(server.response_body).encode("utf-8")  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        """Silent: test output is for assertions, not access logs."""


def local_systemone_body() -> dict[str, Any]:
    """A well-formed System One response from a local 4B-class model."""
    return {
        "model": "kev-4b-local",
        "answers": {
            # Sharp on purpose: a well-calibrated local model is confident on a
            # clearly routine task, and the derived certainty of these
            # distributions clears the default policy floors (0.7 / 0.8).
            "complexity": {
                "probabilities": {"trivial": 0.005, "standard": 0.98, "hard": 0.01, "frontier": 0.005},
                "choice": "standard",
            },
            "sensitivity": {
                "probabilities": {"public": 0.98, "internal": 0.01, "confidential": 0.005, "regulated": 0.005},
                "choice": "public",
            },
            "pii_present": {"noul": 0.03},
            "domain": {
                "probabilities": {
                    "code": 0.005,
                    "writing": 0.98,
                    "analysis": 0.005,
                    "chat": 0.005,
                    "data-extraction": 0.005,
                },
                "choice": "writing",
            },
        },
    }


@pytest.fixture
def systemone_server() -> Iterator[ThreadingHTTPServer]:
    """A real TCP server on 127.0.0.1, ephemeral port, stdlib only.

    Loopback is what conftest leaves open on purpose: serving a local HTTP stub
    is still offline.
    """
    server = ThreadingHTTPServer((LOCAL_HOST, 0), _SystemoneHandler)
    server.daemon_threads = True
    server.requests = []  # type: ignore[attr-defined]
    server.response_body = local_systemone_body()  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def server_base_url(server: ThreadingHTTPServer) -> str:
    return f"http://{LOCAL_HOST}:{server.server_address[1]}"


# --------------------------------------------------------------------------- #
# Base URL joining
# --------------------------------------------------------------------------- #
class TestSystemoneEndpoint:
    """The whole air-gapped swap is: keep the path, swap the host."""

    def test_a_url_already_ending_in_the_path_is_verbatim(self) -> None:
        url = f"http://{LOCAL_HOST}:8765/v1/systemone"
        assert systemone_endpoint(url) == url
        assert systemone_endpoint(url + "/") == f"http://{LOCAL_HOST}:8765/v1/systemone"

    def test_the_default_cloud_endpoint_is_verbatim(self) -> None:
        # The shipped default is the cloud endpoint itself; it must survive the
        # base-URL logic unchanged, or every existing deployment changes route.
        assert systemone_endpoint(DEFAULT_API_URL) == DEFAULT_API_URL

    def test_a_v1_base_gets_the_path_appended(self) -> None:
        assert systemone_endpoint(f"http://{LOCAL_HOST}:8765/v1") == f"http://{LOCAL_HOST}:8765/v1/systemone"
        assert systemone_endpoint(f"http://{LOCAL_HOST}:8765/v1/") == f"http://{LOCAL_HOST}:8765/v1/systemone"

    def test_a_server_root_gets_the_path_appended(self) -> None:
        assert systemone_endpoint(f"http://{LOCAL_HOST}:8765") == f"http://{LOCAL_HOST}:8765/systemone"

    def test_any_other_path_is_a_base_and_gets_the_path_appended(self) -> None:
        assert (
            systemone_endpoint(f"http://{LOCAL_HOST}:8765/proxy/v2")
            == f"http://{LOCAL_HOST}:8765/proxy/v2/systemone"
        )


# --------------------------------------------------------------------------- #
# End to end: Router.from_policy_file against the local server, zero credentials
# --------------------------------------------------------------------------- #
def _airgapped_policy_doc(log_path: Path, api_url: str) -> dict[str, Any]:
    return make_policy_doc(
        log_path=log_path,
        backend={
            "name": "jev",
            "airgapped": True,
            "api_url": api_url,
            "model": "kev-4b",
            "timeout_seconds": 5.0,
            "max_retries": 0,
        },
    )


class TestAirgappedEndToEnd:
    async def test_from_policy_file_routes_with_zero_credentials(
        self, systemone_server: ThreadingHTTPServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The acceptance path: fresh config, local server, no cloud key at all."""
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        assert os.environ.get("TYPESAFE_API_KEY") is None, "the test must start credential-free"

        doc = _airgapped_policy_doc(tmp_path / "decisions.jsonl", f"{server_base_url(systemone_server)}/v1")
        path = tmp_path / "policy.yaml"
        path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")

        router = Router.from_policy_file(path)
        decision = await router.route_text(CLEAN_PROSE)
        await router.aclose()

        # Routed, classified by the local model, not degraded.
        assert decision.tier == "cheap"
        assert decision.backend == "jev"
        assert decision.backend_model_version == "kev-4b-local"
        assert decision.degraded is False
        assert decision.escalated == ()
        assert decision.answers.complexity.choice == "standard"
        assert decision.answers.sensitivity.choice == "public"
        assert decision.answers.pii.value == pytest.approx(0.03)

        # Exactly one wire call, to the derived endpoint, with the placeholder key.
        assert len(systemone_server.requests) == 1  # type: ignore[attr-defined]
        request = systemone_server.requests[0]  # type: ignore[attr-defined]
        assert request["path"] == "/v1/systemone"
        assert request["headers"]["Authorization"] == "Bearer local"
        payload = request["json"]
        assert payload["model"] == "kev-4b"
        assert set(payload["questions"]) == {"complexity", "sensitivity", "pii_present", "domain"}
        assert CLEAN_PROSE in str(payload["state"].get("prompt_excerpt", ""))

    async def test_an_explicit_endpoint_url_is_used_verbatim(
        self, systemone_server: ThreadingHTTPServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """api_url may already be the full endpoint; it is not re-extended."""
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        endpoint = f"{server_base_url(systemone_server)}/v1/systemone"
        doc = _airgapped_policy_doc(tmp_path / "decisions.jsonl", endpoint)
        path = tmp_path / "policy.yaml"
        path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")

        router = Router.from_policy_file(path)
        decision = await router.route_text(CLEAN_PROSE)
        await router.aclose()

        assert decision.backend == "jev"
        request = systemone_server.requests[0]  # type: ignore[attr-defined]
        assert request["path"] == "/v1/systemone"
        assert request["headers"]["Authorization"] == "Bearer local"


# --------------------------------------------------------------------------- #
# Construction: airgapped relaxes the credential rule, cloud does not
# --------------------------------------------------------------------------- #
class TestConstruction:
    def test_airgapped_without_a_key_uses_the_placeholder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        backend = JevBackend(
            api_key=None,
            api_url=f"http://{LOCAL_HOST}:8765/v1",
            airgapped=True,
        )
        assert backend.airgapped is True
        assert backend.api_key == JevBackend.DEFAULT_LOCAL_KEY

    def test_airgapped_does_not_read_a_stray_environment_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A leftover cloud key on the machine is not sent to a local server."""
        monkeypatch.setenv("TYPESAFE_API_KEY", "stray-cloud-key")
        backend = JevBackend(api_key=None, api_url=f"http://{LOCAL_HOST}:8765/v1", airgapped=True)
        assert backend.api_key == JevBackend.DEFAULT_LOCAL_KEY
        assert "stray" not in backend.api_key

    def test_airgapped_honours_an_explicit_key_for_a_fronting_proxy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        backend = JevBackend(api_key="front-proxy-token", api_url=f"http://{LOCAL_HOST}:8765/v1", airgapped=True)
        assert backend.api_key == "front-proxy-token"

    def test_cloud_mode_still_refuses_to_start_without_a_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pinned behaviour: the loud failure names the env var and the offline path."""
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        with pytest.raises(ValueError) as excinfo:
            JevBackend()
        message = str(excinfo.value)
        assert "TYPESAFE_API_KEY" in message
        assert "MockBackend" in message


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
class TestFactory:
    def test_build_backend_airgapped_without_a_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        backend = build_backend(
            {
                "name": "jev",
                "airgapped": True,
                "api_url": f"http://{LOCAL_HOST}:8765/v1",
                "model": "kev-4b",
            }
        )
        assert isinstance(backend, JevBackend)
        assert backend.airgapped is True
        assert backend.api_key == JevBackend.DEFAULT_LOCAL_KEY

    def test_build_backend_reads_the_flag_from_a_policy_object(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        doc = _airgapped_policy_doc(Path("./unused.jsonl"), f"http://{LOCAL_HOST}:8765/v1")
        policy = Policy.from_dict(doc)
        backend = build_backend(policy)
        assert isinstance(backend, JevBackend)
        assert backend.airgapped is True

    def test_build_backend_cloud_still_requires_a_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
            build_backend({"name": "jev"})


# --------------------------------------------------------------------------- #
# The calibration profile: backend.profile.confidence_adjust
# --------------------------------------------------------------------------- #
def _profile_router(
    factor: Any,
    backend: FakeBackend,
    cache: Any = None,
) -> Router:
    """A router whose policy carries a backend profile, or none when factor is None."""
    backend_cfg: dict[str, Any] = {"name": "mock"}
    if factor is not None:
        backend_cfg["profile"] = {"confidence_adjust": factor}
    doc = make_policy_doc(
        log_path="./unused-decision-log.jsonl",
        backend=backend_cfg,
        on_uncertain={"complexity_confidence_below": 0.7, "sensitivity_confidence_below": 0.3},
    )
    return Router(
        Policy.from_dict(doc),
        backend,
        cache=cache if cache is not None else NullCache(),
        sink=NullSink(),
    )


class TestConfidenceProfile:
    async def test_the_factor_scales_the_three_choice_confidences_only(self) -> None:
        fake = FakeBackend(complexity_confidence=0.9, sensitivity_confidence=0.9, domain_confidence=0.9, pii=0.02)
        router = _profile_router(0.5, fake)
        decision = await router.route_text(CLEAN_PROSE)
        await router.aclose()

        # Each choice confidence is multiplied ...
        assert decision.answers.complexity.confidence == pytest.approx(0.45)
        assert decision.answers.sensitivity.confidence == pytest.approx(0.45)
        assert decision.answers.domain.confidence == pytest.approx(0.45)
        # ... while the distributions stay exactly what the backend said.
        assert decision.answers.complexity.probabilities["standard"] == pytest.approx(0.9, abs=1e-4)
        # The PII noul is a probability, not a confidence: out of scope.
        assert decision.answers.pii.value == pytest.approx(0.02)

    async def test_the_factor_is_clamped_to_unit(self) -> None:
        fake = FakeBackend(complexity_confidence=0.9, sensitivity_confidence=0.9, domain_confidence=0.9)
        router = _profile_router(2.0, fake)
        decision = await router.route_text(CLEAN_PROSE)
        await router.aclose()

        assert decision.answers.complexity.confidence == 1.0
        assert decision.answers.sensitivity.confidence == 1.0
        assert decision.escalated == ()

    async def test_without_a_profile_confidences_are_consumed_as_is(self) -> None:
        fake = FakeBackend(complexity_confidence=0.9)
        router = _profile_router(None, fake)
        decision = await router.route_text(CLEAN_PROSE)
        await router.aclose()
        assert decision.answers.complexity.confidence == pytest.approx(0.9)

    async def test_the_adjusted_confidence_flips_an_escalation(self) -> None:
        """0.8 clears the 0.7 floor; 0.8 * 0.5 = 0.4 does not, so the route changes."""
        text = CLEAN_PROSE
        plain = _profile_router(None, FakeBackend(complexity_confidence=0.8, pii=0.02))
        before = await plain.route_text(text)
        await plain.aclose()
        assert before.escalated == ()
        assert before.effective_complexity == "standard"
        assert before.tier == "cheap"

        adjusted = _profile_router(0.5, FakeBackend(complexity_confidence=0.8, pii=0.02))
        after = await adjusted.route_text(text)
        await adjusted.aclose()
        assert len(after.escalated) == 1
        assert after.escalated[0].startswith("complexity standard->hard")
        assert after.effective_complexity == "hard"
        assert after.tier == "strong"

    async def test_cached_results_are_adjusted_but_never_mutated(self) -> None:
        """The cache stores the raw result; every read gets a new, corrected copy."""
        cache = InMemoryTTLCache(ttl_seconds=60.0, max_entries=8)

        class KeepingFake(FakeBackend):
            def __init__(self) -> None:
                super().__init__(complexity_confidence=0.9, pii=0.02)
                self.last_result = None

            async def decide(self, request: Any) -> Any:
                result = await super().decide(request)
                self.last_result = result
                return result

        fake = KeepingFake()
        router = _profile_router(0.5, fake, cache=cache)
        first = await router.route_text(CLEAN_PROSE)
        second = await router.route_text(CLEAN_PROSE)
        await router.aclose()

        assert first.cached is False
        assert second.cached is True
        assert first.answers.complexity.confidence == pytest.approx(0.45)
        assert second.answers.complexity.confidence == pytest.approx(0.45)
        # The object the backend produced -- and the one the cache holds -- is raw.
        assert fake.last_result.answers.complexity.confidence == pytest.approx(0.9)
        stored = next(iter(cache._data.values()))[1]
        assert stored is fake.last_result
        assert stored.answers.complexity.confidence == pytest.approx(0.9)

    @pytest.mark.parametrize(
        ("factor", "match"),
        [
            (True, "must be a number"),
            (False, "must be a number"),
            ("0.5", "must be a number"),
            (None, None),  # absent knob: no profile, no error (covered separately)
            (float("nan"), "must be finite"),
            (float("inf"), "must be finite"),
            (-0.5, "must be >= 0"),
        ],
        ids=["bool-true", "bool-false", "string", "none", "nan", "inf", "negative"],
    )
    def test_invalid_factors_are_refused_at_construction(self, factor: Any, match: Any) -> None:
        if factor is None:
            router = _profile_router(None, FakeBackend())
            assert router._confidence_adjust is None
            return
        with pytest.raises(ValueError, match=match):
            _profile_router(factor, FakeBackend())

    def test_a_non_mapping_profile_is_refused_at_construction(self) -> None:
        doc = make_policy_doc(
            log_path="./unused-decision-log.jsonl",
            backend={"name": "mock", "profile": ["confidence_adjust", 0.5]},
        )
        with pytest.raises(ValueError, match="must be a mapping"):
            Router(Policy.from_dict(doc), FakeBackend(), sink=NullSink())


# --------------------------------------------------------------------------- #
# Shadow window: per-backend partitioning, legacy rows included
# --------------------------------------------------------------------------- #
AGREE = (True, 0.9)   # (decision_fired, shadow_score) at threshold 0.5
DISAGREE = (True, 0.1)


def _ts(index: int) -> float:
    return 1_000_000.0 + index * 3600.0


class TestShadowPartitioning:
    def test_untagged_records_partition_under_the_configured_label(self, tmp_path: Path) -> None:
        metrics = ShadowMetrics(
            tmp_path / "window.jsonl", window_size=100, min_decisions=10, min_days=1, baseline_size=2
        )
        for i in range(3):  # legacy shape: no tag at all
            fired, score = AGREE
            metrics.record(decision_fired=fired, shadow_score=score, threshold=0.5, ts=_ts(i))
        for i in range(3, 5):  # tagged traffic from a second, air-gapped backend
            fired, score = DISAGREE
            metrics.record(decision_fired=fired, shadow_score=score, threshold=0.5, ts=_ts(i), backend_id="kev")

        by_backend = metrics.window_status_by_backend()
        assert set(by_backend) == {"jev", "kev"}
        assert by_backend["jev"].n_decisions == 3
        assert by_backend["kev"].n_decisions == 2
        assert by_backend["jev"].agreement_rate == pytest.approx(1.0)
        assert by_backend["kev"].agreement_rate == pytest.approx(0.0)

    def test_record_returns_the_pinned_shape_when_untagged(self, tmp_path: Path) -> None:
        metrics = ShadowMetrics(tmp_path / "window.jsonl")
        record = metrics.record(decision_fired=True, shadow_score=0.9, threshold=0.5, ts=_ts(0))
        # The pre-v0.5 on-disk shape: exactly the four original keys, nothing more.
        assert set(record) == {"decision_fired", "shadow_score", "threshold", "ts"}
        record = metrics.record(decision_fired=True, shadow_score=0.9, threshold=0.5, ts=_ts(1), backend_id="kev")
        assert set(record) == {"decision_fired", "shadow_score", "threshold", "ts", "backend"}
        assert record["backend"] == "kev"

    def test_a_legacy_file_reads_back_as_a_single_partition(self, tmp_path: Path) -> None:
        """A file written before the tag existed is one ``jev`` window, unchanged."""
        path = tmp_path / "window.jsonl"
        lines = []
        for i in range(4):
            fired, score = AGREE
            lines.append(
                json.dumps({"decision_fired": fired, "shadow_score": score, "threshold": 0.5, "ts": _ts(i)})
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        for line in path.read_text(encoding="utf-8").splitlines():
            assert set(json.loads(line)) == {"decision_fired", "shadow_score", "threshold", "ts"}

        metrics = ShadowMetrics(path)
        by_backend = metrics.window_status_by_backend()
        assert set(by_backend) == {"jev"}
        assert by_backend["jev"].n_decisions == 4
        # The whole-window arithmetic is exactly what it was before v0.5.
        assert metrics.window_status().n_decisions == 4
        assert metrics.window_status().agreement_rate == pytest.approx(1.0)

    def test_a_window_with_a_custom_label_reads_legacy_rows_under_it(self, tmp_path: Path) -> None:
        path = tmp_path / "window.jsonl"
        metrics = ShadowMetrics(path, backend_id="nimble")
        fired, score = AGREE
        metrics.record(decision_fired=fired, shadow_score=score, threshold=0.5, ts=_ts(0))
        reloaded = ShadowMetrics(path, backend_id="nimble")
        assert set(reloaded.window_status_by_backend()) == {"nimble"}
        assert reloaded.window_status_by_backend()["nimble"].n_decisions == 1

    def test_a_late_backend_is_graded_on_its_own_baseline(self, tmp_path: Path) -> None:
        """The whole window alarms on drift; each partition, on its own history, does not."""
        metrics = ShadowMetrics(
            tmp_path / "window.jsonl", window_size=100, min_decisions=100, min_days=30, baseline_size=2
        )
        for i in range(2):  # first backend: two agreeing records
            fired, score = AGREE
            metrics.record(decision_fired=fired, shadow_score=score, threshold=0.5, ts=_ts(i))
        for i in range(2, 4):  # the air-gapped backend joins, and disagrees on both records
            fired, score = DISAGREE
            metrics.record(decision_fired=fired, shadow_score=score, threshold=0.5, ts=_ts(i), backend_id="kev")

        whole = metrics.window_status()
        assert whole.baseline_disagreement == pytest.approx(0.0)
        assert whole.current_disagreement == pytest.approx(1.0)
        assert whole.drift_alarm is True

        by_backend = metrics.window_status_by_backend()
        # The kev partition's baseline is its OWN two records, not the window's first two.
        assert by_backend["kev"].baseline_disagreement == pytest.approx(1.0)
        assert by_backend["kev"].drift_alarm is False
        assert by_backend["jev"].drift_alarm is False
