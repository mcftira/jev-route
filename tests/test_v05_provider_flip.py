"""The cross-provider quota tandem: same model, someone else's quota."""

from __future__ import annotations

from typing import Any

from jev_route.backends.mock import MockBackend
from jev_route.cache import NullCache
from jev_route.logging_sink import NullSink
from jev_route.policy import Policy
from jev_route.router import Router
from tests.conftest import FakeBackend


class QuotaThenOkBackend(FakeBackend):
    """Degrades with a 429 on the primary; serves on the alternate."""

    def __init__(self, alt: MockBackend) -> None:
        super().__init__(degraded=True, degrade_reason="HTTP 429 quota exceeded")
        self._alt = alt

    def swap_provider(self, provider_cfg: dict) -> Any:
        return self._alt


def make_policy_doc() -> dict:
    from tests.conftest import make_policy_doc as _base

    doc = _base(log_path="/tmp/v05_flip_log.jsonl")
    doc["providers"] = {
        "typesafe": {"api_url": "https://api.typesafe.ai/v1", "api_key_env": "TYPESAFE_API_KEY"},
        "bai": {"api_url": "https://api.b.ai/v1", "api_key_env": "BAI_API_KEY"},
    }
    return doc


async def test_quota_flips_to_alternate_provider_same_model() -> None:
    sink = NullSink()
    alt = MockBackend()
    backend = QuotaThenOkBackend(alt)
    router = Router(Policy.from_dict(make_policy_doc()), backend, sink=sink, cache=NullCache())
    decision = await router.route_text("say hello")
    assert not decision.degraded
    assert any("provider quota flip" in e for e in decision.escalated)
    assert decision.backend.endswith(":bai")


async def test_no_providers_config_means_tandem_path_unchanged() -> None:
    sink = NullSink()
    backend = FakeBackend(degraded=True, degrade_reason="HTTP 429 quota exceeded")
    doc = make_policy_doc()
    del doc["providers"]
    router = Router(Policy.from_dict(doc), backend, sink=sink, cache=NullCache())
    decision = await router.route_text("say hello")
    assert decision.degraded or "quota tandem" in str(decision.escalated) or decision.tier == "cheap"
