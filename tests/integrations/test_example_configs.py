"""The examples are part of the product, so they are tested like part of the product.

An example config that no longer loads is worse than no example: it is a
confident instruction that fails on a reader's machine, in a project whose whole
pitch is "run it". These tests load both proxy configs and the SDK script
without a network and without any API key, using LiteLLM's *own* resolution and
validation functions -- ``get_instance_fn`` for the dotted paths and
``ComplexityRouterConfig`` for the router block -- so they fail when LiteLLM
changes shape, not when our reading of it goes stale.
"""

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.types_utils.utils import get_instance_fn
from litellm.router_strategy.complexity_router.config import ComplexityRouterConfig
from litellm.types.router import ClassifierPlugin

from jev_route import Policy
from jev_route.integrations import _shared

#: These load real LiteLLM config paths, so they belong to the integration marker.
pytestmark = pytest.mark.integration


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "examples"
CLASSIFIER_CONFIG = EXAMPLES / "litellm-proxy-classifier" / "config.yaml"
HOOK_CONFIG = EXAMPLES / "litellm-proxy-hook" / "config.yaml"
SDK_EXAMPLE = EXAMPLES / "litellm-sdk-plugin.py"
POLICY_FILE = REPO_ROOT / "policies" / "default.yaml"

#: Anything that looks like a real credential literal. `Bearer $VAR` in a curl
#: example is fine; `Bearer sk-...` is not.
KEY_LIKE = re.compile(r"(?:sk-|LTAI|ghp_|xox[baprs]-)[A-Za-z0-9_\-]{8,}")


def load_config(path: Path) -> dict[str, Any]:
    """Parse an example config, failing loudly when it is missing rather than skipping."""
    assert path.is_file(), f"missing example: {path}"
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict), f"{path.name} is not a mapping at the top level"
    return loaded


def model_names(config: dict[str, Any]) -> set[str]:
    """Every ``model_name`` in ``model_list`` (an operator-facing alias)."""
    return {str(entry["model_name"]) for entry in config["model_list"]}


def provider_models(config: dict[str, Any], group: str | None = None) -> set[str]:
    """Every ``litellm_params.model``, optionally only within one model group."""
    return {
        str(entry["litellm_params"]["model"])
        for entry in config["model_list"]
        if group is None or str(entry["model_name"]) == group
    }


def find_auto_router(config: dict[str, Any]) -> dict[str, Any]:
    """The ``auto_router/complexity_router`` deployment, or a clear failure."""
    for entry in config["model_list"]:
        params = entry["litellm_params"]
        if str(params.get("model", "")).startswith("auto_router/complexity_router"):
            return entry
    raise AssertionError("the classifier example declares no auto_router/complexity_router deployment")


def iter_api_keys(config: dict[str, Any]) -> list[str]:
    return [str(entry["litellm_params"].get("api_key", "")) for entry in config["model_list"]]


# --------------------------------------------------------------------------- #
# Both configs: hygiene that applies to everything in examples/
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", [CLASSIFIER_CONFIG, HOOK_CONFIG, SDK_EXAMPLE])
def test_examples_inline_no_secrets(path: Path) -> None:
    """A public repo: the only acceptable spellings of a key are references."""
    text = path.read_text(encoding="utf-8")
    assert not KEY_LIKE.search(text), f"{path.name} contains something that looks like a credential literal"
    # `Bearer` may only ever be followed by a shell/env reference in an example.
    for match in re.finditer(r"Bearer\s+(\S+)", text):
        assert match.group(1).startswith(("$", "{")), f"{path.name} shows a literal bearer token"


@pytest.mark.parametrize("path", [CLASSIFIER_CONFIG, HOOK_CONFIG])
def test_config_api_keys_are_references(path: Path) -> None:
    """Every api_key is either the llama.cpp placeholder or an env reference."""
    config = load_config(path)
    for entry in config["model_list"]:
        model = str(entry["litellm_params"].get("model", ""))
        if model.startswith("auto_router/"):
            # A strategy-router deployment forwards to other deployments and
            # carries no credential of its own.
            assert not entry["litellm_params"].get("api_key"), f"{path.name}: {model} should have no api_key"
            continue
        key = str(entry["litellm_params"].get("api_key", ""))
        assert key, f"{path.name}: {model} has no api_key"
        assert key == "EMPTY" or key.startswith("os.environ/"), f"{path.name}: api_key {key!r} is inlined"
    assert any(key.startswith("os.environ/ALIBABA_API_KEY") for key in iter_api_keys(config))


def test_examples_never_contain_the_environment_key_value() -> None:
    """Belt and braces: compare against the real value if this machine has one."""
    real = os.environ.get("ALIBABA_API_KEY") or os.environ.get("TYPESAFE_API_KEY")
    if not real or len(real) < 12:
        pytest.skip("no key in the environment to check against")
    for candidate in (CLASSIFIER_CONFIG, HOOK_CONFIG, SDK_EXAMPLE):
        assert real not in candidate.read_text(encoding="utf-8"), candidate.name


# --------------------------------------------------------------------------- #
# Mode 2: the native complexity router + ClassifierPlugin
# --------------------------------------------------------------------------- #
def test_classifier_config_resolves_its_plugin_through_litellm() -> None:
    """The proxy resolves this dotted path at startup and rejects a sync classify."""
    config = load_config(CLASSIFIER_CONFIG)
    path = find_auto_router(config)["litellm_params"]["complexity_router_config"]["classifier_plugin"]
    assert path == "jev_route.integrations.litellm_classifier.classifier"

    resolved = get_instance_fn(value=path, config_file_path=str(CLASSIFIER_CONFIG))

    assert isinstance(resolved, ClassifierPlugin)
    assert inspect_iscoroutine(resolved.classify)


def inspect_iscoroutine(fn: Any) -> bool:
    import inspect

    return inspect.iscoroutinefunction(fn)


def test_classifier_config_block_validates_against_litellm() -> None:
    """Load the operator's YAML through LiteLLM's own validator."""
    config = load_config(CLASSIFIER_CONFIG)
    params = find_auto_router(config)["litellm_params"]
    raw = dict(params["complexity_router_config"])
    raw["classifier_plugin"] = get_instance_fn(value=raw["classifier_plugin"], config_file_path=str(CLASSIFIER_CONFIG))

    validated = ComplexityRouterConfig(**raw)

    assert validated.classifier_type == "custom"
    assert tuple(validated.tier_names()) == ("local", "cheap", "strong")
    assert validated.fallback_tier == "local"
    assert params.get("complexity_router_default_model"), "litellm needs this spelling for its model graph"


def test_classifier_config_does_not_combine_the_two_fallback_knobs() -> None:
    """Documented LiteLLM behaviour, pinned so the example cannot regress into it.

    ``classifier_fallback: default_model`` and ``tier_definitions`` are mutually
    exclusive; with a custom tier set, ``fallback_tier`` is the knob. The example
    config's long comment quotes LiteLLM's own reasoning, and this test is what
    keeps that comment true.
    """
    config = load_config(CLASSIFIER_CONFIG)
    raw = dict(find_auto_router(config)["litellm_params"]["complexity_router_config"])
    assert "classifier_fallback" not in raw, "would be rejected at startup alongside tier_definitions"
    assert raw.get("tier_definitions")

    plugin = get_instance_fn(value=raw["classifier_plugin"], config_file_path=str(CLASSIFIER_CONFIG))
    with pytest.raises(Exception, match="cannot be combined with tier_definitions"):
        ComplexityRouterConfig(**{**raw, "classifier_plugin": plugin, "classifier_fallback": "default_model"})


def test_classifier_config_tiers_cover_every_deployment() -> None:
    """Every model a tier names must exist in model_list, or the tier is a 404."""
    config = load_config(CLASSIFIER_CONFIG)
    raw = find_auto_router(config)["litellm_params"]["complexity_router_config"]
    defined = {str(entry["name"]) for entry in raw["tier_definitions"]}
    names = model_names(config)

    assert set(raw["tiers"]) == defined, "tiers keys must match tier_definitions exactly"
    for tier, models in raw["tiers"].items():
        for model in models:
            assert model in names, f"tier {tier} names {model!r}, which is not in model_list"
    assert raw["fallback_tier"] in defined
    assert raw["default_model"] in names


def test_classifier_config_tier_names_match_the_policy_file() -> None:
    """The classifier returns a tier NAME, so both files must agree on the names.

    If they drift, every classification declines and ``fallback_tier`` decides --
    which is safe, silent, and exactly the kind of misconfiguration a test should
    catch instead of an operator.
    """
    config = load_config(CLASSIFIER_CONFIG)
    raw = find_auto_router(config)["litellm_params"]["complexity_router_config"]
    policy = Policy.from_file(POLICY_FILE)

    assert {str(entry["name"]) for entry in raw["tier_definitions"]} == set(policy.tiers)


# --------------------------------------------------------------------------- #
# Mode 3: the pre-call hook
# --------------------------------------------------------------------------- #
def test_hook_config_callback_resolves_to_a_live_custom_logger() -> None:
    config = load_config(HOOK_CONFIG)
    callbacks = config["litellm_settings"]["callbacks"]
    path = callbacks[0] if isinstance(callbacks, list) else callbacks
    assert path == "jev_route.integrations.litellm_hook.proxy_handler_instance"

    resolved = get_instance_fn(value=path, config_file_path=str(HOOK_CONFIG))

    assert isinstance(resolved, CustomLogger)
    # ProxyLogging only dispatches the hook when the leaf class defines it.
    assert "async_pre_call_hook" in vars(type(resolved))
    assert resolved.enforces_request_content is False


def test_hook_config_routed_alias_exists_and_fails_closed() -> None:
    """``auto`` must exist, and its own deployment should be the local one.

    When the hook declines, LiteLLM routes the model the caller asked for -- so
    the alias's deployment is the fallback destination, and pointing it at the
    self-hosted model is what makes the fail-open integration layer fail closed
    in practice.
    """
    config = load_config(HOOK_CONFIG)
    assert "auto" in model_names(config)
    assert provider_models(config, "auto") == {"openai/qwen3.8"}

    local_bases = {
        str(entry["litellm_params"].get("api_base", ""))
        for entry in config["model_list"]
        if entry["litellm_params"].get("model") == "openai/qwen3.8"
    }
    assert all(not base.startswith("https://") for base in local_bases), "the local tier must not be a cloud base"


def test_hook_config_exposes_every_policy_tier_as_a_model_name() -> None:
    """The hook rewrites ``data["model"]`` to a policy tier's model name."""
    config = load_config(HOOK_CONFIG)
    policy = Policy.from_file(POLICY_FILE)
    names = model_names(config)

    for tier, models in policy.tiers.items():
        for model in models:
            assert model in names, f"policy tier {tier} routes to {model!r}, which is not in model_list"


def test_hook_config_scopes_the_managed_models() -> None:
    """The example must tell the reader to opt in; "*" would rewrite pinned calls."""
    text = HOOK_CONFIG.read_text(encoding="utf-8")
    assert _shared.MANAGED_MODELS_ENV_VAR in text


# --------------------------------------------------------------------------- #
# Mode 1: the SDK script
# --------------------------------------------------------------------------- #
def load_sdk_module() -> Any:
    spec = importlib.util.spec_from_file_location("jev_route_sdk_example", SDK_EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sdk_example_is_internally_consistent() -> None:
    """Importing the script must be side-effect free, and its tables must agree."""
    module = load_sdk_module()

    group = {str(entry["litellm_params"]["model"]) for entry in module.MODEL_LIST if entry["model_name"] == "auto"}
    for tier, models in module.TIER_MODELS.items():
        for model in models:
            assert model in group, f"tier {tier} names {model!r}, which is not in the routed group"
    assert len(group) == len(module.TIER_MODELS), (
        "the routed group must hold every tier's deployment, or the plugin has nothing to narrow"
    )


def test_sdk_example_declares_its_stub() -> None:
    """The one stubbed boundary is named in the file, so nobody mistakes it for a live run."""
    text = SDK_EXAMPLE.read_text(encoding="utf-8")
    assert "install_stub_upstream" in text
    assert "monkeypatches ``litellm.acompletion``" in text
    assert "--live" in text


def test_shared_helpers_the_examples_rely_on_exist() -> None:
    for name in ("get_router", "build_router", "configure_router", "extract_messages", "extract_metadata"):
        assert callable(getattr(_shared, name)), name
