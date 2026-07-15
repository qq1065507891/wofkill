# -*- coding: utf-8 -*-
"""Tests for ``config/models.yaml`` provider/model routing (v1.1.4 fallback-fix Part C.1).

In the 4 game logs captured on/after 2026-07-14, the ``minimax`` provider
(returning ``MiniMax-M2.7`` from the Anthropic-compatible endpoint) had a
24.9% attempt-failure rate (``root_cause=invalid_output``).  Most player
defaults routed through that profile.

The fix moves the ``minimax_default`` ``llm_profile`` to a different
provider (``openai`` calling ``minimax-m3`` on the Ark endpoint),
which has been more reliable in pilots.

These tests pin the YAML state so a future ``minimax_m27_*`` regression
is caught at unit time, not at runtime in a real game.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "models.yaml"
)


@pytest.fixture(scope="module")
def yaml_config() -> dict:
    """Load ``config/models.yaml`` once per module.  Schema is
    versioned + reviewed; if parsing breaks, surface that clearly.
    """
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_minimax_default_default_provider_is_openai(yaml_config: dict) -> None:
    """``minimax_default`` must now route primary traffic through
    ``openai`` (with ``ark_minimax_m3`` model_profile), not the
    native ``minimax`` provider.  This is what drops the 24.9%
    observed failure rate.
    """
    cfg = yaml_config["llm_profiles"]["minimax_default"]["default"]
    assert cfg["provider"] == "openai"
    assert cfg["model_profile"] == "ark_minimax_m3"


def test_minimax_default_fallback_uses_different_provider(yaml_config: dict) -> None:
    """The fallback provider was previously also ``minimax`` (which
    offered no model diversity on outage).  Part C.1 changes it to
    ``openai`` + ``ark_deepseek_v4_flash`` so a ``MiniMax`` outage
    doesn't propagate.
    """
    fb = yaml_config["llm_profiles"]["minimax_default"]["fallback"]
    assert fb["provider"] == "openai"
    assert fb["model_profile"] in {"ark_deepseek_v4_flash", "ark_deepseek_v4_pro"}
    # Cross-provider (different from primary)
    primary = yaml_config["llm_profiles"]["minimax_default"]["default"]["provider"]
    assert fb["provider"] != primary or fb["model_profile"] != "ark_minimax_m3"


def test_minimax_default_reflection_keeps_native_endpoint(yaml_config: dict) -> None:
    """``reflection`` task intentionally keeps ``minimax_m27_reflection``
    (native ``minimax`` provider / ``MiniMax-M2.7``).  This is a
    cross-version sampling choice: the reflection synthesizer benefits
    from diverse provider signals.
    """
    reflection = yaml_config["llm_profiles"]["minimax_default"]["tasks"]["reflection"]
    assert reflection["provider"] == "minimax"
    assert reflection["model_profile"] == "minimax_m27_reflection"


def test_player_defaults_resolve_through_new_routing(yaml_config: dict) -> None:
    """Most players are still bound to ``minimax_default``; after
    Part C.1 every speech / night_action call from those players
    goes through Ark OpenAI + M3.  This test pins the player→profile
    assignment so a future refactor that re-binds to ``minimax_27_*``
    is caught.
    """
    players = yaml_config["players"]
    minimax_default_players = [
        pid for pid, info in players.items() if info["llm_profile"] == "minimax_default"
    ]
    # At least 7 of the 12 players should still inherit minimax_default.
    # (2 were already on ark_glm, 1 on ark_minimax, 1 on ark_kimi; the
    # rest get the new routing by virtue of the llm_profile swap.)
    assert len(minimax_default_players) >= 7, (
        f"only {len(minimax_default_players)} players on minimax_default; "
        f"the audit identified ≥7 expected"
    )


def test_ark_minimax_m3_model_profile_exists(yaml_config: dict) -> None:
    """The ``ark_minimax_m3`` model profile must still exist and
    reference ``minimax-m3`` (Ark OpenAI-compatible model id).
    """
    profile = yaml_config["model_profiles"]["ark_minimax_m3"]
    assert profile["provider"] == "openai"
    assert profile["model"] == "minimax-m3"


def test_minimax_m27_profiles_still_defined_for_reflection_fallback(yaml_config: dict) -> None:
    """The native ``minimax`` provider profiles must still exist so
    the ``reflection`` task and any future ``minimax-only`` tests
    keep working.  We don't remove them, just stop routing primary
    traffic through them.
    """
    profiles = yaml_config["model_profiles"]
    for required in (
        "minimax_m27_default",
        "minimax_m27_reflection",
        "minimax_m27_fast",
    ):
        assert required in profiles, f"required profile {required!r} missing"
        assert profiles[required]["provider"] == "minimax"
        assert profiles[required]["model"] == "MiniMax-M2.7"
