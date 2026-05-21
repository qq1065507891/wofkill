"""Tests for ruleset capability registration and playability decisions."""

from __future__ import annotations

import pytest

from werewolf_agent.runtime.game_runner import GameRunner, GameRunnerConfig


def test_default_ruleset_is_playable() -> None:
    from werewolf_agent.customization.ruleset_registry import RulesetRegistry

    registry = RulesetRegistry()

    entry = registry.get("pre_witch_hunter_idiot_mixed")

    assert entry.status == "playable"
    assert "werewolf" in entry.capabilities.supported_roles
    assert "hybrid" in entry.capabilities.supported_roles
    assert entry.compatibility.status == "playable"


def test_unimplemented_roles_are_display_only() -> None:
    from werewolf_agent.customization.ruleset_registry import RulesetRegistry

    registry = RulesetRegistry()

    entry = registry.from_normalized(
        {
            "ruleset_id": "wolf_king_guard_demo",
            "roles": {"wolf_king": {"count": 1}, "guard": {"count": 1}},
            "player_count": 2,
        }
    )

    assert entry.status == "display_only"
    assert "wolf_king" in entry.unsupported_roles
    assert "guard" in entry.unsupported_roles


def test_game_runner_rejects_display_only_ruleset() -> None:
    from werewolf_agent.customization.ruleset_registry import RulesetRegistry, RulesetRegistryEntry
    from werewolf_agent.customization.compatibility import CompatibilityMatrix, RuleEngineCapabilities

    registry = RulesetRegistry()
    caps = RuleEngineCapabilities()
    entry = RulesetRegistryEntry(
        ruleset_id="mock_display_only",
        status="display_only",
        capabilities=caps,
        compatibility=CompatibilityMatrix(status="display_only"),
        path=None,
    )
    registry._entries["mock_display_only"] = entry

    with pytest.raises(ValueError, match="is display_only"):
        GameRunner(GameRunnerConfig(ruleset_id="mock_display_only", ruleset_registry=registry))
