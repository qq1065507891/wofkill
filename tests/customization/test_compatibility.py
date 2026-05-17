"""Tests for customization compatibility matrix helpers."""

from __future__ import annotations


def test_ruleset_compatibility_matrix_reports_missing_abilities() -> None:
    from werewolf_agent.customization.ruleset_registry import RulesetRegistry

    registry = RulesetRegistry()

    entry = registry.from_normalized(
        {
            "ruleset_id": "wolf_king_guard_demo",
            "roles": {"wolf_king": {"count": 1}, "guard": {"count": 1}},
            "abilities": ["wolf_king_shot", "guard_protect"],
            "player_count": 2,
        }
    )

    assert entry.compatibility.status == "display_only"
    assert "wolf_king_shot" in entry.compatibility.missing_abilities
    assert "guard_protect" in entry.compatibility.missing_abilities


def test_supported_ruleset_has_no_missing_role_or_ability() -> None:
    from werewolf_agent.customization.ruleset_registry import RulesetRegistry

    registry = RulesetRegistry()

    entry = registry.from_normalized(
        {
            "ruleset_id": "supported_custom_demo",
            "roles": {
                "werewolf": {"count": 4},
                "villager": {"count": 3},
                "seer": {"count": 1},
                "witch": {"count": 1},
                "hunter": {"count": 1},
                "idiot": {"count": 1},
                "hybrid": {"count": 1},
            },
            "abilities": ["wolf_kill", "witch_potion", "seer_check", "hunter_shot", "idiot_reveal", "hybrid_bind"],
            "player_count": 12,
        }
    )

    assert entry.compatibility.status == "playable"
    assert entry.compatibility.unsupported_roles == []
    assert entry.compatibility.missing_abilities == []
