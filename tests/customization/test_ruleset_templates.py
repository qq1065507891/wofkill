"""Tests for ruleset upload templates and validation."""

from __future__ import annotations

from pathlib import Path

from werewolf_agent.customization.validators import validate_ruleset_yaml


def test_default_custom_ruleset_template_validates() -> None:
    text = Path("config/rulesets/templates/custom_ruleset_template.yaml").read_text(encoding="utf-8")

    result = validate_ruleset_yaml(text)

    assert result.valid is True
    assert result.summary["player_count"] == 12
    assert result.normalized["ruleset_id"]
    assert result.normalized["status"] == "playable"
    assert "wolf_timeout_default" not in result.normalized["constraints"]


def test_ruleset_rejects_obsolete_wolf_timeout_default_constraint() -> None:
    """Runtime 超时语义已删除，定制规则不得继续暴露旧字段。"""
    text = """
ruleset_id: obsolete_timeout_constraint
player_count: 1
roles:
  villager: {count: 1, faction: good}
constraints:
  wolf_timeout_default: no_kill
"""

    result = validate_ruleset_yaml(text)

    assert result.valid is False
    assert any(
        error.field == "constraints.wolf_timeout_default"
        and error.code == "unknown_constraint"
        for error in result.errors
    )


def test_ruleset_rejects_role_count_mismatch() -> None:
    text = """
ruleset_id: bad_count
name: Bad Count
version: 1
player_count: 12
roles:
  werewolf: {count: 4, faction: werewolf}
  villager: {count: 7, faction: good}
night_order: [werewolf]
victory:
  good: [eliminate_all_wolves]
  werewolf: [slaughter_villagers]
constraints: {}
"""

    result = validate_ruleset_yaml(text)

    assert result.valid is False
    assert any("player_count" in err.message for err in result.errors)


def test_ruleset_with_future_roles_is_display_only() -> None:
    text = """
ruleset_id: wolf_king_guard_demo
name: Wolf King Guard Demo
version: 1
player_count: 12
roles:
  wolf_king: {count: 1, faction: werewolf}
  guard: {count: 1, faction: good}
  werewolf: {count: 3, faction: werewolf}
  villager: {count: 7, faction: good}
abilities: [wolf_king_shot, guard_protect]
victory:
  good: [eliminate_all_wolves]
  werewolf: [slaughter_villagers]
constraints: {}
"""

    result = validate_ruleset_yaml(text)

    assert result.valid is True
    assert result.normalized["status"] == "display_only"
    assert "wolf_king" in result.normalized["unsupported_roles"]
    assert "guard_protect" in result.normalized["missing_abilities"]


def test_ruleset_rejects_unknown_fields() -> None:
    text = """
ruleset_id: bad_unknown
name: Bad Unknown
version: 1
player_count: 1
roles:
  villager: {count: 1, faction: good}
victory:
  good: [eliminate_all_wolves]
constraints: {}
script: rm -rf /
"""

    result = validate_ruleset_yaml(text)

    assert result.valid is False
    assert any("unknown field" in err.message for err in result.errors)


def test_ruleset_returns_diff_against_default() -> None:
    text = """
ruleset_id: changed_constraints
name: Changed Constraints
version: 1
player_count: 12
roles:
  werewolf: {count: 4, faction: werewolf}
  villager: {count: 3, faction: good}
  seer: {count: 1, faction: good}
  witch: {count: 1, faction: good}
  hunter: {count: 1, faction: good}
  idiot: {count: 1, faction: good}
  hybrid: {count: 1, faction: special_bound_to_master}
victory:
  good: [eliminate_all_wolves]
  werewolf: [slaughter_villagers]
constraints:
  witch_can_self_save: true
"""

    result = validate_ruleset_yaml(text)

    assert result.valid is True
    assert {
        "path": "constraints.witch_can_self_save",
        "default": False,
        "uploaded": True,
    } in result.diff_against_default
