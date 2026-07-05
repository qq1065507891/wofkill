# -*- coding: utf-8 -*-
"""
功能描述：**：上传或内置规则集的兼容性矩阵，仅反映当前 RuleEngine 实际能力。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


PLAYABLE_ROLES = frozenset(
    {
        "werewolf",
        "villager",
        "seer",
        "witch",
        "hunter",
        "idiot",
        "hybrid",
    }
)

PLAYABLE_ABILITIES = frozenset(
    {
        "wolf_kill",
        "wolf_no_kill",
        "witch_potion",
        "seer_check",
        "hunter_shot",
        "idiot_reveal",
        "hybrid_bind",
        "sheriff_badge",
        "last_words",
        "exile_vote",
        "self_destruct",
    }
)

PLAYABLE_VICTORY_CONDITIONS = frozenset(
    {
        "all_werewolves_out",
        "eliminate_all_wolves",
        "slaughter_one_side",
        "slaughter_villagers",
        "slaughter_gods",
        "hybrid_follows_master",
    }
)


@dataclass(frozen=True)
class RuleEngineCapabilities:
    """Current built-in capability surface exposed by RuleEngine."""

    supported_roles: tuple[str, ...] = tuple(sorted(PLAYABLE_ROLES))
    supported_abilities: tuple[str, ...] = tuple(sorted(PLAYABLE_ABILITIES))
    supported_victory_conditions: tuple[str, ...] = tuple(sorted(PLAYABLE_VICTORY_CONDITIONS))


@dataclass(frozen=True)
class CompatibilityMatrix:
    """Structured playability report for a ruleset template."""

    status: str
    supported_roles: list[str] = field(default_factory=list)
    unsupported_roles: list[str] = field(default_factory=list)
    supported_abilities: list[str] = field(default_factory=list)
    missing_abilities: list[str] = field(default_factory=list)
    supported_victory_conditions: list[str] = field(default_factory=list)
    unsupported_victory_conditions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def extract_role_ids(normalized: dict[str, Any]) -> list[str]:
    roles = normalized.get("roles", {})
    if isinstance(roles, dict):
        return [str(role) for role in roles]
    if isinstance(roles, list):
        role_ids: list[str] = []
        for item in roles:
            if isinstance(item, dict):
                role_ids.append(str(item.get("id") or item.get("role") or item.get("name")))
            else:
                role_ids.append(str(item))
        return [role for role in role_ids if role and role != "None"]
    return []


def extract_ability_ids(normalized: dict[str, Any]) -> list[str]:
    abilities = normalized.get("abilities", [])
    if isinstance(abilities, dict):
        return [str(ability) for ability in abilities]
    if isinstance(abilities, list):
        ability_ids: list[str] = []
        for item in abilities:
            if isinstance(item, dict):
                ability_ids.append(str(item.get("id") or item.get("ability_id") or item.get("name")))
            else:
                ability_ids.append(str(item))
        return [ability for ability in ability_ids if ability and ability != "None"]
    return []


def extract_victory_conditions(normalized: dict[str, Any]) -> list[str]:
    victory = normalized.get("victory", {})
    conditions: list[str] = []
    if isinstance(victory, dict):
        for value in victory.values():
            if isinstance(value, str):
                conditions.append(value)
            elif isinstance(value, list):
                conditions.extend(str(item) for item in value)
            elif isinstance(value, dict):
                condition = value.get("condition")
                if condition:
                    conditions.append(str(condition))
    elif isinstance(victory, list):
        conditions.extend(str(item) for item in victory)
    return conditions


def build_compatibility_matrix(
    normalized: dict[str, Any],
    capabilities: RuleEngineCapabilities | None = None,
) -> CompatibilityMatrix:
    """Compare a normalized ruleset against current RuleEngine support."""

    caps = capabilities or RuleEngineCapabilities()
    supported_roles_set = set(caps.supported_roles)
    supported_abilities_set = set(caps.supported_abilities)
    supported_victory_set = set(caps.supported_victory_conditions)

    roles = extract_role_ids(normalized)
    abilities = extract_ability_ids(normalized)
    victories = extract_victory_conditions(normalized)

    unsupported_roles = sorted(role for role in roles if role not in supported_roles_set)
    missing_abilities = sorted(ability for ability in abilities if ability not in supported_abilities_set)
    unsupported_victories = sorted(condition for condition in victories if condition not in supported_victory_set)
    warnings: list[str] = []
    if unsupported_roles:
        warnings.append("Some roles are not implemented by RuleEngine and are display-only.")
    if missing_abilities:
        warnings.append("Some abilities are not implemented by RuleEngine and are display-only.")
    if unsupported_victories:
        warnings.append("Some victory conditions are not implemented by RuleEngine.")

    status = "playable"
    if not roles:
        status = "display_only"
    if unsupported_roles or missing_abilities or unsupported_victories:
        status = "display_only"

    return CompatibilityMatrix(
        status=status,
        supported_roles=sorted(role for role in roles if role in supported_roles_set),
        unsupported_roles=unsupported_roles,
        supported_abilities=sorted(ability for ability in abilities if ability in supported_abilities_set),
        missing_abilities=missing_abilities,
        supported_victory_conditions=sorted(condition for condition in victories if condition in supported_victory_set),
        unsupported_victory_conditions=unsupported_victories,
        warnings=warnings,
    )
