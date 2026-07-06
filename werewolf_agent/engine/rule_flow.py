# -*- coding: utf-8 -*-
"""
RuleEngine 的角色分配和昼夜流程 helper。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.engine.rule_flow import assign_roles
"""

from __future__ import annotations

import random
from typing import Any

from werewolf_agent.core.models import PlayerState


def role_count(raw: dict[str, Any], role: str) -> int:
    return int(raw["roles"][role]["count"])


def assign_roles(
    raw: dict[str, Any],
    player_ids: list[str],
    *,
    player_count: int,
    seed: int | None = None,
) -> dict[str, PlayerState]:
    if len(player_ids) != player_count:
        raise ValueError(
            f"Expected {player_count} players, got {len(player_ids)}"
        )
    role_list: list[str] = []
    for role, cfg in raw["roles"].items():
        role_list.extend([role] * int(cfg["count"]))
    rng = random.Random(seed)
    shuffled_roles = list(role_list)
    rng.shuffle(shuffled_roles)
    return {
        pid: PlayerState(id=pid, role=role)
        for pid, role in zip(player_ids, shuffled_roles)
    }


def night_order(raw: dict[str, Any]) -> list[str]:
    return [item["node"] for item in raw["night_flow"]["order"]]


def day_flow(raw: dict[str, Any], day_number: int) -> list[str]:
    if day_number != 1:
        return [
            node
            for node in raw["day_flow"]["standard_order"]
            if node != "first_day_sheriff_election"
        ]
    return list(raw["day_flow"]["standard_order"])


__all__ = ["assign_roles", "day_flow", "night_order", "role_count"]
