# -*- coding: utf-8 -*-
"""
RuleEngine 的遗言资格判定 helper。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.engine.rule_last_words import can_leave_last_words
"""

from __future__ import annotations

from typing import Any


def can_leave_last_words(
    raw: dict[str, Any],
    *,
    death_reason: str,
    timing: str,
    night_number: int,
) -> bool:
    lw = raw["last_words"]
    # 先按 death_reason 判定，避免夜间猎人开枪目标被误判为普通夜死。
    if death_reason == "exile" and timing == "day_vote":
        return lw["day_exile"]
    if death_reason == "hunter_shot":
        return lw["hunter_shot_target"]
    if death_reason == "self_destruct":
        return lw["self_destruct"]
    if timing == "night":
        if night_number == 1:
            return lw["first_night_death"]
        return lw["later_night_death"]
    return False


__all__ = ["can_leave_last_words"]
