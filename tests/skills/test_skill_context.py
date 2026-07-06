# -*- coding: utf-8 -*-
"""
验证技能上下文 helper 拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/skills/test_skill_context.py -q
"""

from __future__ import annotations


def test_skill_context_helpers_remain_compatibly_importable() -> None:
    from werewolf_agent.skills import skill_context
    from werewolf_agent.skills import werewolf_skills

    assert werewolf_skills._count_seer_claimants is skill_context._count_seer_claimants
    assert werewolf_skills._get_seer_claimants is skill_context._get_seer_claimants
    assert werewolf_skills._alive_wolves is skill_context._alive_wolves
    assert werewolf_skills._alive_non_wolves is skill_context._alive_non_wolves
    assert werewolf_skills._vote_targets_for_player is skill_context._vote_targets_for_player
    assert werewolf_skills._seer_checks_on_target is skill_context._seer_checks_on_target
    assert werewolf_skills._alerts_for_player is skill_context._alerts_for_player
    assert werewolf_skills._belief_top_suspects is skill_context._belief_top_suspects
    assert werewolf_skills._wolf_teammates_exposed is skill_context._wolf_teammates_exposed


def test_skill_context_helpers_handle_missing_context() -> None:
    from werewolf_agent.skills import skill_context

    assert skill_context._count_seer_claimants(None) == 0
    assert skill_context._get_seer_claimants(None) == []
    assert skill_context._alive_wolves(None) == []
    assert skill_context._alive_non_wolves(None) == []
    assert skill_context._vote_targets_for_player(None, "p01") == []
    assert skill_context._seer_checks_on_target(None, "p01") == []
    assert skill_context._alerts_for_player([], "") == []
    assert skill_context._belief_top_suspects(None) == []
    assert skill_context._wolf_teammates_exposed(None, ["p01"]) == []
