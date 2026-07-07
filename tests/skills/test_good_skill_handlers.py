# -*- coding: utf-8 -*-
"""
验证好人侧技能 handler 拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/skills/test_good_skill_handlers.py -q
"""

from __future__ import annotations


def test_good_skill_handlers_remain_compatibly_importable() -> None:
    from werewolf_agent.skills import good_skill_handlers
    from werewolf_agent.skills import werewolf_skills

    assert werewolf_skills.bold_claim_handler is good_skill_handlers.bold_claim_handler
    assert werewolf_skills.counter_claim_handler is good_skill_handlers.counter_claim_handler
    assert werewolf_skills.push_vote_handler is good_skill_handlers.push_vote_handler
    assert werewolf_skills.swing_vote_handler is good_skill_handlers.swing_vote_handler
    assert werewolf_skills.find_power_handler is good_skill_handlers.find_power_handler
    assert werewolf_skills.protect_power_handler is good_skill_handlers.protect_power_handler
    assert werewolf_skills.resist_push_handler is good_skill_handlers.resist_push_handler
    assert werewolf_skills.last_words_handler is good_skill_handlers.last_words_handler


def test_good_skill_handler_groups_are_importable() -> None:
    from werewolf_agent.skills import good_claim_handlers
    from werewolf_agent.skills import good_power_handlers
    from werewolf_agent.skills import good_skill_handlers
    from werewolf_agent.skills import good_vote_handlers

    assert good_claim_handlers.bold_claim_handler is good_skill_handlers.bold_claim_handler
    assert good_claim_handlers.counter_claim_handler is good_skill_handlers.counter_claim_handler
    assert good_vote_handlers.push_vote_handler is good_skill_handlers.push_vote_handler
    assert good_vote_handlers.swing_vote_handler is good_skill_handlers.swing_vote_handler
    assert good_vote_handlers.resist_push_handler is good_skill_handlers.resist_push_handler
    assert good_power_handlers.find_power_handler is good_skill_handlers.find_power_handler
    assert good_power_handlers.protect_power_handler is good_skill_handlers.protect_power_handler
