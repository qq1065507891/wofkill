# -*- coding: utf-8 -*-
"""
验证狼人侧技能 handler 拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/skills/test_wolf_skill_handlers.py -q
"""

from __future__ import annotations


def test_wolf_skill_handlers_remain_compatibly_importable() -> None:
    from werewolf_agent.skills import werewolf_skills
    from werewolf_agent.skills import wolf_skill_handlers

    assert werewolf_skills.deep_hook_handler is wolf_skill_handlers.deep_hook_handler
    assert werewolf_skills.hide_identity_handler is wolf_skill_handlers.hide_identity_handler
    assert werewolf_skills.wolf_pit_handler is wolf_skill_handlers.wolf_pit_handler
