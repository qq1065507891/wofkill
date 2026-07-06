# -*- coding: utf-8 -*-
"""
验证复盘纠错技能 handler 拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/skills/test_review_skill_handlers.py -q
"""

from __future__ import annotations


def test_review_skill_handlers_remain_compatibly_importable() -> None:
    from werewolf_agent.skills import review_skill_handlers
    from werewolf_agent.skills import werewolf_skills

    assert (
        werewolf_skills.review_correction_handler
        is review_skill_handlers.review_correction_handler
    )
    assert werewolf_skills._review_correction_wolf is review_skill_handlers._review_correction_wolf
    assert werewolf_skills._review_correction_good is review_skill_handlers._review_correction_good
