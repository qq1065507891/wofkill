# -*- coding: utf-8 -*-
"""
验证技能 handler registry 拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/skills/test_skill_handler_registry.py -q
"""

from __future__ import annotations

import subprocess
import sys


def test_skill_handler_registry_remains_compatibly_importable() -> None:
    from werewolf_agent.skills import skill_handler_registry
    from werewolf_agent.skills import werewolf_skills

    assert werewolf_skills.register_handler is skill_handler_registry.register_handler
    assert werewolf_skills.get_handler is skill_handler_registry.get_handler
    assert werewolf_skills._SKILL_HANDLERS is skill_handler_registry.SKILL_HANDLERS


def test_skill_handler_registry_direct_import_populates_default_handlers() -> None:
    code = (
        "from werewolf_agent.skills.schemas import SkillName\n"
        "from werewolf_agent.skills.skill_handler_registry import get_handler, SKILL_HANDLERS\n"
        "handler = get_handler(SkillName.PUSH_VOTE)\n"
        "assert handler is not None\n"
        "assert len(SKILL_HANDLERS) == 12\n"
    )

    subprocess.run([sys.executable, "-c", code], check=True)
