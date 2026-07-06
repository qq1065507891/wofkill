# -*- coding: utf-8 -*-
"""
维护狼人杀技能 handler 的注册表。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.skills.skill_handler_registry import get_handler
"""

from __future__ import annotations

from typing import Callable

from werewolf_agent.skills.schemas import SkillName


SKILL_HANDLERS: dict[SkillName, Callable] = {}
_SKILL_HANDLERS = SKILL_HANDLERS
_DEFAULT_HANDLERS_IMPORTED = False


def register_handler(name: SkillName):
    """Decorator to register a skill handler in the dispatch table."""
    def decorator(fn: Callable) -> Callable:
        _SKILL_HANDLERS[name] = fn
        return fn
    return decorator


def ensure_default_handlers_registered() -> None:
    """导入默认 handler 模块，确保注册表对直接导入者可用。"""
    global _DEFAULT_HANDLERS_IMPORTED
    if _DEFAULT_HANDLERS_IMPORTED:
        return
    _DEFAULT_HANDLERS_IMPORTED = True
    # 导入模块会执行 @register_handler 装饰器，填充共享注册表。
    from werewolf_agent.skills import good_skill_handlers  # noqa: F401
    from werewolf_agent.skills import review_skill_handlers  # noqa: F401
    from werewolf_agent.skills import wolf_skill_handlers  # noqa: F401


def get_handler(name: SkillName) -> Callable | None:
    """Look up a registered skill handler by name."""
    ensure_default_handlers_registered()
    return _SKILL_HANDLERS.get(name)
