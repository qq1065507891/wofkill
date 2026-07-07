# -*- coding: utf-8 -*-
"""
好人侧技能 handler 兼容 facade。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-07

使用示例:
    >>> from werewolf_agent.skills.good_skill_handlers import push_vote_handler
"""

from __future__ import annotations

from werewolf_agent.skills.good_claim_handlers import (
    bold_claim_handler,
    counter_claim_handler,
)
from werewolf_agent.skills.good_power_handlers import (
    find_power_handler,
    protect_power_handler,
)
from werewolf_agent.skills.good_vote_handlers import (
    last_words_handler,
    push_vote_handler,
    resist_push_handler,
    swing_vote_handler,
)

__all__ = [
    "bold_claim_handler",
    "counter_claim_handler",
    "find_power_handler",
    "last_words_handler",
    "protect_power_handler",
    "push_vote_handler",
    "resist_push_handler",
    "swing_vote_handler",
]
