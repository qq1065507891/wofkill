# -*- coding: utf-8 -*-
"""
运行时日间行动 agent 适配器的兼容 facade。

作者: Project contributors
创建日期: 2026-07-08

使用示例:
    >>> from werewolf_agent.runtime.agent_day_actions import agent_day_speech
    >>> agent_day_speech(...)
"""

from __future__ import annotations

from werewolf_agent.runtime.agent_day_speech_actions import (
    agent_day_speech,
    agent_defense_speech,
    agent_exile_last_words,
    agent_pk_speech,
)
from werewolf_agent.runtime.agent_day_vote_actions import agent_day_vote

__all__ = [
    "agent_defense_speech",
    "agent_day_speech",
    "agent_pk_speech",
    "agent_day_vote",
    "agent_exile_last_words",
]