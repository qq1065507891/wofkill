# -*- coding: utf-8 -*-
"""
Agent schema 公开兼容 facade。

作者: Project contributors
创建日期: 2025-01-15
修改日期: 2026-07-16

使用示例:
    >>> from werewolf_agent.agents.schemas import PlayerAction
"""

from __future__ import annotations

from werewolf_agent.agents.action_schemas import (
    ActionType,
    BadgeTearPlayerAction,
    BadgeTransferPlayerAction,
    CheckAlignmentPlayerAction,
    ChooseMasterPlayerAction,
    FallbackAction,
    HunterShotPlayerAction,
    JudgeBroadcast,
    NoOpPlayerAction,
    PLAYER_ACTION_VARIANTS,
    PlayerAction,
    SeerStance,
    SelfDestructPlayerAction,
    SheriffRegisterPlayerAction,
    SheriffVotePlayerAction,
    SheriffWithdrawPlayerAction,
    SpeechPlayerAction,
    UseAntidotePlayerAction,
    UsePoisonPlayerAction,
    VoteBasis,
    VotePlayerAction,
    WolfKillPlayerAction,
    WolfNoKillPlayerAction,
    WolfDiscussionSpeechPlayerAction,
    WolfTargetStance,
    WolfTargetStanceAction,
    WolfTeamPlan,
    _PLAYER_ACTION_ADAPTER,
)
from werewolf_agent.agents.prompt_schemas import (
    AgentContext,
    OutputMode,
    TaskType,
)
from werewolf_agent.agents.trace_schemas import (
    ActionTrace,
    FactionGoal,
    PrivateIntent,
    RetryInfo,
    RiskFlag,
)

__all__ = [
    "ActionTrace",
    "ActionType",
    "AgentContext",
    "BadgeTearPlayerAction",
    "BadgeTransferPlayerAction",
    "CheckAlignmentPlayerAction",
    "ChooseMasterPlayerAction",
    "FactionGoal",
    "FallbackAction",
    "HunterShotPlayerAction",
    "JudgeBroadcast",
    "NoOpPlayerAction",
    "OutputMode",
    "PLAYER_ACTION_VARIANTS",
    "PlayerAction",
    "PrivateIntent",
    "RetryInfo",
    "RiskFlag",
    "SeerStance",
    "SelfDestructPlayerAction",
    "SheriffRegisterPlayerAction",
    "SheriffVotePlayerAction",
    "SheriffWithdrawPlayerAction",
    "SpeechPlayerAction",
    "TaskType",
    "UseAntidotePlayerAction",
    "UsePoisonPlayerAction",
    "VoteBasis",
    "VotePlayerAction",
    "WolfKillPlayerAction",
    "WolfNoKillPlayerAction",
    "WolfDiscussionSpeechPlayerAction",
    "WolfTargetStance",
    "WolfTargetStanceAction",
    "WolfTeamPlan",
    "_PLAYER_ACTION_ADAPTER",
]
