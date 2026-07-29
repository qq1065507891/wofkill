# -*- coding: utf-8 -*-
"""
自主玩家智能体的严格、版本化且无副作用的公共契约。

作者: Project contributors
创建日期: 2026-07-29
"""

from werewolf_agent.player_agents.contracts.errors import (
    ProposalFailure,
    ValidationErrorCode,
)
from werewolf_agent.player_agents.contracts.revisions import (
    ReadReference,
    RevisionContext,
)
from werewolf_agent.player_agents.contracts.turns import (
    AgentTurn,
    AgentTurnStatus,
    ConflictClass,
    LegalActionWindow,
    TurnBudget,
    transition_turn,
)

__all__ = [
    "AgentTurn",
    "AgentTurnStatus",
    "ConflictClass",
    "LegalActionWindow",
    "ProposalFailure",
    "ReadReference",
    "RevisionContext",
    "TurnBudget",
    "ValidationErrorCode",
    "transition_turn",
]
