# -*- coding: utf-8 -*-
"""
功能描述：所有游戏状态的不可变数据容器，frozen dataclass + __post_init__ 防御性拷贝，零外部依赖。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：内部模块，无对外接口
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


_GOOD_ROLES = frozenset({"villager", "seer", "witch", "hunter", "idiot"})


def _derive_faction(role: str) -> str | None:
    """E3 (post-review-v2): 从 role 推导 faction。"""
    if role == "werewolf":
        return "werewolf"
    if role in _GOOD_ROLES:
        return "good"
    # hybrid 在未选 master 时 faction 是 None
    return None


@dataclass(frozen=True)
class PlayerState:
    id: str
    role: str
    alive: bool = True
    revealed_idiot: bool = False
    vote_enabled: bool = True
    badge_eligible: bool = True
    exile_immune: bool = False
    faction: str | None = None  # E3: 缺省从 role 推导（hybrid → None）

    def __post_init__(self) -> None:
        # E3 (post-review-v2): faction 缺省时按 role 推导
        if self.faction is None and self.role:
            derived = _derive_faction(self.role)
            if derived is not None:
                object.__setattr__(self, "faction", derived)


@dataclass(frozen=True)
class Death:
    player_id: str
    reason: str
    timing: str
    resolution_batch: str
    source_player_id: str | None = None
    can_leave_last_words: bool | None = None
    triggered_skills: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GameState:
    ruleset_id: str = "pre_witch_hunter_idiot_mixed"
    players: dict[str, PlayerState] = field(default_factory=dict)
    game_id: str = ""
    phase: str = "setup"
    day_number: int = 0
    night_number: int = 0
    hybrid_master_id: str | None = None
    hybrid_master_faction: str | None = None
    hybrid_result: str | None = None
    sheriff_id: str | None = None
    sheriff_badge_state: str = "none"
    sheriff_candidates: list[str] = field(default_factory=list)
    sheriff_interrupt_count: int = 0
    sheriff_tie_count: int = 0
    sheriff_pk_candidates: list[str] = field(default_factory=list)
    votes: dict[str, str] = field(default_factory=dict)
    private_intents: dict[str, dict[str, Any]] = field(default_factory=dict)
    antidote_used: bool = False
    poison_used: bool = False
    deaths: list[Death] = field(default_factory=list)
    events: list[GameEvent] = field(default_factory=list)
    winning_faction: str | None = None
    paused: bool = False

    def __post_init__(self) -> None:
        # 防御性浅拷贝：防止外部可变容器被意外修改
        object.__setattr__(self, "players", dict(self.players) if self.players else {})
        object.__setattr__(self, "events", list(self.events) if self.events else [])
        object.__setattr__(self, "deaths", list(self.deaths) if self.deaths else [])
        object.__setattr__(self, "votes", dict(self.votes) if self.votes else {})
        object.__setattr__(self, "private_intents", dict(self.private_intents) if self.private_intents else {})
        object.__setattr__(self, "sheriff_candidates", list(self.sheriff_candidates) if self.sheriff_candidates else [])
        object.__setattr__(self, "sheriff_pk_candidates", list(self.sheriff_pk_candidates) if self.sheriff_pk_candidates else [])


@dataclass(frozen=True)
class Action:
    type: str
    target_id: str | None = None


@dataclass(frozen=True)
class RuleResult:
    accepted: bool
    error_code: str | None = None


@dataclass(frozen=True)
class AlignmentResult:
    alignment: str
    role: str | None = None


@dataclass(frozen=True)
class VictoryResult:
    winner: str | None
    reason: str | None = None


@dataclass(frozen=True)
class BadgeDecisionOptions:
    can_transfer: bool
    can_tear: bool
    transfer_targets: list[str]


@dataclass(frozen=True)
class VoteResult:
    exiled_player_id: str | None
    next_phase: str
    reason: str
    tied_player_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GameEvent:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VisibleContext:
    view_mode: str
    visible_sections: set[str]
