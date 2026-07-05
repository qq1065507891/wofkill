# -*- coding: utf-8 -*-
"""
功能描述：SheriffRules 封装警长相关的全部判决逻辑，包括 death_policy（死后徽章处理）、
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import BadgeDecisionOptions, GameEvent, GameState


class SheriffRules:
    """Deterministic sheriff election and badge management rules."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw

    def badge_options_after_sheriff_death(
        self,
        game_state: GameState,
        *,
        sheriff_id: str,
        death_reason: str,
    ) -> BadgeDecisionOptions:
        cfg = self._raw["sheriff"]["death_policy"]
        can_act = death_reason in cfg["may_transfer_or_tear_on_death_reasons"]
        if not can_act:
            return BadgeDecisionOptions(can_transfer=False, can_tear=False, transfer_targets=[])

        targets = [
            pid for pid, p in game_state.players.items()
            if p.alive and pid != sheriff_id and p.badge_eligible
        ]
        return BadgeDecisionOptions(can_transfer=True, can_tear=True, transfer_targets=targets)

    def resolve_badge_decision(
        self, game_state: GameState, *, decision: str, target_id: str | None = None
    ) -> GameState:
        if decision == "tear":
            return replace(game_state, sheriff_id=None, sheriff_badge_state="torn")
        if decision == "transfer" and target_id is not None:
            return replace(game_state, sheriff_id=target_id, sheriff_badge_state="active")
        return game_state

    def sheriff_register(
        self, game_state: GameState, *, candidates: list[str]
    ) -> tuple[GameState, GameEvent]:
        event = GameEvent(type="sheriff_registered", payload={"candidates": candidates})
        return game_state, event

    def sheriff_withdraw(
        self, game_state: GameState, *, candidates: list[str], withdrawing: list[str]
    ) -> tuple[GameState, GameEvent]:
        remaining = [c for c in candidates if c not in withdrawing]
        event = GameEvent(
            type="sheriff_withdraw",
            payload={"remaining": remaining, "withdrew": withdrawing},
        )
        return game_state, event

    def resolve_sheriff_vote(
        self, game_state: GameState, *, votes: dict[str, str], candidates: list[str]
    ) -> tuple[GameState, GameEvent]:
        tally: dict[str, int] = {}
        for target_id in votes.values():
            if target_id in candidates:
                tally[target_id] = tally.get(target_id, 0) + 1
        if not tally:
            return game_state, GameEvent(type="sheriff_no_election", payload={})
        max_votes = max(tally.values())
        top = [pid for pid, count in tally.items() if count == max_votes]
        if len(top) != 1:
            return game_state, GameEvent(type="sheriff_vote_tie", payload={"tied": top})
        winner = top[0]
        new_state = replace(game_state, sheriff_id=winner, sheriff_badge_state="active")
        return new_state, GameEvent(type="sheriff_elected", payload={"sheriff_id": winner})

    def speech_order_policy(self, game_state: GameState) -> str:
        cfg = self._raw["day_flow"]["speech"]
        if game_state.sheriff_badge_state == "active":
            return cfg.get("sheriff_order_policy", "random_start")
        if game_state.sheriff_badge_state == "torn":
            return cfg["torn_badge_order_policy"]
        return cfg["no_sheriff_order_policy"]
