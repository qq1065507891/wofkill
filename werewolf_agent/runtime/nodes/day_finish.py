# -*- coding: utf-8 -*-
"""
原子提交胜利条件并结束日间阶段游戏。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-13

使用示例:
    内部运行时节点模块。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.nodes._shared import (
    logger,
    RuntimeState,
    _judge_broadcast,
)


def _commit_victory(state: RuntimeState) -> dict[str, Any]:
    """通过 RuleEngine 原子检查并提交唯一的胜负事件。"""
    engine: RuleEngine = state["engine"]
    gs: GameState = state["game_state"]
    if gs.winning_faction is not None:
        return {"game_state": gs}
    alive_player_ids = sorted(pid for pid, player in gs.players.items() if player.alive)
    for event in reversed(gs.events):
        if event.type != "victory_checked":
            continue
        if event.payload.get("alive_player_ids") == alive_player_ids:
            return {"game_state": gs}
        break
    result = engine.check_victory(gs)
    checked_payload = {
        "winner": result.winner,
        "reason": result.reason,
        "alive_player_ids": alive_player_ids,
    }
    gs = replace(gs, events=gs.events + [GameEvent(type="victory_checked", payload=checked_payload)])

    if result.winner is not None:
        wf = result.winner
        faction_label = "好人阵营" if wf == "good" else "狼人阵营"
        logger.debug(f"\n{'='*60}")
        logger.debug(f"  【游戏结束】胜利方: {faction_label} ({result.reason})")
        logger.debug(f"{'='*60}")

        # 公开胜利公告。
        identity_reveal = ", ".join(
            f"{pid}({p.role})" for pid, p in gs.players.items()
        )
        gs, _ = _judge_broadcast(
            phase="victory_announce",
            message=f"游戏结束，{faction_label}获胜！({result.reason})",
            gs=gs, day_number=gs.day_number,
            extra_payload={
                "winner": wf,
                "reason": result.reason,
                "identities": identity_reveal,
            },
            visibility="public",
        )
        hr = None
        if wf == "good" and gs.hybrid_master_faction == "good":
            hr = "win"
        elif wf == "good" and gs.hybrid_master_faction == "werewolf":
            hr = "lose"
        elif wf == "werewolf" and gs.hybrid_master_faction == "werewolf":
            hr = "win"
        elif wf == "werewolf" and gs.hybrid_master_faction == "good":
            hr = "lose"
        gs = replace(gs, winning_faction=wf, hybrid_result=hr,
                     events=gs.events + [GameEvent(
                         type="victory",
                         payload={
                             "winner": wf,
                             "winning_faction": wf,
                             "reason": result.reason,
                             "hybrid_master_id": gs.hybrid_master_id,
                             "hybrid_master_faction": gs.hybrid_master_faction,
                             "hybrid_result": hr,
                         },
                     )])
    return {"game_state": gs, "_victory_result": result}


def check_victory(state: RuntimeState) -> dict[str, Any]:
    """兼容图节点入口；胜负已提交时保持幂等。"""
    return _commit_victory(state)


def finish_game(state: RuntimeState) -> dict[str, Any]:
    gs: GameState = state["game_state"]

    # 最终身份公开公告。
    identity_lines = []
    for pid, p in gs.players.items():
        status = "存活" if p.alive else "死亡"
        identity_lines.append(f"{pid}({p.role}, {status})")
    gs, _ = _judge_broadcast(
        phase="game_end_reveal",
        message="游戏结束，公布所有玩家身份：" + "；".join(identity_lines),
        gs=gs, day_number=gs.day_number,
        extra_payload={"identities": {pid: p.role for pid, p in gs.players.items()}},
        visibility="public",
    )

    gs = replace(gs, phase="finished",
                 events=gs.events + [GameEvent(type="game_finished", payload={})])
    return {"game_state": gs}
