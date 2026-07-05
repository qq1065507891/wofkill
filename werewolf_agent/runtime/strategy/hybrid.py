# -*- coding: utf-8 -*-
"""
功能描述：混血儿策略评估函数。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：内部模块，无对外接口
"""
from __future__ import annotations

from typing import Any

from werewolf_agent.core.models import GameState


def evaluate_hybrid_master_candidates(
    gs: GameState,
    hybrid_id: str,
    candidates: list[str],
) -> dict[str, Any]:
    """Evaluate master candidates for the hybrid on N1.

    The hybrid doesn't know any player's role, so scoring is based on
    observable public signals from the sheriff election and speeches.
    """
    scores: dict[str, dict[str, Any]] = {}
    for pid in candidates:
        sig: list[str] = []
        value = 0

        # Registered for sheriff election — likely power role or confident player
        for e in gs.events:
            if e.type == "sheriff_registration" and e.payload.get("player_id") == pid:
                sig.append("ran_for_sheriff")
                value += 3
                break

        # Gave a substantive sheriff speech — engaged and likely experienced
        for e in gs.events:
            if e.type == "sheriff_speech" and e.payload.get("speaker") == pid:
                text = str(e.payload.get("text", ""))
                if len(text) > 50:
                    sig.append("substantive_sheriff_speech")
                    value += 2
                break

        # Claimed a power role in speech — could be real or fake, but either way influential
        for e in gs.events:
            if e.type not in ("speech", "sheriff_speech"):
                continue
            if e.payload.get("speaker") != pid:
                continue
            claims = e.payload.get("claims") or []
            for claim in claims:
                if claim.get("type") == "role":
                    sig.append(f"claimed_{claim['value']}")
                    value += 4
                    break
            break

        # Position-based: prefer players in middle positions (less likely to be
        # first-night wolf targets in a positional meta)
        scores[pid] = {"value": value, "signals": sig}

    ranked = sorted(scores.items(), key=lambda x: x[1]["value"], reverse=True)
    total_candidates = len(candidates)
    # Derive role counts from actual game state
    wolf_count = sum(1 for pid, p in gs.players.items() if p.role == "werewolf" and pid != hybrid_id)
    god_count = sum(1 for p in gs.players.values() if p.role in ("seer", "witch", "hunter", "idiot"))
    villager_count = total_candidates - wolf_count - god_count

    return {
        "description": "主人候选评估（分数越高，玩家影响力越大，对混血儿越有价值）",
        "probability_framework": {
            "p_good_faction": f"~{(god_count + villager_count) / max(1, total_candidates):.0%}（{god_count + villager_count}/{total_candidates} 好人阵营）",
            "p_wolf_faction": f"~{wolf_count / max(1, total_candidates):.0%}（{wolf_count}/{total_candidates} 狼人阵营）",
            "note": "你不知道主人阵营，选到好人和狼人的概率都有，策略需要灵活适应",
        },
        "ranked_candidates": [
            {"target": t, "value": d["value"], "signals": d["signals"]}
            for t, d in ranked
        ],
        "strategy_guidance": (
            "选主人策略考量：\n"
            "1) 选择影响力大且存活概率高的玩家（上警、发言积极、声称神职但不拉满仇恨）——"
            "影响力大的主人意味着你的胜利条件更容易实现，但太拉仇恨的玩家（如悍跳预言家）"
            "可能活不过两轮。平衡影响力和生存性：警上发言有条理但不激进的人是好选择\n"
            "2) 避免选择自己——你不能选自己\n"
            "3) 不要过于纠结概率——7:4 的好人:狼人比例下，你更大概率是好人阵营，"
            "但游戏进程会告诉你主人的真正阵营\n"
            "4) 选定后无法更改，请在考虑后做出决定"
        ),
    }
