# -*- coding: utf-8 -*-
"""
构造不依赖 LLM 的法官公共播报。

作者: Project contributors
创建日期: 2026-07-08
修改日期: 2026-07-08

使用示例:
    >>> from werewolf_agent.agents.judge_static_broadcasts import build_phase_broadcast
    >>> build_phase_broadcast("day", day_number=1).broadcast_type
    'day'
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.agents.schemas import JudgeBroadcast
from werewolf_agent.runtime.timeline import phase_label


def build_phase_broadcast(
    phase: str,
    day_number: int = 0,
    night_number: int = 0,
    public_data: dict[str, Any] | None = None,
) -> JudgeBroadcast:
    """构造阶段切换播报。"""
    templates: dict[str, str] = {
        "night": f"天黑请闭眼。{phase_label('night', night_number)} 开始。",
        "day": f"天亮了。{phase_label('day', day_number)} 开始。",
        "wolf_discussion": "狼人请睁眼，讨论击杀目标。",
        "witch_turn": "女巫请睁眼。",
        "seer_turn": "预言家请睁眼。",
        "vote": "投票阶段开始。",
        "sheriff_registration": "警长竞选开始，请上警玩家举手。",
        "sheriff_vote": "请警下玩家投票选举警长。",
        "free_discussion": "自由发言阶段开始。",
        "pk_speech": "平票PK发言开始。",
        "victory_good": "好人阵营获胜！",
        "victory_werewolf": "狼人阵营获胜！",
        "finished": "对局结束。",
    }

    message = templates.get(phase, f"进入 {phase} 阶段。")
    if public_data:
        deaths = public_data.get("deaths", [])
        if deaths:
            players_str = "、".join(death.get("player_id", "???") for death in deaths)
            message += f" 昨夜倒牌：{players_str}。"
        exiled = public_data.get("exiled")
        if exiled:
            message += f" {exiled} 被放逐。"
        revealed = public_data.get("revealed_idiot")
        if revealed:
            message += f" {revealed} 翻牌自证白痴身份。"

    return JudgeBroadcast(
        broadcast_type=phase,
        message=message,
        phase=phase,
        day_number=day_number,
        night_number=night_number,
        public_data=public_data or {},
    )


def build_death_announcement_broadcast(
    deaths: list[dict[str, Any]],
    day_number: int,
) -> JudgeBroadcast:
    """将死亡记录转换为公开死亡播报。"""
    if not deaths:
        return JudgeBroadcast(
            broadcast_type="death_announcement",
            message=f"{phase_label('day', day_number)}：昨夜是平安夜，无人倒牌。",
            phase="day",
            day_number=day_number,
            public_data={"death_count": 0, "death_ids": ""},
        )

    dead_names = [death.get("player_id", "???") for death in deaths]
    msg = f"{phase_label('day', day_number)}：昨夜倒牌：{'、'.join(dead_names)}。"
    return JudgeBroadcast(
        broadcast_type="death_announcement",
        message=msg,
        phase="day",
        day_number=day_number,
        public_data={
            "death_count": len(dead_names),
            "death_ids": ",".join(dead_names),
        },
    )


__all__ = [
    "build_death_announcement_broadcast",
    "build_phase_broadcast",
]
