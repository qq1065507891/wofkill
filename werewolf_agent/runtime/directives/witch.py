# -*- coding: utf-8 -*-
"""Witch day-speech directive builder.
    作者: Mike
    创建日期: 2025-01-15
    修改日期: 2026-07-05
    使用示例: 内部模块，无对外接口
D-1: extracted from agent_adapter's hard-coded ``witch_speech_constraint``
block.  D-7: now consults ``evaluate_death_cause_claims`` so the witch's
day-speech guidance is informed by her private knowledge of what poison /
antidote actions she has taken this game.
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.core.models import GameState


def build_witch_directive(
    gs: GameState | None,
    witch_id: str,
) -> dict[str, Any]:
    """Build the day-speech directive for the witch.

    Returns a dict with at least ``witch_speech_directive`` (the
    structured strategy text shown to the LLM).  When ``gs`` is provided,
    the directive is enriched with the witch's private view of
    death-cause claims via :func:`evaluate_death_cause_claims`.
    """
    parts: dict[str, Any] = {}

    base_text = (
        "你是女巫，你掌握的夜间信息（谁被刀、药水使用情况、救了谁、毒了谁）"
        "是你的核心优势。不要轻易暴露这些信息——一旦公开，狼人会知道你的"
        "药水状态并针对性调整策略。但在以下情况可以适度透露：\n"
        "1) 你即将死亡需要传递关键信息；\n"
        "2) 场上好人阵营信息严重不足，需要你站出来带队；\n"
        "3) 有人假冒女巫需要你自证身份。\n"
        "透露时也要衡量利弊，不要在第一天就全部交底。\n\n"
        "【毒药公开依据校验 P0-G3223805846-5】\n"
        "你的毒药目标必须能在 day_speech 公开事件里追溯到以下任一来源：\n"
        "- 已被预言家（含悍跳预言家）报为'查杀'的具体玩家 ID\n"
        "- 已被多人明确指控并形成证据链的具体玩家 ID\n"
        "- 已公开跳神职/悍跳且证据不足自洽的具体玩家 ID\n"
        "**禁止**仅凭'看起来像狼'、'语气不对'、'可能是神'这类模糊理由用毒。"
        "若你 reason 中引用的 player_id 实际是由别人报的查杀（如 p05 报'p06 是狼'），"
        "请在 reason 中明确写出'基于 p05 的查杀'以避免自己记错。\n"
        "【解药策略 P0-G3223805846-5】\n"
        "解药是否使用应结合目标价值和公开证据：高价值好人或可信神职倾向救；"
        "若目标身份逻辑严重破产或存在强狼证据，保留解药也合理。"
        "夜间刀口本身不证明目标是好人。"
    )
    parts["witch_speech_directive"] = base_text

    if gs is None:
        return parts

    # D-7: enrich with private-knowledge death-cause evaluation so the
    # witch's day speech can decide when to surface / hide poison claims.
    try:
        from werewolf_agent.runtime.strategy.death import evaluate_death_cause_claims

        # Collect wolf kill target for this night from public events.
        wolf_kill_target_id: str | None = None
        for e in gs.events:
            if e.type == "wolf_kill_selected":
                night = e.payload.get("night_number", 0)
                target = e.payload.get("target_id", "")
                if night and target and night == gs.night_number:
                    wolf_kill_target_id = target
                    break

        evaluations = evaluate_death_cause_claims(
            gs, witch_id, "witch", wolf_kill_target_id=wolf_kill_target_id,
        )
        if evaluations:
            parts["witch_death_cause_evaluations"] = evaluations
    except Exception:
        # The directive must never break the agent's day speech; if the
        # strategy evaluator chokes (e.g., on partial state in tests),
        # silently skip the enrichment.
        pass

    return parts
