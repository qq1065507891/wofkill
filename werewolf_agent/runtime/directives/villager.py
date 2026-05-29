"""Villager day-speech directive builder."""

from __future__ import annotations

from typing import Any

from werewolf_agent.core.models import GameState
from werewolf_agent.runtime.directives._shared import collect_death_order, collect_public_vote_history


def build_villager_directive(
    gs: GameState,
    villager_id: str,
) -> dict[str, Any]:
    """Build day speech directive for villager/idiot -- pure analysis, no private info."""
    # TODO: Circular dependency — will be resolved when _public_seer_claimants moves to runtime/strategy/ (Task 2)
    from werewolf_agent.runtime.agent_adapter import _public_seer_claimants  # noqa: TID251

    parts: dict[str, Any] = {}

    # Collect public information for analysis
    seer_claimants = _public_seer_claimants(gs)
    vote_history = collect_public_vote_history(gs)
    death_order = collect_death_order(gs)

    # Build seer claim analysis if there are competing claims
    seer_analysis = ""
    if len(seer_claimants) >= 2:
        seer_analysis = (
            "\n\n【对跳预言家分析】场上有多个预言家声明，你需要独立判断：\n"
            "1) 验人逻辑链：谁的验人报告与死亡、投票数据吻合？\n"
            "2) 警徽流一致性：谁在遵守自己的警徽流承诺？\n"
            "3) 发言质量：谁的发言有实质信息，谁只是在泛泛而谈？\n"
            "4) 站边分析：谁在帮好人说话，谁在帮狼人打掩护？\n"
            f"对跳预言家: {sorted(seer_claimants)}"
        )
    elif len(seer_claimants) == 1:
        seer_analysis = (
            f"\n\n【单边预言家】场上只有一个预言家声明: {sorted(seer_claimants)}，"
            "单边预言家可信度较高，但仍需关注其验人逻辑是否合理。"
        )

    # Build vote pattern analysis
    vote_analysis = ""
    if vote_history:
        vote_analysis = "\n\n【投票数据参考】" + vote_history

    # Build death order analysis
    death_analysis = ""
    if death_order:
        death_analysis = "\n\n【死亡顺序】" + death_order

    role_label = "白痴" if gs.players.get(villager_id, None) and gs.players[villager_id].role == "idiot" else "普通村民"

    # Check if this villager is seer-verified gold water
    gold_water_duty = ""
    for e in gs.events:
        if e.type == "seer_check" and e.payload.get("target_id") == villager_id:
            if e.payload.get("alignment") == "good":
                seer = next(
                    (pid for pid, p in gs.players.items() if p.role == "seer"), None
                )
                if seer:
                    gold_water_duty = (
                        f"\n\n【你是预言家{seer}的金水】你在好人视角是最高身份。"
                        f"你有义务为预言家站边——你是场上最应该信他的人。"
                        f"如果预言家被多人踩，你必须站出来帮他拉票、分析谁在冲票。"
                    )
            break

    parts["villager_speech_directive"] = (
        f"你是{role_label}，没有夜间技能和私有信息，你的核心价值是逻辑分析能力。\n\n"
        "发言策略：\n"
        "1) 不要复述别人的观点——提出你自己的分析和判断\n"
        "2) 引用具体的发言内容和投票数据来支撑你的论点\n"
        "3) 如果你有独立的怀疑对象，说明理由；不要无证据跟风\n"
        "4) 不要冒充任何角色——你没有信息来支撑冒充\n"
        "5) 如果预言家已死或被怀疑，好人阵营需要你站出来做逻辑整理"
        f"{gold_water_duty}{seer_analysis}{vote_analysis}{death_analysis}"
    )

    return parts
