"""Villager day-speech directive builder."""

from __future__ import annotations

import re
from typing import Any

from werewolf_agent.core.models import GameState
from werewolf_agent.runtime.directives._shared import collect_death_order, collect_public_vote_history


def build_villager_directive(
    gs: GameState,
    villager_id: str,
) -> dict[str, Any]:
    """Build day speech directive for villager -- pure analysis, no private info.

    Note: the idiot role has its own directive (idiot.py) and is not
    routed through this function, so we do not branch on role_label here.
    """
    from werewolf_agent.runtime.strategy.seer import public_seer_claimants as _public_seer_claimants

    parts: dict[str, Any] = {}

    # Collect public information for analysis
    seer_claimants = _public_seer_claimants(gs)
    # M3-2: cap public history to current day so day-5 LLM
    # does not dilute focus with day-1 patterns.
    vote_history = collect_public_vote_history(gs, current_day=gs.day_number)
    death_order = collect_death_order(gs, current_day=gs.day_number)

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

    # Determine gold-water targets strictly from public seer claims + their
    # public speeches. We MUST NOT read private seer_check events here, since
    # villager agents must never see private night information belonging to
    # the seer. A target is "publicly announced as gold water" only if a
    # public seer-claimant explicitly named them as a good check in speech.
    gold_water_targets: set[str] = set()
    if seer_claimants:
        for e in gs.events:
            if e.type not in ("speech", "sheriff_speech"):
                continue
            if e.payload.get("speaker") not in seer_claimants:
                continue
            text = str(e.payload.get("text", ""))
            for match in re.finditer(
                r"验[了过]?\s*(p\d+).*?好人|验[了过]?\s*(p\d+).*?金水",
                text,
            ):
                target = match.group(1) or match.group(2)
                if target:
                    gold_water_targets.add(target)

    if villager_id in gold_water_targets:
        parts["gold_water_duty"] = (
            f"你是{villager_id}，公开场上预言家已报你为金水。"
            "请基于此身份积极帮助好人阵营：\n"
            "1) 主动站边公开预言家，传递信任信号；\n"
            "2) 发言中可适度引用预言家给你的金水身份，但不要伪造额外查杀信息；\n"
            "3) 投票时优先跟随查杀方归票，保留对悍跳狼的质疑能力。"
        )

    parts["villager_speech_directive"] = (
        f"你是普通村民，没有夜间技能和私有信息，你的核心价值是逻辑分析能力。\n\n"
        "发言策略：\n"
        "1) 不要复述别人的观点——提出你自己的分析和判断\n"
        "2) 引用具体的发言内容和投票数据来支撑你的论点\n"
        "3) 如果你有独立的怀疑对象，说明理由；不要无证据跟风\n"
        "4) 不要冒充任何角色——你没有信息来支撑冒充\n"
        "5) 如果预言家已死或被怀疑，好人阵营需要你站出来做逻辑整理"
        f"{seer_analysis}{vote_analysis}{death_analysis}"
    )

    return parts
