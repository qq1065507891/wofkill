# -*- coding: utf-8 -*-
"""
实现归票、冲票、抗推和遗言分析类技能 handler。

作者: Project contributors
创建日期: 2026-07-07

使用示例:
    >>> from werewolf_agent.skills.good_vote_handlers import push_vote_handler
"""

from __future__ import annotations

from werewolf_agent.skills.advice_frames import (
    _cap_prompt_injectable,
    _push_vote_advice_frame,
)
from werewolf_agent.skills.schemas import SkillDefinition, SkillInput, SkillName, SkillOutput
from werewolf_agent.skills.skill_context import (
    _alerts_for_player,
    _alive_non_wolves,
    _alive_wolves,
    _belief_top_suspects,
    _seer_checks_on_target,
    _vote_targets_for_player,
)
from werewolf_agent.skills.skill_handler_registry import register_handler

@register_handler(SkillName.PUSH_VOTE)
def push_vote_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    # P1-K5: branch on task_type. push_vote during VOTE phase = "who to
    # actually vote for". push_vote during SPEECH phase = "how to
    # rhetorically rally others to vote for X". The two need different
    # advice and confidence framing.
    is_vote_task = inp.task_type == "vote"
    is_speech_task = inp.task_type == "speech"
    if gs is None:
        # static fallback — task_type-specific phrasing
        if is_vote_task:
            prompt = (
                "归票建议（投票阶段）：根据发言逻辑和验人信息，"
                "选择嫌疑最大的人作为你的投票目标。"
            )
            speech = ["确认你的最终投票目标", "回顾其嫌疑证据", "准备投出选票"]
        else:
            # Default (and speech-task): rhetoric-focused push.
            prompt = (
                "归票建议：根据发言逻辑和验人信息，选择嫌疑最大的人归票。"
                "陈述理由时需要有理有据，号召全场跟随。"
            )
            speech = ["陈述归票理由", "分析目标嫌疑", "号召全场归票"]
        risks = ["归票错误目标可能导致好人损失"]
        return SkillOutput(
            skill_name=skill.name.value,
            speech_structure=speech,
            risk_alerts=risks,
            confidence=0.6,
            reasoning="归票需要有充分的逻辑依据和说服力",
            prompt_injectable=_cap_prompt_injectable(prompt),
            advice_frame=_push_vote_advice_frame(
                inp,
                skill,
                prompt=prompt,
                risks=risks,
                confidence=0.6,
            ),
        )
    # dynamic analysis
    ws = inp.world_state
    bs = inp.belief_state

    top_suspects = _belief_top_suspects(bs, count=3)

    if not top_suspects:
        if is_vote_task:
            prompt = (
                "归票建议（投票阶段）：当前信息不足，没有明确嫌疑目标。"
                "选择一个相对最可疑的目标投票，避免弃票。"
            )
        elif is_speech_task:
            prompt = (
                "归票建议（发言阶段）：当前信息不足，没有明确嫌疑目标。"
                "在发言中表示需要观察，避免无依据地号召归票。"
            )
        else:
            prompt = "归票建议：当前信息不足，建议观察发言后再决定归票方向。"
        return SkillOutput(
            skill_name=skill.name.value,
            confidence=0.4,
            reasoning="当前无明确嫌疑目标",
            prompt_injectable=_cap_prompt_injectable(prompt),
            advice_frame=_push_vote_advice_frame(
                inp,
                skill,
                prompt=prompt,
                risks=[],
                confidence=0.4,
            ),
        )

    primary, lean, trust = top_suspects[0]
    reasons = []
    if lean == "wolf_lean":
        reasons.append(f"行为模式偏向狼人")
    if trust < 0.3:
        reasons.append(f"可信度极低({trust:.0%})")

    checks = _seer_checks_on_target(ws, primary)
    for c in checks:
        if "wolf" in c.get("value", "").lower() or "狼" in c.get("value", ""):
            reasons.append(f"被{c['source']}查杀为狼")
            break

    votes_on = _vote_targets_for_player(ws, primary)
    if len(votes_on) >= 2:
        reasons.append(f"已有多人({len(votes_on)}票)指向该玩家")

    reason_text = "；".join(reasons) if reasons else "综合行为分析"
    if is_vote_task:
        # Vote phase: emphasize the target pick + evidence, no rhetoric.
        prompt = (
            f"归票建议（投票阶段）：{primary} 的嫌疑最大。"
            f"理由：{reason_text}。"
            f"请直接选 {primary} 作为你的投票目标。"
        )
        speech = [f"确认{primary}为最终投票目标", f"回顾{primary}的嫌疑证据", "投出选票"]
        risks = ["归票错误目标可能导致好人损失"]
    elif is_speech_task:
        # Speech phase: emphasize rhetoric, herd rallying.
        prompt = (
            f"归票建议（发言阶段）：{primary} 的嫌疑最大。"
            f"理由：{reason_text}。"
            f"在发言中陈述理由，号召全场集中票数归出 {primary}。"
        )
        speech = [f"陈述{primary}的嫌疑理由", "分析其行为链", "号召全场归票"]
        risks = ["归票错误目标可能导致好人损失"]
    else:
        # Unknown / default: keep original generic phrasing.
        prompt = (
            f"归票建议：根据场上信息，{primary} 的嫌疑最大。"
            f"理由：{reason_text}。号召全场集中票数归出 {primary}。"
        )
        speech = [f"陈述{primary}的嫌疑理由", "分析其行为链", "号召全场归票"]
        risks = ["归票错误目标可能导致好人损失"]

    conf = 0.6 + min(0.2, len(reasons) * 0.05)
    return SkillOutput(
        skill_name=skill.name.value,
        speech_structure=speech,
        risk_alerts=risks,
        confidence=conf,
        reasoning=f"动态分析：{primary} 有{len(reasons)}个嫌疑信号"
                  + ("（vote task）" if is_vote_task else
                     "（speech task）" if is_speech_task else ""),
        prompt_injectable=_cap_prompt_injectable(prompt),
        advice_frame=_push_vote_advice_frame(
            inp,
            skill,
            prompt=prompt,
            risks=risks,
            confidence=conf,
            primary=primary,
        ),
    )

@register_handler(SkillName.SWING_VOTE)
def swing_vote_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    # S-03: branch on task_type. `wolf_discussion` is a NIGHT phase —
    # the wolves are picking a night-kill target, not a day-vote target.
    # Default (day-vote) behavior is preserved.
    is_wolf_discussion = inp.task_type == "wolf_discussion"

    if gs is None:
        # static fallback
        risks = ["冲票暴露投票链：好人可能通过投票链锁定狼人"]
        if is_wolf_discussion:
            return SkillOutput(
                skill_name=skill.name.value,
                speech_structure=["在狼讨论中提出冲刀目标", "分析目标的投票压力", "协调队友分散或集中夜杀"],
                risk_alerts=risks,
                confidence=0.5,
                reasoning="冲刀需要考虑夜杀链暴露风险",
                prompt_injectable=_cap_prompt_injectable((
                    "冲刀建议（狼队夜杀讨论）：选择场上已有投票压力的好人作为冲刀目标。"
                    "与队友协调夜杀方向，避免夜杀链暴露狼人身份。"
                )),
            )
        return SkillOutput(
            skill_name=skill.name.value,
            speech_structure=["在狼讨论中提出冲票目标", "分析目标的投票压力", "协调队友分散或集中票"],
            risk_alerts=risks,
            confidence=0.5,
            reasoning="冲票需要考虑投票链暴露风险",
            prompt_injectable=_cap_prompt_injectable("冲票建议：选择场上已有投票压力的好人作为冲票目标。与队友协调投票方向，避免投票链暴露狼人身份。"),
        )
    # dynamic analysis
    ws = inp.world_state

    wolves = _alive_wolves(gs)
    non_wolves = _alive_non_wolves(gs)

    if not non_wolves:
        return SkillOutput(
            skill_name=skill.name.value,
            confidence=0.0,
            reasoning="无冲票目标" if not is_wolf_discussion else "无冲刀目标",
        )

    # Find non-wolf with most vote pressure
    # Weight by game phase: later phases have more settled opinions, so
    # votes and suspect claims carry more weight.
    day = gs.day_number
    vote_weight = 1.0 + min(day * 0.3, 1.5)    # 1.0 → 2.5 over 5 days
    suspect_weight = 1.5 + min(day * 0.3, 1.5)  # 1.5 → 3.0 over 5 days

    pressure: dict[str, float] = {pid: 0.0 for pid in non_wolves}
    for f in (ws.facts_of_type("vote") if ws else []):
        if f.target_player in pressure:
            pressure[f.target_player] += vote_weight
    for f in (ws.facts_of_type("claimed_suspect") if ws else []):
        if f.target_player in pressure:
            pressure[f.target_player] += suspect_weight

    best_target = max(non_wolves, key=lambda p: pressure.get(p, 0))
    best_pressure = pressure.get(best_target, 0)

    wolf_count = len(wolves)
    risks = ["冲票暴露投票链：好人可能通过投票链锁定狼人"]
    if wolf_count <= 2:
        risks.append(f"仅剩{wolf_count}狼，冲票暴露风险极高")

    if is_wolf_discussion:
        # Night-kill semantics: rephrase the prompt and action.
        if best_pressure > 0:
            prompt = (
                f"冲刀建议（狼队夜杀讨论）：集中狼队夜杀 {best_target}。"
                f"理由：{best_target} 已有{best_pressure}个怀疑信号，"
                f"冲刀成功率高，避免暴露狼队。"
            )
            conf = 0.5 + min(0.2, best_pressure * 0.04)
        else:
            prompt = (
                "冲刀建议（狼队夜杀讨论）：当前无明确冲刀目标，"
                "建议根据好人发言集中度选择夜杀目标。"
            )
            conf = 0.35
        return SkillOutput(
            skill_name=skill.name.value,
            risk_alerts=risks,
            confidence=conf,
            reasoning=(
                f"动态分析：{best_target} 有{best_pressure}个压力信号，"
                f"{wolf_count}狼存活（夜杀任务）"
            ),
            prompt_injectable=_cap_prompt_injectable(prompt),
        )

    if best_pressure > 0:
        prompt = (
            f"冲票建议：集中狼队票数冲 {best_target}。"
            f"理由：{best_target} 已有{best_pressure}个怀疑信号，冲票成功率高。"
        )
        conf = 0.5 + min(0.2, best_pressure * 0.04)
    else:
        prompt = "冲票建议：当前无明确冲票目标，建议分散投票避免暴露。"
        conf = 0.35

    return SkillOutput(
        skill_name=skill.name.value,
        risk_alerts=risks,
        confidence=conf,
        reasoning=f"动态分析：{best_target} 有{best_pressure}个压力信号，{wolf_count}狼存活",
        prompt_injectable=_cap_prompt_injectable(prompt),
    )

@register_handler(SkillName.RESIST_PUSH)
def resist_push_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    if gs is None:
        # static fallback
        return SkillOutput(
            skill_name=skill.name.value,
            speech_structure=["承认疑虑合理性", "逻辑反驳关键指控", "提出建设性站边"],
            risk_alerts=["过度防御可能加深怀疑", "攻击质疑者会适得其反"],
            confidence=0.55,
            reasoning="抗推需要冷静的逻辑反驳，而非情绪对抗",
            prompt_injectable=_cap_prompt_injectable("抗推建议：冷静反驳质疑，针对关键指控逐一回应。如果被查杀，质疑预言家身份；如果仅被怀疑，补充自己的逻辑线和站边理由。"),
        )
    # dynamic analysis
    ws = inp.world_state
    pid = inp.player_id
    alerts = inp.contradiction_alerts

    # Check if player is seer-checked (查杀)
    seer_checks = _seer_checks_on_target(ws, pid)
    is_seer_checked = any(
        "wolf" in c.get("value", "").lower() or "狼" in c.get("value", "")
        for c in seer_checks
    )

    # Find who is pushing against us
    votes_against = _vote_targets_for_player(ws, pid)
    pushers = [v["source"] for v in votes_against if v.get("day", 0) == inp.game_state.day_number]

    risks = ["过度防御可能加深怀疑", "攻击质疑者会适得其反"]
    if is_seer_checked:
        checker = seer_checks[0]["source"] if seer_checks else "未知"
        prompt = (
            f"抗推建议：你被 {checker} 查杀为狼。这是最危险的推票信号。"
            f"必须质疑该玩家的预言家身份——指出其验人动机和逻辑漏洞。"
            f"可以考虑：1) 质疑其预言家身份 2) 提出自己的身份线 3) 分析其站边的可疑性。"
        )
        conf = 0.6
        speech = [f"质疑{checker}的预言家身份", "提出自身行为的一致性", "分析查杀动机的合理性"]
    elif pushers:
        prompt = (
            f"抗推建议：你被 {', '.join(pushers)} 等人怀疑/推票，但无查杀等实质证据。"
            f"冷静反驳：指出指控缺乏事实基础，转而质疑推你的人的逻辑。"
            f"用事实和逻辑回应，避免情绪化。"
        )
        conf = 0.55
        speech = ["指出指控缺乏实质证据", "逻辑反驳关键指控", "转而质疑推票者的动机"]
    else:
        prompt = (
            f"抗推建议：你目前被轻微怀疑但无明确推票压力。"
            f"保持冷静，适度解释自己的立场，避免过度反应引起更多怀疑。"
        )
        conf = 0.5
        speech = ["适度解释自身立场", "保持冷静客观", "贡献有价值分析转移注意力"]

    # Check for contradictions in pushers' behavior (counter-attack material)
    for pusher in pushers[:1]:
        pusher_alerts = _alerts_for_player(alerts, pusher)
        if pusher_alerts:
            prompt += f" 注意：{pusher} 自身存在发言矛盾，可作为反击材料。"
            conf += 0.05

    return SkillOutput(
        skill_name=skill.name.value,
        speech_structure=speech,
        risk_alerts=risks,
        confidence=conf,
        reasoning="动态分析：根据推票来源和强度调整抗推策略",
        prompt_injectable=_cap_prompt_injectable(prompt),
    )

@register_handler(SkillName.LAST_WORDS_ANALYSIS)
def last_words_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    # S-18: unify the static fallback (gs is None) and the no-ws
    # branch (gs set but world_state is None) into a single early
    # return. Previously the two branches had duplicated code with
    # identical output — a divergence risk if one branch was edited
    # and the other was not.
    if gs is None or inp.world_state is None:
        return SkillOutput(
            skill_name=skill.name.value,
            speech_structure=["提取遗言关键信息", "分析遗言与已知信息的矛盾", "评估遗言可信度"],
            confidence=0.55,
            reasoning="遗言分析需要结合已有信息判断遗言内容的真实性",
            prompt_injectable=_cap_prompt_injectable("遗言分析建议：关注出局玩家最后发言中的角色声明、验人信息和站边逻辑。与已知信息交叉验证，判断遗言内容的可信度。"),
        )
    # dynamic analysis
    ws = inp.world_state
    alerts = inp.contradiction_alerts

    # Find recent deaths
    recent_deaths = list(ws.facts_of_type("player_died"))

    if not recent_deaths:
        return SkillOutput(
            skill_name=skill.name.value,
            confidence=0.3,
            reasoning="暂无死亡事件可分析",
            prompt_injectable=_cap_prompt_injectable("遗言分析：当前无遗言可分析。"),
        )

    # Analyze all deaths, prioritizing the most recent
    all_prompts: list[str] = []
    has_contradiction = False
    last_dead_player = None

    for death in reversed(recent_deaths):
        dead_player = death.target_player
        if dead_player is None:
            continue
        if last_dead_player is None:
            last_dead_player = dead_player

        # Find last speech of dead player
        speech_count = sum(
            1 for f in ws.facts_of_type("speech")
            if f.source_player == dead_player
        )

        # Check for contradictions
        dead_alerts = _alerts_for_player(alerts, dead_player)
        if dead_alerts:
            has_contradiction = True

        # Collect claims
        claims = [
            f"{f.value}(Day{f.day})"
            for f in ws.facts_of_type("claimed_role")
            if f.source_player == dead_player
        ]

        parts = [f"{dead_player}的遗言："]
        if claims:
            parts.append(f"身份声明：{', '.join(claims)}。")
        if dead_alerts:
            parts.append(f"发言矛盾：{'; '.join(a.description for a in dead_alerts[:2])}。")
        if speech_count:
            parts.append(f"有{speech_count}条发言记录。")
        # NEW-R4-P2-3: if the dead player has no claims, no
        # contradictions, and no speeches, parts is just the bare
        # "p05的遗言：" label with no body — a useless artifact that
        # wastes prompt budget. Fall back to a placeholder so the
        # LLM has something to read.
        if len(parts) == 1:
            parts.append("无具体遗言内容可分析。")
        all_prompts.append("".join(parts))

    if not all_prompts:
        return SkillOutput(
            skill_name=skill.name.value,
            confidence=0.3,
            reasoning="无有效遗言数据",
            prompt_injectable=_cap_prompt_injectable("遗言分析：无有效遗言可分析。"),
        )

    prompt = f"遗言分析（{len(all_prompts)}人死亡）：\n" + "\n".join(all_prompts)
    prompt += "\n需要结合已知信息判断遗言内容的真实性和意图。"

    return SkillOutput(
        skill_name=skill.name.value,
        speech_structure=["提取遗言关键信息", "对比已知信息一致性", "评估可信度"],
        risk_alerts=["遗言可能是狼人的误导"] if has_contradiction else [],
        confidence=0.55 if not has_contradiction else 0.65,
        reasoning="动态分析：根据遗言内容与已知信息的对比判断可信度",
        prompt_injectable=_cap_prompt_injectable(prompt),
    )
