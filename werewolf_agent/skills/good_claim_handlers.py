# -*- coding: utf-8 -*-
"""
实现好人侧声明和对跳类技能 handler。

作者: Project contributors
创建日期: 2026-07-07
修改日期: 2026-07-09

使用示例:
    >>> from werewolf_agent.skills.good_claim_handlers import bold_claim_handler
"""

from __future__ import annotations

from werewolf_agent.skills.advice_frames import (
    _cap_prompt_injectable,
    _counter_claim_advice_frame,
)
from werewolf_agent.skills.schemas import SkillDefinition, SkillInput, SkillName, SkillOutput
from werewolf_agent.skills.skill_context import (
    _alerts_for_player,
    _alive_wolves,
    _count_seer_claimants,
    _get_seer_claimants,
)
from werewolf_agent.skills.skill_handler_registry import register_handler

@register_handler(SkillName.BOLD_CLAIM)
def bold_claim_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    if gs is None:
        # static fallback — day-conditional branching (NEW-R4-P2-6).
        # Day 1 = 窗口开放 (strongest encouragement). Day 2 = 过渡期
        # (still possible, conditional on team plan). Day 3+ = 窗口
        # 关闭 (deprioritize, advise only if teammate already failed).
        risks = ["悍跳风险：如果对跳方是真预言家，可信度会大幅下降"]
        if inp.day >= 2:
            risks.append("悍跳需要完整的假时间线，发言越多越容易暴露")
        if inp.day >= 3:
            risks.append("晚期悍跳风险更高：已发言轮次多，矛盾点容易被抓")
        if inp.day <= 1:
            conf = 0.6
            prompt = (
                "悍跳建议：当前为Day 1，悍跳窗口最佳。"
                "尽早跳预言家并报出假查验结果，构建完整时间线和警徽流。"
                "队友可配合站边，把假预言家身份坐实。"
            )
        elif inp.day == 2:
            conf = 0.45
            prompt = (
                "悍跳建议：当前为Day 2，仍有悍跳窗口但风险上升。"
                "如果队内尚无假预言家且场上无人跳，可考虑悍跳，"
                "需准备更精细的假时间线和与Day 1发言的兼容性。"
                "如果已有队友尝试失败，悍跳会进一步暴露狼队，建议放弃。"
            )
        else:  # day >= 3
            conf = 0.3
            prompt = (
                "悍跳建议：当前已过Day 2，悍跳窗口基本关闭。"
                "除非队内无人尝试且对跳方明显是悍跳失败者，"
                "否则不建议此时悍跳——风险远大于收益，"
                "应转为深水或倒钩策略。"
            )
        return SkillOutput(
            skill_name=skill.name.value,
            speech_structure=["报查验结果", "声明警徽流", "攻击对立面逻辑"],
            risk_alerts=risks,
            confidence=conf,
            reasoning="悍跳需要前期执行，后期风险增大",
            prompt_injectable=_cap_prompt_injectable(prompt),
        )
    # dynamic analysis
    ws = inp.world_state
    day = gs.day_number
    seer_count = _count_seer_claimants(ws)
    wolves = _alive_wolves(gs)

    # If wolf_team_plan assigns a different wolf as fake_seer, skip bold claim advice
    wolf_plan = inp.extra.get("wolf_team_plan") if inp.extra else None
    if wolf_plan and wolf_plan.get("fake_seer") and wolf_plan["fake_seer"] != inp.player_id:
        # S-14: do NOT name the fake_seer teammate.  Role-neutral
        # phrasing — the teammate's player_id is a wolf-team secret
        # and must not leak into a prompt that may be inspected by
        # good-faction analysis tools.
        return SkillOutput(
            skill_name=skill.name.value,
            confidence=0.3,
            reasoning="已有队友占据预言家身份，你不需要悍跳",
            prompt_injectable=_cap_prompt_injectable(
                "悍跳建议：已有队友占据预言家身份，你不需要悍跳。"
                "配合TA的预言家身份进行站边和推人即可。"
            ),
        )

    risks: list[str] = ["悍跳风险：如果对跳方是真预言家，可信度会大幅下降"]

    if day > 2:
        risks.append("晚期悍跳风险更高：已发言轮次多，矛盾点容易被抓")

    if seer_count == 0:
        conf = 0.75 if day <= 1 else 0.55
        prompt = (
            f"悍跳建议：场上无人跳预言家（单边），悍跳窗口极佳。"
            f"建议立即跳预言家并报出假查验结果，同时声明警徽流。"
        )
        speech = ["报假查验结果", "声明完整警徽流", "攻击对立面逻辑"]
    elif seer_count == 1:
        claimant = _get_seer_claimants(ws)[0]
        conf = 0.55 if day <= 1 else 0.3
        prompt = (
            f"悍跳建议：场上已有 {claimant} 跳预言家，悍跳将形成对跳。"
            f"必须准备完整的假验人时间线来对抗。找出 {claimant} 发言的漏洞。"
        )
        speech = [f"指出{claimant}的发言漏洞", "报出完整假验人时间线", "声明警徽流对比"]
        risks.append(f"对跳 {claimant} 需要时间线高度一致，任何矛盾都会暴露")
    else:
        claimants = _get_seer_claimants(ws)
        conf = 0.3
        prompt = (
            f"悍跳建议：场上已有多人({', '.join(claimants)})跳预言家，"
            f"继续悍跳会导致多方混战，建议转为深水或倒钩策略。"
        )
        speech = ["暂不悍跳", "保持中立发言", "观察对跳结果"]
        risks.append("多方混战中悍跳极易被识破")

    if len(wolves) <= 2:
        risks.append(f"存活狼人仅{len(wolves)}人，悍跳失败代价极高")

    return SkillOutput(
        skill_name=skill.name.value,
        speech_structure=speech,
        risk_alerts=risks,
        confidence=conf,
        reasoning="动态分析：根据场上预言家声明情况调整悍跳策略",
        prompt_injectable=_cap_prompt_injectable(prompt),
    )

@register_handler(SkillName.COUNTER_CLAIM)
def counter_claim_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    # S-10: branch on `inp.role`. A REAL seer countering a fake seer
    # needs to defend their check result and rip the faker's timeline.
    # A WOLF countering a real seer (i.e. doing the "悍跳" pair-up)
    # needs to fabricate a matching timeline and steer the room.
    is_seer = inp.role == "seer"
    is_wolf = inp.role == "werewolf"
    if gs is None:
        # static fallback — role-tailored phrasing
        if is_seer:
            risks = ["对跳时一定要保持自己验人时间线的一致性"]
            prompt = (
                "对跳建议（真预言家视角）：用你的真实查验结果逐条对比对方的"
                "假时间线。任何不匹配都是暴露对方假预言家的机会。"
                "重点攻击验人动机、警徽流矛盾。"
            )
            return SkillOutput(
                skill_name=skill.name.value,
                speech_structure=["展示自己的真查验结果", "攻击对方时间线漏洞", "对比警徽流"],
                risk_alerts=risks,
                confidence=0.6,
                reasoning="真预言家对跳：核心是维护自己的查验时间线",
                prompt_injectable=_cap_prompt_injectable(prompt),
                advice_frame=_counter_claim_advice_frame(
                    inp,
                    skill,
                    prompt=prompt,
                    risks=risks,
                    confidence=0.6,
                ),
            )
        if is_wolf:
            risks = ["悍跳风险：如果对方是真预言家，可信度会大幅下降"]
            prompt = (
                "对跳建议（狼队悍跳视角）：你作为狼的悍跳者，需要准备完整的"
                "假验人时间线来对跳真预言家。重点攻击对方的验人动机和警徽流漏洞，"
                "并用排坑占边把节奏拉到自己这边。"
            )
            return SkillOutput(
                skill_name=skill.name.value,
                speech_structure=["准备完整的假验人记录", "攻击真预言家的逻辑漏洞", "排坑占边"],
                risk_alerts=risks,
                confidence=0.55,
                reasoning="悍跳对跳：核心是构建与队友一致的假时间线",
                prompt_injectable=_cap_prompt_injectable(prompt),
                advice_frame=_counter_claim_advice_frame(
                    inp,
                    skill,
                    prompt=prompt,
                    risks=risks,
                    confidence=0.55,
                ),
            )
        # Other roles (villager/hunter/witch/...) — neutral counter-claim.
        risks = ["对跳风险：真预言家对跳时好人会倾向真预言家"]
        prompt = "对跳建议：如果有人跳预言家，准备好完整的假验人记录来对跳。重点攻击对方的验人时间线和警徽流漏洞。"
        return SkillOutput(
            skill_name=skill.name.value,
            speech_structure=["指出对方漏洞", "报自身查验信息", "建立时间线对比"],
            risk_alerts=risks,
            confidence=0.55,
            reasoning="对跳需要充分的逻辑支撑和时间线一致性",
            prompt_injectable=_cap_prompt_injectable(prompt),
            advice_frame=_counter_claim_advice_frame(
                inp,
                skill,
                prompt=prompt,
                risks=risks,
                confidence=0.55,
            ),
        )
    # dynamic analysis
    ws = inp.world_state
    claimants = _get_seer_claimants(ws)
    alerts = inp.contradiction_alerts

    if not claimants:
        prompt = "对跳分析：场上无人跳预言家，无需对跳，保持观察并保留发言弹性。"
        return SkillOutput(
            skill_name=skill.name.value,
            confidence=0.35,
            reasoning="场上无人跳预言家，无需对跳",
            prompt_injectable=_cap_prompt_injectable(prompt),
            advice_frame=_counter_claim_advice_frame(
                inp,
                skill,
                prompt=prompt,
                risks=[],
                confidence=0.35,
            ),
        )

    target = claimants[0]
    target_alerts = _alerts_for_player(alerts, target)
    has_claim_conflict = any(a.alert_type == "claim_conflict" for a in target_alerts)

    # S-10: role-tailored dynamic prompt
    if is_seer:
        # Real seer: defend own timeline, rip faker's inconsistencies.
        if has_claim_conflict:
            prompt = (
                f"对跳分析（真预言家视角）：{target} 的发言存在矛盾点（claim_conflict）。"
                f"用你真实的查验结果逐条对照对方的假时间线，"
                f"任何不匹配都是暴露对方假预言家的机会。"
            )
            conf = 0.7
            speech = [
                f"展示你的真实查验结果对比{target}的假时间线",
                f"指出{target}验人时间线上的具体矛盾",
                "建立你自己的完整时间线作为锚点",
            ]
        else:
            prompt = (
                f"对跳分析（真预言家视角）：{target} 的发言看起来较一致。"
                f"重点从侧面找漏洞：验人动机、警徽流合理性、站边逻辑。"
                f"如果对方是悍跳者，时间线上一定会有逻辑漏洞。"
            )
            conf = 0.55
            speech = [
                f"质疑{target}的验人动机",
                f"分析{target}的警徽流是否合理",
                "用你自己的金水作为锚点",
            ]
        risks = [f"对跳 {target} 需保持查验时间线一致"]
    elif is_wolf:
        # Wolf (fake seer / counter-claimer): fabricate, attack, herd.
        if has_claim_conflict:
            prompt = (
                f"对跳分析（悍跳视角）：{target} 的发言已经暴露矛盾（claim_conflict）。"
                f"抓住这个矛盾猛攻，把场上风向拉到对你有利的方向。"
                f"用你准备的假验人时间线作为正面证据。"
            )
            conf = 0.65
            speech = [
                f"放大{target}的发言矛盾",
                "用你准备好的假时间线作为正面证据",
                "排坑占边把节奏拉到自己这边",
            ]
        else:
            prompt = (
                f"对跳分析（悍跳视角）：{target} 的发言暂时没明显漏洞。"
                f"不要正面硬刚，从侧面找漏洞（验人动机、警徽流合理性）。"
                f"如果真预言家时间线无漏洞，转为排坑占边策略。"
            )
            conf = 0.45
            speech = [
                f"侧面质疑{target}的验人动机",
                "准备完整的假时间线作为预案",
                "用排坑占边为团队创造空间",
            ]
        risks = [f"悍跳 {target} 需要完整的假验人记录，时间线断裂会暴露"]
    else:
        # Other roles (villager/...) — neutral dynamic advice.
        if has_claim_conflict:
            prompt = (
                f"对跳分析：{target} 的发言存在矛盾点（claim_conflict）。"
                f"建议集中攻击以下疑点，质疑其预言家身份的真实性。"
            )
            conf = 0.65
            speech = [f"指出{target}验人时间线的矛盾", f"对比{target}前后不一致的发言", "建立自己的完整时间线"]
        else:
            prompt = (
                f"对跳分析：{target} 的发言和验人时间线较一致，直接对跳风险较高。"
                f"建议从侧面寻找漏洞：验人动机、警徽流合理性、站边逻辑。"
            )
            conf = 0.45
            speech = [f"质疑{target}的验人动机", f"分析{target}的警徽流是否合理", "找侧面漏洞而非正面硬刚"]
        risks = [f"对跳 {target} 需要完整的假验人记录，任何时间线断裂都会暴露"]

    return SkillOutput(
        skill_name=skill.name.value,
        speech_structure=speech,
        risk_alerts=risks,
        confidence=conf,
        reasoning=f"动态分析（S-10: {inp.role} 对跳）：根据{target}的发言一致性调整策略",
        prompt_injectable=_cap_prompt_injectable(prompt),
        advice_frame=_counter_claim_advice_frame(
            inp,
            skill,
            prompt=prompt,
            risks=risks,
            confidence=conf,
            target=target,
        ),
    )
