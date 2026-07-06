# -*- coding: utf-8 -*-
"""
实现狼人侧技能 handler。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.skills.wolf_skill_handlers import deep_hook_handler
"""

from __future__ import annotations

from werewolf_agent.skills.advice_frames import (
    _cap_prompt_injectable,
    _hide_identity_advice_frame,
)
from werewolf_agent.skills.schemas import SkillDefinition, SkillInput, SkillName, SkillOutput
from werewolf_agent.skills.skill_context import (
    _alerts_for_player,
    _alive_wolves,
    _wolf_teammates_exposed,
)
from werewolf_agent.skills.skill_handler_registry import register_handler


@register_handler(SkillName.DEEP_HOOK)
def deep_hook_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    if gs is None:
        # static fallback
        return SkillOutput(
            skill_name=skill.name.value,
            speech_structure=["站边好人逻辑", "适度攻击可疑队友", "建立可信度"],
            risk_alerts=["倒钩策略需要长期维持一致性", "过度攻击队友可能被识别"],
            confidence=0.55,
            reasoning="倒钩核心是在好人阵营中建立长期可信度",
            prompt_injectable=_cap_prompt_injectable("倒钩建议：伪装成好人，站边好人逻辑线，适度攻击被怀疑的队友来建立信任。注意保持发言一致性，不要前后矛盾。"),
        )
    # dynamic analysis
    ws = inp.world_state

    wolves = _alive_wolves(gs)
    exposed = _wolf_teammates_exposed(ws, wolves)

    # If wolf role assignment is pusher or fake_seer, deep hook doesn't apply
    wolf_plan = inp.extra.get("wolf_team_plan") if inp.extra else None
    if wolf_plan:
        my_role_key = None
        for key in ("fake_seer", "pusher", "hooker", "deep_cover"):
            if wolf_plan.get(key) == inp.player_id:
                my_role_key = key
                break
        if my_role_key in ("fake_seer", "pusher"):
            return SkillOutput(
                skill_name=skill.name.value,
                confidence=0.3,
                reasoning=f"你是{my_role_key}，不需要倒钩策略",
                prompt_injectable=_cap_prompt_injectable("倒钩建议：你的角色分工是冲锋/悍跳，不需要倒钩。专注于你的主要任务。"),
            )

    day = gs.day_number
    risks = ["倒钩策略需要长期维持一致性", "过度攻击队友可能被识别"]

    if exposed:
        teammate = exposed[0]["teammate"]
        checker = exposed[0]["checked_by"]
        prompt = (
            f"倒钩建议：队友 {teammate} 已被 {checker} 查杀暴露。"
            f"建议适度攻击 {teammate} 来获取好人信任——但不要太用力，保持可信度。"
            f"可以用'我也觉得{teammate}可疑'的方式自然切入。"
        )
        conf = 0.65
        speech = [
            f"适度质疑{teammate}的发言",
            "站边好人逻辑但不过激",
            "贡献独立分析建立可信度",
        ]
    else:
        prompt = (
            f"倒钩建议：场上狼队暂无暴露，继续保持深水伪装。"
            f"保持中立发言，偶尔贡献独立分析，不要主动引起关注。"
            f"等待合适的时机（如队友暴露后）再适度切割。"
        )
        conf = 0.55
        speech = ["保持中立发言节奏", "贡献独立分析", "避免过度暴露信息优势"]

    if day > 3:
        risks.append("已过Day 3，倒钩需更加谨慎——好人的分析会越来越细")
        conf = max(0.4, conf - 0.1)

    return SkillOutput(
        skill_name=skill.name.value,
        speech_structure=speech,
        risk_alerts=risks,
        confidence=conf,
        reasoning="动态分析：根据队友暴露状态调整倒钩策略",
        prompt_injectable=_cap_prompt_injectable(prompt),
    )

@register_handler(SkillName.HIDE_IDENTITY)
def hide_identity_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    if inp.role == "seer":
        prompt = (
            "藏身份提示（预言家视角）：预言家不应默认使用藏身份。"
            "如果已有验人结果，或处于警上、对跳、PK、后置发言、需要带队等公开窗口，"
            "必须准确公开身份与真实验人结果。"
            "只有在尚未进入公开窗口时，才可以短暂避免暴露验人动机，"
            "但不能用藏身份覆盖报验人和带队的硬要求。"
        )
        return SkillOutput(
            skill_name=skill.name.value,
            speech_structure=["准确公开验人", "说明信息来源", "按局势带队"],
            risk_alerts=[
                "藏身份会延误真实验人信息，可能污染好人视角",
                "公开窗口继续隐藏会削弱预言家带队价值",
            ],
            confidence=0.25,
            reasoning="预言家以真实验人信息输出为主，藏身份只能是短暂信息控制",
            prompt_injectable=_cap_prompt_injectable(prompt),
            advice_frame=_hide_identity_advice_frame(
                inp,
                skill,
                prompt=prompt,
                risks=[
                    "藏身份会延误真实验人信息，可能污染好人视角",
                    "公开窗口继续隐藏会削弱预言家带队价值",
                ],
                confidence=0.25,
                exposure="seer_public_window",
            ),
        )

    gs = inp.game_state
    if gs is None:
        # NEW-R4-P2-5: role-tailored static fallback. Witch hides 药剂 /
        # 救人时机; wolf hides 夜杀 / 队友 / 夜间信息. Villagers (and
        # other roles with no night info to leak) keep the generic advice.
        risks = ["藏身份过久可能导致无法在关键时刻发挥作用"]
        if inp.role == "witch":
            prompt = (
                "藏身份建议（女巫视角）：不要暴露你的药剂状态——"
                "既不要让人知道解药是否已用，也不要暗示毒药还在。"
                "发言中避免讨论'该救谁'或'该毒谁'。"
                "保持低调，让狼队无法锁定你的身份。"
            )
        elif inp.role == "werewolf":
            prompt = (
                "藏身份建议（狼队视角）：不要暴露任何夜间信息——"
                "夜杀目标、队友配合、悍跳分工都属高度机密。"
                "发言中避免提及'昨晚'、'夜里'、'队友'等暗示性词汇。"
                "用归票和站边制造好人之间的内讧来掩盖狼队身份。"
            )
        else:
            # Villager / hunter / idiot / hybrid / unknown — generic
            # advice (villagers have no night info to leak).
            prompt = (
                "藏身份建议：发言保持中立，不要暴露你知道的夜晚信息。"
                "如果被质疑，适度释放信息自证但不要全露底牌。"
            )
        return SkillOutput(
            skill_name=skill.name.value,
            speech_structure=["保持中立发言", "避免暴露信息优势", "控制发言节奏"],
            risk_alerts=risks,
            confidence=0.6,
            reasoning="藏身份需要在隐匿和发挥作用之间找到平衡",
            prompt_injectable=_cap_prompt_injectable(prompt),
            advice_frame=_hide_identity_advice_frame(
                inp,
                skill,
                prompt=prompt,
                risks=risks,
                confidence=0.6,
                exposure="static_fallback",
            ),
        )
    # dynamic analysis
    ws = inp.world_state
    pid = inp.player_id
    alerts = inp.contradiction_alerts
    day = gs.day_number

    my_claims = []
    if ws is not None:
        for f in ws.facts_of_type("claimed_role"):
            if f.source_player == pid:
                my_claims.append(f.value)

    my_alerts = _alerts_for_player(alerts, pid)
    under_pressure = len(my_alerts) > 0

    risks = ["藏身份过久可能导致无法在关键时刻发挥作用"]

    if my_claims:
        prompt = (
            f"藏身份建议：你已公开声明身份（{', '.join(my_claims)}），"
            f"身份已不完全隐蔽。建议继续维持已声明的人设，保持一致性。"
        )
        conf = 0.4
        exposure = "claimed_role"
    elif under_pressure:
        pressure_desc = "; ".join(a.description for a in my_alerts[:2])
        prompt = (
            f"藏身份建议：你正被怀疑（{pressure_desc}），"
            f"需要适度释放信息来自证，但不要完全暴露身份。"
            f"可以给出部分信息来降低怀疑，同时保留核心身份信息。"
        )
        conf = 0.55
        risks.append("被怀疑时过度隐蔽反而加深嫌疑")
        exposure = "under_pressure"
    else:
        prompt = (
            f"藏身份建议：你目前身份隐蔽状态良好，没有公开声明也没有被重点怀疑。"
            f"继续保持中立发言节奏，避免暴露信息优势。"
        )
        conf = 0.65
        exposure = "hidden"

    if day > 3:
        risks.append("已过Day 3，继续藏身份可能导致无法在关键时刻发挥作用")
        conf = max(0.4, conf - 0.1)

    return SkillOutput(
        skill_name=skill.name.value,
        speech_structure=["保持中立发言", "避免暴露信息优势", "控制发言节奏"],
        risk_alerts=risks,
        confidence=conf,
        reasoning="动态分析：根据自身暴露状态和被怀疑程度调整策略",
        prompt_injectable=_cap_prompt_injectable(prompt),
        advice_frame=_hide_identity_advice_frame(
            inp,
            skill,
            prompt=prompt,
            risks=risks,
            confidence=conf,
            exposure=exposure,
        ),
    )

@register_handler(SkillName.WOLF_PIT_ANALYSIS)
def wolf_pit_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    if gs is None:
        # NEW-R4-P2-7: when no game_state is provided, the handler
        # has no suspects/excludes to analyze. The previous
        # fallback used abstract "系统性分析" advice that gave the
        # LLM no concrete next step. Replace with an explicit
        # "wait" — the dynamic branch is the real value-add; the
        # fallback is a placeholder.
        return SkillOutput(
            skill_name=skill.name.value,
            speech_structure=["列出嫌疑人", "分析各嫌疑人证据", "排除法缩小范围"],
            confidence=0.5,
            reasoning="盘狼坑需要系统性分析所有嫌疑人的行为链",
            prompt_injectable=_cap_prompt_injectable(
                "盘狼坑建议：当前信息不足，等待关键发言出现后再下判断。"
                "重点关注发言矛盾、投票链异常、验人冲突等维度，"
                "用排除法缩小狼坑范围。"
            ),
        )
    # dynamic analysis
    ws = inp.world_state
    bs = inp.belief_state

    # Build suspect and exclude lists
    suspects: list[tuple[str, str]] = []   # (player, reason)
    excluded: list[tuple[str, str]] = []   # (player, reason)

    alive_count = sum(1 for p in gs.players.values() if p.alive)

    # From belief state
    if bs is not None:
        for pid, belief in bs.beliefs.items():
            player = gs.players.get(pid)
            if not player or not player.alive:
                continue
            if belief.faction_lean == "wolf_lean":
                suspects.append((pid, f"行为偏向狼人(信任{belief.trust:.0%})"))
            elif belief.faction_lean == "good_lean" and belief.trust > 0.7:
                excluded.append((pid, f"行为像好人(信任{belief.trust:.0%})"))

    # From seer checks
    if ws is not None:
        for fact_type in ("seer_check_claim",):
            for f in ws.facts_of_type(fact_type):
                target = f.target_player
                val = (f.value or "").lower()
                # NEW-S19-B: skip dead players. A dead player with a
                # seer_check_claim would be added to suspects/excluded
                # and then dropped by the S-19 filter (or worse, the
                # prompt would carry an "illegal" target). Mirror the
                # belief-state loop above which already guards on
                # `player.alive`.
                target_player = gs.players.get(target) if target else None
                if not target_player or not target_player.alive:
                    continue
                if target and ("wolf" in val or "狼" in (f.value or "")):
                    suspects.append((target, f"被{f.source_player}查杀"))
                elif target and ("good" in val or "金水" in (f.value or "")):
                    excluded.append((target, f"被{f.source_player}发金水"))

    # Deduplicate
    suspect_ids = set()
    unique_suspects = []
    for pid, reason in suspects:
        if pid not in suspect_ids:
            suspect_ids.add(pid)
            unique_suspects.append((pid, reason))

    exclude_ids = set()
    unique_excluded = []
    for pid, reason in excluded:
        if pid not in exclude_ids:
            exclude_ids.add(pid)
            unique_excluded.append((pid, reason))

    suspect_lines = [f"{pid}({reason})" for pid, reason in unique_suspects[:5]]
    exclude_lines = [f"{pid}({reason})" for pid, reason in unique_excluded[:5]]

    # NEW-R4-P2-4: when the visible list is a slice of the full
    # deduped list, surface the (shown/total) count so the LLM has
    # one truncation signal, not two (the second one being the
    # `...（已省略）` from `_cap_prompt_injectable`). When nothing
    # was truncated, the bare `N人` form is fine.
    def _count_label(shown: int, total: int) -> str:
        if shown == total:
            return f"({total}人)"
        return f"({shown}/{total}人)"

    suspect_count = _count_label(len(suspect_lines), len(unique_suspects))
    exclude_count = _count_label(len(exclude_lines), len(unique_excluded))

    prompt = (
        f"盘狼坑分析：当前存活{alive_count}人中，"
        f"嫌疑区{suspect_count}：{'；'.join(suspect_lines) if suspect_lines else '暂无明确嫌疑人'}。"
        f"排除区{exclude_count}：{'；'.join(exclude_lines) if exclude_lines else '暂无排除'}。"
        f"需要继续关注投票链和发言一致性来缩小范围。"
    )

    return SkillOutput(
        skill_name=skill.name.value,
        speech_structure=["列出嫌疑人及其证据", "分析排除区", "缩小嫌疑范围"],
        confidence=0.5 + min(0.2, len(unique_suspects) * 0.05),
        reasoning=f"动态分析：{len(unique_suspects)}个嫌疑人，{len(unique_excluded)}个排除",
        prompt_injectable=_cap_prompt_injectable(prompt),
    )
