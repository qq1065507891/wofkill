"""Werewolf skill definitions: 12 core gameplay skills per design doc §11.1.

Each skill provides a deterministic, game-state-aware tactical suggestion.
When game_state is provided, handlers analyze real signals and produce
actionable Chinese-language advice. When game_state is None, handlers fall
back to static output for backward compatibility.
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.skills.schemas import (
    SkillDefinition,
    SkillFaction,
    SkillInput,
    SkillName,
    SkillOutput,
)


# ---------------------------------------------------------------------------
# 12 core werewolf skills
# ---------------------------------------------------------------------------

SKILL_DEFINITIONS: list[SkillDefinition] = [
    SkillDefinition(
        name=SkillName.BOLD_CLAIM,
        display_name="悍跳",
        description="冒充神职角色，通过假查验或假身份获取信任和话语权",
        applicable_roles=["werewolf"],
        applicable_phases=["speech", "sheriff_speech", "pk_speech", "wolf_discussion"],
        faction=SkillFaction.WOLF,
        tags=["deception", "aggressive"],
    ),
    SkillDefinition(
        name=SkillName.COUNTER_CLAIM,
        display_name="对跳",
        description="针对已经起跳的玩家进行身份对跳，争夺话语权",
        applicable_roles=["seer", "werewolf"],
        applicable_phases=["speech", "pk_speech"],
        faction=SkillFaction.COMMON,
        tags=["confrontation", "identity_claim"],
    ),
    SkillDefinition(
        name=SkillName.PUSH_VOTE,
        display_name="归票",
        description="引导全场投票方向，集中票数归出目标玩家",
        applicable_roles=["seer", "werewolf", "villager", "hunter", "witch", "idiot", "hybrid"],
        applicable_phases=["speech", "vote"],
        faction=SkillFaction.COMMON,
        tags=["leadership", "voting"],
    ),
    SkillDefinition(
        name=SkillName.SWING_VOTE,
        display_name="冲票",
        description="集中阵营力量冲票特定目标，多用于狼队协同冲票",
        applicable_roles=["werewolf"],
        applicable_phases=["vote", "wolf_discussion"],
        faction=SkillFaction.WOLF,
        tags=["coordination", "aggressive"],
    ),
    SkillDefinition(
        name=SkillName.DEEP_HOOK,
        display_name="倒钩",
        description="在好人阵营中建立可信度，通过适度攻击狼队友来获取信任",
        applicable_roles=["werewolf"],
        applicable_phases=["speech", "vote", "wolf_discussion"],
        faction=SkillFaction.WOLF,
        tags=["deception", "long_term"],
    ),
    SkillDefinition(
        name=SkillName.FIND_POWER,
        display_name="找神",
        description="通过发言和行为模式分析找出神职玩家",
        applicable_roles=["werewolf", "villager", "hybrid"],
        applicable_phases=["speech", "night_action", "wolf_discussion"],
        faction=SkillFaction.COMMON,
        tags=["analysis", "information"],
    ),
    SkillDefinition(
        name=SkillName.HIDE_IDENTITY,
        display_name="藏身份",
        description="隐藏自己的真实角色，避免被过早识别",
        applicable_roles=["seer", "witch", "hunter", "werewolf", "hybrid"],
        applicable_phases=["speech", "sheriff_speech", "sheriff_registration"],
        faction=SkillFaction.COMMON,
        tags=["stealth", "defense"],
    ),
    SkillDefinition(
        name=SkillName.RESIST_PUSH,
        display_name="抗推",
        description="在被怀疑或被推票时进行有效防守和反驳",
        applicable_roles=["werewolf", "villager", "seer", "witch", "hunter", "idiot", "hybrid"],
        applicable_phases=["defense_speech", "pk_speech"],
        faction=SkillFaction.COMMON,
        tags=["defense", "persuasion"],
    ),
    SkillDefinition(
        name=SkillName.WOLF_PIT_ANALYSIS,
        display_name="盘狼坑",
        description="系统性分析可能的狼人分布，缩小嫌疑范围",
        applicable_roles=["villager", "seer", "witch", "hunter", "idiot", "hybrid"],
        applicable_phases=["speech", "sheriff_speech"],
        faction=SkillFaction.GOOD,
        tags=["analysis", "logic"],
    ),
    SkillDefinition(
        name=SkillName.PROTECT_POWER,
        display_name="保护强神",
        description="保护关键神职角色不被狼人发现或冲票",
        applicable_roles=["villager", "seer", "witch", "hunter", "idiot", "hybrid"],
        applicable_phases=["speech", "vote"],
        faction=SkillFaction.GOOD,
        tags=["protection", "team_play"],
    ),
    SkillDefinition(
        name=SkillName.LAST_WORDS_ANALYSIS,
        display_name="遗言分析",
        description="分析遗言内容，提取信息，判断发言者真实身份",
        applicable_roles=["villager", "seer", "witch", "hunter", "idiot", "hybrid", "werewolf"],
        applicable_phases=["speech"],
        faction=SkillFaction.COMMON,
        tags=["analysis", "information"],
    ),
    SkillDefinition(
        name=SkillName.REVIEW_CORRECTION,
        display_name="复盘纠错",
        description="复盘本局关键判断，识别错误，提出改进建议",
        applicable_roles=["villager", "seer", "witch", "hunter", "idiot", "hybrid", "werewolf"],
        applicable_phases=["review"],
        faction=SkillFaction.UNIVERSAL,
        tags=["review", "improvement"],
    ),
]


# ---------------------------------------------------------------------------
# Shared helpers for game-state-aware analysis
# ---------------------------------------------------------------------------

def _count_seer_claimants(ws: Any) -> int:
    """Count distinct players who publicly claimed seer."""
    if ws is None:
        return 0
    claimants: set[str] = set()
    for f in ws.facts_of_type("claimed_role"):
        if f.value == "seer" and f.source_player:
            claimants.add(f.source_player)
    return len(claimants)


def _get_seer_claimants(ws: Any) -> list[str]:
    """Return list of players who publicly claimed seer."""
    if ws is None:
        return []
    claimants: set[str] = set()
    for f in ws.facts_of_type("claimed_role"):
        if f.value == "seer" and f.source_player:
            claimants.add(f.source_player)
    return sorted(claimants)


def _alive_wolves(gs: Any) -> list[str]:
    """Return alive wolf teammates."""
    if gs is None:
        return []
    return [
        pid for pid, p in gs.players.items()
        if p.alive and p.role == "werewolf"
    ]


def _alive_non_wolves(gs: Any) -> list[str]:
    """Return alive non-wolf players."""
    if gs is None:
        return []
    return [
        pid for pid, p in gs.players.items()
        if p.alive and p.role != "werewolf"
    ]


def _vote_targets_for_player(ws: Any, player_id: str) -> list[dict[str, Any]]:
    """Get vote facts targeting a specific player."""
    if ws is None:
        return []
    return [
        {"source": f.source_player, "day": f.day, "value": f.value}
        for f in ws.facts_of_type("vote")
        if f.target_player == player_id
    ]


def _seer_checks_on_target(ws: Any, target_id: str) -> list[dict[str, Any]]:
    """Get seer check claims targeting a specific player."""
    if ws is None:
        return []
    results = []
    for f in ws.facts_of_type("seer_check_claim"):
        if f.target_player == target_id:
            results.append({"source": f.source_player, "value": f.value, "day": f.day})
    for f in ws.facts_of_type("seer_check"):
        if f.target_player == target_id:
            results.append({"source": f.source_player, "value": f.value, "day": f.day or f.night})
    return results


def _alerts_for_player(alerts: list[Any], player_id: str) -> list[Any]:
    """Filter contradiction alerts that mention the player."""
    return [
        a for a in alerts
        if player_id in a.player_id
    ]


def _belief_top_suspects(bs: Any, count: int = 3) -> list[tuple[str, str, float]]:
    """Return top suspects from belief state (wolf_lean, lowest trust)."""
    if bs is None:
        return []
    suspects: list[tuple[str, str, float]] = []
    for pid, belief in bs.beliefs.items():
        if belief.faction_lean == "wolf_lean" or belief.trust < 0.35:
            suspects.append((pid, belief.faction_lean, belief.trust))
    suspects.sort(key=lambda x: x[2])
    return suspects[:count]


def _wolf_teammates_exposed(ws: Any, wolf_ids: list[str]) -> list[dict[str, Any]]:
    """Check which wolf teammates have been publicly seer-checked as wolf."""
    if ws is None:
        return []
    exposed = []
    for wid in wolf_ids:
        checks = _seer_checks_on_target(ws, wid)
        for c in checks:
            if "wolf" in c.get("value", "").lower() or "狼" in c.get("value", ""):
                exposed.append({"teammate": wid, "checked_by": c["source"]})
    return exposed


# ---------------------------------------------------------------------------
# Skill dispatch
# ---------------------------------------------------------------------------

def apply_skill(skill_name: SkillName, skill_input: SkillInput) -> SkillOutput:
    """Apply a skill to generate a tactical suggestion."""
    skill_def = _find_definition(skill_name)
    if skill_def is None:
        return SkillOutput(
            skill_name=skill_name.value,
            confidence=0.0,
            risk_alerts=["未知技能"],
        )

    handler = _SKILL_HANDLERS.get(skill_name, _default_handler)
    return handler(skill_input, skill_def)


def _find_definition(name: SkillName) -> SkillDefinition | None:
    for s in SKILL_DEFINITIONS:
        if s.name == name:
            return s
    return None


# ---------------------------------------------------------------------------
# Handlers — each has a static fallback and a dynamic branch
# ---------------------------------------------------------------------------

def _default_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="analyze",
        confidence=0.5,
        reasoning=f"技能 {skill.display_name} 适用，需要更多局势信息",
    )


# --- BOLD_CLAIM (悍跳) ---

def _bold_claim_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    if gs is None:
        return _bold_claim_static(inp, skill)
    return _bold_claim_dynamic(inp, skill)


def _bold_claim_static(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    risks = ["悍跳风险：如果对跳方是真预言家，可信度会大幅下降"]
    if inp.day > 2:
        risks.append("晚期悍跳风险更高：已发言轮次多，矛盾点容易被抓")
    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="claim_role",
        speech_structure=["报查验结果", "声明警徽流", "攻击对立面逻辑"],
        risk_alerts=risks,
        confidence=0.6 if inp.day <= 1 else 0.3,
        reasoning="悍跳需要前期执行，后期风险增大",
    )


def _bold_claim_dynamic(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    ws = inp.world_state
    gs = inp.game_state
    day = gs.day_number
    seer_count = _count_seer_claimants(ws)
    wolves = _alive_wolves(gs)

    # If wolf_team_plan assigns a different wolf as fake_seer, skip bold claim advice
    wolf_plan = inp.extra.get("wolf_team_plan") if inp.extra else None
    if wolf_plan and wolf_plan.get("fake_seer") and wolf_plan["fake_seer"] != inp.player_id:
        return SkillOutput(
            skill_name=skill.name.value,
            recommended_action="speech",
            confidence=0.3,
            reasoning=f"队友 {wolf_plan['fake_seer']} 负责悍跳，你不需要悍跳",
            prompt_injectable=f"悍跳建议：你的队友 {wolf_plan['fake_seer']} 负责悍跳预言家，"
                              f"你不需要悍跳。配合TA的预言家身份进行站边和推人。",
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
        recommended_action="claim_role" if seer_count <= 1 else "speech",
        speech_structure=speech,
        risk_alerts=risks,
        confidence=conf,
        reasoning="动态分析：根据场上预言家声明情况调整悍跳策略",
        prompt_injectable=prompt,
    )


# --- COUNTER_CLAIM (对跳) ---

def _counter_claim_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    if gs is None:
        return _counter_claim_static(inp, skill)
    return _counter_claim_dynamic(inp, skill)


def _counter_claim_static(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    risks = ["对跳风险：真预言家对跳时好人会倾向真预言家"]
    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="counter_claim",
        speech_structure=["指出对方漏洞", "报自身查验信息", "建立时间线对比"],
        risk_alerts=risks,
        confidence=0.55,
        reasoning="对跳需要充分的逻辑支撑和时间线一致性",
    )


def _counter_claim_dynamic(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    ws = inp.world_state
    claimants = _get_seer_claimants(ws)
    alerts = inp.contradiction_alerts

    if not claimants:
        return SkillOutput(
            skill_name=skill.name.value,
            recommended_action="speech",
            confidence=0.35,
            reasoning="场上无人跳预言家，无需对跳",
        )

    target = claimants[0]
    target_alerts = _alerts_for_player(alerts, target)
    has_claim_conflict = any(a.alert_type == "claim_conflict" for a in target_alerts)

    if has_claim_conflict:
        prompt = (
            f"对跳分析：{target} 的发言存在矛盾点（已检测到claim_conflict）。"
            f"建议集中攻击以下疑点，质疑其预言家身份的真实性。"
        )
        conf = 0.65
        speech = [f"指出{target}验人时间线的矛盾", "对比{target}前后不一致的发言", "建立自己的完整时间线"]
    else:
        prompt = (
            f"对跳分析：{target} 的发言和验人时间线较一致，直接对跳风险较高。"
            f"建议从侧面寻找漏洞：验人动机、警徽流合理性、站边逻辑。"
        )
        conf = 0.45
        speech = [f"质疑{target}的验人动机", "分析{target}的警徽流是否合理", "找侧面漏洞而非正面硬刚"]

    risks = [f"对跳 {target} 需要完整的假验人记录，任何时间线断裂都会暴露"]

    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="counter_claim",
        recommended_target=target,
        speech_structure=speech,
        risk_alerts=risks,
        confidence=conf,
        reasoning=f"动态分析：根据{target}的发言一致性调整对跳策略",
        prompt_injectable=prompt,
    )


# --- PUSH_VOTE (归票) ---

def _push_vote_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    if gs is None:
        return _push_vote_static(inp, skill)
    return _push_vote_dynamic(inp, skill)


def _push_vote_static(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="vote",
        speech_structure=["陈述归票理由", "分析目标嫌疑", "号召全场归票"],
        risk_alerts=["归票错误目标可能导致好人损失"],
        confidence=0.6,
        reasoning="归票需要有充分的逻辑依据和说服力",
    )


def _push_vote_dynamic(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    ws = inp.world_state
    bs = inp.belief_state
    gs = inp.game_state

    top_suspects = _belief_top_suspects(bs, count=3)
    day = gs.day_number

    if not top_suspects:
        return SkillOutput(
            skill_name=skill.name.value,
            recommended_action="vote",
            confidence=0.4,
            reasoning="当前无明确嫌疑目标",
            prompt_injectable="归票建议：当前信息不足，建议观察发言后再决定归票方向。",
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
    prompt = (
        f"归票建议：根据场上信息，{primary} 的嫌疑最大。"
        f"理由：{reason_text}。号召全场集中票数归出 {primary}。"
    )

    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="vote",
        recommended_target=primary,
        speech_structure=[f"陈述{primary}的嫌疑理由", "分析其行为链", "号召全场归票"],
        risk_alerts=["归票错误目标可能导致好人损失"],
        confidence=0.6 + min(0.2, len(reasons) * 0.05),
        reasoning=f"动态分析：{primary} 有{len(reasons)}个嫌疑信号",
        prompt_injectable=prompt,
    )


# --- SWING_VOTE (冲票) ---

def _swing_vote_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    if gs is None:
        return _swing_vote_static(inp, skill)
    return _swing_vote_dynamic(inp, skill)


def _swing_vote_static(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    risks = ["冲票暴露投票链：好人可能通过投票链锁定狼人"]
    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="vote",
        risk_alerts=risks,
        confidence=0.5,
        reasoning="冲票需要考虑投票链暴露风险",
    )


def _swing_vote_dynamic(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    ws = inp.world_state

    wolves = _alive_wolves(gs)
    non_wolves = _alive_non_wolves(gs)

    if not non_wolves:
        return SkillOutput(
            skill_name=skill.name.value,
            confidence=0.0,
            reasoning="无冲票目标",
        )

    # Find non-wolf with most vote pressure
    pressure: dict[str, int] = {pid: 0 for pid in non_wolves}
    for f in (ws.facts_of_type("vote") if ws else []):
        if f.target_player in pressure:
            pressure[f.target_player] += 1
    for f in (ws.facts_of_type("claimed_suspect") if ws else []):
        if f.target_player in pressure:
            pressure[f.target_player] += 2

    best_target = max(non_wolves, key=lambda p: pressure.get(p, 0))
    best_pressure = pressure.get(best_target, 0)

    wolf_count = len(wolves)
    risks = ["冲票暴露投票链：好人可能通过投票链锁定狼人"]
    if wolf_count <= 2:
        risks.append(f"仅剩{wolf_count}狼，冲票暴露风险极高")

    if best_pressure > 0:
        prompt = (
            f"冲票建议：集中狼队票数冲 {best_target}。"
            f"理由：{best_target} 已有{best_pressure}个怀疑信号，冲票成功率高。"
        )
        conf = 0.5 + min(0.2, best_pressure * 0.05)
    else:
        prompt = "冲票建议：当前无明确冲票目标，建议分散投票避免暴露。"
        conf = 0.35

    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="vote",
        recommended_target=best_target if best_pressure > 0 else None,
        risk_alerts=risks,
        confidence=conf,
        reasoning=f"动态分析：{best_target} 有{best_pressure}个压力信号，{wolf_count}狼存活",
        prompt_injectable=prompt,
    )


# --- DEEP_HOOK (倒钩) ---

def _deep_hook_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    if gs is None:
        return _deep_hook_static(inp, skill)
    return _deep_hook_dynamic(inp, skill)


def _deep_hook_static(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="speech",
        speech_structure=["站边好人逻辑", "适度攻击可疑队友", "建立可信度"],
        risk_alerts=["倒钩策略需要长期维持一致性", "过度攻击队友可能被识别"],
        confidence=0.55,
        reasoning="倒钩核心是在好人阵营中建立长期可信度",
    )


def _deep_hook_dynamic(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    ws = inp.world_state
    bs = inp.belief_state

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
                recommended_action="speech",
                confidence=0.3,
                reasoning=f"你是{my_role_key}，不需要倒钩策略",
                prompt_injectable="倒钩建议：你的角色分工是冲锋/悍跳，不需要倒钩。专注于你的主要任务。",
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
        recommended_action="speech",
        speech_structure=speech,
        risk_alerts=risks,
        confidence=conf,
        reasoning="动态分析：根据队友暴露状态调整倒钩策略",
        prompt_injectable=prompt,
    )


# --- FIND_POWER (找神) ---

def _find_power_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    if gs is None:
        return _find_power_static(inp, skill)
    return _find_power_dynamic(inp, skill)


def _find_power_static(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="analyze",
        speech_structure=["分析发言信息量", "观察投票倾向", "识别保护行为"],
        confidence=0.5,
        reasoning="找神需要综合多个信号源进行推断",
    )


def _find_power_dynamic(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    ws = inp.world_state
    bs = inp.belief_state

    power_roles = {"seer", "witch", "hunter"}
    candidates: list[dict[str, Any]] = []

    if bs is not None:
        for pid, belief in bs.beliefs.items():
            for role, prob in belief.role_probabilities.items():
                if role in power_roles and prob > 0.3:
                    candidates.append({
                        "player": pid,
                        "role": role,
                        "probability": prob,
                        "source": "belief",
                    })

    if ws is not None:
        for f in ws.facts_of_type("badge_flow_claim"):
            if f.source_player:
                candidates.append({
                    "player": f.source_player,
                    "role": "seer",
                    "probability": 0.6,
                    "source": "badge_flow",
                })

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for c in sorted(candidates, key=lambda x: x["probability"], reverse=True):
        key = f"{c['player']}_{c['role']}"
        if key not in seen:
            seen.add(key)
            unique.append(c)

    if not unique:
        return SkillOutput(
            skill_name=skill.name.value,
            recommended_action="analyze",
            confidence=0.3,
            reasoning="暂无足够信号推断神职",
            prompt_injectable="找神分析：当前信息不足，建议继续观察发言信息量和投票模式。",
        )

    lines = []
    for c in unique[:3]:
        lines.append(f"{c['player']} 大概率是 {c['role']}（依据：{c['source']}，置信{c['probability']:.0%}）")
    prompt = f"找神分析：{'；'.join(lines)}"

    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="analyze",
        recommended_target=unique[0]["player"] if unique else None,
        speech_structure=["分析发言信息量", "观察投票倾向", "识别保护行为"],
        confidence=0.5 + min(0.2, len(unique) * 0.05),
        reasoning=f"动态分析：识别到{len(unique)}个疑似神职信号",
        prompt_injectable=prompt,
    )


# --- HIDE_IDENTITY (藏身份) ---

def _hide_identity_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    if gs is None:
        return _hide_identity_static(inp, skill)
    return _hide_identity_dynamic(inp, skill)


def _hide_identity_static(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    risks = ["藏身份过久可能导致无法在关键时刻发挥作用"]
    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="speech",
        speech_structure=["保持中立发言", "避免暴露信息优势", "控制发言节奏"],
        risk_alerts=risks,
        confidence=0.6,
        reasoning="藏身份需要在隐匿和发挥作用之间找到平衡",
    )


def _hide_identity_dynamic(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    ws = inp.world_state
    pid = inp.player_id
    alerts = inp.contradiction_alerts
    day = inp.game_state.day_number

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
    elif under_pressure:
        pressure_desc = "; ".join(a.description for a in my_alerts[:2])
        prompt = (
            f"藏身份建议：你正被怀疑（{pressure_desc}），"
            f"需要适度释放信息来自证，但不要完全暴露身份。"
            f"可以给出部分信息来降低怀疑，同时保留核心身份信息。"
        )
        conf = 0.55
        risks.append("被怀疑时过度隐蔽反而加深嫌疑")
    else:
        prompt = (
            f"藏身份建议：你目前身份隐蔽状态良好，没有公开声明也没有被重点怀疑。"
            f"继续保持中立发言节奏，避免暴露信息优势。"
        )
        conf = 0.65

    if day > 3:
        risks.append("已过Day 3，继续藏身份可能导致无法在关键时刻发挥作用")
        conf = max(0.4, conf - 0.1)

    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="speech",
        speech_structure=["保持中立发言", "避免暴露信息优势", "控制发言节奏"],
        risk_alerts=risks,
        confidence=conf,
        reasoning="动态分析：根据自身暴露状态和被怀疑程度调整策略",
        prompt_injectable=prompt,
    )


# --- RESIST_PUSH (抗推) ---

def _resist_push_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    if gs is None:
        return _resist_push_static(inp, skill)
    return _resist_push_dynamic(inp, skill)


def _resist_push_static(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="defense_speech",
        speech_structure=["承认疑虑合理性", "逻辑反驳关键指控", "提出建设性站边"],
        risk_alerts=["过度防御可能加深怀疑", "攻击质疑者会适得其反"],
        confidence=0.55,
        reasoning="抗推需要冷静的逻辑反驳，而非情绪对抗",
    )


def _resist_push_dynamic(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
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
        recommended_action="defense_speech",
        recommended_target=pushers[0] if pushers else None,
        speech_structure=speech,
        risk_alerts=risks,
        confidence=conf,
        reasoning="动态分析：根据推票来源和强度调整抗推策略",
        prompt_injectable=prompt,
    )


# --- WOLF_PIT_ANALYSIS (盘狼坑) ---

def _wolf_pit_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    if gs is None:
        return _wolf_pit_static(inp, skill)
    return _wolf_pit_dynamic(inp, skill)


def _wolf_pit_static(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="analyze",
        speech_structure=["列出嫌疑人", "分析各嫌疑人证据", "排除法缩小范围"],
        confidence=0.5,
        reasoning="盘狼坑需要系统性分析所有嫌疑人的行为链",
    )


def _wolf_pit_dynamic(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    ws = inp.world_state
    bs = inp.belief_state
    gs = inp.game_state

    # Build suspect and exclude lists
    suspects: list[tuple[str, str]] = []   # (player, reason)
    excluded: list[tuple[str, str]] = []   # (player, reason)

    total_wolves = sum(1 for p in gs.players.values() if p.role == "werewolf")

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
        for f in ws.facts_of_type("seer_check_claim"):
            target = f.target_player
            if target and "wolf" in f.value.lower() or (target and "狼" in f.value):
                suspects.append((target, f"被{f.source_player}查杀"))
            elif target and ("good" in f.value.lower() or "金水" in f.value):
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

    prompt = (
        f"盘狼坑分析：场上{total_wolves}狼中，"
        f"嫌疑区：{'；'.join(suspect_lines) if suspect_lines else '暂无明确嫌疑人'}。"
        f"排除区：{'；'.join(exclude_lines) if exclude_lines else '暂无排除'}。"
        f"需要继续关注投票链和发言一致性来缩小范围。"
    )

    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="analyze",
        speech_structure=["列出嫌疑人及其证据", "分析排除区", "缩小嫌疑范围"],
        confidence=0.5 + min(0.2, len(unique_suspects) * 0.05),
        reasoning=f"动态分析：{len(unique_suspects)}个嫌疑人，{len(unique_excluded)}个排除",
        prompt_injectable=prompt,
    )


# --- PROTECT_POWER (保护强神) ---

def _protect_power_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    if gs is None:
        return _protect_power_static(inp, skill)
    return _protect_power_dynamic(inp, skill)


def _protect_power_static(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="speech",
        speech_structure=["暗示关键角色需要保护", "引导怀疑方向远离神职", "分散狼队注意力"],
        risk_alerts=["过度保护某个玩家反而暴露其身份"],
        confidence=0.5,
        reasoning="保护强神需要隐蔽的引导而非明显的保护行为",
    )


def _protect_power_dynamic(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    ws = inp.world_state
    bs = inp.belief_state

    power_roles = {"seer", "witch", "hunter"}
    at_risk: list[dict[str, Any]] = []

    if bs is not None:
        for pid, belief in bs.beliefs.items():
            top_role, prob = belief.top_role_guess()
            if top_role in power_roles and prob > 0.3:
                votes_on = _vote_targets_for_player(ws, pid)
                if votes_on:
                    at_risk.append({
                        "player": pid,
                        "likely_role": top_role,
                        "votes": len(votes_on),
                    })

    risks = ["过度保护某个玩家反而暴露其身份"]

    if at_risk:
        target = at_risk[0]
        prompt = (
            f"保护强神建议：疑似{target['likely_role']}的 {target['player']} "
            f"正被推票（{target['votes']}票）。"
            f"建议发言引导怀疑方向远离TA：提出其他嫌疑人、质疑推票逻辑。"
            f"注意保护要隐蔽，不要让狼队察觉你在保人。"
        )
        conf = 0.6
    else:
        prompt = (
            f"保护强神建议：场上疑似神职暂时安全，无被推票压力。"
            f"继续观察，注意保护已识别的疑似神职不被狼队发现。"
        )
        conf = 0.45

    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="speech",
        recommended_target=at_risk[0]["player"] if at_risk else None,
        speech_structure=["引导怀疑方向远离", "提出替代嫌疑人", "隐蔽保护"],
        risk_alerts=risks,
        confidence=conf,
        reasoning="动态分析：根据神职受压情况调整保护策略",
        prompt_injectable=prompt,
    )


# --- LAST_WORDS_ANALYSIS (遗言分析) ---

def _last_words_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    if gs is None:
        return _last_words_static(inp, skill)
    return _last_words_dynamic(inp, skill)


def _last_words_static(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="analyze",
        speech_structure=["提取遗言关键信息", "分析遗言与已知信息的矛盾", "评估遗言可信度"],
        confidence=0.55,
        reasoning="遗言分析需要结合已有信息判断遗言内容的真实性",
    )


def _last_words_dynamic(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    ws = inp.world_state
    alerts = inp.contradiction_alerts

    if ws is None:
        return _last_words_static(inp, skill)

    # Find recent deaths
    recent_deaths = []
    for f in ws.facts_of_type("player_died"):
        recent_deaths.append(f)

    if not recent_deaths:
        return SkillOutput(
            skill_name=skill.name.value,
            recommended_action="analyze",
            confidence=0.3,
            reasoning="暂无死亡事件可分析",
            prompt_injectable="遗言分析：当前无遗言可分析。",
        )

    last_death = recent_deaths[-1]
    dead_player = last_death.target_player

    # Find last speech of dead player
    last_speech_facts = [
        f for f in ws.facts_of_type("speech")
        if f.source_player == dead_player
    ]

    # Check for contradictions in dead player's statements
    dead_alerts = _alerts_for_player(alerts, dead_player or "")

    claims: list[str] = []
    if ws is not None:
        for f in ws.facts_of_type("claimed_role"):
            if f.source_player == dead_player:
                claims.append(f"{f.value}(Day{f.day})")

    prompt_parts = [f"遗言分析：{dead_player} 的遗言关键信息："]
    if claims:
        prompt_parts.append(f"身份声明：{', '.join(claims)}。")
    if dead_alerts:
        prompt_parts.append(f"发言存在矛盾：{'; '.join(a.description for a in dead_alerts[:2])}。")
    if last_speech_facts:
        prompt_parts.append(f"有{len(last_speech_facts)}条发言记录可供分析。")
    prompt_parts.append("需要结合已知信息判断遗言内容的真实性和意图。")

    prompt = "".join(prompt_parts)
    has_contradiction = len(dead_alerts) > 0

    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="analyze",
        recommended_target=dead_player,
        speech_structure=["提取遗言关键信息", "对比已知信息一致性", "评估可信度"],
        risk_alerts=["遗言可能是狼人的误导"] if has_contradiction else [],
        confidence=0.55 if not has_contradiction else 0.65,
        reasoning="动态分析：根据遗言内容与已知信息的对比判断可信度",
        prompt_injectable=prompt,
    )


# --- REVIEW_CORRECTION (复盘纠错) ---

def _review_correction_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    if gs is None:
        return _review_correction_static(inp, skill)
    return _review_correction_dynamic(inp, skill)


def _review_correction_static(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="review",
        speech_structure=["回顾关键判断点", "识别错误和原因", "总结改进方向"],
        confidence=0.7,
        reasoning="复盘纠错以事实为基础，系统性地回顾决策过程",
    )


def _review_correction_dynamic(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    ws = inp.world_state
    gs = inp.game_state

    winner = gs.winning_faction or "unknown"
    total_events = len(gs.events) if gs.events else 0
    day = gs.day_number

    deaths_by_wolf = 0
    deaths_by_exile = 0
    if ws is not None:
        for f in ws.facts_of_type("player_died"):
            reason = f.metadata.get("reason", "")
            if "wolf" in reason:
                deaths_by_wolf += 1
            elif "exile" in reason:
                deaths_by_exile += 1

    prompt = (
        f"复盘分析：游戏进行到Day {day}，共{total_events}个事件。"
        f"狼刀死亡{deaths_by_wolf}人，放逐{deaths_by_exile}人。"
        f"{'获胜方：' + winner if winner != 'unknown' else '游戏尚未结束'}。"
        f"关键判断点：回顾每个Day的投票决策和站边选择，识别错误和原因，总结改进方向。"
    )

    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="review",
        speech_structure=["回顾关键判断点", "识别错误和原因", "总结改进方向"],
        confidence=0.7,
        reasoning="动态分析：基于完整游戏时间线进行复盘",
        prompt_injectable=prompt,
    )


# ---------------------------------------------------------------------------
# Handler dispatch table
# ---------------------------------------------------------------------------

_SKILL_HANDLERS = {
    SkillName.BOLD_CLAIM: _bold_claim_handler,
    SkillName.COUNTER_CLAIM: _counter_claim_handler,
    SkillName.PUSH_VOTE: _push_vote_handler,
    SkillName.SWING_VOTE: _swing_vote_handler,
    SkillName.DEEP_HOOK: _deep_hook_handler,
    SkillName.FIND_POWER: _find_power_handler,
    SkillName.HIDE_IDENTITY: _hide_identity_handler,
    SkillName.RESIST_PUSH: _resist_push_handler,
    SkillName.WOLF_PIT_ANALYSIS: _wolf_pit_handler,
    SkillName.PROTECT_POWER: _protect_power_handler,
    SkillName.LAST_WORDS_ANALYSIS: _last_words_handler,
    SkillName.REVIEW_CORRECTION: _review_correction_handler,
}
