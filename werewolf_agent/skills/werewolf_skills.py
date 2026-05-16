"""Werewolf skill definitions: 12 core gameplay skills per design doc §11.1.

Each skill provides a deterministic suggestion skeleton. LLM agents fill in
the natural language and make the final decision — skills do not override
agent autonomy or rule engine authority.
"""

from __future__ import annotations

from werewolf_agent.skills.schemas import (
    SkillDefinition,
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
        applicable_phases=["speech", "sheriff_speech", "pk_speech"],
        tags=["deception", "aggressive"],
    ),
    SkillDefinition(
        name=SkillName.COUNTER_CLAIM,
        display_name="对跳",
        description="针对已经起跳的玩家进行身份对跳，争夺话语权",
        applicable_roles=["seer", "werewolf"],
        applicable_phases=["speech", "pk_speech"],
        tags=["confrontation", "identity_claim"],
    ),
    SkillDefinition(
        name=SkillName.PUSH_VOTE,
        display_name="归票",
        description="引导全场投票方向，集中票数归出目标玩家",
        applicable_roles=["seer", "werewolf", "villager", "hunter", "witch", "idiot", "hybrid"],
        applicable_phases=["speech", "vote"],
        tags=["leadership", "voting"],
    ),
    SkillDefinition(
        name=SkillName.SWING_VOTE,
        display_name="冲票",
        description="集中阵营力量冲票特定目标，多用于狼队协同冲票",
        applicable_roles=["werewolf"],
        applicable_phases=["vote"],
        tags=["coordination", "aggressive"],
    ),
    SkillDefinition(
        name=SkillName.DEEP_HOOK,
        display_name="倒钩",
        description="在好人阵营中建立可信度，通过适度攻击狼队友来获取信任",
        applicable_roles=["werewolf"],
        applicable_phases=["speech", "vote"],
        tags=["deception", "long_term"],
    ),
    SkillDefinition(
        name=SkillName.FIND_POWER,
        display_name="找神",
        description="通过发言和行为模式分析找出神职玩家",
        applicable_roles=["werewolf", "villager", "hybrid"],
        applicable_phases=["speech", "night_action"],
        tags=["analysis", "information"],
    ),
    SkillDefinition(
        name=SkillName.HIDE_IDENTITY,
        display_name="藏身份",
        description="隐藏自己的真实角色，避免被过早识别",
        applicable_roles=["seer", "witch", "hunter", "werewolf", "hybrid"],
        applicable_phases=["speech", "sheriff_speech"],
        tags=["stealth", "defense"],
    ),
    SkillDefinition(
        name=SkillName.RESIST_PUSH,
        display_name="抗推",
        description="在被怀疑或被推票时进行有效防守和反驳",
        applicable_roles=["werewolf", "villager", "seer", "witch", "hunter", "idiot", "hybrid"],
        applicable_phases=["defense_speech", "pk_speech"],
        tags=["defense", "persuasion"],
    ),
    SkillDefinition(
        name=SkillName.WOLF_PIT_ANALYSIS,
        display_name="盘狼坑",
        description="系统性分析可能的狼人分布，缩小嫌疑范围",
        applicable_roles=["villager", "seer", "witch", "hunter", "idiot", "hybrid"],
        applicable_phases=["speech", "sheriff_speech"],
        tags=["analysis", "logic"],
    ),
    SkillDefinition(
        name=SkillName.PROTECT_POWER,
        display_name="保护强神",
        description="保护关键神职角色不被狼人发现或冲票",
        applicable_roles=["villager", "seer", "witch", "hunter", "idiot", "hybrid"],
        applicable_phases=["speech", "vote"],
        tags=["protection", "team_play"],
    ),
    SkillDefinition(
        name=SkillName.LAST_WORDS_ANALYSIS,
        display_name="遗言分析",
        description="分析遗言内容，提取信息，判断发言者真实身份",
        applicable_roles=["villager", "seer", "witch", "hunter", "idiot", "hybrid", "werewolf"],
        applicable_phases=["speech"],
        tags=["analysis", "information"],
    ),
    SkillDefinition(
        name=SkillName.REVIEW_CORRECTION,
        display_name="复盘纠错",
        description="复盘本局关键判断，识别错误，提出改进建议",
        applicable_roles=["villager", "seer", "witch", "hunter", "idiot", "hybrid", "werewolf"],
        applicable_phases=["review"],
        tags=["review", "improvement"],
    ),
]


# ---------------------------------------------------------------------------
# Skill applicability logic — deterministic suggestion skeleton
# ---------------------------------------------------------------------------

def apply_skill(skill_name: SkillName, skill_input: SkillInput) -> SkillOutput:
    """Apply a skill to generate a suggestion skeleton.

    Returns structured output with recommended action, speech structure,
    risk alerts, and confidence. The agent makes the final decision.
    """
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


def _default_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="analyze",
        confidence=0.5,
        reasoning=f"技能 {skill.display_name} 适用，需要更多局势信息",
    )


def _bold_claim_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
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


def _counter_claim_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    risks = ["对跳风险：真预言家对跳时好人会倾向真预言家"]
    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="counter_claim",
        speech_structure=["指出对方漏洞", "报自身查验信息", "建立时间线对比"],
        risk_alerts=risks,
        confidence=0.55,
        reasoning="对跳需要充分的逻辑支撑和时间线一致性",
    )


def _push_vote_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="vote",
        speech_structure=["陈述归票理由", "分析目标嫌疑", "号召全场归票"],
        risk_alerts=["归票错误目标可能导致好人损失"],
        confidence=0.6,
        reasoning="归票需要有充分的逻辑依据和说服力",
    )


def _swing_vote_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    risks = ["冲票暴露投票链：好人可能通过投票链锁定狼人"]
    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="vote",
        risk_alerts=risks,
        confidence=0.5,
        reasoning="冲票需要考虑投票链暴露风险",
    )


def _deep_hook_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="speech",
        speech_structure=["站边好人逻辑", "适度攻击可疑队友", "建立可信度"],
        risk_alerts=["倒钩策略需要长期维持一致性", "过度攻击队友可能被识别"],
        confidence=0.55,
        reasoning="倒钩核心是在好人阵营中建立长期可信度",
    )


def _find_power_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="analyze",
        speech_structure=["分析发言信息量", "观察投票倾向", "识别保护行为"],
        confidence=0.5,
        reasoning="找神需要综合多个信号源进行推断",
    )


def _hide_identity_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    risks = ["藏身份过久可能导致无法在关键时刻发挥作用"]
    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="speech",
        speech_structure=["保持中立发言", "避免暴露信息优势", "控制发言节奏"],
        risk_alerts=risks,
        confidence=0.6,
        reasoning="藏身份需要在隐匿和发挥作用之间找到平衡",
    )


def _resist_push_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="defense_speech",
        speech_structure=["承认疑虑合理性", "逻辑反驳关键指控", "提出建设性站边"],
        risk_alerts=["过度防御可能加深怀疑", "攻击质疑者会适得其反"],
        confidence=0.55,
        reasoning="抗推需要冷静的逻辑反驳，而非情绪对抗",
    )


def _wolf_pit_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="analyze",
        speech_structure=["列出嫌疑人", "分析各嫌疑人证据", "排除法缩小范围"],
        confidence=0.5,
        reasoning="盘狼坑需要系统性分析所有嫌疑人的行为链",
    )


def _protect_power_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="speech",
        speech_structure=["暗示关键角色需要保护", "引导怀疑方向远离神职", "分散狼队注意力"],
        risk_alerts=["过度保护某个玩家反而暴露其身份"],
        confidence=0.5,
        reasoning="保护强神需要隐蔽的引导而非明显的保护行为",
    )


def _last_words_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="analyze",
        speech_structure=["提取遗言关键信息", "分析遗言与已知信息的矛盾", "评估遗言可信度"],
        confidence=0.55,
        reasoning="遗言分析需要结合已有信息判断遗言内容的真实性",
    )


def _review_correction_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    return SkillOutput(
        skill_name=skill.name.value,
        recommended_action="review",
        speech_structure=["回顾关键判断点", "识别错误和原因", "总结改进方向"],
        confidence=0.7,
        reasoning="复盘纠错以事实为基础，系统性地回顾决策过程",
    )


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
