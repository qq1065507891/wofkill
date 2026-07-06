# -*- coding: utf-8 -*-
"""
构建技能建议帧并限制可注入 prompt 文本长度。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.skills.advice_frames import _ensure_skill_advice_frame
"""

from __future__ import annotations

from werewolf_agent.skills.schemas import (
    SkillAdviceFrame,
    SkillDefinition,
    SkillInput,
    SkillName,
    SkillOutput,
)


PROMPT_INJECTABLE_CAP = 800

PROMPT_INJECTABLE_MARKER_TAIL = "...（已省略）"

_GENERIC_SKILL_FRAME_DEFAULTS: dict[str, tuple[list[str], str]] = {
    SkillName.BOLD_CLAIM.value: (
        [
            "没有前期窗口、没有完整假验人时间线或场上已有多人对跳时，不要强行悍跳。",
            "队友已经承担预言家身份或自己分工不是悍跳位时，应降级为配合站边。",
        ],
        "不得把悍跳建议当成真实身份信息；不得泄漏狼队分工或队友身份。",
    ),
    SkillName.SWING_VOTE.value: (
        [
            "目标不在合法投票/夜杀范围、票型压力不足或狼队人数过少时，不要冲票/冲刀。",
            "冲票会暴露投票链，夜杀讨论不能泄漏队友协作细节。",
        ],
        "不得把冲票/冲刀目标当成裁判身份真相；不得推荐非法目标。",
    ),
    SkillName.DEEP_HOOK.value: (
        [
            "自己承担悍跳或冲锋分工时，不要切换成倒钩。",
            "队友尚未暴露且过度攻击队友会制造反常行为链时，不要强行倒钩。",
        ],
        "不得泄漏狼队队友身份；不得用倒钩建议覆盖当前分工。",
    ),
    SkillName.FIND_POWER.value: (
        [
            "没有发言信息量、警徽流、保护行为或票型信号时，不要硬猜神职。",
            "候选人已死亡或信号来源不可公开时，不要把该候选注入行动建议。",
        ],
        "不得把神职推断当成裁判身份真相；不得编造夜间信息。",
    ),
    SkillName.RESIST_PUSH.value: (
        [
            "没有被推票、查杀或明确质疑时，不要过度防御。",
            "回应压力时不能情绪化攻击质疑者，应优先引用公开事实。",
        ],
        "不得把抗推话术当成免疫放逐的硬约束；不得虚构证据自证。",
    ),
    SkillName.WOLF_PIT_ANALYSIS.value: (
        [
            "公开事实不足、嫌疑区为空或排除区依据不足时，不要强行定狼坑。",
            "被查验/投票链证据冲突时，需要保留不确定性。",
        ],
        "不得把狼坑排序当成裁判身份真相；不得忽略死亡/非法目标过滤。",
    ),
    SkillName.PROTECT_POWER.value: (
        [
            "没有疑似神职候选或候选未受压时，不要明显保人。",
            "保护动作过强会反向暴露神职身份，需要降级为轻度站边。",
        ],
        "不得公开声称他人神职身份；不得把保护建议当成身份确认。",
    ),
    SkillName.LAST_WORDS_ANALYSIS.value: (
        [
            "没有死亡事件、没有遗言内容或遗言与公开事实冲突时，不能直接采信。",
            "遗言可能是狼队误导，需要与已知声明、票型和矛盾记录交叉验证。",
        ],
        "不得把遗言当成裁判真相；不得从遗言推导隐藏夜间信息。",
    ),
    SkillName.REVIEW_CORRECTION.value: (
        [
            "复盘只用于总结历史决策，不应影响当前局不可见信息判断。",
            "缺少完整事件链时，只能记录待验证问题，不能给出绝对结论。",
        ],
        "不得把复盘结论当成当前局事实；不得跨局泄漏具体玩家身份或隐藏信息。",
    ),
}

def _cap_prompt_injectable(text: str, cap: int = PROMPT_INJECTABLE_CAP) -> str:
    """Truncate `text` to `cap` chars, appending a marker on truncation.

    S-06: late-game review output can exceed 1KB and bloat the
    renderer prompt. Cap to 800 chars and signal truncation to the
    LLM so it knows the advice is partial.
    """
    if not text:
        return text
    if len(text) <= cap:
        return text
    marker = PROMPT_INJECTABLE_MARKER_TAIL
    return text[: cap - len(marker)] + marker

def _advice_frame(
    *,
    skill: SkillDefinition,
    inp: SkillInput,
    recommended_use: str,
    risk_alerts: list[str] | None,
    counter_signals: list[str],
    forbidden_use: str,
    confidence: float,
    relevance: float | None = None,
    situation_detail: str = "",
    evidence_refs: list[str] | None = None,
) -> SkillAdviceFrame:
    """Build native structured skill advice for prompt injection."""
    conf = max(0.0, min(1.0, float(confidence)))
    rel = conf if relevance is None else max(0.0, min(1.0, float(relevance)))
    task = inp.task_type or inp.phase
    situation = (
        f"role={inp.role} task={task} phase={inp.phase} day={inp.day}"
    )
    if inp.player_id:
        situation += f" player={inp.player_id}"
    if situation_detail:
        situation += f" {situation_detail}"
    return SkillAdviceFrame(
        skill=skill.name.value,
        situation_signature=situation,
        recommended_use=recommended_use,
        risk_alerts=list(risk_alerts or []),
        counter_signals=counter_signals,
        forbidden_use=forbidden_use,
        confidence=conf,
        relevance=rel,
        evidence_refs=list(evidence_refs or []),
    )

def _push_vote_advice_frame(
    inp: SkillInput,
    skill: SkillDefinition,
    *,
    prompt: str,
    risks: list[str],
    confidence: float,
    primary: str = "",
) -> SkillAdviceFrame:
    target_detail = f"target={primary}" if primary else "target=unknown"
    return _advice_frame(
        skill=skill,
        inp=inp,
        recommended_use=prompt,
        risk_alerts=risks,
        counter_signals=[
            "没有明确公开证据或信任模型未形成狼倾向时，不要强行归票。",
            "目标已死亡、不在合法投票范围或票型信息不足时，需要降级为观察。",
        ],
        forbidden_use="不得把嫌疑排序当成裁判身份真相；不得虚构查验、票型或发言。",
        confidence=confidence,
        relevance=0.7 if primary else 0.35,
        situation_detail=target_detail,
    )

def _counter_claim_advice_frame(
    inp: SkillInput,
    skill: SkillDefinition,
    *,
    prompt: str,
    risks: list[str],
    confidence: float,
    target: str = "",
) -> SkillAdviceFrame:
    target_detail = f"target={target}" if target else "target=none"
    return _advice_frame(
        skill=skill,
        inp=inp,
        recommended_use=prompt,
        risk_alerts=risks,
        counter_signals=[
            "场上无人跳预言家或不存在可对跳对象时，不要使用对跳话术。",
            "没有完整查验时间线、验人动机或警徽流解释时，不要硬跳。",
        ],
        forbidden_use="不得把对跳建议当成真实身份证明；不得泄漏隐藏阵营信息或编造裁判结果。",
        confidence=confidence,
        relevance=0.7 if target else 0.25,
        situation_detail=target_detail,
    )

def _hide_identity_advice_frame(
    inp: SkillInput,
    skill: SkillDefinition,
    *,
    prompt: str,
    risks: list[str],
    confidence: float,
    exposure: str = "",
) -> SkillAdviceFrame:
    detail = f"exposure={exposure}" if exposure else ""
    return _advice_frame(
        skill=skill,
        inp=inp,
        recommended_use=prompt,
        risk_alerts=risks,
        counter_signals=[
            "进入公开窗口、需要带队或必须报验人时，不要继续隐藏。",
            "已经被重点怀疑时，不能只沉默，需要释放可公开信息自证。",
        ],
        forbidden_use="不得用藏身份覆盖强制公开信息、合法行动要求或已经公开的身份声明。",
        confidence=confidence,
        relevance=0.6,
        situation_detail=detail,
    )

def _generic_skill_advice_frame(
    output: SkillOutput,
    inp: SkillInput,
    skill: SkillDefinition,
) -> SkillAdviceFrame:
    recommended_use = (
        (output.prompt_injectable or "").strip()
        or (output.reasoning or "").strip()
        or "该技能当前只提供低优先级参考。"
    )
    counter_signals, forbidden_use = _GENERIC_SKILL_FRAME_DEFAULTS.get(
        skill.name.value,
        (
            ["当前局公开事实不足以支撑该技能建议时不要使用。"],
            "不得把技能建议当成裁判真相、隐藏身份真相或当前局公开记录。",
        ),
    )
    confidence = max(0.0, min(1.0, float(output.confidence or 0.0)))
    relevance = confidence
    if not output.prompt_injectable:
        relevance = min(relevance, 0.3)
    return _advice_frame(
        skill=skill,
        inp=inp,
        recommended_use=recommended_use,
        risk_alerts=list(output.risk_alerts or []),
        counter_signals=counter_signals,
        forbidden_use=forbidden_use,
        confidence=confidence,
        relevance=relevance,
        evidence_refs=list((output.metadata or {}).get("evidence_refs", []) or []),
    )

def _ensure_skill_advice_frame(
    output: SkillOutput,
    inp: SkillInput,
    skill: SkillDefinition,
) -> SkillOutput:
    if output.advice_frame is None:
        output.advice_frame = _generic_skill_advice_frame(output, inp, skill)
    return output
