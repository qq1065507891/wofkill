# -*- coding: utf-8 -*-
"""
把技能输出转换为 AgentContext 中的战术建议，并收集技能调用监控记录。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-10

使用示例:
    >>> from werewolf_agent.runtime.context_skill_advice import _inject_skill_output
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import GameState
from werewolf_agent.skills.registry import SkillRegistry
from werewolf_agent.skills.schemas import SkillAdviceFrame, SkillInput

_MAX_SKILL_TACTICAL_ADVICE_ITEMS = 3
_NEGATIVE_SIGNAL_TOKENS = ("不需要", "不要", "避免", "禁止", "不能", "不建议", "无需")
_LOW_CONFIDENCE_SKILL_THRESHOLD = 0.4


def _has_negative_signal_advice(*parts: Any) -> bool:
    text = " ".join(str(part or "") for part in parts)
    return any(token in text for token in _NEGATIVE_SIGNAL_TOKENS)


def _skill_output_to_advice_frame(
    output: Any,
    skill_input: SkillInput,
) -> SkillAdviceFrame:
    if getattr(output, "advice_frame", None) is not None:
        frame = output.advice_frame
        if (
            frame.confidence < _LOW_CONFIDENCE_SKILL_THRESHOLD
            and _has_negative_signal_advice(
                getattr(output, "reasoning", ""),
                frame.recommended_use,
            )
        ):
            return replace(frame, relevance=max(frame.relevance, 0.65))
        return frame
    advice = (getattr(output, "prompt_injectable", "") or "").strip()
    reasoning = (getattr(output, "reasoning", "") or "").strip()
    recommended_use = advice or reasoning or "该技能当前只提供低优先级参考。"
    confidence = float(getattr(output, "confidence", 0.5) or 0.0)
    relevance = max(0.0, min(1.0, confidence))
    if (
        confidence < _LOW_CONFIDENCE_SKILL_THRESHOLD
        and _has_negative_signal_advice(recommended_use)
    ):
        relevance = max(relevance, 0.65)
    skill_name = getattr(output, "skill_name", "") or ""
    return SkillAdviceFrame(
        skill=skill_name,
        situation_signature=(
            f"role={skill_input.role} task={skill_input.task_type or skill_input.phase} "
            f"phase={skill_input.phase} day={skill_input.day}"
        ),
        recommended_use=recommended_use,
        risk_alerts=list(getattr(output, "risk_alerts", []) or []),
        counter_signals=["当前局公开事实不足以支撑该技能建议时不要使用。"],
        forbidden_use="不得把技能建议当成裁判真相、隐藏身份真相或当前局公开记录。",
        confidence=confidence,
        relevance=relevance,
        evidence_refs=list((getattr(output, "metadata", {}) or {}).get("evidence_refs", []) or []),
    )


def _skill_advice_frame_to_prompt_dict(
    frame: SkillAdviceFrame,
    *,
    advice: str,
) -> dict[str, Any]:
    return {
        "skill": frame.skill,
        "advice": advice,
        "situation_signature": frame.situation_signature,
        "recommended_use": frame.recommended_use,
        "risk_alerts": frame.risk_alerts,
        "counter_signals": frame.counter_signals,
        "forbidden_use": frame.forbidden_use,
        "confidence": frame.confidence,
        "relevance": frame.relevance,
        "evidence_refs": frame.evidence_refs,
    }


def _inject_skill_output(
    strategy_directive: dict[str, Any],
    gs: GameState,
    player_id: str,
    world_state: Any,
    belief_state: Any,
    contradiction_alerts: list[Any],
    task_type: str,
    legal_targets: list[str] | None = None,
    wolf_team_plan: dict[str, Any] | None = None,
    skill_call_records: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """把技能建议注入 strategy_directive，并返回技能分析摘要。"""

    player = gs.players.get(player_id)
    if not player or not player.alive:
        return strategy_directive, {}

    try:
        from werewolf_agent.runtime import context as context_mod
        registry_factory = getattr(context_mod, "SkillRegistry", SkillRegistry)
    except Exception:
        registry_factory = SkillRegistry
    registry = registry_factory()
    skill_input = SkillInput(
        role=player.role,
        phase=task_type,
        day=gs.day_number,
        game_state=gs,
        world_state=world_state,
        belief_state=belief_state,
        contradiction_alerts=contradiction_alerts,
        player_id=player_id,
        legal_targets=legal_targets or [],
        extra={"wolf_team_plan": wolf_team_plan} if wolf_team_plan else {},
        task_type=task_type,
    )
    dispatch_kwargs: dict[str, Any] = {"task_type": task_type, "gs": gs}
    if skill_call_records is not None:
        dispatch_kwargs["audit_records"] = skill_call_records
    outputs = registry.dispatch_for_role(
        player.role,
        task_type,
        skill_input,
        **dispatch_kwargs,
    )

    sortable: list[tuple[float, str, Any]] = []
    skill_analyses: dict[str, str] = {}
    for output in outputs:
        if not output.prompt_injectable:
            skill_analyses.setdefault(output.skill_name, output.prompt_injectable or "")
            continue
        skill_analyses[output.skill_name] = output.prompt_injectable
        sortable.append((output.confidence, output.prompt_injectable, output))

    sortable.sort(key=lambda item: -item[0])
    structured: list[dict[str, Any]] = []
    seen: set[str] = set()
    legal_set = set(legal_targets or [])

    from werewolf_agent.skills.schemas import SkillName as _SkillName

    analysis_exempt_skills: set[str] = set()
    for skill_name_enum in _SkillName:
        if skill_name_enum.value in ("last_words", "review_correct"):
            analysis_exempt_skills.add(skill_name_enum.value)

    for confidence, prompt, source_output in sortable:
        skill_name = source_output.skill_name if source_output else ""
        if skill_name in seen:
            continue
        seen.add(skill_name)
        if legal_set and prompt and skill_name not in analysis_exempt_skills:
            import re as _re

            mentioned: set[str] = set()
            for match in _re.finditer(r"\b[pP](\d+)\b", prompt):
                mentioned.add(f"p{int(match.group(1)):02d}")
            for match in _re.finditer(r"(\d+)\s*号\s*玩家?", prompt):
                mentioned.add(f"p{int(match.group(1)):02d}")
            for match in _re.finditer(r"玩家\s*(\d+)", prompt):
                mentioned.add(f"p{int(match.group(1)):02d}")
            illegal = mentioned - legal_set
            if illegal:
                continue
        frame = _skill_output_to_advice_frame(source_output, skill_input)
        structured.append(
            _skill_advice_frame_to_prompt_dict(frame, advice=prompt)
        )

    structured.sort(
        key=lambda item: (
            float(item.get("relevance", 0.0) or 0.0),
            float(item.get("confidence", 0.0) or 0.0),
        ),
        reverse=True,
    )
    structured = structured[:_MAX_SKILL_TACTICAL_ADVICE_ITEMS]
    structured.sort(
        key=lambda item: (
            float(item.get("confidence", 0.0) or 0.0),
            float(item.get("relevance", 0.0) or 0.0),
        ),
        reverse=True,
    )
    if structured:
        strategy_directive["skill_tactical_advice"] = structured
    return strategy_directive, skill_analyses
