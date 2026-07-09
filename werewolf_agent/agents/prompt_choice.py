# -*- coding: utf-8 -*-
"""
构建 choice 模式的候选映射、候选摘要和提示词。

作者: Project contributors
创建日期: 2026-07-08
修改日期: 2026-07-09

使用示例:
    >>> from werewolf_agent.agents.prompt_choice import vote_choice_map
    >>> vote_choice_map(type("Ctx", (), {"legal_targets": ["p01", "p02"]})())
    {'A': 'p01', 'B': 'p02'}
"""

from __future__ import annotations

import json

from werewolf_agent.agents.schemas import ActionType, AgentContext, TaskType


def _vote_reason_privacy_guard() -> str:
    """延迟读取旧 facade 常量，避免模块导入环。"""
    from werewolf_agent.agents.prompt_output import _VOTE_REASON_PRIVACY_GUARD

    return _VOTE_REASON_PRIVACY_GUARD


def is_exile_vote_context(context: AgentContext) -> bool:
    """判断当前 choice prompt 是否属于白天放逐投票。"""
    return context.task_type == TaskType.VOTE and ActionType.VOTE in context.legal_actions


def vote_choice_map(context: AgentContext) -> dict[str, str]:
    """把合法目标映射为 A/B/C choice。"""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return {
        letters[idx]: target
        for idx, target in enumerate(context.legal_targets[:len(letters)])
    }


def vote_candidate_summary(context: AgentContext, target_id: str) -> str:
    """为放逐投票候选人生成公开证据摘要。"""
    clues: list[str] = []
    for item in context.salience_items:
        if not isinstance(item, dict):
            continue
        target = item.get("target") or item.get("target_id") or item.get("player_id")
        if target != target_id:
            continue
        item_type = item.get("type") or item.get("event")
        if item_type == "seer_claim":
            speaker = item.get("speaker") or item.get("seer_id")
            result = item.get("result") or item.get("alignment")
            if speaker and result:
                clues.append(f"{speaker}报{target_id}为{result}")
        elif item_type in {"vote_resolved", "vote"}:
            clues.append(f"{target_id}出现在关键票型中")
        elif item_type in {"player_died", "death"}:
            clues.append(f"{target_id}关联死亡事件")
    if clues:
        return "；".join(clues[:2])
    return (
        "暂无该候选的公开证据摘要；如选择该目标，必须另引公开发言、"
        "票型、查验或警徽流，不得把候选身份当作证据"
    )


def target_candidate_summary(context: AgentContext, target_id: str) -> str:
    """为技能或夜间目标候选人生成摘要。"""
    action = context.legal_actions[0] if context.legal_actions else ActionType.NO_ACTION
    action_reasons = {
        ActionType.WOLF_KILL: "作为狼队夜间击杀目标",
        ActionType.USE_POISON: "作为女巫毒药目标",
        ActionType.CHECK_ALIGNMENT: "作为预言家查验目标",
        ActionType.CHOOSE_MASTER: "作为混血儿主人选择目标",
        ActionType.HUNTER_SHOT: "作为猎人开枪目标",
        ActionType.BADGE_TRANSFER: "作为警徽移交目标",
        ActionType.SHERIFF_VOTE: "作为警长投票目标",
    }
    clues: list[str] = []
    for item in context.salience_items:
        if not isinstance(item, dict):
            continue
        item_text = json.dumps(item, ensure_ascii=False)
        if target_id in item_text:
            clues.append(item_text[:80])
    basis = f"；依据：{'；'.join(clues[:2])}" if clues else ""
    reason = action_reasons.get(action, "作为当前合法目标")
    return f"{target_id}{reason}较合适{basis}"


def format_choice_prompt(context: AgentContext) -> str:
    """渲染 target-choice / vote-choice 输出提示。"""
    is_vote = is_exile_vote_context(context)
    header = "投票候选枚举" if is_vote else "目标候选枚举"
    choice_map = vote_choice_map(context)
    lines = [
        f"{header}（必须从中选择一个choice，不要直接编写target_id）："
    ]
    if is_vote:
        lines.append(
            "候选枚举不是公开证据；摘要只用于识别候选，不能复制成 vote reason。"
            "reason 必须引用当前局公开事实：查验、对跳、警徽流、票型或具体发言。"
        )
    for choice, target_id in choice_map.items():
        summary = (
            vote_candidate_summary(context, target_id)
            if is_vote
            else target_candidate_summary(context, target_id)
        )
        lines.append(f"{choice} = {target_id}，摘要：{summary}")
    if is_vote:
        role = context.own_role or "villager"
        choice_vote_basis = "seer_check" if role == "seer" else "seer_siding"
        example = (
            '{"choice":"A","reason":"投票公开理由",'
            '"seer_stance":"trust",'
            f'"vote_basis":"{choice_vote_basis}",'
            '"standing_with_seer":"站边的预言家或逻辑线",'
            '"suspect_reason":"为什么怀疑该候选",'
            '"not_voting_reason":"为什么不投其他候选",'
            '"candidate_comparison":"至少两名候选人的公开证据对比",'
            '"private_reason":"完整内心理由",'
            '"confidence":0.7}'
        )
    else:
        example = '{"choice":"A","reason":"选择该目标的简明理由","confidence":0.7}'
    lines.extend([
        "只需要输出choice决策JSON，程序会把choice映射为target_id并组装PlayerAction。",
        "示例：",
        example,
    ])
    if is_vote:
        return _vote_reason_privacy_guard() + "\n".join(lines)
    return "\n".join(lines)


__all__ = [
    "format_choice_prompt",
    "is_exile_vote_context",
    "target_candidate_summary",
    "vote_candidate_summary",
    "vote_choice_map",
]
