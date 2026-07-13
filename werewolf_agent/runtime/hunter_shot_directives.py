# -*- coding: utf-8 -*-
"""
构建猎人开枪阶段的策略指令和结果。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-13

使用示例:
    >>> from werewolf_agent.runtime.hunter_shot_directives import build_hunter_death_label
    >>> build_hunter_death_label("exile")
"""

from __future__ import annotations

from typing import Any, Mapping

from werewolf_agent.agents.schemas import ActionType


def build_hunter_death_label(death_reason: str) -> str:
    """把死亡原因转换成猎人可读提示。"""
    return {"wolf_kill": "被狼人袭击", "exile": "被投票放逐"}.get(
        death_reason,
        f"因{death_reason}",
    )


def build_hunter_shoot_encouragement(
    shot_assessment: Mapping[str, Any] | None,
) -> str:
    """根据目标评估分值构建猎人是否开枪的提示。"""
    has_suspects = (
        shot_assessment
        and shot_assessment.get("ranked_targets")
        and len(shot_assessment["ranked_targets"]) > 0
    )
    top_value = 0
    if has_suspects:
        top_value = int(shot_assessment["ranked_targets"][0].get("value", 0))
    if top_value >= 6:
        return "存在明确查杀、对跳或强公共证据目标，可以优先开枪带走。"
    if top_value >= 3:
        return "存在一定公共证据目标；开枪前仍要比较出错成本，避免误伤好人。"
    return (
        "当前没有明确查杀、强票型或强对跳失败目标，优先选择不开枪（NO_ACTION），"
        "避免误伤好人；只有你能指出具体硬证据时才开枪。"
    )


def build_hunter_shot_directive(
    *,
    death_reason: str,
    shot_assessment: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """构建猎人开枪阶段的完整策略指令。"""
    death_label = build_hunter_death_label(death_reason)
    shoot_encouragement = build_hunter_shoot_encouragement(shot_assessment)
    strategy_directive: dict[str, Any] = {
        "hunter_shot_directive": (
            f"你是猎人，{death_label}导致死亡。你现在可以开枪带走一名玩家。\n"
            "开枪是一次性的，但你的判断是场上最好的武器之一。\n"
            f"{shoot_encouragement}\n"
            "注意：本局没有守卫，如果你被女巫毒杀（而非被狼杀或放逐），你无法开枪。\n"
            "speech字段留空。"
        ),
        "alternative_comparison": {
            "legal_alternatives": [],
            "no_legal_alternative": True,
        },
        "friendly_fire_risk": {
            "status": "not_applicable",
            "targets": [],
            "basis": "无合法目标",
        },
        "retain_option": {
            "action": "no_action",
            "available": True,
            "required": True,
            "reason": "无合法开枪目标",
        },
    }
    if shot_assessment:
        strategy_directive["shot_value_assessment"] = {
            key: value
            for key, value in shot_assessment.items()
            if key not in {"alternative_comparison", "friendly_fire_risk"}
        }
        ranked = list(shot_assessment.get("ranked_targets") or [])
        strategy_directive["alternative_comparison"] = shot_assessment.get(
            "alternative_comparison",
            {
                "legal_alternatives": [item.get("target") for item in ranked],
                "no_legal_alternative": len(ranked) <= 1,
            },
        )
        strategy_directive["friendly_fire_risk"] = shot_assessment.get(
            "friendly_fire_risk",
            {
                "targets": [
                    item.get("target") for item in ranked
                    if int(item.get("value", 0)) < 0
                ],
            },
        )
        strategy_directive["retain_option"] = {
            "action": "no_action",
            "available": True,
            "required": False,
            "reason": "可在误伤风险过高时保留不开枪选项",
        }
    return strategy_directive


def build_hunter_shot_result(
    *,
    action_type: ActionType,
    target_id: str | None,
    action_trace: dict[str, Any],
) -> dict[str, Any]:
    """把猎人行动转换为运行时返回结构。"""
    if action_type == ActionType.HUNTER_SHOT and target_id:
        return {
            "hunter_shot_target_id": target_id,
            "action_trace": action_trace,
        }
    return {
        "hunter_shot_target_id": None,
        "action_trace": action_trace,
    }
