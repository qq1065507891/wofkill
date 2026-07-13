# -*- coding: utf-8 -*-
"""
测试猎人开枪阶段的策略指令构建函数。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-13

使用示例:
    >>> from werewolf_agent.runtime.hunter_shot_directives import build_hunter_death_label
    >>> build_hunter_death_label("exile")
"""

import json

from werewolf_agent.agents.schemas import ActionType
from werewolf_agent.runtime.hunter_shot_directives import (
    build_hunter_death_label,
    build_hunter_shoot_encouragement,
    build_hunter_shot_directive,
    build_hunter_shot_result,
)


def test_build_hunter_death_label_maps_known_reasons() -> None:
    """猎人死亡原因标签应映射已知原因，未知原因保留原值。"""
    assert build_hunter_death_label("wolf_kill") == "被狼人袭击"
    assert build_hunter_death_label("exile") == "被投票放逐"
    assert build_hunter_death_label("poison") == "因poison"


def test_build_hunter_shoot_encouragement_uses_top_target_value() -> None:
    """开枪鼓励文案应根据最高候选分值分级。"""
    strong = {"ranked_targets": [{"target": "p02", "value": 6}]}
    medium = {"ranked_targets": [{"target": "p02", "value": 3}]}
    weak = {"ranked_targets": [{"target": "p02", "value": 1}]}

    assert "优先开枪带走" in build_hunter_shoot_encouragement(strong)
    assert "比较出错成本" in build_hunter_shoot_encouragement(medium)
    assert "优先选择不开枪" in build_hunter_shoot_encouragement(weak)
    assert "优先选择不开枪" in build_hunter_shoot_encouragement(None)


def test_build_hunter_shot_directive_adds_assessment_when_present() -> None:
    """猎人开枪策略指令应包含死亡原因和可选目标评估。"""
    shot_assessment = {"ranked_targets": [{"target": "p02", "value": 6}]}

    directive = build_hunter_shot_directive(
        death_reason="exile",
        shot_assessment=shot_assessment,
    )

    assert "被投票放逐导致死亡" in directive["hunter_shot_directive"]
    assert directive["shot_value_assessment"] == shot_assessment


def test_hunter_directive_exposes_structured_alternative_and_friendly_fire_evidence() -> None:
    shot_assessment = {
        "ranked_targets": [
            {"target": "p02", "value": 6, "signals": ["public_suspect_by_p04"]},
            {"target": "p03", "value": -6, "signals": ["public_good_claim_by_p05"]},
        ],
        "alternative_comparison": {
            "legal_alternatives": ["p02", "p03"],
            "no_legal_alternative": False,
        },
        "friendly_fire_risk": {"targets": ["p03"]},
    }
    directive = build_hunter_shot_directive(
        death_reason="exile", shot_assessment=shot_assessment
    )
    assert directive["alternative_comparison"]["legal_alternatives"] == ["p02", "p03"]
    assert directive["friendly_fire_risk"]["targets"] == ["p03"]
    assert "alternative_comparison" not in directive["shot_value_assessment"]
    assert "friendly_fire_risk" not in directive["shot_value_assessment"]
    serialized = json.dumps(directive, ensure_ascii=False)
    assert serialized.count('"alternative_comparison"') == 1
    assert serialized.count('"friendly_fire_risk"') == 1


def test_hunter_directive_marks_single_target_as_no_legal_alternative() -> None:
    assessment = {"ranked_targets": [{"target": "p02", "value": 1, "signals": []}]}
    directive = build_hunter_shot_directive(
        death_reason="exile", shot_assessment=assessment
    )
    assert directive["alternative_comparison"]["no_legal_alternative"] is True


def test_hunter_zero_targets_still_emits_structured_retain_directive() -> None:
    directive = build_hunter_shot_directive(
        death_reason="exile",
        shot_assessment=None,
    )

    assert directive["alternative_comparison"] == {
        "legal_alternatives": [],
        "no_legal_alternative": True,
    }
    assert directive["friendly_fire_risk"] == {
        "status": "not_applicable",
        "targets": [],
        "basis": "无合法目标",
    }
    assert directive["retain_option"] == {
        "action": "no_action",
        "available": True,
        "required": True,
        "reason": "无合法开枪目标",
    }


def test_build_hunter_shot_result_maps_action_type() -> None:
    """猎人行动结果应仅在 hunter_shot 且有目标时返回目标。"""
    shot = build_hunter_shot_result(
        action_type=ActionType.HUNTER_SHOT,
        target_id="p02",
        action_trace={"action": "hunter_shot"},
    )
    no_shot = build_hunter_shot_result(
        action_type=ActionType.NO_ACTION,
        target_id="p02",
        action_trace={"action": "no_action"},
    )

    assert shot == {
        "hunter_shot_target_id": "p02",
        "action_trace": {"action": "hunter_shot"},
    }
    assert no_shot == {
        "hunter_shot_target_id": None,
        "action_trace": {"action": "no_action"},
    }
