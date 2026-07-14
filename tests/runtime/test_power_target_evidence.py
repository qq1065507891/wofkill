# -*- coding: utf-8 -*-
"""
验证神职伤害决策保留目标证据与选择对比。

作者: Project contributors
创建日期: 2026-07-14
"""

from __future__ import annotations

from werewolf_agent.runtime.agent_special_actions import _damage_decision_evidence
from werewolf_agent.runtime.witch_night_directives import build_witch_action_evidence


def test_hunter_damage_evidence_preserves_ranked_selected_and_alternative_scores() -> None:
    evidence = _damage_decision_evidence({
        "ranked_targets": [
            {"target": "p02", "value": 7, "signals": ["seer_check_wolf"]},
            {"target": "p03", "value": 2, "signals": ["vote_pattern"]},
        ],
        "alternative_comparison": {
            "legal_alternatives": ["p02", "p03"],
            "alternative_target": "p03", "no_legal_alternative": False,
        },
        "friendly_fire_risk": {
            "status": "assessed", "targets": [], "basis": "公开证据评分",
        },
        "retain_option": {
            "action": "no_action", "available": True,
            "required": False, "reason": "可以不开枪",
        },
    }, target_id="p02")

    assert evidence["target_evidence"] == {
        "selected_score": 7, "selected_signals": ["seer_check_wolf"],
    }
    assert evidence["target_comparison"] == {
        "selected_score": 7, "selected_signals": ["seer_check_wolf"],
        "alternative_target": "p03", "alternative_score": 2,
        "alternative_signals": ["vote_pattern"],
        "comparison_basis": "ranked public evidence score",
    }


def test_witch_action_evidence_builds_explainable_target_ranking() -> None:
    evidence = build_witch_action_evidence(
        legal_targets=["p02", "p03"], poison_targets=["p02", "p03"],
        poison_candidates=[{"player_id": "p03", "reason": "公开查杀"}],
        wolf_kill_target_id=None,
    )

    assert evidence["ranked_targets"] == [
        {"target": "p03", "value": 1, "signals": ["公开查杀"]},
        {"target": "p02", "value": 0, "signals": ["no_structured_public_support"]},
    ]
