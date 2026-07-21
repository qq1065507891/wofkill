# -*- coding: utf-8 -*-
"""
功能描述：**：定义 HARD_CONSTRAINT_KEYS / SUGGESTION_KEYS / REFERENCE_KEYS 三个 frozenset，供 prompt builder 按优先级组装指令。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-21
使用示例：内部模块，无对外接口
"""

from __future__ import annotations


HARD_CONSTRAINT_KEYS: frozenset[str] = frozenset({
    "wolf_fake_seer_execution",
    "must_address_alerts",
    "first_night_killed",
    "speech_silent",
    "vote_silent",
    "witch_night_action",
    "seer_night_check",
    "role_alerts",
    "vote_pressure",
    "wolf_sheriff_must_claim_seer",
    "wolf_no_reveal_seer",
    "wolf_fake_seer_teammate",
    "wolf_kill_instruction",
    "wolf_team_discussion",
    "wolf_universal_rules",
    "anti_herd",
    "hybrid_wolf_master_directive",
    "hybrid_good_master_directive",
    "hunter_shot_directive",
    "last_words",
    "badge_decision",
    "sheriff_silent",
    "witch_poison_deterrent",
    "required_evaluation",
    "vote_basis_hint",
    "gold_water_duty",
    "unreported_checks",
    "my_check_history",
    # 2026-07-21: must emit target_stance in plan-envelope mode, otherwise
    # _planned_wolf_kill force-strategic_abstain 空刀。提升到 HARD 后会被
    # PromptStrategyMixin 渲染为【硬约束】(MUST) 块，LLM 在 envelope 模式下
    # 会显式产出该字段，planning 层透传修复才能被触发。
    "target_stance_contract",
    # PR1: post-game reflection directive. It carries role-family
    # section headers (【投票错误】 / 【保留的优点】 / 【悍跳分析】) that
    # the aggregation layer parses. Unclassified it fell through to
    # 【参考】 and was rendered as a JSON string value, so the LLM saw
    # a background field instead of MUST text and emitted in-game speech
    # instead of sectioned reflection (game g_415624166, 12/12 no header).
    "reflection_task",
})

SUGGESTION_KEYS: frozenset[str] = frozenset({
    "wolf_speech_directive",
    "villager_speech_directive",
    "seer_speech_directive",
    "witch_speech_directive",
    "hunter_speech_directive",
    "idiot_speech_directive",
    "hybrid_speech_directive",
    "wolf_vote_strategy",
    "wolf_vote_role_hint",
    "wolf_vote_target",
    "seer_vote_strategy",
    "witch_vote_strategy",
    "hunter_vote_strategy",
    "villager_vote_strategy",
    "hybrid_vote_strategy",
    "no_sheriff_vote_hint",
    "good_vote_decision_guard",
    "sheriff_vote_push",
    "speech_originality",
    "anti_following_and_peace_night_rule",
})

REFERENCE_KEYS: frozenset[str] = frozenset({
    "skill_tactical_advice",
    "wolf_day_push_target",
    "wolf_high_priority_target",
    "wolf_plan_target",
    "master_behavior_summary",
    "witch_pressure",
    "witch_strategy_hint",
    "day_discussion_summary",
    "vote_pressure_context",
})
