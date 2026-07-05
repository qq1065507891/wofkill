# -*- coding: utf-8 -*-
"""
功能描述：策略评估函数包——从 agent_adapter 提取的纯确定性评分辅助函数，无 LLM 调用或副作用。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：内部模块，无对外接口
"""
from __future__ import annotations

from werewolf_agent.runtime.strategy.death import evaluate_death_cause_claims
from werewolf_agent.runtime.strategy.hunter import evaluate_hunter_shot_target
from werewolf_agent.runtime.strategy.hybrid import evaluate_hybrid_master_candidates
from werewolf_agent.runtime.strategy.seer import (
    evaluate_seer_check_value,
    public_seer_claimants,
)
from werewolf_agent.runtime.strategy.witch import (
    build_witch_pressure_targets,
    estimate_witch_save_value,
)
from werewolf_agent.runtime.strategy.wolf import (
    evaluate_wolf_kill_target,
    get_wolf_role_assignment,
    has_publicly_claimed_seer,
)
from werewolf_agent.runtime.strategy.poison import collect_witch_poison_candidates

__all__ = [
    "build_witch_pressure_targets",
    "collect_witch_poison_candidates",
    "evaluate_death_cause_claims",
    "evaluate_hunter_shot_target",
    "evaluate_hybrid_master_candidates",
    "evaluate_seer_check_value",
    "evaluate_wolf_kill_target",
    "get_wolf_role_assignment",
    "has_publicly_claimed_seer",
    "public_seer_claimants",
    "estimate_witch_save_value",
]
