"""Strategy and evaluation functions extracted from agent_adapter.

Pure deterministic scoring helpers — no LLM calls, no side effects.
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

__all__ = [
    "build_witch_pressure_targets",
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
