# -*- coding: utf-8 -*-
"""
运行时图节点共享的 RuntimeState 类型、默认规则路径和引擎构造。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-23

使用示例:
    >>> from werewolf_agent.runtime.nodes.runtime_state import _new_engine
    >>> _new_engine()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.evaluation.trace_identity import DecisionIdentity
from werewolf_agent.runtime._stable_seed import _stable_seed


RULESET_PATH = str(
    Path(__file__).resolve().parent.parent.parent.parent
    / "config"
    / "rulesets"
    / "pre_witch_hunter_idiot_mixed.yaml"
)


class RuntimeState(TypedDict, total=False):
    game_state: GameState
    engine: RuleEngine
    wolf_kill_target_id: str | None
    wolf_action: str
    wolf_action_reason: str
    use_antidote: bool
    poison_target_id: str | None
    seer_target_id: str | None
    seer_action_trace: dict[str, Any]
    witch_action_trace: dict[str, Any]
    hybrid_master_target_id: str | None
    self_destruct_wolf_id: str | None
    current_speaker_id: str | None
    speech_order: list[str]
    speech_index: int
    speech_text: str
    exile_votes: dict[str, str]
    exile_vote_day: int
    exile_vote_revote: bool
    pk_candidates: list[str]
    vote_action_traces: dict[str, Any]
    vote_decision_identities: dict[str, DecisionIdentity]
    revote: bool
    sheriff_candidates: list[str]
    sheriff_votes: dict[str, str]
    sheriff_withdrawing: list[str]
    badge_decision: str | None
    badge_target_id: str | None
    hunter_shot_target_id: str | None
    agent_registry: Any
    rag_service: Any
    restored_memory: Any
    cognition_state_manager: Any
    consecutive_no_exile_days: int
    wolf_discussion_round: int
    wolf_team_plan: dict[str, Any]
    wolf_consensus_evidence: str
    repository: Any
    discussion_positions_version: int
    discussion_positions: dict[str, dict[str, Any]]
    discussion_summary_audit_records: list[dict[str, str]]
    judge_agent: Any
    judge_llm_enabled: bool
    judge_hitl: Any
    judge_hitl_enabled: bool
    hitl_auto_pause_after: list[str]
    agent_call_delay_ms: int
    action_index_by_game: dict[str, int]
    pending_exposure_events_by_trace: dict[str, list[GameEvent]]
    prompt_proof_key_provider: Any


def _new_engine() -> RuleEngine:
    return RuleEngine.from_yaml(RULESET_PATH)


__all__ = [
    "RULESET_PATH",
    "RuntimeState",
    "_new_engine",
    "_stable_seed",
]
