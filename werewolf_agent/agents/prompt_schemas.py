# -*- coding: utf-8 -*-
"""
提供代理 prompt 构建所需的任务、输出模式和上下文 schema。

作者: Project contributors
创建日期: 2026-07-07
修改日期: 2026-07-27

使用示例:
    >>> from werewolf_agent.agents.prompt_schemas import AgentContext
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from werewolf_agent.agents.action_schemas import ActionType


class TaskType(str, Enum):
    SPEECH = "speech"
    DISCUSSION_SUMMARY = "discussion_summary"
    VOTE = "vote"
    NIGHT_ACTION = "night_action"
    DECEPTION = "deception"
    LAST_WORDS = "last_words"
    SHERIFF_SPEECH = "sheriff_speech"
    SHERIFF_REGISTRATION = "sheriff_registration"
    DEFENSE_SPEECH = "defense_speech"
    REFLECTION = "reflection"
    WOLF_DISCUSSION = "wolf_discussion"
    WOLF_TEAM_PLAN = "wolf_team_plan"
    HUNTER_SHOT = "hunter_shot"
    PK_SPEECH = "pk_speech"
    JUDGE_PHASE = "judge_phase"
    JUDGE_DEATH = "judge_death"
    JUDGE_VOTE_CALLING = "judge_vote_calling"
    JUDGE_VOTE_TALLY = "judge_vote_tally"
    JUDGE_SKILL_GUIDE = "judge_skill_guide"
    JUDGE_SHERIFF = "judge_sheriff"
    JUDGE_EXILE = "judge_exile"


class OutputMode(str, Enum):
    FULL_ACTION = "full_action"
    TARGET_CHOICE = "target_choice"
    SPEECH_INTENT = "speech_intent"


# ---------------------------------------------------------------------------
# Agent context input — what an agent receives
# ---------------------------------------------------------------------------

class AgentContext(BaseModel):
    """Input context for a player or judge agent call."""
    # P2-1: AgentContext is constructed in 100+ call sites (cognition,
    # runtime, tests). Adding extra="forbid" required auditing every
    # call site — all 24 kwargs used by callers (agent_id,
    # belief_state, cognition_matrix_hint, contradiction_alerts,
    # day_number, hybrid_master_faction, legal_actions, legal_targets,
    # night_number, own_role, persona_snapshot, phase,
    # internal_discussion_summary, private_memory_caveat,
    # private_memory_hints, profile_memory_hint,
    # public_summary, rag_hints, recent_transcript,
    # reflection_memory_hints, salience_items, skill_analyses,
    # skill_analysis_hints, strategy_directive, task_type,
    # visible_world_state) are schema-defined. The strict guard
    # surfaces typos and unintended fields at construction time.
    model_config = ConfigDict(extra="forbid")
    agent_id: str
    task_type: TaskType
    phase: str = ""
    day_number: int = 0
    night_number: int = 0
    public_summary: str = ""
    internal_discussion_summary: str = Field(
        default="",
        exclude=True,
        description="仅供当前玩家决策的讨论总结，不得作为公开证据。",
    )
    own_role: str | None = None
    # P1-2: hybrid's master faction ("good" or "werewolf") — set by
    # runtime from gs.hybrid_master_faction. Controls whether the
    # anti-herd section in the user prompt frames herding as expected
    # (wolf-side) or warns against it (good-side). Unset/None defaults
    # to good-side (safe default — over-warn > silent team-coordination
    # cue leak).
    hybrid_master_faction: str | None = None
    legal_actions: list[ActionType] = Field(default_factory=list)
    legal_targets: list[str] = Field(default_factory=list)
    visible_world_state: dict[str, Any] = Field(default_factory=dict)
    salience_items: list[dict[str, Any]] = Field(default_factory=list)
    rag_hints: list[dict[str, Any]] = Field(default_factory=list)
    private_memory_hints: dict[str, Any] = Field(default_factory=dict)
    # MEM-02: P1-M10 caveat string. Populated by build_agent_context
    # from build_private_memory["_llm_aware_hint"]. The prompt renderer
    # (prompt_builder._build_private_memory_hints) emits this as a
    # separate line BEFORE the logic_flaws / valid_points section so
    # the LLM treats those keyword signals as crude, not authoritative.
    private_memory_caveat: str = ""
    reflection_memory_hints: list[dict[str, Any]] = Field(default_factory=list)
    profile_memory_hint: dict[str, Any] = Field(default_factory=dict)
    cognition_matrix_hint: dict[str, Any] = Field(default_factory=dict)
    # reflect-cross-1: aggregated error pattern across past reflections
    # (e.g. "你最常犯的 2 类错误: vote_mistake(3 次), claim_failed(2 次)")。
    # 不依赖 LLM 解析,纯 section header regex 提取 + 频率统计。
    error_pattern_hint: dict[str, Any] = Field(default_factory=dict)
    belief_state: dict[str, Any] = Field(default_factory=dict)
    contradiction_alerts: list[dict[str, Any]] = Field(default_factory=list)
    seer_credibility: dict[str, Any] = Field(default_factory=dict)
    strategy_directive: dict[str, Any] = Field(default_factory=dict)
    persona_snapshot: dict[str, Any] = Field(default_factory=dict)
    model_config_snapshot: dict[str, Any] = Field(default_factory=dict)
    possible_worlds: dict[str, Any] = Field(default_factory=dict)
    authoritative_world_identities: list[dict[str, Any]] = Field(
        default_factory=list,
        exclude=True,
        description="仅供 moderator 审计重算 possible-world ID 的完整身份快照。",
    )
    public_world_evidence_ids: list[str] = Field(default_factory=list, exclude=True)
    public_claim_ledger: list[dict[str, Any]] = Field(default_factory=list, exclude=True)
    public_fact_ledger: dict[str, list[dict[str, Any]]] = Field(
        default_factory=dict,
        exclude=True,
        description="当前玩家可见的规则事实、声明和冲突结构化投影。",
    )
    simulation_predictions: dict[str, Any] = Field(default_factory=dict)
    decision_plan_audit: dict[str, Any] = Field(default_factory=dict)
    dialogue_plan_audit: dict[str, Any] = Field(default_factory=dict)
    recent_transcript: list[dict[str, Any]] = Field(default_factory=list)
    output_schema_hint: str = ""
    skill_analyses: dict[str, str] = Field(
        default_factory=dict,
        description="Pre-computed skill analysis results keyed by tool name.",
    )
    skill_analysis_hints: dict[str, str] = Field(default_factory=dict)
    decision_identity: Any | None = Field(default=None, exclude=True)
    exposure_collector: Any | None = Field(default=None, exclude=True)
    # P2-G11: counts RAG service anomalies observed while building this
    # context. Increments by 1 per unexpected retrieve_live_hints()
    # failure. Expected misses (rag_service=None, no hits returned) do
    # NOT increment. Used by tests and metrics; not consumed by the
    # prompt renderer.
    rag_anomaly_count: int = 0


__all__ = [
    "AgentContext",
    "OutputMode",
    "TaskType",
]
