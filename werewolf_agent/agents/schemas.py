"""Structured output schemas for agent actions.

All player and judge outputs must pass through these schemas.
Enums constrain legal actions and targets per RuleEngine rules.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums — constrain the action space
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    NO_ACTION = "no_action"
    VOTE = "vote"
    WOLF_KILL = "wolf_kill"
    WOLF_NO_KILL = "wolf_no_kill"
    USE_ANTIDOTE = "use_antidote"
    USE_POISON = "use_poison"
    CHECK_ALIGNMENT = "check_alignment"
    CHOOSE_MASTER = "choose_master"
    HUNTER_SHOT = "hunter_shot"
    SELF_DESTRUCT = "self_destruct"
    SHERIFF_REGISTER = "sheriff_register"
    SHERIFF_WITHDRAW = "sheriff_withdraw"
    SHERIFF_VOTE = "sheriff_vote"
    BADGE_TRANSFER = "badge_transfer"
    BADGE_TEAR = "badge_tear"
    SPEECH = "speech"


class SeerStance(str, Enum):
    TRUST = "trust"
    DISTRUST = "distrust"
    UNDECIDED = "undecided"
    NO_CLAIM = "no_claim"


class VoteBasis(str, Enum):
    SEER_CHECK = "seer_check"
    SEER_SIDING = "seer_siding"
    SPEECH_LOGIC = "speech_logic"
    VOTE_PATTERN = "vote_pattern"
    PRESSURE_TEST = "pressure_test"
    ANTI_HERD = "anti_herd"
    FALLBACK = "fallback"


class TaskType(str, Enum):
    SPEECH = "speech"
    VOTE = "vote"
    NIGHT_ACTION = "night_action"
    DECEPTION = "deception"
    LAST_WORDS = "last_words"
    SHERIFF_SPEECH = "sheriff_speech"
    SHERIFF_REGISTRATION = "sheriff_registration"
    DEFENSE_SPEECH = "defense_speech"
    REFLECTION = "reflection"
    WOLF_DISCUSSION = "wolf_discussion"
    HUNTER_SHOT = "hunter_shot"
    PK_SPEECH = "pk_speech"


class OutputMode(str, Enum):
    FULL_ACTION = "full_action"
    TARGET_CHOICE = "target_choice"
    SPEECH_INTENT = "speech_intent"


class FactionGoal(str, Enum):
    PUSH_GOOD_PLAYER_OUT = "push_good_player_out"
    PROTECT_TEAMMATE = "protect_teammate"
    FIND_WOLVES = "find_wolves"
    SURVIVE = "survive"
    HELP_MASTER_FACTION = "help_master_faction"
    CONFUSE_GOOD = "confuse_good"
    DEEP_HOOK = "deep_hook"
    AGGRESSIVE_PUSH = "aggressive_push"


class RiskFlag(str, Enum):
    AVOID_NIGHT_KILL_LEAK = "avoid_night_kill_leak"
    AVOID_TEAMMATE_EXPOSURE = "avoid_teammate_exposure"
    HIGH_VISIBILITY = "high_visibility"
    LOW_TRUST = "low_trust"
    SUSPECTED = "suspected"


# ---------------------------------------------------------------------------
# Private intent — not written to public timeline
# ---------------------------------------------------------------------------

class PrivateIntent(BaseModel):
    """Agent's private strategic snapshot. Only enters debug/audit views."""
    true_role: str = Field(..., description="Agent's actual role")
    faction_goal: FactionGoal = Field(..., description="Current faction objective")
    claimed_view: str = Field(
        ..., description="Identity perspective the agent is claiming publicly"
    )
    pressure_target: str | None = Field(
        None, description="Player the agent intends to pressure"
    )
    risk_flags: list[RiskFlag] = Field(
        default_factory=list, description="Active risk markers"
    )


# ---------------------------------------------------------------------------
# Player action output — schema-constrained
# ---------------------------------------------------------------------------

class ActionTrace(BaseModel):
    """Moderator/audit trace for a model action attempt."""
    raw_text: str = ""
    parsed_action: dict[str, Any] | None = None
    final_action_type: str = ""
    legal_actions: list[str] = Field(default_factory=list)
    legal_targets: list[str] = Field(default_factory=list)
    retry: dict[str, Any] | None = None
    fallback_reason: str | None = None
    # Task 9: Structured output metadata
    tool_call_required: bool = False
    tool_call_received: bool = False
    tool_call_name: str = ""
    parse_success: bool = False
    parse_error: str | None = None
    retry_count: int = 0
    structured_failure_reason: str | None = None


class PlayerAction(BaseModel):
    """Structured output from a player agent. Must pass schema validation."""
    action_type: ActionType
    target_id: str | None = Field(
        None, description="Target player id, must be in legal_targets"
    )
    speech: str = Field(default="", description="Public speech text")
    reason: str = Field(default="", description="Reason for the action")
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence in this action"
    )
    private_intent: PrivateIntent | None = None
    seer_stance: SeerStance = Field(
        default=SeerStance.UNDECIDED,
        description="Vote audit enum: trust, distrust, undecided, or no_claim.",
    )
    vote_basis: VoteBasis = Field(
        default=VoteBasis.FALLBACK,
        description="Vote audit enum: primary structured basis for this vote.",
    )
    standing_with_seer: str = Field(
        default="", description="Private vote audit: seer or logic line the agent stands with"
    )
    suspect_reason: str = Field(
        default="", description="Private vote audit: why the final vote target is suspicious"
    )
    not_voting_reason: str = Field(
        default="", description="Private vote audit: why other major candidates were not selected"
    )
    private_reason: str = Field(
        default="", description="Private vote audit: full non-public reasoning for moderator audit"
    )
    trace: ActionTrace | None = None

    @model_validator(mode="after")
    def validate_target_required(self) -> "PlayerAction":
        target_required = {
            ActionType.VOTE,
            ActionType.WOLF_KILL,
            ActionType.USE_POISON,
            ActionType.CHECK_ALIGNMENT,
            ActionType.CHOOSE_MASTER,
            ActionType.HUNTER_SHOT,
            ActionType.BADGE_TRANSFER,
            ActionType.SHERIFF_VOTE,
        }
        if self.action_type in target_required and self.target_id is None:
            raise ValueError(
                f"action_type={self.action_type.value} requires target_id"
            )
        return self


# ---------------------------------------------------------------------------
# Judge broadcast output
# ---------------------------------------------------------------------------

class JudgeBroadcast(BaseModel):
    """Structured output from the judge agent."""
    broadcast_type: str = Field(..., description="Phase announcement type")
    message: str = Field(..., description="Natural language broadcast")
    phase: str = Field(..., description="Current game phase")
    day_number: int = Field(0, ge=0)
    night_number: int = Field(0, ge=0)
    public_data: dict[str, Any] = Field(
        default_factory=dict, description="Structured public data for this broadcast"
    )


# ---------------------------------------------------------------------------
# Retry / fallback metadata
# ---------------------------------------------------------------------------

class RetryInfo(BaseModel):
    """Tracks retry attempts for illegal/invalid outputs."""
    attempt: int = 1
    max_retries: int = 3
    error_code: str | None = None
    error_message: str | None = None
    correction_hint: str | None = None


class FallbackAction(BaseModel):
    """Fallback when retries are exhausted."""
    action_type: ActionType = ActionType.NO_ACTION
    target_id: str | None = None
    speech: str = ""
    reason: str = "fallback: retries exhausted"
    trace: ActionTrace | None = None


# ---------------------------------------------------------------------------
# Agent context input — what an agent receives
# ---------------------------------------------------------------------------

class AgentContext(BaseModel):
    """Input context for a player or judge agent call."""
    agent_id: str
    task_type: TaskType
    phase: str = ""
    day_number: int = 0
    night_number: int = 0
    public_summary: str = ""
    own_role: str | None = None
    legal_actions: list[ActionType] = Field(default_factory=list)
    legal_targets: list[str] = Field(default_factory=list)
    visible_world_state: dict[str, Any] = Field(default_factory=dict)
    salience_items: list[dict[str, Any]] = Field(default_factory=list)
    rag_hints: list[dict[str, Any]] = Field(default_factory=list)
    private_memory_hints: dict[str, Any] = Field(default_factory=dict)
    reflection_memory_hints: list[dict[str, Any]] = Field(default_factory=list)
    profile_memory_hint: dict[str, Any] = Field(default_factory=dict)
    cognition_matrix_hint: dict[str, Any] = Field(default_factory=dict)
    belief_state: dict[str, Any] = Field(default_factory=dict)
    contradiction_alerts: list[dict[str, Any]] = Field(default_factory=list)
    strategy_directive: dict[str, Any] = Field(default_factory=dict)
    persona_snapshot: dict[str, Any] = Field(default_factory=dict)
    model_config_snapshot: dict[str, Any] = Field(default_factory=dict)
    recent_transcript: list[dict[str, Any]] = Field(default_factory=list)
    output_schema_hint: str = ""
    skill_tools: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Optional skill analysis tools for LLM to call on-demand.",
    )
    skill_analyses: dict[str, str] = Field(
        default_factory=dict,
        description="Pre-computed skill analysis results keyed by tool name.",
    )
    skill_analysis_hints: dict[str, str] = Field(default_factory=dict)
