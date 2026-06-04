"""Structured output schemas for agent actions.

All player and judge outputs must pass through these schemas.
Enums constrain legal actions and targets per RuleEngine rules.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


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
    # Task 1: Track whether a fallback target was used (decoupled from reason string)
    fallback_target_used: bool = False
    fallback_target_id: str | None = None
    # Task 9: Structured output metadata
    tool_call_required: bool = False
    tool_call_received: bool = False
    tool_call_name: str = ""
    parse_success: bool = False
    parse_error: str | None = None
    retry_count: int = 0
    structured_failure_reason: str | None = None


# ---------------------------------------------------------------------------
# PlayerAction: discriminated Union of action-type-specific variants.
# ---------------------------------------------------------------------------
#
# Pipeline-optimization Task 5: PlayerAction used to be a flat BaseModel
# that forced 6 vote-only fields on every action (vote_basis, seer_stance,
# suspect_reason, etc.). The schema noise increased LLM parse error rate
# and confused the model. We now expose a discriminated Union so each
# variant carries only the fields that make sense for it.
#
# The discriminator is ``action_type`` because:
#   * ActionType is a str-based Enum, so Pydantic v2 accepts both the
#     string form ("vote") and the Enum form (ActionType.VOTE).
#   * The LLM already emits action_type natively, so no extra
#     "action_kind" field needs to be advertised in the tool schema.
#   * Discriminator lookups work whether the payload came from a tool
#     call (action_type present) or from the text-tool fallback path
#     (action_type also present, just maybe as a string).
#
# `PlayerAction(...)` and `PlayerAction.model_validate(...)` are still
# valid call patterns — the ``__new__`` / ``model_validate`` overrides
# below route through the Union's TypeAdapter so existing call sites
# (and isinstance checks) keep working unchanged.


class PlayerAction(BaseModel):
    """Structured output from a player agent. Must pass schema validation.

    This is the base of a discriminated Union: every concrete action
    type is a subclass (``VotePlayerAction``, ``SpeechPlayerAction``,
    ``WolfKillPlayerAction``, ...). Direct construction on this class
    (e.g. ``PlayerAction(action_type=ActionType.VOTE, ...)``) dispatches
    to the right variant via the ``action_type`` discriminator; the
    returned instance is one of the concrete variant classes, so
    ``isinstance`` checks against both this base and any variant class
    keep working unchanged.

    Fields shared by every variant are declared on this base. Each
    variant narrows ``action_type`` to a single ``Literal[ActionType.X]``
    value (so Pydantic can route on it) and adds the audit/reasoning
    fields that only make sense for that action.

    ``action_kind`` is a derived string tag (Literal on each variant)
    exposed for type narrowing and introspection. It is NOT the
    discriminator — we route on ``action_type`` because that is the
    field the LLM already emits natively.

    P0-S8: every variant declares ``extra="forbid"``. Game trace
    ``g_3528592081`` showed 67 successful speech actions containing
    ``vote_basis: "fallback"`` even though the speech action doesn't
    ask for it — the LLM was being defensive. With strict validation,
    such deflections become a parse error that the retry loop can
    surface back to the LLM, so it learns to stop filling in fields
    the prompt never requested.
    """

    model_config = ConfigDict(extra="forbid")

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
    trace: ActionTrace | None = None

    @model_validator(mode="after")
    def _validate_target_required(self) -> "PlayerAction":
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

    def __new__(cls, *args: Any, **data: Any) -> "PlayerAction":
        # Direct construction on the base class dispatches to the right
        # variant via the Union's TypeAdapter. Construction on a
        # specific variant class (e.g. ``VotePlayerAction(...)``) falls
        # through to the normal Pydantic __init__.
        if cls is PlayerAction and (args or data):
            return _PLAYER_ACTION_ADAPTER.validate_python(data)
        return super().__new__(cls)

    @classmethod
    def model_validate(  # type: ignore[override]
        cls,
        *args: Any,
        **kwargs: Any,
    ) -> "PlayerAction":
        # When called on the base PlayerAction class, route through
        # the Union adapter so callers don't need to know which variant
        # they're targeting. Specific variant classes use Pydantic's
        # normal validate path.
        if cls is PlayerAction:
            return _PLAYER_ACTION_ADAPTER.validate_python(*args, **kwargs)
        return super().model_validate(*args, **kwargs)


# Action-type-specific variants. Each one narrows ``action_type`` to a
# single value and adds the audit/reasoning fields that only make sense
# for that action. Variants that share a no-target contract (e.g.
# ``SELF_DESTRUCT``) reuse the same shape via inheritance.
#
# Note: target_id stays ``str | None = None`` on the target-required
# variants too — the base class's ``_validate_target_required`` raises
# the consistent "requires target_id" error when a target-requiring
# action is built without one. Doing it in the validator (instead of
# via a non-Optional field type) preserves the historical error
# message that existing tests assert against.
# ---------------------------------------------------------------------------


class VotePlayerAction(PlayerAction):
    """Day-vote action with the full vote-audit trail (used by the moderator)."""

    model_config = ConfigDict(extra="forbid")
    action_type: Literal[ActionType.VOTE] = ActionType.VOTE
    action_kind: Literal["vote"] = "vote"
    target_id: str | None = None
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


class SpeechPlayerAction(PlayerAction):
    """Public speech action — speech text is the primary payload."""

    model_config = ConfigDict(extra="forbid")
    action_type: Literal[ActionType.SPEECH] = ActionType.SPEECH
    action_kind: Literal["speech"] = "speech"


class WolfKillPlayerAction(PlayerAction):
    """Wolf-team night-kill target."""

    model_config = ConfigDict(extra="forbid")
    action_type: Literal[ActionType.WOLF_KILL] = ActionType.WOLF_KILL
    action_kind: Literal["wolf_kill"] = "wolf_kill"
    target_id: str | None = None


class CheckAlignmentPlayerAction(PlayerAction):
    """Seer night-check target."""

    model_config = ConfigDict(extra="forbid")
    action_type: Literal[ActionType.CHECK_ALIGNMENT] = ActionType.CHECK_ALIGNMENT
    action_kind: Literal["check_alignment"] = "check_alignment"
    target_id: str | None = None


class UsePoisonPlayerAction(PlayerAction):
    """Witch poison target."""

    model_config = ConfigDict(extra="forbid")
    action_type: Literal[ActionType.USE_POISON] = ActionType.USE_POISON
    action_kind: Literal["use_poison"] = "use_poison"
    target_id: str | None = None


class ChooseMasterPlayerAction(PlayerAction):
    """Hybrid first-night master choice target."""

    model_config = ConfigDict(extra="forbid")
    action_type: Literal[ActionType.CHOOSE_MASTER] = ActionType.CHOOSE_MASTER
    action_kind: Literal["choose_master"] = "choose_master"
    target_id: str | None = None


class HunterShotPlayerAction(PlayerAction):
    """Hunter shot target (when triggered by death)."""

    model_config = ConfigDict(extra="forbid")
    action_type: Literal[ActionType.HUNTER_SHOT] = ActionType.HUNTER_SHOT
    action_kind: Literal["hunter_shot"] = "hunter_shot"
    target_id: str | None = None


class BadgeTransferPlayerAction(PlayerAction):
    """Sheriff badge transfer target (when leaving the game)."""

    model_config = ConfigDict(extra="forbid")
    action_type: Literal[ActionType.BADGE_TRANSFER] = ActionType.BADGE_TRANSFER
    action_kind: Literal["badge_transfer"] = "badge_transfer"
    target_id: str | None = None


class SheriffVotePlayerAction(PlayerAction):
    """Sheriff election vote target."""

    model_config = ConfigDict(extra="forbid")
    action_type: Literal[ActionType.SHERIFF_VOTE] = ActionType.SHERIFF_VOTE
    action_kind: Literal["sheriff_vote"] = "sheriff_vote"
    target_id: str | None = None


class NoOpPlayerAction(PlayerAction):
    """Catch-all for actions that carry no target and no public payload.

    Covers the ``ActionType.NO_ACTION`` value (the LLM may emit any of
    several no-target action types — see the additional siblings below
    for the literal action types other than NO_ACTION that we also
    support). The action_type field stays narrowed to a single value
    per variant in the schema registry below so the discriminated
    union can route on it.
    """

    model_config = ConfigDict(extra="forbid")
    action_type: Literal[ActionType.NO_ACTION] = ActionType.NO_ACTION
    action_kind: Literal["no_action"] = "no_action"


# Additional no-target variants. Each narrows ``action_type`` to one
# of the remaining ActionType values so the discriminated union
# accepts them. They all share the same no-target/no-payload shape
# and exist purely to keep the union exhaustive across the full
# ActionType enum.
# ---------------------------------------------------------------------------


class WolfNoKillPlayerAction(PlayerAction):
    """Wolf team explicitly chooses not to kill tonight."""

    model_config = ConfigDict(extra="forbid")
    action_type: Literal[ActionType.WOLF_NO_KILL] = ActionType.WOLF_NO_KILL
    action_kind: Literal["wolf_no_kill"] = "wolf_no_kill"


class UseAntidotePlayerAction(PlayerAction):
    """Witch uses antidote (saves the wolf's victim). Self-targeted by default."""

    model_config = ConfigDict(extra="forbid")
    action_type: Literal[ActionType.USE_ANTIDOTE] = ActionType.USE_ANTIDOTE
    action_kind: Literal["use_antidote"] = "use_antidote"


class SelfDestructPlayerAction(PlayerAction):
    """Self-destruct action (idiot reveal, etc.). No target."""

    model_config = ConfigDict(extra="forbid")
    action_type: Literal[ActionType.SELF_DESTRUCT] = ActionType.SELF_DESTRUCT
    action_kind: Literal["self_destruct"] = "self_destruct"


class SheriffRegisterPlayerAction(PlayerAction):
    """Player registers to run for sheriff."""

    model_config = ConfigDict(extra="forbid")
    action_type: Literal[ActionType.SHERIFF_REGISTER] = ActionType.SHERIFF_REGISTER
    action_kind: Literal["sheriff_register"] = "sheriff_register"


class SheriffWithdrawPlayerAction(PlayerAction):
    """Player withdraws from sheriff candidacy."""

    model_config = ConfigDict(extra="forbid")
    action_type: Literal[ActionType.SHERIFF_WITHDRAW] = ActionType.SHERIFF_WITHDRAW
    action_kind: Literal["sheriff_withdraw"] = "sheriff_withdraw"


class BadgeTearPlayerAction(PlayerAction):
    """Sheriff tears the badge — ends the sheriff role for this game."""

    model_config = ConfigDict(extra="forbid")
    action_type: Literal[ActionType.BADGE_TEAR] = ActionType.BADGE_TEAR
    action_kind: Literal["badge_tear"] = "badge_tear"


# Explicit per-action-type variant registry. Each entry maps a single
# ActionType literal to its variant class. The full Union below is the
# public type; the registry is used by `_PLAYER_ACTION_ADAPTER` and
# also lets callers introspect "which variant handles action_type X".
PLAYER_ACTION_VARIANTS = (
    VotePlayerAction,
    SpeechPlayerAction,
    WolfKillPlayerAction,
    CheckAlignmentPlayerAction,
    UsePoisonPlayerAction,
    ChooseMasterPlayerAction,
    HunterShotPlayerAction,
    BadgeTransferPlayerAction,
    SheriffVotePlayerAction,
    NoOpPlayerAction,
    # The 6 no-target/no-payload siblings above cover the rest of the
    # ActionType enum so the union accepts every legal action value.
    WolfNoKillPlayerAction,
    UseAntidotePlayerAction,
    SelfDestructPlayerAction,
    SheriffRegisterPlayerAction,
    SheriffWithdrawPlayerAction,
    BadgeTearPlayerAction,
)

_PLAYER_ACTION_ADAPTER: TypeAdapter = TypeAdapter(
    Annotated[
        Union[PLAYER_ACTION_VARIANTS],  # type: ignore[valid-type]
        Field(discriminator="action_type"),
    ]
)


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
# Judge structured inputs — per-call payloads for judge broadcast methods
# ---------------------------------------------------------------------------

class JudgeVoteCallingInput(BaseModel):
    """Input for per-player vote calling (唱票)."""
    voter_id: str
    voter_name: str
    candidates: list[str] = Field(default_factory=list)
    position: int = Field(ge=1, description="第N位投票者")
    total: int = Field(ge=1, description="总投票人数")
    day_number: int = 1
    sheriff_weight: float = 1.0


class JudgeSkillGuideInput(BaseModel):
    """Input for role-specific skill guidance dialogue."""
    role: str
    player_id: str
    player_name: str
    available_actions: list[str] = Field(default_factory=list)
    context_hints: dict[str, Any] = Field(default_factory=dict)


class JudgeTallyInput(BaseModel):
    """Input for vote tally announcement."""
    tally: dict[str, float] = Field(default_factory=dict)
    player_names: dict[str, str] = Field(default_factory=dict)
    sheriff_id: str | None = None
    sheriff_weight: float = 1.5
    day_number: int = 1


class JudgeExileInput(BaseModel):
    """Input for exile result announcement."""
    exiled_player_id: str | None = None
    exiled_player_name: str = ""
    reason: str = ""
    tied_player_ids: list[str] = Field(default_factory=list)
    day_number: int = 1


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
    # Pipeline-optimization Task 1: set when the retry loop short-circuits
    # because two consecutive attempts produced the same (error_code,
    # raw_text[:50]) signature. Saves wasted LLM calls when the model is
    # stuck repeating the same broken output.
    early_exit_reason: str | None = None
    # Pipeline-optimization Task 3: attribution for empty_response — one of
    # "timeout", "token_limit", "provider_error", "network_error", "unknown".
    # None when the response was not empty or the cause could not be inferred.
    failure_category: str | None = None


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
    skill_analyses: dict[str, str] = Field(
        default_factory=dict,
        description="Pre-computed skill analysis results keyed by tool name.",
    )
    skill_analysis_hints: dict[str, str] = Field(default_factory=dict)
    # P2-G11: counts RAG service anomalies observed while building this
    # context. Increments by 1 per unexpected retrieve_live_hints()
    # failure. Expected misses (rag_service=None, no hits returned) do
    # NOT increment. Used by tests and metrics; not consumed by the
    # prompt renderer.
    rag_anomaly_count: int = 0
