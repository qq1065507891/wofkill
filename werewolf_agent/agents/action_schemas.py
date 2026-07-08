# -*- coding: utf-8 -*-
"""
提供玩家动作、法官广播和狼队计划兼容导出的结构化 schema。

作者: Project contributors
创建日期: 2026-07-07
修改日期: 2026-07-08

使用示例:
    >>> from werewolf_agent.agents.action_schemas import PlayerAction
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from werewolf_agent.agents.trace_schemas import ActionTrace, PrivateIntent
from werewolf_agent.agents.wolf_team_plan_schema import WolfTeamPlan


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

    @model_validator(mode="after")
    def _validate_reason_fields_non_empty(self) -> "VotePlayerAction":
        # P2-6: the user prompt forbids writing 「未说明」 in the three
        # reason fields — enforce that at the schema level so the
        # retry loop can surface the parse error and the LLM learns
        # to fill in actual reasoning. ``standing_with_seer`` is
        # intentionally NOT validated: a seer stands with their OWN
        # check (own ID is implicit, so empty is the documented
        # default for them) and non-seer roles with no seer claim
        # to stand with also pass empty.
        empty_fields = [
            name
            for name, value in (
                ("suspect_reason", self.suspect_reason),
                ("not_voting_reason", self.not_voting_reason),
                ("private_reason", self.private_reason),
            )
            if not value or not value.strip()
        ]
        if empty_fields:
            raise ValueError(
                "vote action reason fields must be non-empty (the prompt "
                "forbids 「未说明」): " + ", ".join(empty_fields)
            )
        return self


class SpeechPlayerAction(PlayerAction):
    """Public speech action — speech text is the primary payload."""

    model_config = ConfigDict(extra="forbid")
    action_type: Literal[ActionType.SPEECH] = ActionType.SPEECH
    action_kind: Literal["speech"] = "speech"
    intent: str = Field(default="", description="Structured public-speech intent")


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
    # P2-1: LLM-generated; unknown fields must raise. Mirrors the
    # pattern applied to PrivateIntent (P1-1) and the PlayerAction
    # variants (P0-S8). Without this, the judge LLM was stuffing
    # defensive fields (e.g. ``moderator_internal_notes``) into the
    # broadcast payload and the audit log happily recorded them.
    model_config = ConfigDict(extra="forbid")
    broadcast_type: str = Field(..., description="Phase announcement type")
    message: str = Field(..., description="Natural language broadcast")
    phase: str = Field(..., description="Current game phase")
    day_number: int = Field(0, ge=0)
    night_number: int = Field(0, ge=0)
    public_data: dict[str, str | int | float | bool] = Field(
        default_factory=dict, description="Structured public data for this broadcast"
    )


class FallbackAction(BaseModel):
    """Fallback when retries are exhausted."""
    # P2-1: populated by upstream code; unknown fields must raise.
    model_config = ConfigDict(extra="forbid")
    action_type: ActionType = ActionType.NO_ACTION
    target_id: str | None = None
    speech: str = ""
    reason: str = "fallback: retries exhausted"
    trace: ActionTrace | None = None


__all__ = [
    "ActionType",
    "BadgeTearPlayerAction",
    "BadgeTransferPlayerAction",
    "CheckAlignmentPlayerAction",
    "ChooseMasterPlayerAction",
    "FallbackAction",
    "HunterShotPlayerAction",
    "JudgeBroadcast",
    "NoOpPlayerAction",
    "PLAYER_ACTION_VARIANTS",
    "PlayerAction",
    "SeerStance",
    "SelfDestructPlayerAction",
    "SheriffRegisterPlayerAction",
    "SheriffVotePlayerAction",
    "SheriffWithdrawPlayerAction",
    "SpeechPlayerAction",
    "UseAntidotePlayerAction",
    "UsePoisonPlayerAction",
    "VoteBasis",
    "VotePlayerAction",
    "WolfKillPlayerAction",
    "WolfNoKillPlayerAction",
    "WolfTeamPlan",
    "_PLAYER_ACTION_ADAPTER",
]
