"""Tests for agent action schemas and validation.

Covers:
- Player action schema validation (legal/illegal actions and targets)
- Private intent isolation
- Judge broadcast schema
- Default action validator
- PlayerAction as discriminated Union of 10 action variants
"""

from __future__ import annotations

import importlib

import pytest
from pydantic import ValidationError

from werewolf_agent.agents.schemas import (
    ActionTrace,
    ActionType,
    AgentContext,
    BadgeTransferPlayerAction,
    CheckAlignmentPlayerAction,
    ChooseMasterPlayerAction,
    FactionGoal,
    HunterShotPlayerAction,
    JudgeBroadcast,
    NoOpPlayerAction,
    PlayerAction,
    PrivateIntent,
    RiskFlag,
    SheriffVotePlayerAction,
    SpeechPlayerAction,
    TaskType,
    UsePoisonPlayerAction,
    VotePlayerAction,
    WolfKillPlayerAction,
)
from werewolf_agent.agents.player import (
    DefaultActionValidator,
)


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


def test_schema_groups_remain_compatibly_importable() -> None:
    action_schemas = importlib.import_module("werewolf_agent.agents.action_schemas")
    prompt_schemas = importlib.import_module("werewolf_agent.agents.prompt_schemas")
    trace_schemas = importlib.import_module("werewolf_agent.agents.trace_schemas")

    assert action_schemas.ActionType is ActionType
    assert action_schemas.PlayerAction is PlayerAction
    assert action_schemas.JudgeBroadcast is JudgeBroadcast
    assert prompt_schemas.AgentContext is AgentContext
    assert prompt_schemas.TaskType is TaskType
    assert trace_schemas.ActionTrace is ActionTrace
    assert trace_schemas.PrivateIntent is PrivateIntent


class TestPlayerActionSchema:
    def test_valid_vote_action(self) -> None:
        action = PlayerAction(
            action_type=ActionType.VOTE,
            target_id="p07",
            speech="我归7。",
            reason="7的视角不对",
            confidence=0.72,
            suspect_reason="p07的视角不对",
            not_voting_reason="p08也没充分证据",
            candidate_comparison="p07有公开发言矛盾；p08目前只有跟票嫌疑，证据弱于p07。",
            private_reason="我信p03的查杀",
            private_intent=PrivateIntent(
                true_role="werewolf",
                faction_goal=FactionGoal.PUSH_GOOD_PLAYER_OUT,
                claimed_view="good_player_without_night_info",
                pressure_target="p07",
                risk_flags=[RiskFlag.AVOID_NIGHT_KILL_LEAK],
            ),
        )
        assert action.action_type == ActionType.VOTE
        assert action.target_id == "p07"
        assert action.private_intent is not None
        assert action.private_intent.true_role == "werewolf"

    def test_vote_action_requires_candidate_comparison(self) -> None:
        with pytest.raises(ValidationError, match="candidate_comparison"):
            VotePlayerAction(
                action_type=ActionType.VOTE,
                target_id="p07",
                reason="p07的发言矛盾最多",
                suspect_reason="p07前后逻辑不一致",
                not_voting_reason="p08目前只有轻微跟票嫌疑",
                private_reason="公开证据下p07更可疑",
            )

    def test_action_requiring_target_fails_without_target(self) -> None:
        with pytest.raises(ValidationError, match="requires target_id"):
            PlayerAction(action_type=ActionType.VOTE, speech="归票")

    def test_no_action_does_not_require_target(self) -> None:
        action = PlayerAction(action_type=ActionType.NO_ACTION, speech="过")
        assert action.target_id is None

    def test_confidence_clamped(self) -> None:
        with pytest.raises(ValidationError):
            PlayerAction(
                action_type=ActionType.NO_ACTION,
                confidence=1.5,
            )

    def test_private_intent_isolation(self) -> None:
        """Private intent is stored separately and not in public fields."""
        action = PlayerAction(
            action_type=ActionType.SPEECH,
            speech="我是好人",
            private_intent=PrivateIntent(
                true_role="werewolf",
                faction_goal=FactionGoal.CONFUSE_GOOD,
                claimed_view="good_player_without_night_info",
            ),
        )
        # speech is public, private_intent is private
        assert action.speech == "我是好人"
        assert action.private_intent.true_role == "werewolf"
        # serializing: private_intent should be accessible for audit
        data = action.model_dump()
        assert "private_intent" in data
        assert data["private_intent"]["true_role"] == "werewolf"

    def test_all_action_types_with_required_targets(self) -> None:
        """All target-requiring types fail without target_id."""
        for at in [
            ActionType.WOLF_KILL, ActionType.USE_POISON,
            ActionType.CHECK_ALIGNMENT, ActionType.CHOOSE_MASTER,
            ActionType.HUNTER_SHOT, ActionType.BADGE_TRANSFER,
            ActionType.SHERIFF_VOTE,
        ]:
            with pytest.raises(ValidationError):
                PlayerAction(action_type=at)

    def test_all_action_types_with_optional_targets(self) -> None:
        """These action types work without target_id."""
        for at in [ActionType.NO_ACTION, ActionType.SPEECH,
                    ActionType.SELF_DESTRUCT, ActionType.BADGE_TEAR,
                    ActionType.SHERIFF_REGISTER, ActionType.SHERIFF_WITHDRAW,
                    ActionType.USE_ANTIDOTE]:
            action = PlayerAction(action_type=at)
            assert action.action_type == at


class TestJudgeBroadcastSchema:
    def test_valid_broadcast(self) -> None:
        b = JudgeBroadcast(
            broadcast_type="death_announcement",
            message="昨夜p03倒牌。",
            phase="day",
            day_number=2,
            public_data={"death_count": 1, "death_ids": "p03"},
        )
        assert b.broadcast_type == "death_announcement"
        assert b.day_number == 2

    def test_minimal_broadcast(self) -> None:
        b = JudgeBroadcast(broadcast_type="phase", message="test", phase="night")
        assert b.day_number == 0

    def test_judge_broadcast_rejects_unknown_fields(self) -> None:
        """P2-1: JudgeBroadcast is LLM-generated; unknown fields must raise."""
        with pytest.raises(ValidationError, match="extra_forbidden|Extra"):
            JudgeBroadcast(
                broadcast_type="phase",
                message="x",
                phase="day",
                extra_unwanted_field="value",  # type: ignore[call-arg]
            )

    def test_judge_broadcast_rejects_nested_dict_in_public_data(self) -> None:
        """J-8: public_data values must be scalar (str|int|float|bool), not nested dicts/lists."""
        with pytest.raises(ValidationError, match="dict_value|public_data|Input should be"):
            JudgeBroadcast(
                broadcast_type="death_announcement",
                message="p03倒牌。",
                phase="day",
                public_data={"deaths": [{"player_id": "p03", "reason": "wolf_kill"}]},
            )

    def test_judge_broadcast_rejects_list_value_in_public_data(self) -> None:
        """J-8: list values in public_data are not allowed."""
        with pytest.raises(ValidationError, match="dict_value|public_data|Input should be"):
            JudgeBroadcast(
                broadcast_type="death_announcement",
                message="p03倒牌。",
                phase="day",
                public_data={"deaths": ["p03", "p05"]},
            )

    def test_judge_broadcast_accepts_scalar_public_data(self) -> None:
        """J-8: scalar values (str|int|float|bool) in public_data are allowed."""
        b = JudgeBroadcast(
            broadcast_type="death_announcement",
            message="p03倒牌。",
            phase="day",
            public_data={"death_count": 1, "death_ids": "p03", "is_peaceful": False},
        )
        assert b.public_data == {"death_count": 1, "death_ids": "p03", "is_peaceful": False}


# ---------------------------------------------------------------------------
# P2-1: JudgeBroadcast/RetryInfo/FallbackAction/AgentContext must reject
# unknown fields (extra="forbid").
# ---------------------------------------------------------------------------
#
# Audit P2-1 finding: 4 schema classes were constructed without
# ``extra="forbid"``. JudgeBroadcast is LLM-generated (so unknown
# fields leak LLM-defensive noise into the broadcast payload).
# RetryInfo / FallbackAction / AgentContext are populated by upstream
# code, but without the strict field guard a typo or future regression
# silently writes an unknown key that downstream consumers won't
# notice. Mirror the pattern already applied to PrivateIntent (P1-1)
# and the PlayerAction variants (P0-S8).
#
# AgentContext is constructed in 100+ call sites — adding extra="forbid"
# requires the call-site audit done in P2-1 prep. All 24 kwargs used
# by callers (`agent_id`, `belief_state`, `cognition_matrix_hint`,
# `contradiction_alerts`, `day_number`, `hybrid_master_faction`,
# `legal_actions`, `legal_targets`, `night_number`, `own_role`,
# `persona_snapshot`, `phase`, `private_memory_caveat`,
# `private_memory_hints`, `profile_memory_hint`, `public_summary`,
# `rag_hints`, `recent_transcript`, `reflection_memory_hints`,
# `salience_items`, `skill_analyses`, `skill_analysis_hints`,
# `strategy_directive`, `task_type`, `visible_world_state`) are
# schema fields, so the audit passes.

class TestP2_1ExtraForbid:
    """P2-1: 4 schema classes must reject unknown fields."""

    def test_retry_info_rejects_unknown_fields(self) -> None:
        from werewolf_agent.agents.schemas import RetryInfo
        with pytest.raises(ValidationError, match="extra_forbidden|Extra"):
            RetryInfo(
                attempt=1,
                unknown_field="x",  # type: ignore[call-arg]
            )

    def test_retry_info_reason_codes_default_to_an_empty_list(self) -> None:
        from werewolf_agent.agents.schemas import RetryInfo

        assert RetryInfo().reason_codes == []

    def test_fallback_action_rejects_unknown_fields(self) -> None:
        from werewolf_agent.agents.schemas import FallbackAction
        with pytest.raises(ValidationError, match="extra_forbidden|Extra"):
            FallbackAction(
                action_type=ActionType.NO_ACTION,
                unknown_field="x",  # type: ignore[call-arg]
            )

    def test_agent_context_rejects_unknown_fields(self) -> None:
        """P2-1: AgentContext is synthesized for tests/audit; unknown
        fields must raise. The test only asserts on a synthesized
        instance, not on real call sites (which were audited in P2-1
        prep and confirmed to only use schema-defined keys)."""
        with pytest.raises(ValidationError, match="extra_forbidden|Extra"):
            AgentContext(
                agent_id="p01",
                task_type=TaskType.SPEECH,
                unknown_field="x",  # type: ignore[call-arg]
            )

    def test_agent_context_accepts_known_fields(self) -> None:
        """Regression: all documented AgentContext fields still validate."""
        AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            phase="day",
            day_number=1,
            night_number=0,
            public_summary="x",
            own_role="villager",
            hybrid_master_faction="good",
            legal_actions=[ActionType.VOTE],
            legal_targets=["p07"],
            visible_world_state={},
            salience_items=[],
            rag_hints=[],
            private_memory_hints={},
            private_memory_caveat="",
            reflection_memory_hints=[],
            profile_memory_hint={},
            cognition_matrix_hint={},
            belief_state={},
            contradiction_alerts=[],
            strategy_directive={},
            persona_snapshot={},
            model_config_snapshot={},
            possible_worlds={},
            simulation_predictions={},
            decision_plan_audit={},
            dialogue_plan_audit={},
            recent_transcript=[],
            output_schema_hint="",
            skill_analyses={},
            skill_analysis_hints={},
            rag_anomaly_count=0,
        )

    def test_agent_context_world_model_placeholders_default_empty(self) -> None:
        """Phase 0: future world-model fields exist but default to empty dicts."""
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
        )

        assert ctx.possible_worlds == {}
        assert ctx.simulation_predictions == {}
        assert ctx.decision_plan_audit == {}
        assert ctx.dialogue_plan_audit == {}


# ---------------------------------------------------------------------------
# P2-6: VotePlayerAction must reject empty reason fields
# ---------------------------------------------------------------------------
#
# Audit P2-6 finding: ``suspect_reason`` / ``not_voting_reason`` /
# ``private_reason`` default to ``""``, but the user prompt says
# "理由字段不能写「未说明」" — the validator never enforced the
# rule. A vote with empty reasons slipped through and the audit
# log happily recorded it. We add a model_validator that rejects
# empty-string reason fields when ``action_type == VOTE``.
#
# Note: ``standing_with_seer`` is intentionally NOT in the rejection
# list — a seer stands with their OWN check (own ID is implicit),
# so empty string is the documented default for them. Non-seer roles
# that legitimately have no seer claim to stand with also pass
# empty. Only the three reason fields are required to be non-empty.


class TestVoteActionRejectsEmptyReason:
    """P2-6: VotePlayerAction must reject empty suspect_reason,
    not_voting_reason, private_reason. The prompt explicitly
    forbids writing 「未说明」 — the schema must enforce it.
    """

    def test_vote_action_rejects_empty_reason(self) -> None:
        """P2-6: empty suspect_reason on a vote raises ValidationError."""
        with pytest.raises(ValidationError, match="suspect_reason|non-empty"):
            VotePlayerAction(
                action_type=ActionType.VOTE,
                target_id="p05",
                suspect_reason="",
                not_voting_reason="x",
                private_reason="y",
            )

    def test_vote_action_rejects_empty_not_voting_reason(self) -> None:
        with pytest.raises(ValidationError, match="not_voting_reason|non-empty"):
            VotePlayerAction(
                action_type=ActionType.VOTE,
                target_id="p05",
                suspect_reason="x",
                not_voting_reason="",
                private_reason="y",
            )

    def test_vote_action_rejects_empty_private_reason(self) -> None:
        with pytest.raises(ValidationError, match="private_reason|non-empty"):
            VotePlayerAction(
                action_type=ActionType.VOTE,
                target_id="p05",
                suspect_reason="x",
                not_voting_reason="y",
                private_reason="",
            )

    def test_vote_action_accepts_filled_reasons(self) -> None:
        """Regression: a vote with all three reason fields filled
        must still validate."""
        action = VotePlayerAction(
            action_type=ActionType.VOTE,
            target_id="p05",
            suspect_reason="p05没有回应p03的查杀逻辑",
            not_voting_reason="p07虽然被踩但无明确证据",
            candidate_comparison="p05有查杀压力；p07只有轻微被踩，证据更弱",
            private_reason="心里活动：更信p03的预言家线",
        )
        assert action.suspect_reason == "p05没有回应p03的查杀逻辑"

    def test_vote_action_allows_empty_standing_with_seer(self) -> None:
        """P2-6: ``standing_with_seer`` is intentionally NOT in the
        rejection list — a seer stands with their OWN check (own
        ID is implicit, so empty string is the documented default).
        """
        action = VotePlayerAction(
            action_type=ActionType.VOTE,
            target_id="p05",
            standing_with_seer="",
            suspect_reason="x",
            not_voting_reason="y",
            candidate_comparison="x 比 y 的公开证据更强",
            private_reason="z",
        )
        assert action.standing_with_seer == ""


# ---------------------------------------------------------------------------
# Action validator tests
# ---------------------------------------------------------------------------


class TestDefaultActionValidator:
    def test_legal_action_and_target(self) -> None:
        v = DefaultActionValidator()
        ok, err = v.validate(
            ActionType.VOTE, "p07",
            [ActionType.VOTE, ActionType.NO_ACTION],
            ["p07", "p08"],
        )
        assert ok is True
        assert err is None

    def test_illegal_action(self) -> None:
        v = DefaultActionValidator()
        ok, err = v.validate(
            ActionType.WOLF_KILL, None,
            [ActionType.VOTE],
            [],
        )
        assert ok is False
        assert "not in legal_actions" in (err or "")

    def test_illegal_target(self) -> None:
        v = DefaultActionValidator()
        ok, err = v.validate(
            ActionType.VOTE, "p99",
            [ActionType.VOTE],
            ["p07", "p08"],
        )
        assert ok is False
        assert "not in legal_targets" in (err or "")

    def test_empty_legal_sets_pass(self) -> None:
        """When no legal sets provided, everything passes (for non-LLM modes)."""
        v = DefaultActionValidator()
        ok, _ = v.validate(ActionType.NO_ACTION, None, [], [])
        assert ok is True

    def test_target_requiring_action_rejects_target_when_legal_targets_missing(self) -> None:
        v = DefaultActionValidator()
        ok, err = v.validate(
            ActionType.VOTE,
            "p99",
            [ActionType.VOTE],
            [],
        )
        assert ok is False
        assert "no legal_targets" in (err or "")


# ---------------------------------------------------------------------------
# PlayerAction discriminated Union tests (pipeline-optimization Task 5)
# ---------------------------------------------------------------------------


class TestPlayerActionUnion:
    """PlayerAction is a discriminated Union of action-specific variants.

    Each variant carries only the fields that make sense for that action
    type (e.g. vote_basis only on VotePlayerAction). The discriminator
    is ``action_type`` — the LLM emits it natively, and each variant's
    ``action_type`` field is narrowed to a single ``ActionType`` value.
    """

    def test_vote_action_validates_with_vote_fields(self) -> None:
        action = VotePlayerAction(
            action_type=ActionType.VOTE,
            target_id="p05",
            vote_basis="speech_logic",
            seer_stance="undecided",
            suspect_reason="p05发言矛盾",
            not_voting_reason="p07没有明显证据",
            candidate_comparison="p05有发言矛盾；p07没有明显证据",
            private_reason="我投p05",
        )
        assert action.action_kind == "vote"
        assert action.action_type == ActionType.VOTE

    def test_speech_action_does_not_require_vote_fields(self) -> None:
        action = SpeechPlayerAction(
            action_type=ActionType.SPEECH,
            speech="I think p05 is suspicious",
        )
        assert action.action_kind == "speech"
        # vote-only fields are not present on SpeechPlayerAction
        assert not hasattr(action, "vote_basis") or action.vote_basis is None

    def test_parse_vote_payload_returns_vote_action(self) -> None:
        data = {
            "action_type": "vote",
            "target_id": "p05",
            "vote_basis": "speech_logic",
            "seer_stance": "undecided",
            "suspect_reason": "p05发言矛盾",
            "not_voting_reason": "p07没有明显证据",
            "candidate_comparison": "p05有发言矛盾；p07没有明显证据",
            "private_reason": "我投p05",
        }
        action = PlayerAction.model_validate(data)
        assert isinstance(action, VotePlayerAction)
        assert action.target_id == "p05"

    def test_parse_speech_payload_returns_speech_action(self) -> None:
        data = {
            "action_type": "speech",
            "speech": "hello world",
        }
        action = PlayerAction.model_validate(data)
        assert isinstance(action, SpeechPlayerAction)

    def test_parse_wolf_kill_payload_returns_wolf_kill_action(self) -> None:
        data = {"action_type": "wolf_kill", "target_id": "p09"}
        action = PlayerAction.model_validate(data)
        assert isinstance(action, WolfKillPlayerAction)

    def test_parse_no_action_returns_no_op_action(self) -> None:
        data = {"action_type": "no_action"}
        action = PlayerAction.model_validate(data)
        assert isinstance(action, NoOpPlayerAction)

    def test_direct_call_dispatches_to_variant(self) -> None:
        # `PlayerAction(...)` should route to VotePlayerAction when action_type=vote
        action = PlayerAction(
            action_type=ActionType.VOTE,
            target_id="p07",
            vote_basis="speech_logic",
            seer_stance="undecided",
            suspect_reason="p07的视角不对",
            not_voting_reason="p08没明显证据",
            candidate_comparison="p07视角问题更明显；p08没有同等证据",
            private_reason="我信p03的查杀",
        )
        assert isinstance(action, VotePlayerAction)
        assert action.target_id == "p07"

    def test_missing_action_type_raises(self) -> None:
        # No discriminator → no variant matches → ValidationError
        with pytest.raises(ValidationError):
            PlayerAction.model_validate({"target_id": "p05"})

    def test_invalid_action_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            PlayerAction.model_validate({"action_type": "made_up_action"})

    def test_vote_variant_requires_target_id(self) -> None:
        # target-required variants must still fail without target_id
        with pytest.raises(ValidationError, match="requires target_id"):
            PlayerAction(action_type=ActionType.VOTE, speech="归票")

    def test_speech_variant_does_not_require_target_id(self) -> None:
        action = PlayerAction(action_type=ActionType.SPEECH, speech="过")
        assert isinstance(action, SpeechPlayerAction)
        assert action.target_id is None

    def test_all_ten_variants_exist_and_are_subclasses(self) -> None:
        """Sanity check: all 10 variants are importable and are PlayerAction subclasses."""
        variant_classes = [
            VotePlayerAction, SpeechPlayerAction, WolfKillPlayerAction,
            CheckAlignmentPlayerAction, UsePoisonPlayerAction,
            ChooseMasterPlayerAction, HunterShotPlayerAction,
            BadgeTransferPlayerAction, SheriffVotePlayerAction,
            NoOpPlayerAction,
        ]
        assert len(variant_classes) == 10
        for cls in variant_classes:
            assert issubclass(cls, PlayerAction)

    def test_all_target_requiring_variants_fail_without_target(self) -> None:
        """Vote, wolf_kill, poison, check, choose_master, hunter_shot, badge_transfer, sheriff_vote."""
        target_required = [
            (ActionType.VOTE, "p07"),
            (ActionType.WOLF_KILL, "p09"),
            (ActionType.USE_POISON, "p09"),
            (ActionType.CHECK_ALIGNMENT, "p09"),
            (ActionType.CHOOSE_MASTER, "p09"),
            (ActionType.HUNTER_SHOT, "p09"),
            (ActionType.BADGE_TRANSFER, "p09"),
            (ActionType.SHERIFF_VOTE, "p07"),
        ]
        for action_type, target in target_required:
            with pytest.raises(ValidationError):
                PlayerAction(action_type=action_type)
            # Sanity: with target, it succeeds. Vote also requires
            # the three reason fields (P2-6); fill them in here.
            if action_type == ActionType.VOTE:
                action = PlayerAction(
                    action_type=action_type,
                    target_id=target,
                    suspect_reason="x",
                    not_voting_reason="y",
                    candidate_comparison="x 比 y 的公开证据更强",
                    private_reason="z",
                )
            else:
                action = PlayerAction(action_type=action_type, target_id=target)
            assert action.target_id == target

    def test_all_optional_target_variants_succeed_without_target(self) -> None:
        """Speech, no_action, and other no-target types work without target_id."""
        no_target_types = [
            ActionType.NO_ACTION, ActionType.SPEECH, ActionType.SELF_DESTRUCT,
            ActionType.BADGE_TEAR, ActionType.SHERIFF_REGISTER,
            ActionType.SHERIFF_WITHDRAW, ActionType.USE_ANTIDOTE,
        ]
        for at in no_target_types:
            action = PlayerAction(action_type=at)
            assert action.action_type == at


# ---------------------------------------------------------------------------
# P0-S8: PlayerAction variants reject unknown fields (extra="forbid")
# ---------------------------------------------------------------------------
#
# Game trace g_3528592081 shows 67 successful speech actions containing
# `vote_basis: "fallback"` even though the speech action does not need
# it — the LLM is being defensive. With ``extra="forbid"`` on each
# variant, those deflections are now schema errors that the retry loop
# can report back to the LLM, so it learns to stop filling fields
# the prompt did not ask for.


class TestPlayerActionExtraForbid:
    """PlayerAction variants reject unknown fields.

    Each discriminated-Union variant declares only the fields that
    make sense for that action. With ``extra="forbid"``, an LLM that
    fills in vote-audit fields on a speech action (or any other
    cross-variant field) is rejected at parse time instead of having
    those fields silently dropped.
    """

    def test_speech_action_rejects_vote_basis(self) -> None:
        # P0-S8: SpeechPlayerAction must NOT silently accept vote_basis
        # — game trace g_3528592081 had 67 such actions.
        with pytest.raises(ValidationError, match="extra_forbidden|Extra"):
            SpeechPlayerAction(
                action_type=ActionType.SPEECH,
                speech="hello",
                vote_basis="fallback",  # type: ignore[call-arg]
            )

    def test_speech_action_rejects_seer_stance(self) -> None:
        with pytest.raises(ValidationError, match="extra_forbidden|Extra"):
            SpeechPlayerAction(
                action_type=ActionType.SPEECH,
                speech="hello",
                seer_stance="trust",  # type: ignore[call-arg]
            )

    def test_wolf_kill_action_rejects_seer_stance(self) -> None:
        with pytest.raises(ValidationError, match="extra_forbidden|Extra"):
            WolfKillPlayerAction(
                action_type=ActionType.WOLF_KILL,
                target_id="p05",
                seer_stance="trust",  # type: ignore[call-arg]
            )

    def test_wolf_kill_action_rejects_vote_basis(self) -> None:
        with pytest.raises(ValidationError, match="extra_forbidden|Extra"):
            WolfKillPlayerAction(
                action_type=ActionType.WOLF_KILL,
                target_id="p05",
                vote_basis="fallback",  # type: ignore[call-arg]
            )

    def test_speech_via_model_validate_rejects_vote_basis(self) -> None:
        # End-to-end through PlayerAction.model_validate (the dispatch
        # path used by parse_action). vote_basis on a speech is rejected.
        with pytest.raises(ValidationError, match="extra_forbidden|Extra"):
            PlayerAction.model_validate({
                "action_type": "speech",
                "speech": "hello",
                "vote_basis": "fallback",
            })

    def test_speech_via_model_validate_rejects_seer_stance(self) -> None:
        with pytest.raises(ValidationError, match="extra_forbidden|Extra"):
            PlayerAction.model_validate({
                "action_type": "speech",
                "speech": "hello",
                "seer_stance": "trust",
            })

    def test_wolf_kill_via_model_validate_rejects_vote_basis(self) -> None:
        with pytest.raises(ValidationError, match="extra_forbidden|Extra"):
            PlayerAction.model_validate({
                "action_type": "wolf_kill",
                "target_id": "p05",
                "vote_basis": "fallback",
            })

    def test_no_action_rejects_vote_basis(self) -> None:
        # Sanity: even no-action variants (no_target/no_payload) reject
        # vote-audit fields.
        with pytest.raises(ValidationError, match="extra_forbidden|Extra"):
            PlayerAction.model_validate({
                "action_type": "no_action",
                "vote_basis": "fallback",
            })

    def test_vote_action_accepts_vote_basis(self) -> None:
        # Sanity check: VotePlayerAction is the only variant that
        # accepts vote-audit fields, and it must still work.
        action = VotePlayerAction(
            action_type=ActionType.VOTE,
            target_id="p07",
            vote_basis="seer_check",
            seer_stance="trust",
            standing_with_seer="p03",
            suspect_reason="p07没有回应",
            not_voting_reason="p08没查验",
            candidate_comparison="p07没有回应查杀；p08只是没查验",
            private_reason="我信p03",
        )
        assert action.vote_basis == "seer_check"
        assert action.seer_stance == "trust"

    def test_player_action_base_rejects_unknown(self) -> None:
        # Direct construction on the base class also routes through the
        # Union, so an unknown field is rejected.
        with pytest.raises(ValidationError, match="extra_forbidden|Extra"):
            PlayerAction(
                action_type=ActionType.SPEECH,
                speech="hello",
                random_unknown_field="x",  # type: ignore[call-arg]
            )

    def test_all_target_variants_reject_vote_basis(self) -> None:
        # Sanity sweep: every target-required variant rejects vote_basis.
        for at in [
            ActionType.WOLF_KILL, ActionType.USE_POISON,
            ActionType.CHECK_ALIGNMENT, ActionType.CHOOSE_MASTER,
            ActionType.HUNTER_SHOT, ActionType.BADGE_TRANSFER,
            ActionType.SHERIFF_VOTE,
        ]:
            with pytest.raises(ValidationError, match="extra_forbidden|Extra"):
                PlayerAction.model_validate({
                    "action_type": at.value,
                    "target_id": "p05",
                    "vote_basis": "fallback",
                })


# ---------------------------------------------------------------------------
# P1-1: PrivateIntent must reject unknown fields (extra="forbid")
# ---------------------------------------------------------------------------
#
# Audit P1-1 finding: all PlayerAction variants have ``extra="forbid"`` (P0-S8)
# so the LLM cannot attach arbitrary fields to actions. But the nested
# ``PrivateIntent`` BaseModel has no such guard — the LLM can stuff
# `leaked_secret: "p05 is wolf"` into private_intent and have it pass
# schema validation, then it ends up in the audit log.


class TestPrivateIntentExtraForbid:
    """PrivateIntent is the LLM's free-form audit payload — every variant
    field is a deliberate schema field. With ``extra="forbid"`` an
    LLM that attaches an extra key (defensive field, leaked secret, etc.)
    is rejected at parse time, and the retry loop can surface the
    parse error back to the LLM.
    """

    def test_private_intent_rejects_unknown_fields(self) -> None:
        """P1-1: an unknown key on PrivateIntent must raise ValidationError."""
        with pytest.raises(ValidationError, match="extra_forbidden|Extra"):
            PrivateIntent(
                true_role="werewolf",
                faction_goal=FactionGoal.PUSH_GOOD_PLAYER_OUT,
                claimed_view="good_player_without_night_info",
                some_unknown="x",  # type: ignore[call-arg]
            )

    def test_private_intent_model_validate_rejects_unknown(self) -> None:
        """P1-1: model_validate path also rejects unknown fields."""
        with pytest.raises(ValidationError, match="extra_forbidden|Extra"):
            PrivateIntent.model_validate({
                "true_role": "seer",
                "faction_goal": "find_wolves",
                "claimed_view": "seer",
                "leaked_secret": "p05 is wolf",
            })

    def test_private_intent_known_fields_still_pass(self) -> None:
        """Regression: all 5 documented PrivateIntent fields still validate."""
        intent = PrivateIntent(
            true_role="werewolf",
            faction_goal=FactionGoal.CONFUSE_GOOD,
            claimed_view="good_player_without_night_info",
            pressure_target="p05",
            risk_flags=[RiskFlag.AVOID_TEAMMATE_EXPOSURE],
        )
        assert intent.true_role == "werewolf"
        assert intent.faction_goal == FactionGoal.CONFUSE_GOOD
        assert intent.claimed_view == "good_player_without_night_info"
        assert intent.pressure_target == "p05"
        assert intent.risk_flags == [RiskFlag.AVOID_TEAMMATE_EXPOSURE]


# ---------------------------------------------------------------------------
# P1-3: claimed_view must be enum-enforced (not free-form str)
# ---------------------------------------------------------------------------
#
# Audit P1-3 finding: P0-S7 added a prompt-side enum constraint for
# `claimed_view` (good_player_without_night_info / seer / etc.), but the
# PrivateIntent schema still accepted any string. Game trace g_3528592081
# showed wolves writing `claimed_view: "我是好人，混水摸鱼"` — a natural-
# language strategy note — and the audit log happily recording it. The
# schema must reject values that aren't in the documented enum.
#
# Fix: change `claimed_view: str` to a `Literal[...]` over the
# documented set, so the retry loop can report "claimed_view '我是狼'
# not in enum" back to the LLM and the model learns to stop.

_ALLOWED_CLAIMED_VIEWS = frozenset({
    "good_player_without_night_info",
    "seer", "witch", "hunter", "idiot", "hybrid", "werewolf",
})


class TestClaimedViewEnum:
    """P1-3: claimed_view must be one of the documented enum values."""

    def test_claimed_view_accepts_only_known_values(self) -> None:
        """P1-3: a free-form string like '我是狼' must raise ValidationError."""
        with pytest.raises(ValidationError, match="claimed_view|Input should be"):
            PrivateIntent(
                true_role="werewolf",
                faction_goal=FactionGoal.CONFUSE_GOOD,
                claimed_view="我是狼",
            )

    def test_claimed_view_rejects_strategy_note_string(self) -> None:
        """P1-3: a natural-language strategy note is rejected.

        Mirrors the g_3528592081 leak: wolves wrote
        ``claimed_view: "我是好人，混水摸鱼"`` and the schema accepted it.
        """
        with pytest.raises(ValidationError, match="claimed_view|Input should be"):
            PrivateIntent(
                true_role="werewolf",
                faction_goal=FactionGoal.PUSH_GOOD_PLAYER_OUT,
                claimed_view="我是好人，混水摸鱼",
            )

    def test_claimed_view_accepts_documented_enum_values(self) -> None:
        """Regression: all 7 documented enum values still validate."""
        for allowed in _ALLOWED_CLAIMED_VIEWS:
            intent = PrivateIntent(
                true_role="werewolf",
                faction_goal=FactionGoal.CONFUSE_GOOD,
                claimed_view=allowed,
            )
            assert intent.claimed_view == allowed, (
                f"claimed_view={allowed!r} must validate; got "
                f"{intent.claimed_view!r}"
            )

    def test_claimed_view_model_validate_rejects_free_form(self) -> None:
        """P1-3: model_validate path also rejects free-form claimed_view."""
        with pytest.raises(ValidationError, match="claimed_view|Input should be"):
            PrivateIntent.model_validate({
                "true_role": "seer",
                "faction_goal": "find_wolves",
                "claimed_view": "随便写点啥",  # noqa: E501 — not in enum
            })


class TestActionTraceRetryField:
    """P3-G3223805846-1: ActionTrace 加 total_retry_count_until_success 字段。

    语义：成功路径上 LLM 一共重试了多少次才成功。
    - 第一次尝试就成功 → 0
    - 第 N 次尝试才成功 → N-1
    """

    def test_default_total_retry_count_is_zero(self) -> None:
        """未指定时，total_retry_count_until_success 必须默认为 0。"""
        trace = ActionTrace()
        assert trace.total_retry_count_until_success == 0

    def test_explicit_total_retry_count_round_trips(self) -> None:
        """显式赋值必须可被读取（model_dump 也得带上）。"""
        trace = ActionTrace.model_validate({
            "total_retry_count_until_success": 3,
        })
        assert trace.total_retry_count_until_success == 3
        dumped = trace.model_dump()
        assert dumped["total_retry_count_until_success"] == 3

    def test_total_retry_count_in_model_fields(self) -> None:
        """字段必须声明在 ActionTrace.model_fields 里（避免 typo 静默写入）。"""
        assert "total_retry_count_until_success" in ActionTrace.model_fields
        field = ActionTrace.model_fields["total_retry_count_until_success"]
        # 必须有默认值
        assert field.default == 0
        # 必须是 int 类型
        assert field.annotation is int


def test_action_trace_rejects_boolean_runtime_timeout_count() -> None:
    with pytest.raises(ValidationError, match="runtime_timeout_count"):
        ActionTrace(runtime_timeout_count=True)
