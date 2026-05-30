"""Tests for agent action schemas and validation.

Covers:
- Player action schema validation (legal/illegal actions and targets)
- Private intent isolation
- Judge broadcast schema
- Default action validator
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from werewolf_agent.agents.schemas import (
    ActionType,
    FactionGoal,
    JudgeBroadcast,
    PlayerAction,
    PrivateIntent,
    RiskFlag,
)
from werewolf_agent.agents.player import (
    DefaultActionValidator,
)


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


class TestPlayerActionSchema:
    def test_valid_vote_action(self) -> None:
        action = PlayerAction(
            action_type=ActionType.VOTE,
            target_id="p07",
            speech="我归7。",
            reason="7的视角不对",
            confidence=0.72,
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
                claimed_view="villager",
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
            public_data={"deaths": [{"player_id": "p03"}]},
        )
        assert b.broadcast_type == "death_announcement"
        assert b.day_number == 2

    def test_minimal_broadcast(self) -> None:
        b = JudgeBroadcast(broadcast_type="phase", message="test", phase="night")
        assert b.day_number == 0


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
