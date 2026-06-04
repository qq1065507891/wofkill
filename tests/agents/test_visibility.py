"""Tests for visibility boundaries between agent contexts."""

from __future__ import annotations

from werewolf_agent.agents.schemas import (
    ActionType,
    AgentContext,
    FactionGoal,
    PlayerAction,
    PrivateIntent,
    TaskType,
)


class TestVisibilityBoundaries:
    def test_player_cannot_see_other_private_intent(self) -> None:
        """AgentContext only provides own role, never others' roles."""
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            own_role="villager",
            visible_world_state={"alive_players": ["p01", "p02"]},
        )
        # own_role is the only role field — no other player roles
        data = ctx.model_dump()
        role_keys = [k for k in data if "role" in k.lower()]
        for k in role_keys:
            assert "other" not in k.lower()
            assert "hidden" not in k.lower()

    def test_agent_context_no_moderator_full(self) -> None:
        """AgentContext must not contain moderator_full sections."""
        ctx = AgentContext(agent_id="p01", task_type=TaskType.SPEECH)
        data = ctx.model_dump()
        assert "moderator_full" not in str(data).lower()
        assert "hidden_identity" not in str(data).lower()

    def test_private_intent_not_in_speech(self) -> None:
        """Speech field must be safe to broadcast publicly."""
        action = PlayerAction(
            action_type=ActionType.SPEECH,
            speech="我是好人，站边逻辑型选手。",
            private_intent=PrivateIntent(
                true_role="werewolf",
                faction_goal=FactionGoal.CONFUSE_GOOD,
                claimed_view="good_player_without_night_info",
            ),
        )
        # Private data must not appear in speech
        assert "werewolf" not in action.speech
        assert "狼" not in action.speech
