"""P1-K6: anti-herd warning should only fire for the good side.

Audit finding: `_build_phase_context` in `werewolf_agent/agents/prompt_builder.py`
emits the anti-herd text "反跟票警告：不要无条件跟随任何人的归票..." for EVERY
role, including wolves. But 抱团 (herding) is the wolves' core day-vote
strategy — non-fake_seer wolves are expected to follow the fake_seer's lead
on the kill target. Telling wolves to be wary of herding inverts the wolf
team's actual coordination goal.

Fix: only render the anti-herd text when `own_role` is in
{`villager`, `seer`, `witch`, `hunter`, `idiot`}. For wolves (and the
hybrid-wolf-master case), render a wolf-specific herd message:
"狼队抱团是正常策略；投票时跟队友一致是预期行为；只有在倒钩场景下需独立判断。"
"""

from __future__ import annotations

import pytest

from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
from werewolf_agent.agents.schemas import (
    ActionType,
    AgentContext,
    RetryInfo,
    TaskType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vote_ctx(role: str) -> AgentContext:
    """Build an AgentContext for a vote task, with the given own_role.

    The anti-herd text only renders when VOTE is a legal action, so we
    configure the context for the day-vote phase.
    """
    return AgentContext(
        agent_id="p08",
        task_type=TaskType.VOTE,
        phase="day",
        day_number=2,
        own_role=role,
        legal_actions=[ActionType.VOTE],
        legal_targets=["p01", "p02", "p03"],
        public_summary="D2 vote",
        strategy_directive={},
    )


# ---------------------------------------------------------------------------
# K6.1: anti-herd text appears for good-side roles
# ---------------------------------------------------------------------------

class TestAntiHerdForGoodSide:
    """Good-side roles (villager / seer / witch / hunter / idiot) get
    the anti-herd text in `_build_phase_context`."""

    @pytest.mark.parametrize("role", [
        "villager", "seer", "witch", "hunter", "idiot",
    ])
    def test_anti_herd_text_present_for_good_side(self, role: str):
        ctx = _make_vote_ctx(role)
        prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
        # The pre-fix code emits "反跟票警告" for every role. Post-fix
        # the good-side roles still receive this text.
        assert "反跟票" in prompt, (
            f"good-side role {role!r} should receive anti-herd text. "
            f"Prompt: {prompt!r}"
        )


# ---------------------------------------------------------------------------
# K6.2: anti-herd text does NOT appear for wolves; wolf-specific text does
# ---------------------------------------------------------------------------

class TestAntiHerdForWolves:
    """Wolves (and hybrid-master-wolf) should NOT receive the generic
    anti-herd text. Instead they should receive a wolf-specific herd
    message that frames 抱团 as expected wolf coordination."""

    @pytest.mark.parametrize("role", [
        "werewolf", "hybrid",
    ])
    def test_anti_herd_text_absent_for_wolf_side(self, role: str):
        ctx = _make_vote_ctx(role)
        prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
        # Pre-fix code emits "反跟票警告" for every role. Post-fix wolves
        # and hybrid get a wolf-specific message instead.
        assert "反跟票警告" not in prompt, (
            f"wolf-side role {role!r} should NOT receive the generic "
            f"反跟票警告 text. Prompt: {prompt!r}"
        )

    @pytest.mark.parametrize("role", [
        "werewolf", "hybrid",
    ])
    def test_wolf_specific_herd_message_present(self, role: str):
        """Wolves get a wolf-specific herd/coordination message."""
        ctx = _make_vote_ctx(role)
        prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
        # The new message frames 抱团 as normal wolf strategy and mentions
        # the 倒钩 (deep-hook) exception.
        assert "狼队抱团" in prompt or "抱团" in prompt, (
            f"wolf-side role {role!r} should receive a wolf-specific "
            f"herd message. Prompt: {prompt!r}"
        )


# ---------------------------------------------------------------------------
# K6.3: anti-herd text is gated on role (not on phase)
# ---------------------------------------------------------------------------

class TestAntiHerdRoleGatingContract:
    """Pin down the gating contract: anti-herd text depends on own_role,
    not on phase / task_type. The contract is:
    - own_role ∈ {villager, seer, witch, hunter, idiot}  → anti-herd text
    - own_role ∈ {werewolf, hybrid}                       → wolf-specific text
    - own_role not set / unknown role                     → anti-herd text (safe default)
    """

    def test_anti_herd_gating_is_role_based(self):
        """The same vote task, different roles, different anti-herd
        text — pinning the role-based gating."""
        villager_ctx = _make_vote_ctx("villager")
        wolf_ctx = _make_vote_ctx("werewolf")
        villager_prompt = PlayerPromptBuilder(villager_ctx).build_user_prompt(RetryInfo())
        wolf_prompt = PlayerPromptBuilder(wolf_ctx).build_user_prompt(RetryInfo())
        # The two prompts must differ in the anti-herd section.
        assert "反跟票警告" in villager_prompt
        assert "反跟票警告" not in wolf_prompt
        # The wolf prompt must mention wolf-team coordination.
        assert "狼队" in wolf_prompt or "抱团" in wolf_prompt

    def test_unknown_role_falls_back_to_anti_herd(self):
        """If own_role is None or an unknown string, render the
        anti-herd text (safe default for the good side)."""
        ctx = _make_vote_ctx("unknown_role_xyz")
        prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
        # Unknown role → safe default = anti-herd text.
        assert "反跟票" in prompt, (
            f"unknown role should fall back to anti-herd text (safe default). "
            f"Prompt: {prompt!r}"
        )
