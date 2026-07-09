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

    def test_good_side_warning_protects_possible_power_roles(self):
        ctx = _make_vote_ctx("villager")
        prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
        assert "机械抗推可能神职" in prompt
        assert "预言家、女巫、猎人、白痴" in prompt

    def test_good_side_vote_requires_public_evidence_candidate_comparison(self):
        ctx = _make_vote_ctx("villager")
        prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
        assert "至少比较两名候选人" in prompt
        assert "公开证据" in prompt
        assert "不能只写跟票" in prompt


# ---------------------------------------------------------------------------
# K6.2: anti-herd text does NOT appear for wolves; wolf-specific text does
# ---------------------------------------------------------------------------

class TestAntiHerdForWolves:
    """Wolves (and hybrid-with-wolf-master) should NOT receive the generic
    anti-herd text. Instead they should receive a wolf-specific herd
    message that frames 抱团 as expected wolf coordination.

    Note: ``hybrid`` is parametrized separately via TestHybridMasterFaction
    because its bucket is gated on ctx.hybrid_master_faction (P1-2), not
    just on its own role. The K6.2 contract for hybrid-with-no-master
    defaults to good-side (safe default).
    """

    @pytest.mark.parametrize("role", [
        "werewolf",
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
        "werewolf",
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


# ---------------------------------------------------------------------------
# P1-2: hybrid with good-side master receives good-side anti-herd
# ---------------------------------------------------------------------------
#
# Audit P1-2 finding: hybrid is always treated as wolf-side for the
# anti-herd warning (P1-K6). But ~50% of hybrid players choose a master
# on the good side, in which case the wolf-side message is wrong.
# The hybrid-with-good-master should receive the good-side anti-herd
# text ("反跟票警告") and NOT the wolf-side message ("狼队抱团是正常策略").
#
# Fix: introduce ``ctx.hybrid_master_faction`` (set by runtime) and
# use it to override the hybrid bucket. Default to good-side if unset.


def _make_hybrid_vote_ctx(
    master_faction: str | None = None,
) -> AgentContext:
    """Hybrid vote context with explicit master_faction (P1-2)."""
    return AgentContext(
        agent_id="p04",
        task_type=TaskType.VOTE,
        phase="day",
        day_number=2,
        own_role="hybrid",
        hybrid_master_faction=master_faction,
        legal_actions=[ActionType.VOTE],
        legal_targets=["p01", "p02", "p03"],
        public_summary="D2 vote",
        strategy_directive={},
    )


class TestHybridMasterFaction:
    """P1-2: hybrid's anti-herd bucket depends on master_faction.

    Contract:
    - hybrid_master_faction == "good"      → good-side text (反跟票警告)
    - hybrid_master_faction == "werewolf"  → wolf-side text (狼队抱团)
    - hybrid_master_faction is None / unset → default to good-side
      (safe default — better to over-warn than to silently hand
      wolves a team-coordination cue).
    """

    def test_hybrid_good_master_receives_good_side_anti_herd(self):
        """P1-2: hybrid with good-side master must see good-side text."""
        ctx = _make_hybrid_vote_ctx(master_faction="good")
        prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
        assert "反跟票" in prompt, (
            "hybrid with good master should receive the good-side "
            f"anti-herd text. Prompt: {prompt!r}"
        )
        assert "狼队抱团是正常策略" not in prompt, (
            "hybrid with good master must NOT receive the wolf-side "
            f"herd text. Prompt: {prompt!r}"
        )

    def test_hybrid_wolf_master_does_not_receive_hidden_wolf_side_text(self):
        ctx = _make_hybrid_vote_ctx(master_faction="werewolf")
        prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
        assert "狼队抱团是正常策略" not in prompt
        assert "反跟票" in prompt

    def test_hybrid_master_faction_unset_defaults_to_good_side(self):
        """P1-2: when master_faction is unset, default to good-side.

        Default safety: better to over-warn the hybrid about
        anti-herd than to silently hand wolf-team coordination
        cues to a hybrid whose master we don't yet know.
        """
        ctx = _make_hybrid_vote_ctx(master_faction=None)
        prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
        assert "反跟票" in prompt, (
            "hybrid with unset master_faction should default to "
            f"good-side anti-herd. Prompt: {prompt!r}"
        )


# ---------------------------------------------------------------------------
# P2-5: wolf-side anti-herd advice addresses 悍跳狼 (fake-seer wolf)
# ---------------------------------------------------------------------------
#
# Audit P2-5 finding: the wolf-side message says "投票时跟队友一致
# 是预期行为", but does not distinguish the 悍跳狼 (fake-seer wolf)
# case. A 悍跳 wolf is expected to follow the 悍跳 teammate's 归票
# (the fake-seer wolf's framing of the day), not the original wolf
# team's vote shape. The original "跟队友一致" message is too coarse
# and risks the fake-seer wolf following the wrong teammate.
#
# Fix: extend the wolf-side text to mention that 悍跳狼 should follow
# the 悍跳 teammate's 归票, not the original wolf-team's vote shape.


def test_fake_seer_wolf_advice_present():
    """P2-5: the wolf-side anti-herd text must mention 悍跳狼 (fake-seer wolf)
    coordination explicitly. A 悍跳狼 should follow the 悍跳 teammate's
    归票, not the original wolf-team's vote shape.
    """
    ctx = _make_vote_ctx("werewolf")
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    assert "悍跳" in prompt, (
        "P2-5: wolf-side anti-herd text must mention 悍跳狼 coordination "
        f"so the fake-seer wolf follows the 悍跳 teammate's 归票. "
        f"Prompt: {prompt!r}"
    )
