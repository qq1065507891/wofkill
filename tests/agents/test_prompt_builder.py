"""P0-S5 strategy_directive grouping: split keys into [硬约束/建议/参考] sections.

Per consolidated audit (P0-S5):
- Current `_build_strategy_directive` renders the whole `strategy_directive`
  dict as a single flat JSON block, so the LLM has no way to distinguish
  hard constraints (e.g., `must_address_alerts`, `wolf_fake_seer_execution`)
  from soft suggestions (e.g., `anti_herd`, `speech_originality`) from
  background context (e.g., `master_behavior_summary`).
- Game trace g_3528592081 shows p08 wolf actually follows
  `wolf_fake_seer_execution` directive (claim seer) — confirms directives
  are received and acted on, but the LLM has to guess priority.

Fix: render 3 sub-sections in the prompt:
- 【硬约束】 keys: must be obeyed (MUST)
- 【建议】   keys: recommended, may deviate with reason (SHOULD)
- 【参考】   keys: background, optional context (REFERENCE)
"""

from __future__ import annotations

import pytest

from werewolf_agent.agents.prompt_builder import (
    HARD_CONSTRAINT_KEYS,
    REFERENCE_KEYS,
    SUGGESTION_KEYS,
    PlayerPromptBuilder,
)
from werewolf_agent.agents.schemas import (
    ActionType,
    AgentContext,
    RetryInfo,
    TaskType,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_ctx_with_directive(directive: dict) -> AgentContext:
    """Build an AgentContext with the given strategy_directive populated."""
    return AgentContext(
        agent_id="p08",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="werewolf",
        legal_actions=[ActionType.SPEECH],
        legal_targets=[],
        public_summary="D2 wolf speech",
        strategy_directive=directive,
    )


# ---------------------------------------------------------------------------
# Category-membership sanity checks (module-level constants)
# ---------------------------------------------------------------------------


def test_hard_constraint_keys_include_critical_directives():
    """The most binding directives must be in the HARD group, not softer tiers."""
    must_be_hard = {
        "wolf_fake_seer_execution",
        "must_address_alerts",
        "first_night_killed",
        "speech_silent",
        "vote_silent",
        "witch_night_action",
        "role_alerts",
        "vote_pressure",
    }
    for k in must_be_hard:
        assert k in HARD_CONSTRAINT_KEYS, (
            f"{k!r} must be classified as 硬约束 (hard constraint); got none"
        )


def test_suggestion_keys_include_behavioral_directives():
    """Behavioral guidance (recommended but not mandatory) lives in 建议."""
    must_be_suggestion = {
        "wolf_speech_directive",
        "wolf_universal_rules",
        "good_vote_decision_guard",
        "anti_herd",
        "sheriff_vote_push",
        "speech_originality",
        "seer_speech_directive",
        "witch_speech_constraint",
        "anti_following_and_peace_night_rule",
    }
    for k in must_be_suggestion:
        assert k in SUGGESTION_KEYS, (
            f"{k!r} must be classified as 建议 (suggestion); got none"
        )


def test_reference_keys_include_background_context():
    """Background context (read-only) lives in 参考."""
    must_be_reference = {
        "skill_tactical_advice",
        "wolf_day_push_target",
        "wolf_high_priority_target",
        "wolf_plan_target",
        "master_behavior_summary",
        "witch_pressure",
        "witch_strategy_hint",
        "day_discussion_summary",
        "vote_pressure_context",
    }
    for k in must_be_reference:
        assert k in REFERENCE_KEYS, (
            f"{k!r} must be classified as 参考 (reference); got none"
        )


def test_no_key_appears_in_multiple_categories():
    """Each directive key must live in exactly one category."""
    overlap = (
        HARD_CONSTRAINT_KEYS & SUGGESTION_KEYS
        | HARD_CONSTRAINT_KEYS & REFERENCE_KEYS
        | SUGGESTION_KEYS & REFERENCE_KEYS
    )
    assert not overlap, (
        f"Directive keys must be in exactly one category, but found overlap: {overlap}"
    )


# ---------------------------------------------------------------------------
# Prompt-rendering: 3 distinct sections
# ---------------------------------------------------------------------------


def test_strategy_directive_renders_three_section_headers():
    """When directive covers all 3 categories, prompt must have all 3 headers."""
    ctx = _make_ctx_with_directive(
        {
            # 硬约束
            "must_address_alerts": ["p07 accused me"],
            "speech_silent": True,
            # 建议
            "anti_herd": "不要盲目跟票",
            "speech_originality": "避免模板化",
            # 参考
            "master_behavior_summary": "master 上一轮攻击 p05",
            "skill_tactical_advice": {"role": "werewolf"},
        }
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    assert "【硬约束】" in prompt, "Prompt must contain 硬约束 header"
    assert "【建议】" in prompt, "Prompt must contain 建议 header"
    assert "【参考】" in prompt, "Prompt must contain 参考 header"


def test_strategy_directive_only_hard_section_when_only_hard_keys():
    """Only hard keys → only 硬约束 header (建议/参考 headers absent)."""
    ctx = _make_ctx_with_directive(
        {
            "must_address_alerts": ["p07 accused me"],
            "speech_silent": True,
        }
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    assert "【硬约束】" in prompt
    assert "【建议】" not in prompt
    assert "【参考】" not in prompt


def test_strategy_directive_only_suggestion_section_when_only_suggestion_keys():
    """Only suggestion keys → only 建议 header."""
    ctx = _make_ctx_with_directive(
        {
            "anti_herd": "不要盲目跟票",
            "speech_originality": "避免模板化",
        }
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    assert "【硬约束】" not in prompt
    assert "【建议】" in prompt
    assert "【参考】" not in prompt


def test_strategy_directive_only_reference_section_when_only_reference_keys():
    """Only reference keys → only 参考 header."""
    ctx = _make_ctx_with_directive(
        {
            "master_behavior_summary": "master 上一轮攻击 p05",
            "skill_tactical_advice": {"role": "werewolf"},
        }
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    assert "【硬约束】" not in prompt
    assert "【建议】" not in prompt
    assert "【参考】" in prompt


def test_strategy_directive_omits_section_when_empty():
    """Empty directive → no directive section at all (no empty headers)."""
    ctx = _make_ctx_with_directive({})
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    assert "【硬约束】" not in prompt
    assert "【建议】" not in prompt
    assert "【参考】" not in prompt


# ---------------------------------------------------------------------------
# Key placement: each key under the right section
# ---------------------------------------------------------------------------


def test_hard_key_appears_under_hard_section():
    """`must_address_alerts` must be rendered under 【硬约束】, not other sections."""
    ctx = _make_ctx_with_directive(
        {
            "must_address_alerts": ["p07 accused me"],
            # Also add a suggestion key to ensure both sections render
            "anti_herd": "不要盲目跟票",
        }
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    hard_idx = prompt.find("【硬约束】")
    sugg_idx = prompt.find("【建议】")
    assert hard_idx >= 0 and sugg_idx >= 0
    assert hard_idx < sugg_idx, "硬约束 must come before 建议 in the prompt"
    # The hard-section body should mention the must_address_alerts key
    hard_body = prompt[hard_idx:sugg_idx]
    assert "must_address_alerts" in hard_body
    # The hard body should NOT contain keys from other sections
    assert "anti_herd" not in hard_body


def test_suggestion_key_appears_under_suggestion_section():
    """`anti_herd` must be rendered under 【建议】."""
    ctx = _make_ctx_with_directive(
        {
            "anti_herd": "不要盲目跟票",
            # Add a hard key to ensure both sections render
            "must_address_alerts": ["p07 accused me"],
        }
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    hard_idx = prompt.find("【硬约束】")
    sugg_idx = prompt.find("【建议】")
    assert hard_idx >= 0 and sugg_idx >= 0
    sugg_body = prompt[sugg_idx:]
    # Reference section absent, so whole rest of prompt is 建议
    assert "anti_herd" in sugg_body
    # Hard keys must not appear in the suggestion body
    assert "must_address_alerts" not in sugg_body


def test_reference_key_appears_under_reference_section():
    """`skill_tactical_advice` must be rendered under 【参考】."""
    ctx = _make_ctx_with_directive(
        {
            "skill_tactical_advice": {"role": "werewolf"},
            # Add a hard key for context
            "must_address_alerts": ["p07 accused me"],
        }
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    hard_idx = prompt.find("【硬约束】")
    ref_idx = prompt.find("【参考】")
    assert hard_idx >= 0 and ref_idx >= 0
    assert hard_idx < ref_idx
    ref_body = prompt[ref_idx:]
    assert "skill_tactical_advice" in ref_body
    assert "must_address_alerts" not in ref_body


def test_unknown_keys_fall_through_to_reference():
    """Keys not in any category list must be rendered under 【参考】 as fallback.

    Forward-compat: new directive keys added in the future default to 参考
    rather than being silently dropped or mis-grouped as hard constraints.
    """
    ctx = _make_ctx_with_directive(
        {
            "future_directive_xyz": "some new directive",
            "must_address_alerts": ["p07 accused me"],
        }
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    ref_idx = prompt.find("【参考】")
    assert ref_idx >= 0
    ref_body = prompt[ref_idx:]
    assert "future_directive_xyz" in ref_body


# ---------------------------------------------------------------------------
# Hard-section must lead with "必须遵守" framing
# ---------------------------------------------------------------------------


def test_hard_section_includes_must_obey_framing():
    """LLM must see explicit MUST-obey language for the hard section."""
    ctx = _make_ctx_with_directive(
        {"must_address_alerts": ["p07 accused me"]}
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    hard_idx = prompt.find("【硬约束】")
    assert hard_idx >= 0
    # Look for "必须" / "MUST" / "必" within 200 chars after 【硬约束】 header
    window = prompt[hard_idx:hard_idx + 200]
    assert "必须" in window or "必" in window, (
        "Hard section must be framed as mandatory (e.g., '必须遵守')"
    )


# ---------------------------------------------------------------------------
# Section ordering: hard → suggestion → reference
# ---------------------------------------------------------------------------


def test_section_ordering_is_hard_then_suggestion_then_reference():
    """3 sections must appear in priority order: 硬约束 → 建议 → 参考."""
    ctx = _make_ctx_with_directive(
        {
            "must_address_alerts": ["x"],
            "anti_herd": "y",
            "master_behavior_summary": "z",
        }
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    hard_idx = prompt.find("【硬约束】")
    sugg_idx = prompt.find("【建议】")
    ref_idx = prompt.find("【参考】")
    assert hard_idx >= 0 and sugg_idx >= 0 and ref_idx >= 0
    assert hard_idx < sugg_idx < ref_idx, (
        "Sections must appear in priority order: 硬约束 < 建议 < 参考"
    )


# ---------------------------------------------------------------------------
# Smoke: directive section also present (header prefix)
# ---------------------------------------------------------------------------


def test_strategy_directive_still_has_section_prefix():
    """After refactor, the section still introduces itself with a header prefix."""
    ctx = _make_ctx_with_directive(
        {"must_address_alerts": ["p07 accused me"]}
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    # Look for the directive introduction prefix
    assert "策略指令" in prompt or "策略" in prompt, (
        "Prompt should still label the section (e.g., '本轮策略指令')"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
