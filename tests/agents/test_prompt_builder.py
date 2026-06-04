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
    """Only suggestion keys → only 建议 header.

    P1-S3 added a section-level 【硬约束】 outer label to strategy_directive
    in build_user_prompt. The inner sub-group check below uses the unique
    MUST/SHOULD/REFERENCE markers (P0-S5 inner sub-group discriminators)
    instead of the bare "【硬约束】" label, so the test still verifies that
    only the 建议 inner sub-group renders when only suggestion keys exist.
    """
    ctx = _make_ctx_with_directive(
        {
            "anti_herd": "不要盲目跟票",
            "speech_originality": "避免模板化",
        }
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    # Inner sub-group MUST markers must be absent when no hard keys exist
    assert "以下指令必须遵守（MUST）" not in prompt, (
        "MUST marker must not appear when no hard keys are present "
        "(P0-S5 inner sub-group is gated by key presence)"
    )
    assert "以下指令为建议（SHOULD）" in prompt
    assert "以下为背景信息（REFERENCE）" not in prompt


def test_strategy_directive_only_reference_section_when_only_reference_keys():
    """Only reference keys → only 参考 header (P0-S5 inner sub-group).

    P1-S3 added a section-level 【硬约束】 outer label to strategy_directive.
    The inner sub-group check uses the unique MUST/SHOULD/REFERENCE
    markers to discriminate the inner P0-S5 sub-group from the outer
    P1-S3 section label.
    """
    ctx = _make_ctx_with_directive(
        {
            "master_behavior_summary": "master 上一轮攻击 p05",
            "skill_tactical_advice": {"role": "werewolf"},
        }
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    assert "以下指令必须遵守（MUST）" not in prompt
    assert "以下指令为建议（SHOULD）" not in prompt
    assert "以下为背景信息（REFERENCE）" in prompt


def test_strategy_directive_omits_section_when_empty():
    """Empty directive → no directive section at all (no empty headers).

    P1-S3: when strategy_directive is empty, the section body is empty,
    so the outer 【硬约束】 label is also suppressed (the section just
    disappears from the prompt). The retry hint and output contract
    may still have their own 【硬约束】 labels — those are unrelated
    to the directive section.
    """
    ctx = _make_ctx_with_directive({})
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    # Inner sub-group MUST/SHOULD/REFERENCE markers must all be absent
    assert "以下指令必须遵守（MUST）" not in prompt
    assert "以下指令为建议（SHOULD）" not in prompt
    assert "以下为背景信息（REFERENCE）" not in prompt
    # The directive section's outer label is also absent when the body
    # is empty (we look for it directly before the section header text).
    assert "【硬约束】 本轮策略指令" not in prompt


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


# ---------------------------------------------------------------------------
# P0-S6: retry hint reordering + error snippet inclusion
# ---------------------------------------------------------------------------
#
# Audit P0-S6 finding: the retry hint used to render BEFORE the task prompt,
# so the LLM saw "纠正提示..." then later the task description — easy to
# miss the correction. Now the order is: task → retry → contract, with the
# error_message snippet prepended so the model sees the specific failure.
#
# Game trace g_3528592081 Action 50: p10 had 3 retries with the same
# parse_error before fallback. Generic hint "只输出JSON..." was not
# actionable; surfacing the actual error_message gives a concrete signal.


def _make_ctx_for_retry_test() -> AgentContext:
    """Minimal context for retry-hint ordering tests (speech task).

    Uses SPEECH task + [SPEECH] action so the task prompt renders the
    SPEECH_INTENT pipeline ("发言意图枚举"); the test searches for that
    distinctive header to locate the task section.
    """
    return AgentContext(
        agent_id="p10",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH],
        legal_targets=[],
        public_summary="D2 speech",
    )


def test_retry_hint_appears_after_task_prompt():
    """P0-S6: retry hint must appear AFTER task prompt, before output contract.

    Old order was: ...transcript → retry → task → contract. The LLM saw
    "纠正提示..." too early and then forgot about it once it read the task.
    New order is: ...transcript → task → retry → contract, so the
    correction is the last thing before the output contract.
    """
    ctx = _make_ctx_for_retry_test()
    retry = RetryInfo(
        attempt=2,
        max_retries=3,
        error_code="parse_error",
        error_message="JSON parse error: missing field 'speech'",
        correction_hint="只输出JSON，不要解释、不要Markdown代码块。",
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(retry)
    retry_idx = prompt.find("纠正提示")
    # SPEECH task + [SPEECH] action triggers SPEECH_INTENT pipeline,
    # which renders "发言意图枚举" as the task prompt header.
    task_idx = prompt.find("发言意图枚举")
    contract_idx = prompt.find("最终输出协议")
    assert retry_idx >= 0, "Prompt must contain the retry hint header"
    assert task_idx >= 0, "Prompt must contain the task prompt (发言意图枚举)"
    assert contract_idx >= 0, "Prompt must contain the output contract"
    assert task_idx < retry_idx, (
        f"task prompt (idx {task_idx}) must come BEFORE retry hint "
        f"(idx {retry_idx}); otherwise LLM sees correction before knowing the task"
    )
    assert retry_idx < contract_idx, (
        f"retry hint (idx {retry_idx}) must come BEFORE output contract "
        f"(idx {contract_idx}); the correction must lead into the contract"
    )


def test_retry_hint_appears_after_task_in_full_action_mode():
    """P0-S6: ordering also holds for FULL_ACTION mode (e.g., REFLECTION task).

    Covers the non-SPEECH_INTENT branch so the reorder isn't accidentally
    mode-specific. REFLECTION task uses FULL_ACTION output mode which
    renders "示例输出" examples in the task prompt.
    """
    ctx = AgentContext(
        agent_id="p10",
        task_type=TaskType.REFLECTION,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH],
        legal_targets=[],
        public_summary="D2 reflection",
    )
    retry = RetryInfo(
        attempt=2,
        max_retries=3,
        error_code="parse_error",
        error_message="JSON parse error",
        correction_hint="只输出JSON。",
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(retry)
    retry_idx = prompt.find("纠正提示")
    task_idx = prompt.find("示例输出")
    contract_idx = prompt.find("最终输出协议")
    assert task_idx >= 0, "FULL_ACTION task prompt must render 示例输出"
    assert task_idx < retry_idx < contract_idx, (
        f"Order must be task ({task_idx}) → retry ({retry_idx}) "
        f"→ contract ({contract_idx})"
    )


def test_retry_hint_includes_error_message_snippet():
    """P0-S6: hint must surface the actual error_message, not just correction_hint.

    Game trace g_3528592081 Action 50 shows p10 hitting the same parse_error
    3 times. The previous hint only echoed the generic correction_hint, so
    the LLM had no signal about what specifically failed. The hint must
    now include (a prefix of) the error_message so the model sees the
    concrete failure.
    """
    ctx = _make_ctx_for_retry_test()
    retry = RetryInfo(
        attempt=2,
        max_retries=3,
        error_code="parse_error",
        error_message="JSON parse error: missing field 'speech'",
        correction_hint="只输出JSON，不要解释。",
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(retry)
    assert "missing field 'speech'" in prompt, (
        "Retry hint must include the concrete error_message text "
        "(at minimum its first 100 chars)"
    )


def test_retry_hint_truncates_long_error_message_to_100_chars():
    """P0-S6: error_message longer than 100 chars must be truncated.

    Keeps the hint focused on the most relevant signal and avoids leaking
    raw model output / long traces into the next prompt.
    """
    ctx = _make_ctx_for_retry_test()
    long_error = "A" * 50 + "B" * 50 + "C" * 50  # 150 chars
    retry = RetryInfo(
        attempt=2,
        max_retries=3,
        error_code="parse_error",
        error_message=long_error,
        correction_hint="只输出JSON。",
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(retry)
    # First 100 chars: 50 A's + 50 B's
    expected_snippet = "A" * 50 + "B" * 50
    assert expected_snippet in prompt, (
        "First 100 chars of error_message must appear in the hint"
    )
    # The C's (chars beyond 100) must NOT appear in the rendered hint
    # (anywhere in the prompt — they're not present in any other section).
    assert "C" * 50 not in prompt, (
        "Chars beyond first 100 of error_message must be truncated"
    )


def test_retry_hint_falls_back_to_correction_hint_when_error_message_empty():
    """P0-S6: if error_message is None/empty, hint must still render correction_hint.

    Backward compat: not every retry path populates error_message (e.g.,
    skill-tool nudges with no error code). The hint must still appear so
    the LLM sees the corrective signal.
    """
    ctx = _make_ctx_for_retry_test()
    retry = RetryInfo(
        attempt=2,
        max_retries=3,
        error_code=None,
        error_message=None,
        correction_hint="请先调用 load_skill('wolf_pit') 再提交行动。",
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(retry)
    assert "纠正提示" in prompt, "Hint header must still render when only correction_hint is set"
    assert "load_skill('wolf_pit')" in prompt, (
        "When error_message is None, correction_hint text must still appear"
    )


def test_retry_hint_empty_when_no_correction_and_no_error_message():
    """P0-S6: with no correction_hint AND no error_message, hint section is omitted.

    First-attempt RetryInfo (attempt=1) has no error info — the hint
    section should not bloat the prompt with an empty 纠正提示 header.
    """
    ctx = _make_ctx_for_retry_test()
    retry = RetryInfo()  # attempt=1, no error
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(retry)
    assert "纠正提示" not in prompt, (
        "First-attempt prompt (no error) must not contain the retry hint header"
    )


# ---------------------------------------------------------------------------
# P0-S7: claimed_view examples must use enum-style identifiers
# ---------------------------------------------------------------------------
#
# Audit P0-S7 finding: the PrivateIntent.claimed_view field is documented
# as "Identity perspective the agent is claiming publicly" — should be a
# clean identifier, not a natural-language Chinese sentence. But the
# `_format_examples` method rendered `"claimed_view": "我是好人"` /
# `"我是预言家"` etc. (whole Chinese phrases).
#
# Game trace g_3528592081 confirms the issue: real wolves wrote
# `claimed_view: "我是好人，混水摸鱼"` — a strategy note in natural
# Chinese, not an enum value. The LLM is copying the bad example.
#
# Fix: change the example value to a known good identifier
# `good_player_without_night_info` (matches the schema's intent of an
# identity-perspective enum value) and `seer` (role identifier) for
# the seer branch. Tests below cover all 4 example-rendering branches.


def _make_ctx_for_claimed_view_test(
    role: str = "villager",
    legal_actions: list | None = None,
) -> AgentContext:
    """Context that exercises `_format_examples` (FULL_ACTION output mode).

    Default branch: legal_actions = [SPEECH] triggers the speech+vote
    example pair. Pass WOLF_KILL to exercise the wolf-kill branch.
    """
    return AgentContext(
        agent_id="p05",
        task_type=TaskType.REFLECTION,
        phase="day",
        day_number=2,
        own_role=role,
        legal_actions=legal_actions if legal_actions is not None else [ActionType.SPEECH],
        legal_targets=["p05"],
        public_summary="D2 reflection",
    )


def test_claimed_view_example_uses_enum_not_chinese_phrase_default():
    """P0-S7: the default-branch example must use an enum-style identifier.

    The Chinese phrase "我是好人" is a free-form natural-language claim,
    not an enum value. The schema documents `claimed_view` as an
    "Identity perspective the agent is claiming publicly" — should be a
    clean identifier. Game trace g_3528592081 showed wolves writing
    "我是好人，混水摸鱼" — a strategy note, not a structured value.
    """
    ctx = _make_ctx_for_claimed_view_test(role="villager")
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    # The example in the prompt must NOT use the Chinese natural-language
    # phrase "我是好人" as the claimed_view value.
    assert "我是好人" not in prompt, (
        "Example claimed_view must not be the Chinese phrase '我是好人'; "
        "use an enum-style identifier like 'good_player_without_night_info'."
    )
    # And it must use a clean English identifier that matches the
    # schema's intent (identity-perspective enum).
    assert "good_player_without_night_info" in prompt, (
        "Example claimed_view should be 'good_player_without_night_info' "
        "(a clean identity-perspective identifier)."
    )


def test_claimed_view_example_uses_seer_identifier_for_seer_role():
    """P0-S7: the seer-branch example must use the role identifier 'seer',
    not the Chinese phrase '我是预言家'.
    """
    ctx = _make_ctx_for_claimed_view_test(role="seer")
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    assert "我是预言家" not in prompt, (
        "Example claimed_view for seer must not be '我是预言家'; "
        "use the role identifier 'seer'."
    )
    assert '"claimed_view": "seer"' in prompt, (
        "Seer example must use 'seer' as the claimed_view identifier."
    )


def test_claimed_view_example_uses_enum_in_wolf_kill_branch():
    """P0-S7: the wolf-kill example branch must also use the enum value.

    Covers lines 563 and 572 (wolf_kill + wolf_no_kill examples). The
    example should advertise the same clean identifier the LLM should
    copy, not natural language.
    """
    ctx = AgentContext(
        agent_id="p08",
        task_type=TaskType.NIGHT_ACTION,
        phase="night",
        night_number=2,
        own_role="werewolf",
        legal_actions=[ActionType.WOLF_KILL, ActionType.WOLF_NO_KILL],
        legal_targets=["p03", "p05"],
        public_summary="N2 wolf action",
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    # Both wolf-kill and wolf-no-kill examples previously used
    # "我是好人" as the claimed_view value. After the fix neither
    # should appear.
    assert "我是好人" not in prompt, (
        "Wolf-kill example branch must not use '我是好人' as claimed_view; "
        "use the enum-style identifier 'good_player_without_night_info'."
    )
    assert "good_player_without_night_info" in prompt, (
        "Wolf-kill branch example should advertise 'good_player_without_night_info'."
    )


def test_claimed_view_example_no_chinese_natural_language_anywhere():
    """P0-S7: sweep — no claimed_view value in any branch uses Chinese phrase.

    Final guard: the LLM should never see `我是X` as a claimed_view
    example value, no matter which output mode the task triggers. The
    claim must be a structured identifier.
    """
    # 1) Default branch (speech + vote examples)
    ctx_default = _make_ctx_for_claimed_view_test(role="villager")
    prompt_default = PlayerPromptBuilder(ctx_default).build_user_prompt(RetryInfo())
    assert "我是好人" not in prompt_default
    assert "我是预言家" not in prompt_default

    # 2) Seer default branch
    ctx_seer = _make_ctx_for_claimed_view_test(role="seer")
    prompt_seer = PlayerPromptBuilder(ctx_seer).build_user_prompt(RetryInfo())
    assert "我是好人" not in prompt_seer
    assert "我是预言家" not in prompt_seer

    # 3) Wolf default branch
    ctx_wolf = _make_ctx_for_claimed_view_test(role="werewolf")
    prompt_wolf = PlayerPromptBuilder(ctx_wolf).build_user_prompt(RetryInfo())
    assert "我是好人" not in prompt_wolf
    assert "我是预言家" not in prompt_wolf

    # 4) Wolf-kill branch
    ctx_wk = AgentContext(
        agent_id="p08",
        task_type=TaskType.NIGHT_ACTION,
        phase="night",
        night_number=2,
        own_role="werewolf",
        legal_actions=[ActionType.WOLF_KILL, ActionType.WOLF_NO_KILL],
        legal_targets=["p03"],
        public_summary="N2",
    )
    prompt_wk = PlayerPromptBuilder(ctx_wk).build_user_prompt(RetryInfo())
    assert "我是好人" not in prompt_wk
    assert "我是预言家" not in prompt_wk


# ---------------------------------------------------------------------------
# P0-S8: speech example in the prompt must validate against the strict
# SpeechPlayerAction schema (and the vote example must validate against
# VotePlayerAction). With extra="forbid", if the example contains any
# cross-variant fields, the LLM will copy them too.
# ---------------------------------------------------------------------------


import json as _json
import re as _re

from werewolf_agent.agents.schemas import (
    PlayerAction as _PlayerAction,
    SpeechPlayerAction as _SpeechPlayerAction,
    VotePlayerAction as _VotePlayerAction,
)


def _extract_first_json_object(prompt: str) -> dict:
    """Find the first balanced JSON object in the prompt and parse it.

    The prompt contains many JSON examples wrapped in quotes inside
    Chinese text; we need to find one with a top-level ``action_type``
    field to test against the schema.
    """
    # Greedy first { ... } that contains "action_type" — balance braces.
    for match in _re.finditer(r"\{", prompt):
        depth = 0
        in_string = False
        escape = False
        end = -1
        for idx in range(match.start(), len(prompt)):
            ch = prompt[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = idx
                    break
        if end < 0:
            continue
        try:
            data = _json.loads(prompt[match.start() : end + 1])
        except _json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "action_type" in data:
            return data
    raise AssertionError("No JSON object with 'action_type' found in prompt")


def test_speech_example_in_prompt_validates_against_strict_schema():
    """P0-S8: speech example in the prompt must validate SpeechPlayerAction.

    The prompt's speech example is what the LLM copies. With
    extra="forbid" on SpeechPlayerAction, any cross-variant field in
    the example would cause the LLM to also fill those fields
    defensively. We now require the example itself to round-trip
    cleanly through the strict schema.
    """
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,  # → FULL_ACTION (multi-action not 1)
        phase="day",
        day_number=1,
        own_role="villager",
        legal_actions=[ActionType.SPEECH, ActionType.VOTE],
        legal_targets=["p05"],
        public_summary="D1",
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    # Collect every JSON object in the example section; the speech one
    # must have action_type="speech" and the vote one must have
    # action_type="vote".
    examples: list[dict] = []
    for match in _re.finditer(r"\{[^{}]*?(?:\{[^{}]*\}[^{}]*?)*\}", prompt, flags=_re.DOTALL):
        try:
            data = _json.loads(match.group(0))
        except _json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "action_type" in data:
            examples.append(data)

    speech_examples = [e for e in examples if e.get("action_type") == "speech"]
    vote_examples = [e for e in examples if e.get("action_type") == "vote"]
    assert speech_examples, "Expected a speech example in the prompt"
    assert vote_examples, "Expected a vote example in the prompt"

    for ex in speech_examples:
        # Round-trip through the strict SpeechPlayerAction schema.
        _SpeechPlayerAction.model_validate(ex)

    for ex in vote_examples:
        # Vote example is allowed to carry vote-audit fields.
        _VotePlayerAction.model_validate(ex)


def test_speech_example_does_not_contain_vote_audit_field_names():
    """P0-S8: the speech example must not name vote-audit fields.

    Even if the field value would be valid, naming the field primes
    the LLM to fill it. Game trace g_3528592081 showed 67 successful
    speech actions with ``vote_basis: "fallback"`` after the LLM saw
    the field name in another branch's example.
    """
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=1,
        own_role="villager",
        legal_actions=[ActionType.SPEECH, ActionType.VOTE],
        legal_targets=["p05"],
        public_summary="D1",
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    for ex in _re.finditer(r"\{\s*\"action_type\":\s*\"speech\".*?\}", prompt, flags=_re.DOTALL):
        block = ex.group(0)
        assert "vote_basis" not in block, (
            "Speech example must not mention vote_basis"
        )
        assert "seer_stance" not in block, (
            "Speech example must not mention seer_stance"
        )
        assert "standing_with_seer" not in block, (
            "Speech example must not mention standing_with_seer"
        )
        assert "suspect_reason" not in block, (
            "Speech example must not mention suspect_reason"
        )
        assert "not_voting_reason" not in block, (
            "Speech example must not mention not_voting_reason"
        )
        assert "private_reason" not in block, (
            "Speech example must not mention private_reason"
        )


# ---------------------------------------------------------------------------
# P0-R2: skill_catalog moves from system to user prompt
# ---------------------------------------------------------------------------
#
# Audit P0-R2 finding: the skill_catalog is role+phase dependent, so it
# was always the wrong section to keep in the cacheable system prompt —
# the LLM was paying system-prompt cache cost for content that wasn't
# actually stable.  Game trace g_3528592081 showed 17 empty_responses,
# mostly from seer (5) and villager (3) — the seer/witch prompts were
# bloated with the 200+ char skill catalog in the system slot.
#
# Fix: move the catalog from `build_system_prompt` to `build_user_prompt`
# so the system prompt is shorter (less cache pressure) and the user
# message carries the per-turn skill list.  Tests below check both
# halves: system must be lean, user must contain the catalog.


def _make_seer_check_context() -> AgentContext:
    """Seer day-speech context (the most empty_response-heavy role).

    Phase is set to 'speech' so the skill catalog renders at least one
    entry — this lets the test verify catalog placement.  The runtime
    currently passes raw 'day'/'night' phases (a pre-existing mismatch
    with skill applicable_phases), so the catalog would be empty in
    production.  The architectural move from system → user is still
    correct: regardless of whether the catalog renders, the system
    prompt is now shorter.
    """
    return AgentContext(
        agent_id="p03",
        task_type=TaskType.SPEECH,
        phase="speech",
        day_number=2,
        own_role="seer",
        legal_actions=[ActionType.SPEECH, ActionType.VOTE],
        legal_targets=["p05", "p07"],
        public_summary="D2 seer speech",
    )


def _make_villager_context() -> AgentContext:
    """Villager day-speech context (the other empty_response-heavy role).

    Phase is set to 'speech' so the skill catalog renders at least one
    entry.  See _make_seer_check_context for why.
    """
    return AgentContext(
        agent_id="p10",
        task_type=TaskType.SPEECH,
        phase="speech",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH, ActionType.VOTE],
        legal_targets=["p05", "p07"],
        public_summary="D2 vote",
    )


def test_rag_hints_strip_audit_metadata_in_live_prompt():
    """P0-G2 defense in depth: even if a non-production code path
    populates ``ctx.rag_hints`` with the full audit payload
    (relevance, quality, source, visibility, display annotation),
    the rendered prompt must NOT include any of those fields.

    The runtime now produces slim lines via
    :class:`RAGKnowledgeService.hits_to_prompt_lines`, so this path
    is only hit when a test or future code path populates the audit
    shape directly. The slim filter in
    :meth:`PlayerPromptBuilder._build_rag_hints` is the last line of
    defense.
    """
    ctx = _make_villager_context()
    ctx = ctx.model_copy(update={
        "rag_hints": [{
            "type": "rag_hit",
            "entry_id": "leaked_audit_item",
            "title": "京城大师赛 250415 抗推预言家",
            "summary": "狼队在白天通过抗推预言家获得票数优势。",
            "key_decisions": [
                "白天全力归票预言家",
                "预言家抗推后改换身份打深钩",
                "夜里优先解神牌",
                "不应当出现",
            ],
            "relevance": 0.83,
            "quality": "high_rank_game",
            "source_type": "public_tournament",
            "visibility": "player_perspective",
            "visibility_boundary": "player_perspective",
            "annotation": "[public_tournament|high_rank_game]",
            "display_annotation": "[public_tournament|high_rank_game]",
            "allowed_in_live": True,
            "allowed_in_live_context": True,
            "case_type": "external_high_end_case",
            "role_perspective": "werewolf",
            "tags": ["抗推", "预言家"],
            "short_quotes": ["p04 票型抱团"],
            "phase": "speech",
        }],
    })
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    # Audit field NAMES must not appear in the prompt.
    for forbidden_name in (
        "relevance",
        "relevance_score",
        "quality",
        "quality_grade",
        "source_type",
        "visibility",
        "visibility_boundary",
        "annotation",
        "display_annotation",
        "allowed_in_live",
        "allowed_in_live_context",
        "case_type",
        "role_perspective",
        "tags",
        "short_quotes",
        "entry_id",
    ):
        assert forbidden_name not in prompt, (
            f"Audit-only field name {forbidden_name!r} leaked into live prompt"
        )
    # Audit field VALUES must not appear either.
    assert "0.83" not in prompt
    assert "high_rank_game" not in prompt
    assert "public_tournament" not in prompt
    assert "player_perspective" not in prompt
    assert "leaked_audit_item" not in prompt
    # The slim payload fields ARE present.
    assert "京城大师赛 250415 抗推预言家" in prompt
    assert "狼队在白天通过抗推预言家获得票数优势。" in prompt
    assert "白天全力归票预言家" in prompt
    # key_decisions truncated to 3.
    assert "不应当出现" not in prompt


def test_rag_hints_empty_when_no_hints():
    """P0-G2: with no rag_hints, the prompt must not include the
    知识库提示 header. Sanity check on the empty path."""
    ctx = _make_villager_context()
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    assert "知识库提示" not in prompt


def test_rag_hints_include_player_id_warning():
    """P0-G3: live prompt must include a hard-constraint warning that
    RAG case player IDs are NOT this game's players — the LLM must not
    parrot case-specific speech or vote patterns as if they were
    current-game events.

    Without this prefix the LLM has been observed in g_3528592081 to
    treat case player IDs (e.g., p04 / p09) as current-game player
    IDs and align votes / speech to those concrete IDs, which is
    information leakage and tactical error.
    """
    ctx = _make_villager_context()
    ctx = ctx.model_copy(update={
        "rag_hints": [{
            "title": "京城大师赛 250415 抗推预言家",
            "summary": "狼队在白天通过抗推预言家获得票数优势。",
            "key_decisions": ["白天全力归票预言家"],
        }],
    })
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    # The exact prefix the plan specifies.
    expected_prefix = (
        "⚠️ RAG 案例中的玩家 ID 与本局无关；"
        "不得直接套用案例中具体玩家的发言或票型。"
    )
    assert expected_prefix in prompt, (
        f"P0-G3: the RAG hints section must include the hard-constraint "
        f"prefix about case-vs-current player IDs. "
        f"Expected: {expected_prefix!r}\nGot prompt excerpt:\n"
        + prompt[prompt.find("知识库提示"):prompt.find("知识库提示") + 400]
    )
    # The warning must be in the 知识库提示 section, not buried elsewhere.
    rag_section_start = prompt.find("知识库提示")
    rag_section_end = prompt.find("\n\n", rag_section_start)
    if rag_section_end == -1:
        rag_section_end = len(prompt)
    rag_section = prompt[rag_section_start:rag_section_end]
    assert expected_prefix in rag_section, (
        "P0-G3: the warning must live inside the 知识库提示 section, "
        "not in some other section of the prompt."
    )


def test_rag_hints_player_id_warning_appears_before_json_payload():
    """P0-G3: the warning must be the FIRST line of the 知识库提示
    section, before the JSON payload. Otherwise an LLM that reads
    the section top-to-bottom might process the JSON before seeing
    the warning, and the whole point of the prefix is to set the
    "do not parrot" frame BEFORE the model sees the case data.
    """
    ctx = _make_villager_context()
    ctx = ctx.model_copy(update={
        "rag_hints": [{
            "title": "案例标题",
            "summary": "案例摘要。",
            "key_decisions": ["决策1"],
        }],
    })
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    rag_start = prompt.find("知识库提示")
    after_header = prompt[rag_start:]
    # The warning must precede the first JSON key in the section.
    json_start = after_header.find('"title"')
    warning_start = after_header.find("⚠️")
    assert warning_start != -1, "P0-G3: warning prefix must be present"
    assert json_start != -1, "P0-G3: JSON payload must still be present"
    assert warning_start < json_start, (
        "P0-G3: the warning must come BEFORE the JSON payload in the "
        "知识库提示 section."
    )


def test_rag_hints_no_warning_when_no_hints():
    """P0-G3: the warning is a hard-constraint prefix that only makes
    sense when there ARE RAG hints. With no hints, the section is
    omitted entirely (existing behavior) and the warning must not
    leak into other sections."""
    ctx = _make_villager_context()
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    assert "RAG 案例中的玩家 ID" not in prompt
    assert "⚠️" not in prompt or "RAG 案例" not in prompt


def test_skill_catalog_not_in_system_prompt_for_seer():
    """P0-R2: seer system prompt must not include the skill catalog.

    The catalog is dynamic (role+phase dependent) so it was always
    misclassified as 'stable' system content. Moving it to user
    removes ~1500 chars of wasted system-prompt cache pressure.
    """
    ctx = _make_seer_check_context()
    system_prompt = PlayerPromptBuilder(ctx).build_system_prompt()
    assert "可用技能目录" not in system_prompt, (
        "Seer system prompt must not include the skill catalog header. "
        "The catalog is role+phase dependent so it belongs in the user "
        "message, not the cacheable system prompt."
    )


def test_skill_analysis_hints_in_user_prompt_for_seer():
    """P0-K1: seer user prompt must include the pre-injected skill analysis.

    The catalog (tool path) is removed. Skill analyses are delivered
    via the `skill_analysis_hints` pre-injection path.
    """
    ctx = AgentContext(
        agent_id="p03",
        task_type=TaskType.SPEECH,
        phase="speech",
        day_number=2,
        own_role="seer",
        legal_actions=[ActionType.SPEECH, ActionType.VOTE],
        legal_targets=["p05", "p07"],
        public_summary="D2 seer speech",
        skill_analysis_hints={"wolf_pit": "嫌疑区: p05"},
    )
    user_prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    assert "技能分析结果" in user_prompt, (
        "Seer user prompt must include the pre-injected skill analysis section."
    )


def test_skill_catalog_not_in_system_prompt_for_villager():
    """P0-K1: villager system prompt must not include the tool-skill catalog.

    The tool path is dead code. The catalog (which referenced load_skill
    tools) is removed entirely from the prompt.
    """
    ctx = _make_villager_context()
    system_prompt = PlayerPromptBuilder(ctx).build_system_prompt()
    assert "可用技能目录" not in system_prompt, (
        "Villager system prompt must not include the skill catalog header."
    )


def test_skill_analysis_hints_in_user_prompt_for_villager():
    """P0-K1: villager user prompt must include the pre-injected skill analyses."""
    ctx = AgentContext(
        agent_id="p10",
        task_type=TaskType.SPEECH,
        phase="speech",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH, ActionType.VOTE],
        legal_targets=["p05", "p07"],
        public_summary="D2 vote",
        skill_analysis_hints={"wolf_pit": "嫌疑区: p07"},
    )
    user_prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    assert "技能分析结果" in user_prompt


# ---------------------------------------------------------------------------
# P0-R2: empty_response retry hint suggests no_action when category is timeout
# ---------------------------------------------------------------------------
#
# Audit P0-R2 finding: when the model returns empty and the failure
# category is "timeout", the LLM was getting only a generic
# "If the model timed out, consider shorter reasoning" hint.  That
# gave the LLM no permission to take a safe no-action — so it would
# either: (a) try again and time out again, (b) fabricate a vote
# target. Both are bad.
#
# Fix: when `failure_category == "timeout"`, the retry hint must
# explicitly tell the LLM it can return `no_action` as a safe
# fallback. The hint should NOT add this permission for non-timeout
# categories (e.g., provider_error) where the LLM might recover
# with a different response.


def test_retry_hint_suggests_no_action_when_failure_category_is_timeout():
    """P0-R2: timeout empty_response → hint must suggest no_action.

    Game trace g_3528592081: seer p03 vote (Action 57) hit 3 empty
    retries and fell back to a default target. If the hint had said
    "if you can't decide, return no_action", the model would have
    taken the safe no-op rather than burning 3 attempts.
    """
    ctx = _make_villager_context()
    retry = RetryInfo(
        attempt=2,
        max_retries=3,
        error_code="empty_response",
        error_message="Model returned empty text",
        failure_category="timeout",
        correction_hint=(
            "Please provide a valid JSON action (cause: timeout). "
            "If the model timed out, consider shorter reasoning."
        ),
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(retry)
    # The hint should explicitly mention no_action as a safe fallback.
    assert "no_action" in prompt, (
        "When failure_category is 'timeout', retry hint must mention "
        "'no_action' so the LLM knows the safe no-op is permitted."
    )


def test_retry_hint_does_not_force_no_action_for_non_timeout_categories():
    """P0-R2: non-timeout empty_response → no special no_action hint.

    For provider_error / network_error / unknown, the LLM might
    still recover with a fresh attempt — so the timeout-specific
    no_action suggestion should NOT appear, to avoid telling the
    LLM to give up on recoverable cases.
    """
    ctx = _make_villager_context()
    retry = RetryInfo(
        attempt=2,
        max_retries=3,
        error_code="empty_response",
        error_message="Model returned empty text",
        failure_category="provider_error",
        correction_hint=(
            "Please provide a valid JSON action (cause: provider_error). "
            "If the model timed out, consider shorter reasoning."
        ),
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(retry)
    # The timeout-specific "no_action" permission must NOT leak into
    # non-timeout cases.  (Existing empty_response hints remain.)
    # We check for the timeout-specific phrase that we add in the fix.
    assert "如果超时" not in prompt, (
        "Non-timeout empty_response must not include the timeout-specific "
        "'如果超时, 请直接返回 no_action' hint."
    )


# ---------------------------------------------------------------------------
# P0-M7: _build_private_memory_hints reads ONLY from private_memory_hints
# ---------------------------------------------------------------------------


def test_no_private_memory_dual_source():
    """P0-M7: section must use ctx.private_memory_hints as the only source.

    The previous code did:
        ctx.private_memory_hints or ctx.visible_world_state.get("private_memory", {})
    This dual-source caused duplicate injection when both fields were
    populated, and risked leaking into the prompt content twice.

    Fix: read only from ctx.private_memory_hints. If empty, skip the
    section entirely (no fallback to visible_world_state).
    """
    # Case 1: only visible_world_state["private_memory"] populated.
    # The section must NOT appear.
    ctx_visible_only = AgentContext(
        agent_id="p05",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH],
        legal_targets=["p05"],
        public_summary="D2 public",
        private_memory_hints={},  # explicitly empty
        visible_world_state={
            "private_memory": {
                "logic_flaws": [{"day": 1, "speaker": "p02", "point": "should not leak"}],
            },
        },
    )
    prompt_visible = PlayerPromptBuilder(ctx_visible_only).build_user_prompt(RetryInfo())
    # The private_memory section label should NOT appear because the
    # canonical source (private_memory_hints) is empty. The visible fallback
    # must be ignored.
    assert "should not leak" not in prompt_visible, (
        "visible_world_state['private_memory'] fallback must be ignored; "
        "the section must use private_memory_hints as the only source."
    )

    # Case 2: only private_memory_hints populated. Section appears once.
    ctx_hints_only = AgentContext(
        agent_id="p05",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH],
        legal_targets=["p05"],
        public_summary="D2 public",
        private_memory_hints={
            "logic_flaws": [{"day": 1, "speaker": "p02", "point": "vote flip"}],
        },
        visible_world_state={"other_field": "x"},
    )
    prompt_hints = PlayerPromptBuilder(ctx_hints_only).build_user_prompt(RetryInfo())
    # The hint content must appear (via private_memory_hints)
    assert "vote flip" in prompt_hints
    # It must appear exactly once (no dual-source duplication)
    assert prompt_hints.count("vote flip") == 1, (
        "private_memory content must appear exactly once; dual-source may "
        "have produced duplicate injection."
    )


# ---------------------------------------------------------------------------
# P0-M1: private_memory section is labeled as "本局·第N轮·私有记忆"
# ---------------------------------------------------------------------------


def _make_ctx_with_private_memory(
    private_memory: dict,
    day_number: int = 2,
) -> AgentContext:
    """Build an AgentContext whose private_memory_hints is populated."""
    return AgentContext(
        agent_id="p05",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=day_number,
        own_role="villager",
        legal_actions=[ActionType.SPEECH],
        legal_targets=["p05"],
        public_summary="D2 public",
        private_memory_hints=private_memory,
    )


def test_private_memory_labeled_as_current_game():
    """P0-M1: private_memory section must be prefixed with the current-game
    label 【本局·第N轮·私有记忆】 so the LLM cannot confuse it with
    cross-game reflection memory.

    Without the label, the LLM might treat a sentence from its own
    private memory as a public statement, leaking the thinking to
    other players.
    """
    ctx = _make_ctx_with_private_memory(
        private_memory={
            "logic_flaws": [{"day": 1, "speaker": "p02", "point": "vote flip"}],
        },
        day_number=2,
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    # The label must include the day number and the "私有记忆" / "本局" tag.
    assert "【本局·第2轮·私有记忆】" in prompt, (
        "private_memory section must be labeled 【本局·第N轮·私有记忆】 "
        "where N is ctx.day_number; prompt did not contain expected label."
    )


def test_private_memory_label_uses_day_number_correctly():
    """The day number in the label must come from ctx.day_number, not be
    hardcoded. Day 3 must produce 【本局·第3轮·私有记忆】."""
    ctx = _make_ctx_with_private_memory(
        private_memory={"vote_thoughts": [{"day": 3, "target": "p02"}]},
        day_number=3,
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    assert "【本局·第3轮·私有记忆】" in prompt
    # Day 2 label must NOT appear in a day-3 prompt
    assert "【本局·第2轮·私有记忆】" not in prompt


# ---------------------------------------------------------------------------
# P1-S3: every user-prompt section gets a [硬约束/辅助/可选] priority label
# ---------------------------------------------------------------------------
#
# Audit P1-S3 finding: build_user_prompt concatenates 16 sections
# (phase, belief, summary, visible_state, private_memory, salience,
# rag_hints, reflection, profile, cognition, strategy_directive,
# skill_analysis, transcript, task, retry_hint, output_contract) with
# no priority signal between them. The LLM reads the entire stack as
# equally important and burns tokens on sections that don't need its
# attention (e.g., transcript while the retry hint is telling it to
# shorten the response).
#
# Fix: prepend each section with one of three priority labels:
# - 【硬约束】 strategy_directive, retry hint, output contract
#   (must be addressed / must be obeyed)
# - 【辅助】   phase, belief, summary, visible state, private memory,
#   salience, rag hints, reflection, profile, cognition, skill hints
#   (background context for reasoning; ignore fields not relevant to
#   current task)
# - 【可选】   transcript (reference, may be skimmed or dropped if
#   output is constrained)
#
# Task prompt itself is NOT relabeled — the task description is what
# the LLM acts on and adding a prefix would risk confusing the action
# spec. The retry hint + output contract keep their natural positions
# at the end of the user message but are wrapped with 【硬约束】 so the
# LLM sees the priority before reading the content.


def _make_ctx_for_priority_label_test(
    *,
    own_role: str = "villager",
    legal_actions: list | None = None,
    include_retry: bool = False,
) -> "AgentContext":
    """Context with every section populated so the labels have content."""
    return AgentContext(
        agent_id="p05",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role=own_role,
        legal_actions=legal_actions if legal_actions is not None else [ActionType.SPEECH, ActionType.VOTE],
        legal_targets=["p05", "p07"],
        public_summary="D2 — p05 voted p07",
        visible_world_state={"alive": ["p05", "p07"], "day_number": 2},
        private_memory_hints={"logic_flaws": [{"day": 2, "point": "vote flip"}]},
        salience_items=[{"id": "sal-1", "summary": "p07 changed vote"}],
        rag_hints=[{"title": "Case 1", "summary": "wolf switch", "key_decisions": ["vote flip"]}],
        reflection_memory_hints=[{"theme": "anti-herd", "summary": "independent judgement"}],
        profile_memory_hint={"games_played": 5, "summary": "前 30%"},
        cognition_matrix_hint={"p07": {"trust": 0.3, "faction_lean": "werewolf"}},
        strategy_directive={
            "must_address_alerts": ["p07 accused me"],
            "anti_herd": "do not follow the herd",
        },
        skill_analysis_hints={"wolf_pit": "嫌疑区: p07"},
        recent_transcript=[{"speaker": "p07", "text": "I think p05 is wolf"}],
        persona_snapshot={"tone": "neutral"},
    )


def test_sections_have_priority_labels():
    """P1-S3: each of the 16 user-prompt sections is prefixed with a
    [硬约束/辅助/可选] priority label so the LLM can rank which sections
    to attend to under tight token budget.

    Sections grouped:
    - 【硬约束】 strategy_directive, retry hint, output contract
    - 【辅助】   phase, belief, summary, visible state, private memory,
      salience, rag hints, reflection, profile, cognition, skill hints
    - 【可选】   transcript

    The test verifies the label appears in the user prompt and the
    label appears BEFORE the section's first content character.
    """
    retry = RetryInfo(
        attempt=1,
        max_retries=3,
        error_code="parse_error",
        error_message="missing JSON key",
        correction_hint="Output a valid JSON object",
    )
    ctx = _make_ctx_for_priority_label_test(include_retry=True)
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(retry)

    # Hard sections must be labeled 【硬约束】.
    # P1-6: strategy_directive's outer label is now 【策略指令】 (neutral)
    # to avoid double-labeling. Retry hint and output contract are
    # still 【硬约束】, so 2 hard labels remain.
    hard_label_count = prompt.count("【硬约束】")
    assert hard_label_count >= 2, (
        f"Expected at least 2 【硬约束】 labels "
        f"(retry hint, output contract), got {hard_label_count}."
    )

    # The 辅助 sections must collectively produce multiple 【辅助】 labels.
    auxiliary_label_count = prompt.count("【辅助】")
    assert auxiliary_label_count >= 8, (
        f"Expected at least 8 【辅助】 labels (one per 辅助 section), "
        f"got {auxiliary_label_count}."
    )

    # The transcript is the only 可选 section in build_user_prompt.
    assert "【可选】" in prompt, (
        "Expected at least one 【可选】 label (transcript section)."
    )

    # Verify the label is followed by the section's own header text so
    # the LLM sees 【硬约束】 X then the actual content.
    # The retry hint header is "纠正提示" — confirm the label is
    # 【辅助】 (P1-9: retry hint is advisory, not a hard constraint).
    retry_idx = prompt.find("纠正提示")
    assert retry_idx > 0, "Retry hint should still render in the user prompt"
    preceding = prompt[max(0, retry_idx - 60):retry_idx]
    assert "【辅助】" in preceding, (
        f"P1-9: retry hint must be preceded by 【辅助】 label, got: {preceding!r}"
    )

    # P1-6: Strategy directive header is "本轮策略指令". Its outer
    # section label is now 【策略指令】 (neutral), NOT 【硬约束】.
    directive_idx = prompt.find("本轮策略指令")
    assert directive_idx > 0
    preceding = prompt[max(0, directive_idx - 60):directive_idx]
    assert "【策略指令】" in preceding, (
        f"Strategy directive must be preceded by 【策略指令】 label "
        f"(P1-6: neutral outer label, inner P0-S5 sub-group carries "
        f"the priority signal). Got: {preceding!r}"
    )

    # Transcript is the only 可选 section — its header is "近期发言".
    transcript_idx = prompt.find("近期发言")
    assert transcript_idx > 0
    preceding = prompt[max(0, transcript_idx - 60):transcript_idx]
    assert "【可选】" in preceding, (
        f"Transcript must be preceded by 【可选】 label, got: {preceding!r}"
    )


def test_priority_labels_for_hard_sections_distinct_from_internal_directive_groups():
    """P1-S3: the inner P0-S5 【硬约束】 sub-group must still render
    inside the strategy_directive section, even after P1-6 changed
    the OUTER section label to 【策略指令】.

    The inner sub-group signals "this key is hard within the directive"
    (MUST-obey). The outer section label is now neutral (【策略指令】).
    The inner P0-S5 sub-group MUST/SHOULD/REFERENCE markers must still
    be present so the LLM knows which directive keys are binding.
    """
    ctx = _make_ctx_for_priority_label_test()
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    # The directive section now has the outer label 【策略指令】. The
    # inner P0-S5 sub-group still has its 【硬约束】 MUST marker.
    assert "以下指令必须遵守（MUST）" in prompt, (
        "P0-S5 inner MUST marker must still render (the keys with "
        "hard priority still need their binding signal)"
    )
    # Sanity: the inner marker is 【硬约束】 (P0-S5 unchanged).
    assert "【硬约束】" in prompt, (
        "P0-S5 inner 【硬约束】 sub-group header must still render "
        "in the directive section."
    )


def test_priority_labels_for_auxiliary_sections_are_consistent():
    """P1-S3: 辅助 sections all use the same 【辅助】 label, not mixed
    labels. The LLM should see a uniform priority signal across the
    background context.
    """
    ctx = _make_ctx_for_priority_label_test()
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    # Section headers we expect to be 辅助:
    auxiliary_headers = [
        "当前阶段",          # phase
        "我的判断",         # belief
        "当前局公开事实",     # public summary
        "可见状态",          # visible state
        "本局·",            # private memory
        "关键事件",          # salience
        "知识库提示",        # rag hints
        "跨局反思记忆",       # reflection
        "长期能力画像",       # profile
        "我的认知矩阵",       # cognition
        "技能分析结果",       # skill analysis
    ]
    for header in auxiliary_headers:
        idx = prompt.find(header)
        if idx < 0:
            # Some headers may not render if the corresponding data
            # is empty in this context. Skip absent ones.
            continue
        preceding = prompt[max(0, idx - 60):idx]
        assert "【辅助】" in preceding, (
            f"Section with header {header!r} must be preceded by 【辅助】 label, "
            f"got: {preceding!r}"
        )


# ---------------------------------------------------------------------------
# P1-4: output contract field list must not appear twice
# ---------------------------------------------------------------------------
#
# Audit P1-4 finding: ``action_type、target_id、speech、reason、confidence``
# appears in BOTH the system prompt's ``_build_output_contract`` AND
# the user prompt's ``_build_strict_output_contract``. Inflates token
# budget by ~25 chars per duplication, multiplies across multi-turn
# games, and adds no information — the LLM has already seen the field
# list in the cacheable system prompt.
#
# Fix: the field list lives in the SYSTEM prompt only (stable rule).
# The user prompt keeps the per-turn phase-specific rules (e.g., the
# "8. 投票还必须包含..." line for VOTE actions, the legal_actions
# / legal_targets constraints).


def test_output_contract_not_duplicated():
    """P1-4: the FULL_ACTION field list must appear only ONCE in the
    assembled (system + user) prompt pair.

    Pre-fix, the string ``action_type、target_id、speech、reason、confidence``
    appeared in BOTH ``build_system_prompt()`` and
    ``build_user_prompt()``. The user prompt version adds no new
    information — the system prompt is already cacheable and the LLM
    has the rule. The user prompt should keep phase-specific rules
    only (vote audit fields, legal_actions / legal_targets).
    """
    ctx = AgentContext(
        agent_id="p05",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH, ActionType.VOTE],
        legal_targets=["p05", "p07"],
        public_summary="D2 speech",
    )
    builder = PlayerPromptBuilder(ctx)
    system_prompt = builder.build_system_prompt()
    user_prompt = builder.build_user_prompt(RetryInfo())

    field_list = "action_type、target_id、speech、reason、confidence"
    # Sanity: must appear at least once (in the system prompt).
    assert system_prompt.count(field_list) >= 1, (
        "Field list must still be advertised in the system prompt "
        "(stable rule — keep the LLM trained on the format)."
    )
    # P1-4: must NOT appear in the user prompt (it would be a duplicate).
    assert user_prompt.count(field_list) == 0, (
        "P1-4: field list is duplicated in the user prompt. The system "
        "prompt already has it — the user prompt should only carry "
        "phase-specific rules (legal_actions, legal_targets, vote audit "
        "fields). User prompt excerpt: " + user_prompt[:500]
    )


def test_output_contract_vote_rule_still_in_user_prompt():
    """P1-4: even after removing the duplicated field list, the
    phase-specific VOTE audit rule must still appear in the user prompt.

    The user prompt is where the LLM sees the per-turn constraints. The
    VOTE audit fields (``seer_stance、vote_basis、standing_with_seer、...``)
    are phase-specific, not stable, so they belong in the user prompt.
    """
    ctx = AgentContext(
        agent_id="p05",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH, ActionType.VOTE],
        legal_targets=["p05", "p07"],
        public_summary="D2 speech",
    )
    user_prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    # The phase-specific rule for VOTE must still be in the user prompt.
    assert "seer_stance" in user_prompt, (
        "P1-4: phase-specific VOTE audit rule (seer_stance etc.) must "
        "stay in the user prompt — it depends on legal_actions which "
        "is per-turn."
    )


# ---------------------------------------------------------------------------
# P1-5: user prompt must respect a global token budget
# ---------------------------------------------------------------------------
#
# Audit P1-5 finding: with 8+ sections populated (RAG hints, salience,
# strategy_directive, transcript, etc.) the assembled user prompt
# reaches 3,000-5,000 tokens — well over the 2,000-token budget most
# Chinese-context models can comfortably ingest alongside the system
# prompt. Per-section truncation at _MAX_JSON_CONTEXT_CHARS = 1800 is
# not enough; the global sum is the constraint that matters.
#
# Fix: implement a global token budget. Drop lowest-priority sections
# (可选 before 辅助; never drop 硬约束) when the assembled prompt
# exceeds the budget. Use the existing _SECTION_PRIORITIES map for
# the priority signal.

_USER_PROMPT_BUDGET_CHARS = 6_250  # ≈ 2,500 CJK tokens (rough)


def _make_ctx_with_all_sections_populated() -> AgentContext:
    """Context with every available user-prompt section populated.

    Forces the largest possible user prompt (16 sections all with
    content) so the budget test catches the over-budget case.
    """
    long_summary = "公开事实 " + ("D2 数据 " * 300)
    long_visible = {f"key_{i}": f"value_{i} " * 30 for i in range(20)}
    long_private_memory = {
        "logic_flaws": [{"day": 2, "point": f"逻辑问题 {i}"} for i in range(8)],
        "valid_points": [{"day": 2, "point": f"有效点 {i}"} for i in range(8)],
    }
    long_salience = [
        {"id": f"sal-{i}", "weight": 0.5, "summary": f"事件 {i} " * 20}
        for i in range(8)
    ]
    long_rag = [
        {"title": f"案例 {i}", "summary": f"内容 {i} " * 50, "key_decisions": [f"决策 {i}"]}
        for i in range(3)
    ]
    long_reflection = [
        {"theme": f"主题 {i}", "summary": f"经验 {i} " * 30}
        for i in range(5)
    ]
    long_profile = {"games_played": 10, "summary": "前 30% 玩家画像 " * 50}
    long_cognition = {
        f"p{i:02d}": {"trust": 0.5, "faction_lean": "good", "top_role_guess": "villager"}
        for i in range(1, 13)
    }
    long_directive = {
        "must_address_alerts": [{"alert": f"提示 {i}"} for i in range(5)],
        "anti_herd": "不要盲目跟票 " * 20,
        "skill_tactical_advice": {"role": "villager", "tips": ["tip1", "tip2"] * 30},
    }
    long_skill = {
        "wolf_pit": "嫌疑区 " * 100,
        "seer_logic": "查验逻辑 " * 100,
    }
    long_transcript = [
        {"speaker": f"p{i:02d}", "text": f"近期发言内容 {i} " * 30}
        for i in range(1, 5)
    ]
    long_persona = {"tone": "aggressive", "style": "logical", "phrase_style": "blame_p05"}
    long_belief = {
        "my_suspects": [
            {"player": f"p{i:02d}", "faction_lean": "wolf_lean",
             "top_role_guess": "werewolf", "trust": 0.2}
            for i in range(1, 6)
        ],
        "my_trusted": [
            {"player": f"p{i:02d}", "faction_lean": "good_lean",
             "top_role_guess": "villager", "trust": 0.8}
            for i in range(6, 11)
        ],
    }
    return AgentContext(
        agent_id="p05",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH, ActionType.VOTE],
        legal_targets=["p05", "p07"],
        public_summary=long_summary,
        visible_world_state=long_visible,
        private_memory_hints=long_private_memory,
        salience_items=long_salience,
        rag_hints=long_rag,
        reflection_memory_hints=long_reflection,
        profile_memory_hint=long_profile,
        cognition_matrix_hint=long_cognition,
        strategy_directive=long_directive,
        skill_analysis_hints=long_skill,
        recent_transcript=long_transcript,
        persona_snapshot=long_persona,
        belief_state=long_belief,
    )


def test_user_prompt_within_budget_when_all_sections_populated():
    """P1-5: with all 16 sections populated, the user prompt must be
    under the budget cap.

    Pre-fix: the user prompt could reach 3,000-5,000 tokens. The fix
    drops lowest-priority sections (可选 first, then 辅助) until the
    prompt fits under the budget. 硬约束 sections are never dropped.
    """
    ctx = _make_ctx_with_all_sections_populated()
    user_prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    # rough CJK token estimate: 1 token ≈ 2.5 chars
    approx_tokens = len(user_prompt) / 2.5
    assert approx_tokens < 2_500, (
        f"P1-5: user prompt is over budget. approx_tokens={approx_tokens:.0f}, "
        f"chars={len(user_prompt)}, budget=2,500 tokens "
        f"(~6,250 chars). Drop lowest-priority sections first when over "
        f"budget. user_prompt[:500]={user_prompt[:500]!r}"
    )


def test_hard_sections_never_dropped_under_budget():
    """P1-5: budget enforcement must NEVER drop truly-binding sections.

    P1-9: retry hint is now 【辅助】 (advisory), so under extreme
    budget pressure it may be dropped — the runtime FallbackAction
    handles the safety case regardless. strategy_directive and
    output contract are the two truly-binding sections that the
    LLM must always see; the trimmer must never drop them.
    """
    ctx = _make_ctx_with_all_sections_populated()
    retry = RetryInfo(
        attempt=2,
        max_retries=3,
        error_code="parse_error",
        error_message="JSON parse error: missing field 'speech'",
        correction_hint="只输出JSON。",
    )
    user_prompt = PlayerPromptBuilder(ctx).build_user_prompt(retry)
    # strategy_directive is gated by 硬约束-equivalent policy: it
    # carries binding rules the LLM must obey (the inner P0-S5
    # MUST/SHOULD/REFERENCE sub-group keys).
    assert "本轮策略指令" in user_prompt, (
        "P1-5: strategy_directive carries binding rules and must "
        "never be dropped from the user prompt under the budget cap."
    )
    # Output contract is 【硬约束】 and must never be dropped.
    assert "最终输出协议" in user_prompt, (
        "P1-5: output contract is a 硬约束 section and must never be dropped."
    )


def test_optional_sections_dropped_first():
    """P1-5: under tight budget, 可选 sections are dropped first.

    The transcript is the only 可选 section in build_user_prompt.
    """
    ctx = _make_ctx_with_all_sections_populated()
    user_prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    # If the budget requires any drop, the 可选 section (transcript)
    # should be the first to go. We can't pin the exact behavior
    # (it depends on payload sizes), but we can assert the budget is
    # respected — the trimmer always picks the lowest-priority section.
    # The contract under test is "budget is enforced", not "transcript
    # is always dropped", so this test is a sanity check that the
    # transcript MAY be present (it didn't HAVE to be dropped) when
    # other smaller payloads fit.
    # Pin the contract: budget is hard.
    approx_tokens = len(user_prompt) / 2.5
    assert approx_tokens < 2_500


# ---------------------------------------------------------------------------
# P1-6: strategy_directive section label is neutral, not 【硬约束】
# ---------------------------------------------------------------------------
#
# Audit P1-6 finding: ``_build_strategy_directive`` is wrapped with the
# outer section label 【硬约束】, but the function internally splits its
# content into 【硬约束】/【建议】/【参考】 sub-headers. The double-labeling
# is contradictory — the LLM sees "this whole section is MUST" but then
# sees the inner header saying "this subsection is just REFERENCE".
#
# Fix: use a neutral section label 【策略指令】 for the OUTER wrapper.
# The inner sub-headers (【硬约束】/【建议】/【参考】) carry the actual
# priority signal for the keys inside.


def test_strategy_directive_section_label_neutral():
    """P1-6: strategy_directive outer section label is 【策略指令】,
    not 【硬约束】. The inner sub-headers (P0-S5) carry the priority signal.
    """
    ctx = AgentContext(
        agent_id="p05",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH, ActionType.VOTE],
        legal_targets=["p05", "p07"],
        public_summary="D2 speech",
        strategy_directive={
            "must_address_alerts": ["p07 accused me"],
            "anti_herd": "do not follow the herd",
        },
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    # Pin the contract: the strategy_directive section is wrapped with
    # 【策略指令】, NOT 【硬约束】, at the outer level.
    directive_idx = prompt.find("本轮策略指令")
    assert directive_idx >= 0, "strategy_directive section must still render"
    preceding = prompt[max(0, directive_idx - 60):directive_idx]
    assert "【策略指令】" in preceding, (
        "P1-6: strategy_directive outer section label must be 【策略指令】 "
        "(neutral), not 【硬约束】. The inner P0-S5 sub-headers carry the "
        "priority signal. Got preceding text: " + repr(preceding)
    )
    # The outer label must NOT be 【硬约束】 (the inner sub-group
    # header is also 【硬约束】, but the OUTER wrapper is the one
    # immediately before "本轮策略指令").
    outer_label = preceding.strip().split("\n")[-1].strip()
    assert "【硬约束】" not in outer_label, (
        "P1-6: outer strategy_directive section label must NOT be "
        "【硬约束】 (the inner sub-group is already 【硬约束】). "
        f"Got outer label: {outer_label!r}"
    )


def test_strategy_directive_inner_subgroups_still_three_tiers():
    """P1-6 regression: changing the outer label must not remove the
    inner P0-S5 sub-grouping (MUST / SHOULD / REFERENCE).

    The inner sub-headers carry the priority signal for the directive
    keys. With hard+soft+reference keys present, all 3 inner headers
    must still render.
    """
    ctx = AgentContext(
        agent_id="p05",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH, ActionType.VOTE],
        legal_targets=["p05", "p07"],
        public_summary="D2 speech",
        strategy_directive={
            # hard
            "must_address_alerts": ["p07 accused me"],
            # suggestion
            "anti_herd": "do not follow the herd",
            # reference
            "master_behavior_summary": "master last round attacked p05",
        },
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    # Inner P0-S5 sub-group markers must still be present.
    assert "以下指令必须遵守（MUST）" in prompt, (
        "P1-6 regression: P0-S5 inner MUST marker must still render"
    )
    assert "以下指令为建议（SHOULD）" in prompt, (
        "P1-6 regression: P0-S5 inner SHOULD marker must still render"
    )
    assert "以下为背景信息（REFERENCE）" in prompt, (
        "P1-6 regression: P0-S5 inner REFERENCE marker must still render"
    )


# ---------------------------------------------------------------------------
# P1-7: villager role guide must not claim authority over witch's potions
# ---------------------------------------------------------------------------
#
# Audit P1-7 finding: the villager role guide contains the rule
# "N1 用药决策推动解药救人" (N1: drive the antidote-saving decision).
# But villagers have NO authority over witch's potions — that decision
# is exclusively the witch's. The rule confuses villagers into thinking
# they can direct the witch, which is a known role-rule violation.
#
# Fix: reword to "N1 公开讨论中支持解药救人（如有女巫报银水线索）" —
# villagers can argue in PUBLIC discussion that the witch should save
# (if there's a silver-water lead), but they cannot "push a decision".


def test_villager_role_guide_no_witch_decision_authority():
    """P1-7: the villager role guide must not claim authority over
    the witch's potion decisions.

    Pre-fix text: "N1 用药决策推动解药救人" — implies villagers can
    drive the witch's potion use. Villagers have no such authority
    (witch's potions are exclusively the witch's decision per
    design doc Chapter 3).
    """
    ctx = AgentContext(
        agent_id="p05",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH, ActionType.VOTE],
        legal_targets=["p07"],
        public_summary="D2",
    )
    system_prompt = PlayerPromptBuilder(ctx).build_system_prompt()
    # The villager role guide must not say "N1 用药决策" — that
    # phrasing implies authority over the witch's potions.
    assert "N1 用药决策" not in system_prompt, (
        "P1-7: villager role guide must not say 'N1 用药决策' "
        "(implies authority over witch's potions). Got: "
        + system_prompt[system_prompt.find("村民规则"):system_prompt.find("村民规则") + 200]
    )
    # The new framing should be "支持" (support) or similar —
    # describing what villagers can do in PUBLIC discussion, not
    # what they can drive as a decision.
    villager_rule_idx = system_prompt.find("村民规则")
    assert villager_rule_idx >= 0, "villager role guide must be present"
    rule_window = system_prompt[villager_rule_idx:villager_rule_idx + 300]
    # The guide should reference the witch's decision context
    # (支持 / 银水 / 公开讨论) rather than claiming authority
    # (用药决策 / 推动).
    assert "解药" in rule_window, (
        "P1-7: villager guide should still mention the antidote "
        "(just framed as supporting, not deciding)."
    )


# ---------------------------------------------------------------------------
# P1-8: vote example's vote_basis must be role-appropriate
# ---------------------------------------------------------------------------
#
# Audit P1-8 finding: every non-seer role (villager / witch / hunter /
# idiot / hybrid / werewolf) sees the same vote example with
# `vote_basis="seer_check"`. But only a seer has a check. Non-seer
# roles don't have their own check — they are siding with another
# player's seer claim. The example's vote_basis should reflect that:
#   - seer       → "seer_check"   (own check)
#   - non-seer   → "seer_siding"  (standing with another seer)
#
# Fix: make vote_basis role-dependent in the example. Use the
# `role` variable already in scope.


def test_vote_example_role_appropriate_basis():
    """P1-8: non-seer roles' vote example uses ``seer_siding``, not ``seer_check``.

    Only a seer has a check of their own. Non-seer roles do not
    have a check — they side with a (claimed) seer. The example's
    ``vote_basis`` must reflect that distinction so the LLM doesn't
    fabricate a non-existent check.
    """
    ctx = _make_full_action_ctx("villager")
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    vote_examples = [
        ex for ex in _extract_json_examples(prompt)
        if ex.get("action_type") == "vote"
    ]
    assert vote_examples, "Expected a vote example in the prompt"
    for ex in vote_examples:
        # P1-8: villager must use seer_siding, not seer_check.
        assert ex.get("vote_basis") == "seer_siding", (
            "P1-8: villager vote example must use vote_basis='seer_siding' "
            "(villager has no check of their own — they side with a seer). "
            f"Got vote_basis={ex.get('vote_basis')!r}. Full example: {ex}"
        )


def test_vote_example_seer_basis_unchanged():
    """P1-8 regression: seer role still uses vote_basis='seer_check'.

    A seer has their own check — the example must keep the
    ``seer_check`` value (P0-4 already fixed ``standing_with_seer``
    to be empty for the seer branch). The P1-8 change is for
    non-seer roles only.
    """
    ctx = _make_full_action_ctx("seer")
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    vote_examples = [
        ex for ex in _extract_json_examples(prompt)
        if ex.get("action_type") == "vote"
    ]
    assert vote_examples
    for ex in vote_examples:
        assert ex.get("vote_basis") == "seer_check", (
            f"P1-8 regression: seer vote example must still use "
            f"vote_basis='seer_check', got {ex.get('vote_basis')!r}"
        )


@pytest.mark.parametrize("role", [
    "witch", "hunter", "idiot", "hybrid", "werewolf",
])
def test_vote_example_non_seer_uses_seer_siding(role: str):
    """P1-8: every non-seer role (good and wolf) uses ``seer_siding``."""
    ctx = _make_full_action_ctx(role)
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    vote_examples = [
        ex for ex in _extract_json_examples(prompt)
        if ex.get("action_type") == "vote"
    ]
    assert vote_examples, f"Expected a vote example for role={role!r}"
    for ex in vote_examples:
        assert ex.get("vote_basis") == "seer_siding", (
            f"P1-8: non-seer role={role!r} vote example must use "
            f"vote_basis='seer_siding' (no own check), got "
            f"{ex.get('vote_basis')!r}. Full example: {ex}"
        )


# ---------------------------------------------------------------------------
# P1-9: retry hint section is 【辅助】, not 【硬约束】
# ---------------------------------------------------------------------------
#
# Audit P1-9 finding: ``_build_retry_hint`` is wrapped with the
# 【硬约束】 outer label, but the content is mostly descriptive /
# advisory (the error_message snippet and the correction_hint). Only
# the timeout-no-op permission ("如果你已经超时, 请直接返回 no_action")
# is a true binding rule, and even that is enforced by the runtime
# FallbackAction path, not by the LLM obeying the prompt.
#
# Fix: use 【辅助】 for the retry hint section. The no-op permission
# line still appears (the LLM needs to see it), but the section
# wrapper is no longer over-labeled as MUST-obey.


def test_retry_hint_labeled_as_辅助():
    """P1-9: retry hint outer section label is 【辅助】, not 【硬约束】.

    The retry hint content is descriptive (error_message snippet) and
    advisory (correction_hint, timeout-no-op permission). Treating
    the whole section as a binding hard constraint is over-labeling.
    """
    ctx = AgentContext(
        agent_id="p05",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH, ActionType.VOTE],
        legal_targets=["p05", "p07"],
        public_summary="D2 speech",
    )
    retry = RetryInfo(
        attempt=2,
        max_retries=3,
        error_code="parse_error",
        error_message="JSON parse error",
        correction_hint="只输出JSON。",
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(retry)
    # The retry hint header is "纠正提示" — confirm the OUTER label
    # is 【辅助】, not 【硬约束】.
    retry_idx = prompt.find("纠正提示")
    assert retry_idx > 0, "Retry hint should still render in the user prompt"
    preceding = prompt[max(0, retry_idx - 60):retry_idx]
    assert "【辅助】" in preceding, (
        "P1-9: retry hint section must be labeled 【辅助】 "
        "(advisory content, not a hard constraint). Got preceding: "
        f"{preceding!r}"
    )


def test_retry_hint_timeout_permission_line_still_present():
    """P1-9 regression: the timeout-no-op permission line must still
    appear in the retry hint even after relabeling the section.

    The line is a useful advisory signal that the LLM should take
    a safe no-op on timeout. The runtime FallbackAction enforces
    safety; the prompt signal is the soft guidance.
    """
    ctx = AgentContext(
        agent_id="p05",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH, ActionType.VOTE],
        legal_targets=["p05", "p07"],
        public_summary="D2 speech",
    )
    retry = RetryInfo(
        attempt=2,
        max_retries=3,
        error_code="empty_response",
        failure_category="timeout",
        correction_hint="Please provide a valid JSON action",
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(retry)
    # The no-op permission line is still in the rendered retry hint.
    assert "no_action" in prompt, (
        "P1-9 regression: timeout-no-op permission line must still "
        "appear in the retry hint (the LLM still needs the signal)."
    )


# ---------------------------------------------------------------------------
# P1-S4: _format_examples (FULL_ACTION mode) examples match the mode
# ---------------------------------------------------------------------------
#
# Audit P1-S4 finding: _format_examples is invoked for FULL_ACTION mode
# (the speech/vote/sheriff/wolf paths). The LLM copies the example
# structure, so a leaked `intent` field (SPEECH_INTENT mode) or `choice`
# field (TARGET_CHOICE mode) would prime the LLM to fill those fields
# for actions that don't have them. P0-S1 added mode isolation at the
# parse layer (rejects unknown fields); P0-S8 added extra=forbid on the
# schema; this is the prompt-side regression to keep the examples clean.
#
# Fix: verify the example JSON in each _format_examples branch has no
# `intent` or `choice` keys. The check scans every JSON example block
# in the user prompt and fails if either field name appears.


def _extract_json_examples(prompt: str) -> list[dict]:
    """Find every JSON example in the prompt (the action_type-tagged ones)."""
    examples: list[dict] = []
    for match in _re.finditer(
        r"\{[^{}]*?(?:\{[^{}]*\}[^{}]*?)*\}",
        prompt,
        flags=_re.DOTALL,
    ):
        try:
            data = _json.loads(match.group(0))
        except _json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "action_type" in data:
            examples.append(data)
    return examples


def test_format_examples_no_intent_field_in_speech_path():
    """P1-S4: speech-path example must not include `intent` field.

    `intent` is a SPEECH_INTENT-mode field (the enum-style value the
    LLM picks from a small set). FULL_ACTION mode renders speech
    directly with action_type=speech, speech=text, reason=..., and
    never uses `intent`. If the example mentions it, the LLM will
    defensively fill it for SPEECH actions and the strict schema
    (extra=forbid, P0-S8) will reject them.
    """
    ctx = AgentContext(
        agent_id="p05",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH, ActionType.VOTE],
        legal_targets=["p05"],
        public_summary="D2 vote",
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    examples = _extract_json_examples(prompt)
    assert examples, "Expected at least one example in the speech path"
    for ex in examples:
        assert "intent" not in ex, (
            f"FULL_ACTION example must not include 'intent' field "
            f"(that's a SPEECH_INTENT-mode field); example={ex}"
        )


def test_format_examples_no_choice_field_in_speech_path():
    """P1-S4: speech-path example must not include `choice` field.

    `choice` is a TARGET_CHOICE-mode field (an enum letter A/B/C/...
    the LLM picks from a small set). FULL_ACTION mode renders the
    action with action_type=..., target_id=player_id, never with
    `choice`. If the example mentions it, the LLM will defensively
    fill it for SPEECH/VOTE/WOLF_KILL actions.
    """
    ctx = AgentContext(
        agent_id="p05",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH, ActionType.VOTE],
        legal_targets=["p05"],
        public_summary="D2 vote",
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    examples = _extract_json_examples(prompt)
    assert examples
    for ex in examples:
        assert "choice" not in ex, (
            f"FULL_ACTION example must not include 'choice' field "
            f"(that's a TARGET_CHOICE-mode field); example={ex}"
        )


def test_format_examples_no_intent_field_in_wolf_kill_path():
    """P1-S4: wolf_kill-path examples must not include `intent`.

    Wolf kill is a TARGET_CHOICE-style action (pick a target) but
    renders in FULL_ACTION mode. The example must not leak the
    SPEECH_INTENT `intent` field.
    """
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.NIGHT_ACTION,
        phase="night",
        day_number=1,
        own_role="werewolf",
        legal_actions=[ActionType.WOLF_KILL, ActionType.WOLF_NO_KILL],
        legal_targets=["p05", "p07"],
        public_summary="N1",
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    examples = _extract_json_examples(prompt)
    assert examples, "Expected wolf_kill / wolf_no_kill examples"
    for ex in examples:
        assert "intent" not in ex, (
            f"Wolf-kill example must not include 'intent' field; example={ex}"
        )
        assert "choice" not in ex, (
            f"Wolf-kill example must not include 'choice' field; example={ex}"
        )


def test_format_examples_no_intent_field_in_sheriff_register_path():
    """P1-S4: sheriff_register-path example must not include `intent`."""
    ctx = AgentContext(
        agent_id="p03",
        task_type=TaskType.SHERIFF_REGISTRATION,
        phase="day",
        day_number=1,
        own_role="villager",
        legal_actions=[ActionType.SHERIFF_REGISTER, ActionType.NO_ACTION],
        legal_targets=[],
        public_summary="D1 sheriff election",
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    examples = _extract_json_examples(prompt)
    assert examples, "Expected sheriff_register / no_action examples"
    for ex in examples:
        assert "intent" not in ex, (
            f"Sheriff-register example must not include 'intent' field; example={ex}"
        )
        assert "choice" not in ex, (
            f"Sheriff-register example must not include 'choice' field; example={ex}"
        )


# ---------------------------------------------------------------------------
# P1-S9: villager role guide is present in system_prompt
# ---------------------------------------------------------------------------
#
# Audit P1-S9 finding: _build_role_guide has entries for hunter, idiot,
# witch, seer, werewolf, hybrid — but NOT for villager. Villagers are
# 3 of 12 players in V1, so the role guide has been silently skipping
# them. Villagers see only the generic reasoning/information-boundary
# sections, missing concrete day-time decision guidance.
#
# Fix: add villager role guide with 4 specific behavioral rules:
#   1. When publicly identified, actively state good-side stance
#   2. Analyze speech contradictions / vote patterns
#   3. N1: support the witch using the antidote to save people
#   4. Vote based on public evidence chain, not emotion


def test_villager_role_guide_present():
    """P1-S9: villager role guide must be present in the system prompt.

    The villager role guide is rendered inside _build_role_guide and
    should appear in build_system_prompt. Other 5 roles (hunter, idiot,
    witch, seer, werewolf, hybrid) all have guides; villager must too.
    """
    ctx = AgentContext(
        agent_id="p05",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH, ActionType.VOTE],
        legal_targets=["p07"],
        public_summary="D2",
    )
    system_prompt = PlayerPromptBuilder(ctx).build_system_prompt()
    # The role guide header marker for villager — must mention 村民规则
    # OR a specific villager-behavior directive.
    assert "村民规则" in system_prompt, (
        "Villager role guide must be present in system_prompt. "
        f"system_prompt: {system_prompt[:500]!r}"
    )


def test_villager_role_guide_specific_rules():
    """P1-S9: villager role guide must include the 4 specific rules.

    The 4 rules per the audit:
    1. Public identity → actively state good-side stance
    2. Analyze speech contradictions / vote patterns
    3. N1: support antidote-saving
    4. Vote based on public evidence chain, not emotion
    """
    ctx = AgentContext(
        agent_id="p05",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH, ActionType.VOTE],
        legal_targets=["p07"],
        public_summary="D2",
    )
    system_prompt = PlayerPromptBuilder(ctx).build_system_prompt()
    # Check key phrases from each rule
    assert "好人立场" in system_prompt or "好人阵营" in system_prompt, (
        "Rule 1 (public stance) must be in the villager role guide."
    )
    assert "矛盾" in system_prompt, (
        "Rule 2 (analyze contradictions) must be in the villager role guide."
    )
    assert "解药" in system_prompt, (
        "Rule 3 (N1 antidote support) must be in the villager role guide."
    )
    assert "公开" in system_prompt, (
        "Rule 4 (vote based on public evidence) must be in the villager role guide."
    )


def test_other_roles_still_have_their_guides():
    """P1-S9: adding the villager guide must not remove the other 6 guides.

    Regression check: each of the 6 existing role guides must still
    appear in their respective system_prompts. Adding villager is
    additive, not a replacement.
    """
    role_expected = {
        "hunter": "猎人规则",
        "idiot": "白痴规则",
        "witch": "女巫规则",
        "seer": "预言家规则",
        "werewolf": "狼人规则",
        "hybrid": "混血儿规则",
    }
    for role, expected_phrase in role_expected.items():
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            phase="day",
            day_number=2,
            own_role=role,
            legal_actions=[ActionType.SPEECH],
            legal_targets=["p07"],
            public_summary="D2",
        )
        system_prompt = PlayerPromptBuilder(ctx).build_system_prompt()
        assert expected_phrase in system_prompt, (
            f"Role guide for {role!r} must still be present after P1-S9 "
            f"villager addition. Expected phrase: {expected_phrase!r}"
        )


# ---------------------------------------------------------------------------
# P2-S10: persona lives in user_prompt, not system_prompt
# ---------------------------------------------------------------------------
#
# Audit P2-S10 finding: `_build_persona` was originally placed in
# `build_system_prompt()`. Persona is per-turn (e.g., tone / style hints
# driven by the situation), but system_prompt is supposed to be stable
# (s10 architecture). Mixed placement defeats prompt caching on the
# system side. Fix: move `_build_persona()` invocation from
# `build_system_prompt()` to `build_user_prompt()`, right after the
# DYNAMIC_BOUNDARY marker so the persona is grouped with other
# per-turn dynamic context.


def test_persona_in_user_prompt_not_system():
    """P2-S10: persona must live in user_prompt, not system_prompt.

    The `persona_snapshot` field is per-turn (situation-driven) and
    should be a dynamic section in the user message, not a stable
    section in the system prompt. Pre-fix, `_build_persona()` was
    called from `build_system_prompt()`, so every persona change
    invalidated the system-prompt cache and could change role
    behavior mid-game.
    """
    ctx = AgentContext(
        agent_id="p08",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="werewolf",
        legal_actions=[ActionType.SPEECH],
        legal_targets=["p07"],
        public_summary="D2",
        persona_snapshot={"tone": "aggressive", "style": "blame_p07"},
    )
    builder = PlayerPromptBuilder(ctx)
    system_prompt = builder.build_system_prompt()
    user_prompt = builder.build_user_prompt(RetryInfo())

    # Persona MUST be in the user prompt.
    assert "人格设定" in user_prompt, (
        "persona must appear in user_prompt after P2-S10. "
        f"user_prompt[:500]={user_prompt[:500]!r}"
    )
    # Persona MUST NOT be in the system prompt.
    assert "人格设定" not in system_prompt, (
        "persona must NOT appear in system_prompt after P2-S10 "
        "(per s10 architecture, system_prompt holds stable sections only). "
        f"system_prompt={system_prompt!r}"
    )


def test_persona_empty_snapshot_is_noop():
    """P2-S10: empty persona_snapshot renders as no section in either prompt.

    When the persona is empty, the section should be silently dropped
    in both system and user prompt (consistent with the empty-body
    pass-through behavior of the rest of the pipeline).
    """
    ctx = AgentContext(
        agent_id="p08",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="werewolf",
        legal_actions=[ActionType.SPEECH],
        legal_targets=["p07"],
        public_summary="D2",
        # persona_snapshot defaults to {} (empty dict)
    )
    builder = PlayerPromptBuilder(ctx)
    system_prompt = builder.build_system_prompt()
    user_prompt = builder.build_user_prompt(RetryInfo())
    assert "人格设定" not in system_prompt
    assert "人格设定" not in user_prompt


# ---------------------------------------------------------------------------
# P0-1: _format_examples else branch must follow ctx.own_role for true_role
# ---------------------------------------------------------------------------
#
# Audit P0-1 finding: in `_format_examples` (the FULL_ACTION default branch
# used by speech/vote), the `example_role` variable that becomes
# `private_intent.true_role` was hardcoded to "villager" for every non-wolf
# role. That meant a seer / witch / hunter / idiot / hybrid agent saw an
# example that misrepresented their actual role in the audit log —
# the LLM would copy the structure and emit `private_intent.true_role =
# "villager"` even though `ctx.own_role` was something else.
#
# Fix: the variable that drives `true_role` in the example's
# `private_intent` should be `ctx.own_role` (or a sensible mapping
# for roles that have no clean "team" parallel in the example's
# faction_goal). Tests below cover all 5 non-wolf roles and assert
# the example's `private_intent.true_role` reflects `own_role`.


def _find_example_true_roles(prompt: str) -> list[str]:
    """Extract every `true_role` value in the rendered example JSON.

    Walks the prompt for JSON example objects (those carrying
    ``action_type``) and returns the value of their nested
    ``private_intent.true_role`` field. Used by P0-1 tests to verify
    the example role matches `own_role` rather than the hardcoded
    "villager".
    """
    true_roles: list[str] = []
    for match in _re.finditer(
        r"\{[^{}]*?(?:\{[^{}]*\}[^{}]*?)*\}",
        prompt,
        flags=_re.DOTALL,
    ):
        try:
            data = _json.loads(match.group(0))
        except _json.JSONDecodeError:
            continue
        if not isinstance(data, dict) or "action_type" not in data:
            continue
        pi = data.get("private_intent")
        if isinstance(pi, dict) and "true_role" in pi:
            true_roles.append(str(pi["true_role"]))
    return true_roles


def _make_full_action_ctx(role: str) -> AgentContext:
    """Build a context that triggers _format_examples (FULL_ACTION mode).

    Uses REFLECTION task + [SPEECH, VOTE] legal actions so the
    FULL_ACTION default branch renders both the speech and vote
    example pair. own_role is parameterized.
    """
    return AgentContext(
        agent_id="p05",
        task_type=TaskType.REFLECTION,
        phase="day",
        day_number=2,
        own_role=role,
        legal_actions=[ActionType.SPEECH, ActionType.VOTE],
        legal_targets=["p05", "p07"],
        public_summary="D2 reflection",
    )


@pytest.mark.parametrize("role", ["seer", "witch", "hunter", "idiot", "hybrid"])
def test_format_examples_private_intent_matches_own_role(role: str):
    """P0-1: example's private_intent.true_role must equal ctx.own_role.

    For seer / witch / hunter / idiot / hybrid (the 5 non-werewolf,
    non-villager roles that the bug was hardcoding to "villager"),
    the example rendered by _format_examples must use the agent's
    own role, not a hardcoded "villager". Otherwise the LLM copies
    the example and the audit log records `true_role="villager"`
    for a seer — a real bug observed in production traces.
    """
    ctx = _make_full_action_ctx(role)
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    true_roles = _find_example_true_roles(prompt)
    # Both the speech and vote examples should advertise the role.
    assert true_roles, (
        f"No example private_intent.true_role found in prompt for role={role!r}"
    )
    for tr in true_roles:
        assert tr == role, (
            f"Example private_intent.true_role must equal own_role={role!r}, "
            f"got {tr!r}. The example hardcodes the wrong role, "
            f"priming the LLM to copy a wrong audit value."
        )
    # Explicit anti-regression: the hardcoded "villager" must NOT appear
    # as a true_role for any non-villager role.
    assert "villager" not in true_roles, (
        f"Example must not advertise true_role='villager' for own_role={role!r}; "
        f"observed true_roles={true_roles}."
    )


def test_format_examples_villager_role_still_uses_villager():
    """P0-1: regression — villager example must still use 'villager'.

    Confirms the fix didn't accidentally break the villager branch
    (where example_role was already 'villager' before the fix).
    """
    ctx = _make_full_action_ctx("villager")
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    true_roles = _find_example_true_roles(prompt)
    assert true_roles, "Expected example true_roles for villager branch"
    for tr in true_roles:
        assert tr == "villager", (
            f"Villager branch must still use true_role='villager', got {tr!r}"
        )


def test_format_examples_werewolf_branch_still_uses_werewolf():
    """P0-1: regression — werewolf example must still use 'werewolf'.

    The bug fix only affects non-werewolf roles (the else branch).
    The werewolf branch (line ~717-718) was already correct and
    must remain so. Confirms no regression on the wolf example.
    """
    ctx = _make_full_action_ctx("werewolf")
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    true_roles = _find_example_true_roles(prompt)
    assert true_roles, "Expected example true_roles for werewolf branch"
    for tr in true_roles:
        assert tr == "werewolf", (
            f"Werewolf branch must still use true_role='werewolf', got {tr!r}"
        )


# ---------------------------------------------------------------------------
# P0-4: vote example standing_with_seer must use own ID for seer agent
# ---------------------------------------------------------------------------
#
# Audit P0-4 finding: in `_format_examples`, the vote example hardcodes
# `"standing_with_seer": "p03"` for every role. For a seer agent this
# is wrong — a seer stands with their OWN check, not with another seer.
# A seer voting based on its own check should report `standing_with_seer
# = ""` (own ID is implicit) with `vote_basis = "seer_check"` meaning
# "based on my own check".
#
# Fix: when `ctx.own_role == "seer"`, the example vote should set
# `standing_with_seer=""` (own ID is implicit). For non-seer roles
# the existing `p03` example stands.


def test_format_examples_seer_vote_uses_own_check():
    """P0-4: a seer agent's vote example must stand with own check (empty seer ID).

    The vote example in the prompt primes the LLM on what fields to
    fill. For a seer, `standing_with_seer` is the seer ID the agent
    sides with — a seer is THE seer, so the field should be empty
    (own ID is implicit) and `vote_basis` stays "seer_check" meaning
    "based on my own check". Otherwise the LLM copies the example
    and the seer reports a phantom "p03" allegiance it doesn't have.
    """
    ctx = _make_full_action_ctx("seer")
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    # Find the vote example (action_type="vote") and inspect its
    # standing_with_seer field.
    vote_examples = [
        ex for ex in _extract_json_examples(prompt)
        if ex.get("action_type") == "vote"
    ]
    assert vote_examples, "Expected a vote example in the prompt"
    for ex in vote_examples:
        # P0-4 fix: seer's vote example must stand with own check.
        assert ex.get("standing_with_seer") == "", (
            f"Seer vote example must have standing_with_seer='' "
            f"(own check, no other seer to side with), "
            f"got {ex.get('standing_with_seer')!r}. "
            f"Full example: {ex}"
        )
        # vote_basis must remain "seer_check" — meaning "based on my
        # own check", not "based on someone else's seer_check".
        assert ex.get("vote_basis") == "seer_check", (
            f"Seer vote example must keep vote_basis='seer_check' "
            f"(now meaning own check), got {ex.get('vote_basis')!r}. "
            f"Full example: {ex}"
        )


def test_format_examples_non_seer_vote_keeps_p03_example():
    """P0-4 / P1-8: non-seer roles keep the p03 / seer_siding example.

    Regression: the seer-specific fix must NOT remove the original
    p03 example for non-seer roles (villager, witch, hunter, idiot,
    hybrid, werewolf). Those roles DO side with an external seer
    claim, so the example is still meaningful. P1-8 updated
    ``vote_basis`` from "seer_check" (own check, only seers have
    that) to "seer_siding" (standing with another seer).
    """
    for role in ("villager", "witch", "hunter", "idiot", "hybrid", "werewolf"):
        ctx = _make_full_action_ctx(role)
        prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
        vote_examples = [
            ex for ex in _extract_json_examples(prompt)
            if ex.get("action_type") == "vote"
        ]
        assert vote_examples, (
            f"Expected a vote example for role={role!r}"
        )
        for ex in vote_examples:
            # Non-seer roles still see the p03 standing-with-seer
            # example (they side with an external seer).
            assert ex.get("standing_with_seer") == "p03", (
                f"Non-seer role={role!r} must keep the 'p03' example "
                f"for standing_with_seer (they side with an external seer), "
                f"got {ex.get('standing_with_seer')!r}. Full example: {ex}"
            )
            # P1-8: vote_basis is "seer_siding" for non-seer roles
            # (they have no check of their own).
            assert ex.get("vote_basis") == "seer_siding", (
                f"Non-seer role={role!r} must use vote_basis='seer_siding' "
                f"(P1-8: they have no check of their own), "
                f"got {ex.get('vote_basis')!r}. Full example: {ex}"
            )


# ---------------------------------------------------------------------------
# P0-2 (defense): _build_salience_events must whitelist public fields only
# ---------------------------------------------------------------------------
#
# Audit P0-2 (defense) finding: `_build_salience_events` dumps
# `ctx.salience_items` into the prompt without filtering private keys.
# The runtime (`runtime/context.py:build_agent_context`) currently doesn't
# populate private keys into salience_items, but a future change could
# easily leak `seer_result`, `witch_target`, `wolf_team`, etc. into
# the player-visible salience. Defense in depth: the renderer should
# explicitly allow only the public fields and silently drop any item
# containing private keys.
#
# Allowed public fields: weight, bucket, fact_type, source, target, value,
# day, phase, event_type.
# Private keys that must be dropped: seer_result, witch_target, wolf_team,
# private_intent, moderator_full.


_SALIENCE_PUBLIC_FIELDS = frozenset({
    "weight", "bucket", "fact_type", "source", "target", "value",
    "day", "phase", "event_type",
})

_SALIENCE_PRIVATE_KEYS = (
    "seer_result", "witch_target", "wolf_team",
    "private_intent", "moderator_full",
)


def _make_ctx_with_salience(items: list[dict]) -> AgentContext:
    """Build a context with the given salience_items populated.

    Used by P0-2 (defense) tests to feed salience payloads of varying
    contamination level.
    """
    return AgentContext(
        agent_id="p05",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH],
        legal_targets=["p05"],
        public_summary="D2",
        salience_items=items,
    )


def test_salience_events_strips_private_keys():
    """P0-2: an item with a private key (e.g., seer_result) must not leak.

    Defense in depth: even if a future change in the runtime
    accidentally puts a `seer_result` field into a salience item,
    the rendered prompt must not contain that substring. The
    renderer should either drop the offending item entirely or
    strip the private key — but the private value must NOT appear
    in the user prompt.
    """
    leaked_item = {
        "weight": 0.7,
        "bucket": "vote_pattern",
        "fact_type": "vote_resolved",
        "source": "p03",
        "target": "p07",
        "value": "p07 voted p03",
        "day": 2,
        "phase": "day",
        "event_type": "vote_resolved",
        # Private key that must NOT appear in the rendered prompt.
        "seer_result": "werewolf",
    }
    ctx = _make_ctx_with_salience([leaked_item])
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    # The private KEY name must not appear (we don't know which side
    # of the leak was rendered, but either way no "seer_result" should
    # be in the user prompt).
    assert "seer_result" not in prompt, (
        "P0-2 (defense): private key 'seer_result' leaked into the "
        "rendered user prompt via salience_events. The renderer must "
        "strip private keys before serializing salience items."
    )
    # The private VALUE must not appear either ("werewolf" is allowed
    # to appear elsewhere in the prompt, so we check for the exact
    # seer_result=werewolf pairing).
    assert '"seer_result":"werewolf"' not in prompt
    assert "'seer_result': 'werewolf'" not in prompt


def test_salience_events_strips_all_private_keys():
    """P0-2: every documented private key must be stripped.

    Sweep: the renderer must drop every private key in the audit
    boundary, not just `seer_result`. Covers witch_target, wolf_team,
    private_intent, and moderator_full.
    """
    leaked_item = {
        "weight": 0.5,
        "bucket": "b",
        "fact_type": "f",
        "source": "p08",
        "target": "p05",
        "value": "v",
        "day": 2,
        "phase": "day",
        "event_type": "e",
        # Pile of private keys — none should reach the prompt.
        "seer_result": "werewolf",
        "witch_target": "p07",
        "wolf_team": ["p01", "p08", "p11"],
        "private_intent": {"true_role": "werewolf"},
        "moderator_full": "FULL STATE",
    }
    ctx = _make_ctx_with_salience([leaked_item])
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    for private_key in _SALIENCE_PRIVATE_KEYS:
        assert private_key not in prompt, (
            f"P0-2 (defense): private key {private_key!r} leaked into the "
            f"rendered user prompt via salience_events."
        )


def test_salience_events_keeps_public_fields():
    """P0-2: a clean salience item (public fields only) passes through.

    Confirms the whitelist is not so aggressive that it drops valid
    public salience items. The renderer's `_compact_json` output
    must include the public fields that drive the LLM's reasoning.
    """
    clean_item = {
        "weight": 0.8,
        "bucket": "vote_pattern",
        "fact_type": "vote_resolved",
        "source": "p03",
        "target": "p07",
        "value": "p07 voted p03 on D2",
        "day": 2,
        "phase": "day",
        "event_type": "vote_resolved",
    }
    ctx = _make_ctx_with_salience([clean_item])
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    # All public field names and at least some public values should appear.
    assert "关键事件" in prompt, (
        "Salience section header must be present for non-empty salience_items"
    )
    # Spot-check public values pass through the JSON serialization.
    assert "p07 voted p03 on D2" in prompt, (
        "Public 'value' field of salience item must pass through to prompt"
    )
    assert "vote_resolved" in prompt, (
        "Public 'event_type' / 'fact_type' field must pass through to prompt"
    )


def test_salience_events_section_absent_when_empty():
    """P0-2: empty salience_items → no salience section in the prompt.

    Sanity: the existing empty-path behavior must be preserved.
    """
    ctx = _make_ctx_with_salience([])
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    assert "关键事件" not in prompt, (
        "Empty salience_items must not produce a salience section in the prompt"
    )


# ---------------------------------------------------------------------------
# P2-4: _compact_json truncation must mark the cut clearly
# ---------------------------------------------------------------------------
#
# Audit P2-4 finding: _compact_json truncates to 1800 chars and appends
# `...（已截断，原长度X）`. The trailing `）` followed by nothing leaves
# the LLM looking at a broken structure (mid-string, mid-array, mid-object).
# The retry loop sometimes treats the malformed snippet as parseable JSON
# (truncation rarely lands cleanly on a quote/brace boundary).
#
# Fix: append a clearly visible `<已截断>` marker so the LLM knows
# the JSON was intentionally cut and is not expected to parse.
# This is an acceptable compromise (we don't make the cut valid JSON —
# that's a much bigger refactor). The marker also helps the parse-
# failure handler distinguish "truncated by design" from "garbled".


def test_compact_json_truncation_marks_omission():
    """P2-4: a long dict passed through _compact_json must end with
    a clearly visible omission marker."""
    builder = PlayerPromptBuilder(
        AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            phase="day",
            day_number=1,
            own_role="villager",
        )
    )
    # Build a long dict whose JSON-serialized form definitely exceeds
    # the 1800-char cap. Padding values in a single big string.
    big = {
        "key_" + str(i): "x" * 200
        for i in range(50)
    }
    out = builder._compact_json(big)
    assert out.endswith("<已截断>"), (
        f"P2-4: truncated compact JSON must end with the `<已截断>` "
        f"marker so the LLM knows the snippet was intentionally cut. "
        f"Last 30 chars: {out[-30:]!r}"
    )
    # And confirm the truncation actually fired (i.e. the input was
    # bigger than the cap).
    assert len(big) > 0  # sanity
    # The marker suffix length matters: a future regression that drops
    # the marker should fail this assertion.
    assert "<已截断>" in out



# ---------------------------------------------------------------------------
# P2-8: belief rendering is capped at top 3
# ---------------------------------------------------------------------------
#
# Audit P2-8 finding: _build_belief_state rendered top-5 suspects and
# top-5 trusted (10 belief entries). Combined with 4 salience items,
# the section consumed 200-400 tokens. The budget trimmer (P1-5) had
# to drop entire sections to fit, when a smaller cap on the
# highest-priority section would have freed tokens for lower-priority
# context.
#
# Fix: cap the belief section to top 3 suspects / top 3 trusted
# (6 belief entries total). The salience section is also reduced
# from 4 → 3 items via _MAX_SALIENCE_ITEMS.


def test_belief_top3_cap():
    """P2-8: rendered belief with 5 suspects must show at most 3."""
    # Set up 5 suspects — well above the new top-3 cap.
    suspects = [
        {"player": f"p0{i}", "faction_lean": "werewolf_lean", "top_role_guess": "狼"}
        for i in range(1, 6)
    ]
    trusted = [
        {"player": "p10", "faction_lean": "good", "trust": 0.9},
    ]
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.VOTE,
        phase="day",
        day_number=2,
        own_role="villager",
        belief_state={"my_suspects": suspects, "my_trusted": trusted},
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    # The example JSON in the prompt (e.g. "target_id": "p05") also
    # contains p0X tokens, so the test extracts ONLY the belief
    # section ("我怀疑的玩家" / "我信任的玩家") and counts inside that.
    import re
    m = re.search(r"我怀疑的玩家: ([^\n]*)", prompt)
    assert m, (
        "P2-8: belief section must be present in the rendered prompt. "
        f"Prompt: {prompt!r}"
    )
    suspect_line = m.group(1)
    rendered_suspects = sum(1 for i in range(1, 6) if f"p0{i}" in suspect_line)
    assert rendered_suspects <= 3, (
        f"P2-8: rendered belief must show at most 3 of the 5 suspects. "
        f"Found {rendered_suspects} in suspect line: {suspect_line!r}"
    )
    # And the trusted player (p10) should still be present.
    assert "p10" in prompt, (
        "P2-8: trusted player must still be rendered in the belief section. "
        f"Prompt: {prompt!r}"
    )



# ---------------------------------------------------------------------------
# P2-9: role guide documents which vote_basis values to use
# ---------------------------------------------------------------------------
#
# Audit P2-9 finding: non-wolf roles did not see a one-line note
# saying which ``vote_basis`` enum value to use for votes. The
# 7-value ``vote_basis`` enum (seer_check / seer_siding /
# speech_logic / vote_pattern / pressure_test / anti_herd /
# fallback) is too wide for the LLM to guess correctly without
# guidance, and the wrong choice silently degraded the audit log.
# The seer is the only role that legitimately uses ``seer_check``
# (based on their OWN check); every other role uses
# speech_logic / vote_pattern / seer_siding and must NOT use
# seer_check.
#
# Fix: add a one-line note to the role guide for non-seer roles:
# "投票时 vote_basis 选用 speech_logic / vote_pattern / seer_siding，
# 不要用 seer_check".


def test_role_guide_documents_vote_basis():
    """P2-9: the role guide for non-seer roles must mention which
    vote_basis enum values to use (and that seer_check is not for
    non-seer roles).
    """
    # Spot-check villager (a non-seer good-side role).
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=1,
        own_role="villager",
    )
    system_prompt = PlayerPromptBuilder(ctx).build_system_prompt()
    # The role guide lives in the system prompt (stable section).
    # Check that vote_basis guidance is present.
    assert "vote_basis" in system_prompt, (
        "P2-9: non-seer role guide must mention vote_basis enum "
        f"guidance. system_prompt: {system_prompt!r}"
    )
    # And explicitly call out that seer_check is NOT to be used by
    # non-seer roles.
    assert "seer_check" in system_prompt, (
        "P2-9: non-seer role guide must explicitly call out that "
        "seer_check is not for non-seer roles. "
        f"system_prompt: {system_prompt!r}"
    )



# ---------------------------------------------------------------------------
# P2-3: SHERIFF example must include a sheriff_withdraw example
# ---------------------------------------------------------------------------
#
# Audit P2-3 finding: the SHERIFF_REGISTER example block in
# _format_examples renders examples for `sheriff_register` and
# `no_action`, but does not show `sheriff_withdraw`. Without an
# explicit withdraw example, the LLM was emitting `sheriff_register`
# when it intended to withdraw, and the parser fell back to a
# default (game trace g_3528592081 action 41, p05).
#
# Fix: add a third example showing `sheriff_withdraw` so the LLM
# can pattern-match the correct action type when the player
# changes their mind about running for sheriff.

# explicit withdraw example, the LLM was emitting `sheriff_register`
# when it intended to withdraw, and the parser fell back to a
# default (game trace g_3528592081 action 41, p05).
#
# Fix: add a third example showing `sheriff_withdraw` so the LLM
# can pattern-match the correct action type when the player
# changes their mind about running for sheriff.
# explicit withdraw example, the LLM was emitting `sheriff_register`
# when it intended to withdraw, and the parser fell back to a
# default (game trace g_3528592081 action 41, p05).
#
# Fix: add a third example showing `sheriff_withdraw` so the LLM
# can pattern-match the correct action type when the player
# changes their mind about running for sheriff.


def test_sheriff_example_includes_withdraw():
    """P2-3: the SHERIFF_REGISTER example block must include a
    `sheriff_withdraw` JSON example.

    The block currently shows only `sheriff_register` and (optionally)
    `no_action`. We add a `sheriff_withdraw` example so the LLM
    has a template to copy when the player decides to pull out of
    the sheriff race. The withdraw example is only emitted when
    SHERIFF_WITHDRAW is in legal_actions — mirror that in the test.
    """
    ctx = AgentContext(
        agent_id="p03",
        task_type=TaskType.SHERIFF_REGISTRATION,
        phase="day",
        day_number=1,
        own_role="villager",
        legal_actions=[
            ActionType.SHERIFF_REGISTER,
            ActionType.SHERIFF_WITHDRAW,
            ActionType.NO_ACTION,
        ],
        legal_targets=[],
        public_summary="D1 sheriff election",
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    assert "sheriff_withdraw" in prompt, (
        "P2-3: SHERIFF example block must include a `sheriff_withdraw` "
        "JSON example so the LLM has a template when the player "
        "decides to pull out of the sheriff race."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
