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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
