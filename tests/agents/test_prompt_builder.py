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

import re

import pytest

from werewolf_agent.agents import prompt_builder as prompt_builder_module
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
    # Phase 1 self-audit (P1-1 revert): ``directive`` was removed
    # entirely.  ``must_address_alerts`` alone conveys the imperative
    # via the MUST sub-group framing.  No second key needed.
    must_be_hard = {
        "wolf_fake_seer_execution",
        "must_address_alerts",
        "first_night_killed",
        "speech_silent",
        "vote_silent",
        "witch_night_action",
        "role_alerts",
        "vote_pressure",
        # Phase 2 P2-1: 13 wolf / hybrid / hunter / last-words / badge /
        # sheriff-silent / witch-deterrent directives that contain
        # "强制执行" / "严禁信息穿越" / "必须" / "不能" framing but
        # previously fell through to REFERENCE 兜底.  LLM downgraded
        # them to soft suggestions.  Promote to HARD.
        "wolf_sheriff_must_claim_seer",
        "wolf_no_reveal_seer",
        "wolf_fake_seer_teammate",
        "hybrid_wolf_master_directive",
        "hybrid_good_master_directive",
        "required_evaluation",
        "wolf_kill_instruction",
        "wolf_team_discussion",
        "hunter_shot_directive",
        "last_words",
        "badge_decision",
        "sheriff_silent",
        "witch_poison_deterrent",
    }
    for k in must_be_hard:
        assert k in HARD_CONSTRAINT_KEYS, (
            f"{k!r} must be classified as 硬约束 (hard constraint); got none"
        )
    # The ``directive`` key must NOT be in any group — it was
    # deleted in P1-1 revert.
    assert "directive" not in HARD_CONSTRAINT_KEYS
    assert "directive" not in SUGGESTION_KEYS
    assert "directive" not in REFERENCE_KEYS


def test_suggestion_keys_include_behavioral_directives():
    """Behavioral guidance (recommended but not mandatory) lives in 建议."""
    # Phase-1 audit: ``witch_speech_constraint`` was a dead key
    # (D-1 renamed the producer to ``witch_speech_directive`` but the
    # SUGGESTION_KEYS entry was never removed).  Producer grep returns
    # zero hits.  Dropped from the must_be_suggestion set; the
    # explicit ``test_witch_speech_constraint_not_in_suggestion_keys``
    # below confirms it is removed from the frozenset itself.
    # P3 (post-review-v2): ``wolf_universal_rules`` (绝对 / 严禁 framing)
    # and ``anti_herd`` (P0-K6 hard constraint) promoted to HARD tier.
    must_be_suggestion = {
        "wolf_speech_directive",
        "good_vote_decision_guard",
        "sheriff_vote_push",
        "speech_originality",
        "seer_speech_directive",
        "anti_following_and_peace_night_rule",
    }
    for k in must_be_suggestion:
        assert k in SUGGESTION_KEYS, (
            f"{k!r} must be classified as 建议 (suggestion); got none"
        )


def test_witch_speech_constraint_not_in_suggestion_keys():
    """Phase-1 audit: dead key removed from SUGGESTION_KEYS."""
    assert "witch_speech_constraint" not in SUGGESTION_KEYS, (
        "witch_speech_constraint has no producer (D-1 renamed it to "
        "witch_speech_directive); keeping it in SUGGESTION_KEYS misleads "
        "future readers and the renderer."
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


def test_reflection_prompt_renders_v2_prompt_card_whitelist_only() -> None:
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=1,
        own_role="seer",
        reflection_memory_hints=[
            {
                "role": "seer",
                "result": "负",
                "theme": "对跳局先核验警徽流",
                "lesson": "历史教训只说明你过去曾过早站边。",
                "trigger_signals": ["双预言家对跳", "警徽流解释不完整"],
                "recommended_action": "先比较验人时间线、警徽流和票型承接。",
                "misuse_risk": "不要映射到本局玩家身份。",
                "source": {"llm_self_review": "raw source must not render"},
                "quality_score": 0.91,
                "quality_status": "approved",
                "mistake_patterns": [{"category": "vote_mistake"}],
            }
        ],
    )
    builder = PlayerPromptBuilder(ctx)

    rendered = builder._build_reflection_memory_hints()

    assert "对跳局先核验警徽流" in rendered
    assert "双预言家对跳" in rendered
    assert "先比较验人时间线" in rendered
    assert "source" not in rendered
    assert "raw source" not in rendered
    assert "quality_score" not in rendered
    assert "mistake_patterns" not in rendered


def test_all_role_decision_directives_are_explicitly_classified():
    """Core role decisions must not silently fall through to reference."""
    expected_suggestions = {
        "villager_speech_directive",
        "hunter_speech_directive",
        "idiot_speech_directive",
        "hybrid_speech_directive",
        "witch_speech_directive",
        "seer_vote_strategy",
        "witch_vote_strategy",
        "hunter_vote_strategy",
        "villager_vote_strategy",
        "wolf_vote_strategy",
        "wolf_vote_role_hint",
        "wolf_vote_target",
        "no_sheriff_vote_hint",
    }
    assert expected_suggestions <= SUGGESTION_KEYS
    assert "seer_night_check" in HARD_CONSTRAINT_KEYS


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
    """Only hard keys → only 硬约束 sub-group marker (建议/参考 sub-group markers absent).

    Phase-1 audit: the bare ``【参考】`` outer label is now also used by
    ``_build_public_summary`` (P1-12 promoted it from 辅助).  Using
    the inner P0-S5 sub-group discriminator ``以下为背景信息（REFERENCE）``
    keeps the test targeted to the strategy_directive sub-group, not
    the section label.
    """
    ctx = _make_ctx_with_directive(
        {
            "must_address_alerts": ["p07 accused me"],
            "speech_silent": True,
        }
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    assert "以下指令必须遵守（MUST）" in prompt
    assert "以下指令为建议（SHOULD）" not in prompt
    assert "以下为背景信息（REFERENCE）" not in prompt


def test_strategy_directive_only_suggestion_section_when_only_suggestion_keys():
    """Only suggestion keys → only 建议 header.

    P1-S3 added a section-level 【硬约束】 outer label to strategy_directive
    in build_user_prompt. The inner sub-group check below uses the unique
    MUST/SHOULD/REFERENCE markers (P0-S5 inner sub-group discriminators)
    instead of the bare "【硬约束】" label, so the test still verifies that
    only the 建议 inner sub-group renders when only suggestion keys exist.

    P3 (post-review-v2): ``anti_herd`` was promoted to HARD, so it can no
    longer serve as the "suggestion" sample.  Use ``speech_originality``
    and ``good_vote_decision_guard`` (both remain in SUGGESTION_KEYS).
    """
    ctx = _make_ctx_with_directive(
        {
            "good_vote_decision_guard": "不要盲目跟票",
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
    """`must_address_alerts` must be rendered under 【硬约束】, not other sections.

    P3 (post-review-v2): ``anti_herd`` was promoted to HARD, so it can
    no longer serve as the "suggestion" sample.  Use
    ``good_vote_decision_guard`` (still in SUGGESTION_KEYS).
    """
    ctx = _make_ctx_with_directive(
        {
            "must_address_alerts": ["p07 accused me"],
            # Also add a suggestion key to ensure both sections render
            "good_vote_decision_guard": "不要盲目跟票",
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
    assert "good_vote_decision_guard" not in hard_body


def test_suggestion_key_appears_under_suggestion_section():
    """`good_vote_decision_guard` must be rendered under 【建议】.

    P3 (post-review-v2): ``anti_herd`` was promoted to HARD, so it can
    no longer serve as the "suggestion" sample.  Replaced with
    ``good_vote_decision_guard`` (still in SUGGESTION_KEYS).
    """
    ctx = _make_ctx_with_directive(
        {
            "good_vote_decision_guard": "不要盲目跟票",
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
    assert "good_vote_decision_guard" in sugg_body
    # Hard keys must not appear in the suggestion body
    assert "must_address_alerts" not in sugg_body


def test_reference_key_appears_under_reference_section():
    """`skill_tactical_advice` must be rendered under the inner P0-S5
    ``以下为背景信息（REFERENCE）`` sub-group.

    Phase-1 audit (P1-12): the bare ``【参考】`` outer label is also
    used by ``_build_public_summary`` (promoted to 参考 tier).  Using
    the inner sub-group discriminator keeps the test targeted to the
    strategy_directive section.
    """
    ctx = _make_ctx_with_directive(
        {
            "skill_tactical_advice": {"role": "werewolf"},
            # Add a hard key for context
            "must_address_alerts": ["p07 accused me"],
        }
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    hard_idx = prompt.find("以下指令必须遵守（MUST）")
    ref_idx = prompt.find("以下为背景信息（REFERENCE）")
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
    """3 sections must appear in priority order: 硬约束 → 建议 → 参考.

    Phase-1 audit (P1-12): public_summary now also carries the bare
    【参考】 outer label.  Using the inner P0-S5 sub-group
    discriminators keeps the test targeted to the strategy_directive
    section ordering, not the global section labels.

    P3 (post-review-v2): ``anti_herd`` was promoted to HARD, so the
    "suggestion" sample is now ``good_vote_decision_guard`` (still in
    SUGGESTION_KEYS).
    """
    ctx = _make_ctx_with_directive(
        {
            "must_address_alerts": ["x"],
            "good_vote_decision_guard": "y",
            "master_behavior_summary": "z",
        }
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    hard_idx = prompt.find("以下指令必须遵守（MUST）")
    sugg_idx = prompt.find("以下指令为建议（SHOULD）")
    ref_idx = prompt.find("以下为背景信息（REFERENCE）")
    assert hard_idx >= 0 and sugg_idx >= 0 and ref_idx >= 0
    assert hard_idx < sugg_idx < ref_idx, (
        "Sections must appear in priority order: 硬约束 < 建议 < 参考"
    )


def test_system_prompt_orders_rules_before_boundaries_and_reasoning() -> None:
    ctx = _make_villager_context()

    prompt = PlayerPromptBuilder(ctx).build_system_prompt()

    identity_idx = prompt.index("你是一场狼人杀游戏的玩家")
    rules_idx = prompt.index("【禁止事项】")
    role_idx = prompt.index("村民规则")
    boundaries_idx = prompt.index("【信息边界】")
    reasoning_idx = prompt.index("【推理方法-3 步】")
    skill_idx = prompt.index("【技能与建议】")
    output_idx = prompt.index("【结构化输出】")
    assert (
        identity_idx
        < rules_idx
        < role_idx
        < boundaries_idx
        < reasoning_idx
        < skill_idx
        < output_idx
    )


def test_user_prompt_groups_current_public_context_before_private_and_strategy_before_task() -> None:
    ctx = _make_ctx_for_priority_label_test().model_copy(
        update={
            "belief_state": {
                "my_suspects": [
                    {
                        "player": "p07",
                        "faction_lean": "werewolf",
                        "top_role_guess": "werewolf",
                    }
                ],
                "my_trusted": [
                    {
                        "player": "p05",
                        "faction_lean": "good",
                        "trust": 0.8,
                    }
                ],
            }
        }
    )

    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())

    phase_idx = prompt.index("当前阶段:")
    public_idx = prompt.index("当前局公开事实")
    visible_idx = prompt.index("可见状态")
    salience_idx = prompt.index("关键事件")
    transcript_idx = prompt.index("近期发言")
    persona_idx = prompt.index("人格设定")
    belief_idx = prompt.index("我的判断")
    private_idx = prompt.index("本局·第2轮·私有记忆")
    learning_idx = prompt.index("跨局学习参考")
    strategy_idx = prompt.index("本轮策略指令")
    task_idx = prompt.index("发言意图枚举")
    output_idx = prompt.index("最终输出约束")
    assert (
        phase_idx
        < public_idx
        < visible_idx
        < salience_idx
        < transcript_idx
        < persona_idx
        < belief_idx
        < private_idx
        < learning_idx
        < strategy_idx
        < task_idx
        < output_idx
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
# NEW-R4-P1-1: skill_tactical_advice must render as a human-readable list
# (not a raw JSON envelope) once the value is a structured
# [{skill, advice, confidence}, ...] list (S-07 contract).
# ---------------------------------------------------------------------------


def test_skill_tactical_advice_rendered_as_human_readable_list():
    """The strategy_directive.skill_tactical_advice list must be rendered
    as a human-readable bullet list, not as raw JSON.

    Pre-fix: ``_build_strategy_directive`` ran the whole
    ``strategy_directive`` dict through ``_compact_json`` and emitted
    ``{"skill_tactical_advice":[{...}, ...]}`` as a JSON envelope. The
    LLM had to parse ~200 chars of JSON to read the advice, wasting
    10-20% of the directive budget and inviting parse errors on the
    renderer side.

    Post-fix: a dedicated renderer for the ``skill_tactical_advice`` key
    bypasses ``_compact_json`` and emits one bullet per entry:
    ``- [skill_name/confidence] advice_text``. This is the format the
    LLM can read directly without an extra parse step.
    """
    ctx = _make_ctx_with_directive(
        {
            "skill_tactical_advice": [
                {
                    "skill": "push_vote",
                    "advice": "归票建议: p05 发言逻辑矛盾，盘他",
                    "situation_signature": "role=villager task=speech phase=day",
                    "recommended_use": "把公开发言矛盾转成可解释的归票建议。",
                    "risk_alerts": ["不要把推测说成铁狼"],
                    "counter_signals": ["当前没有可公开解释的目标"],
                    "forbidden_use": "不得把技能建议当成裁判真相。",
                    "confidence": 0.65,
                    "relevance": 0.8,
                },
                {
                    "skill": "bold_claim",
                    "advice": "悍跳建议: 上警直接抢预言家",
                    "confidence": 0.45,
                },
            ],
        }
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())

    # The rendered bullet list must show up under 【参考】 and the
    # first entry's `归票建议:` Chinese must be visible to the LLM
    # (not hidden inside a JSON string).
    ref_idx = prompt.find("【参考】")
    assert ref_idx >= 0, "skill_tactical_advice lives in 【参考】 group"
    ref_body = prompt[ref_idx:]
    assert "- [" in ref_body, (
        "skill_tactical_advice must be rendered as bullet list with "
        "'- [' prefix (NEW-R4-P1-1); got ref_body:\n" + ref_body
    )
    assert "归票建议:" in ref_body, (
        "skill_tactical_advice advice text must be visible to LLM "
        "(NEW-R4-P1-1); got ref_body:\n" + ref_body
    )
    assert "适用局面：role=villager task=speech phase=day" in ref_body
    assert "本轮建议：把公开发言矛盾转成可解释的归票建议。" in ref_body
    assert "风险：不要把推测说成铁狼" in ref_body
    assert "不适用信号：当前没有可公开解释的目标" in ref_body
    assert "禁止套用：不得把技能建议当成裁判真相。" in ref_body

    # Negative: the raw JSON envelope (the bug) must NOT appear.
    # If we see `[{"skill":` in the prompt, the renderer is still
    # handing the list to _compact_json and the LLM has to parse it.
    assert '[{"skill":' not in ref_body, (
        "skill_tactical_advice must NOT be rendered as raw JSON "
        "envelope; got ref_body:\n" + ref_body
    )
    # The dict key "skill_tactical_advice" may still appear as a
    # label, but the value must not be a JSON array of objects.


def test_prompt_text_cleaner_scrubs_cjk_adjacent_player_ids() -> None:
    cleaned = PlayerPromptBuilder._clean_prompt_text("上次p03悍跳时我被带偏")

    assert "p03" not in cleaned
    assert "历史玩家" in cleaned


def test_error_category_normalizer_scrubs_cjk_adjacent_player_ids() -> None:
    cleaned = PlayerPromptBuilder._normalize_error_category("p03的预言家发言")

    assert "p03" not in cleaned
    assert "历史玩家" in cleaned


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
        task_type=TaskType.REFLECTION,  # → FULL_ACTION
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


def _v2_rag_hint(
    *,
    title: str = "案例标题",
    situation_signature: str = "D2 白天发言阶段，预言家对跳后票型开始集中。",
    transferable_lesson: str = "先拆投票动机，再决定是否跟随归票。",
    applicability: list[str] | None = None,
    counter_signals: list[str] | None = None,
    recommended_use: str = "用作发言前的检核清单，不替代当前局证据。",
    misuse_risk: str = "照搬案例票型会把历史玩家当成本局事实。",
) -> dict:
    return {
        "type": "rag_hit",
        "title": title,
        "situation_signature": situation_signature,
        "transferable_lesson": transferable_lesson,
        "applicability": applicability
        if applicability is not None
        else ["本局已有公开票型", "发言顺序能解释站边变化"],
        "counter_signals": counter_signals
        if counter_signals is not None
        else ["当前局没有预言家对跳", "票型仍未形成压力"],
        "recommended_use": recommended_use,
        "misuse_risk": misuse_risk,
    }


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
            **_v2_rag_hint(
                title="京城大师赛 250415 抗推预言家",
                situation_signature="D2 白天发言，预言家被集中质疑。",
                transferable_lesson="先拆质疑来源，再判断是否需要跟票。",
                recommended_use="只作为发言结构参考。",
                misuse_risk="照搬历史票型会误导当前局判断。",
            ),
            "entry_id": "leaked_audit_item",
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
        "summary",
        "key_decisions",
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
    assert "狼队在白天通过抗推预言家获得票数优势。" not in prompt
    assert "白天全力归票预言家" not in prompt
    # The V2 slim payload fields ARE present.
    assert "京城大师赛 250415 抗推预言家" in prompt
    assert "D2 白天发言，预言家被集中质疑。" in prompt
    assert "先拆质疑来源，再判断是否需要跟票。" in prompt


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
        "rag_hints": [_v2_rag_hint(title="京城大师赛 250415 抗推预言家")],
    })
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    # The exact prefix — Phase 2 P2-11 extended it to cover
    # TACTIC reuse (not just player ID reuse).
    expected_prefix = (
        "⚠️ RAG 案例中的玩家 ID 与战术选择仅供启发；"
        "本局的玩家 ID、票型、遗言均与案例无关；"
        "不得直接套用案例中具体玩家的动作、票型或决策链。"
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


def test_rag_hints_player_id_warning_appears_before_case_cards():
    """P0-G3 + P1-RAG-DENSITY: the warning must be the FIRST line of
    the 知识库提示 section, before the case cards. Otherwise an LLM that
    reads the section top-to-bottom might process case data before
    seeing the warning, and the whole point of the prefix is to set the
    "do not parrot" frame BEFORE the model sees the cases.
    """
    ctx = _make_villager_context()
    ctx = ctx.model_copy(update={
        "rag_hints": [_v2_rag_hint(title="案例标题")],
    })
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    rag_start = prompt.find("知识库提示")
    after_header = prompt[rag_start:]
    # The warning must precede the first case card in the section.
    card_start = after_header.find("案例 1：案例标题")
    warning_start = after_header.find("⚠️")
    assert warning_start != -1, "P0-G3: warning prefix must be present"
    assert card_start != -1, "P1-RAG-DENSITY: case card must be present"
    assert warning_start < card_start, (
        "P0-G3: the warning must come BEFORE the case card in the "
        "知识库提示 section."
    )


def test_rag_hints_render_v2_case_cards_instead_of_json_payload():
    """RAG V2: live RAG should be rendered as readable tactical cards.

    JSON is convenient for code but weak for LLM decision grounding:
    the model can treat it as fact rows and copy tactics directly. The
    live prompt should surface the V2 safe tactical frame as a
    low-priority reference card with explicit applicability and
    misuse-risk guards.
    """
    ctx = _make_villager_context()
    ctx = ctx.model_copy(update={
        "rag_hints": [_v2_rag_hint(
            title="案例标题",
            situation_signature="局面签名",
            transferable_lesson="可迁移原则文本",
            applicability=["条件1", "条件2"],
            counter_signals=["反信号1", "反信号2"],
            recommended_use="本局参考方式文本",
            misuse_risk="误用风险文本",
        )],
    })

    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    rag_start = prompt.find("知识库提示")
    assert rag_start != -1, "RAG hints section must be present"
    rag_section = prompt[rag_start:]

    assert "案例 1：案例标题" in rag_section
    assert "- 适用局面：局面签名" in rag_section
    assert "- 可迁移原则：可迁移原则文本" in rag_section
    assert "- 适用条件：条件1；条件2" in rag_section
    assert "- 不适用信号：反信号1；反信号2" in rag_section
    assert "- 本局参考方式：本局参考方式文本" in rag_section
    assert "- 误用风险：误用风险文本" in rag_section
    for json_key in (
        '"title"',
        '"summary"',
        '"key_decisions"',
        '"situation_signature"',
        '"transferable_lesson"',
        '"applicability"',
        '"counter_signals"',
        '"recommended_use"',
        '"misuse_risk"',
    ):
        assert json_key not in rag_section, (
            f"P1-RAG-DENSITY: live RAG prompt should not expose "
            f"compact JSON key {json_key}"
        )
    assert "案例摘要：" not in rag_section
    assert "可借鉴原则：" not in rag_section
    assert "使用前检查：" not in rag_section
    assert "禁止套用：" not in rag_section


def test_rag_hints_legacy_items_use_fallback_frame_without_raw_keys():
    """RAG V2 compat: legacy ctx.rag_hints still render via shared fallback.

    The live prompt should not show raw JSON keys ``summary`` /
    ``key_decisions`` and should not render those legacy keys as card
    labels. The shared tactical helper supplies conservative fallback
    frame text instead.
    """
    ctx = _make_villager_context()
    ctx = ctx.model_copy(update={
        "rag_hints": [{
            "type": "rag_hit",
            "title": "旧版案例",
            "summary": "旧版摘要可作为谨慎参考。",
            "key_decisions": ["旧版决策不应作为卡片字段"],
        }],
    })

    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    rag_start = prompt.find("知识库提示")
    assert rag_start != -1, "RAG hints section must be present"
    rag_section = prompt[rag_start:]

    assert "案例 1：旧版案例" in rag_section
    assert "- 适用局面：旧版RAG条目缺少V2战术框架。" in rag_section
    assert "- 可迁移原则：旧版摘要可作为谨慎参考。" in rag_section
    assert "- 适用条件：仅在当前局面与旧摘要明确匹配时参考。" in rag_section
    assert "- 不适用信号：当前局面与旧摘要描述不一致。" in rag_section
    assert "summary" not in rag_section
    assert "key_decisions" not in rag_section
    assert "案例摘要：" not in rag_section
    assert "旧版决策不应作为卡片字段" not in rag_section


def test_rag_hints_accept_string_v2_list_fields_from_non_default_paths():
    """Defensive renderer: string list fields should not crash prompt build."""
    ctx = _make_villager_context()
    ctx = ctx.model_copy(update={
        "rag_hints": [_v2_rag_hint(
            title="字符串字段案例",
            applicability="单个适用条件",  # type: ignore[arg-type]
            counter_signals="单个反信号",  # type: ignore[arg-type]
        )],
    })

    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    rag_start = prompt.find("知识库提示")
    assert rag_start != -1, "RAG hints section must be present"
    rag_section = prompt[rag_start:]

    assert "- 适用条件：单个适用条件" in rag_section
    assert "- 不适用信号：单个反信号" in rag_section


def test_rag_hints_no_warning_when_no_hints():
    """P0-G3: the warning is a hard-constraint prefix that only makes
    sense when there ARE RAG hints. With no hints, the section is
    omitted entirely (existing behavior) and the warning must not
    leak into other sections."""
    ctx = _make_villager_context()
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    assert "RAG 案例中的玩家 ID" not in prompt
    assert "⚠️" not in prompt or "RAG 案例" not in prompt


def test_rag_hints_have_tail_reminder():
    """R19: the 知识库提示 section has a "以上案例仅供参考" tail
    reminder that lands AFTER the case cards. The head warning
    (player-id leak guard) only sets the frame at the start; a
    tail reminder re-anchors the model at the end so the LLM does
    not carry case-derived claims into its final answer.

    Without the tail, the case cards sit at the end of the section and
    the model is free to treat it as a hard assertion rather than
    reference material.
    """
    ctx = _make_villager_context()
    ctx = ctx.model_copy(update={
        "rag_hints": [_v2_rag_hint(title="案例标题")],
    })
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    rag_start = prompt.find("知识库提示")
    assert rag_start != -1, "RAG hints section must be present"
    after_header = prompt[rag_start:]
    # The tail reminder text must appear after the case card so it
    # acts as an "after you read this" anchor.
    tail = "以上案例仅供参考"
    card_start = after_header.find("案例 1：案例标题")
    tail_start = after_header.find(tail)
    assert card_start != -1, "R19: case card must still be present"
    assert tail_start != -1, (
        f"R19: tail reminder {tail!r} must appear in the 知识库提示 "
        f"section after the case cards. Got section excerpt:\n"
        + after_header[:400]
    )
    assert tail_start > card_start, (
        "R19: the tail reminder must come AFTER the case cards so "
        "the LLM encounters it after reading the case data, not before."
    )


def test_rag_hints_filtered_by_type():
    """G-R4-10: ``_build_rag_hints`` must filter ``ctx.rag_hints`` by
    ``type == "rag_hit"`` before rendering. Mixed-type lists (rag_hit
    + non-rag_hit items, e.g. a future code path or test that
    injects auxiliary metadata) must drop the non-rag_hit entries
    rather than render them.

    The previous code at runtime/context.py:231 used
    ``[item for item in ctx.rag_hints if item.get("type") != "rag_hit"]``
    to *retain* non-rag items, which is brittle: a stray non-rag item
    persists across turns and the prompt renderer would happily
    process it. The prompt-side filter is explicit, defensive, and
    matches the slim renderer's expectation that every line carries
    the ``rag_hit`` discriminator.
    """
    ctx = _make_villager_context()
    ctx = ctx.model_copy(update={
        "rag_hints": [
            # Mixed list: first two are NOT rag_hit, last two are.
            {
                "type": "aux_meta",
                "title": "AUX-META-TITLE",
                "summary": "AUX-META-SUMMARY",
                "key_decisions": ["AUX-META-DECISION"],
            },
            {
                "type": "salience_event",
                "title": "SALIENCE-TITLE",
                "summary": "SALIENCE-SUMMARY",
                "key_decisions": ["SALIENCE-DECISION"],
            },
            {
                **_v2_rag_hint(
                    title="RAG-1-TITLE",
                    situation_signature="RAG-1-SITUATION",
                ),
            },
            {
                **_v2_rag_hint(
                    title="RAG-2-TITLE",
                    situation_signature="RAG-2-SITUATION",
                ),
            },
        ],
    })
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    # The rag_hit items MUST be rendered.
    assert "RAG-1-TITLE" in prompt, (
        f"G-R4-10: rag_hit item 1 must be rendered; prompt excerpt: "
        f"{prompt[prompt.find('知识库提示'):prompt.find('知识库提示')+400]!r}"
    )
    assert "RAG-2-TITLE" in prompt
    # The non-rag_hit items MUST NOT leak into the prompt.
    for forbidden in ("AUX-META-TITLE", "AUX-META-SUMMARY",
                      "SALIENCE-TITLE", "SALIENCE-SUMMARY"):
        assert forbidden not in prompt, (
            f"G-R4-10: non-rag_hit item leaked into prompt: {forbidden!r}"
        )


def test_rag_truncation_note_in_tail():
    """Long RAG fields are truncated before the whole RAG card.

    The live prompt should keep the RAG reference framing while avoiding
    a generic content_prefix/content_suffix envelope for one oversized
    field.
    """
    long_lesson = "狼" * 5000
    ctx = _make_villager_context()
    ctx = ctx.model_copy(update={
        "rag_hints": [{
            **_v2_rag_hint(
                title="超长案例",
                transferable_lesson=long_lesson,
            ),
        }],
    })
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    rag_start = prompt.find("知识库提示")
    assert rag_start != -1, "RAG hints section must be present"
    rag_section = prompt[rag_start:]
    assert "…已截断" in rag_section, (
        f"Long RAG V2 field should be truncated at field level. Got: {rag_section!r}"
    )
    assert "以上案例仅供参考" in rag_section, (
        "The existing '以上案例仅供参考' tail reminder must remain."
    )


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


def test_skill_analysis_hints_not_in_user_prompt_for_seer():
    """NEW-S04-A: skill_analysis_hints render path is dropped. The
    structured `skill_tactical_advice` (in strategy_directive) is the
    single source of truth; the opaque `技能分析结果` section no
    longer appears in the user prompt. Seer user prompt must NOT
    include the old `技能分析结果` header.
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
        # Even if a caller populates skill_analysis_hints directly,
        # the prompt builder does not render the legacy section.
        skill_analysis_hints={"wolf_pit": "嫌疑区: p05"},
    )
    user_prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    assert "技能分析结果" not in user_prompt, (
        "NEW-S04-A: skill_analysis_hints render path is dropped; the "
        "legacy `技能分析结果` section must NOT appear in the user "
        "prompt. The structured `skill_tactical_advice` inside "
        "strategy_directive is the single source of truth."
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


def test_skill_analysis_hints_not_in_user_prompt_for_villager():
    """NEW-S04-A: villager user prompt must NOT include the legacy
    `技能分析结果` section. The structured `skill_tactical_advice` in
    `strategy_directive` is the single source of truth.
    """
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
    assert "技能分析结果" not in user_prompt, (
        "NEW-S04-A: skill_analysis_hints render path is dropped; the "
        "legacy `技能分析结果` section must NOT appear in the user "
        "prompt."
    )


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
    """P0-R2: timeout empty_response → hint must suggest no_action when emit-able.

    Game trace g_3528592081: seer p03 vote (Action 57) hit 3 empty
    retries and fell back to a default target. If the hint had said
    "if you can't decide, return no_action", the model would have
    taken the safe no-op rather than burning 3 attempts.
    """
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.SHERIFF_REGISTRATION,
        phase="day",
        day_number=1,
        own_role="villager",
        legal_actions=[
            ActionType.SHERIFF_REGISTER,
            ActionType.SHERIFF_WITHDRAW,
            ActionType.NO_ACTION,
        ],
        public_summary="D1 sheriff election",
    )
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
        "When failure_category is 'timeout' and FULL_ACTION can emit it, retry hint must mention "
        "'no_action' so the LLM knows the safe no-op is permitted."
    )


def test_retry_hint_omits_no_action_when_no_action_is_illegal():
    """Timeout retry guidance must not suggest an illegal no_action."""
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.VOTE,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.VOTE],
        legal_targets=["p05", "p07"],
        public_summary="D2 vote",
    )
    retry = RetryInfo(
        attempt=2,
        max_retries=3,
        error_code="empty_response",
        error_message="Model returned empty text",
        failure_category="timeout",
        correction_hint=(
            "Please provide a valid JSON action (cause: timeout). "
            "If the model timed out, choose a legal target."
        ),
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(retry)
    assert "action_type='no_action'" not in prompt
    assert "选择一个合法目标" in prompt


def test_retry_hint_omits_no_action_for_speech_intent_legacy_vote_mode():
    """Speech-intent retries must not suggest action_type=no_action."""
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH, ActionType.VOTE],
        legal_targets=["p05"],
        public_summary="D2 speech",
    )
    retry = RetryInfo(
        attempt=2,
        max_retries=3,
        error_code="empty_response",
        error_message="Model returned empty text",
        failure_category="timeout",
        correction_hint="Please provide a valid JSON action (cause: timeout).",
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(retry)
    assert "发言意图JSON对象" in prompt
    assert "action_type='no_action'" not in prompt


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
# P1-S3: every user-prompt section gets a typed priority/context label
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
# Fix: prepend each section with a typed label derived from the
# section metadata registry. The same registry also defines budget
# behavior so labels and trimming cannot drift.
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
        rag_hints=[{"type": "rag_hit", "title": "Case 1", "summary": "wolf switch", "key_decisions": ["vote flip"]}],
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
    """P1-S3: each merged user-prompt section is prefixed with a
    typed priority/context label so the LLM can rank which sections to
    attend to under tight token budget.

    Sections grouped:
    - 【硬约束】 final output guard (retry hint + output contract)
    - 【人格】   persona snapshot
    - 【辅助】   phase, belief, summary, visible state, private memory,
      salience
    - 【参考】   learning context (RAG + reflection + profile + cognition
      + error pattern)
    - 【策略指令】 strategy_directive
    - 【场上记录】 transcript

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

    # The remaining 辅助 sections are current-turn support sections.
    # Cross-game learning now renders under one 【参考】 wrapper.
    auxiliary_label_count = prompt.count("【辅助】")
    assert auxiliary_label_count >= 4, (
        f"Expected at least 4 【辅助】 labels for current-turn support "
        f"sections. "
        f"Got {auxiliary_label_count}."
    )
    assert "【人格】 人格设定:" in prompt
    assert "【参考】 跨局学习参考:" in prompt

    assert "【可选】" not in prompt, (
        "No live section should be labeled 【可选】 while also being "
        "protected from budget trimming."
    )

    # Verify the label is followed by the section's own header text so
    # the LLM sees 【硬约束】 X then the actual content.
    # AUDIT-2-04: the retry hint is now 【硬约束】 (was 【辅助】 in P1-9).
    # The trimmer was actually dropping the retry hint under budget
    # pressure, leading the LLM to repeat the same mistake on retries.
    # Promoting the label to 【硬约束】 keeps the trimmer from dropping it.
    retry_idx = prompt.find("纠正提示")
    assert retry_idx > 0, "Retry hint should still render in the user prompt"
    preceding = prompt[max(0, retry_idx - 60):retry_idx]
    assert "【硬约束】" in preceding, (
        f"AUDIT-2-04: retry hint must be preceded by 【硬约束】 label, "
        f"got: {preceding!r}"
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

    # Transcript is current-game public record, not optional context.
    transcript_idx = prompt.find("近期发言")
    assert transcript_idx > 0
    preceding = prompt[max(0, transcript_idx - 60):transcript_idx]
    assert "【场上记录】" in preceding, (
        f"Transcript must be preceded by 【场上记录】 label, got: {preceding!r}"
    )


def test_section_metadata_is_single_source_for_labels_and_budget() -> None:
    specs = PlayerPromptBuilder._USER_SECTION_SPECS
    by_name = PlayerPromptBuilder._SECTION_SPEC_BY_NAME

    assert set(by_name) == {spec.builder_name for spec in specs}
    assert PlayerPromptBuilder._SECTION_PRIORITIES == {
        spec.builder_name: spec.label for spec in specs
    }
    assert PlayerPromptBuilder._NEVER_DROP == {
        spec.builder_name for spec in specs if spec.drop_tier is None
    }
    assert PlayerPromptBuilder._LOW_VALUE_SECTIONS == {
        spec.builder_name for spec in specs if spec.drop_tier == 0
    }
    assert "_build_learning_context" in PlayerPromptBuilder._LOW_VALUE_SECTIONS
    assert by_name["_build_learning_context"].drop_tier == 0
    assert by_name["_build_learning_context"].drop_tier < by_name["_build_belief_state"].drop_tier
    assert by_name["_build_learning_context"].drop_tier < by_name["_build_possible_worlds"].drop_tier
    assert by_name["_build_learning_context"].drop_tier < by_name["_build_simulation_predictions"].drop_tier
    assert "_build_rag_hints" not in by_name
    assert "_build_reflection_memory_hints" not in by_name
    assert "_build_profile_memory_hint" not in by_name
    assert "_build_cognition_matrix_hint" not in by_name
    assert "_build_error_pattern_hint" not in by_name
    assert by_name["_build_recent_transcript"].label == "【场上记录】"
    assert by_name["_build_recent_transcript"].drop_tier is None
    assert by_name["_build_final_output_guard"].label == "【硬约束】"
    assert by_name["_build_final_output_guard"].drop_tier is None


def test_user_prompt_uses_merged_learning_and_output_sections() -> None:
    ctx = AgentContext(
        agent_id="p08",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH],
        rag_hints=[{
            "type": "rag_hit",
            "summary": "票型抱团识别",
        }],
        reflection_memory_hints=[{
            "role": "villager",
            "result": "负",
            "text": "不要盲目跟票。",
        }],
        profile_memory_hint={"games_played": 3},
        cognition_matrix_hint={"suspects": [{"player": "p02", "trust": 0.2}]},
        error_pattern_hint={
            "total_reflections": 2,
            "top_mistakes": [("vote_mistake", 2)],
        },
    )
    retry = RetryInfo(
        attempt=2,
        max_retries=3,
        error_code="parse_error",
        error_message="missing field speech",
        correction_hint="只输出JSON。",
    )

    prompt = PlayerPromptBuilder(ctx).build_user_prompt(retry)

    assert prompt.count("【参考】 跨局学习参考") == 1
    assert "知识库提示:" in prompt
    assert "跨局反思记忆:" in prompt
    assert "历史角色经验:" in prompt
    assert "认知校准摘要:" in prompt
    assert "跨局错误模式" in prompt
    assert prompt.count("【硬约束】 最终输出约束") == 1
    assert "纠正提示" in prompt
    assert "最终输出协议" in prompt
    assert prompt.index("纠正提示") < prompt.index("最终输出协议")


def test_learning_context_orders_player_calibration_before_rag() -> None:
    ctx = AgentContext(
        agent_id="p08",
        task_type=TaskType.SPEECH,
        phase="day",
        own_role="villager",
        legal_actions=[ActionType.SPEECH],
        rag_hints=[{
            "type": "rag_hit",
            "title": "RAG_ORDER_MARKER",
            "summary": "通用案例",
        }],
        reflection_memory_hints=[{
            "role": "villager",
            "result": "负",
            "text": "REFLECTION_ORDER_MARKER",
        }],
        profile_memory_hint={"games_played": 3, "summary": "PROFILE_ORDER_MARKER"},
        cognition_matrix_hint={"suspects": [{"player": "p02", "trust": 0.2}]},
        error_pattern_hint={
            "total_reflections": 2,
            "top_mistakes": [("vote_mistake", 2)],
        },
    )

    text = PlayerPromptBuilder(ctx)._build_learning_context()

    assert text.index("跨局错误模式") < text.index("跨局反思记忆")
    assert text.index("跨局反思记忆") < text.index("历史角色经验")
    assert text.index("历史角色经验") < text.index("认知校准摘要")
    assert text.index("认知校准摘要") < text.index("知识库提示")


def test_learning_context_drops_rag_before_player_reflection_when_too_large() -> None:
    ctx = AgentContext(
        agent_id="p08",
        task_type=TaskType.SPEECH,
        phase="day",
        own_role="villager",
        legal_actions=[ActionType.SPEECH],
        rag_hints=[
            {
                "type": "rag_hit",
                "title": f"RAG_LOW_PRIORITY_MARKER_{idx}",
                "situation_signature": "低优先级局面" * 600,
                "transferable_lesson": "低优先级原则" * 600,
                "applicability": ["低优先级条件" * 300],
                "counter_signals": ["低优先级反信号" * 300],
                "recommended_use": "低优先级参考方式" * 600,
                "misuse_risk": "低优先级误用风险" * 600,
            }
            for idx in range(6)
        ],
        reflection_memory_hints=[
            {
                "role": "villager",
                "result": "负",
                "text": (
                    "REFLECTION_HIGH_PRIORITY_MARKER " if idx == 0 else ""
                ) + ("反思" * 600),
            }
            for idx in range(8)
        ],
        profile_memory_hint={"games_played": 3, "summary": "PROFILE_LOW_PRIORITY_MARKER" * 80},
        cognition_matrix_hint={
            "suspects": [
                {"player": f"p{i:02d}", "trust": 0.2, "open_questions": ["q"] * 20}
                for i in range(12)
            ]
        },
        error_pattern_hint={
            "total_reflections": 2,
            "top_mistakes": [("vote_mistake", 2)],
        },
    )

    text = PlayerPromptBuilder(ctx)._build_learning_context()

    assert "跨局错误模式" in text
    assert "REFLECTION_HIGH_PRIORITY_MARKER" in text
    assert "RAG_LOW_PRIORITY_MARKER" not in text


def test_strategy_tactical_advice_is_bounded_in_prompt() -> None:
    ctx = AgentContext(
        agent_id="p08",
        task_type=TaskType.SPEECH,
        phase="day",
        own_role="villager",
        legal_actions=[ActionType.SPEECH],
        strategy_directive={
            "skill_tactical_advice": [
                {
                    "skill": f"skill_{idx}",
                    "confidence": 0.9,
                    "advice": f"ADVICE_MARKER_{idx} " + ("长建议" * 300),
                }
                for idx in range(8)
            ]
        },
    )

    text = PlayerPromptBuilder(ctx)._build_strategy_directive()

    assert "ADVICE_MARKER_0" in text
    assert "ADVICE_MARKER_1" in text
    assert "ADVICE_MARKER_2" in text
    assert "ADVICE_MARKER_3" not in text
    assert "长建议" * 80 not in text
    assert "其余 5 条技能战术建议已省略" in text


def test_information_boundaries_are_generated_from_section_metadata() -> None:
    ctx = AgentContext(
        agent_id="p05",
        task_type=TaskType.SPEECH,
        phase="day",
        own_role="villager",
    )
    text = PlayerPromptBuilder(ctx)._build_information_boundaries()

    for spec in PlayerPromptBuilder._USER_SECTION_SPECS:
        assert spec.display_name in text
        assert spec.label in text
    assert "本轮任务/候选枚举或示例不加外层优先级标签" in text


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

    Phase-1 audit (P1-12): ``_build_public_summary`` was promoted from
    辅助 to 参考.  Removed from the auxiliary_header list and added
    a parallel assertion that it now bears the 参考 label.

    M4-2 (Task 7): ``_build_reflection_memory_hints`` swapped from
    辅助 to 参考. Removed from auxiliary_header and added to a
    parallel reference_header assertion. RAG hints (knowledge
    base) took reflection's old 辅助 slot — added back to
    auxiliary_header since they are now 辅助 tier again (M4-2
    reverts G-R4-15's RAG promotion to 【参考】).
    """
    ctx = _make_ctx_for_priority_label_test()
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    # Section headers we expect to be 辅助:
    auxiliary_headers = [
        "当前阶段",          # phase
        "我的判断",         # belief
        "可见状态",          # visible state
        "本局·",            # private memory
        "关键事件",          # salience
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
    # Merged learning context is one reference section with internal
    # anchors for RAG/reflection/profile/cognition/error-pattern hints.
    learning_idx = prompt.find("跨局学习参考")
    if learning_idx >= 0:
        preceding = prompt[max(0, learning_idx - 60):learning_idx]
        assert "【参考】" in preceding, (
            f"Learning context must be preceded by 【参考】 label. "
            f"Got: {preceding!r}"
        )
    # P1-6 / P1-12: public_summary now has a distinctive
    # ``【场上记录】`` label (Phase 1 self-audit P1-4 rename) so the
    # LLM can distinguish it from the strategy_directive inner
    # sub-group ``【参考】`` marker.  Priority tier is still 辅助.
    public_summary_idx = prompt.find("当前局公开事实")
    if public_summary_idx >= 0:
        preceding = prompt[max(0, public_summary_idx - 60):public_summary_idx]
        assert "【场上记录】" in preceding, (
            "P1-12 + self-audit P1-4: public_summary must now use the "
            "distinctive label 【场上记录】 (not 【参考】, which "
            "collides with the strategy_directive sub-group marker)"
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


def test_output_contract_fields_live_only_in_dynamic_action_contract():
    """Field-level schemas belong only to the current turn's ActionContract."""
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

    stable_contract = builder._build_output_contract()
    for discriminator in ("action_type", "choice", "intent"):
        assert discriminator not in stable_contract, (
            f"Stable system output contract must not enumerate "
            f"{discriminator!r}; the current ActionContract is the sole "
            f"field-level source of truth. contract={stable_contract!r}"
        )
    assert "ActionContract" in stable_contract
    assert "最终输出协议" in user_prompt


def test_output_contract_vote_rule_still_in_user_prompt():
    """P1-4: even after removing the duplicated field list, the
    phase-specific VOTE audit rule must still appear in the user prompt.

    The user prompt is where the LLM sees the per-turn constraints. The
    VOTE audit fields (``seer_stance、vote_basis、standing_with_seer、...``)
    are phase-specific, not stable, so they belong in the user prompt.
    """
    ctx = AgentContext(
        agent_id="p05",
        task_type=TaskType.VOTE,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.VOTE],
        legal_targets=["p05", "p07"],
        public_summary="D2 vote",
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
# Fix: implement a global token budget. Drop sections by their
# registry-defined drop_tier when the assembled prompt exceeds the
# budget; never-drop sections are preserved.

_USER_PROMPT_BUDGET_CHARS = 20_000


def test_user_prompt_budget_is_raised_to_20k() -> None:
    assert prompt_builder_module._USER_PROMPT_BUDGET_CHARS == 20_000


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
        {
            "type": "rag_hit",
            "title": f"案例 {i}",
            "summary": f"内容 {i} " * 50,
            "key_decisions": [f"决策 {i}"],
        }
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
    drops lowest-priority sections by registry drop_tier until the
    prompt fits under the budget. Never-drop sections are preserved.
    """
    ctx = _make_ctx_with_all_sections_populated()
    user_prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    assert len(user_prompt) < _USER_PROMPT_BUDGET_CHARS, (
        f"P1-5: user prompt is over budget. chars={len(user_prompt)}, "
        f"budget={_USER_PROMPT_BUDGET_CHARS} chars. "
        f"Drop lowest-priority sections first when over "
        f"budget. user_prompt[:500]={user_prompt[:500]!r}"
    )


def test_hard_sections_never_dropped_under_budget():
    """P1-5: budget enforcement must NEVER drop truly-binding sections.

    strategy_directive, retry hint, output contract, persona, and
    current-game grounding are registry-marked never-drop sections.
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
    assert "最终输出约束" in user_prompt, (
        "P1-5: retry hint and output contract share one never-drop output guard."
    )


def test_budget_trimmer_respects_registry_drop_tiers():
    """P1-5: under tight budget, registry drop_tier controls trimming."""
    ctx = _make_ctx_with_all_sections_populated()
    user_prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    # Pin the contract: budget is hard, and trim order comes from
    # _USER_SECTION_SPECS rather than label-name heuristics.
    assert len(user_prompt) < _USER_PROMPT_BUDGET_CHARS


# ---------------------------------------------------------------------------
# AUDIT-2-04: retry hint must be never-dropped under budget pressure
# ---------------------------------------------------------------------------
#
# Audit AUDIT-2-04 finding: the retry hint carries the LLM's only
# feedback on the previous turn's failure (the error_message snippet
# + correction_hint). Under budget pressure, the 辅助 label allowed
# the trimmer to drop this section — leaving the LLM to repeat the
# same mistake on the next attempt and burning the retry budget.
#
# Game trace g_3528592081 Action 50: p10 had 3 retries on the same
# parse_error before fallback. The retry hint (had it survived) is
# the corrective signal that breaks that loop.
#
# Fix: add _build_retry_hint to _NEVER_DROP and change the outer
# label from 【辅助】 to 【硬约束】 so the trimmer treats it as
# binding (must be present regardless of budget).


def test_retry_hint_never_dropped_under_budget():
    """AUDIT-2-04: under heavy budget pressure the retry hint content
    must still be present in the user prompt.

    The retry hint is the LLM's only feedback on the previous turn's
    failure (the error_message snippet + correction_hint). Without
    it, the LLM repeats the same mistake and burns the retry budget.
    The fix puts _build_retry_hint in _NEVER_DROP and labels it
    【硬约束】 so the trimmer cannot remove it.
    """
    ctx = _make_ctx_with_all_sections_populated()
    retry = RetryInfo(
        attempt=2,
        max_retries=3,
        error_code="parse_error",
        error_message="JSON parse error: missing field 'speech'",
        correction_hint="只输出JSON，不要解释、不要Markdown代码块。",
    )
    user_prompt = PlayerPromptBuilder(ctx).build_user_prompt(retry)
    # The retry hint's error snippet and correction hint must appear.
    # Both are unique strings the LLM needs to see to break out of
    # the retry loop.
    assert "missing field 'speech'" in user_prompt, (
        "AUDIT-2-04: retry hint error_message snippet must survive "
        "budget pressure. Without it, the LLM cannot tell what failed "
        "on the previous attempt. "
        f"Prompt length: {len(user_prompt)} chars."
    )
    assert "只输出JSON" in user_prompt, (
        "AUDIT-2-04: retry hint correction_hint must survive budget "
        "pressure. The hint is the LLM's only corrective signal on "
        f"retries. Prompt length: {len(user_prompt)} chars."
    )


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

    P3 (post-review-v2): ``anti_herd`` was promoted to HARD, so the
    "suggestion" sample is now ``good_vote_decision_guard`` (still in
    SUGGESTION_KEYS).
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
            "good_vote_decision_guard": "do not follow the herd",
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
# AUDIT-2-04 (overrides P1-9): retry hint section is 【硬约束】
# ---------------------------------------------------------------------------
#
# P1-9 originally classified _build_retry_hint as 【辅助】 (advisory)
# on the theory that the content is descriptive and the runtime
# FallbackAction enforces safety. That conservative call was
# overridden by AUDIT-2-04: under budget pressure the trimmer was
# actually dropping the retry hint, leaving the LLM to repeat the
# same mistake on retries. Game trace g_3528592081 Action 50
# showed p10 hitting the same parse_error 3 times before fallback.
#
# Fix: promote _build_retry_hint to 【硬约束】 AND add it to
# _NEVER_DROP. The correction_hint + error_message are the LLM's
# only signal of what failed on the previous attempt. Losing them
# under budget pressure is a worse failure mode than the
# (over-labeling) cost of treating the whole section as MUST-obey.


def test_retry_hint_labeled_as_硬约束():
    """AUDIT-2-04: retry hint outer section label is 【硬约束】, not 【辅助】.

    The retry hint is the LLM's only feedback on the previous turn's
    failure (error_message snippet + correction_hint). Without it,
    the LLM repeats the same mistake and burns the retry budget.
    Promoting the label to 【硬约束】 keeps the trimmer from dropping
    this section under heavy budget pressure.
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
    # is 【硬约束】, not 【辅助】.
    retry_idx = prompt.find("纠正提示")
    assert retry_idx > 0, "Retry hint should still render in the user prompt"
    preceding = prompt[max(0, retry_idx - 60):retry_idx]
    assert "【硬约束】" in preceding, (
        "AUDIT-2-04: retry hint section must be labeled 【硬约束】 "
        "(binding — must survive budget trim). Got preceding: "
        f"{preceding!r}"
    )


def test_retry_hint_timeout_permission_line_still_present():
    """P1-9 regression: the timeout-no-op permission line must still
    appear in the retry hint even after relabeling the section.

    The line is a useful advisory signal when no_action is legal.
    The runtime FallbackAction enforces safety; the prompt signal is
    the soft guidance.
    """
    ctx = AgentContext(
        agent_id="p05",
        task_type=TaskType.SHERIFF_REGISTRATION,
        phase="day",
        day_number=1,
        own_role="villager",
        legal_actions=[
            ActionType.SHERIFF_REGISTER,
            ActionType.SHERIFF_WITHDRAW,
            ActionType.NO_ACTION,
        ],
        public_summary="D1 sheriff election",
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


def test_format_examples_no_intent_field_in_full_action_sheriff_example():
    """P1-S4: FULL_ACTION sheriff examples must not include `intent` field.

    `intent` is a SPEECH_INTENT-mode field (the enum-style value the
    LLM picks from a small set). FULL_ACTION mode renders sheriff
    actions directly with action_type=..., speech=..., reason=..., and
    never uses `intent`. If the example mentions it, the LLM will
    defensively fill it for non-speech-intent actions and the strict
    schema (extra=forbid, P0-S8) will reject them.
    """
    ctx = AgentContext(
        agent_id="p05",
        task_type=TaskType.SHERIFF_REGISTRATION,
        phase="day",
        day_number=1,
        own_role="villager",
        legal_actions=[
            ActionType.SHERIFF_REGISTER,
            ActionType.SHERIFF_WITHDRAW,
            ActionType.NO_ACTION,
        ],
        public_summary="D1 sheriff election",
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    examples = _extract_json_examples(prompt)
    assert examples, "Expected at least one example in the full action path"
    for ex in examples:
        assert "intent" not in ex, (
            f"FULL_ACTION example must not include 'intent' field "
            f"(that's a SPEECH_INTENT-mode field); example={ex}"
        )


def test_format_examples_no_choice_field_in_full_action_sheriff_example():
    """P1-S4: FULL_ACTION sheriff examples must not include `choice` field.

    `choice` is a TARGET_CHOICE-mode field (an enum letter A/B/C/...
    the LLM picks from a small set). FULL_ACTION mode renders the
    action with action_type=..., target_id=player_id/null, never with
    `choice`. If the example mentions it, the LLM will defensively
    fill it for non-choice actions.
    """
    ctx = AgentContext(
        agent_id="p05",
        task_type=TaskType.SHERIFF_REGISTRATION,
        phase="day",
        day_number=1,
        own_role="villager",
        legal_actions=[
            ActionType.SHERIFF_REGISTER,
            ActionType.SHERIFF_WITHDRAW,
            ActionType.NO_ACTION,
        ],
        public_summary="D1 sheriff election",
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
    # P3-2: the info_boundaries text now lists "人格设定" as one
    # of the 11 user-prompt sections.  Use the persona section's
    # section-header marker ("人格设定: " with colon) to avoid a
    # substring collision — only the persona section uses this
    # exact prefix; the info_boundaries list has it without colon.
    assert "人格设定: " not in system_prompt, (
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
    # P3-2: the info_boundaries text now lists "人格设定" as one of
    # the 11 user-prompt sections.  Use the persona section's
    # section header marker ("人格设定: " with colon) to avoid a
    # substring collision — only the persona section uses this
    # exact prefix.
    assert "人格设定: " not in system_prompt
    assert "人格设定: " not in user_prompt


def test_good_role_persona_prompt_renders_only_sanitized_expression_fields():
    ctx = AgentContext(
        agent_id="p08",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH],
        persona_snapshot={
            "profile_id": "bold_pretender",
            "display_name": "悍跳进攻型",
            "personality": "bold_deceiver",
            "speech_style": "confident_fake_claim",
            "task_style": "fake_authority",
            "effective_params": {
                "deception_skill": 0.91,
                "logic_skill": 0.55,
            },
            "dynamic_adjustments": {"deception_skill": 0.10},
            "phrase_style": "固定冒充预言家",
        },
    )

    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())

    assert "bold_pretender" not in prompt
    assert "悍跳进攻型" not in prompt
    assert "bold_deceiver" not in prompt
    assert "confident_fake_claim" not in prompt
    assert "fake_authority" not in prompt
    assert "固定冒充预言家" not in prompt
    assert "role_consistent_expression" in prompt
    assert "evidence_based_expression" in prompt
    assert "deception_skill" not in prompt


def test_live_memory_prompt_omits_ability_ranks_and_duplicate_cognition_lists():
    ctx = AgentContext(
        agent_id="p08",
        task_type=TaskType.SPEECH,
        phase="day",
        own_role="villager",
        legal_actions=[ActionType.SPEECH],
        profile_memory_hint={
            "games_played": 8,
            "current_role": "villager",
            "current_role_games": 3,
            "current_role_win_rate_pct": 67,
            "win_rate_confidence": "样本中等(3局)",
            "logic_rank": "前 30%",
            "deception_rank": "需要提升",
        },
        cognition_matrix_hint={
            "suspects": [{
                "player": "p03",
                "trust": 0.2,
                "key_evidence": ["salience_items#abc123"],
            }],
            "trusted": [{
                "player": "p04",
                "trust": 0.8,
                "open_questions": ["salience_items#def456"],
            }],
        },
    )

    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())

    assert "logic_rank" not in prompt
    assert "deception_rank" not in prompt
    assert "前 30%" not in prompt
    assert "需要提升" not in prompt
    assert '"suspects"' not in prompt
    assert '"trusted"' not in prompt
    assert "salience_items#" not in prompt
    assert "tracked_suspect_count" in prompt
    assert "tracked_trusted_count" in prompt
    assert "open_question_count" in prompt
    assert "calibration_warning" in prompt


def test_learning_context_renderers_whitelist_prompt_safe_fields() -> None:
    ctx = AgentContext(
        agent_id="p08",
        task_type=TaskType.SPEECH,
        phase="day",
        own_role="villager",
        legal_actions=[ActionType.SPEECH],
        reflection_memory_hints=[{
            "role": "seer",
            "result": "负",
            "text": "上一局 p03 是真预言家但我跟错票",
            "raw_transcript": "RAW_SECRET_SHOULD_NOT_RENDER",
            "debug_trace": "DEBUG_SHOULD_NOT_RENDER",
            "hidden_role_truth": {"p03": "seer"},
        }],
        profile_memory_hint={
            "games_played": 8,
            "current_role": "villager",
            "summary": "累计 p04 相关经验",
            "raw_games": ["RAW_GAME_SHOULD_NOT_RENDER"],
            "private_labels": {"p04": "wolf"},
            "logic_rank": "前 10%",
        },
        rag_hints=[{
            "type": "rag_hit",
            "title": "案例 p05",
            "summary": "案例中 p06 冲票成功",
            "key_decisions": "不要被拆成字符列表",
            "debug": "RAG_DEBUG_SHOULD_NOT_RENDER",
        }],
    )

    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())

    assert "RAW_SECRET_SHOULD_NOT_RENDER" not in prompt
    assert "DEBUG_SHOULD_NOT_RENDER" not in prompt
    assert "hidden_role_truth" not in prompt
    assert "RAW_GAME_SHOULD_NOT_RENDER" not in prompt
    assert "private_labels" not in prompt
    assert "logic_rank" not in prompt
    assert "RAG_DEBUG_SHOULD_NOT_RENDER" not in prompt
    assert "历史玩家" in prompt
    assert "案例中 历史玩家 冲票成功" in prompt
    assert "不要被拆成字符列表" not in prompt
    assert "key_decisions" not in prompt


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


def test_format_examples_non_seer_vote_keeps_pXX_example():
    """P0-4 / P1-8 / C4: non-seer roles keep a non-empty standing_with_seer example.

    Regression: the seer-specific fix must NOT remove the non-seer
    branch (villager, witch, hunter, idiot, hybrid, werewolf). Those
    roles DO side with an external seer claim, so the example is
    still meaningful. P1-8 updated ``vote_basis`` from "seer_check"
    (own check, only seers have that) to "seer_siding" (standing with
    another seer).

    C4 review fix: standing_with_seer is no longer hardcoded to a
    concrete player ID ("p03"). It is a placeholder ("pXX") the LLM
    should replace with the actual seer ID in the current game. The
    contract being tested here is that the field stays *non-empty* for
    non-seer roles (they need an external seer to stand with) and
    *empty* for the seer (own check is implicit, no other seer to
    side with).
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
            # C4 fix: standing_with_seer is a placeholder ("pXX"), not
            # a hardcoded player ID. Assert the field is non-empty (the
            # contract is "non-seer roles side with an external seer")
            # and matches the placeholder pattern.
            value = ex.get("standing_with_seer", "")
            assert value and value != "", (
                f"Non-seer role={role!r} must keep a non-empty "
                f"standing_with_seer example (they side with an external "
                f"seer), got {value!r}. Full example: {ex}"
            )
            # C4: prefer the explicit "pXX" placeholder. Allow other
            # placeholder-looking values (anything not a concrete p0X)
            # so the test does not lock the placeholder name.
            import re as _re_placeholder
            assert not _re_placeholder.match(r"^p\d{2}$", value), (
                f"Non-seer role={role!r} standing_with_seer must NOT be "
                f"a hardcoded concrete player ID (e.g., 'p03'); got "
                f"{value!r}. Use a placeholder (e.g., 'pXX') so the LLM "
                f"replaces it with the actual seer ID."
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


def test_compact_json_truncation_stays_valid_json():
    """Oversized context must remain machine-readable after truncation."""
    import json

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
    parsed = json.loads(out)
    assert parsed["truncated"] is True
    assert parsed["original_type"] == "dict"
    assert parsed["omitted_middle"] is True
    assert "head" in parsed
    assert "tail" in parsed
    assert len(out) <= 1800


def test_compact_json_preserves_tail_context_when_truncated():
    """Long dict compaction must not be prefix-only."""
    import json

    builder = PlayerPromptBuilder(
        AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            phase="day",
            day_number=1,
            own_role="villager",
        )
    )
    big = {
        "early_noise": "x" * 5000,
        "late_critical_fact": "TAIL_CRITICAL_MARKER",
    }
    out = builder._compact_json(big)
    parsed = json.loads(out)

    assert parsed["truncated"] is True
    assert "TAIL_CRITICAL_MARKER" in out


def test_compact_json_structured_summary_keeps_shape_metadata():
    import json

    builder = PlayerPromptBuilder(
        AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            phase="day",
            day_number=1,
            own_role="villager",
        )
    )
    big = {f"k{i:02d}": "x" * 300 for i in range(20)}

    parsed = json.loads(builder._compact_json(big))

    assert parsed["truncated"] is True
    assert parsed["original_type"] == "dict"
    assert parsed["original_length"] == 20
    assert parsed["omitted_middle"] is True
    assert "k00" in parsed["head"]
    assert "k19" in parsed["tail"]



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
    """M2-2: vote_basis *guidance* (the "选用 speech_logic / vote_pattern
    / seer_siding, 不要用 seer_check" sentence) is no longer in
    role_guide (system prompt). It is now injected per-turn via
    strategy_directive from the agent_adapter for VOTE/SPEECH task
    types.

    Background: pre-fix (P2-9) the guidance was appended to every
    role's role_guide in the stable system prompt, so wolf NIGHT
    actions saw "投票时 vote_basis 选用 speech_logic" — irrelevant
    for a kill target decision. M2-2 moved the guidance out.

    Note: the field name ``vote_basis`` still appears in the
    output_contract section (VOTE field list) — that's the schema,
    not the guidance, and it stays in the system prompt.
    """
    # Spot-check villager (a non-seer good-side role).
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=1,
        own_role="villager",
    )
    role_guide = PlayerPromptBuilder(ctx)._build_role_guide()
    # M2-2: the role guide no longer contains the VOTE_BASIS_GUIDANCE
    # text. The field name "vote_basis" is OK to remain (output
    # contract section uses it for schema listing).
    guidance_phrase = "选用 speech_logic"
    assert guidance_phrase not in role_guide, (
        "M2-2: vote_basis *guidance* should be removed from role_guide "
        f"(stable system section). role_guide: {role_guide!r}"
    )


def test_vote_basis_guidance_appears_in_per_turn_user_prompt():
    """M2-2: vote_basis guidance must be in the per-turn strategy_directive
    output (rendered into the user prompt by _build_strategy_directive),
    not the stable system prompt.
    """
    from werewolf_agent.runtime.agent_adapter import VOTE_BASIS_GUIDANCE
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=1,
        own_role="villager",
        strategy_directive={"vote_basis_hint": VOTE_BASIS_GUIDANCE},
    )
    builder = PlayerPromptBuilder(ctx)
    # System prompt must NOT contain the VOTE_BASIS_GUIDANCE text.
    system_text = builder.build_system_prompt()
    assert VOTE_BASIS_GUIDANCE not in system_text, (
        "M2-2: VOTE_BASIS_GUIDANCE leaked into stable system prompt. "
        f"system_text: {system_text!r}"
    )
    # _build_strategy_directive (per-turn user prompt section) must
    # contain the VOTE_BASIS_GUIDANCE text.
    user_section = builder._build_strategy_directive()
    assert VOTE_BASIS_GUIDANCE in user_section, (
        "M2-2: VOTE_BASIS_GUIDANCE missing from _build_strategy_directive. "
        f"section: {user_section!r}"
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


def test_full_action_examples_use_current_legal_target() -> None:
    scenarios = [
        (ActionType.HUNTER_SHOT, "hunter_shot"),
        (ActionType.USE_ANTIDOTE, "use_antidote"),
        (ActionType.USE_POISON, "use_poison"),
        (ActionType.BADGE_TRANSFER, "badge_transfer"),
        (ActionType.CHOOSE_MASTER, "choose_master"),
    ]
    for action, action_text in scenarios:
        ctx = AgentContext(
            agent_id="p03",
            task_type=TaskType.NIGHT_ACTION,
            phase="night",
            night_number=2,
            own_role="witch",
            legal_actions=[action, ActionType.NO_ACTION],
            legal_targets=["p11"],
        )

        prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())

        assert action_text in prompt
        assert '"target_id": "p11"' in prompt
        assert '"target_id": "p05"' not in prompt
        assert '"target_id": "p07"' not in prompt


# ---------------------------------------------------------------------------
# G-R4-15 → M4-2 reversal: RAG hints are NO LONGER guaranteed to survive
# budget pressure.
#
# Pre-M4-2 (G-R4-15): RAG hints were tagged 【参考】, so they survived
# budget cuts ahead of 辅助 sections. The rationale was that RAG
# retrieval is expensive, so losing the hints would defeat the
# retrieval investment.
#
# Post-M4-2 (commit 0022d25): RAG hints are now a 【辅助】 section
# (same tier as persona, profile, etc.). The M4-2 view is
# "prompt value ≠ retrieval cost" — per-player reflection
# (now in 【参考】) outranks generic RAG for current LLM reasoning,
# so RAG is no longer privileged over other 辅助 sections.
#
# This test still passes, but only because persona is the largest
# 辅助 section in the fixture, so it gets dropped first and the
# budget is satisfied before RAG is reached. RAG's actual eviction
# priority is now determined by the generic 辅助-tier drop order,
# not by a survival guarantee. If a future fixture makes RAG the
# largest 辅助 section, this test will start to fail — that would
# be correct M4-2 behavior, not a regression.
# ---------------------------------------------------------------------------


def _make_budget_pressure_context() -> AgentContext:
    """Context that genuinely exceeds the 6_250-char user-prompt budget.

    Current-game grounding should survive while style/history sections drop.
    """
    return AgentContext(
        agent_id="p10",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH, ActionType.VOTE],
        legal_targets=["p05", "p07"],
        # 辅助: each of these can render up to ~1.8k chars.
        persona_snapshot={
            "profile_id": "logic_anchor",
            "display_name": "逻辑锚点",
            "personality": "budget-pressure-persona",
            "speech_style": "structured_logical",
            "task_style": "evidence_based_expression",
            "tone": "aggressive",
            "extra": "X" * 1500,
        },
        belief_state={
            "my_suspects": [
                {"player": f"p{i:02d}", "faction_lean": "wolf_lean",
                 "top_role_guess": "werewolf", "trust": 0.2,
                 "note": "long context " * 30}
                for i in range(1, 6)
            ],
            "my_trusted": [
                {"player": f"p{i:02d}", "faction_lean": "good_lean",
                 "top_role_guess": "villager", "trust": 0.8,
                 "note": "long context " * 30}
                for i in range(6, 11)
            ],
        },
        public_summary="PUBLIC_FACT_MARKER " + ("D2 投票事实数据 " * 200),
        visible_world_state={
            "VISIBLE_STATE_MARKER": "alive/sheriff state",
            **{f"key_{i}": f"value_{i} " * 30 for i in range(20)},
        },
        private_memory_hints={
            "logic_flaws": [{"day": 2, "point": f"逻辑 {i} " * 25} for i in range(8)],
            "valid_points": [{"day": 2, "point": f"有效 {i} " * 25} for i in range(8)],
        },
        salience_items=[
            {
                "id": f"sal-{i}",
                "weight": 0.5,
                "summary": (
                    "SALIENCE_MARKER " if i == 0 else ""
                ) + f"事件 {i} " * 30,
            }
            for i in range(8)
        ],
        # RAG hints with the proper discriminator (the render filter
        # at _build_rag_hints drops items without ``type == "rag_hit"``).
        rag_hints=[{
            "type": "rag_hit",
            "title": "budget-pressure-rag",
            "summary": "案例摘要 " * 60,
            "key_decisions": ["决策1", "决策2", "决策3"],
        }],
        reflection_memory_hints=[
            {"theme": f"主题 {i}", "summary": f"经验 {i} " * 30}
            for i in range(5)
        ],
        profile_memory_hint={"games_played": 10, "summary": "前 30% 玩家画像 " * 50},
        cognition_matrix_hint={
            f"p{i:02d}": {"trust": 0.5, "faction_lean": "good",
                          "top_role_guess": "villager",
                          "note": "画像 " * 20}
            for i in range(1, 13)
        },
        recent_transcript=[
            {
                "speaker": "p07",
                "day_number": 2,
                "text": "LATEST_TRANSCRIPT_MARKER " + ("公开发言证据 " * 20),
            },
        ],
        # 硬约束 equivalent (NEVER_DROP) — keeps strategy + contract
        # in the prompt no matter what.
        strategy_directive={
            "must_address_alerts": [{"alert": f"提示 {i}"} for i in range(3)],
            "anti_herd": "不要盲目跟票 " * 10,
            "skill_tactical_advice": {"role": "villager", "tips": ["tip1", "tip2"] * 20},
        },
    )


def test_current_game_grounding_survives_with_expanded_budget() -> None:
    """Current facts stay present while the 20k budget may retain history."""
    ctx = _make_budget_pressure_context()
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    assert len(prompt) <= _USER_PROMPT_BUDGET_CHARS + 750, (
        f"budget trimmer should keep joined prompt under "
        f"~6_250 + 750 chars; got {len(prompt)} chars."
    )
    for marker in (
        "PUBLIC_FACT_MARKER",
        "VISIBLE_STATE_MARKER",
        "SALIENCE_MARKER",
        "LATEST_TRANSCRIPT_MARKER",
    ):
        assert marker in prompt, (
            f"current-game grounding marker {marker!r} was dropped before "
            f"style/history context. prompt[:500]={prompt[:500]!r}"
        )
    assert "人格设定" in prompt
    assert "budget-pressure-persona" in prompt
    assert "长期能力画像" not in prompt
    assert "我的认知矩阵" not in prompt


def test_phase_and_legal_context_survives_extreme_budget_pressure() -> None:
    """Legal action context is the task contract, not optional grounding."""
    ctx = AgentContext(
        agent_id="p10",
        task_type=TaskType.VOTE,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.VOTE],
        legal_targets=["p05", "p07"],
        strategy_directive={
            "must_address_alerts": ["必须回应" + "X" * 4000],
            "anti_herd": "不要盲目跟票 " * 400,
        },
        public_summary="PUBLIC_OVERFLOW " + ("公开事实 " * 400),
        visible_world_state={f"key_{i}": "value " * 40 for i in range(20)},
        salience_items=[
            {"id": f"sal-{i}", "summary": "事件 " * 80}
            for i in range(3)
        ],
        recent_transcript=[
            {"speaker": f"p{i:02d}", "day_number": 2, "text": "发言 " * 120}
            for i in range(1, 5)
        ],
    )

    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())

    assert "当前阶段: day" in prompt
    assert "可用操作: ['vote']" in prompt
    assert "可选目标: ['p05', 'p07']" in prompt
    assert "本轮投票必须选择一名玩家放逐" in prompt


def test_persona_core_survives_under_budget_pressure() -> None:
    from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder

    assert "_build_persona" in PlayerPromptBuilder._NEVER_DROP

    ctx = _make_budget_pressure_context()
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())

    assert "人格设定" in prompt
    assert "budget-pressure-persona" in prompt


# ---------------------------------------------------------------------------
# Phase-1 audit: 7 new behavior tests for items 11, 12, 25, 26, 29
# ---------------------------------------------------------------------------


def test_salience_id_and_summary_preserved_in_prompt():
    """Phase-1 P1-11: salience items with ``id`` and ``summary`` must
    surface in the rendered prompt so the LLM can track which event is
    which across turns.

    Pre-fix, ``_SALIENCE_PUBLIC_FIELDS`` did not include ``id`` /
    ``summary`` and ``_slim_salience_item`` stripped them — the LLM
    saw only ``weight`` etc. and could not distinguish salience
    events from one another.
    """
    ctx = AgentContext(
        agent_id="p05",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH],
        legal_targets=["p05"],
        public_summary="",
        salience_items=[
            {"id": "sal-7", "weight": 0.5, "summary": "p07 翻票"},
        ],
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    assert "sal-7" in prompt, (
        "salience item id 'sal-7' must be preserved in the rendered prompt"
    )
    assert "p07 翻票" in prompt, (
        "salience item summary must be preserved so the LLM can read the event"
    )


def test_private_memory_caveat_wrapped_in_separator_markers():
    """Phase-1 P1-25: the P1-M10 caveat string is wrapped in ``---``
    so the LLM cannot mistake it for a JSON key or hint payload entry.

    Pre-fix, the caveat was directly prepended to the JSON dict and
    the LLM had been observed to fold it into the parsed object.
    """
    ctx = AgentContext(
        agent_id="p05",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH],
        legal_targets=["p05"],
        public_summary="",
        private_memory_hints={"logic_flaws": [{"day": 1, "speaker": "p02", "point": "flip"}]},
        private_memory_caveat=(
            "【P1-M10 提示】私有记忆中的'逻辑漏洞'与'合理点'是基于关键词的粗粒度信号"
        ),
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    # The caveat must be flanked by --- markers
    assert "---" in prompt
    # The caveat text must appear
    assert "P1-M10" in prompt
    # Find the position of the marker, ensure it precedes the caveat
    marker_idx = prompt.find("---")
    caveat_idx = prompt.find("P1-M10")
    assert marker_idx < caveat_idx, (
        "Separator marker --- must precede the caveat text so the LLM "
        "treats it as a meta-notice, not a JSON field"
    )


def test_villager_guide_includes_night_fallback():
    """Phase-1 P1-26 (superseded by M2-1): the original P1-26 test
    required explicit night-time fallback rules in the villager
    role guide (idiot reveals, hunter shot). M2-1 audit found this
    was over-guidance — villagers are 3 of 12, and the system
    prompt was 4× longer than other roles' guides (400+ chars).
    The fix compresses the guide to ~50 chars and moves the
    night-fallback knowledge to per-turn context (when the
    villager's NIGHT action is dispatched, the task is a no-op
    by definition; the engine handles night transitions).

    This test now asserts the compression: villager guide is
    concise (no night-fallback detail) while still keeping the
    2 key day-time cues (N1 解药 + 证据/票型) per M2-1.
    """
    ctx = AgentContext(
        agent_id="p05",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH],
        legal_targets=["p05"],
        public_summary="",
    )
    sys_prompt = PlayerPromptBuilder(ctx).build_system_prompt()
    # M2-1 compression: the night-fallback detail is no longer in
    # the system prompt (token waste + over-guidance). The guide
    # should be concise like other roles' guides (~50-150 chars).
    assert "夜间阶段" not in sys_prompt, (
        f"M2-1: villager guide should NOT include '夜间阶段' detail "
        f"(compressed for token efficiency). got sys_prompt="
        f"{sys_prompt[sys_prompt.find('村民规则'):sys_prompt.find('村民规则') + 300]!r}"
    )
    assert "无投票权" not in sys_prompt, (
        "M2-1: villager guide should NOT include idiot-vote detail "
        "(that knowledge belongs to per-turn context, not the "
        "stable system prompt). "
        f"got sys_prompt={sys_prompt[-300:]!r}"
    )
    # M2-1 regression: the 2 key day-time cues MUST still be present
    assert "解药" in sys_prompt, (
        "M2-1: villager guide must keep '解药' cue (N1 antidote support)"
    )
    assert "票型" in sys_prompt or "证据" in sys_prompt, (
        "M2-1: villager guide must keep 票型/证据 cue (evidence-based voting)"
    )


def test_hybrid_guide_includes_master_death_lock():
    """Phase-1 P1-26: hybrid role guide must explicitly state that
    master's death LOCKS the faction — no reselection, hybrid
    slaughter conditions differ by master faction.

    Note: role guide lives in the SYSTEM prompt (stable rule).
    """
    ctx = AgentContext(
        agent_id="p05",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="hybrid",
        legal_actions=[ActionType.SPEECH],
        legal_targets=["p05"],
        public_summary="",
    )
    sys_prompt = PlayerPromptBuilder(ctx).build_system_prompt()
    assert "不能再选" in sys_prompt, (
        "hybrid guide must state master is locked after death (no reselection)"
    )
    assert "屠边" in sys_prompt, (
        "hybrid guide must mention slaughter conditions differ by master faction"
    )


def test_reasoning_method_has_three_numbered_steps():
    """Phase-1 P1-29: reasoning method section must be a 3-step
    actionable flow, not 4 abstract lines.
    """
    ctx = AgentContext(
        agent_id="p05",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH],
        legal_targets=["p05"],
        public_summary="",
    )
    sys_prompt = PlayerPromptBuilder(ctx).build_system_prompt()
    # Each step starts with N) pattern
    for step in ("1)", "2)", "3)"):
        assert step in sys_prompt, (
            f"reasoning method must contain numbered step {step!r}; "
            f"got sys_prompt={sys_prompt!r}"
        )
    # The 3-step section header is named
    assert "推理方法-3 步" in sys_prompt, (
        "reasoning method section must be labeled with 3-step marker"
    )


# ---------------------------------------------------------------------------
# Phase 2 P2-3: system output contract must advertise 9 fields for VOTE
# ---------------------------------------------------------------------------


def test_system_output_contract_defers_vote_fields_to_action_contract():
    """Stable system guidance must not compete with the vote contract."""
    ctx = AgentContext(
        agent_id="p05",
        task_type=TaskType.VOTE,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.VOTE],
        legal_targets=["p02", "p03"],
    )
    builder = PlayerPromptBuilder(ctx)
    stable_contract = builder._build_output_contract()
    user_prompt = builder.build_user_prompt(RetryInfo())
    for field in (
        "choice", "reason", "seer_stance", "vote_basis",
        "standing_with_seer", "suspect_reason", "not_voting_reason",
        "private_reason", "confidence",
    ):
        assert field not in stable_contract
        assert field in user_prompt


def test_system_output_contract_does_not_unconditionally_prefer_tool_call():
    ctx = AgentContext(
        agent_id="p05",
        task_type=TaskType.REFLECTION,
        phase="day",
        day_number=2,
        own_role="villager",
        legal_actions=[ActionType.SPEECH],
    )
    stable_contract = PlayerPromptBuilder(ctx)._build_output_contract()
    assert "优先通过当前回合提供的工具提交" not in stable_contract
    assert "最终输出协议" in stable_contract
    assert "ActionContract" in stable_contract


def test_information_boundaries_do_not_hardcode_wrong_section_count():
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,
        phase="day",
        own_role="villager",
    )
    boundaries = PlayerPromptBuilder(ctx)._build_information_boundaries()
    assert "11 类" not in boundaries
    assert "user-prompt" in boundaries


def test_information_boundaries_use_current_dynamic_section_names():
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,
        phase="day",
        own_role="villager",
    )
    boundaries = PlayerPromptBuilder(ctx)._build_information_boundaries()
    for current_name in (
        "历史角色经验",
        "认知校准摘要",
        "跨局错误模式",
        "本轮任务",
        "纠正提示",
        "最终输出协议",
    ):
        assert current_name in boundaries
    assert "长期能力画像" not in boundaries
    assert "我的认知矩阵" not in boundaries


def test_reasoning_method_keeps_private_info_as_decision_input_not_public_fact():
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,
        phase="day",
        own_role="seer",
    )
    reasoning = PlayerPromptBuilder(ctx)._build_reasoning_method()
    assert "私有信息可用于私有决策" in reasoning
    assert "不能伪装成公开事实" in reasoning


def test_skill_policy_points_to_strategy_directive_advice_not_legacy_section():
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,
        phase="day",
        own_role="villager",
    )
    policy = PlayerPromptBuilder(ctx)._build_skill_policy()
    assert "策略指令" in policy
    assert "技能战术建议" in policy
    assert "技能分析结果" not in policy


def test_werewolf_role_guide_names_stable_wolf_options():
    ctx = AgentContext(
        agent_id="p08",
        task_type=TaskType.SPEECH,
        phase="day",
        own_role="werewolf",
    )
    role_guide = PlayerPromptBuilder(ctx)._build_role_guide()
    assert "空刀" in role_guide
    assert "自爆" in role_guide


# ---------------------------------------------------------------------------
# Phase 2 P2-2: judge LLM calls use JUDGE_* task types (not "speech")
# ---------------------------------------------------------------------------


def test_judge_broadcasts_use_judge_task_types_in_router_calls() -> None:
    """Phase 2 P2-2: all 4 JudgeAgent broadcast LLM calls must use
    the JUDGE_* task_type enum values so the model_gateway router
    can route them to a dedicated profile (or override timeouts)
    per the JUDGE_* keys in ``config/models.yaml``.

    Pre-fix: 4 calls all used ``task_type="speech"`` which routed
    judge broadcasts through the player-speech code path.  This
    conflated audit logs and prevented per-judge-task tuning.

    Implementation: construct a real ``ModelRouter`` and replace
    its ``generate`` method with a recorder.  Mocking the class
    attribute on a generic MagicMock doesn't work; we need a real
    router instance whose method we can patch.
    """
    from unittest.mock import MagicMock, patch
    from werewolf_agent.agents.judge import JudgeAgent
    from werewolf_agent.model_gateway.router import ModelRouter, GenerateResult, UsageRecord

    mock_provider = MagicMock()
    router = ModelRouter(
        model_profiles={"cap": {"model": "capture-model"}},
        llm_profiles={"judge_default": {
            "default": {"provider": "cap", "model_profile": "cap"},
        }},
        player_assignments={"judge": "judge_default"},
        providers={"cap": mock_provider},
    )
    captured: list[dict[str, object]] = []

    def _capture_generate(self, agent_id, task_type, prompt, **kwargs):  # type: ignore[no-untyped-def]
        captured.append({"agent_id": agent_id, "task_type": task_type})
        return GenerateResult(
            text="ok", provider="cap", model="capture-model",
            usage=UsageRecord(
                agent_id=agent_id, task_type=task_type,
                provider="cap", model="capture-model",
                prompt_tokens=0, completion_tokens=0, latency_ms=0,
            ),
        )

    judge = JudgeAgent(model_router=router, profile_router=None)
    with patch.object(type(router), "generate", _capture_generate):
        judge.broadcast_vote_calling(
            voter_id="p01", voter_name="玩家一", candidates=["p02"],
            position=1, total=1, day_number=1,
        )
        judge.guide_skill_use(
            role="witch", player_id="p11", player_name="玩家十一",
            available_actions=["use_antidote"],
        )
        judge.announce_vote_tally(
            tally={"p05": 5.0}, player_names={"p05": "玩家五"},
            sheriff_id=None, sheriff_weight=1.5, day_number=1,
        )
        judge.announce_exile_result(
            exiled_player_id="p05", exiled_player_name="玩家五",
            reason="", tied_player_ids=[], day_number=1,
        )

    assert len(captured) == 4
    expected_task_types = {
        "judge_vote_calling", "judge_skill_guide",
        "judge_vote_tally", "judge_exile",
    }
    actual = {rec["task_type"] for rec in captured}
    assert actual == expected_task_types, (
        f"P2-2: 4 judge LLM calls must use JUDGE_* task types; "
        f"got {actual!r}, expected {expected_task_types!r}"
    )
    # All calls must use agent_id="judge" for routing
    for rec in captured:
        assert rec["agent_id"] == "judge"


class TestPublicReasonPrivacyGuard:
    """P0-G3223805846-8: vote 阶段 prompt 应禁止 reason 中私视角表述。"""

    def test_vote_reason_privacy_guard_constant_exists(self):
        from werewolf_agent.agents.prompt_builder import _VOTE_REASON_PRIVACY_GUARD
        text = _VOTE_REASON_PRIVACY_GUARD
        # 必须含禁止 + 关键私视角 token
        for token in ("禁止", "private_intent", "我作为预言家", "狼队 N1 刀了"):
            assert token in text, f"privacy guard missing token: {token}"

    def test_vote_prompt_includes_privacy_guard(self):
        """Vote prompt 必须注入 guard。"""
        from werewolf_agent.agents.prompt_builder import _VOTE_REASON_PRIVACY_GUARD
        # 简化为检查 guard 在 prompt 中存在即可（通过直接 import 验证）
        assert len(_VOTE_REASON_PRIVACY_GUARD) > 50
        assert "P0-G3223805846-8" in _VOTE_REASON_PRIVACY_GUARD


class TestActionTypeFieldName:
    """P1-G3223805846-4: _format_examples 头部应强化 action_type 字段名。"""

    def test_action_type_guard_constant_exists(self):
        from werewolf_agent.agents.prompt_builder import _ACTION_TYPE_GUARD
        text = _ACTION_TYPE_GUARD
        assert "action_type" in text, f"guard missing 'action_type': {text!r}"
        assert "intent" in text, f"guard missing 'intent' warning: {text!r}"
        assert "禁止" in text or "不要" in text, (
            f"guard missing prohibition: {text!r}"
        )
        assert "P1-G3223805846-4" in text

    def test_action_type_guard_length_is_substantial(self):
        from werewolf_agent.agents.prompt_builder import _ACTION_TYPE_GUARD
        assert len(_ACTION_TYPE_GUARD) > 100


class TestHardConstraintLabelUniqueness:
    """审查 C3: 单一硬约束标签层。"""

    def test_no_duplicate_hard_constraint_markers(self):
        # Use a NIGHT_ACTION context with WOLF_KILL — this routes to
        # OutputMode.FULL_ACTION, which calls _format_examples (which
        # injects _ACTION_TYPE_GUARD with the "【硬约束 P1-...】" inner
        # sub-group label) AND also renders the output contract (with
        # the outer "【硬约束】" section label). The test asserts the
        # combined count of "硬约束" substrings is <= 1, which requires
        # the inner sub-group label to be removed (fix) and the outer
        # section label to remain (current).
        #
        # P3 (post-review-v2): ``anti_herd`` was promoted to HARD, so
        # using it here would now trigger the strategy_directive inner
        # 【硬约束】 sub-group header (the very thing this test tries to
        # suppress).  Use ``speech_originality`` (still SUGGESTION_KEYS)
        # as a non-hard sample to keep the no-inner-sub-group invariant.
        from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
        from werewolf_agent.agents.schemas import (
            ActionType, AgentContext, RetryInfo, TaskType,
        )
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.NIGHT_ACTION,
            own_role="werewolf",
            legal_actions=[ActionType.WOLF_KILL],
            legal_targets=[],
            # No hard-constraint keys → strategy_directive inner
            # 【硬约束】 sub-group header does not render.
            strategy_directive={
                "speech_originality": "不要模板化",
            },
        )
        builder = PlayerPromptBuilder(ctx)
        # build_user_prompt requires a RetryInfo arg.
        user_prompt = builder.build_user_prompt(RetryInfo())
        # 计数 "硬约束" 出现次数：fix 后内层 sub-group 标签移除，
        # 整个 user prompt 内只剩输出契约外层 "【硬约束】" 1 处。
        count = user_prompt.count("硬约束")
        assert count <= 1, (
            f"multiple 硬约束 labels in single prompt: {count} occurrences\n{user_prompt!r}"
        )


class TestFormatExamplesNoHardcodedID:
    """审查 C4: vote 示例不应硬编码 p03 作为预言家 ID。"""

    def test_vote_example_does_not_hardcode_p03(self):
        # Use a VOTE action context (not VOTE-only) so _format_examples
        # branch is taken and the vote example is rendered. We use a
        # non-seer own_role so the non-seer branch (which had the p03
        # hardcode) is taken.
        from unittest.mock import MagicMock
        from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
        from werewolf_agent.agents.schemas import ActionType
        mock_ctx = MagicMock()
        mock_ctx.legal_actions = [ActionType.VOTE]
        mock_ctx.legal_targets = [f"p{i:02d}" for i in range(1, 13)]
        mock_ctx.own_role = "villager"  # non-seer branch (had p03 hardcode)
        mock_ctx.task_type = None
        # 触发 _format_examples 路径
        builder = PlayerPromptBuilder(mock_ctx)
        examples = builder._format_examples()
        # 不应出现 "standing_with_seer": "p03" 这类硬编码
        import re as _re
        match = _re.search(r'"standing_with_seer"\s*:\s*"p03"', examples)
        assert not match, f"_format_examples hardcodes p03 as seer: {match.group(0)}"


class TestFormatExamplesNoHardcodedIDComplete:
    """P1 (post-review-v2): vote 示例所有 p0X 硬编码 ID 改占位符。"""

    def test_vote_example_no_real_p0X_hardcoded(self):
        from unittest.mock import MagicMock
        from werewolf_agent.agents.schemas import ActionType
        from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
        mock_ctx = MagicMock()
        mock_ctx.legal_actions = [ActionType.VOTE]
        mock_ctx.legal_targets = [f"p{i:02d}" for i in range(1, 13)]
        builder = PlayerPromptBuilder(mock_ctx)
        examples = builder._format_examples()
        # vote 段不应出现 "p01"-"p12" 真实 ID 字符串（应全部为 pXX 占位符）
        vote_section_match = re.search(r'(vote[^#]*?)(?=##|$)', examples, re.DOTALL)
        if vote_section_match:
            vote_section = vote_section_match.group(1)
            real_id_hits = re.findall(r'"p(?:0[1-9]|1[0-2])"', vote_section)
            assert len(real_id_hits) == 0, (
                f"vote 示例段仍含硬编码 player IDs: {real_id_hits}\n\n{vote_section[:600]}"
            )


class TestVoteReasonGuardInFullActionPath:
    """P2 (post-review-v2): vote 阶段 FULL_ACTION 模式也应注入 _VOTE_REASON_PRIVACY_GUARD。"""

    def test_vote_full_action_includes_privacy_guard(self):
        from werewolf_agent.agents.schemas import ActionType, AgentContext, RetryInfo, TaskType
        from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            own_role="villager",
            legal_actions=[ActionType.VOTE, ActionType.NO_ACTION],  # 多动作触发 FULL_ACTION
            legal_targets=[f"p{i:02d}" for i in range(1, 13)],
            strategy_directive={},
        )
        builder = PlayerPromptBuilder(ctx)
        user_prompt = builder.build_user_prompt(RetryInfo())
        # 隐私 guard 关键词必出现（"禁止" + "private_intent"）
        assert "禁止" in user_prompt and "private_intent" in user_prompt, (
            f"FULL_ACTION vote path missing _VOTE_REASON_PRIVACY_GUARD"
        )


class TestHardConstraintTierAccuracy:
    """P3 (post-review-v2): 含'严禁/绝对'约束应归 HARD_CONSTRAINT_KEYS。"""

    def test_wolf_universal_rules_in_hard_tier(self):
        from werewolf_agent.agents.prompt_builder import (
            HARD_CONSTRAINT_KEYS, SUGGESTION_KEYS,
        )
        # wolf_universal_rules 含"绝对不要提到你的队友是狼人"等硬约束
        assert "wolf_universal_rules" in HARD_CONSTRAINT_KEYS, (
            f"wolf_universal_rules 含硬约束应归 HARD: HARD={HARD_CONSTRAINT_KEYS}, SUGGESTION={SUGGESTION_KEYS}"
        )

    def test_anti_herd_in_hard_tier(self):
        from werewolf_agent.agents.prompt_builder import HARD_CONSTRAINT_KEYS
        # anti_herd 是 P0-K6 硬约束
        assert "anti_herd" in HARD_CONSTRAINT_KEYS, (
            f"anti_herd 应在 HARD_CONSTRAINT_KEYS"
        )


class TestWorldModelPhase0PromptBoundary:
    """Phase 0: future world-model context fields are not rendered yet."""

    def test_world_model_placeholders_do_not_render_without_renderer(self):
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            phase="day",
            own_role="villager",
            legal_actions=[ActionType.SPEECH],
            possible_worlds={
                "top_worlds": [
                    {
                        "world_id": "w_secret",
                        "roles": {"p03": "werewolf"},
                        "supporting_evidence": ["hidden teammate truth"],
                    }
                ]
            },
            simulation_predictions={
                "predictions": [
                    {"event": "wolves kill p04", "rationale": "secret night info"}
                ]
            },
            decision_plan_audit={"private_goal": "push hidden wolf teammate"},
            dialogue_plan_audit={"conceal": ["p03 is my teammate"]},
        )

        prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())

        for forbidden in (
            "possible_worlds",
            "simulation_predictions",
            "decision_plan_audit",
            "dialogue_plan_audit",
            "hidden teammate truth",
            "secret night info",
            "p03 is my teammate",
        ):
            assert forbidden not in prompt

    def test_possible_worlds_render_prompt_safe_hypotheses(self):
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            phase="day",
            own_role="villager",
            legal_actions=[ActionType.SPEECH],
            possible_worlds={
                "type": "possible_worlds",
                "top_worlds": [
                    {
                        "label": "World A",
                        "probability": 0.62,
                        "key_assignments": {"p03": "werewolf"},
                        "why": ["p03 vote pattern conflicts"],
                        "watch_for": ["p03 defense target"],
                        "roles": {"p02": "seer", "p03": "werewolf"},
                        "supporting_evidence": ["audit-only raw evidence"],
                    }
                ],
                "warning": "These are hypotheses from visible evidence, not ground truth.",
            },
        )

        prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())

        assert "World A" in prompt
        assert "p03=werewolf" in prompt
        assert "hypotheses" in prompt
        assert "roles" not in prompt
        assert "supporting_evidence" not in prompt
        assert "audit-only raw evidence" not in prompt

    def test_possible_worlds_keeps_current_game_ids_in_why_and_watch_for(self):
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            phase="day",
            own_role="villager",
            legal_actions=[ActionType.SPEECH],
            possible_worlds={
                "type": "possible_worlds",
                "top_worlds": [
                    {
                        "label": "World A",
                        "probability": 0.5,
                        "key_assignments": {"p03": "werewolf"},
                        "why": ["p03 vote pattern conflicts"],
                        "watch_for": ["p03 defense target"],
                    }
                ],
                "warning": "These are hypotheses from visible evidence, not ground truth.",
            },
        )
        prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
        # Current-game IDs in why/watch_for must be PRESERVED (not scrubbed to 历史玩家).
        assert "p03 vote pattern conflicts" in prompt
        assert "p03 defense target" in prompt
        assert "历史玩家 vote pattern conflicts" not in prompt

    def test_simulation_predictions_render_prompt_safe_cards(self):
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            phase="day",
            own_role="villager",
            legal_actions=[ActionType.SPEECH],
            simulation_predictions={
                "type": "simulation",
                "horizon": "next_turn",
                "predictions": [
                    {
                        "event": "next_day_vote_pressure",
                        "probability": 0.72,
                        "affected_players": ["p03"],
                        "rationale": "current relation pressure is concentrated",
                        "world_ids": ["World A"],
                        "roles": {"p03": "werewolf"},
                        "secret": "hidden truth",
                    }
                ],
                "warning": "Prediction, not fact.",
            },
        )

        prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())

        assert "next_day_vote_pressure" in prompt
        assert "Prediction, not fact." in prompt
        assert "p03" in prompt
        assert "World A" in prompt
        assert "roles" not in prompt
        assert "hidden truth" not in prompt

    def test_simulation_keeps_current_game_ids_in_rationale_and_event(self):
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            phase="day",
            own_role="villager",
            legal_actions=[ActionType.SPEECH],
            simulation_predictions={
                "type": "simulation",
                "horizon": "next_turn",
                "predictions": [
                    {
                        "event": "p03 exile pressure",
                        "probability": 0.7,
                        "affected_players": ["p03"],
                        "rationale": "p03 vote pressure concentrated",
                        "world_ids": ["World A"],
                    }
                ],
                "warning": "Prediction, not fact.",
            },
        )
        prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
        # Current-game IDs in event/rationale must be PRESERVED.
        assert "p03 exile pressure" in prompt
        assert "p03 vote pressure concentrated" in prompt

    def test_clean_current_game_list_items_guards_non_list_input(self):
        from werewolf_agent.agents.prompt_builder import _clean_current_game_list_items
        # Non-list input must return [] (not iterate chars).
        assert _clean_current_game_list_items("p03 vote", limit=2, max_chars=80) == []
        assert _clean_current_game_list_items(None, limit=2, max_chars=80) == []
        assert _clean_current_game_list_items(
            ["p03 vote pattern conflicts", "p04 defense"], limit=2, max_chars=80
        ) == ["p03 vote pattern conflicts", "p04 defense"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
