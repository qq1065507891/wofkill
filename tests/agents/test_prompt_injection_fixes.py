"""Tests for M4 prompt-injection audit fixes.

Tasks 1, 2, 3, 4, 7, 8 all live in this module — one per issue in
``docs/superpowers/plans/2026-06-09-prompt-injection-audit-fixes.md``.

The first test (M4-1) was the only one in scope for the initial commit;
later tasks will add their own tests below.
"""

from __future__ import annotations


def test_reflection_hints_slice_uses_card_budget_3() -> None:
    """Reflection live prompt rendering must cap cards at the V2 card budget."""
    from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
    from werewolf_agent.agents.schemas import AgentContext, TaskType

    # Build 10 reflection hints.
    hints = [
        {
            "role": "seer",
            "result": "胜" if i % 2 == 0 else "负",
            "text": f"反思 {i}",
            "situation": "{}",
        }
        for i in range(10)
    ]
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,
        phase="day",
        day_number=5,
        night_number=5,
        own_role="seer",
        reflection_memory_hints=hints,
    )
    builder = PlayerPromptBuilder.__new__(PlayerPromptBuilder)
    builder.context = ctx
    text = builder._build_reflection_memory_hints()
    # Count hint entries in the rendered JSON array. The header itself
    # contains "反思" once, so plain ``text.count("反思")`` would be
    # off-by-one; instead count the unique per-entry prefix
    # ``"text":"反思 `` (note trailing space — the header has no JSON
    # ``"text":"反思 "``).
    item_count = text.count('"text":"反思 ')
    assert item_count == 3, (
        f"Expected 3 reflection hints in output, got {item_count}. "
        "Reflection V2 live prompt card budget is 3."
    )


def test_output_schema_constants_used_by_both_renderers():
    """M2-3: both renderers must reference the same field constants."""
    from werewolf_agent.agents import prompt_builder
    assert hasattr(prompt_builder, "_OUTPUT_SCHEMA_VOTE_FIELDS")
    assert hasattr(prompt_builder, "_OUTPUT_SCHEMA_SPEECH_FIELDS")
    assert hasattr(prompt_builder, "_OUTPUT_SCHEMA_SKILL_FIELDS")
    assert len(prompt_builder._OUTPUT_SCHEMA_VOTE_FIELDS) >= 5
    assert len(prompt_builder._OUTPUT_SCHEMA_SPEECH_FIELDS) >= 3
    assert len(prompt_builder._OUTPUT_SCHEMA_SKILL_FIELDS) >= 3


def test_vote_audit_fields_derived_from_constant():
    """M2-3: 投票审计字段必须从 _OUTPUT_SCHEMA_VOTE_FIELDS 派生,不重复字面。"""
    from werewolf_agent.agents import prompt_builder
    # 必须有 _VOTE_AUDIT_FIELDS 常量
    assert hasattr(prompt_builder, "_VOTE_AUDIT_FIELDS")
    # 必须等于 VOTE 减去 {choice, reason, confidence}
    vote = prompt_builder._OUTPUT_SCHEMA_VOTE_FIELDS
    audit = prompt_builder._VOTE_AUDIT_FIELDS
    expected = tuple(f for f in vote if f not in ("choice", "reason", "confidence"))
    assert audit == expected, (
        f"_VOTE_AUDIT_FIELDS 必须是 VOTE 减去 {{choice, reason, confidence}}. "
        f"Got: {audit!r}, expected: {expected!r}"
    )
    # 必须 non-empty (VOTE 9 fields - 3 = 6 期望)
    assert len(audit) >= 3, f"_VOTE_AUDIT_FIELDS 必须 >= 3 fields, got {len(audit)}"


def test_vote_basis_guidance_only_in_vote_or_speech_prompts():
    """M2-2: _VOTE_BASIS_GUIDANCE must NOT appear in night_action prompts."""
    from werewolf_agent.agents.schemas import AgentContext, TaskType
    from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder

    # Night task (wolf kill = NIGHT_ACTION): vote_basis guidance
    # should NOT be in role_guide (stable system section).
    ctx_night = AgentContext(
        agent_id="p01", task_type=TaskType.NIGHT_ACTION,
        phase="night", day_number=1, night_number=1,
        own_role="werewolf",
    )
    builder = PlayerPromptBuilder.__new__(PlayerPromptBuilder)
    builder.context = ctx_night
    role_guide = builder._build_role_guide()
    assert "vote_basis" not in role_guide.lower(), (
        f"M2-2: vote_basis guidance leaked into role_guide (stable system "
        f"section). Got: {role_guide!r}"
    )


def test_vote_basis_guidance_present_in_speech_via_strategy_directive():
    """M2-2: vote_basis guidance must be injected per-turn via
    strategy_directive (the adapter does it, role_guide doesn't)."""
    from werewolf_agent.agents.schemas import AgentContext, TaskType
    from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
    from werewolf_agent.runtime.agent_adapter import VOTE_BASIS_GUIDANCE

    ctx = AgentContext(
        agent_id="p01", task_type=TaskType.SPEECH,
        phase="day", day_number=2, night_number=2,
        own_role="villager",
        strategy_directive={"vote_basis_hint": VOTE_BASIS_GUIDANCE},
    )
    builder = PlayerPromptBuilder.__new__(PlayerPromptBuilder)
    builder.context = ctx
    text = builder._build_strategy_directive()
    assert "vote_basis" in text.lower()


def test_vote_basis_hint_registers_in_hard_or_suggestion_tier():
    """M2-2 follow-up: vote_basis_hint must not silently fall to
    【参考】 tier — the guidance contains '不要用 seer_check' which
    is a command, not a suggestion. Budget trimmer must preserve it.
    """
    from werewolf_agent.agents import prompt_builder
    hard = prompt_builder.HARD_CONSTRAINT_KEYS
    sugg = prompt_builder.SUGGESTION_KEYS
    ref = prompt_builder.REFERENCE_KEYS
    in_hard = "vote_basis_hint" in hard
    in_sugg = "vote_basis_hint" in sugg
    in_ref = "vote_basis_hint" in ref
    assert in_hard or in_sugg, (
        f"M2-2 follow-up: vote_basis_hint must register in HARD or "
        f"SUGGESTION tier (command-language: '不要用 seer_check'). "
        f"Currently in_hard={in_hard} in_sugg={in_sugg} in_ref={in_ref}. "
        f"Falling through to 【参考】 allows budget trimmer to drop the "
        f"seer_check prohibition under tight token budgets."
    )
    assert not in_ref, "vote_basis_hint should NOT be in REFERENCE tier"


def test_vote_basis_hint_renders_in_hard_section():
    """M2-2 follow-up: when strategy_directive contains vote_basis_hint,
    the rendered output must place it under the 【硬约束】 header, not
    【参考】. Verifies the actual render path, not just the registry."""
    from werewolf_agent.agents.schemas import AgentContext, TaskType
    from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
    from werewolf_agent.runtime.agent_adapter import VOTE_BASIS_GUIDANCE

    ctx = AgentContext(
        agent_id="p01", task_type=TaskType.SPEECH,
        phase="day", day_number=2, night_number=2,
        own_role="villager",
        strategy_directive={"vote_basis_hint": VOTE_BASIS_GUIDANCE},
    )
    builder = PlayerPromptBuilder.__new__(PlayerPromptBuilder)
    builder.context = ctx
    text = builder._build_strategy_directive()
    # The hard-constraint section header is 【硬约束】 (defined in
    # _STRATEGY_GROUP_ORDER). The reference header is 【参考】.
    hard_pos = text.find("【硬约束】")
    ref_pos = text.find("【参考】")
    vote_basis_pos = text.find("vote_basis")
    assert vote_basis_pos != -1, "vote_basis should appear in the rendered text"
    # If both headers exist, the vote_basis text must come after the
    # hard header and (if reference exists) before the reference header.
    if hard_pos != -1 and ref_pos != -1:
        assert hard_pos < vote_basis_pos < ref_pos, (
            f"vote_basis must be rendered under 【硬约束】, not 【参考】. "
            f"hard_pos={hard_pos} vote_basis_pos={vote_basis_pos} ref_pos={ref_pos}"
        )
    elif ref_pos != -1:
        # Hard section didn't render (no hard keys other than this) — but
        # vote_basis IS a hard key, so this branch shouldn't be reached.
        # If it is, the registry still has it under REFERENCE — fail loudly.
        assert False, (
            f"vote_basis_hint rendered under 【参考】 at pos {ref_pos}, "
            f"but it should be in 【硬约束】. Text: {text!r}"
        )


def test_inject_vote_basis_helper_centralized():
    """M2-2 follow-up: 6 injection sites consolidated to a helper."""
    from werewolf_agent.runtime import agent_adapter
    from werewolf_agent.core.models import GameState, PlayerState
    assert hasattr(agent_adapter, "_inject_vote_basis_hint")
    gs = GameState(
        game_id="t", phase="day", day_number=2, night_number=2,
        players={
            "p01": PlayerState(id="p01", role="seer", alive=True),
            "p02": PlayerState(id="p02", role="villager", alive=True),
        },
    )
    sd: dict = {}
    # Seer: no injection
    agent_adapter._inject_vote_basis_hint(sd, gs, "p01")
    assert "vote_basis_hint" not in sd, (
        "Seer should be exempt from vote_basis_hint injection"
    )
    # Villager: injected
    agent_adapter._inject_vote_basis_hint(sd, gs, "p02")
    assert "vote_basis_hint" in sd, (
        "Non-seer should get vote_basis_hint injection"
    )
    assert "speech_logic" in sd["vote_basis_hint"] or "vote_basis" in sd["vote_basis_hint"].lower()


def test_villager_role_guide_is_concise():
    """M2-1: villager guide was 4 paragraphs (~400 chars), other
    roles are 1 paragraph. Token waste + over-guidance."""
    from werewolf_agent.agents.schemas import AgentContext, TaskType
    from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder

    ctx = AgentContext(
        agent_id="p01", task_type=TaskType.SPEECH,
        phase="day", day_number=2, night_number=2,
        own_role="villager",
    )
    builder = PlayerPromptBuilder.__new__(PlayerPromptBuilder)
    builder.context = ctx
    guide = builder._build_role_guide()
    # Compress to ~200 chars while keeping the 2 key behaviors
    assert len(guide) < 200, (
        f"Villager role_guide too long: {len(guide)} chars. "
        f"Other roles are ~100 chars; this is over-guidance."
    )
    # But must keep the 2 key behaviors
    assert "N1" in guide or "解药" in guide, "should keep N1 解药 cue"
    assert "票型" in guide or "证据" in guide, "should keep evidence-based voting cue"


def test_learning_context_priority_wraps_rag_and_reflection():
    """Merged cross-game learning renders as one 【参考】 section."""
    from werewolf_agent.agents import prompt_builder
    prios = prompt_builder.PlayerPromptBuilder._SECTION_PRIORITIES
    assert prios["_build_learning_context"] == "【参考】"
    assert "_build_reflection_memory_hints" not in prios
    assert "_build_rag_hints" not in prios


def test_skill_policy_distinguishes_from_identity_rules():
    """M5-1: _build_skill_policy should clearly state that identity
    rules (above in role_guide) outrank skill advice on conflict.

    Without this 边界, LLM may conflate 'skill said vote X' with
    'role said vote X'.
    """
    import re
    from werewolf_agent.agents.schemas import AgentContext, TaskType
    from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder

    ctx = AgentContext(
        agent_id="p01", task_type=TaskType.SPEECH,
        phase="day", day_number=2, night_number=2,
        own_role="villager",
    )
    builder = PlayerPromptBuilder.__new__(PlayerPromptBuilder)
    builder.context = ctx
    policy = builder._build_skill_policy()
    # Must mention both concepts
    assert "身份" in policy and "技能" in policy, (
        f"Skill policy must mention both '身份' and '技能' "
        f"to establish the precedence 边界. Got: {policy!r}"
    )
    # Precedence direction: 身份规则 followed by 优先 (the precedence verb)
    assert re.search(r"身份规则.*优先", policy) is not None, (
        f"Skill policy must state 身份规则 优先 (over 技能建议); got: {policy!r}"
    )
