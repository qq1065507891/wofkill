"""Tests for M4 prompt-injection audit fixes.

Tasks 1, 2, 3, 4, 7, 8 all live in this module — one per issue in
``docs/superpowers/plans/2026-06-09-prompt-injection-audit-fixes.md``.

The first test (M4-1) was the only one in scope for the initial commit;
later tasks will add their own tests below.
"""

from __future__ import annotations


def test_reflection_hints_slice_uses_budget_8() -> None:
    """M4-1: prompt_builder must slice reflection hints with budget=8, not 5."""
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
    assert item_count == 8, (
        f"Expected 8 reflection hints in output, got {item_count}. "
        f"Budget is 8 in context.py but prompt_builder slices with [:5]."
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
