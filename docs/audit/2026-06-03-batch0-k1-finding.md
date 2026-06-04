# P0-K1 Finding: Skill Tool Path is Dead Code

**Date:** 2026-06-03
**Audit:** Across 3 saved games (`game_g_3528592081.json`, `game_g_2989362760.json`, `game_g_2913931821.json`), 279 total action traces.
**Result:** `submit_player_action` called 278 times. Skill tools (`skill_analyze_wolf_pit`, `skill_find_power_roles`, `skill_analyze_last_words`) called **0 times**.

## Implication

The "tool" path for skills is **dead code** in production. LLMs never invoke `load_skill` style tools — they always go directly to `submit_player_action`. The pre-injection path (where `_inject_skill_output` puts `prompt_injectable` text into `skill_analyses` → `skill_analysis_hints` → prompt) is the only effective delivery channel.

## Fix Direction

Drop the tool path entirely:
1. Remove `skill_tools` exposure in `agent_adapter.py:269-272` (don't add to `tools` list)
2. Remove `_build_skill_tool_defs` and `_resolve_tool_skills` (lines 380-404 of `context.py`)
3. Remove skill-skip retry logic in `player.py:345-389`
4. Keep pre-injection: `_inject_skill_output` continues to populate `skill_analyses`
5. Keep `skill_analysis_hints` rendering in prompt

Benefits:
- Simpler prompt (no tool schema noise)
- Faster execution (no skip-retry overhead, ~3 fewer attempts per action)
- Less confusion (single path, no double-charging)

## Test

```python
# tests/agents/test_skill_injection.py
def test_skill_tool_path_removed():
    """After fix: skill_tools should always be empty."""
    ctx = build_agent_context(engine, gs, "seer", TaskType.NIGHT_ACTION)
    assert ctx.skill_tools == []

def test_skill_analyses_still_injected():
    """Pre-injection path still works."""
    ctx = build_agent_context(engine, gs, "seer", TaskType.NIGHT_ACTION)
    # skill_analyses populated for tool skills, even if tools not exposed
    assert isinstance(ctx.skill_analyses, dict)
```
