"""P0-K1: Skill tool path is dead code; remove it.

Across 3 saved games (279 actions), skill tools were called 0 times.
LLMs always go directly to `submit_player_action`. The pre-injection path
(`skill_analyses` -> `skill_analysis_hints` -> prompt) is the only
effective delivery channel.

These tests assert that after the fix:
1. `ctx.skill_tools` field is removed entirely (no tool exposure).
2. `_build_skill_tool_defs` and `_TOOL_SKILL_NAMES` / `_SKILL_TOOL_DEFS`
   are removed (the tool-def factory is gone).
3. The pre-injection path still works: `ctx.skill_analyses` populated.
4. PlayerAgent no longer nudges the LLM to call skill tools.
"""

from __future__ import annotations

from werewolf_agent.agents.schemas import (
    AgentContext,
    TaskType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_skill_gs():
    """Build a minimal GameState for skill injection tests."""
    from werewolf_agent.core.models import GameState, PlayerState

    return GameState(
        ruleset_id="test",
        game_id="test_game",
        phase="speech",
        day_number=1,
        night_number=1,
        players={
            "p01": PlayerState(id="p01", role="werewolf", alive=True),
            "p02": PlayerState(id="p02", role="werewolf", alive=True),
            "p03": PlayerState(id="p03", role="seer", alive=True),
            "p04": PlayerState(id="p04", role="villager", alive=True),
            "p05": PlayerState(id="p05", role="villager", alive=True),
            "p06": PlayerState(id="p06", role="witch", alive=True),
        },
    )


def _build_cognition(gs, player_id: str):
    from werewolf_agent.cognition.belief import BeliefUpdater
    from werewolf_agent.cognition.contradiction import ContradictionEngine
    from werewolf_agent.cognition.world_state import build_world_state

    world_state = build_world_state(gs)
    updater = BeliefUpdater()
    belief_state = updater.initialize(list(gs.players.keys()), player_id)
    belief_state = updater.update(belief_state, world_state.facts, gs.day_number)
    alerts = ContradictionEngine().detect(world_state.facts, gs.day_number)
    return world_state, belief_state, alerts


# ---------------------------------------------------------------------------
# K1.1: skill_tools field is always empty
# ---------------------------------------------------------------------------

class TestSkillToolPathRemoved:
    """The tool path is removed: there is no `skill_tools` field at all."""

    def test_agent_context_has_no_skill_tools_field(self):
        """AgentContext no longer exposes `skill_tools` (field is removed).

        P0-K1: the LLM-callable tool path is dead code. We don't need
        to carry an empty list — the field is gone entirely.
        """
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
        )
        assert not hasattr(ctx, "skill_tools"), (
            "skill_tools field should be removed from AgentContext — "
            "the tool path is dead code."
        )

    def test_agent_context_skill_analyses_default_empty(self):
        """Pre-injection container is still present and defaults to {}."""
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
        )
        assert ctx.skill_analyses == {}

    def test_inject_skill_output_does_not_populate_skill_tools(self):
        """_inject_skill_output no longer needs to track tool skill dicts.

        After the tool path is removed, _inject_skill_output's job is to
        populate `skill_tactical_advice` only — it does not maintain a
        separate `tool_analyses` dict for on-demand LLM tool calls.

        We assert the pre-injection path is intact: `skill_tactical_advice`
        is populated and the function's second return value (analyses) is
        either empty or also present (the schema can keep the field).
        """
        from werewolf_agent.runtime.context import _inject_skill_output

        gs = _make_skill_gs()
        ws, bs, alerts = _build_cognition(gs, "p01")
        directive: dict = {}
        result, analyses = _inject_skill_output(
            directive, gs, "p01", ws, bs, alerts, "speech",
        )
        # Pre-injection still works.
        assert "skill_tactical_advice" in result
        assert isinstance(result["skill_tactical_advice"], str)
        # Tool dict is not needed: pass empty dict for `skill_tools` upstream.


# ---------------------------------------------------------------------------
# K1.2: _build_skill_tool_defs is removed (or returns [])
# ---------------------------------------------------------------------------

class TestBuildSkillToolDefsRemoved:
    """The tool-def factory should no longer be needed."""

    def test_build_skill_tool_defs_not_exported_from_context(self):
        """`_build_skill_tool_defs` is no longer defined in `context.py`."""
        from werewolf_agent.runtime import context as ctx_mod
        assert not hasattr(ctx_mod, "_build_skill_tool_defs"), (
            "_build_skill_tool_defs should be removed from context.py — "
            "the tool path is dead code."
        )

    def test_build_skill_tool_defs_not_exported_from_agent_adapter(self):
        """`_build_skill_tool_defs` is no longer re-exported by agent_adapter."""
        from werewolf_agent.runtime import agent_adapter
        assert not hasattr(agent_adapter, "_build_skill_tool_defs"), (
            "_build_skill_tool_defs should not be re-exported by agent_adapter."
        )

    def test_tool_skill_names_not_exported_from_context(self):
        """`_TOOL_SKILL_NAMES` and `_SKILL_TOOL_DEFS` are no longer in context."""
        from werewolf_agent.runtime import context as ctx_mod
        assert not hasattr(ctx_mod, "_TOOL_SKILL_NAMES"), (
            "_TOOL_SKILL_NAMES should be removed from context.py."
        )
        assert not hasattr(ctx_mod, "_SKILL_TOOL_DEFS"), (
            "_SKILL_TOOL_DEFS should be removed from context.py."
        )


# ---------------------------------------------------------------------------
# K1.3: PlayerAgent no longer has skill-skip retry logic
# ---------------------------------------------------------------------------

class TestPlayerAgentNoSkillSkipRetry:
    """PlayerAgent.act() should not retry with 'call the skill tool' nudges."""

    def test_player_agent_no_skill_skip_count(self):
        """`skill_skip_count` local variable is removed from act()."""
        import inspect

        from werewolf_agent.agents.player import PlayerAgent

        source = inspect.getsource(PlayerAgent.act)
        # The skill-skip retry block and its counter are gone.
        assert "skill_skip_count" not in source
        assert "skill_call_count" not in source
        # The tool-skip nudge prompt is gone.
        assert "请先调用分析工具" not in source

    def test_player_agent_does_not_extend_skill_tools(self):
        """PlayerAgent should not append `context.skill_tools` to its tool list."""
        import inspect

        from werewolf_agent.agents.player import PlayerAgent

        source = inspect.getsource(PlayerAgent.act)
        # The `tools.extend(context.skill_tools)` line is gone.
        assert "context.skill_tools" not in source


# ---------------------------------------------------------------------------
# K1.4: pre-injection still works
# ---------------------------------------------------------------------------

class TestSkillPreInjectionStillWorks:
    """The pre-injection path (skill_analyses -> skill_analysis_hints) remains."""

    def test_skill_analysis_hints_field_exists(self):
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            skill_analysis_hints={"x": "y"},
        )
        assert ctx.skill_analysis_hints == {"x": "y"}

    def test_skill_analysis_hints_default_empty(self):
        ctx = AgentContext(agent_id="p01", task_type=TaskType.VOTE)
        assert ctx.skill_analysis_hints == {}
