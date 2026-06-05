"""Shared pytest fixtures for the Werewolf Agent test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from werewolf_agent.core.models import GameState
from werewolf_agent.engine.rule_engine import RuleEngine

_RULESET_PATH = str(
    Path(__file__).resolve().parent.parent
    / "config" / "rulesets" / "pre_witch_hunter_idiot_mixed.yaml"
)


@pytest.fixture(scope="session")
def ruleset_path() -> str:
    """Absolute path to the default V1 ruleset YAML."""
    return _RULESET_PATH


@pytest.fixture(scope="session")
def engine(ruleset_path: str) -> RuleEngine:
    """Session-scoped RuleEngine — read-only after construction, safe to share."""
    return RuleEngine.from_yaml(ruleset_path)


@pytest.fixture(scope="session")
def game_graph():
    """Session-scoped LangGraph StateGraph for runtime/integration tests.

    ``build_game_graph()`` is expensive (compiles all 20+ node definitions
    into a CompiledStateGraph). 49 test call sites invoke it; sharing
    one session-scoped instance saves ~20-30s of redundant compilation.
    The graph has no per-test state — tests stream events through it
    without mutating graph structure — so sharing is safe.
    """
    from werewolf_agent.runtime.graph import build_game_graph
    return build_game_graph()


@pytest.fixture
def new_gs() -> GameState:
    """Fresh GameState with a unique game_id per test invocation."""
    import uuid
    return GameState(game_id=uuid.uuid4().hex[:8])
