from __future__ import annotations

import pytest

from werewolf_agent.runtime import graph


def test_phase_timeout_table_matches_real_game_contract() -> None:
    from werewolf_agent.runtime.timeouts import AGENT_TIMEOUTS

    assert AGENT_TIMEOUTS.wolf_discussion_per_player == 120.0
    assert AGENT_TIMEOUTS.wolf_consensus == 120.0
    assert AGENT_TIMEOUTS.seer == 180.0
    assert AGENT_TIMEOUTS.witch == 180.0
    assert AGENT_TIMEOUTS.day_speech == 180.0
    assert AGENT_TIMEOUTS.day_vote == 180.0
    assert AGENT_TIMEOUTS.hunter_shot == 120.0


@pytest.mark.parametrize(
    "node_name,expected",
    [
        ("wolf_discussion", "AGENT_TIMEOUTS.wolf_discussion_per_player"),
        ("wolf_consensus", "AGENT_TIMEOUTS.wolf_consensus"),
        ("night_witch", "AGENT_TIMEOUTS.witch"),
        ("night_seer", "AGENT_TIMEOUTS.seer"),
        ("free_discussion", "AGENT_TIMEOUTS.day_speech"),
        ("day_vote", "AGENT_TIMEOUTS.day_vote"),
        ("resolve_hunter_shot", "AGENT_TIMEOUTS.hunter_shot"),
    ],
)
def test_graph_uses_named_phase_timeouts(node_name: str, expected: str) -> None:
    import inspect

    source = inspect.getsource(getattr(graph, node_name))

    assert expected in source
    assert "timeout_override=60.0" not in source
    assert "timeout_override=30.0" not in source
