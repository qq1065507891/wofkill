from __future__ import annotations

import pytest

from werewolf_agent.runtime import graph


def test_phase_timeout_table_matches_real_game_contract() -> None:
    from werewolf_agent.runtime.timeouts import AGENT_TIMEOUTS

    assert AGENT_TIMEOUTS.wolf_discussion_per_player == 180.0
    assert AGENT_TIMEOUTS.wolf_discussion_total == 600.0
    assert AGENT_TIMEOUTS.wolf_consensus == 180.0
    # P0-R2: god-role timeouts bumped 2x to reduce empty_response rate.
    # Game trace g_3528592081: 17/82 actions ended in empty_response,
    # mostly seer (5) and villager (3). Seer check and witch action
    # benefit from 2x timeout (180s → 360s) so the model has more
    # headroom before the connection is closed.
    assert AGENT_TIMEOUTS.seer_check == 360.0, (
        "Seer check timeout must be 360s (2x of 180s baseline). "
        "P0-R2: bumped to reduce empty_response rate on seer actions."
    )
    assert AGENT_TIMEOUTS.witch_action == 360.0, (
        "Witch action timeout must be 360s (2x of 180s baseline). "
        "P0-R2: bumped to reduce empty_response rate on witch actions."
    )
    assert AGENT_TIMEOUTS.day_speech == 240.0
    assert AGENT_TIMEOUTS.day_vote == 180.0
    assert AGENT_TIMEOUTS.hunter_shot == 120.0


@pytest.mark.parametrize(
    "node_name,expected",
    [
        ("wolf_discussion", "AGENT_TIMEOUTS.wolf_discussion_per_player"),
        ("night_witch", "AGENT_TIMEOUTS.witch_action"),
        ("night_seer", "AGENT_TIMEOUTS.seer_check"),
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
