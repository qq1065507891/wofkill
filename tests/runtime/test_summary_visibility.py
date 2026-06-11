from werewolf_agent.core.models import GameState, PlayerState
from werewolf_agent.runtime.nodes.summary import (
    _sleep_between_agent_calls,
    summarize_context,
)


def test_public_context_summary_excludes_private_player_positions() -> None:
    gs = GameState(
        game_id="summary_visibility",
        day_number=1,
        players={
            "p01": PlayerState(id="p01", role="werewolf"),
            "p02": PlayerState(id="p02", role="villager"),
        },
    )

    result = summarize_context({
        "game_state": gs,
        "discussion_positions": {
            "p01": "我是狼人，我准备推动p02出局。",
            "p02": "我怀疑p01。",
        },
    })

    event = result["game_state"].events[-1]
    assert event.type == "context_summary"
    assert event.payload["visibility"] == "public"
    assert "position_summary" not in event.payload
    assert "我是狼人" not in str(event.payload)


def test_agent_call_delay_can_be_disabled(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(
        "werewolf_agent.runtime.nodes.summary.time.sleep",
        sleeps.append,
    )

    _sleep_between_agent_calls({"agent_call_delay_ms": -1}, default_ms=10000)

    assert sleeps == []


def test_agent_call_delay_uses_node_default_when_configured_as_zero(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(
        "werewolf_agent.runtime.nodes.summary.time.sleep",
        sleeps.append,
    )

    _sleep_between_agent_calls({"agent_call_delay_ms": 0}, default_ms=20000)

    assert sleeps == [20.0]
