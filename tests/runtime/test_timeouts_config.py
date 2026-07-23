from __future__ import annotations

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


def test_runtime_does_not_keep_per_call_timeout_plumbing() -> None:
    import inspect
    from pathlib import Path

    from werewolf_agent.runtime.game_runner_config import GameRunnerConfig
    from werewolf_agent.runtime.nodes.node_helpers import _dispatch_agent
    from werewolf_agent.runtime.nodes.runtime_state import RuntimeState

    assert "timeout_override" not in inspect.signature(_dispatch_agent).parameters
    assert "agent_call_timeout" not in GameRunnerConfig.__dataclass_fields__
    assert "agent_call_timeout" not in RuntimeState.__annotations__
    assert "wolf_vote_timeout" not in RuntimeState.__annotations__

    root = Path(__file__).resolve().parents[2]
    scanned = [root / "werewolf_agent" / "runtime", root / "scripts" / "run_real_game.py"]
    forbidden = ("agent_call_timeout", "--no-timeout", "--timeout", "timeout_override")
    for path in scanned:
        paths = [path] if path.is_file() else path.rglob("*.py")
        for candidate in paths:
            source = candidate.read_text(encoding="utf-8")
            assert not any(token in source for token in forbidden), candidate


def test_current_operator_docs_do_not_advertise_removed_runtime_timeouts() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    operator_docs = [
        root / "狼人杀多智能体项目小白指南 v1.1.1.md",
        root / "handoff.md",
    ]
    forbidden = ("--timeout", "--no-timeout", "agent_call_timeout", "-TimeoutSeconds")

    for path in operator_docs:
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path
        assert "同步" in text
        assert "config/models.yaml" in text


def test_wolf_consensus_log_does_not_mislabel_generic_failure_as_timeout() -> None:
    import importlib
    import inspect

    wolf_consensus_module = importlib.import_module(
        "werewolf_agent.runtime.nodes.wolf_consensus"
    )

    source = inspect.getsource(wolf_consensus_module)
    assert "Agent调用超时" not in source
    assert "Agent调用失败或未返回有效结果" in source
