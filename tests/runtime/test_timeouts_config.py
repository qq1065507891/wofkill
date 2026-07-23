# -*- coding: utf-8 -*-
"""
验证废弃 timeout 兼容门面及运行时 deadline 清理合同。

作者: Project contributors
修改日期: 2026-07-23
"""

from __future__ import annotations


def test_timeout_compatibility_facades_share_one_inert_object() -> None:
    from werewolf_agent.runtime import agent_action_pipeline, timeouts
    from werewolf_agent.runtime.nodes import _shared, day, night, sheriff

    facades = (
        timeouts.AGENT_TIMEOUTS,
        agent_action_pipeline.AGENT_TIMEOUTS,
        _shared.AGENT_TIMEOUTS,
        day.AGENT_TIMEOUTS,
        night.AGENT_TIMEOUTS,
        sheriff.AGENT_TIMEOUTS,
    )

    assert all(facade is facades[0] for facade in facades)


def test_runtime_does_not_keep_deadline_plumbing() -> None:
    import inspect
    from pathlib import Path

    from werewolf_agent.runtime.game_runner_config import GameRunnerConfig
    from werewolf_agent.runtime.nodes.node_helpers import _dispatch_agent
    from werewolf_agent.runtime.nodes.runtime_state import RuntimeState

    assert "timeout_override" not in inspect.signature(_dispatch_agent).parameters
    assert "agent_call_timeout" not in GameRunnerConfig.__dataclass_fields__
    assert "agent_call_timeout" not in RuntimeState.__annotations__
    assert "wolf_vote_timeout" not in RuntimeState.__annotations__
    assert "runtime" + "_timer" not in RuntimeState.__annotations__
    assert "speech" + "_timed_out" not in RuntimeState.__annotations__
    assert "speech_seconds_limit" not in RuntimeState.__annotations__

    root = Path(__file__).resolve().parents[2]
    scanned = [root / "werewolf_agent" / "runtime", root / "scripts" / "run_real_game.py"]
    forbidden = (
        "agent_call_timeout",
        "--no-timeout",
        "--timeout",
        "timeout_override",
        "timed_call",
        "runtime" + "_timer",
        "_deadlines",
        "Manual" + "Timer",
        "Real" + "Timer",
        "Noop" + "Timer",
        "_timer_expired",
        "speech" + "_timed_out",
    )
    for path in scanned:
        paths = [path] if path.is_file() else path.rglob("*.py")
        for candidate in paths:
            source = candidate.read_text(encoding="utf-8")
            assert not any(token in source for token in forbidden), candidate
            if candidate.name != "timeouts.py":
                assert "AGENT_TIMEOUTS." not in source, candidate


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
