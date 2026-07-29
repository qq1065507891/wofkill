# -*- coding: utf-8 -*-
"""
验证废弃 timeout 兼容门面及运行时 deadline 清理合同。

作者: Project contributors
修改日期: 2026-07-23
"""

from __future__ import annotations

import ast
from pathlib import Path


_DEADLINE_STATE_TOKENS = frozenset({
    "runtime_timer",
    "speech_timed_out",
    "speech_seconds_limit",
})
_ACTIVE_DEADLINE_SYMBOLS = frozenset({
    "runtime_timer",
    "_timer_expired",
    "timed_call",
})
_DELETED_TIMER_IMPORTS = frozenset({"ManualTimer", "RealTimer", "NoopTimer"})


def _literal_string(node: ast.AST) -> str | None:
    """递归求值测试源码中的纯字符串拼接。"""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_string(node.left)
        right = _literal_string(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def test_runtime_and_tests_do_not_reintroduce_active_deadline_symbols() -> None:
    root = Path(__file__).resolve().parents[2]
    scanned = [
        *sorted((root / "werewolf_agent" / "runtime").rglob("*.py")),
        *sorted((root / "tests").rglob("*.py")),
    ]
    for path in scanned:
        if path.resolve() == Path(__file__).resolve():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        strings = {
            value
            for node in ast.walk(tree)
            if (value := _literal_string(node)) is not None
        }
        assert not strings.intersection(_DEADLINE_STATE_TOKENS), path
        assert not any(
            isinstance(node, ast.Name) and node.id in _ACTIVE_DEADLINE_SYMBOLS
            for node in ast.walk(tree)
        ), path
        assert not any(
            isinstance(node, ast.Attribute) and node.attr in _ACTIVE_DEADLINE_SYMBOLS
            for node in ast.walk(tree)
        ), path
        assert not any(
            isinstance(node, ast.ImportFrom)
            and node.module == "werewolf_agent.runtime.timers"
            for node in ast.walk(tree)
        ), path
        assert not any(
            isinstance(node, ast.alias) and node.name in _DELETED_TIMER_IMPORTS
            for node in ast.walk(tree)
        ), path
        assert not any(
            isinstance(node, ast.ClassDef) and node.name in _DELETED_TIMER_IMPORTS
            for node in ast.walk(tree)
        ), path


def test_authoritative_ruleset_and_docs_do_not_advertise_runtime_deadlines() -> None:
    root = Path(__file__).resolve().parents[2]
    ruleset_text = (
        root / "config" / "rulesets" / "pre_witch_hunter_idiot_mixed.yaml"
    ).read_text(encoding="utf-8")
    design_text = (
        root / "docs" / "design" / "werewolf-agent-v1-design.md"
    ).read_text(encoding="utf-8")
    guide_text = (
        root / "狼人杀多智能体项目小白指南 v1.1.1.md"
    ).read_text(encoding="utf-8")

    forbidden_ruleset = (
        "timeout_policy:",
        "timers:",
        "wolf_discussion_seconds:",
        "day_speech_seconds:",
        "per_player_time_limit_seconds:",
        "timeout_event:",
        "speech_timeout",
    )
    assert not any(token in ruleset_text for token in forbidden_ruleset)

    assert "同步调用" in design_text
    assert "provider HTTP timeout" in design_text
    assert "`timer_expired`" in design_text
    assert "仅用于读取历史 V1" in design_text
    assert "wolf_discussion_seconds" not in design_text
    assert "speech_timeout" not in design_text
    assert "计时、检查点" not in guide_text


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


def test_timeout_compatibility_module_is_explicitly_deprecated_and_inert() -> None:
    from werewolf_agent.runtime import timeouts

    module_doc = timeouts.__doc__ or ""
    source = Path(timeouts.__file__).read_text(encoding="utf-8")

    assert "deprecated" in module_doc.lower()
    assert "inert" in module_doc.lower()
    assert "不控制 provider" in module_doc
    assert "closed by the provider" not in source
    assert "wall-clock headroom" not in source


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
    ]
    forbidden = (
        "--timeout",
        "--no-timeout",
        "agent_call_timeout",
        "-TimeoutSeconds",
        "Timeout:",
    )

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
