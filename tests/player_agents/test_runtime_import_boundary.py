# -*- coding: utf-8 -*-
"""
验证自主玩家新运行时不会导入旧玩家决策与模型调用链。

作者: Project contributors
创建日期: 2026-07-30
"""

from __future__ import annotations

import ast
from pathlib import Path

RUNTIME_ROOT = (
    Path(__file__).resolve().parents[2]
    / "werewolf_agent"
    / "player_agents"
    / "runtime"
)
FORBIDDEN_PREFIXES = (
    "werewolf_agent.agents",
    "werewolf_agent.model_gateway",
    "werewolf_agent.runtime.agent_action_pipeline",
    "werewolf_agent.runtime.agent_adapter",
    "werewolf_agent.runtime.nodes",
    "werewolf_agent.runtime.strategy",
)


def _imported_modules(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
    return tuple(modules)


def test_player_agent_runtime_is_isolated_from_legacy_decision_modules() -> None:
    assert RUNTIME_ROOT.is_dir(), "player_agents runtime package must exist"
    runtime_files = tuple(sorted(RUNTIME_ROOT.rglob("*.py")))
    assert runtime_files, "player_agents runtime package must contain Python modules"

    violations = [
        (path.relative_to(RUNTIME_ROOT), module)
        for path in runtime_files
        for module in _imported_modules(path)
        if module.startswith(FORBIDDEN_PREFIXES)
    ]

    assert violations == []
