# -*- coding: utf-8 -*-
"""
验证新玩家运行时包不依赖已废弃的玩家决策模块。

作者: Project contributors
创建日期: 2026-07-29
"""

from __future__ import annotations

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[2] / "werewolf_agent" / "player_agents"
FORBIDDEN_PREFIXES = (
    "werewolf_agent.agents.player",
    "werewolf_agent.agents.action_schemas",
    "werewolf_agent.agents.schemas",
    "werewolf_agent.agents.speech_act_schemas",
    "werewolf_agent.runtime.agent_action_pipeline",
    "werewolf_agent.runtime.agent_adapter",
    "werewolf_agent.runtime.directives",
    "werewolf_agent.runtime.strategy",
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_player_agents_package_exists_and_has_no_legacy_decision_imports() -> None:
    assert PACKAGE_ROOT.is_dir()
    violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        for module in sorted(_imported_modules(path)):
            if module.startswith(FORBIDDEN_PREFIXES):
                violations.append(f"{path.relative_to(PACKAGE_ROOT)}: {module}")
    assert violations == []
