# -*- coding: utf-8 -*-
"""
验证自主玩家新运行时不会导入旧玩家决策与模型调用链。

作者: Project contributors
创建日期: 2026-07-30
"""

from __future__ import annotations

import ast
from importlib.util import resolve_name
from pathlib import Path

RUNTIME_ROOT = (
    Path(__file__).resolve().parents[2]
    / "werewolf_agent"
    / "player_agents"
    / "runtime"
)
PACKAGE_NAME = "werewolf_agent.player_agents.runtime"
FORBIDDEN_PREFIXES = (
    "werewolf_agent.agents",
    "werewolf_agent.model_gateway",
    "werewolf_agent.runtime.agent_action_pipeline",
    "werewolf_agent.runtime.agent_adapter",
    "werewolf_agent.runtime.nodes",
    "werewolf_agent.runtime.strategy",
)


def _imported_modules(
    path: Path,
    *,
    package_root: Path = RUNTIME_ROOT,
) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    relative_parent = path.relative_to(package_root).parent
    current_package = ".".join((PACKAGE_NAME, *relative_parent.parts))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = resolve_name(f"{'.' * node.level}{module}", current_package)
            if module:
                modules.add(module)
            modules.update(
                f"{module}.{alias.name}" if module else alias.name
                for alias in node.names
                if alias.name != "*"
            )
    return modules


def test_imported_modules_expands_absolute_from_import_members(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "werewolf_agent" / "player_agents" / "runtime"
    probe = package_root / "absolute_probe.py"
    probe.parent.mkdir(parents=True)
    probe.write_text(
        "from werewolf_agent import model_gateway\n",
        encoding="utf-8",
    )

    assert "werewolf_agent.model_gateway" in _imported_modules(
        probe,
        package_root=package_root,
    )


def test_imported_modules_resolves_relative_from_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "werewolf_agent" / "player_agents" / "runtime"
    probe = package_root / "probes" / "relative_probe.py"
    probe.parent.mkdir(parents=True)
    probe.write_text("from ....runtime import nodes\n", encoding="utf-8")

    assert "werewolf_agent.runtime.nodes" in _imported_modules(
        probe,
        package_root=package_root,
    )


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
