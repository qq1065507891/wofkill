# -*- coding: utf-8 -*-
"""
验证新玩家运行时包不依赖已废弃的玩家决策模块。

作者: Project contributors
创建日期: 2026-07-29
"""

from __future__ import annotations

import ast
from importlib.util import resolve_name
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[2] / "werewolf_agent" / "player_agents"
PACKAGE_NAME = "werewolf_agent.player_agents"
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


def _imported_modules(
    path: Path,
    *,
    package_root: Path = PACKAGE_ROOT,
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


def _is_forbidden_module(module: str) -> bool:
    return module.startswith(FORBIDDEN_PREFIXES)


def test_imported_modules_expands_absolute_from_import_members(tmp_path: Path) -> None:
    package_root = tmp_path / "werewolf_agent" / "player_agents"
    probe = package_root / "absolute_probe.py"
    probe.parent.mkdir(parents=True)
    probe.write_text(
        "from werewolf_agent.agents import player\n",
        encoding="utf-8",
    )

    assert "werewolf_agent.agents.player" in _imported_modules(
        probe,
        package_root=package_root,
    )


def test_imported_modules_resolves_relative_from_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "werewolf_agent" / "player_agents"
    probe = package_root / "contracts" / "relative_probe.py"
    probe.parent.mkdir(parents=True)
    probe.write_text("from ...runtime import agent_adapter\n", encoding="utf-8")

    assert "werewolf_agent.runtime.agent_adapter" in _imported_modules(
        probe,
        package_root=package_root,
    )


def test_forbidden_matching_preserves_existing_prefix_scope() -> None:
    assert _is_forbidden_module("werewolf_agent.agents.player") is True
    assert _is_forbidden_module("werewolf_agent.agents.player.helpers") is True
    assert _is_forbidden_module("werewolf_agent.agents.player_v2") is True


def test_player_agents_package_exists_and_has_no_legacy_decision_imports() -> None:
    assert PACKAGE_ROOT.is_dir()
    assert (PACKAGE_ROOT / "contracts" / "__init__.py").is_file()
    assert (PACKAGE_ROOT / "observation" / "__init__.py").is_file()
    violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        for module in sorted(_imported_modules(path)):
            if _is_forbidden_module(module):
                violations.append(f"{path.relative_to(PACKAGE_ROOT)}: {module}")
    assert violations == []
