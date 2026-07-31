# -*- coding: utf-8 -*-
"""
验证观察投影包不连接旧玩家、模型供应商、工具结果或物理玩家目录。

作者: Project contributors
创建日期: 2026-07-31
"""

from __future__ import annotations

import ast
from importlib.util import resolve_name
from pathlib import Path

OBSERVATION_ROOT = (
    Path(__file__).resolve().parents[2]
    / "werewolf_agent"
    / "player_agents"
    / "observation"
)
PACKAGE_NAME = "werewolf_agent.player_agents.observation"
FORBIDDEN_OBSERVATION_PREFIXES = (
    "werewolf_agent.agents",
    "werewolf_agent.core.models",
    "werewolf_agent.model_gateway",
    "werewolf_agent.runtime",
    "werewolf_agent.tools.schemas",
)
FORBIDDEN_SYMBOLS = frozenset({
    "PlayerAgent",
    "GameRunner",
    "ModelRouter",
    "_dispatch_agent",
    "ToolResultMarkdownProjection",
})
FORBIDDEN_PROVIDER_MODULES = frozenset({
    "anthropic",
    "cohere",
    "google.generativeai",
    "mistralai",
    "openai",
})
FORBIDDEN_PROVIDER_SYMBOLS = frozenset({
    "Anthropic",
    "AsyncAnthropic",
    "AsyncOpenAI",
    "OpenAI",
})


def _tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imported_modules(
    path: Path,
    *,
    package_root: Path = OBSERVATION_ROOT,
) -> set[str]:
    tree = _tree(path)
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
    return any(
        module == prefix or module.startswith(f"{prefix}.")
        for prefix in FORBIDDEN_OBSERVATION_PREFIXES
    )


def _is_provider_client_module(module: str) -> bool:
    components = module.split(".")
    return any(
        module == prefix or module.startswith(f"{prefix}.")
        for prefix in FORBIDDEN_PROVIDER_MODULES
    ) or any(
        component in {"provider", "providers"}
        or component.endswith("_client")
        for component in components
    )


def _is_provider_client_symbol(symbol: str) -> bool:
    return (
        symbol in FORBIDDEN_PROVIDER_SYMBOLS
        or symbol.endswith(("Provider", "Client"))
    )


def _referenced_symbols(tree: ast.AST) -> set[str]:
    symbols = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }
    symbols.update(
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    )
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            symbols.update(
                alias.name.rsplit(".", maxsplit=1)[-1]
                for alias in node.names
            )
            symbols.update(
                alias.asname
                for alias in node.names
                if alias.asname is not None
            )
    return symbols


def _forbidden_symbols(tree: ast.AST) -> set[str]:
    return {
        symbol
        for symbol in _referenced_symbols(tree)
        if symbol in FORBIDDEN_SYMBOLS or _is_provider_client_symbol(symbol)
    }


def _direct_call_strings(tree: ast.AST) -> set[str]:
    values: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        arguments = (*node.args, *(keyword.value for keyword in node.keywords))
        values.update(
            argument.value
            for argument in arguments
            if isinstance(argument, ast.Constant)
            and isinstance(argument.value, str)
        )
    return values


def _is_forbidden_dispatched_value(value: str) -> bool:
    return (
        value in FORBIDDEN_SYMBOLS
        or _is_forbidden_module(value)
        or _is_provider_client_module(value)
        or _is_provider_client_symbol(value)
    )


def _physical_player_paths(tree: ast.AST) -> set[str]:
    paths: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        normalized = node.value.replace("\\", "/")
        if (
            normalized == "players"
            or normalized.startswith("players/")
            or "/players/" in normalized
        ):
            paths.add(node.value)
    return paths


def test_observation_module_matching_respects_python_boundaries() -> None:
    assert _is_forbidden_module("werewolf_agent.agents") is True
    assert _is_forbidden_module("werewolf_agent.agents.player") is True
    assert _is_forbidden_module("werewolf_agent.core.models") is True
    assert _is_forbidden_module("werewolf_agent.agents_v2") is False
    assert _is_forbidden_module("werewolf_agent.runtime_v2") is False


def test_observation_scanner_detects_symbols_dynamic_calls_and_player_paths(
    tmp_path: Path,
) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from werewolf_agent.model_gateway import ModelRouter\n"
        "from openai import OpenAI\n"
        "from bridge import ToolResultMarkdownProjection as Projection\n"
        "factory('PlayerAgent')\n"
        "factory('openai')\n"
        "client = OpenAIProvider()\n"
        "workspace = Path('players/p01/PLAYER.md')\n",
        encoding="utf-8",
    )
    tree = _tree(probe)

    assert "werewolf_agent.model_gateway" in _imported_modules(
        probe,
        package_root=tmp_path,
    )
    assert _is_provider_client_module("openai") is True
    assert _forbidden_symbols(tree) == {
        "ModelRouter",
        "OpenAI",
        "OpenAIProvider",
        "ToolResultMarkdownProjection",
    }
    assert {
        value
        for value in _direct_call_strings(tree)
        if _is_forbidden_dispatched_value(value)
    } == {"PlayerAgent", "openai"}
    assert _physical_player_paths(tree) == {"players/p01/PLAYER.md"}


def test_observation_package_has_no_forbidden_architecture_edges() -> None:
    assert OBSERVATION_ROOT.is_dir()
    observation_files = tuple(sorted(OBSERVATION_ROOT.rglob("*.py")))
    assert OBSERVATION_ROOT / "service.py" in observation_files
    assert OBSERVATION_ROOT / "workspace.py" in observation_files

    violations: list[str] = []
    for path in observation_files:
        relative_path = path.relative_to(OBSERVATION_ROOT)
        tree = _tree(path)
        for module in sorted(_imported_modules(path)):
            if _is_forbidden_module(module) or _is_provider_client_module(module):
                violations.append(f"{relative_path}: import {module}")
        for symbol in sorted(_forbidden_symbols(tree)):
            violations.append(f"{relative_path}: symbol {symbol}")
        for value in sorted(_direct_call_strings(tree)):
            if _is_forbidden_dispatched_value(value):
                violations.append(f"{relative_path}: dynamic {value}")
        for physical_path in sorted(_physical_player_paths(tree)):
            violations.append(f"{relative_path}: path {physical_path}")

    assert violations == []
