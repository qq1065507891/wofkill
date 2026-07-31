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
    "AnthropicProvider",
    "AsyncAnthropic",
    "AsyncOpenAI",
    "GLMProvider",
    "LLMProvider",
    "MiniMaxProvider",
    "MockProvider",
    "OpenAI",
    "OpenAIProvider",
    "SiliconFlowEmbeddingClient",
    "SiliconFlowRerankerClient",
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
        component in {"client", "clients", "provider", "providers"}
        or component.endswith(("_client", "_clients"))
        for component in components
    )


def _imported_provider_symbols(tree: ast.AST) -> set[str]:
    """只把 provider 模块或已知 SDK 导入名视为 provider symbol。"""

    symbols: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        for alias in node.names:
            original_name = alias.name.rsplit(".", maxsplit=1)[-1]
            if (
                _is_provider_client_module(module)
                or original_name in FORBIDDEN_PROVIDER_SYMBOLS
            ):
                symbols.add(original_name)
    return symbols


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
    forbidden = {
        symbol
        for symbol in _referenced_symbols(tree)
        if symbol in FORBIDDEN_SYMBOLS
    }
    return forbidden | _imported_provider_symbols(tree)


_DYNAMIC_LOOKUP_ARGUMENTS = {
    "__import__": 0,
    "getattr": 1,
    "getitem": 1,
    "import_module": 0,
    "resolve_name": 0,
}
def _dynamic_lookup_aliases(tree: ast.AST) -> dict[str, str]:
    """解析标准动态查找函数的本地导入别名。"""

    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module not in {"builtins", "importlib", "importlib.util", "operator"}:
            continue
        for alias in node.names:
            if alias.name in _DYNAMIC_LOOKUP_ARGUMENTS:
                aliases[alias.asname or alias.name] = alias.name
    return aliases


def _callee_name(function: ast.expr) -> str | None:
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


def _string_argument(call: ast.Call, index: int) -> str | None:
    if index >= len(call.args):
        return None
    argument = call.args[index]
    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
        return argument.value
    return None


def _dispatched_strings(tree: ast.AST) -> set[str]:
    """只收集真正参与动态查找或调用目标选择的字符串。"""

    parents = {
        child: node
        for node in ast.walk(tree)
        for child in ast.iter_child_nodes(node)
    }
    aliases = _dynamic_lookup_aliases(tree)
    values: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        for candidate in ast.walk(node.func):
            if not isinstance(candidate, ast.Subscript):
                continue
            if isinstance(candidate.slice, ast.Constant) and isinstance(
                candidate.slice.value,
                str,
            ):
                values.add(candidate.slice.value)

        callee_name = _callee_name(node.func)
        canonical_name = aliases.get(callee_name or "", callee_name)
        if canonical_name in _DYNAMIC_LOOKUP_ARGUMENTS:
            value = _string_argument(
                node,
                _DYNAMIC_LOOKUP_ARGUMENTS[canonical_name],
            )
            if value is not None:
                values.add(value)
            continue

        parent = parents.get(node)
        if isinstance(parent, ast.Call) and parent.func is node:
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
        or value in FORBIDDEN_PROVIDER_SYMBOLS
    )


def _physical_player_paths(tree: ast.AST) -> set[str]:
    paths: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        normalized = node.value.replace("\\", "/")
        if "players" in (segment.casefold() for segment in normalized.split("/")):
            paths.add(node.value)
    return paths


def test_observation_module_matching_respects_python_boundaries() -> None:
    assert _is_forbidden_module("werewolf_agent.agents") is True
    assert _is_forbidden_module("werewolf_agent.agents.player") is True
    assert _is_forbidden_module("werewolf_agent.core.models") is True
    assert _is_forbidden_module("werewolf_agent.agents_v2") is False
    assert _is_forbidden_module("werewolf_agent.runtime_v2") is False


def test_dispatch_scanner_tracks_only_dynamic_lookup_targets(tmp_path: Path) -> None:
    probe = tmp_path / "dispatch_probe.py"
    probe.write_text(
        "from app.registry import resolve as choose\n"
        "from importlib import import_module as load_module\n"
        "from operator import getitem as lookup\n"
        "registry['PlayerAgent']()\n"
        "registry.get('GameRunner')()\n"
        "getattr(target, 'ModelRouter')()\n"
        "load_module('werewolf_agent.agents.player')\n"
        "lookup(registry, 'ToolResultMarkdownProjection')()\n"
        "choose('_dispatch_agent')()\n"
        "factory('werewolf_agent.runtime')()\n",
        encoding="utf-8",
    )

    assert _dispatched_strings(_tree(probe)) == {
        "GameRunner",
        "ModelRouter",
        "PlayerAgent",
        "ToolResultMarkdownProjection",
        "_dispatch_agent",
        "werewolf_agent.agents.player",
        "werewolf_agent.runtime",
    }

    ordinary_probe = tmp_path / "ordinary_call_probe.py"
    ordinary_probe.write_text(
        "render('PlayerAgent')\n"
        "render('ProjectionProvider')\n",
        encoding="utf-8",
    )
    assert _dispatched_strings(_tree(ordinary_probe)) == set()


def test_provider_scanner_uses_import_context_not_local_suffixes(
    tmp_path: Path,
) -> None:
    probe = tmp_path / "provider_probe.py"
    probe.write_text(
        "from acme.client import RemoteClient as Remote\n"
        "from acme.clients.api import Session\n"
        "from openai import OpenAI as SDK\n"
        "class WorkspaceClient:\n"
        "    pass\n"
        "class ProjectionProvider:\n"
        "    pass\n"
        "WorkspaceClient()\n"
        "ProjectionProvider()\n",
        encoding="utf-8",
    )
    tree = _tree(probe)

    assert _is_provider_client_module("acme.client") is True
    assert _is_provider_client_module("acme.clients.api") is True
    assert _is_provider_client_module("acme.providers.openai") is True
    assert _is_provider_client_module("acme.client_tools") is False
    assert _forbidden_symbols(tree) == {"OpenAI", "RemoteClient", "Session"}


def test_physical_player_path_scanner_uses_exact_path_segments(
    tmp_path: Path,
) -> None:
    probe = tmp_path / "path_probe.py"
    probe.write_text(
        "paths = (\n"
        "    'players',\n"
        "    '/players',\n"
        "    '/srv/players/p01',\n"
        "    '/srv/players',\n"
        "    'C:\\\\players',\n"
        "    'C:\\\\Players\\\\p02',\n"
        "    'C:\\\\srv\\\\players\\\\p01',\n"
        "    'C:\\\\srv\\\\players',\n"
        "    'multiplayers',\n"
        "    '/srv/players_backup',\n"
        "    '/srv/gameplayers/p01',\n"
        "    'players.md',\n"
        ")\n",
        encoding="utf-8",
    )

    assert _physical_player_paths(_tree(probe)) == {
        "players",
        "/players",
        "/srv/players/p01",
        "/srv/players",
        "C:\\players",
        "C:\\Players\\p02",
        "C:\\srv\\players\\p01",
        "C:\\srv\\players",
    }


def test_observation_scanner_detects_symbols_dynamic_calls_and_player_paths(
    tmp_path: Path,
) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from werewolf_agent.model_gateway import ModelRouter\n"
        "from werewolf_agent.model_gateway.providers.openai import OpenAIProvider\n"
        "from openai import OpenAI\n"
        "from bridge import ToolResultMarkdownProjection as Projection\n"
        "registry['PlayerAgent']()\n"
        "__import__('openai')\n"
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
        for value in _dispatched_strings(tree)
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
        for value in sorted(_dispatched_strings(tree)):
            if _is_forbidden_dispatched_value(value):
                violations.append(f"{relative_path}: dynamic {value}")
        for physical_path in sorted(_physical_player_paths(tree)):
            violations.append(f"{relative_path}: path {physical_path}")

    assert violations == []
