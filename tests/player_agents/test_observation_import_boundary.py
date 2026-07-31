# -*- coding: utf-8 -*-
"""
验证观察投影包不连接旧玩家、模型供应商、工具结果或物理玩家目录。

作者: Project contributors
创建日期: 2026-07-31
修改日期: 2026-07-31
"""

from __future__ import annotations

import ast
from importlib.util import resolve_name
from pathlib import Path

import pytest

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

_DYNAMIC_LOOKUPS = {
    "builtins.__import__": (0, "name"),
    "builtins.getattr": (1, "name"),
    "importlib.import_module": (0, "name"),
    "importlib.util.resolve_name": (0, "name"),
    "operator.getitem": (1, "key"),
}
_DYNAMIC_LOOKUP_MODULES = frozenset({
    "builtins",
    "importlib",
    "importlib.util",
    "operator",
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
    return symbols


def _forbidden_symbols(tree: ast.AST) -> set[str]:
    referenced = _referenced_symbols(tree)
    forbidden = referenced & (FORBIDDEN_SYMBOLS | FORBIDDEN_PROVIDER_SYMBOLS)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if _is_provider_client_module(node.module or ""):
            forbidden.update(alias.name for alias in node.names)
    return forbidden


def _dotted_name(node: ast.expr) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return ".".join((node.id, *reversed(parts)))


def _dynamic_lookup_aliases(tree: ast.AST) -> dict[str, str]:
    """收集显式导入的标准库动态查找别名，不模拟 Python 作用域。"""

    aliases = {
        "__import__": "builtins.__import__",
        "getattr": "builtins.getattr",
    }
    module_aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name not in _DYNAMIC_LOOKUP_MODULES:
                    continue
                local_name = imported.asname or imported.name.split(".", maxsplit=1)[0]
                module_aliases[local_name] = (
                    imported.name if imported.asname else local_name
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for imported in node.names:
                qualified = f"{module}.{imported.name}" if module else imported.name
                if qualified in _DYNAMIC_LOOKUP_MODULES:
                    module_aliases[imported.asname or imported.name] = qualified
                elif qualified in _DYNAMIC_LOOKUPS:
                    aliases[imported.asname or imported.name] = qualified

    for local_name, module in module_aliases.items():
        for qualified in _DYNAMIC_LOOKUPS:
            if qualified == module or not qualified.startswith(f"{module}."):
                continue
            aliases[f"{local_name}{qualified[len(module):]}"] = qualified
    return aliases


def _string_argument(
    call: ast.Call,
    index: int,
    keyword_name: str,
) -> str | None:
    if index < len(call.args):
        argument = call.args[index]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            return argument.value
    for keyword in call.keywords:
        if (
            keyword.arg == keyword_name
            and isinstance(keyword.value, ast.Constant)
            and isinstance(keyword.value.value, str)
        ):
            return keyword.value.value
    return None


def _direct_string_arguments(call: ast.Call) -> set[str]:
    arguments = (*call.args, *(keyword.value for keyword in call.keywords))
    return {
        argument.value
        for argument in arguments
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
    }


def _dispatched_strings(tree: ast.AST) -> set[str]:
    """收集边界关心的显式字符串分派形态。"""

    aliases = _dynamic_lookup_aliases(tree)
    values: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if (
            isinstance(node.func, ast.Subscript)
            and isinstance(node.func.slice, ast.Constant)
            and isinstance(node.func.slice.value, str)
        ):
            values.add(node.func.slice.value)
        if isinstance(node.func, ast.Call):
            values.update(_direct_string_arguments(node.func))

        spelling = _dotted_name(node.func)
        canonical = aliases.get(spelling or "")
        if canonical is None and spelling in _DYNAMIC_LOOKUPS:
            canonical = spelling
        if canonical is not None:
            index, keyword_name = _DYNAMIC_LOOKUPS[canonical]
            value = _string_argument(node, index, keyword_name)
            if value is not None:
                values.add(value)
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


def test_module_and_provider_matching_use_explicit_boundaries() -> None:
    assert _is_forbidden_module("werewolf_agent.agents.player") is True
    assert _is_forbidden_module("werewolf_agent.agents_v2") is False
    assert _is_provider_client_module("acme.clients.api") is True
    assert _is_provider_client_module("openai.resources") is True
    assert _is_provider_client_module("acme.client_tools") is False


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("registry['PlayerAgent']()", {"PlayerAgent"}),
        ("registry.get('GameRunner')()", {"GameRunner"}),
        ("factory('werewolf_agent.runtime')()", {"werewolf_agent.runtime"}),
        ("getattr(target, 'ModelRouter')", {"ModelRouter"}),
        (
            (
                "from importlib import import_module as load\n"
                "load(name='werewolf_agent.agents.player')"
            ),
            {"werewolf_agent.agents.player"},
        ),
        (
            (
                "import importlib as imports\n"
                "imports.import_module('werewolf_agent.model_gateway')"
            ),
            {"werewolf_agent.model_gateway"},
        ),
        (
            (
                "from importlib import util as iu\n"
                "iu.resolve_name(name='PlayerAgent', package='pkg')"
            ),
            {"PlayerAgent"},
        ),
        (
            (
                "from operator import getitem as lookup\n"
                "lookup(registry, key='ToolResultMarkdownProjection')"
            ),
            {"ToolResultMarkdownProjection"},
        ),
        ("render('PlayerAgent')", set()),
        ("registry['PlayerAgent']", set()),
        ("factory('werewolf_agent.runtime')", set()),
        ("registry.get('GameRunner')", set()),
    ],
    ids=[
        "subscript-callee",
        "invoked-resolver",
        "invoked-factory",
        "builtin-lookup",
        "imported-function-alias",
        "imported-module-alias",
        "from-imported-module-alias",
        "getitem-alias-keyword",
        "ordinary-call",
        "unused-subscript",
        "uninvoked-factory",
        "uninvoked-resolver",
    ],
)
def test_dispatch_scanner_handles_explicit_boundary_shapes(
    source: str,
    expected: set[str],
) -> None:
    assert _dispatched_strings(ast.parse(source)) == expected


def test_symbol_scanner_uses_import_context_not_local_suffixes() -> None:
    tree = ast.parse(
        "from acme.client import RemoteClient as Remote\n"
        "from openai import OpenAI as SDK\n"
        "from bridge import ToolResultMarkdownProjection as Projection\n"
        "class WorkspaceClient: pass\n"
        "class ProjectionProvider: pass\n"
    )

    assert _forbidden_symbols(tree) == {
        "OpenAI",
        "RemoteClient",
        "ToolResultMarkdownProjection",
    }


def test_physical_player_path_scanner_uses_exact_casefolded_segments() -> None:
    tree = ast.parse(
        "paths = ('players', '/srv/players/p01', 'C:\\\\Players\\\\p02', "
        "'multiplayers', '/srv/players_backup', 'players.md')"
    )

    assert _physical_player_paths(tree) == {
        "players",
        "/srv/players/p01",
        "C:\\Players\\p02",
    }


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
