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
_DYNAMIC_LOOKUP_KEYWORDS = {
    "__import__": "name",
    "getattr": "name",
    "getitem": "key",
    "import_module": "name",
    "resolve_name": "name",
}
_STANDARD_DYNAMIC_MODULE_CALLS = {
    "builtins.__import__": "__import__",
    "builtins.getattr": "getattr",
    "importlib.import_module": "import_module",
    "importlib.util.resolve_name": "resolve_name",
    "operator.getitem": "getitem",
}
_BUILTIN_DYNAMIC_LOOKUPS = frozenset({"__import__", "getattr"})
_STANDARD_DYNAMIC_MODULES = frozenset({
    "builtins",
    "importlib",
    "importlib.util",
    "operator",
})
_LOCAL_BINDING = ("local", "")


class _LookupScope:
    """保存单个词法作用域内、可按来源解析的名字绑定。"""

    def __init__(
        self,
        parent: _LookupScope | None = None,
        *,
        kind: str = "module",
    ) -> None:
        self.parent = parent
        self.kind = kind
        self.bindings: dict[str, tuple[str, str]] = {}

    def resolve(self, name: str) -> tuple[str, str] | None:
        binding = self.bindings.get(name)
        if binding is not None:
            return binding
        if self.parent is not None:
            return self.parent.resolve(name)
        return None


def _argument_names(arguments: ast.arguments) -> set[str]:
    return {
        argument.arg
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        )
    } | {
        argument.arg
        for argument in (arguments.vararg, arguments.kwarg)
        if argument is not None
    }


class _FunctionLocalCollector(ast.NodeVisitor):
    """预收集一个函数自己的 local 名字，不穿透嵌套作用域。"""

    def __init__(self, arguments: ast.arguments) -> None:
        self.names = _argument_names(arguments)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)

    def visit_Import(self, node: ast.Import) -> None:
        self.names.update(
            alias.asname or alias.name.split(".", maxsplit=1)[0]
            for alias in node.names
        )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.names.update(
            alias.asname or alias.name
            for alias in node.names
            if alias.name != "*"
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name is not None:
            self.names.add(node.name)
        self.generic_visit(node)

    def _visit_comprehension(self, generators: list[ast.comprehension]) -> None:
        for generator in generators:
            self.visit(generator.iter)
            for condition in generator.ifs:
                self.visit(condition)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node.generators)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node.generators)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node.generators)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node.generators)


def _function_local_names(
    arguments: ast.arguments,
    body: list[ast.stmt],
) -> set[str]:
    collector = _FunctionLocalCollector(arguments)
    for statement in body:
        collector.visit(statement)
    return collector.names


class _DispatchedStringVisitor(ast.NodeVisitor):
    """按源码顺序和词法作用域追踪动态查找调用的来源。"""

    def __init__(self, tree: ast.AST) -> None:
        self.parents = {
            child: node
            for node in ast.walk(tree)
            for child in ast.iter_child_nodes(node)
        }
        self.scope = _LookupScope()
        self.values: set[str] = set()

    def _bind(self, name: str, binding: tuple[str, str] = _LOCAL_BINDING) -> None:
        self.scope.bindings[name] = binding

    def _bind_target(self, target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            self._bind(target.id)
        elif isinstance(target, (ast.List, ast.Tuple)):
            for element in target.elts:
                self._bind_target(element)
        elif isinstance(target, ast.Starred):
            self._bind_target(target.value)

    def _canonical_lookup(self, function: ast.expr) -> str | None:
        if isinstance(function, ast.Name):
            binding = self.scope.resolve(function.id)
            if binding is not None:
                kind, provenance = binding
                return provenance if kind == "lookup" else None
            if function.id in _BUILTIN_DYNAMIC_LOOKUPS:
                return function.id
            return None

        attributes: list[str] = []
        while isinstance(function, ast.Attribute):
            attributes.append(function.attr)
            function = function.value
        if not isinstance(function, ast.Name):
            return None
        binding = self.scope.resolve(function.id)
        if binding is None or binding[0] != "module":
            return None
        dotted = ".".join((binding[1], *reversed(attributes)))
        return _STANDARD_DYNAMIC_MODULE_CALLS.get(dotted)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            local_name = alias.asname or alias.name.split(".", maxsplit=1)[0]
            imported_name = alias.name if alias.asname else local_name
            binding = (
                ("module", imported_name)
                if imported_name in _STANDARD_DYNAMIC_MODULES
                else _LOCAL_BINDING
            )
            self._bind(local_name, binding)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            if alias.name == "*":
                continue
            local_name = alias.asname or alias.name
            qualified_name = f"{module}.{alias.name}" if module else alias.name
            canonical = _STANDARD_DYNAMIC_MODULE_CALLS.get(qualified_name)
            if canonical is not None:
                self._bind(local_name, ("lookup", canonical))
            elif qualified_name in _STANDARD_DYNAMIC_MODULES:
                self._bind(local_name, ("module", qualified_name))
            else:
                self._bind(local_name)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self._bind_target(target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
        self._bind_target(node.target)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        self._bind_target(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self._bind_target(node.target)

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        for expression in (
            *node.decorator_list,
            *node.args.defaults,
            *(default for default in node.args.kw_defaults if default is not None),
        ):
            self.visit(expression)
        self._bind(node.name)

        parent = self.scope.parent if self.scope.kind == "class" else self.scope
        function_scope = _LookupScope(parent, kind="function")
        function_scope.bindings.update({
            name: _LOCAL_BINDING
            for name in _function_local_names(node.args, node.body)
        })
        previous_scope = self.scope
        self.scope = function_scope
        for statement in node.body:
            self.visit(statement)
        self.scope = previous_scope

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        lambda_scope = _LookupScope(self.scope, kind="function")
        lambda_scope.bindings.update({
            name: _LOCAL_BINDING
            for name in _argument_names(node.args)
        })
        previous_scope = self.scope
        self.scope = lambda_scope
        self.visit(node.body)
        self.scope = previous_scope

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expression in (*node.decorator_list, *node.bases, *node.keywords):
            self.visit(expression.value if isinstance(expression, ast.keyword) else expression)
        self._bind(node.name)
        previous_scope = self.scope
        self.scope = _LookupScope(previous_scope, kind="class")
        for statement in node.body:
            self.visit(statement)
        self.scope = previous_scope

    def visit_Call(self, node: ast.Call) -> None:
        for candidate in ast.walk(node.func):
            if (
                isinstance(candidate, ast.Subscript)
                and isinstance(candidate.slice, ast.Constant)
                and isinstance(candidate.slice.value, str)
            ):
                self.values.add(candidate.slice.value)

        canonical_name = self._canonical_lookup(node.func)
        if canonical_name in _DYNAMIC_LOOKUP_ARGUMENTS:
            value = _string_argument(
                node,
                _DYNAMIC_LOOKUP_ARGUMENTS[canonical_name],
                _DYNAMIC_LOOKUP_KEYWORDS[canonical_name],
            )
            if value is not None:
                self.values.add(value)
        else:
            parent = self.parents.get(node)
            if isinstance(parent, ast.Call) and parent.func is node:
                arguments = (
                    *node.args,
                    *(keyword.value for keyword in node.keywords),
                )
                self.values.update(
                    argument.value
                    for argument in arguments
                    if isinstance(argument, ast.Constant)
                    and isinstance(argument.value, str)
                )
        self.generic_visit(node)


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
        if keyword.arg != keyword_name:
            continue
        if isinstance(keyword.value, ast.Constant) and isinstance(
            keyword.value.value,
            str,
        ):
            return keyword.value.value
    return None


def _dispatched_strings(tree: ast.AST) -> set[str]:
    """只收集真正参与动态查找或调用目标选择的字符串。"""

    visitor = _DispatchedStringVisitor(tree)
    visitor.visit(tree)
    return visitor.values


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
        "from importlib.util import resolve_name as resolve_alias\n"
        "from operator import getitem as lookup\n"
        "registry['PlayerAgent']()\n"
        "registry.get('GameRunner')()\n"
        "getattr(target, 'ModelRouter')()\n"
        "load_module(name='werewolf_agent.agents.player')\n"
        "resolve_alias(name='werewolf_agent.model_gateway', package='pkg')()\n"
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
        "werewolf_agent.model_gateway",
        "werewolf_agent.runtime",
    }

    ordinary_probe = tmp_path / "ordinary_call_probe.py"
    ordinary_probe.write_text(
        "render(name='PlayerAgent')\n"
        "render(label='ProjectionProvider')\n",
        encoding="utf-8",
    )
    assert _dispatched_strings(_tree(ordinary_probe)) == set()

    local_lookup_probe = tmp_path / "local_lookup_probe.py"
    local_lookup_probe.write_text(
        "def import_module(*, name):\n"
        "    return name\n"
        "class Resolver:\n"
        "    def resolve_name(self, *, name, package):\n"
        "        return name\n"
        "import_module(name='PlayerAgent')\n"
        "Resolver().resolve_name(name='ModelRouter', package='pkg')\n",
        encoding="utf-8",
    )
    assert _dispatched_strings(_tree(local_lookup_probe)) == set()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            (
                "from app.helpers import getattr\n"
                "getattr(target, 'PlayerAgent')\n"
            ),
            set(),
        ),
        (
            (
                "def use(getattr):\n"
                "    return getattr(target, 'PlayerAgent')\n"
            ),
            set(),
        ),
        (
            (
                "from importlib import import_module as load_module\n"
                "import importlib as imports\n"
                "load_module = local_loader\n"
                "imports = local_imports\n"
                "load_module(name='werewolf_agent.runtime')\n"
                "imports.import_module(name='werewolf_agent.model_gateway')\n"
            ),
            set(),
        ),
        (
            (
                "def unrelated():\n"
                "    getattr = local_lookup\n"
                "def active():\n"
                "    return getattr(target, 'PlayerAgent')\n"
            ),
            {"PlayerAgent"},
        ),
    ],
    ids=[
        "foreign-import",
        "parameter-shadow",
        "rebound-stdlib-aliases",
        "unrelated-nested-binding",
    ],
)
def test_dispatch_scanner_resolves_provenance_in_lexical_scope(
    source: str,
    expected: set[str],
) -> None:
    assert _dispatched_strings(ast.parse(source)) == expected


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
