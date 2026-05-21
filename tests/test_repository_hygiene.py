from pathlib import Path
import ast


def test_gitignore_excludes_real_game_outputs() -> None:
    patterns = set(Path(".gitignore").read_text(encoding="utf-8").splitlines())

    assert "game_g_*.json" in patterns
    assert "game_stdout.log" in patterns
    assert "game_output.log" in patterns


def test_runtime_graph_has_no_duplicate_top_level_function_names() -> None:
    tree = ast.parse(Path("werewolf_agent/runtime/graph.py").read_text(encoding="utf-8"))
    names: dict[str, int] = {}
    duplicates: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name in names:
                duplicates.append(node.name)
            names[node.name] = node.lineno

    assert duplicates == []
