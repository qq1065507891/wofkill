"""S4 (post-review-v2): GameRepository Protocol completeness tests."""

from __future__ import annotations


class TestGameRepositoryProtocolCompleteness:
    """S4 (post-review-v2): GameRepository Protocol 必须声明 reflection / snapshot 方法。"""

    def test_protocol_declares_reflection_methods(self):
        from werewolf_agent.storage.repository import GameRepository

        required = {
            "save_reflection", "load_reflections_by_game", "load_reflections_by_player",
            "load_all_reflections", "delete_reflection",
            "save_memory_snapshot", "load_memory_snapshot", "list_memory_snapshots",
        }
        # Protocol method definitions live in vars()/__dict__, not __annotations__.
        # Accept either style (annotation-style or def-style) for forward compat.
        annotations = getattr(GameRepository, "__annotations__", {}) or {}
        declared = set(vars(GameRepository).keys()) | set(annotations.keys())
        for method in required:
            assert method in declared, (
                f"GameRepository Protocol missing: {method}"
            )

    def test_in_memory_implements_reflection_methods(self):
        from werewolf_agent.storage.memory_store import InMemoryGameRepository

        for method in (
            "save_reflection", "load_reflections_by_game", "load_reflections_by_player",
            "load_all_reflections", "delete_reflection",
            "save_memory_snapshot", "load_memory_snapshot", "list_memory_snapshots",
        ):
            assert hasattr(InMemoryGameRepository, method), (
                f"InMemoryGameRepository missing: {method}"
            )
