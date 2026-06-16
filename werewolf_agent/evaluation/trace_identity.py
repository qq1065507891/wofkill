"""Shared evaluation trace identity helpers."""

from __future__ import annotations

from dataclasses import dataclass, field


def make_trace_id(
    *,
    game_id: str,
    player_id: str,
    phase: str,
    day_number: int,
    night_number: int,
    task_type: str,
    action_index: int,
) -> str:
    return (
        f"{game_id}:{player_id}:{phase}:"
        f"D{day_number}:N{night_number}:{task_type}:{action_index}"
    )


@dataclass(frozen=True)
class DecisionIdentity:
    game_id: str
    player_id: str
    phase: str
    day_number: int
    night_number: int
    task_type: str
    action_index: int

    def trace_id(self) -> str:
        return make_trace_id(
            game_id=self.game_id,
            player_id=self.player_id,
            phase=self.phase,
            day_number=self.day_number,
            night_number=self.night_number,
            task_type=self.task_type,
            action_index=self.action_index,
        )


@dataclass
class ActionIndexAllocator:
    _next_by_game: dict[str, int] = field(default_factory=dict)

    def next(self, game_id: str) -> int:
        index = self._next_by_game.get(game_id, 0)
        self._next_by_game[game_id] = index + 1
        return index
