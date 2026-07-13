# -*- coding: utf-8 -*-
"""
功能描述：基于可能世界集合生成紧凑预测，并在导出边界校验世界引用。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-13
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from werewolf_agent.cognition.worlds import PossibleWorldSet


_POWER_ROLES = {"seer", "witch", "hunter", "idiot"}


@dataclass(frozen=True)
class FutureEventPrediction:
    event_type: str
    probability: float
    affected_players: list[str] = field(default_factory=list)
    rationale: str = ""
    world_ids: list[str] = field(default_factory=list)

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "event": self.event_type,
            "probability": round(_clamp(self.probability), 3),
            "affected_players": list(self.affected_players[:3]),
            "rationale": self.rationale,
            "world_ids": list(self.world_ids[:3]),
        }


@dataclass(frozen=True)
class SimulationResult:
    viewer_id: str
    horizon: str
    predictions: list[FutureEventPrediction] = field(default_factory=list)
    retained_promptable_world_ids: list[str] = field(default_factory=list)

    def to_prompt_dict(self) -> dict[str, Any]:
        """导出经过边界校验的预测；审计计数按未知 world ID 引用次数累计。"""
        allowed = set(self.retained_promptable_world_ids)
        exported: list[dict[str, Any]] = []
        rejected_unknown_world_id_count = 0
        for prediction in self.predictions:
            unknown_ids = [
                world_id for world_id in prediction.world_ids
                if world_id not in allowed
            ]
            if unknown_ids:
                rejected_unknown_world_id_count += len(unknown_ids)
                continue
            item = prediction.to_prompt_dict()
            item["world_ids"] = list(dict.fromkeys(prediction.world_ids))[:3]
            exported.append(item)
        return {
            "type": "simulation",
            "horizon": self.horizon,
            "predictions": exported,
            "warning": "Prediction, not fact.",
            "rejected_unknown_world_id_count": rejected_unknown_world_id_count,
        }


class BoundedSimulator:
    """Produce compact heuristic predictions without replacing the rule engine."""

    def simulate(
        self,
        *,
        viewer_id: str,
        possible_worlds: PossibleWorldSet | None,
        alive_players: list[str],
        day_number: int,
        pressure_summaries: dict[str, dict[str, Any]] | None = None,
        top_k: int = 2,
    ) -> SimulationResult:
        if not possible_worlds or not possible_worlds.promptable_worlds() or not alive_players:
            return SimulationResult(viewer_id=viewer_id, horizon="next_turn")

        alive = set(alive_players)
        predictions: list[FutureEventPrediction] = []
        pressure = self._pressure_prediction(
            viewer_id=viewer_id,
            alive=alive,
            pressure_summaries=pressure_summaries or {},
            possible_worlds=possible_worlds,
        )
        if pressure is not None:
            predictions.append(pressure)

        night_kill = self._night_kill_pressure_prediction(
            viewer_id=viewer_id,
            alive=alive,
            possible_worlds=possible_worlds,
        )
        if night_kill is not None:
            predictions.append(night_kill)

        if not predictions:
            suspect = self._world_suspect_prediction(
                viewer_id=viewer_id,
                alive=alive,
                possible_worlds=possible_worlds,
            )
            if suspect is not None:
                predictions.append(suspect)

        predictions.sort(key=lambda item: (-item.probability, item.event_type))
        return SimulationResult(
            viewer_id=viewer_id,
            horizon="next_turn",
            predictions=predictions[:max(0, top_k)],
            retained_promptable_world_ids=[
                world.world_id for world in possible_worlds.promptable_worlds()
            ],
        )

    def _pressure_prediction(
        self,
        *,
        viewer_id: str,
        alive: set[str],
        pressure_summaries: dict[str, dict[str, Any]],
        possible_worlds: PossibleWorldSet,
    ) -> FutureEventPrediction | None:
        best_player = ""
        best_score = 0.0
        for player_id, summary in pressure_summaries.items():
            if player_id == viewer_id or player_id not in alive:
                continue
            pressure = _float(summary.get("pressure_score"), 0.0)
            defense = _float(summary.get("defense_score"), 0.0)
            score = max(0.0, pressure - (defense * 0.5))
            if score > best_score:
                best_player = player_id
                best_score = score
        if not best_player or best_score <= 0.0:
            return None
        return FutureEventPrediction(
            event_type="next_day_vote_pressure",
            probability=_clamp(0.4 + best_score * 0.25),
            affected_players=[best_player],
            rationale="current relation pressure is concentrated on this player",
            world_ids=_top_world_ids(possible_worlds),
        )

    def _night_kill_pressure_prediction(
        self,
        *,
        viewer_id: str,
        alive: set[str],
        possible_worlds: PossibleWorldSet,
    ) -> FutureEventPrediction | None:
        scores: dict[str, float] = {}
        world_ids: dict[str, list[str]] = {}
        for world in possible_worlds.promptable_worlds():
            for player_id, role in world.roles.items():
                if player_id == viewer_id or player_id not in alive:
                    continue
                if role in _POWER_ROLES:
                    scores[player_id] = scores.get(player_id, 0.0) + world.probability
                    world_ids.setdefault(player_id, []).append(world.world_id)
        if not scores:
            return None
        target, score = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0]
        return FutureEventPrediction(
            event_type="night_kill_pressure",
            probability=_clamp(0.25 + score * 0.45),
            affected_players=[target],
            rationale="top worlds imply this player may attract night pressure",
            world_ids=world_ids.get(target, [])[:3],
        )

    def _world_suspect_prediction(
        self,
        *,
        viewer_id: str,
        alive: set[str],
        possible_worlds: PossibleWorldSet,
    ) -> FutureEventPrediction | None:
        scores: dict[str, float] = {}
        world_ids: dict[str, list[str]] = {}
        for world in possible_worlds.promptable_worlds():
            for player_id, role in world.roles.items():
                if player_id == viewer_id or player_id not in alive:
                    continue
                if role == "werewolf":
                    scores[player_id] = scores.get(player_id, 0.0) + world.probability
                    world_ids.setdefault(player_id, []).append(world.world_id)
        if not scores:
            return None
        target, score = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0]
        return FutureEventPrediction(
            event_type="next_day_vote_pressure",
            probability=_clamp(0.3 + score * 0.4),
            affected_players=[target],
            rationale="top worlds concentrate suspicion on this player",
            world_ids=world_ids.get(target, [])[:3],
        )


def _top_world_ids(possible_worlds: PossibleWorldSet) -> list[str]:
    return [world.world_id for world in possible_worlds.promptable_worlds()[:3]]


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
