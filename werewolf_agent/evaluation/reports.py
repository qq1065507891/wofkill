"""Leaderboard report generator: JSON reports, experiment comparisons, growth curves.

Design doc §14: standardized benchmark with leaderboard JSON reports.
Dimensions include werewolf win rate, good win rate, anti-push rate,
lie detection rate, stance accuracy, illegal action rate, avg cost, avg latency.
"""

from __future__ import annotations

import json
import time
from typing import Any

from werewolf_agent.evaluation.metrics import MetricsAggregator
from werewolf_agent.evaluation.schemas import (
    BatchConfig,
    ExperimentComparison,
    ExperimentDimension,
    FactionMetrics,
    GrowthPoint,
    LeaderboardEntry,
    LeaderboardReport,
    MetricsSnapshot,
    ReplayRecord,
)


class ReportGenerator:
    """Generates leaderboard reports and experiment comparisons from metrics snapshots."""

    def __init__(self) -> None:
        self._snapshots: list[MetricsSnapshot] = []

    def add_snapshot(self, snapshot: MetricsSnapshot) -> None:
        self._snapshots.append(snapshot)

    def generate_leaderboard(
        self,
        player_model_map: dict[str, str] | None = None,
        player_persona_map: dict[str, str] | None = None,
    ) -> LeaderboardReport:
        """Generate a leaderboard report from all collected snapshots.

        player_model_map: player_id → model name
        player_persona_map: player_id → persona name
        """
        report_id = f"lb_{int(time.time())}"
        player_model_map = player_model_map or {}
        player_persona_map = player_persona_map or {}

        # Aggregate per-player metrics across all snapshots
        player_agg: dict[str, dict[str, float]] = {}
        total_games = 0

        for snap in self._snapshots:
            total_games += snap.total_games
            for pid, pm in snap.player_metrics.items():
                if pid not in player_agg:
                    player_agg[pid] = {"games": 0.0, "wins": 0.0, "good_wins": 0.0, "wolf_wins": 0.0}
                player_agg[pid]["games"] += pm.games
                player_agg[pid]["wins"] += pm.wins
                # Count good/wolf wins from role metrics
                for role, rm in pm.role_metrics.items():
                    if role in ("werewolf",):
                        player_agg[pid]["wolf_wins"] += rm.wins
                    else:
                        player_agg[pid]["good_wins"] += rm.wins

        # Build leaderboard entries
        entries: list[LeaderboardEntry] = []
        for pid, agg in player_agg.items():
            games = int(agg["games"])
            wins = int(agg["wins"])
            good_wins = int(agg["good_wins"])
            wolf_wins = int(agg["wolf_wins"])

            win_rate = wins / games if games else 0.0
            good_win_rate = good_wins / games if games else 0.0
            wolf_win_rate = wolf_wins / games if games else 0.0

            # Aggregate quality/safety metrics from snapshots
            avg_anti_push = 0.0
            avg_lie_detection = 0.0
            avg_stance = 0.0
            avg_illegal = 0.0
            avg_cost = 0.0
            avg_latency = 0
            snap_count = 0

            for snap in self._snapshots:
                if pid in snap.player_metrics:
                    snap_count += 1
                    avg_anti_push += snap.quality_metrics.anti_push_rate
                    avg_lie_detection += snap.quality_metrics.lie_detection_rate
                    avg_stance += snap.quality_metrics.stance_accuracy
                    avg_illegal += snap.safety_metrics.illegal_action_rate
                    avg_cost += snap.cost_metrics.avg_cost_per_game
                    avg_latency += snap.cost_metrics.avg_latency_ms

            if snap_count:
                avg_anti_push /= snap_count
                avg_lie_detection /= snap_count
                avg_stance /= snap_count
                avg_illegal /= snap_count
                avg_cost /= snap_count
                avg_latency = int(avg_latency / snap_count)

            # Overall score: weighted combination
            overall_score = (
                win_rate * 0.3
                + avg_stance * 0.2
                + avg_anti_push * 0.15
                + avg_lie_detection * 0.1
                + (1.0 - avg_illegal) * 0.15
                + (1.0 - min(avg_cost / 1.0, 1.0)) * 0.1
            )

            entries.append(LeaderboardEntry(
                rank=0,
                player_id=pid,
                model=player_model_map.get(pid, ""),
                persona=player_persona_map.get(pid, ""),
                overall_score=overall_score,
                win_rate=win_rate,
                good_win_rate=good_win_rate,
                werewolf_win_rate=wolf_win_rate,
                anti_push_rate=avg_anti_push,
                lie_detection_rate=avg_lie_detection,
                stance_accuracy=avg_stance,
                illegal_action_rate=avg_illegal,
                avg_cost_per_game=avg_cost,
                avg_latency_ms=avg_latency,
                games_played=games,
            ))

        # Sort by overall_score descending and assign ranks
        entries.sort(key=lambda e: e.overall_score, reverse=True)
        for i, entry in enumerate(entries, 1):
            entry.rank = i

        return LeaderboardReport(
            report_id=report_id,
            batch_ids=[s.batch_id for s in self._snapshots],
            entries=entries,
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            total_games=total_games,
        )

    def compare_experiments(
        self,
        snapshot_a: MetricsSnapshot,
        snapshot_b: MetricsSnapshot,
        dimension: str = "",
        label_a: str = "A",
        label_b: str = "B",
    ) -> list[ExperimentComparison]:
        """Compare two experiment snapshots across all metrics."""
        raw = MetricsAggregator.compare_snapshots(
            snapshot_a, snapshot_b,
            dimension=dimension,
            label_a=label_a,
            label_b=label_b,
        )

        comparisons = []
        for item in raw:
            comparisons.append(ExperimentComparison(
                dimension=item["dimension"],
                label_a=item["label_a"],
                label_b=item["label_b"],
                metric_name=item["metric_name"],
                value_a=item["value_a"],
                value_b=item["value_b"],
                delta=item["delta"],
                games_a=item["games_a"],
                games_b=item["games_b"],
            ))
        return comparisons

    def build_growth_curves(
        self,
        snapshots: list[MetricsSnapshot],
        labels: list[str] | None = None,
    ) -> dict[str, list[GrowthPoint]]:
        """Build growth curves from sequential snapshots."""
        curves: dict[str, list[GrowthPoint]] = {}
        labels = labels or [f"experiment_{i}" for i in range(len(snapshots))]

        for label, snap in zip(labels, snapshots):
            if snap.growth_curve:
                # Use only aggregate metrics (not per-player) for clarity
                for point in snap.growth_curve:
                    if not point.metric_name.startswith("player_"):
                        key = f"{label}_{point.metric_name}"
                        if key not in curves:
                            curves[key] = []
                        curves[key].append(point)

        return curves

    @staticmethod
    def to_json(report: LeaderboardReport) -> str:
        """Serialize leaderboard report to JSON string."""
        return json.dumps(report.to_json_dict(), indent=2, ensure_ascii=False)

    @staticmethod
    def from_json(json_str: str) -> LeaderboardReport:
        """Deserialize leaderboard report from JSON string."""
        data = json.loads(json_str)
        entries = []
        for e in data.get("entries", []):
            entries.append(LeaderboardEntry(
                rank=e["rank"],
                player_id=e["player_id"],
                model=e.get("model", ""),
                persona=e.get("persona", ""),
                overall_score=e.get("overall_score", 0.0),
                win_rate=e.get("win_rate", 0.0),
                good_win_rate=e.get("good_win_rate", 0.0),
                werewolf_win_rate=e.get("werewolf_win_rate", 0.0),
                anti_push_rate=e.get("anti_push_rate", 0.0),
                lie_detection_rate=e.get("lie_detection_rate", 0.0),
                stance_accuracy=e.get("stance_accuracy", 0.0),
                illegal_action_rate=e.get("illegal_action_rate", 0.0),
                avg_cost_per_game=e.get("avg_cost_per_game", 0.0),
                avg_latency_ms=e.get("avg_latency_ms", 0),
                games_played=e.get("games_played", 0),
            ))

        comparisons = []
        for c in data.get("comparisons", []):
            comparisons.append(ExperimentComparison(
                dimension=c["dimension"],
                label_a=c["label_a"],
                label_b=c["label_b"],
                metric_name=c["metric_name"],
                value_a=c["value_a"],
                value_b=c["value_b"],
                delta=c["delta"],
                games_a=c["games_a"],
                games_b=c["games_b"],
            ))

        growth_curves: dict[str, list[GrowthPoint]] = {}
        for key, points in data.get("growth_curves", {}).items():
            growth_curves[key] = [
                GrowthPoint(game_number=p["game_number"], metric_name=p["metric_name"], value=p["value"])
                for p in points
            ]

        return LeaderboardReport(
            report_id=data["report_id"],
            batch_ids=data.get("batch_ids", []),
            entries=entries,
            comparisons=comparisons,
            growth_curves=growth_curves,
            generated_at=data.get("generated_at", ""),
            total_games=data.get("total_games", 0),
        )
