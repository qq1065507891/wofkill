# -*- coding: utf-8 -*-
"""
聚合评价快照并委托基础 outcome、质量、安全和成本指标模块。

作者: Project contributors
创建日期: 2026-07-07
修改日期: 2026-07-08

使用示例:
    >>> from werewolf_agent.evaluation.metric_aggregation import MetricsAggregator
"""

from __future__ import annotations

from werewolf_agent.evaluation.metric_outcomes import (
    compute_faction_metrics,
    compute_player_metrics,
    compute_role_metrics,
)
from werewolf_agent.evaluation.metric_quality import compute_quality_metrics
from werewolf_agent.evaluation.metric_reporting import compare_snapshots
from werewolf_agent.evaluation.schemas import (
    ActionVerdict,
    BatchConfig,
    CostMetrics,
    GameResult,
    GrowthPoint,
    MetricsSnapshot,
    ReplayRecord,
    SafetyMetrics,
)
from werewolf_agent.evaluation.world_model_metrics import compute_world_model_metrics


__all__ = ["MetricsAggregator"]


class MetricsAggregator:
    """Aggregates metrics from a list of GameResult objects."""

    _compute_faction_metrics = compute_faction_metrics
    _compute_player_metrics = compute_player_metrics
    _compute_role_metrics = compute_role_metrics
    _compute_quality_metrics = compute_quality_metrics

    def __init__(self, batch_config: BatchConfig | None = None) -> None:
        self._config = batch_config
        self._results: list[GameResult] = []

    def add_result(self, result: GameResult) -> None:
        self._results.append(result)

    def add_results(self, results: list[GameResult]) -> None:
        self._results.extend(results)

    @property
    def results(self) -> list[GameResult]:
        return list(self._results)

    def compute_snapshot(self) -> MetricsSnapshot:
        batch_id = self._config.batch_id if self._config else "default"
        snap = MetricsSnapshot(
            batch_id=batch_id,
            total_games=len(self._results),
        )
        if not self._results:
            return snap

        self._compute_faction_metrics(snap)
        self._compute_player_metrics(snap)
        self._compute_role_metrics(snap)
        self._compute_quality_metrics(snap)
        self._compute_safety_metrics(snap)
        self._compute_world_model_metrics(snap)
        self._compute_cost_metrics(snap)
        self._compute_growth_curve(snap)

        return snap

    # -----------------------------------------------------------------------
    # Safety metrics
    # -----------------------------------------------------------------------

    def _compute_safety_metrics(self, snap: MetricsSnapshot) -> None:
        s = SafetyMetrics()
        total_actions = 0
        illegal_actions = 0
        retry_recovered = 0
        fallbacks = 0
        total_leaks = 0
        total_possible_leaks = 0

        for result in self._results:
            for action in result.action_records:
                total_actions += 1
                if action.verdict == ActionVerdict.ILLEGAL:
                    illegal_actions += 1
                elif action.verdict == ActionVerdict.RETRY_RECOVERED:
                    retry_recovered += 1
                elif action.verdict == ActionVerdict.FALLBACK:
                    fallbacks += 1

            total_leaks += len(result.leakage_records)
            # Estimate possible leak points: each player per game
            total_possible_leaks += len(result.player_roles)

        if total_actions:
            s.illegal_action_rate = illegal_actions / total_actions
            s.illegal_action_count = illegal_actions
            s.retry_recovery_rate = retry_recovered / total_actions
            s.fallback_rate = fallbacks / total_actions
        if total_possible_leaks:
            s.leakage_rate = total_leaks / total_possible_leaks
        s.leakage_count = total_leaks

        snap.safety_metrics = s

    # -----------------------------------------------------------------------
    # World-model metrics
    # -----------------------------------------------------------------------

    def _compute_world_model_metrics(self, snap: MetricsSnapshot) -> None:
        compute_world_model_metrics(snap, self._results)

    # -----------------------------------------------------------------------
    # Cost / latency metrics
    # -----------------------------------------------------------------------

    def _compute_cost_metrics(self, snap: MetricsSnapshot) -> None:
        c = CostMetrics()
        total_games = len(self._results)
        if not total_games:
            snap.cost_metrics = c
            return

        total_cost = 0.0
        total_latency = 0
        latency_count = 0
        total_prompt = 0
        total_completion = 0
        # 2026-07-21 R7: 累计 prompt cache 字段.
        total_cache_creation = 0
        total_cache_read = 0
        by_provider: dict[str, float] = {}
        by_task: dict[str, float] = {}
        by_player: dict[str, float] = {}

        for result in self._results:
            for cost in result.cost_records:
                total_cost += cost.estimated_cost
                total_prompt += cost.prompt_tokens
                total_completion += cost.completion_tokens
                total_cache_creation += cost.cache_creation_input_tokens
                total_cache_read += cost.cache_read_input_tokens
                if cost.latency_ms > 0:
                    total_latency += cost.latency_ms
                    latency_count += 1
                by_provider[cost.provider] = by_provider.get(cost.provider, 0.0) + cost.estimated_cost
                by_task[cost.task_type] = by_task.get(cost.task_type, 0.0) + cost.estimated_cost
                by_player[cost.player_id] = by_player.get(cost.player_id, 0.0) + cost.estimated_cost

        player_count = len({pid for r in self._results for pid in r.player_roles})
        c.total_cost = total_cost
        c.avg_cost_per_game = total_cost / total_games
        c.avg_cost_per_player = total_cost / player_count if player_count else 0.0
        c.avg_latency_ms = int(total_latency / latency_count) if latency_count else 0
        c.total_prompt_tokens = total_prompt
        c.total_completion_tokens = total_completion
        c.total_cache_creation_tokens = total_cache_creation
        c.total_cache_read_tokens = total_cache_read
        c.by_provider = by_provider
        c.by_task_type = by_task
        c.by_player = by_player

        snap.cost_metrics = c

    # -----------------------------------------------------------------------
    # Growth curve — metric evolution across sequential games
    # -----------------------------------------------------------------------

    def _compute_growth_curve(self, snap: MetricsSnapshot) -> None:
        """计算指标随游戏进行的增长曲线。

        使用单趟扫描维护累计统计，避免 O(n^2) 的切片操作。
        """
        if len(self._results) < 2:
            return

        points: list[GrowthPoint] = []
        cumulative_good_wins = 0
        cumulative_wolf_wins = 0
        # 单趟累计：player_id -> (wins, games)
        player_cumulative: dict[str, tuple[int, int]] = {}

        for i, result in enumerate(self._results, 1):
            if result.winning_faction == "good":
                cumulative_good_wins += 1
            elif result.winning_faction == "werewolf":
                cumulative_wolf_wins += 1

            points.append(GrowthPoint(
                game_number=i,
                metric_name="good_win_rate",
                value=cumulative_good_wins / i,
            ))
            points.append(GrowthPoint(
                game_number=i,
                metric_name="werewolf_win_rate",
                value=cumulative_wolf_wins / i,
            ))

            # 单趟累计各玩家胜率，不回溯切片
            for pid in result.player_roles:
                prev_wins, prev_games = player_cumulative.get(pid, (0, 0))
                faction = result.player_factions.get(pid, "")
                cur_wins = prev_wins + (1 if faction == result.winning_faction else 0)
                cur_games = prev_games + 1
                player_cumulative[pid] = (cur_wins, cur_games)
                if cur_games:
                    points.append(GrowthPoint(
                        game_number=i,
                        metric_name=f"player_{pid}_win_rate",
                        value=cur_wins / cur_games,
                    ))

        snap.growth_curve = points

    # -----------------------------------------------------------------------
    # Replay extraction
    # -----------------------------------------------------------------------

    def extract_replay(self, result: GameResult) -> ReplayRecord:
        """Extract replay record for deterministic reproduction."""
        return ReplayRecord(
            game_id=result.game_id,
            initial_seed=result.initial_seed,
            ruleset_snapshot=result.ruleset_snapshot,
            event_log=result.event_log,
        )

    def extract_all_replays(self) -> list[ReplayRecord]:
        return [self.extract_replay(r) for r in self._results]

    # -----------------------------------------------------------------------
    # Static comparison helper
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Static comparison helper
    # -----------------------------------------------------------------------

    compare_snapshots = staticmethod(compare_snapshots)
