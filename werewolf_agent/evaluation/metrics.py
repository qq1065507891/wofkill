"""Metrics aggregator: computes all §14 metrics from game results.

Metrics categories:
- Faction metrics: good/werewolf win rates
- Player/role metrics: per-player and per-role win rates
- Quality metrics: anti-push, lie detection, vote accuracy, skill quality
- Safety metrics: leakage rate, illegal action rate
- Cost/latency metrics: per-game, per-player, per-provider
- Growth curves: metric evolution across games
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.evaluation.schemas import (
    ActionRecord,
    ActionVerdict,
    BatchConfig,
    CostMetrics,
    CostRecord,
    ExperimentDimension,
    FactionMetrics,
    GameResult,
    GrowthPoint,
    LeaderboardEntry,
    LeaderboardReport,
    LeakageRecord,
    MetricsSnapshot,
    PlayerMetrics,
    QualityMetrics,
    ReplayRecord,
    RoleMetrics,
    SafetyMetrics,
)


class MetricsAggregator:
    """Aggregates metrics from a list of GameResult objects."""

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
        self._compute_cost_metrics(snap)
        self._compute_growth_curve(snap)

        return snap

    # -----------------------------------------------------------------------
    # Faction metrics
    # -----------------------------------------------------------------------

    def _compute_faction_metrics(self, snap: MetricsSnapshot) -> None:
        total = len(self._results)
        good_wins = sum(1 for r in self._results if r.winning_faction == "good")
        wolf_wins = sum(1 for r in self._results if r.winning_faction == "werewolf")
        snap.faction_metrics = FactionMetrics(
            good_win_rate=good_wins / total if total else 0.0,
            werewolf_win_rate=wolf_wins / total if total else 0.0,
            good_wins=good_wins,
            werewolf_wins=wolf_wins,
            total_games=total,
        )

    # -----------------------------------------------------------------------
    # Player metrics
    # -----------------------------------------------------------------------

    def _compute_player_metrics(self, snap: MetricsSnapshot) -> None:
        player_stats: dict[str, dict[str, Any]] = {}

        for result in self._results:
            winner = result.winning_faction
            for pid, faction in result.player_factions.items():
                if pid not in player_stats:
                    player_stats[pid] = {"games": 0, "wins": 0, "role_stats": {}}
                stats = player_stats[pid]
                stats["games"] += 1
                if faction == winner:
                    stats["wins"] += 1
                role = result.player_roles.get(pid, "unknown")
                rs = stats["role_stats"]
                if role not in rs:
                    rs[role] = {"games": 0, "wins": 0}
                rs[role]["games"] += 1
                if faction == winner:
                    rs[role]["wins"] += 1

        for pid, stats in player_stats.items():
            games = stats["games"]
            wins = stats["wins"]
            pm = PlayerMetrics(
                player_id=pid,
                win_rate=wins / games if games else 0.0,
                games=games,
                wins=wins,
            )
            for role, rs in stats["role_stats"].items():
                pm.role_metrics[role] = RoleMetrics(
                    role=role,
                    win_rate=rs["wins"] / rs["games"] if rs["games"] else 0.0,
                    games=rs["games"],
                    wins=rs["wins"],
                )
            snap.player_metrics[pid] = pm

    # -----------------------------------------------------------------------
    # Role metrics
    # -----------------------------------------------------------------------

    def _compute_role_metrics(self, snap: MetricsSnapshot) -> None:
        role_stats: dict[str, dict[str, int]] = {}

        for result in self._results:
            winner = result.winning_faction
            for pid, role in result.player_roles.items():
                if role not in role_stats:
                    role_stats[role] = {"games": 0, "wins": 0}
                role_stats[role]["games"] += 1
                faction = result.player_factions.get(pid, "")
                if faction == winner:
                    role_stats[role]["wins"] += 1

        for role, stats in role_stats.items():
            snap.role_metrics[role] = RoleMetrics(
                role=role,
                win_rate=stats["wins"] / stats["games"] if stats["games"] else 0.0,
                games=stats["games"],
                wins=stats["wins"],
            )

    # -----------------------------------------------------------------------
    # Quality metrics
    # -----------------------------------------------------------------------

    def _compute_quality_metrics(self, snap: MetricsSnapshot) -> None:
        q = QualityMetrics()
        total_games = len(self._results)

        anti_push_count = 0
        anti_push_total = 0
        lie_detected = 0
        lie_total = 0
        stance_correct = 0
        stance_total = 0
        vote_correct = 0
        vote_total = 0
        disguise_success = 0
        disguise_total = 0
        bold_claim_success = 0
        bold_claim_total = 0
        hybrid_co_wins = 0
        hybrid_total = 0
        contradiction_hits = 0
        contradiction_total = 0

        for result in self._results:
            # Anti-push: good players who survived exile
            good_exiled = sum(
                1 for d in result.deaths
                if d.get("reason") == "exile"
                and result.player_factions.get(d.get("player_id", "")) == "good"
            )
            good_alive_end = sum(
                1 for pid, p in result.player_roles.items()
                if result.player_factions.get(pid) == "good"
                and not any(dd.get("player_id") == pid for dd in result.deaths)
            )
            total_exiled = sum(1 for d in result.deaths if d.get("reason") == "exile")
            if total_exiled > 0:
                anti_push_total += 1
                if good_exiled == 0 or good_alive_end > 0:
                    anti_push_count += 1

            # Vote accuracy: did the exiled player match the majority faction target?
            exiled_wolves = sum(
                1 for d in result.deaths
                if d.get("reason") == "exile"
                and result.player_factions.get(d.get("player_id", "")) == "werewolf"
            )
            vote_total += total_exiled
            vote_correct += exiled_wolves

            # Identity disguise: wolves who were never correctly identified
            wolves = [pid for pid, r in result.player_roles.items() if r == "werewolf"]
            for wolf_id in wolves:
                disguise_total += 1
                # Check cognition snapshots for correct identification
                for viewer_id, cognition in result.cognition_snapshots.items():
                    if viewer_id == wolf_id:
                        continue
                    entries = cognition.get("entries", {})
                    wolf_entry = entries.get(wolf_id, {})
                    top_guess = wolf_entry.get("top_role_guess", "")
                    if top_guess == "werewolf":
                        break
                else:
                    disguise_success += 1

            # Hybrid co-win rate
            hybrid_id = next(
                (pid for pid, r in result.player_roles.items() if r == "hybrid"),
                None,
            )
            if hybrid_id:
                hybrid_total += 1
                hybrid_faction = result.player_factions.get(hybrid_id, "")
                if hybrid_faction == result.winning_faction:
                    hybrid_co_wins += 1

            # Contradiction hit rate from reviews
            for review in result.reviews:
                alerts = review.get("contradiction_alerts", [])
                adopted = review.get("contradiction_adopted", [])
                if alerts:
                    contradiction_total += len(alerts)
                    contradiction_hits += len(adopted)

        if anti_push_total:
            q.anti_push_rate = anti_push_count / anti_push_total
        if lie_total:
            q.lie_detection_rate = lie_detected / lie_total
        if stance_total:
            q.stance_accuracy = stance_correct / stance_total
        if vote_total:
            q.vote_accuracy = vote_correct / vote_total
        if disguise_total:
            q.identity_disguise_rate = disguise_success / disguise_total
        if bold_claim_total:
            q.bold_claim_success_rate = bold_claim_success / bold_claim_total
        if hybrid_total:
            q.hybrid_co_win_rate = hybrid_co_wins / hybrid_total
        if contradiction_total:
            q.contradiction_hit_rate = contradiction_hits / contradiction_total

        # Compute deep hook benefit from reviews
        deep_hook_wins = sum(
            1 for r in self._results
            if r.winning_faction == "werewolf"
            and any(
                rev.get("strategy", "") == "deep_hook"
                for rev in r.reviews
            )
        )
        wolf_total = sum(1 for r in self._results if r.winning_faction == "werewolf")
        if wolf_total:
            q.deep_hook_benefit = deep_hook_wins / wolf_total

        snap.quality_metrics = q

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
                elif action.verdict == ActionVerdict.FALBACK:
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
        by_provider: dict[str, float] = {}
        by_task: dict[str, float] = {}
        by_player: dict[str, float] = {}

        for result in self._results:
            for cost in result.cost_records:
                total_cost += cost.estimated_cost
                total_prompt += cost.prompt_tokens
                total_completion += cost.completion_tokens
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
        c.by_provider = by_provider
        c.by_task_type = by_task
        c.by_player = by_player

        snap.cost_metrics = c

    # -----------------------------------------------------------------------
    # Growth curve — metric evolution across sequential games
    # -----------------------------------------------------------------------

    def _compute_growth_curve(self, snap: MetricsSnapshot) -> None:
        if len(self._results) < 2:
            return

        points: list[GrowthPoint] = []
        cumulative_good_wins = 0
        cumulative_wolf_wins = 0

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

            # Per-player cumulative win rate
            for pid in result.player_roles:
                player_wins_so_far = sum(
                    1 for r in self._results[:i]
                    if pid in r.player_factions
                    and r.player_factions[pid] == r.winning_faction
                )
                player_games_so_far = sum(
                    1 for r in self._results[:i]
                    if pid in r.player_factions
                )
                if player_games_so_far:
                    points.append(GrowthPoint(
                        game_number=i,
                        metric_name=f"player_{pid}_win_rate",
                        value=player_wins_so_far / player_games_so_far,
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

    @staticmethod
    def compare_snapshots(
        snap_a: MetricsSnapshot,
        snap_b: MetricsSnapshot,
        dimension: str = "",
        label_a: str = "A",
        label_b: str = "B",
    ) -> list[dict[str, Any]]:
        """Compare two metric snapshots across all metric dimensions."""
        comparisons = []

        def _add(metric_name: str, val_a: float, val_b: float) -> None:
            comparisons.append({
                "dimension": dimension,
                "label_a": label_a,
                "label_b": label_b,
                "metric_name": metric_name,
                "value_a": val_a,
                "value_b": val_b,
                "delta": val_b - val_a,
                "games_a": snap_a.total_games,
                "games_b": snap_b.total_games,
            })

        fm_a, fm_b = snap_a.faction_metrics, snap_b.faction_metrics
        _add("good_win_rate", fm_a.good_win_rate, fm_b.good_win_rate)
        _add("werewolf_win_rate", fm_a.werewolf_win_rate, fm_b.werewolf_win_rate)

        sm_a, sm_b = snap_a.safety_metrics, snap_b.safety_metrics
        _add("leakage_rate", sm_a.leakage_rate, sm_b.leakage_rate)
        _add("illegal_action_rate", sm_a.illegal_action_rate, sm_b.illegal_action_rate)

        qm_a, qm_b = snap_a.quality_metrics, snap_b.quality_metrics
        _add("vote_accuracy", qm_a.vote_accuracy, qm_b.vote_accuracy)
        _add("identity_disguise_rate", qm_a.identity_disguise_rate, qm_b.identity_disguise_rate)
        _add("anti_push_rate", qm_a.anti_push_rate, qm_b.anti_push_rate)

        cm_a, cm_b = snap_a.cost_metrics, snap_b.cost_metrics
        _add("avg_cost_per_game", cm_a.avg_cost_per_game, cm_b.avg_cost_per_game)
        _add("avg_latency_ms", float(cm_a.avg_latency_ms), float(cm_b.avg_latency_ms))

        return comparisons
