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

import re
from typing import Any

from werewolf_agent.evaluation.schemas import (
    ActionVerdict,
    BatchConfig,
    CostMetrics,
    FactionMetrics,
    GameResult,
    GrowthPoint,
    MetricProvenance,
    MetricsSnapshot,
    PlayerMetrics,
    QualityMetrics,
    ReplayRecord,
    RoleMetrics,
    SafetyMetrics,
    WorldModelMetrics,
)
from werewolf_agent.evaluation.decision_helpers import (
    decision_is_legal_from_trace as _decision_is_legal_from_trace,
    dialogue_leaked_from_trace as _dialogue_leaked_from_trace,
)
from werewolf_agent.evaluation.trace_builder import EvaluationTraceBuilder
from werewolf_agent.evaluation.world_model_eval import compute_world_model_rank_metrics

_CLAIM_ROLE_MAP = {
    "预言家": "seer",
    "女巫": "witch",
    "猎人": "hunter",
    "白痴": "idiot",
    "村民": "villager",
    "平民": "villager",
    "混血儿": "hybrid",
    "狼人": "werewolf",
}


def _extract_claim_events(event_log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for event in event_log:
        event_type = event.get("type")
        payload = event.get("payload") or {}
        if event_type == "claim_role":
            claims.append(event)
            continue
        if event_type not in ("speech", "sheriff_speech", "pk_speech", "tie_pk_speech"):
            continue
        text = str(payload.get("text") or event.get("text") or "")
        speaker = str(payload.get("speaker") or event.get("player_id") or "")
        match = re.search(
            r"(?:我是|我跳|我认)\s*(预言家|女巫|猎人|白痴|村民|平民|混血儿|狼人)",
            text,
        )
        if not match or not speaker:
            continue
        claims.append({
            "type": "claim_role",
            "payload": {
                "player_id": speaker,
                "claimed_role": _CLAIM_ROLE_MAP[match.group(1)],
            },
        })
    return claims


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
        self._compute_world_model_metrics(snap)
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
        provenance: dict[str, dict[str, Any]] = {}

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
        contradiction_adopted = 0
        contradiction_adopted_total = 0
        games_with_contradictions = 0
        potion_beneficial = 0
        potion_total = 0
        seer_checks_correct = 0
        seer_checks_total = 0
        wolf_strategic_kills = 0
        wolf_kill_total = 0
        badge_decisions_recorded = 0
        badge_beneficial = 0
        speech_influence_aligned = 0
        speech_influence_total = 0
        speech_order_with_ref = 0
        speech_order_total = 0
        compression_sum = 0.0
        compression_count = 0

        for result in self._results:
            exiled_ids = {
                d.get("player_id") for d in result.deaths if d.get("reason") == "exile"
            }
            claim_events = _extract_claim_events(result.event_log)
            seer_check_events = [
                e for e in result.event_log if e.get("type") == "seer_check"
            ]
            potion_events = [
                e for e in result.event_log
                if e.get("type") in (
                    "antidote_used",
                    "poison_used",
                    "witch_antidote_used",
                    "witch_poison_used",
                )
            ]
            wolf_kill_events = [
                e for e in result.event_log
                if e.get("type") in ("wolf_kill", "wolf_kill_selected")
            ]
            badge_events = [
                e for e in result.event_log
                if e.get("type") in (
                    "badge_transfer",
                    "badge_tear",
                    "badge_transferred",
                    "badge_torn",
                )
            ]

            # --- Anti-push ---
            good_exiled = sum(
                1 for d in result.deaths
                if d.get("reason") == "exile"
                and result.player_factions.get(d.get("player_id", "")) == "good"
            )
            total_exiled = sum(1 for d in result.deaths if d.get("reason") == "exile")
            if total_exiled > 0:
                anti_push_total += total_exiled
                anti_push_count += total_exiled - good_exiled

            # --- Vote accuracy ---
            exiled_wolves = sum(
                1 for d in result.deaths
                if d.get("reason") == "exile"
                and result.player_factions.get(d.get("player_id", "")) == "werewolf"
            )
            vote_total += total_exiled
            vote_correct += exiled_wolves

            # --- Lie detection ---
            for ce in claim_events:
                pid = ce.get("payload", {}).get("player_id", "")
                claimed = ce.get("payload", {}).get("claimed_role", "")
                actual = result.player_roles.get(pid, "")
                if actual == "werewolf" and claimed != actual:
                    lie_total += 1
                    # Check if any viewer's cognition detected this lie
                    for viewer_id, cognition in result.cognition_snapshots.items():
                        if viewer_id == pid:
                            continue
                        entries = cognition.get("entries", {})
                        entry = entries.get(pid, {})
                        if entry.get("top_role_guess") == "werewolf":
                            lie_detected += 1
                            break

            # --- Stance accuracy (good voters) ---
            for ar in result.action_records:
                if ar.action_type == "vote" and ar.target_id:
                    voter_faction = result.player_factions.get(ar.player_id, "")
                    target_faction = result.player_factions.get(ar.target_id, "")
                    if voter_faction == "good":
                        stance_total += 1
                        if target_faction == "werewolf":
                            stance_correct += 1

            # --- Identity disguise ---
            wolves = [pid for pid, r in result.player_roles.items() if r == "werewolf"]
            for wolf_id in wolves:
                disguise_total += 1
                for viewer_id, cognition in result.cognition_snapshots.items():
                    if viewer_id == wolf_id:
                        continue
                    entries = cognition.get("entries", {})
                    wolf_entry = entries.get(wolf_id, {})
                    if wolf_entry.get("top_role_guess") == "werewolf":
                        break
                else:
                    disguise_success += 1

            # --- Bold claim success ---
            for ce in claim_events:
                pid = ce.get("payload", {}).get("player_id", "")
                actual = result.player_roles.get(pid, "")
                claimed = ce.get("payload", {}).get("claimed_role", "")
                if actual == "werewolf" and claimed in ("seer", "witch", "hunter"):
                    bold_claim_total += 1
                    if pid not in exiled_ids and result.winning_faction == "werewolf":
                        bold_claim_success += 1

            # --- Hybrid co-win / master choice benefit ---
            hybrid_id = next(
                (pid for pid, r in result.player_roles.items() if r == "hybrid"),
                None,
            )
            if hybrid_id:
                hybrid_total += 1
                hybrid_faction = result.player_factions.get(hybrid_id, "")
                if hybrid_faction == result.winning_faction:
                    hybrid_co_wins += 1

            # --- Witch potion benefit ---
            for pe in potion_events:
                target_id = pe.get("payload", {}).get("target_id", "")
                target_faction = result.player_factions.get(target_id, "")
                potion_type = pe.get("type")
                potion_total += 1
                if potion_type in ("antidote_used", "witch_antidote_used") and target_faction == "good":
                    potion_beneficial += 1
                elif potion_type in ("poison_used", "witch_poison_used") and target_faction == "werewolf":
                    potion_beneficial += 1

            # --- Seer badge-flow quality ---
            for se in seer_check_events:
                target_id = se.get("payload", {}).get("target_id", "")
                alignment = se.get("payload", {}).get("alignment", "")
                seer_checks_total += 1
                if alignment in ("wolf", "werewolf") and target_id in exiled_ids:
                    seer_checks_correct += 1

            # --- Wolf consensus quality ---
            power_roles = {"seer", "witch", "hunter"}
            for we in wolf_kill_events:
                target_id = we.get("payload", {}).get("target_id", "")
                target_role = result.player_roles.get(target_id, "")
                wolf_kill_total += 1
                if target_role in power_roles:
                    wolf_strategic_kills += 1

            # --- Badge decision quality ---
            for be in badge_events:
                badge_decisions_recorded += 1
                payload = be.get("payload", {})
                to_id = payload.get("to_id") or payload.get("new_sheriff_id", "")
                if to_id:
                    to_faction = result.player_factions.get(to_id, "")
                    if to_faction == result.winning_faction:
                        badge_beneficial += 1

            # --- Speech influence rate ---
            speech_events = [
                e for e in result.event_log if e.get("type") == "speech"
            ]
            for speech in speech_events:
                speaker = speech.get("player_id", "")
                day = speech.get("day_number", 0)
                targets = speech.get("mentioned_targets", [])
                if not targets:
                    continue
                # Count subsequent votes on same day from different speakers
                for vote in result.event_log:
                    if (
                        vote.get("type") == "vote"
                        and vote.get("day_number") == day
                        and vote.get("player_id", "") != speaker
                    ):
                        speech_influence_total += 1
                        if vote.get("target_id", "") in targets:
                            speech_influence_aligned += 1

            # --- Multi-speech rate (field: speech_order_utilization) ---
            # Measures fraction of speeches that are not the first speech of their day,
            # i.e. discussion depth beyond the opening speech each round.
            for idx, speech in enumerate(speech_events):
                speech_order_total += 1
                day = speech.get("day_number", 0)
                for prev in speech_events[:idx]:
                    if prev.get("day_number") == day:
                        speech_order_with_ref += 1
                        break

            # --- Cognitive compression rate ---
            for _pid, cognition in result.cognition_snapshots.items():
                orig = cognition.get("original_fact_count", 0)
                comp = cognition.get("compressed_fact_count", 0)
                if orig > 0:
                    compression_sum += comp / orig
                    compression_count += 1

            # --- Contradiction from reviews ---
            for review in result.reviews:
                alerts = review.get("contradiction_alerts", [])
                adopted = review.get("contradiction_adopted", [])
                if alerts:
                    contradiction_total += len(alerts)
                    contradiction_hits += len(adopted)
                    contradiction_adopted_total += len(alerts)
                    contradiction_adopted += len(adopted)
                    games_with_contradictions += 1

        # --- Set rates ---
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
            q.hybrid_master_choice_benefit = hybrid_co_wins / hybrid_total
        if contradiction_total:
            q.contradiction_hit_rate = games_with_contradictions / max(1, len(self._results))
        if contradiction_adopted_total:
            q.contradiction_adopted_rate = contradiction_adopted / contradiction_adopted_total
        if potion_total:
            q.witch_potion_benefit = potion_beneficial / potion_total
        if seer_checks_total:
            q.seer_badge_flow_quality = seer_checks_correct / seer_checks_total
        if wolf_kill_total:
            q.wolf_consensus_quality = wolf_strategic_kills / wolf_kill_total
        if badge_decisions_recorded:
            q.badge_decision_quality = badge_beneficial / badge_decisions_recorded
        if speech_influence_total:
            q.speech_influence_rate = speech_influence_aligned / speech_influence_total
        if speech_order_total:
            q.speech_order_utilization = speech_order_with_ref / speech_order_total
        if compression_count:
            q.cognitive_compression_rate = compression_sum / compression_count

        # Deep hook benefit
        deep_hook_wins = sum(
            1 for r in self._results
            if r.winning_faction == "werewolf"
            and any(rev.get("strategy", "") == "deep_hook" for rev in r.reviews)
        )
        wolf_total = sum(1 for r in self._results if r.winning_faction == "werewolf")
        if wolf_total:
            q.deep_hook_benefit = deep_hook_wins / wolf_total

        snap.quality_metrics = q

        # --- Build provenance ---
        game_ids = [r.game_id for r in self._results]

        def _prov(name: str, method: str, sources: list[str], count: int) -> None:
            provenance[name] = {
                "metric_name": name,
                "computation_method": method,
                "source_types": sources,
                "source_count": count,
                "contributing_games": game_ids,
                "sample_entries": [],
            }

        _prov("anti_push_rate", "non-good exiled / total exiled (lower good exile rate = better defense)",
              ["deaths", "player_factions"], anti_push_total)
        _prov("lie_detection_rate", "detected false claims / total false claims from event_log",
              ["event_log", "cognition_snapshots"], lie_total)
        _prov("stance_accuracy", "good voters targeting wolves / total good voter actions",
              ["action_records", "player_factions"], stance_total)
        _prov("vote_accuracy", "exiled wolves / total exiles",
              ["deaths", "player_factions"], vote_total if vote_total > 0 else len(self._results))
        _prov("identity_disguise_rate", "undetected wolves / total wolves",
              ["cognition_snapshots", "player_roles"], disguise_total)
        _prov("bold_claim_success_rate", "successful wolf power-role claims / total bold claims",
              ["event_log", "deaths", "player_roles"], bold_claim_total)
        _prov("hybrid_co_win_rate", "hybrid faction == winner / games with hybrid",
              ["player_roles", "player_factions"], hybrid_total)
        _prov("hybrid_master_choice_benefit", "same as hybrid co-win rate",
              ["player_roles", "player_factions", "winning_faction"], hybrid_total)
        _prov("witch_potion_benefit", "beneficial potions / total potions",
              ["event_log", "player_factions"], potion_total)
        _prov("seer_badge_flow_quality", "seer checks leading to correct exile / total seer checks",
              ["event_log", "deaths"], seer_checks_total)
        _prov("wolf_consensus_quality", "power-role kills / total wolf kills",
              ["event_log", "player_roles"], wolf_kill_total)
        _prov("badge_decision_quality", "beneficial badge decisions / total badge events",
              ["event_log", "player_factions"], badge_decisions_recorded)
        _prov("contradiction_hit_rate", "games with contradictions / total games",
              ["reviews"], contradiction_total)
        _prov("contradiction_adopted_rate", "adopted alerts / total alerts from reviews",
              ["reviews"], contradiction_adopted_total)
        _prov("deep_hook_benefit", "deep-hook wolf wins / total wolf wins",
              ["reviews", "winning_faction"], wolf_total)
        _prov("speech_influence_rate", "post-speech votes aligned with speaker target / total post-speech votes",
              ["event_log"], speech_influence_total)
        _prov("speech_order_utilization", "speeches with prior same-day reference / total speeches",
              ["event_log"], speech_order_total)
        _prov("cognitive_compression_rate", "avg compressed/original fact ratio from cognition snapshots",
              ["cognition_snapshots"], compression_count)

        snap.provenance = {
            name: MetricProvenance(**data) for name, data in provenance.items()
        }

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
        belief_scores: list[float] = []
        possible_world_hits: list[bool] = []
        simulation_hits: list[bool] = []
        decision_legal: list[bool] = []
        dialogue_leaks: list[bool] = []

        for result in self._results:
            for review in result.reviews:
                audit = review.get("world_model_audit") if isinstance(review, dict) else None
                if not isinstance(audit, dict):
                    continue
                _collect_world_model_audit_samples(
                    audit,
                    player_roles=result.player_roles,
                    belief_scores=belief_scores,
                    possible_world_hits=possible_world_hits,
                    simulation_hits=simulation_hits,
                    decision_legal=decision_legal,
                    dialogue_leaks=dialogue_leaks,
                )
            for event in result.event_log:
                trace = _action_trace_from_event(event)
                if not trace:
                    continue
                audit = trace.get("world_model_audit")
                if isinstance(audit, dict):
                    _collect_world_model_audit_samples(
                        audit,
                        player_roles=result.player_roles,
                        belief_scores=belief_scores,
                        possible_world_hits=possible_world_hits,
                        simulation_hits=simulation_hits,
                        decision_legal=decision_legal,
                        dialogue_leaks=dialogue_leaks,
                    )
                legal = _decision_is_legal_from_trace(trace)
                if legal is not None:
                    decision_legal.append(legal)
                leaked = _dialogue_leaked_from_trace(trace)
                if leaked is not None:
                    dialogue_leaks.append(leaked)

        rank_supported = 0
        rank_unsupported = 0
        rank_top1_hits = 0.0
        rank_top3_hits = 0.0
        rank_sum = 0.0
        rank_overconfident = 0.0
        for result in self._results:
            rank_metrics = compute_world_model_rank_metrics(
                result,
                EvaluationTraceBuilder().build(result, exposure_audits=[]),
            )
            rank_supported += rank_metrics.supported_count
            rank_unsupported += rank_metrics.unsupported_count
            rank_top1_hits += rank_metrics.true_world_top1_rate * rank_metrics.supported_count
            rank_top3_hits += rank_metrics.true_world_top3_rate * rank_metrics.supported_count
            rank_sum += rank_metrics.avg_true_world_rank * rank_metrics.supported_count
            rank_overconfident += (
                rank_metrics.overconfidence_rate * rank_metrics.supported_count
            )

        snap.world_model_metrics = WorldModelMetrics(
            belief_calibration=_avg(belief_scores),
            possible_world_topk_hit_rate=_bool_rate(possible_world_hits),
            simulator_prediction_hit_rate=_bool_rate(simulation_hits),
            decision_legality_rate=_bool_rate(decision_legal),
            dialogue_leakage_rate=_bool_rate(dialogue_leaks),
            true_world_top1_rate=_rate_from_counts(rank_top1_hits, rank_supported),
            true_world_top3_rate=_rate_from_counts(rank_top3_hits, rank_supported),
            avg_true_world_rank=_rate_from_counts(rank_sum, rank_supported),
            world_rank_overconfidence_rate=_rate_from_counts(
                rank_overconfident,
                rank_supported,
            ),
            world_rank_supported_count=rank_supported,
            world_rank_unsupported_count=rank_unsupported,
        )

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
        _add("lie_detection_rate", qm_a.lie_detection_rate, qm_b.lie_detection_rate)
        _add("stance_accuracy", qm_a.stance_accuracy, qm_b.stance_accuracy)
        _add("bold_claim_success_rate", qm_a.bold_claim_success_rate, qm_b.bold_claim_success_rate)
        _add("hybrid_co_win_rate", qm_a.hybrid_co_win_rate, qm_b.hybrid_co_win_rate)
        _add("witch_potion_benefit", qm_a.witch_potion_benefit, qm_b.witch_potion_benefit)
        _add("seer_badge_flow_quality", qm_a.seer_badge_flow_quality, qm_b.seer_badge_flow_quality)
        _add("wolf_consensus_quality", qm_a.wolf_consensus_quality, qm_b.wolf_consensus_quality)
        _add("contradiction_hit_rate", qm_a.contradiction_hit_rate, qm_b.contradiction_hit_rate)

        cm_a, cm_b = snap_a.cost_metrics, snap_b.cost_metrics
        _add("avg_cost_per_game", cm_a.avg_cost_per_game, cm_b.avg_cost_per_game)
        _add("avg_latency_ms", float(cm_a.avg_latency_ms), float(cm_b.avg_latency_ms))

        return comparisons


# ---------------------------------------------------------------------------
# Game pace metrics (from event log / GameState)
# ---------------------------------------------------------------------------


def compute_pace_metrics(
    events: list[dict[str, Any]],
    *,
    deaths: list[dict[str, Any]] | None = None,
    finish_night: int | None = None,
) -> dict[str, Any]:
    """Compute game pace metrics from event log.

    Returns dict with:
    - day_exile_rate: fraction of days that produced an exile
    - max_consecutive_no_exile_days: longest streak of no-exile days
    - second_tie_count: number of second_tie_no_exile events
    - stale_vote_reuse_count: days where votes were identical to a previous day
    - finish_night_number: night the game ended
    - pace_target_met: bool
    """
    vote_events = [
        e for e in events if e.get("type") == "vote_resolved"
    ]

    total_vote_days = len(vote_events)
    exile_days = sum(
        1 for e in vote_events
        if e.get("payload", {}).get("exiled") is not None
    )
    second_tie_count = sum(
        1 for e in vote_events
        if e.get("payload", {}).get("reason") == "second_tie_no_exile"
    )

    day_exile_rate = exile_days / total_vote_days if total_vote_days > 0 else 0.0

    # Consecutive no-exile streak
    max_streak = 0
    current_streak = 0
    for e in vote_events:
        if e.get("payload", {}).get("exiled") is None:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    # Stale vote reuse: check if exile_votes on different days are identical
    stale_count = 0
    seen_votes: list[dict] = []
    for e in events:
        if e.get("type") == "vote_resolved":
            votes_snapshot = e.get("payload", {}).get("votes", {})
            if votes_snapshot:
                for prev in seen_votes:
                    if votes_snapshot == prev:
                        stale_count += 1
                        break
                seen_votes.append(votes_snapshot)

    # Pace target
    pace_target_met = (
        (finish_night is not None and finish_night <= 8)
        and max_streak <= 1
        and stale_count == 0
        and (total_vote_days < 3 or day_exile_rate >= 0.5)
    )

    return {
        "day_exile_rate": round(day_exile_rate, 3),
        "max_consecutive_no_exile_days": max_streak,
        "second_tie_count": second_tie_count,
        "stale_vote_reuse_count": stale_count,
        "finish_night_number": finish_night,
        "pace_target_met": pace_target_met,
    }


def _collect_world_model_audit_samples(
    audit: dict[str, Any],
    *,
    player_roles: dict[str, str],
    belief_scores: list[float],
    possible_world_hits: list[bool],
    simulation_hits: list[bool],
    decision_legal: list[bool],
    dialogue_leaks: list[bool],
) -> None:
    for sample in audit.get("belief_calibration_samples", []) or []:
        if not isinstance(sample, dict):
            continue
        predicted = _bounded_float(sample.get("predicted"))
        actual = 1.0 if bool(sample.get("actual")) else 0.0
        belief_scores.append(1.0 - abs(predicted - actual))
    belief_scores.extend(_belief_scores_from_audit(audit, player_roles))

    possible_world_hits.extend(
        bool(item.get("hit"))
        for item in audit.get("possible_world_checks", []) or []
        if isinstance(item, dict)
    )
    world_hit = _possible_world_hit_from_audit(audit, player_roles)
    if world_hit is not None:
        possible_world_hits.append(world_hit)

    simulation_hits.extend(
        bool(item.get("hit"))
        for item in audit.get("simulation_checks", []) or []
        if isinstance(item, dict)
    )
    decision_legal.extend(
        bool(item.get("legal"))
        for item in audit.get("decision_legality_checks", []) or []
        if isinstance(item, dict)
    )
    dialogue_leaks.extend(
        bool(item.get("leaked"))
        for item in audit.get("dialogue_leak_checks", []) or []
        if isinstance(item, dict)
    )


def _belief_scores_from_audit(
    audit: dict[str, Any],
    player_roles: dict[str, str],
) -> list[float]:
    belief = audit.get("belief")
    if not isinstance(belief, dict) or not player_roles:
        return []
    scores: list[float] = []
    for group in ("my_suspects", "my_trusted"):
        for item in belief.get(group, []) or []:
            if not isinstance(item, dict):
                continue
            player_id = str(item.get("player") or "")
            guessed_role = _normalize_role(item.get("top_role_guess"))
            if not player_id or not guessed_role or player_id not in player_roles:
                continue
            predicted = _bounded_float(item.get("top_role_prob"))
            actual = 1.0 if _normalize_role(player_roles[player_id]) == guessed_role else 0.0
            scores.append(1.0 - abs(predicted - actual))
    for player_id, role_probs in belief.items():
        if player_id in {"my_suspects", "my_trusted"}:
            continue
        if not isinstance(role_probs, dict) or player_id not in player_roles:
            continue
        for role, predicted in role_probs.items():
            normalized = _normalize_role(role)
            if not normalized:
                continue
            actual = 1.0 if _normalize_role(player_roles[player_id]) == normalized else 0.0
            scores.append(1.0 - abs(_bounded_float(predicted) - actual))
    return scores


def _possible_world_hit_from_audit(
    audit: dict[str, Any],
    player_roles: dict[str, str],
) -> bool | None:
    possible_worlds = audit.get("possible_worlds")
    if isinstance(possible_worlds, dict):
        worlds = possible_worlds.get("top_worlds")
    else:
        worlds = possible_worlds
    if not isinstance(worlds, list) or not player_roles:
        return None
    saw_assignments = False
    for world in worlds:
        if not isinstance(world, dict):
            continue
        assignments = world.get("key_assignments")
        if not isinstance(assignments, dict) or not assignments:
            continue
        comparable = {
            str(pid): _normalize_role(role)
            for pid, role in assignments.items()
            if str(pid) in player_roles
        }
        if not comparable:
            continue
        saw_assignments = True
        if all(_normalize_role(player_roles[pid]) == role for pid, role in comparable.items()):
            return True
    return False if saw_assignments else None


def _action_trace_from_event(event: Any) -> dict[str, Any] | None:
    if isinstance(event, dict):
        event_type = event.get("type")
        payload = event.get("payload") or {}
    else:
        event_type = getattr(event, "type", None)
        payload = getattr(event, "payload", {}) or {}
    if event_type != "action_trace_audit" or not isinstance(payload, dict):
        return None
    trace = payload.get("action_trace")
    return trace if isinstance(trace, dict) else None


def _normalize_role(value: Any) -> str:
    role = str(value or "").strip().lower()
    if role in {"wolf", "werewolves"}:
        return "werewolf"
    return role


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _bool_rate(values: list[bool]) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


def _rate_from_counts(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _bounded_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, parsed))
