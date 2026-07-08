# -*- coding: utf-8 -*-
"""
计算评估快照中的质量类指标和指标来源说明。

作者: Project contributors
创建日期: 2026-07-08

使用示例:
    >>> from werewolf_agent.evaluation.metric_quality import compute_quality_metrics
    >>> compute_quality_metrics(aggregator, snapshot)
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.evaluation.claim_metrics import _extract_claim_events
from werewolf_agent.evaluation.schemas import (
    MetricProvenance,
    MetricsSnapshot,
    QualityMetrics,
)


def compute_quality_metrics(aggregator: Any, snap: MetricsSnapshot) -> None:
    """计算对局质量、角色表现和话术影响相关指标。"""
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

    for result in aggregator._results:
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

        # 抗推指标：被放逐者中非好人占比越高，说明防推表现越好。
        good_exiled = sum(
            1 for d in result.deaths
            if d.get("reason") == "exile"
            and result.player_factions.get(d.get("player_id", "")) == "good"
        )
        total_exiled = sum(1 for d in result.deaths if d.get("reason") == "exile")
        if total_exiled > 0:
            anti_push_total += total_exiled
            anti_push_count += total_exiled - good_exiled

        # 投票准确率：放逐狼人次数 / 总放逐次数。
        exiled_wolves = sum(
            1 for d in result.deaths
            if d.get("reason") == "exile"
            and result.player_factions.get(d.get("player_id", "")) == "werewolf"
        )
        vote_total += total_exiled
        vote_correct += exiled_wolves

        # 谎言识别：其他玩家是否把伪装身份的狼人识别为狼人。
        for ce in claim_events:
            pid = ce.get("payload", {}).get("player_id", "")
            claimed = ce.get("payload", {}).get("claimed_role", "")
            actual = result.player_roles.get(pid, "")
            if actual == "werewolf" and claimed != actual:
                lie_total += 1
                for viewer_id, cognition in result.cognition_snapshots.items():
                    if viewer_id == pid:
                        continue
                    entries = cognition.get("entries", {})
                    entry = entries.get(pid, {})
                    if entry.get("top_role_guess") == "werewolf":
                        lie_detected += 1
                        break

        # 好人投票立场：好人投票目标是否为狼人。
        for ar in result.action_records:
            if ar.action_type == "vote" and ar.target_id:
                voter_faction = result.player_factions.get(ar.player_id, "")
                target_faction = result.player_factions.get(ar.target_id, "")
                if voter_faction == "good":
                    stance_total += 1
                    if target_faction == "werewolf":
                        stance_correct += 1

        # 身份伪装：狼人是否未被其他玩家的认知快照识别。
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

        # 悍跳收益：狼人悍跳神职且未被放逐并最终获胜。
        for ce in claim_events:
            pid = ce.get("payload", {}).get("player_id", "")
            actual = result.player_roles.get(pid, "")
            claimed = ce.get("payload", {}).get("claimed_role", "")
            if actual == "werewolf" and claimed in ("seer", "witch", "hunter"):
                bold_claim_total += 1
                if pid not in exiled_ids and result.winning_faction == "werewolf":
                    bold_claim_success += 1

        # 混血儿共赢和主选择收益。
        hybrid_id = next(
            (pid for pid, r in result.player_roles.items() if r == "hybrid"),
            None,
        )
        if hybrid_id:
            hybrid_total += 1
            hybrid_faction = result.player_factions.get(hybrid_id, "")
            if hybrid_faction == result.winning_faction:
                hybrid_co_wins += 1

        # 女巫药水收益：救好人或毒狼人算作有益。
        for pe in potion_events:
            target_id = pe.get("payload", {}).get("target_id", "")
            target_faction = result.player_factions.get(target_id, "")
            potion_type = pe.get("type")
            potion_total += 1
            if potion_type in ("antidote_used", "witch_antidote_used") and target_faction == "good":
                potion_beneficial += 1
            elif potion_type in ("poison_used", "witch_poison_used") and target_faction == "werewolf":
                potion_beneficial += 1

        # 预言家警徽流质量：查杀是否推动正确放逐。
        for se in seer_check_events:
            target_id = se.get("payload", {}).get("target_id", "")
            alignment = se.get("payload", {}).get("alignment", "")
            seer_checks_total += 1
            if alignment in ("wolf", "werewolf") and target_id in exiled_ids:
                seer_checks_correct += 1

        # 狼队共识质量：刀口是否命中强神职。
        power_roles = {"seer", "witch", "hunter"}
        for we in wolf_kill_events:
            target_id = we.get("payload", {}).get("target_id", "")
            target_role = result.player_roles.get(target_id, "")
            wolf_kill_total += 1
            if target_role in power_roles:
                wolf_strategic_kills += 1

        # 警徽决策质量：警徽流向是否有利于获胜阵营。
        for be in badge_events:
            badge_decisions_recorded += 1
            payload = be.get("payload", {})
            to_id = payload.get("to_id") or payload.get("new_sheriff_id", "")
            if to_id:
                to_faction = result.player_factions.get(to_id, "")
                if to_faction == result.winning_faction:
                    badge_beneficial += 1

        # 发言影响：同一天后续投票是否跟随发言者点名目标。
        speech_events = [
            e for e in result.event_log if e.get("type") == "speech"
        ]
        for speech in speech_events:
            speaker = speech.get("player_id", "")
            day = speech.get("day_number", 0)
            targets = speech.get("mentioned_targets", [])
            if not targets:
                continue
            for vote in result.event_log:
                if (
                    vote.get("type") == "vote"
                    and vote.get("day_number") == day
                    and vote.get("player_id", "") != speaker
                ):
                    speech_influence_total += 1
                    if vote.get("target_id", "") in targets:
                        speech_influence_aligned += 1

        # 多轮发言利用：同日非首个发言占比，用来衡量讨论深度。
        for idx, speech in enumerate(speech_events):
            speech_order_total += 1
            day = speech.get("day_number", 0)
            for prev in speech_events[:idx]:
                if prev.get("day_number") == day:
                    speech_order_with_ref += 1
                    break

        # 认知压缩：压缩事实数 / 原始事实数。
        for _pid, cognition in result.cognition_snapshots.items():
            orig = cognition.get("original_fact_count", 0)
            comp = cognition.get("compressed_fact_count", 0)
            if orig > 0:
                compression_sum += comp / orig
                compression_count += 1

        # 复盘矛盾：记录矛盾提示是否被采纳。
        for review in result.reviews:
            alerts = review.get("contradiction_alerts", [])
            adopted = review.get("contradiction_adopted", [])
            if alerts:
                contradiction_total += len(alerts)
                contradiction_hits += len(adopted)
                contradiction_adopted_total += len(alerts)
                contradiction_adopted += len(adopted)
                games_with_contradictions += 1

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
        q.contradiction_hit_rate = games_with_contradictions / max(1, len(aggregator._results))
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

    deep_hook_wins = sum(
        1 for r in aggregator._results
        if r.winning_faction == "werewolf"
        and any(rev.get("strategy", "") == "deep_hook" for rev in r.reviews)
    )
    wolf_total = sum(1 for r in aggregator._results if r.winning_faction == "werewolf")
    if wolf_total:
        q.deep_hook_benefit = deep_hook_wins / wolf_total

    snap.quality_metrics = q

    game_ids = [r.game_id for r in aggregator._results]

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
          ["deaths", "player_factions"], vote_total if vote_total > 0 else len(aggregator._results))
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


__all__ = ["compute_quality_metrics"]
