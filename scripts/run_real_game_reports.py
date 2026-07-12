# -*- coding: utf-8 -*-
"""
输出真实游戏运行后的控制台摘要、节奏、质量和泄漏检查报告。

作者: Project contributors
创建日期: 2026-07-08
修改日期: 2026-07-13

使用示例:
    >>> from scripts.run_real_game_reports import print_quality_audit
    >>> print_quality_audit(runner)
"""

from __future__ import annotations

from typing import Any


def _sep(title: str = "") -> None:
    line = "=" * 60
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


def _role_of(runner: Any, pid: str) -> str:
    player = runner.state.players.get(pid)
    return player.role if player else "?"


def print_game_summary(runner: Any) -> None:
    gs = runner.state
    _sep("GAME SUMMARY")
    print(f"  Game ID:      {gs.game_id}")
    print(f"  Ruleset:      {gs.ruleset_id}")
    print(f"  Phase:        {gs.phase}")
    print(f"  Winner:       {gs.winning_faction or 'N/A'}")
    print(f"  Day / Night:  {gs.day_number} / {gs.night_number}")
    print(f"  Steps:        {runner.step_count}")
    print(f"  Deaths:       {len(gs.deaths)}")

    print("\n  Players:")
    for pid, player in sorted(gs.players.items()):
        status = "ALIVE" if player.alive else "DEAD "
        print(f"    {pid}: {player.role:12s} [{status}]")

    print("\n  Death Log:")
    for death in gs.deaths:
        print(f"    {death.player_id} ({_role_of(runner, death.player_id)}): {death.reason}")

    if gs.sheriff_id:
        print(f"\n  Sheriff:        {gs.sheriff_id}")
    print(f"  Badge state:    {gs.sheriff_badge_state}")
    print(f"  Antidote used:  {gs.antidote_used}")
    print(f"  Poison used:    {gs.poison_used}")

    print(f"\n  Events ({len(gs.events)} total, last 30):")
    for event in gs.events[-30:]:
        visibility = event.payload.get("visibility", "public") if event.payload else ""
        tag = event.type
        if event.payload:
            if "speech" in event.type:
                tag = f"[{event.payload.get('speaker', '?')}] {event.payload.get('text', '')[:80]}"
            elif event.type == "vote_resolved":
                tag = f"exiled={event.payload.get('exiled')} reason={event.payload.get('reason')}"
            elif "wolf_kill" in event.type:
                tag = f"target={event.payload.get('target_id', '?')}"
            elif event.type == "seer_check":
                tag = (
                    f"{event.payload.get('seer_id')} -> "
                    f"{event.payload.get('target_id')}: {event.payload.get('alignment')}"
                )
            elif "witch" in event.type or "badge" in event.type or "hybrid" in event.type:
                tag = str(event.payload)[:100]
        if visibility and visibility != "public":
            tag += f" [{visibility}]"
        print(f"    {event.type:30s} {tag}")
    _sep()


def print_usage_stats(runner: Any) -> None:
    registry = runner._agent_registry
    if registry is None:
        return
    sample_agent = next(iter(registry._agents.values()), None)
    if sample_agent is None:
        return
    usage_log = sample_agent.model_router.get_usage_log()
    if not usage_log:
        return

    _sep("API USAGE")
    total = len(usage_log)
    ok = sum(1 for usage in usage_log if usage.success)
    fail = total - ok
    prompt_tokens = sum(usage.prompt_tokens for usage in usage_log)
    completion_tokens = sum(usage.completion_tokens for usage in usage_log)
    latency_ms = sum(usage.latency_ms for usage in usage_log)

    print(f"  Calls: {total}  (ok {ok}, fail {fail})")
    print(
        f"  Tokens: {prompt_tokens:,} prompt + {completion_tokens:,} completion = "
        f"{prompt_tokens + completion_tokens:,}"
    )
    print(f"  Latency: {latency_ms / 1000:.1f}s total")
    reasoning = _reasoning_evidence_summary(usage_log)
    print(
        "  Reasoning confirmed: "
        f"{reasoning['confirmed_numerator']}/{reasoning['requested_denominator']}"
    )
    for attempt in reasoning["attempts"]:
        print(
            "    attempt "
            f"{attempt['opaque_request_id']}#{attempt['ordinal']} "
            f"{attempt['provider']}/{attempt['model']} "
            f"level={attempt['requested_level']} status={attempt['status']} "
            f"tokens={attempt['reasoning_tokens']} evidence={attempt['evidence']} "
            f"route={attempt['route']} root={attempt['root_cause']} "
            f"outcome={attempt['outcome']}"
        )

    stats: dict[str, dict[str, int]] = {}
    for usage in usage_log:
        stat = stats.setdefault(usage.agent_id, {"calls": 0, "tokens": 0, "lat": 0})
        stat["calls"] += 1
        stat["tokens"] += usage.prompt_tokens + usage.completion_tokens
        stat["lat"] += usage.latency_ms
    print("\n  Per-agent:")
    for agent_id, stat in sorted(stats.items()):
        print(f"    {agent_id}: {stat['calls']} calls, {stat['tokens']:,} tok, {stat['lat'] / 1000:.1f}s")

    failures = {}
    for usage in usage_log:
        if not usage.success:
            reason = usage.fallback_reason or "unknown"
            failures[reason] = failures.get(reason, 0) + 1
    if failures:
        print("\n  Failures:")
        for reason, count in sorted(failures.items(), key=lambda item: -item[1]):
            print(f"    {count}x {reason}")
    _sep()


def _reasoning_evidence_summary(usage_log: list[Any]) -> dict[str, Any]:
    """仅投影允许公开的强类型执行字段，并给出精确支持分母。"""
    attempts = [attempt for usage in usage_log for attempt in usage.attempts]
    requested = [
        attempt for attempt in attempts
        if attempt.requested_reasoning_level.value != "none"
    ]
    confirmed = [
        attempt for attempt in requested
        if attempt.normalized_reasoning_status.value == "confirmed"
    ]
    return {
        "requested_denominator": len(requested),
        "confirmed_numerator": len(confirmed),
        "support_flags": {
            "reasoning_token_evidence": any(
                attempt.evidence_kind.value == "token_count" for attempt in attempts
            ),
            "provider_status_evidence": any(
                attempt.evidence_kind.value == "authoritative_provider_execution"
                for attempt in attempts
            ),
        },
        "attempts": [
            {
                "opaque_request_id": attempt.opaque_request_id.value,
                "ordinal": attempt.ordinal,
                "provider": attempt.provider,
                "model": attempt.model,
                "requested_level": attempt.requested_reasoning_level.value,
                "status": attempt.normalized_reasoning_status.value,
                "reasoning_tokens": attempt.reasoning_token_count,
                "evidence": attempt.evidence_kind.value,
                "route": attempt.route_kind.value,
                "root_cause": attempt.root_cause.value,
                "outcome": attempt.attempt_outcome.value,
            }
            for attempt in attempts
        ],
    }


def print_pace_report(runner: Any) -> None:
    from werewolf_agent.evaluation.metrics import compute_pace_metrics

    gs = runner.state
    _sep("PACE REPORT")
    events = [{"type": event.type, "payload": event.payload} for event in gs.events]
    deaths = [{"player_id": death.player_id, "reason": death.reason} for death in gs.deaths]
    metrics = compute_pace_metrics(events, deaths=deaths, finish_night=gs.night_number)

    wolf_kills = sum(1 for death in gs.deaths if death.reason == "wolf_kill")
    exiles = sum(1 for death in gs.deaths if death.reason == "exile")
    shots = sum(1 for death in gs.deaths if death.reason == "hunter_shot")

    print(f"  Winner:          {gs.winning_faction or 'N/A'}")
    print(f"  Finish:          day {gs.day_number} / night {gs.night_number}")
    print(f"  Deaths:          {len(gs.deaths)} (wolf {wolf_kills}, exile {exiles}, shot {shots})")
    print(f"  Day exile rate:  {metrics['day_exile_rate']:.1%}")
    print(f"  2nd-tie count:   {metrics['second_tie_count']}")
    print(f"  Max no-exile:    {metrics['max_consecutive_no_exile_days']}")
    print(f"  Stale votes:     {metrics['stale_vote_reuse_count']}")
    print(f"  Pace OK:         {'YES' if metrics['pace_target_met'] else 'NO'}")
    _sep()


def print_quality_audit(runner: Any) -> None:
    """按运行事件输出质量审计摘要。"""
    from werewolf_agent.runtime.speech_quality import validate_public_speech

    gs = runner.state
    _sep("QUALITY AUDIT")

    speeches = [event for event in gs.events if event.type == "speech"]
    filler_count = 0
    short_count = 0
    for event in speeches:
        text = event.payload.get("text", "")
        if not text or len(text) < 10:
            short_count += 1
            continue
        result = validate_public_speech(text, phase="day_discussion")
        if not result["valid"]:
            filler_count += 1

    total_speeches = len(speeches)
    ok_speeches = total_speeches - filler_count - short_count
    print(f"  Speeches: {total_speeches} total, {ok_speeches} ok, {filler_count} filler, {short_count} empty")

    votes_with_basis = 0
    votes_without_basis = 0
    for event in gs.events:
        if event.type == "action_trace_audit" and event.payload.get("phase") == "vote":
            trace = event.payload.get("action_trace", {})
            parsed_action = trace.get("parsed_action") or {}
            reason = parsed_action.get("reason", "")
            if reason:
                from werewolf_agent.runtime.vote_quality import extract_vote_basis

                bases = extract_vote_basis(reason)
                if bases:
                    votes_with_basis += 1
                else:
                    votes_without_basis += 1

    print(f"  Votes with basis:    {votes_with_basis}")
    print(f"  Votes without basis: {votes_without_basis}")

    broadcasts = [event for event in gs.events if event.type == "judge_broadcast"]
    phases = {event.payload.get("phase") for event in broadcasts}
    expected = {
        "enter_night",
        "day_announce",
        "wolf_discussion_start",
        "wolf_kill_choice",
        "seer_wake",
        "witch_wake",
        "vote_start",
    }
    missing_broadcasts = expected - phases
    print(f"  Judge broadcasts:    {len(broadcasts)} ({len(phases)} unique phases)")
    if missing_broadcasts:
        print(f"  MISSING broadcasts:  {missing_broadcasts}")

    wolf_discussions = [event for event in gs.events if event.type == "wolf_discussion"]
    silent = sum(1 for event in wolf_discussions if not event.payload.get("text", "").strip())
    print(f"  Wolf discussion:     {len(wolf_discussions)} rounds, {silent} silent")

    traces = [event for event in gs.events if event.type == "action_trace_audit"]
    action_fallback_count = sum(
        1 for event in traces
        if event.payload.get("action_trace", {}).get("fallback_reason")
    )
    wolf_plan_fallback_count = sum(1 for event in gs.events if event.type == "wolf_team_plan_fallback")
    structured_fail = sum(
        1 for event in traces
        if event.payload.get("action_trace", {}).get("structured_failure_reason")
    )
    print(f"  Action traces:       {len(traces)}")
    print(f"  Fallbacks:           {action_fallback_count}")
    print(f"  Wolf plan fallbacks: {wolf_plan_fallback_count}")
    print(f"  Structured failures: {structured_fail}")

    try:
        from werewolf_agent.cognition.contradiction import ContradictionEngine
        from werewolf_agent.cognition.world_state import build_world_state

        world_state = build_world_state(gs)
        engine = ContradictionEngine()
        alerts = engine.detect(world_state.facts, gs.day_number)
        high_alerts = [alert for alert in alerts if alert.priority == "high"]
        print(f"  Contradiction alerts: {len(alerts)} ({len(high_alerts)} high)")
        for alert in high_alerts[:5]:
            print(f"    HIGH: {alert.player_id} {alert.alert_type}: {alert.description}")
    except Exception as exc:
        print(f"  Contradiction check: skipped ({exc})")

    _sep()


def check_leakage(runner: Any) -> None:
    gs = runner.state
    _sep("LEAKAGE CHECK")
    leaks = []
    for event in gs.events:
        visibility = event.payload.get("visibility", "public") if event.payload else ""
        if visibility in ("moderator_only", "seer_only", "witch_private", "werewolf_team_only"):
            continue
        if "wolf_kill" in event.type and visibility == "public":
            if "wolf_kill_target_id" in event.payload or "target_id" in event.payload:
                if "wolf_kill" in event.type:
                    pass
        if event.type == "seer_check" and visibility not in ("seer_only", "moderator_only"):
            leaks.append(f"Seer check leaked: vis={visibility}")

    if leaks:
        print("  POTENTIAL LEAKS:")
        for leak in leaks:
            print(f"    - {leak}")
    else:
        print("  No public-state information leaks detected.")
    _sep()


__all__ = [
    "_sep",
    "check_leakage",
    "print_game_summary",
    "print_pace_report",
    "print_quality_audit",
    "print_usage_stats",
]
