# -*- coding: utf-8 -*-
"""
输出真实游戏运行后的控制台摘要、节奏、质量和泄漏检查报告。

作者: Project contributors
创建日期: 2026-07-08
修改日期: 2026-07-25

使用示例:
    >>> from scripts.run_real_game_reports import print_quality_audit
    >>> print_quality_audit(runner)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from werewolf_agent.core.event_visibility import EventVisibility, event_visibility
from werewolf_agent.runtime.decision_outcomes import summarize_attempt_counts
from werewolf_agent.runtime.vote_display import (
    VotePayloadError,
    decode_vote_resolved_payload,
    format_vote_count,
)


def reflection_verification_metrics(game_state: Any) -> dict[str, int]:
    """对结构化复盘草稿复用终局事实门，并分别累计事实与经验拒绝数。"""
    from werewolf_agent.runtime.reflection_events import canonical_verified_reflections

    rejected_facts = 0
    rejected_lessons = 0
    for verification in canonical_verified_reflections(game_state.events).values():
        rejected_facts += int(verification.get("rejected_fact_count") or 0)
        rejected_lessons += int(verification.get("rejected_lesson_count") or 0)
    return {
        "reflection_rejected_fact_count": rejected_facts,
        "reflection_rejected_lesson_count": rejected_lessons,
    }


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
        visibility = event_visibility(event).value
        tag = event.type
        if event.payload:
            if "speech" in event.type:
                tag = f"[{event.payload.get('speaker', '?')}] {event.payload.get('text', '')[:80]}"
            elif event.type == "vote_resolved":
                tag = (
                    f"exiled={event.payload.get('exiled')} "
                    f"reason={event.payload.get('reason')} "
                    f"tally={_render_vote_tally(event.payload)}"
                )
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


def _render_vote_tally(payload: Mapping[str, Any]) -> str:
    """渲染真实票数；历史 V1 缺少基数时不猜测内部单位。"""
    try:
        decoded = decode_vote_resolved_payload(payload)
    except VotePayloadError:
        return "[unsupported vote payload]"
    if not decoded.display_supported:
        return (
            "[unsupported legacy vote units: "
            f"{decoded.unsupported_reason}]"
        )
    return "、".join(
        f"{player_id}={format_vote_count(value)}票"
        for player_id, value in sorted(
            (decoded.weighted_tally_display or {}).items(),
            key=lambda item: -item[1],
        )
    ) or "(无有效票)"


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
    # 2026-07-21 R7: prompt cache 累计写入 sink.
    cache_creation_tokens = sum(usage.cache_creation_input_tokens for usage in usage_log)
    cache_read_tokens = sum(usage.cache_read_input_tokens for usage in usage_log)
    latency_ms = sum(usage.latency_ms for usage in usage_log)

    print(f"  Calls: {total}  (ok {ok}, fail {fail})")
    print(
        f"  Tokens: {prompt_tokens:,} prompt + {completion_tokens:,} completion = "
        f"{prompt_tokens + completion_tokens:,}"
    )
    # R7: cache_* 写入 sink. 真实数字由 UsageRecord 透传.
    # cache_creation = 首次写入 prefix (Anthropic / MiniMax, 走 1.25x 计费).
    # cache_read = 复用 prefix (Anthropic / MiniMax 0.1x, OpenAI / GLM auto-cache 0.5x).
    print(
        f"  Cache:  {cache_creation_tokens:,} creation + "
        f"{cache_read_tokens:,} read = "
        f"{cache_creation_tokens + cache_read_tokens:,} total"
    )
    # cache_hit_ratio = cache_read / (prompt + cache_read); cache_creation 不计入
    # 因为首次写入不属于"命中".
    cache_hit_ratio = (
        cache_read_tokens / (prompt_tokens + cache_read_tokens)
        if (prompt_tokens + cache_read_tokens) > 0
        else 0.0
    )
    print(f"  Cache hit ratio: {cache_hit_ratio:.1%}")
    print(f"  Latency: {latency_ms / 1000:.1f}s total")
    action_attempts = [
        attempt
        for event in runner.state.events
        if event.type == "action_trace_audit"
        for attempt in (event.payload.get("action_trace") or {}).get(
            "execution_attempts", ()
        )
    ]
    print(
        "Runtime timeouts: "
        f"{summarize_attempt_counts(action_attempts).runtime_timeout_count}"
    )
    reasoning = _reasoning_evidence_summary(
        usage_log,
        action_attempts=action_attempts,
    )
    print(
        "  Reasoning confirmed: "
        f"{reasoning['confirmed_numerator']}/{reasoning['requested_denominator']}"
    )
    for attempt in reasoning["attempts"]:
        print(
            "    attempt "
            f"{attempt['opaque_request_id']}#{attempt['ordinal']} "
            f"{attempt['provider']}/{attempt['model']} "
            f"provider_attempted={'true' if attempt['provider_attempted'] else 'false'} "
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


def _reasoning_evidence_summary(
    usage_log: list[Any],
    *,
    action_attempts: tuple[Any, ...] | list[Any] = (),
) -> dict[str, Any]:
    """按请求与序号去重累计快照，并让最终行动投影覆盖 provider 快照。"""

    def field(attempt: Any, name: str) -> Any:
        value = (
            attempt.get(name)
            if isinstance(attempt, Mapping)
            else getattr(attempt, name)
        )
        if name == "opaque_request_id" and isinstance(value, dict):
            value = value.get("value", "")
        return getattr(value, "value", value)

    def provider_attempted(attempt: Any) -> bool:
        """兼容旧记录缺省值，同时拒绝宽松真假值。"""
        missing = object()
        value = (
            attempt.get("provider_attempted", missing)
            if isinstance(attempt, Mapping)
            else getattr(attempt, "provider_attempted", missing)
        )
        if value is missing:
            return True
        if type(value) is not bool:
            raise TypeError("provider_attempted must be a bool")
        return value

    canonical: dict[tuple[str, int], Any] = {}
    request_order: dict[str, int] = {}
    for usage in usage_log:
        for attempt in usage.attempts:
            opaque_request_id = field(attempt, "opaque_request_id")
            request_order.setdefault(opaque_request_id, len(request_order))
            key = (opaque_request_id, field(attempt, "ordinal"))
            canonical[key] = attempt
    # ActionTrace 包含 parser/validator 完成后的最终投影；同键冲突时，
    # 它必须覆盖较早的 provider usage 快照。
    for attempt in action_attempts:
        opaque_request_id = field(attempt, "opaque_request_id")
        request_order.setdefault(opaque_request_id, len(request_order))
        key = (opaque_request_id, field(attempt, "ordinal"))
        canonical[key] = attempt
    attempts = sorted(
        canonical.values(),
        key=lambda item: (
            request_order[field(item, "opaque_request_id")],
            field(item, "ordinal"),
        ),
    )
    requested = [
        attempt for attempt in attempts
        if field(attempt, "requested_reasoning_level") != "none"
    ]
    confirmed = [
        attempt for attempt in requested
        if field(attempt, "normalized_reasoning_status") == "confirmed"
    ]
    return {
        "requested_denominator": len(requested),
        "confirmed_numerator": len(confirmed),
        "support_flags": {
            "reasoning_token_evidence": any(
                field(attempt, "evidence_kind") == "token_count" for attempt in attempts
            ),
            "provider_status_evidence": any(
                field(attempt, "evidence_kind") == "authoritative_provider_execution"
                for attempt in attempts
            ),
        },
        "attempts": [
            {
                "opaque_request_id": field(attempt, "opaque_request_id"),
                "ordinal": field(attempt, "ordinal"),
                "provider": field(attempt, "provider"),
                "model": field(attempt, "model"),
                "requested_level": field(attempt, "requested_reasoning_level"),
                "status": field(attempt, "normalized_reasoning_status"),
                "reasoning_tokens": field(attempt, "reasoning_token_count"),
                "evidence": field(attempt, "evidence_kind"),
                "route": field(attempt, "route_kind"),
                "root_cause": field(attempt, "root_cause"),
                "outcome": field(attempt, "attempt_outcome"),
                "provider_attempted": provider_attempted(attempt),
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
        visibility = event_visibility(event)
        if visibility is not EventVisibility.PUBLIC:
            continue
        if "wolf_kill" in event.type:
            if "wolf_kill_target_id" in event.payload or "target_id" in event.payload:
                if "wolf_kill" in event.type:
                    pass
        if event.type == "seer_check":
            leaks.append(f"Seer check leaked: vis={visibility.value}")

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
