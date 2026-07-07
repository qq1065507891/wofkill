# -*- coding: utf-8 -*-
"""
将保存的游戏 JSON 渲染为便于人工检查的详细审计报告。

作者: Project contributors
修改日期: 2026-07-07

使用示例:
    python scripts/print_game_audit.py game.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


JUDGE_EVENT_TYPES = {
    "roles_assigned",
    "enter_night",
    "day_announce",
    "night_death_last_words",
    "vote_resolved",
    "player_died",
    "player_exiled",
    "wolf_kill_selected",
    "wolf_no_kill_timeout",
    "wolf_no_kill_declared",
    "victory_checked",
    "victory",
    "sheriff_registered",
    "sheriff_speech",
    "sheriff_withdraw",
    "sheriff_no_election",
    "hunter_idiot_status_confirmed",
    "hybrid_master_chosen",
    "judge_broadcast",
}


def load_game(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def print_section(title: str, items: list[dict], fields: list[str] | None = None):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    if not items:
        print("  (无)")
        return
    for i, item in enumerate(items, 1):
        if fields:
            payload = item.get("payload", item)
            line = " | ".join(str(payload.get(f, "")) for f in fields)
        else:
            line = str(item)
        print(f"  {i}. {line}")


def audit_game(data: dict):
    """Print a structured audit report to stdout."""
    events = data.get("events", [])

    # Separate by type
    judge_broadcasts = [e for e in events if e.get("type") == "judge_broadcast"]
    public_speeches = [e for e in events if e.get("type") == "speech" and e.get("payload", {}).get("visibility", "public") != "werewolf_team_only"]
    wolf_chat = [e for e in events if e.get("type") == "wolf_discussion"]
    wolf_plans = [e for e in events if e.get("type") == "wolf_team_plan"]
    votes = [e for e in events if e.get("type") in ("vote_resolved", "vote")]
    private_actions = [e for e in events if e.get("payload", {}).get("visibility") in ("moderator_only", "witch_private", "seer_only")]
    traces = [e for e in events if e.get("type") == "action_trace_audit"]
    fallbacks = [e for e in events if "fallback" in str(e.get("payload", {}).get("action_trace", ""))]

    print_section("法官时间线", judge_broadcasts, ["phase", "message"])
    print_section("公开发言", public_speeches, ["speaker", "text"])
    print_section("狼人密谈", wolf_chat, ["wolf_id", "round", "text"])
    print_section("狼队计划", wolf_plans)
    print_section("投票记录", votes, ["exiled", "reason"])
    print_section("私有行动", private_actions)
    print_section("行动审计", traces, ["player_id", "phase"])
    print_section("回退/重试", fallbacks)

    print(f"\n{'='*60}")
    print(f"  总计: {len(events)} 个事件")
    print(f"  公开: {len(public_speeches)} | 狼人: {len(wolf_chat)} | 投票: {len(votes)} | 审计: {len(traces)}")
    print(f"{'='*60}")


def find_boundary_violations(game: dict[str, Any]) -> list[dict[str, Any]]:
    """Return machine-checkable game-record boundary violations."""
    violations: list[dict[str, Any]] = []
    events = game.get("events") or []
    players = game.get("players") or {}
    player_roles = {
        str(player_id): str((info or {}).get("role") or "")
        for player_id, info in players.items()
        if isinstance(info, dict)
    }

    required_death_fields = {
        "player_id",
        "reason",
        "timing",
        "resolution_batch",
        "source_player_id",
        "can_leave_last_words",
        "triggered_skills",
    }
    for index, death in enumerate(game.get("deaths") or [], start=1):
        if not isinstance(death, dict):
            continue
        missing = sorted(required_death_fields - set(death))
        if missing:
            violations.append({
                "kind": "incomplete_death_export",
                "event_index": index,
                "detail": f"death for {death.get('player_id') or '?'} missing {', '.join(missing)}",
            })

    dead_roles: set[str] = set()
    pending_hunters: dict[str, dict[str, Any]] = {}
    role_wake_phases = {
        "witch": {"witch_wake", "witch_choose"},
        "seer": {"seer_wake", "seer_choose"},
    }

    for event_index, event in enumerate(events, start=1):
        event_type = str(event.get("type") or "")
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}

        if event_type == "player_died":
            player_id = str(payload.get("player_id") or "")
            reason = str(payload.get("reason") or "")
            if player_id:
                role = player_roles.get(player_id, "")
                if role:
                    dead_roles.add(role)
                if "hunter_shot" in (payload.get("triggered_skills") or []):
                    pending_hunters[player_id] = {
                        "death_event_index": event_index,
                        "prompted": False,
                        "resolved": False,
                    }
            if reason == "hunter_shot":
                shooter = str(payload.get("source_player_id") or "")
                if shooter in pending_hunters:
                    pending_hunters[shooter]["resolved"] = True

        if event_type == "judge_broadcast":
            phase = str(payload.get("phase") or "")
            for role, phases in role_wake_phases.items():
                if role in dead_roles and phase in phases:
                    violations.append({
                        "kind": "dead_role_broadcast",
                        "event_index": event_index,
                        "detail": f"{phase} broadcast after {role} died",
                    })

            hunter_id = str(payload.get("hunter_id") or "")
            if phase == "hunter_shot_prompt" and hunter_id in pending_hunters:
                pending_hunters[hunter_id]["prompted"] = True
            if phase in {"hunter_shot_choice", "hunter_shot_decline"} and hunter_id in pending_hunters:
                pending_hunters[hunter_id]["resolved"] = True

        if event_type == "hunter_shot_declined":
            hunter_id = str(payload.get("hunter_id") or "")
            if hunter_id in pending_hunters:
                pending_hunters[hunter_id]["resolved"] = True

        if event_type == "vote_resolved":
            for vote_index, vote in enumerate(payload.get("votes") or [], start=1):
                if not isinstance(vote, dict):
                    continue
                voter = str(vote.get("voter") or vote.get("voter_id") or "")
                target = str(vote.get("target") or vote.get("target_id") or "")
                reason = str(vote.get("reason") or "").strip()
                if voter and target and voter == target:
                    violations.append({
                        "kind": "self_vote",
                        "event_index": event_index,
                        "detail": f"{voter} voted for self at vote #{vote_index}",
                    })
                if target and not reason:
                    violations.append({
                        "kind": "empty_vote_reason",
                        "event_index": event_index,
                        "detail": f"{voter or '?'} voted {target} without reason",
                    })

    for hunter_id, pending in sorted(pending_hunters.items()):
        if not pending["prompted"] or not pending["resolved"]:
            violations.append({
                "kind": "pending_hunter_shot",
                "event_index": pending["death_event_index"],
                "detail": f"{hunter_id} triggered hunter_shot without prompt and resolution",
            })

    return violations


def render_audit_report(game: dict[str, Any]) -> str:
    """Return a Markdown audit report for a saved game log."""
    lines: list[str] = []
    lines.append(f"# Game Audit: {game.get('game_id', '?')}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Winner: {game.get('winning_faction') or 'N/A'}")
    lines.append(f"- Day: {game.get('day_number', '?')}")
    lines.append(f"- Night: {game.get('night_number', '?')}")
    lines.append(f"- Steps: {game.get('steps', '?')}")
    lines.append("")
    lines.append("## Judge / Runtime Events")
    lines.append("")
    lines.append(
        "Note: this runtime path records deterministic judge/runtime events. "
        "It does not persist JudgeAgent LLM raw output unless JudgeAgent is explicitly called."
    )
    lines.append("")

    events = game.get("events") or []
    for index, event in enumerate(events, start=1):
        event_type = event.get("type", "?")
        if event_type not in JUDGE_EVENT_TYPES:
            continue
        lines.append(f"### Event {index}: {event_type}")
        lines.append("")
        lines.append("```json")
        lines.append(_json(event.get("payload") or {}))
        lines.append("```")
        lines.append("")

    anomalies = _find_rule_order_anomalies(events)
    if anomalies:
        lines.append("## Rule-Order Anomalies")
        lines.append("")
        for anomaly in anomalies:
            lines.append(
                f"- Event {anomaly['event_index']}: {anomaly['kind']} - {anomaly['detail']}"
            )
        lines.append("")

    lines.append("## Player Model Outputs")
    lines.append("")
    lines.append(
        "Provider thinking blocks were not persisted in this game log. "
        "Use raw_text, parsed_action, retry, and fallback_reason below for audit."
    )
    lines.append("")

    count = 0
    for index, event in enumerate(events, start=1):
        for actor, trace in _iter_action_traces(event):
            count += 1
            lines.extend(_render_trace(count, index, event.get("type", "?"), actor, trace))
    if count == 0:
        lines.append("_No player action traces found._")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _find_rule_order_anomalies(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anomalies: list[dict[str, Any]] = []
    dead_players: set[str] = set()
    pending_hunter_shooter: str | None = None
    public_speeches: list[tuple[str, str]] = []

    for index, event in enumerate(events, start=1):
        event_type = event.get("type", "?")
        payload = event.get("payload") or {}

        actor = _event_actor(event_type, payload)
        if actor and actor in dead_players and event_type in {"wolf_discussion", "wolf_kill_selected"}:
            anomalies.append({
                "event_index": index,
                "kind": "dead player wolf action",
                "detail": f"{actor} acted in {event_type} after death",
            })

        if pending_hunter_shooter and event_type in {"wolf_discussion", "wolf_kill_selected"}:
            anomalies.append({
                "event_index": index,
                "kind": "hunter_shot death after wolf action",
                "detail": f"{event_type} occurred before {pending_hunter_shooter}'s hunter shot resolved",
            })

        if event_type == "speech":
            speaker = str(payload.get("speaker") or payload.get("player_id") or "")
            text = str(payload.get("text") or "")
            if _unsupported_public_record_role_claim(text, public_speeches):
                anomalies.append({
                    "event_index": index,
                    "kind": "unsupported public-record role claim",
                    "detail": text,
                })
            if speaker and text:
                public_speeches.append((speaker, text))

        if event_type == "player_died":
            player_id = str(payload.get("player_id") or "")
            reason = str(payload.get("reason") or "")
            if reason == "hunter_shot":
                pending_hunter_shooter = None
            if player_id:
                dead_players.add(player_id)
                if "hunter_shot" in (payload.get("triggered_skills") or []):
                    pending_hunter_shooter = player_id

    return anomalies


def _event_actor(event_type: str, payload: dict[str, Any]) -> str:
    if event_type == "wolf_discussion":
        return str(payload.get("wolf_id") or payload.get("player_id") or "")
    if event_type == "wolf_kill_selected":
        return str(payload.get("killer_id") or payload.get("wolf_id") or payload.get("player_id") or "")
    return ""


_PUBLIC_RECORD_ROLE_CLAIM = re.compile(
    r"(p\d{2}).{0,12}(?:声称自己是|说自己是|自称|认|跳)(狼人|预言家|女巫|猎人|白痴|村民|民).{0,16}公开记录"
)

_ROLE_MARKERS = {
    "狼人": ("我是狼人", "认狼", "狼队视角", "我们狼队"),
    "预言家": ("我是预言家", "我跳预言家", "认预言家", "悍跳预言家"),
    "女巫": ("我是女巫", "我认女巫", "跳女巫"),
    "猎人": ("我是猎人", "我认猎人", "跳猎人"),
    "白痴": ("我是白痴", "我认白痴", "跳白痴"),
    "村民": ("我是村民", "我是民", "我认民"),
    "民": ("我是村民", "我是民", "我认民"),
}


def _unsupported_public_record_role_claim(text: str, public_speeches: list[tuple[str, str]]) -> bool:
    for match in _PUBLIC_RECORD_ROLE_CLAIM.finditer(text):
        player_id, role = match.group(1), match.group(2)
        markers = _ROLE_MARKERS.get(role, (role,))
        supported = any(
            speaker == player_id and any(marker in speech for marker in markers)
            for speaker, speech in public_speeches
        )
        if not supported:
            return True
    return False


def _iter_action_traces(event: dict[str, Any]):
    payload = event.get("payload") or {}
    speaker = payload.get("speaker") or payload.get("player_id") or payload.get("voter")
    if isinstance(payload.get("action_trace"), dict):
        yield speaker or "unknown", payload["action_trace"]
    if isinstance(payload.get("action_traces"), dict):
        for actor, trace in sorted(payload["action_traces"].items()):
            if isinstance(trace, dict):
                yield actor, trace
    if isinstance(payload.get("witch_action_trace"), dict):
        yield payload.get("witch_id") or "witch", payload["witch_action_trace"]
    if isinstance(payload.get("seer_action_trace"), dict):
        yield payload.get("seer_id") or "seer", payload["seer_action_trace"]


def _render_trace(
    count: int,
    event_index: int,
    event_type: str,
    actor: str,
    trace: dict[str, Any],
) -> list[str]:
    lines = [
        f"### Action {count}: {actor} @ Event {event_index} ({event_type})",
        "",
        f"- Final action: `{trace.get('final_action_type') or '?'}`",
        f"- Fallback: {trace.get('fallback_reason') or 'no'}",
        "",
    ]
    retry = trace.get("retry")
    if retry:
        lines.append("Retry:")
        lines.append("")
        lines.append("```json")
        lines.append(_json(retry))
        lines.append("```")
        lines.append("")
    parsed = trace.get("parsed_action")
    if parsed:
        lines.append("Parsed action:")
        lines.append("")
        lines.append("```json")
        lines.append(_json(parsed))
        lines.append("```")
        lines.append("")
    lines.append("Raw model text:")
    lines.append("")
    lines.append("```text")
    lines.append(trace.get("raw_text") or "")
    lines.append("```")
    lines.append("")
    return lines


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a saved game audit report")
    parser.add_argument("game_json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--structured", action="store_true",
                        help="Print structured section-based audit to stdout")
    args = parser.parse_args()

    game = json.loads(args.game_json.read_text(encoding="utf-8"))

    if args.structured:
        audit_game(game)
        return

    report = render_audit_report(game)
    if args.output:
        args.output.write_text(report, encoding="utf-8")
        print(f"Audit report saved to: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
