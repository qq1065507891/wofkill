"""Render a saved game JSON as a detailed audit report.

The report expands player action traces so a human can inspect what the model
actually returned, which action was used, and whether fallback logic intervened.

Includes structured sections for:
  - Judge timeline broadcasts
  - Public speeches
  - Wolf private chat
  - Wolf plan evidence
  - Votes and vote basis
  - Witch/seer/hunter private actions
  - Fallback/retry summary
"""

from __future__ import annotations

import argparse
import json
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
