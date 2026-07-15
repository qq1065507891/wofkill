# -*- coding: utf-8 -*-
"""
为玩家上下文构建公开时间线摘要和近期发言记录。

作者: Mike
创建日期: 2025-01-15
修改日期: 2026-07-15

使用示例:
    >>> from werewolf_agent.runtime.context_public_summary import build_public_summary
    >>> build_public_summary(game_state)
"""

from __future__ import annotations

import re
from typing import Any

from werewolf_agent.core.event_visibility import EventVisibility, event_visibility
from werewolf_agent.core.models import GameState
from werewolf_agent.runtime.timeline import (
    TIMELINE_ORDER_NOTE,
    current_phase_label,
    phase_label,
)


def build_recent_transcript(gs: GameState) -> list[dict[str, Any]]:
    """Build the compact recent speech/vote transcript for prompts."""
    transcript: list[dict[str, Any]] = []
    for event in reversed(gs.events):
        if event_visibility(event) is not EventVisibility.PUBLIC:
            continue
        if event.type in ("speech", "sheriff_speech"):
            if len(transcript) < 10:
                transcript.insert(0, {
                    "speaker": event.payload.get("speaker", ""),
                    "text": event.payload.get("text", ""),
                    "type": event.type,
                })
        elif event.type == "vote_resolved" and len(transcript) < 12:
            votes_detail = event.payload.get("votes", [])
            if votes_detail:
                voter_lines = {
                    vote.get("voter", "?"): vote.get("target", "弃票")
                    for vote in votes_detail
                }
                transcript.insert(0, {
                    "type": "vote_record",
                    "day": event.payload.get("day_number", "?"),
                    "result": event.payload.get("exiled") or "无人出局",
                    "votes": voter_lines,
                })
    return transcript


def build_public_summary(gs: GameState) -> str:
    """Build the public timeline summary rendered into player prompts."""
    summary_items: list[tuple[int, str]] = []

    for event in gs.events:
        if event_visibility(event) is not EventVisibility.PUBLIC:
            continue
        if event.type == "day_announce":
            day = event.payload.get("day", "?")
            try:
                day_label = phase_label("day", int(day))
            except (TypeError, ValueError):
                day_label = f"D{day}"
            summary_items.append((3, f"\n===== {day_label} ====="))

        elif event.type == "judge_broadcast":
            phase = event.payload.get("phase", "")
            msg = event.payload.get("message", "")
            if phase == "death_announce":
                summary_items.append((1, f"[死讯] {msg}"))
            elif phase == "exile":
                summary_items.append((1, f"[法官] {msg}"))
            elif phase == "sheriff_elected":
                summary_items.append((1, f"[警长] {msg}"))
            elif phase == "sheriff_registered":
                summary_items.append((1, f"[上警] {msg}"))
            elif phase in ("vote_tie_pk", "vote_second_tie"):
                summary_items.append((2, f"[法官] {msg}"))
            elif phase == "sheriff_no_election":
                summary_items.append((2, f"[警长] {msg}"))

        elif event.type == "vote_resolved":
            _append_vote_resolution(summary_items, event.payload)

        elif event.type == "idiot_revealed":
            summary_items.append((1, f"[白痴] {event.payload.get('player_id', '?')} 亮牌"))

        elif event.type == "hunter_shot_public":
            hunter = event.payload.get("hunter_id", "?")
            target = event.payload.get("target_id", "?")
            summary_items.append((1, f"[枪声] 猎人{hunter}带走了{target}"))

        elif event.type in ("speech", "sheriff_speech"):
            _append_speech_claims(summary_items, event.payload)

    summary_items = _truncate_summary_items(summary_items)
    public_summary = "\n".join(text for _, text in summary_items)
    if public_summary:
        return f"{TIMELINE_ORDER_NOTE}\n{public_summary}"

    current_label = current_phase_label(
        gs.phase, day_number=gs.day_number, night_number=gs.night_number
    )
    if current_label:
        return f"{TIMELINE_ORDER_NOTE}\n当前时间点：{current_label}"
    return TIMELINE_ORDER_NOTE


def _append_vote_resolution(
    summary_items: list[tuple[int, str]],
    payload: dict[str, Any],
) -> None:
    exiled = payload.get("exiled")
    reason = payload.get("reason", "")
    tied = payload.get("tied", [])
    weighted = payload.get("weighted_tally", {})
    day = payload.get("day_number", "?")
    votes_detail = payload.get("votes", [])

    if exiled:
        if weighted:
            tally_str = "、".join(
                f"{pid}={int(weight)}票"
                for pid, weight in sorted(weighted.items(), key=lambda item: -item[1])[:5]
            )
            summary_items.append((1, f"[放逐] D{day} {exiled}被放逐 ({tally_str})"))
        else:
            summary_items.append((1, f"[放逐] D{day} {exiled}被放逐"))
        _append_vote_lines(summary_items, day, votes_detail)
    elif reason == "second_tie_no_exile":
        summary_items.append((1, "[放逐] 二次平票，无人出局"))
        _append_vote_lines(summary_items, day, votes_detail)
    elif tied:
        summary_items.append((2, f"[放逐] 平票PK: {', '.join(tied)}"))
        _append_vote_lines(summary_items, day, votes_detail)


def _append_vote_lines(
    summary_items: list[tuple[int, str]],
    day: Any,
    votes_detail: list[dict[str, Any]],
) -> None:
    if not votes_detail:
        return
    voter_lines = []
    for vote in votes_detail:
        voter = vote.get("voter", "?")
        target = vote.get("target", "弃票") if vote.get("target") else "弃票"
        voter_lines.append(f"{voter}→{target}")
    summary_items.append((1, f"[投票] D{day}: {'，'.join(voter_lines)}"))


def _append_speech_claims(
    summary_items: list[tuple[int, str]],
    payload: dict[str, Any],
) -> None:
    text = str(payload.get("text", ""))
    speaker = payload.get("speaker", "")
    if "未发表有效言论" in text or not text.strip():
        summary_items.append((3, f"[沉默] {speaker} 未发表任何有效言论"))
        return

    if any(keyword in text for keyword in ("验了", "查验", "查杀", "金水")):
        match = re.search(
            r"(?:第?(\d)夜|N(\d)).*?验[了过]?\s*(p\d+).*?(狼人|查杀|好人|金水)",
            text,
        )
        if match:
            night = match.group(1) or match.group(2)
            target = match.group(3)
            result_raw = match.group(4)
            result_cn = {
                "狼人": "狼人",
                "查杀": "狼人",
                "好人": "好人",
                "金水": "好人",
            }.get(result_raw, result_raw)
            summary_items.append((1, f"[验人] {speaker} 报 N{night} {target}={result_cn}"))

    for pattern, label in [
        (r"(?:我|女巫).{0,4}(?:毒[杀了死]|撒毒).{0,4}(p\d+)", "自称毒杀"),
        (r"(p\d+).{0,6}(?:是|被)(?:女巫)?毒[杀了死]", "被指毒杀"),
        (r"(?:狼[刀杀人]|狼人[刀杀]).{0,4}(p\d+)|(p\d+).{0,4}(?:是|被)狼[刀杀了]", "被指狼刀"),
        (r"(?:我|女巫).{0,4}(?:救[了过]|用解药).{0,4}(p\d+)", "自称救了"),
        (r"(p\d+).{0,4}(?:是)?银水", "被指银水"),
    ]:
        for match in re.finditer(pattern, text):
            target = match.group(1) or match.group(2)
            if target:
                summary_items.append((2, f"[死因] {speaker} 称 {target}{label}"))
                break


def _truncate_summary_items(
    summary_items: list[tuple[int, str]],
    budget: int = 2500,
) -> list[tuple[int, str]]:
    total = sum(len(text) for _, text in summary_items)
    if total <= budget:
        return summary_items

    truncated = list(summary_items)
    for drop_priority in (3, 2, 1):
        if total <= budget:
            break
        for index, (priority, text) in enumerate(truncated):
            if priority == drop_priority and text:
                total -= len(text)
                truncated[index] = (priority, "")
                if total <= budget:
                    break
    return [(priority, text) for priority, text in truncated if text]
