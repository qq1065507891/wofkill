"""Summary and reflection node functions.

- ``summarize_positions`` — deterministic position summary after free discussion
- ``summarize_context`` — daily structured context summary for pruning
- ``reflection`` — post-game per-player reflection using ReflectionMemory
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.agent_adapter import _agent_reflection
from werewolf_agent.runtime.nodes._shared import (
    RuntimeState,
    _action_trace_event,
    _dispatch_agent,
    _judge_broadcast,
    _player_display,
)

logger = logging.getLogger(__name__)


def _route_after_summarize(state: RuntimeState) -> str:
    """Route to sheriff_endorse if sheriff exists, otherwise day_vote."""
    gs: GameState = state["game_state"]
    if gs.sheriff_id and gs.sheriff_badge_state == "active":
        sheriff = gs.players.get(gs.sheriff_id)
        if sheriff and sheriff.alive:
            return "sheriff_endorse"
    return "day_vote"


# ---------------------------------------------------------------------------
# Node 17: summarize_positions
# ---------------------------------------------------------------------------


def summarize_positions(state: RuntimeState) -> dict[str, Any]:
    """Summarise each player's stated positions from current day's speeches.

    Design doc §6.2: after free_discussion, before day_vote, produce a
    structured summary of who suspects whom, who trusts whom, and any
    claimed roles or key claims.
    """
    gs: GameState = state["game_state"]
    day = gs.day_number

    speeches = [
        e for e in gs.events
        if e.type in ("speech", "sheriff_speech")
        and e.payload.get("day_number") == day
    ]

    if not speeches:
        return {"discussion_positions": []}

    positions: list[dict[str, Any]] = []
    for ev in speeches:
        speaker = ev.payload.get("speaker", "?")
        text = str(ev.payload.get("text", "") or "")

        # Simple keyword-based position extraction (deterministic, no LLM)
        suspects = _extract_suspects(text)
        trusts = _extract_trusts(text)
        claimed_role = _extract_role_claim(text)

        positions.append({
            "speaker": speaker,
            "suspects": suspects,
            "trusts": trusts,
            "claimed_role": claimed_role,
        })

    return {"discussion_positions": positions}


def _extract_suspects(text: str) -> list[str]:
    """Extract suspect mentions from speech text."""
    import re
    suspects: list[str] = []
    for m in re.finditer(r"(?:怀疑|标狼|狼面|定狼|抗推|出)\s*(p\d{2})", text):
        pid = m.group(1)
        if pid not in suspects:
            suspects.append(pid)
    return suspects


def _extract_trusts(text: str) -> list[str]:
    """Extract trust mentions from speech text."""
    import re
    trusts: list[str] = []
    for m in re.finditer(r"(?:相信|好人|保|银水|金水|认好)\s*(p\d{2})", text):
        pid = m.group(1)
        if pid not in trusts:
            trusts.append(pid)
    return trusts


def _extract_role_claim(text: str) -> str | None:
    """Extract a claimed role from speech text."""
    import re
    m = re.search(r"(?:我是|跳|身份是|底牌是)\s*(预言家|女巫|猎人|白痴|平民|村民|混血儿)", text)
    if m:
        return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Node 26: summarize_context
# ---------------------------------------------------------------------------


def summarize_context(state: RuntimeState) -> dict[str, Any]:
    """Generate structured daily summary for context pruning.

    Design doc §6.2: at the end of each day, produce a summary of
    stance changes, vote relationships, key contradictions, and
    death / skill clues. This summary is stored as a GameEvent and
    used by future days' agent contexts to keep prompts compact.
    """
    gs: GameState = state["game_state"]
    day = gs.day_number

    summary_parts: dict[str, Any] = {
        "day_number": day,
        "alive_count": sum(1 for p in gs.players.values() if p.alive),
    }

    # Death / skill clues
    day_deaths = [
        {"player_id": d.player_id, "reason": d.reason, "timing": d.timing}
        for d in gs.deaths
        if d.resolution_batch and f"day_{day}" in d.resolution_batch
    ]
    summary_parts["deaths_this_day"] = day_deaths

    # Vote relationships
    vote_event = None
    for e in reversed(gs.events):
        if e.type == "vote_resolved" and e.payload.get("day_number") == day:
            vote_event = e
            break
    if vote_event:
        summary_parts["vote_outcome"] = {
            "exiled": vote_event.payload.get("exiled"),
            "reason": vote_event.payload.get("reason"),
            "tied": vote_event.payload.get("tied", []),
        }

    # Sheriff status
    summary_parts["sheriff"] = {
        "id": gs.sheriff_id,
        "badge_state": gs.sheriff_badge_state,
        "interrupt_count": gs.sheriff_interrupt_count,
    }

    # Stance changes — compare discussion_positions if present
    discussion_positions = state.get("discussion_positions") or []
    if discussion_positions:
        summary_parts["position_summary"] = {
            p["speaker"]: {
                "suspects": p.get("suspects", []),
                "trusts": p.get("trusts", []),
                "claimed_role": p.get("claimed_role"),
            }
            for p in discussion_positions
        }

    # Emit as a GameEvent for audit trail
    event = GameEvent(
        type="context_summary",
        payload={"visibility": "public", **summary_parts},
    )
    gs = replace(gs, events=gs.events + [event])

    logger.debug(f"  [上下文摘要] D{day} 总结: 存活{summary_parts['alive_count']}人")

    return {"game_state": gs}


# ---------------------------------------------------------------------------
# Node 27: reflection
# ---------------------------------------------------------------------------


def reflection(state: RuntimeState) -> dict[str, Any]:
    """Post-game per-player reflection.

    Design doc §6.2 node 27, §10.2: each player generates a reflection
    covering key judgments, mistakes, successful strategies, deception
    experienced, and improvement suggestions. Results are stored into
    ReflectionMemory for future-game retrieval.
    """
    gs: GameState = state["game_state"]
    engine: RuleEngine = state["engine"]

    # Build per-player reflection via agent calls when registry is available
    reflection_entries: list[dict[str, Any]] = []
    for pid, player in gs.players.items():
        reflection_text = ""
        try:
            result = _dispatch_agent(state, _agent_reflection, pid)
            if result:
                reflection_text = result.get("reflection_text", "")
        except Exception:
            logger.warning(
                "Failed to generate reflection for %s", _player_display(state, pid),
                exc_info=True,
            )

        entry = {
            "player_id": pid,
            "role": player.role,
            "alive": player.alive,
            "reflection": reflection_text or f"{player.role} 身份玩家 {pid} 完成对局",
        }
        reflection_entries.append(entry)

    # Persist to ReflectionMemory when available
    _GOOD_ROLES = {"villager", "seer", "witch", "hunter", "idiot"}
    winning = gs.winning_faction or ""
    try:
        from werewolf_agent.memory.reflection import ReflectionMemory
        rm = ReflectionMemory(repo=state.get("repository"))
        for entry in reflection_entries:
            pid = entry["player_id"]
            role = entry["role"]
            pf = "unknown"
            if role == "hybrid":
                pf = gs.hybrid_master_faction or "unknown"
            elif role in _GOOD_ROLES:
                pf = "good"
            elif role == "werewolf":
                pf = "werewolf"
            player_faction_won = pf == winning
            rm.store(
                gs.game_id,
                player_id=pid,
                role=role,
                faction_won=player_faction_won,
                text=entry.get("reflection", ""),
                tags=[role, "post_game"],
                situation={"alive": entry["alive"], "day": gs.day_number},
            )
    except Exception:
        logger.warning("Failed to persist reflection entries", exc_info=True)

    event = GameEvent(
        type="reflection_complete",
        payload={"player_count": len(reflection_entries)},
    )
    gs = replace(gs, events=gs.events + [event])

    logger.debug(f"  [复盘] 完成 {len(reflection_entries)} 位玩家的对局复盘")

    return {"game_state": gs}
