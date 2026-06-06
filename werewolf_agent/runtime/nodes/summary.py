"""Summary and reflection node functions.

- ``summarize_positions`` — per-player LLM summarisation after free discussion
- ``summarize_context`` — daily structured context summary for pruning
- ``reflection`` — post-game per-player reflection using ReflectionMemory
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any

from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.agents.schemas import ActionType, TaskType
from werewolf_agent.runtime.context import build_agent_context
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
    """Each alive player independently summarises today's speeches.

    Design doc SS6.2: after free_discussion, before day_vote. Each agent receives
    the full day transcript and produces a personal summary -- who they suspect,
    trust, and plan to vote for. Deterministic extraction is used as fallback.
    """
    gs: GameState = state["game_state"]
    day = gs.day_number

    speeches = [
        e for e in gs.events
        if e.type in ("speech", "sheriff_speech")
        and e.payload.get("day_number") == day
    ]
    if not speeches:
        return {"discussion_positions": {}, "_day": day}

    # Build transcript text for LLM consumption
    transcript_lines = []
    for ev in speeches:
        speaker = ev.payload.get("speaker", "?")
        text = str(ev.payload.get("text", "") or "")
        if text.strip():
            transcript_lines.append(f"[{speaker}]: {text}")
    transcript_text = "\n".join(transcript_lines)

    # Dispatch each alive player to independently summarize the day
    engine: RuleEngine = state["engine"]
    positions: dict[str, str] = {}
    summarizers: list[str] = [pid for pid, p in gs.players.items() if p.alive]
    for i, pid in enumerate(summarizers):
        # 10s gap between LLM calls (serial, no concurrency)
        if i > 0:
            time.sleep(10)
        summary_text = ""
        try:
            registry = state.get("agent_registry")
            if registry is not None:
                agent = registry.get_agent(pid)
                if agent is not None:
                    context = build_agent_context(
                        engine, gs, pid, TaskType.SPEECH,
                        legal_actions=[ActionType.SPEECH],
                        wolf_team_plan=state.get("wolf_team_plan"),
                        discussion_positions=state.get("discussion_positions"),
                    )
                    extra_directive = {
                        "summary_task": (
                            "请总结今天的讨论：1) 每个玩家说了什么（每人一句话）"
                            "2) 你怀疑谁？为什么？3) 你信任谁？为什么？4) 你打算投谁？"
                            "只输出总结，不要编造发言中不存在的内容。"
                        ),
                        "transcript_text": transcript_text,
                    }
                    sd = context.strategy_directive or {}
                    sd.update(extra_directive)
                    context = context.model_copy(update={"strategy_directive": sd})

                    action, _retry_info = agent.act(context)
                    summary_text = getattr(action, "speech", "") or ""
        except Exception:
            logger.debug("LLM summarisation failed for %s, using deterministic fallback", pid, exc_info=True)

        if not summary_text:
            summary_text = _build_deterministic_summary(pid, speeches)
        positions[pid] = summary_text

    return {"discussion_positions": positions, "_day": day}


def _build_deterministic_summary(player_id: str, speeches: list[GameEvent]) -> str:
    """Deterministic fallback: extract suspects/trusts/claims from speeches."""
    parts = []
    for ev in speeches:
        speaker = ev.payload.get("speaker", "?")
        text = str(ev.payload.get("text", "") or "")
        if not text.strip():
            continue
        s = _extract_suspects(text)
        t = _extract_trusts(text)
        c = _extract_role_claim(text)
        v = _extract_vote_intent(text)
        sn = _first_sentence(text)
        detail = f"{speaker}: {sn}"
        if c:
            detail += f" [声称{c}]"
        if s:
            detail += f" 怀疑{','.join(s)}"
        if t:
            detail += f" 信任{','.join(t)}"
        if v:
            detail += f" 想投{v}"
        parts.append(detail)
    return "\n".join(parts) if parts else "今日无有效发言"


# ---------------------------------------------------------------------------
# Helpers: deterministic speech extraction (fallback)
# ---------------------------------------------------------------------------

def _extract_suspects(text: str) -> list[str]:
    import re
    suspects: list[str] = []
    for m in re.finditer(r"(?:怀疑|标狼|狼面|定狼|抗推|出)\s*(p\d{2})", text):
        pid = m.group(1)
        if pid not in suspects:
            suspects.append(pid)
    return suspects


def _extract_trusts(text: str) -> list[str]:
    import re
    trusts: list[str] = []
    for m in re.finditer(r"(?:相信|好人|保|银水|金水|认好)\s*(p\d{2})", text):
        pid = m.group(1)
        if pid not in trusts:
            trusts.append(pid)
    return trusts


def _extract_role_claim(text: str) -> str | None:
    import re
    m = re.search(r"(?:我是|跳|身份是|底牌是)\s*(预言家|女巫|猎人|白痴|平民|村民|混血儿)", text)
    return m.group(1) if m else None


def _extract_vote_intent(text: str) -> str | None:
    import re
    m = re.search(r"(?:归票|票投|出|投给|投票|上票)\s*(p\d{2})", text)
    return m.group(1) if m else None


def _first_sentence(text: str, max_len: int = 60) -> str:
    for sep in ("。", "！", "？", "\n"):
        idx = text.find(sep)
        if idx > 0:
            sentence = text[:idx + 1].strip()
            return sentence[:max_len]
    return text.strip()[:max_len]


# ---------------------------------------------------------------------------
# Node 26: summarize_context
# ---------------------------------------------------------------------------


def summarize_context(state: RuntimeState) -> dict[str, Any]:
    """Generate structured daily summary for context pruning.

    Design doc SS6.2: at the end of each day, produce a summary of
    stance changes, vote relationships, key contradictions, and
    death / skill clues. Emitted as a GameEvent for audit trail.
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
    }

    # Per-player position summaries
    discussion = state.get("discussion_positions") or {}
    if discussion:
        summary_parts["position_summary"] = discussion

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

    Design doc SS6.2 node 27, SS10.2: each player generates a reflection
    covering key judgments, mistakes, successful strategies, deception
    experienced, and improvement suggestions. Results are stored into
    ReflectionMemory for future-game retrieval.
    """
    gs: GameState = state["game_state"]
    engine: RuleEngine = state["engine"]

    # Build per-player reflection via agent calls when registry is available
    reflection_entries: list[dict[str, Any]] = []
    for i, (pid, player) in enumerate(gs.players.items()):
        # 20s between reflection LLM calls to avoid overwhelming the API at game end
        if i > 0:
            time.sleep(20)
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
