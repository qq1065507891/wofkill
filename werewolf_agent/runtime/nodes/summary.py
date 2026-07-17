# -*- coding: utf-8 -*-
"""处理每日摘要、立场归纳与赛后反思节点。

作者: Project contributors
创建日期: 2025-01-15
修改日期: 2026-07-17
使用示例: 内部模块，无对外接口
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
from werewolf_agent.core.resolution_batches import valid_carrier_resolution_batch
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.agents.schemas import ActionType, TaskType
from werewolf_agent.runtime.context import build_agent_context
from werewolf_agent.runtime.agent_adapter import _agent_reflection
from werewolf_agent.runtime.reflection_events import safe_reflection_verification
from werewolf_agent.runtime.reflection_transaction import (
    PlayerReflectionTransaction,
    ReflectionStage,
    summarize_reflection_transaction,
)
from werewolf_agent.runtime.nodes._shared import (
    RuntimeState,
    _dispatch_agent,
    _player_display,
)

logger = logging.getLogger(__name__)


def _sleep_between_agent_calls(state: RuntimeState, *, default_ms: int) -> None:
    """Apply the shared runtime delay convention between sequential calls."""
    delay_ms = state.get("agent_call_delay_ms", -1)
    if delay_ms == 0:
        delay_ms = default_ms
    if delay_ms > 0:
        time.sleep(delay_ms / 1000.0)


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
        if i > 0:
            _sleep_between_agent_calls(state, default_ms=10000)
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
                        rag_service=state.get("rag_service"),
                        restored_memory=state.get("restored_memory"),
                        cognition_state_manager=state.get("cognition_state_manager"),
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
        if (
            (parsed_batch := valid_carrier_resolution_batch(d)) is not None
            and parsed_batch.phase == "day"
            and parsed_batch.number == day
        )
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

    # Build per-player reflection via agent calls when registry is available
    reflection_entries: list[dict[str, Any]] = []
    transactions: list[PlayerReflectionTransaction] = []
    for i, (pid, player) in enumerate(gs.players.items()):
        if i > 0:
            _sleep_between_agent_calls(state, default_ms=20000)
        decision_id = f"reflection:{gs.game_id}:{pid}"
        verification = safe_reflection_verification(
            {"status": "not_generated"}, decision_id=decision_id,
        )
        dispatch_failed = False
        try:
            result = _dispatch_agent(state, _agent_reflection, pid, post_game=True)
            if result:
                verification = safe_reflection_verification(
                    result.get("reflection_verification"),
                    decision_id=decision_id,
                )
        except Exception:
            dispatch_failed = True
            logger.warning(
                "Failed to generate reflection for %s", _player_display(state, pid),
                exc_info=True,
            )

        try:
            transaction = _reflection_transaction_from_verification(
                pid,
                verification,
                decision_id=decision_id,
                dispatch_failed=dispatch_failed,
            )
        except Exception:
            logger.warning(
                "Failed to reconstruct reflection transaction for %s",
                _player_display(state, pid),
                exc_info=True,
            )
            verification = safe_reflection_verification(
                {
                    "status": "agent_error",
                    "failure_stage": "generated",
                    "failure_code": "reflection_transaction_reconstruction_failed",
                },
                decision_id=decision_id,
            )
            transaction = PlayerReflectionTransaction(pid, decision_id).fail(
                failure_stage="generated",
                failure_code="reflection_transaction_reconstruction_failed",
            )
        transactions.append(transaction)
        entry = {
            "player_id": pid,
            "role": player.role,
            "alive": player.alive,
            "decision_id": decision_id,
            "transaction_state": transaction.stage.value,
            "failure_stage": transaction.failure_stage,
            "failure_code": transaction.failure_code,
            "verification": verification,
        }
        reflection_entries.append(entry)

    transaction_result = summarize_reflection_transaction(transactions)
    game_status = (
        "complete"
        if transactions
        and transaction_result.valid_entry_count == len(transactions)
        and transaction_result.failure_count == 0
        else transaction_result.status
    )
    event = GameEvent(
        type="reflection_complete",
        payload={
            "visibility": "moderator_only",
            "status": game_status,
            "persistence_complete": transaction_result.persistence_complete,
            "player_count": len(reflection_entries),
            "valid_entry_count": transaction_result.valid_entry_count,
            "failure_count": transaction_result.failure_count,
            "entries": reflection_entries,
        },
    )
    events = [*gs.events, event]
    if game_status == "no_valid_entries":
        events.append(GameEvent(
            type="reflection_no_valid_entries",
            payload={
                "visibility": "moderator_only",
                "status": "no_valid_entries",
                "player_count": len(reflection_entries),
                "failures": [
                    {
                        "player_id": item.player_id,
                        "decision_id": item.decision_id,
                        "failure_stage": item.failure_stage,
                        "failure_code": item.failure_code,
                    }
                    for item in transactions
                ],
            },
        ))
    gs = replace(gs, events=events)

    logger.debug(f"  [复盘] 完成 {len(reflection_entries)} 位玩家的对局复盘")

    return {"game_state": gs}


def _reflection_transaction_from_verification(
    player_id: str,
    verification: dict[str, Any],
    *,
    decision_id: str,
    dispatch_failed: bool,
) -> PlayerReflectionTransaction:
    """把安全 verification 映射为逐级事务状态和稳定失败原因。"""
    transaction = PlayerReflectionTransaction(player_id, decision_id)
    status = verification.get("status")
    if dispatch_failed:
        return transaction.fail(
            failure_stage="generated",
            failure_code="agent_error",
        )
    if status in {"not_generated", "agent_error"}:
        return transaction.fail(
            failure_stage=verification.get("failure_stage") or "generated",
            failure_code=verification.get("failure_code") or (
                "agent_error" if status == "agent_error" else "reflection_not_generated"
            ),
        )

    transaction = transaction.advance(ReflectionStage.GENERATED)
    if status == "invalid_structured_draft":
        return transaction.fail(
            failure_stage="schema_validated",
            failure_code="invalid_structured_draft",
        )
    if status != "verified":
        return transaction.fail(
            failure_stage="schema_validated",
            failure_code="invalid_reflection_status",
        )

    transaction = transaction.advance(ReflectionStage.SCHEMA_VALIDATED)
    transaction = transaction.advance(
        ReflectionStage.FACTS_VERIFIED,
        verified_claim_ids=verification.get("verified_claim_ids") or (),
    )
    lessons = verification.get("verified_lessons")
    lesson_ids = [
        lesson.get("lesson_id")
        for lesson in lessons
        if isinstance(lesson, dict)
        and isinstance(lesson.get("lesson_id"), str)
        and lesson.get("lesson_id")
    ] if isinstance(lessons, list) else []
    if not lesson_ids:
        return transaction.fail(
            failure_stage="lessons_verified",
            failure_code="reflection_no_verified_lessons",
        )
    if not transaction.verified_claim_ids:
        return transaction.fail(
            failure_stage="lessons_verified",
            failure_code="reflection_lessons_without_verified_claims",
        )
    return transaction.advance(
        ReflectionStage.LESSONS_VERIFIED,
        verified_lesson_ids=lesson_ids,
    )
