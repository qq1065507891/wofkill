# -*- coding: utf-8 -*-
"""
运行时日间发言、防御发言、PK 发言和遗言行动的 agent 适配器。

作者: Project contributors
创建日期: 2026-07-08

使用示例:
    >>> from werewolf_agent.runtime.agent_day_speech_actions import agent_day_speech
    >>> agent_day_speech(...)
"""

from __future__ import annotations

import logging
from typing import Any

from werewolf_agent.agents.schemas import ActionType, TaskType
from werewolf_agent.core.models import GameState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.evaluation.trace_identity import DecisionIdentity
from werewolf_agent.runtime.agent_action_audit import (
    _audit_context_kwargs,
    _inject_vote_basis_hint,
    _is_sheriff_silenced,
    _seer_credibility_audit_payload,
)
from werewolf_agent.runtime.agent_registry import AgentRegistry
from werewolf_agent.runtime.context import (
    build_agent_context,
    _SPEECH_STYLE_HINTS,
    _action_trace_payload,
    _get_persona_speech_style,
    _merge_strategy_directive,
)
from werewolf_agent.runtime.day_speech_directives import (
    build_day_speech_base_directive,
    build_empty_day_speech_fallback,
    build_sanitized_seer_claim_fallback,
    build_sheriff_election_record,
    build_sheriff_speech_directive,
    build_torn_badge_speech_state,
    collect_sheriff_election_speeches,
)
from werewolf_agent.runtime.defense_speech_directives import (
    build_defense_context_directive,
    build_empty_defense_speech_fallback,
)
from werewolf_agent.runtime.directives import (
    build_hunter_directive as _build_hunter_day_speech_directive,
    build_hybrid_directive as _build_hybrid_day_speech_directive,
    build_idiot_directive as _build_idiot_day_speech_directive,
    build_seer_directive as _build_seer_day_speech_directive,
    build_villager_directive as _build_villager_day_speech_directive,
    build_witch_directive as _build_witch_day_speech_directive,
    build_wolf_day_directive as _build_wolf_day_speech_directive,
)
from werewolf_agent.runtime.directives._shared import (
    build_sheriff_silent_directive as _build_sheriff_silent_directive,
)
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector
from werewolf_agent.runtime.last_words_directives import build_exile_last_words_strategy
from werewolf_agent.runtime.pk_speech_directives import build_pk_speech_strategy

logger = logging.getLogger(__name__)


def agent_defense_speech(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    speaker_id: str,
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any] | None:
    """D-8: handler for TaskType.DEFENSE_SPEECH.

    The defense-speech slot fires when a candidate is accused and gets
    the floor to defend themselves before the room votes.  Pre-fix the
    ``DEFENSE_SPEECH`` task type was registered in the schema and
    referenced in the RAG mapping, but no adapter handler existed —
    any node that tried to delegate defense speeches would crash or
    silently fall through to the scripted fallback.

    This handler delegates to the same machinery as ``agent_day_speech``
    (a public speech is a public speech) but tags the strategy
    directive with a ``defense_context`` block so the model knows it's
    on the spot, not just discussing freely.
    """
    gs: GameState = state["game_state"]
    agent = registry.get_agent(speaker_id)
    if agent is None:
        return None

    context = build_agent_context(
        engine,
        gs,
        speaker_id,
        TaskType.DEFENSE_SPEECH,
        legal_actions=[ActionType.SPEECH],
        wolf_team_plan=state.get("wolf_team_plan"),
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        discussion_positions=state.get("discussion_positions"),
        **_audit_context_kwargs(
            decision_identity, exposure_collector, decision_trace_sink
        ),
    )

    strategy_directive = context.strategy_directive or {}
    strategy_directive["defense_context"] = build_defense_context_directive()
    # M2-2: per-turn VOTE_BASIS_GUIDANCE (seer exempt). Moved out
    # of the stable system prompt so night actions don't see it.
    _inject_vote_basis_hint(strategy_directive, gs, speaker_id)

    context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)
    speech_text = getattr(action, "speech", "") or ""

    # Fallback for empty defense speech
    if not speech_text.strip():
        speech_text = build_empty_defense_speech_fallback(speaker_id)

    return {"speech_text": speech_text, "action_trace": _action_trace_payload(action)}


def agent_day_speech(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    speaker_id: str,
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any] | None:
    """Try to get day speech from agent. Returns None for scripted fallback."""
    gs: GameState = state["game_state"]
    agent = registry.get_agent(speaker_id)
    if agent is None:
        return None

    context = build_agent_context(
        engine,
        gs,
        speaker_id,
        TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH],
        wolf_team_plan=state.get("wolf_team_plan"),
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        discussion_positions=state.get("discussion_positions"),
        **_audit_context_kwargs(
            decision_identity, exposure_collector, decision_trace_sink
        ),
    )

    style_hint = ""
    ss = _get_persona_speech_style(agent)
    if ss and ss in _SPEECH_STYLE_HINTS:
        style_hint = f"\n- 你的发言风格：{_SPEECH_STYLE_HINTS[ss]}"
    strategy_directive = {
        **(context.strategy_directive or {}),
        **build_day_speech_base_directive(style_hint),
    }

    # Role-specific speech constraints
    # M2-2: per-turn VOTE_BASIS_GUIDANCE (seer exempt). Moved out
    # of the stable system prompt so night actions don't see it.
    # Speech adapters also need it because the LLM often frames its
    # current speech in terms of who it intends to vote for.
    _inject_vote_basis_hint(strategy_directive, gs, speaker_id)
    player_role = gs.players[speaker_id].role if speaker_id in gs.players else ""
    if player_role == "werewolf":
        wolf_parts = _build_wolf_day_speech_directive(
            gs,
            speaker_id,
            state.get("wolf_team_plan"),
        )
        strategy_directive.update(wolf_parts)
    elif player_role == "seer":
        # P0-G3223805846-3: pass the day's speech order so the seer directive
        # can enforce the "jump immediately when speaking late" rule.  The
        # order lives on RuntimeState (populated by free_discussion); fall
        # back to None when not yet materialised so the directive still
        # works in unit tests / early-day planning contexts.
        seer_speech_parts = _build_seer_day_speech_directive(
            gs,
            speaker_id,
            speech_order=state.get("speech_order"),
        )
        strategy_directive.update(seer_speech_parts)
    elif player_role == "hunter":
        strategy_directive["hunter_speech_directive"] = (
            _build_hunter_day_speech_directive(gs, speaker_id)
        )
    elif player_role == "hybrid":
        strategy_directive["hybrid_speech_directive"] = (
            _build_hybrid_day_speech_directive(gs, speaker_id)
        )
    elif player_role == "witch":
        # D-1: delegate to the dedicated witch directive module so the
        # day-speech guidance is structured (and D-7 enriches it with
        # the witch's private view of public death-cause claims).
        strategy_directive.update(
            _build_witch_day_speech_directive(gs, speaker_id),
        )
    elif player_role == "idiot":
        strategy_directive.update(_build_idiot_day_speech_directive(gs, speaker_id))
    elif player_role == "villager":
        strategy_directive.update(_build_villager_day_speech_directive(gs, speaker_id))

    # Sheriff gets 归票 (vote push) directive — with silence fallback
    # (P1-D4) and torn-badge election-state directive (P1-D6).
    if gs.sheriff_id == speaker_id and gs.sheriff_badge_state == "active":
        alive_others = [
            pid for pid, p in gs.players.items() if p.alive and pid != speaker_id
        ]
        strategy_directive.update(
            build_sheriff_speech_directive(
                is_silenced=_is_sheriff_silenced(gs, speaker_id),
                alive_others=alive_others,
            )
        )

    # After badge tear → no sheriff for the rest of the game.  Every
    # player (not just the previous sheriff) must know there is no
    # 归票人 and that speech order is now random (design doc §警长规则).
    if gs.sheriff_id is None and gs.sheriff_badge_state == "torn":
        strategy_directive["sheriff_election_state"] = build_torn_badge_speech_state()
        # P0-G3223805846-9: inject 归票 hint so players don't fall back
        # on "loudest voice wins".  Distinct key from `sheriff_silent`
        # (which is reserved for the silenced-but-alive sheriff case).
        strategy_directive.update(
            _build_sheriff_silent_directive(
                gs,
                sheriff_id=None,
                badge_state="torn",
            )
        )

    sheriff_election_record = build_sheriff_election_record(
        collect_sheriff_election_speeches(gs)
    )
    if sheriff_election_record:
        strategy_directive["sheriff_election_record"] = sheriff_election_record

    context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)

    if action.action_type == ActionType.SELF_DESTRUCT:
        return {"speech_text": "", "action_trace": {}, "self_destruct": True}

    speech_text = getattr(action, "speech", "") or ""

    # Reject empty day speeches — provide fallback
    if not speech_text.strip():
        alive_others = [
            pid for pid, p in gs.players.items() if p.alive and pid != speaker_id
        ]
        target_hint = alive_others[0] if alive_others else ""
        speech_text = build_empty_day_speech_fallback(speaker_id, target_hint)

    # Guardrail: enforce the 1-check-per-night rule on public seer claims.
    # If a wolf (or anyone) generated a speech that violates this rule,
    # replace it with a sanitized fallback so the bad claim never reaches
    # the public timeline.
    # D-13: the validator used to be gated on ``player_role == "werewolf"``;
    # the rule is a domain rule (one check per night), not a wolf-specific
    # guard, so any speech — wolf impostor, confused LLM, hybrid, etc. —
    # must be screened.  We now run the validator on every public speech
    # that contains seer-style claim patterns.
    if speech_text:
        from werewolf_agent.runtime.seer_claim_validator import validate_seer_claim

        claim_err = validate_seer_claim(speech_text, day_number=gs.day_number)
        if claim_err:
            logger.warning(
                "Speaker %s (role=%s) speech violated seer claim rule: %s — applying fallback",
                speaker_id,
                player_role,
                claim_err,
            )
            alive_others = [
                pid for pid, p in gs.players.items() if p.alive and pid != speaker_id
            ]
            target_hint = alive_others[0] if alive_others else ""
            speech_text = build_sanitized_seer_claim_fallback(
                speaker_id,
                target_hint,
            )

    return {
        "speech_text": speech_text,
        "action_trace": _action_trace_payload(action),
        "seer_credibility_audit": _seer_credibility_audit_payload(
            context,
            gs.day_number,
        ),
        "self_destruct": False,
    }


def agent_pk_speech(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    speaker_id: str,
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any] | None:
    """Get PK speech from a tied candidate. Returns None for scripted fallback."""
    gs: GameState = state["game_state"]
    agent = registry.get_agent(speaker_id)
    if agent is None:
        return None

    # Build context with prior vote tally info
    prior_tally = {}
    for e in gs.events:
        if e.type == "vote_resolved":
            prior_tally = e.payload
            break

    context = build_agent_context(
        engine,
        gs,
        speaker_id,
        TaskType.PK_SPEECH,
        legal_actions=[ActionType.SPEECH],
        wolf_team_plan=state.get("wolf_team_plan"),
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(
            decision_identity, exposure_collector, decision_trace_sink
        ),
    )
    # Add prior tally to visible state
    if prior_tally:
        updated_visible = {
            **context.visible_world_state,
            "prior_vote_tally": prior_tally,
        }
        context = context.model_copy(update={"visible_world_state": updated_visible})

    pk_strategy = build_pk_speech_strategy(gs, speaker_id)
    # M2-2: per-turn VOTE_BASIS_GUIDANCE (seer exempt). Moved out
    # of the stable system prompt so night actions don't see it.
    _inject_vote_basis_hint(pk_strategy, gs, speaker_id)

    context = _merge_strategy_directive(context, pk_strategy)

    action, retry_info = agent.act(context)
    speech_text = getattr(action, "speech", "") or ""
    return {"speech_text": speech_text, "action_trace": _action_trace_payload(action)}


def agent_exile_last_words(
    state: dict[str, Any],
    engine: RuleEngine,
    registry: AgentRegistry,
    player_id: str,
    *,
    decision_identity: DecisionIdentity | None = None,
    exposure_collector: ModuleExposureAuditCollector | None = None,
    decision_trace_sink: Any | None = None,
) -> dict[str, Any] | None:
    """Exiled player gives last words."""
    gs: GameState = state["game_state"]
    agent = registry.get_agent(player_id)
    if agent is None:
        return None

    player_role = gs.players[player_id].role if player_id in gs.players else ""
    context = build_agent_context(
        engine,
        gs,
        player_id,
        TaskType.LAST_WORDS,
        legal_actions=[ActionType.SPEECH],
        rag_service=state.get("rag_service"),
        restored_memory=state.get("restored_memory"),
        cognition_state_manager=state.get("cognition_state_manager"),
        **_audit_context_kwargs(
            decision_identity, exposure_collector, decision_trace_sink
        ),
    )
    alive_others = [
        pid for pid, player in gs.players.items() if player.alive and pid != player_id
    ]
    strategy_directive = build_exile_last_words_strategy(player_role, alive_others)
    context = _merge_strategy_directive(context, strategy_directive)

    action, retry_info = agent.act(context)
    speech_text = getattr(action, "speech", "") or ""
    return {"speech_text": speech_text, "action_trace": _action_trace_payload(action)}


__all__ = [
    "agent_defense_speech",
    "agent_day_speech",
    "agent_pk_speech",
    "agent_exile_last_words",
]