# -*- coding: utf-8 -*-
"""
封装 target-choice 与 speech-intent 输出模式的解析管线。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.agents.choice_pipeline import uses_choice_pipeline
    >>> uses_choice_pipeline([], [])
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.agents.action_data_extraction import extract_decision_data
from werewolf_agent.agents.action_repair import (
    repair_speech_intent_decision,
    repair_target_decision,
    repair_vote_decision,
)
from werewolf_agent.agents.schemas import ActionType, PlayerAction, SeerStance, VoteBasis

CHOICE_TARGET_ACTIONS = {
    ActionType.VOTE,
    ActionType.WOLF_KILL,
    ActionType.USE_POISON,
    ActionType.CHECK_ALIGNMENT,
    ActionType.CHOOSE_MASTER,
    ActionType.HUNTER_SHOT,
    ActionType.BADGE_TRANSFER,
    ActionType.SHERIFF_VOTE,
}


def _target_choice_action(legal_actions: list[ActionType]) -> ActionType | None:
    for action in legal_actions:
        if action in CHOICE_TARGET_ACTIONS:
            return action
    return None


def parse_choice_action(
    text: str,
    legal_actions: list[ActionType],
    legal_targets: list[str],
    salience_items: list[dict[str, Any]],
) -> tuple[PlayerAction | None, str | None, dict[str, Any] | None]:
    data, parse_error = extract_decision_data(text)
    if data is None:
        return None, parse_error, None
    if "choice" not in data and "target_id" not in data:
        return None, "Choice output must include choice or target_id", data

    target_action = _target_choice_action(legal_actions)
    if target_action is None:
        return None, "No target-required action available for choice output", data

    if target_action == ActionType.VOTE:
        repaired = repair_vote_decision(data, legal_actions, legal_targets, salience_items)
    else:
        repaired = repair_target_decision(data, legal_actions, legal_targets, salience_items)
    if repaired is None:
        return None, "Could not map choice to legal target", data

    # P0-S8: only VOTE actions carry vote-audit fields. With
    # ``extra="forbid"`` on every variant, passing them to e.g.
    # WolfKillPlayerAction is now a parse error — so we only attach
    # them when the action is actually a vote.
    if target_action == ActionType.VOTE:
        action = PlayerAction(
            action_type=target_action,
            target_id=repaired["target_id"],
            speech="",
            reason=repaired["reason"],
            confidence=repaired["confidence"],
            seer_stance=repaired.get("seer_stance", SeerStance.UNDECIDED.value),
            vote_basis=repaired.get("vote_basis", VoteBasis.FALLBACK.value),
            standing_with_seer=repaired.get("standing_with_seer", ""),
            suspect_reason=repaired.get("suspect_reason", ""),
            not_voting_reason=repaired.get("not_voting_reason", ""),
            private_reason=repaired.get("private_reason", ""),
        )
    else:
        action = PlayerAction(
            action_type=target_action,
            target_id=repaired["target_id"],
            speech="",
            reason=repaired["reason"],
            confidence=repaired["confidence"],
        )
    return action, None, repaired


def parse_speech_intent_action(
    text: str,
    context_agent_id: str,
    context_own_role: str | None,
    context_legal_targets: list[str],
    context_salience_items: list[dict[str, Any]],
    context_visible_world_state: dict[str, Any],
    context_recent_transcript: list[dict[str, Any]],
) -> tuple[PlayerAction | None, str | None, dict[str, Any] | None]:
    data, parse_error = extract_decision_data(text)
    if data is None:
        return None, parse_error, None
    if "intent" not in data and "speech" not in data:
        return None, "Speech intent output must include intent or speech", data

    repaired = repair_speech_intent_decision(
        data,
        context_agent_id,
        context_own_role,
        context_legal_targets,
        context_salience_items,
        context_visible_world_state,
        context_recent_transcript,
    )
    action = PlayerAction(
        action_type=ActionType.SPEECH,
        target_id=repaired["target_id"],
        speech=repaired["speech"],
        reason=repaired["reason"],
        confidence=repaired["confidence"],
        intent=repaired["intent"],
    )
    return action, None, repaired


def uses_choice_pipeline(legal_actions: list[ActionType], legal_targets: list[str]) -> bool:
    target_actions = [action for action in legal_actions if action in CHOICE_TARGET_ACTIONS]
    non_target_actions = [action for action in legal_actions if action not in CHOICE_TARGET_ACTIONS]
    if len(legal_actions) == 1:
        return legal_actions[0] in CHOICE_TARGET_ACTIONS and bool(legal_targets)
    return (
        len(target_actions) == 1
        and target_actions[0] == ActionType.SHERIFF_VOTE
        and set(non_target_actions).issubset({ActionType.NO_ACTION})
        and bool(legal_targets)
    )


def uses_speech_intent_pipeline(
    legal_actions: list[ActionType],
    task_type: Any,
    speech_intent_tasks: set,
) -> bool:
    speech_only = legal_actions == [ActionType.SPEECH]
    legacy_speech_vote = set(legal_actions) == {ActionType.SPEECH, ActionType.VOTE}
    return (
        task_type in speech_intent_tasks
        and (speech_only or legacy_speech_vote)
    )
