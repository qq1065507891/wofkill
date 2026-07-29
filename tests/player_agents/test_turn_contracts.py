# -*- coding: utf-8 -*-
"""
验证合法动作窗口、回合快照和显式状态转换。

作者: Project contributors
创建日期: 2026-07-29
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from werewolf_agent.player_agents.contracts.revisions import (
    ReadReference,
    RevisionContext,
)
from werewolf_agent.player_agents.contracts.turns import (
    AgentTurn,
    AgentTurnStatus,
    ConflictClass,
    LegalActionWindow,
    TurnBudget,
    transition_turn,
)

HASH = "a" * 64


def _window() -> LegalActionWindow:
    return LegalActionWindow(
        window_id="speech-d1-p01",
        version=1,
        game_id="game-1",
        task_type="day_speech",
        conflict_class=ConflictClass.SERIAL_PUBLIC,
        participant_ids=("p01",),
        legal_actions=("speech",),
        legal_target_ids=("p02", "p03"),
        opened_revision=4,
        deadline=datetime(2026, 7, 29, 1, tzinfo=timezone.utc),
    )


def _turn() -> AgentTurn:
    return AgentTurn(
        turn_id="turn-1",
        game_id="game-1",
        player_id="p01",
        role_id="villager",
        phase="day_discussion",
        task_type="day_speech",
        revision=RevisionContext(
            base_revision=4,
            window_id="speech-d1-p01",
            window_version=1,
            view_fingerprint=HASH,
        ),
        window=_window(),
        read_set=(
            ReadReference(record_id="public-4", revision=4, content_hash=HASH),
        ),
        model_lease_hash=HASH,
        budget=TurnBudget(model_steps=8, tool_calls=12, repairs=1),
        status=AgentTurnStatus.OPEN,
        idempotency_key="turn-1:submit",
    )


def test_turn_requires_matching_window_and_participant() -> None:
    turn = _turn()
    assert turn.window.conflict_class is ConflictClass.SERIAL_PUBLIC
    with pytest.raises(ValidationError, match="player must be a window participant"):
        AgentTurn.model_validate({**turn.model_dump(), "player_id": "p09"})


def test_window_rejects_duplicate_participants_and_naive_deadline() -> None:
    data = _window().model_dump()
    with pytest.raises(ValidationError):
        LegalActionWindow.model_validate({
            **data,
            "participant_ids": ("p01", "p01"),
        })
    with pytest.raises(ValidationError, match="timezone-aware"):
        LegalActionWindow.model_validate({
            **data,
            "deadline": datetime(2026, 7, 29, 1),  # noqa: DTZ001 - 必须构造无时区输入
        })


def test_transition_turn_allows_only_declared_edges() -> None:
    observing = transition_turn(_turn(), AgentTurnStatus.OBSERVING)
    thinking = transition_turn(observing, AgentTurnStatus.THINKING)
    assert thinking.status is AgentTurnStatus.THINKING
    with pytest.raises(ValueError, match="illegal agent turn transition"):
        transition_turn(thinking, AgentTurnStatus.COMMITTED)
