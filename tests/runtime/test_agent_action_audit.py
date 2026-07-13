# -*- coding: utf-8 -*-
"""
验证 agent action audit 辅助函数拆分后的兼容行为。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/runtime/test_agent_action_audit.py -q
"""

from __future__ import annotations

from werewolf_agent.agents.schemas import AgentContext, TaskType
from werewolf_agent.core.models import GameEvent, GameState, PlayerState


def test_action_audit_event_always_carries_explicit_task_type() -> None:
    from werewolf_agent.runtime.nodes.action_audit import _action_trace_event

    event = _action_trace_event(
        player_id="p01",
        phase="vote",
        action_trace={"execution_attempts": []},
    )

    assert event.payload["task_type"] == "vote"


def test_audit_context_kwargs_requires_identity_and_collector() -> None:
    from werewolf_agent.runtime.agent_action_audit import _audit_context_kwargs

    sink = object()

    assert _audit_context_kwargs(None, object(), sink) == {}
    assert _audit_context_kwargs(object(), None, sink) == {}

    identity = object()
    collector = object()
    result = _audit_context_kwargs(identity, collector, sink)

    assert result == {
        "decision_identity": identity,
        "exposure_collector": collector,
        "decision_trace_sink": sink,
    }


def test_vote_basis_hint_skips_seer_and_applies_to_non_seer() -> None:
    from werewolf_agent.runtime.agent_action_audit import (
        VOTE_BASIS_GUIDANCE,
        _inject_vote_basis_hint,
    )

    gs = GameState(
        players={
            "p01": PlayerState(id="p01", role="seer"),
            "p02": PlayerState(id="p02", role="villager"),
        },
    )

    seer_directive: dict[str, object] = {}
    villager_directive: dict[str, object] = {}

    _inject_vote_basis_hint(seer_directive, gs, "p01")
    _inject_vote_basis_hint(villager_directive, gs, "p02")

    assert "vote_basis_hint" not in seer_directive
    assert villager_directive["vote_basis_hint"] == VOTE_BASIS_GUIDANCE


def test_seer_credibility_audit_payload_keeps_safe_public_fields() -> None:
    from werewolf_agent.runtime.agent_action_audit import _seer_credibility_audit_payload

    context = AgentContext(
        agent_id="p03",
        task_type=TaskType.SPEECH,
        seer_credibility={
            "seer_lines": [
                {
                    "claimant": "p01",
                    "status": "credible",
                    "score": 0.8,
                    "hidden_role": "seer",
                },
            ],
        },
    )

    payload = _seer_credibility_audit_payload(context, day_number=2)

    assert payload == {
        "day_number": 2,
        "visibility": "moderator_only",
        "seer_lines": [{"claimant": "p01", "status": "credible", "score": 0.8}],
    }


def test_is_sheriff_silenced_uses_event_or_badge_state() -> None:
    from werewolf_agent.runtime.agent_action_audit import _is_sheriff_silenced

    gs = GameState(
        sheriff_badge_state="active",
        events=[GameEvent(type="sheriff_silenced", payload={"sheriff_id": "p01"})],
    )
    assert _is_sheriff_silenced(gs, "p01") is True
    assert _is_sheriff_silenced(gs, "p02") is False

    frozen = GameState(sheriff_badge_state="frozen")
    assert _is_sheriff_silenced(frozen, "p03") is True
