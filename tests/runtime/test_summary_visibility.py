# -*- coding: utf-8 -*-
"""
验证公开上下文摘要可见性和内部讨论摘要任务隔离。

作者: Project contributors
修改日期: 2026-07-25
"""

from __future__ import annotations

from werewolf_agent.agents.discussion_summary import (
    DiscussionSummary,
    DiscussionSummaryGenerationError,
)
from werewolf_agent.agents.schemas import AgentContext, TaskType
from werewolf_agent.core.models import Death, GameEvent, GameState, PlayerState
from werewolf_agent.runtime.nodes.summary import (
    _sleep_between_agent_calls,
    summarize_positions,
    summarize_context,
)


def test_public_context_summary_excludes_private_player_positions() -> None:
    gs = GameState(
        game_id="summary_visibility",
        day_number=1,
        players={
            "p01": PlayerState(id="p01", role="werewolf"),
            "p02": PlayerState(id="p02", role="villager"),
        },
    )

    result = summarize_context({
        "game_state": gs,
        "discussion_positions": {
            "p01": "我是狼人，我准备推动p02出局。",
            "p02": "我怀疑p01。",
        },
    })

    event = result["game_state"].events[-1]
    assert event.type == "context_summary"
    assert event.payload["visibility"] == "public"
    assert "position_summary" not in event.payload
    assert "我是狼人" not in str(event.payload)


def test_context_summary_excludes_persisted_failed_death_batch() -> None:
    gs = GameState(
        game_id="summary-failed-batch",
        day_number=1,
        players={"p01": PlayerState(id="p01", role="villager", alive=False)},
        deaths=[
            Death(
                "p01",
                "exile",
                "day_vote",
                "day_1_vote",
                resolution_batch_parse_failed=True,
            )
        ],
    )

    result = summarize_context({"game_state": gs})

    event = result["game_state"].events[-1]
    assert event.type == "context_summary"
    assert event.payload["deaths_this_day"] == []


def test_agent_call_delay_can_be_disabled(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(
        "werewolf_agent.runtime.nodes.summary.time.sleep",
        sleeps.append,
    )

    _sleep_between_agent_calls({"agent_call_delay_ms": -1}, default_ms=10000)

    assert sleeps == []


def test_agent_call_delay_uses_node_default_when_configured_as_zero(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(
        "werewolf_agent.runtime.nodes.summary.time.sleep",
        sleeps.append,
    )

    _sleep_between_agent_calls({"agent_call_delay_ms": 0}, default_ms=20000)

    assert sleeps == [20.0]


def test_summarize_positions_uses_internal_summary_contract(monkeypatch) -> None:
    calls: list[AgentContext] = []

    class _Agent:
        def summarize_discussion(self, context: AgentContext) -> DiscussionSummary:
            calls.append(context)
            return DiscussionSummary(
                summary="p02发言前后矛盾。",
                suspected_players=["p02"],
                vote_target="p02",
                evidence_refs=["speech-1"],
            )

    class _Registry:
        @staticmethod
        def get_agent(player_id: str) -> _Agent | None:
            return _Agent() if player_id == "p01" else None

    monkeypatch.setattr(
        "werewolf_agent.runtime.nodes.summary.build_agent_context",
        lambda _engine, _gs, player_id, task_type, **kwargs: AgentContext(
            agent_id=player_id,
            task_type=task_type,
            strategy_directive=kwargs.get("strategy_directive", {}),
        ),
    )
    gs = GameState(
        game_id="summary-contract",
        day_number=1,
        players={"p01": PlayerState(id="p01", role="villager")},
        events=[
            GameEvent(
                "speech",
                {
                    "speaker": "p02",
                    "text": "我先怀疑p03，现在改投p01。",
                    "day_number": 1,
                },
                event_id="speech-1",
            ),
        ],
    )
    state = {
        "game_state": gs,
        "engine": object(),
        "agent_registry": _Registry(),
        "agent_call_delay_ms": -1,
    }

    result = summarize_positions(state)

    assert [context.task_type for context in calls] == [
        TaskType.DISCUSSION_SUMMARY,
    ]
    assert result["discussion_positions_version"] == 2
    assert result["discussion_positions"]["p01"] == {
        "summary": "p02发言前后矛盾。",
        "suspected_players": ["p02"],
        "trusted_players": [],
        "vote_target": "p02",
        "evidence_refs": ["speech-1"],
    }
    assert gs.events == state["game_state"].events
    assert [event.type for event in gs.events] == ["speech"]


def test_summarize_positions_records_safe_deterministic_fallback(monkeypatch) -> None:
    class _FailingAgent:
        @staticmethod
        def summarize_discussion(_context: AgentContext) -> DiscussionSummary:
            raise RuntimeError("private provider response must not enter audit")

    class _Registry:
        @staticmethod
        def get_agent(_player_id: str) -> _FailingAgent:
            return _FailingAgent()

    monkeypatch.setattr(
        "werewolf_agent.runtime.nodes.summary.build_agent_context",
        lambda _engine, _gs, player_id, task_type, **_kwargs: AgentContext(
            agent_id=player_id,
            task_type=task_type,
        ),
    )
    gs = GameState(
        game_id="summary-fallback",
        day_number=2,
        players={"p01": PlayerState(id="p01", role="villager")},
        events=[
            GameEvent(
                "speech",
                {
                    "speaker": "p02",
                    "text": "我怀疑p03，我投票p03。",
                    "day_number": 2,
                },
            ),
        ],
    )

    result = summarize_positions({
        "game_state": gs,
        "engine": object(),
        "agent_registry": _Registry(),
        "agent_call_delay_ms": -1,
    })

    payload = result["discussion_positions"]["p01"]
    summary = DiscussionSummary.model_validate(payload)
    assert summary.summary.startswith("p02:")
    assert summary.vote_target is None
    assert result["discussion_summary_audit_records"] == [{
        "player_id": "p01",
        "task": "discussion_summary",
        "outcome": "deterministic_fallback",
        "failure_code": "model_failure",
    }]
    assert "private provider response" not in str(
        result["discussion_summary_audit_records"]
    )


def test_summarize_positions_normalizes_unsafe_failure_code(monkeypatch) -> None:
    class _FailingAgent:
        @staticmethod
        def summarize_discussion(_context: AgentContext) -> DiscussionSummary:
            raise DiscussionSummaryGenerationError(
                "provider returned private raw response"
            )

    class _Registry:
        @staticmethod
        def get_agent(_player_id: str) -> _FailingAgent:
            return _FailingAgent()

    monkeypatch.setattr(
        "werewolf_agent.runtime.nodes.summary.build_agent_context",
        lambda _engine, _gs, player_id, task_type, **_kwargs: AgentContext(
            agent_id=player_id,
            task_type=task_type,
        ),
    )
    gs = GameState(
        game_id="summary-safe-failure-code",
        day_number=1,
        players={"p01": PlayerState(id="p01", role="villager")},
        events=[
            GameEvent(
                "speech",
                {
                    "speaker": "p02",
                    "text": "我怀疑p03。",
                    "day_number": 1,
                },
            ),
        ],
    )

    result = summarize_positions({
        "game_state": gs,
        "engine": object(),
        "agent_registry": _Registry(),
        "agent_call_delay_ms": -1,
    })

    assert result["discussion_summary_audit_records"][0]["failure_code"] == (
        "model_failure"
    )
    assert "private raw response" not in str(
        result["discussion_summary_audit_records"]
    )


def test_empty_discussion_writes_v2_mapping() -> None:
    gs = GameState(
        game_id="summary-empty",
        day_number=1,
        players={"p01": PlayerState(id="p01", role="villager")},
    )

    result = summarize_positions({"game_state": gs, "engine": object()})

    assert result == {
        "discussion_positions_version": 2,
        "discussion_positions": {},
        "discussion_summary_audit_records": [],
        "_day": 1,
    }
