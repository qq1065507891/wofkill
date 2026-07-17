# -*- coding: utf-8 -*-
"""
验证狼人夜聊分层上下文与最终系统提示语义合同。

作者: Project contributors
创建日期: 2026-07-18
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
from werewolf_agent.agents.schemas import AgentContext, TaskType
from werewolf_agent.core.event_visibility import EventVisibility
from werewolf_agent.core.models import GameEvent, GameState, PlayerState
def _discussion_state() -> GameState:
    players = {
        **{
            f"w{index}": PlayerState(
                id=f"w{index}", role="werewolf", alive=True,
            )
            for index in range(1, 5)
        },
        "p1": PlayerState(id="p1", role="villager", alive=True),
        "p2": PlayerState(id="p2", role="seer", alive=True),
    }
    events: list[GameEvent] = []
    for sequence_number in range(12):
        wolf_id = f"w{sequence_number % 4 + 1}"
        round_number = sequence_number // 4 + 1
        event_id = f"g-layered:e{sequence_number:06d}"
        marker = f"raw-secret-{sequence_number}"
        text = f"{marker}-" + ("长文本" * 40)
        events.append(GameEvent(
            type="wolf_discussion",
            payload={
                "wolf_id": wolf_id,
                "round": round_number,
                "night_number": 1,
                "text": text,
                "target_stance": {
                    "wolf_id": wolf_id,
                    "target_id": "p1" if sequence_number % 2 == 0 else "p2",
                    "stance": "propose",
                    "priority": "primary",
                    "source_event_id": event_id,
                    "round_number": round_number,
                },
            },
            visibility=EventVisibility.WEREWOLF_TEAM_ONLY,
            event_id=event_id,
            sequence_number=sequence_number,
            occurred_at=datetime(2026, 7, 18, tzinfo=timezone.utc),
            game_id="g-layered",
            schema_version="2",
        ))
    return GameState(
        game_id="g-layered",
        phase="night",
        night_number=1,
        players=players,
        events=events,
    )


def test_layered_context_preserves_all_stances_and_budgets_only_raw_text() -> None:
    from werewolf_agent.runtime import wolf_discussion_directives

    gs = _discussion_state()

    context = wolf_discussion_directives.build_layered_wolf_discussion_context(
        gs,
        ["w1", "w2", "w3", "w4"],
        recent_raw_limit=8,
        older_summary_chars=32,
    )

    assert len(context["structured"]["target_stances"]) == 12
    assert context["structured"]["live_status"] == {
        "night_number": 1,
        "alive_wolves": ["w1", "w2", "w3", "w4"],
        "alive_non_wolves": ["p1", "p2"],
    }
    assert [row["event_id"] for row in context["text"]["recent_raw"]] == [
        f"g-layered:e{index:06d}" for index in range(4, 12)
    ]
    assert [row["event_id"] for row in context["text"]["older_summary"]] == [
        f"g-layered:e{index:06d}" for index in range(4)
    ]
    assert context["audit"] == {
        "injected_event_ids": [f"g-layered:e{index:06d}" for index in range(12)],
        "raw_text_count": 8,
        "summarized_text_count": 4,
        "truncated_text_count": 4,
    }
    raw_serialized = json.dumps(context["text"]["recent_raw"], ensure_ascii=False)
    summary_serialized = json.dumps(
        context["text"]["older_summary"],
        ensure_ascii=False,
    )
    assert "raw-secret-11" in raw_serialized
    assert "长文本" * 40 not in summary_serialized


def test_layered_context_rebuilds_live_status_instead_of_trusting_old_plan() -> None:
    from werewolf_agent.runtime import wolf_discussion_directives

    gs = _discussion_state()
    players = dict(gs.players)
    players["w4"] = PlayerState(id="w4", role="werewolf", alive=False)
    players["p2"] = PlayerState(id="p2", role="seer", alive=False)
    next_night = GameState(
        game_id=gs.game_id,
        phase="night",
        night_number=2,
        players=players,
        events=gs.events,
    )

    context = wolf_discussion_directives.build_layered_wolf_discussion_context(
        next_night,
        ["w1", "w2", "w3"],
    )

    assert context["structured"]["live_status"] == {
        "night_number": 2,
        "alive_wolves": ["w1", "w2", "w3"],
        "alive_non_wolves": ["p1"],
    }
    assert context["structured"]["target_stances"] == []
    assert context["audit"]["injected_event_ids"] == []


def test_werewolf_system_prompt_states_target_and_evidence_semantics() -> None:
    context = AgentContext(
        agent_id="w1",
        task_type=TaskType.WOLF_DISCUSSION,
        phase="night",
        night_number=1,
        own_role="werewolf",
    )

    system_prompt = PlayerPromptBuilder(context).build_system_prompt()

    assert "备刀不是女巫救人后的第二刀" in system_prompt
    assert "死亡玩家不可作为击杀目标" in system_prompt
    assert "系统提供的候选列表不是局内事实" in system_prompt
    assert "队长不得伪造支持者" in system_prompt


def test_wolf_action_injects_layered_context_and_sanitized_audit(monkeypatch) -> None:
    from werewolf_agent.evaluation.trace_identity import DecisionIdentity
    from werewolf_agent.runtime import agent_wolf_actions
    from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector

    gs = _discussion_state()
    identity = DecisionIdentity(
        "g-layered", "w1", "wolf_discussion_round_3", 0, 1,
        "wolf_discussion", 12,
    )
    collector = ModuleExposureAuditCollector()
    base_context = AgentContext(
        agent_id="w1",
        task_type=TaskType.WOLF_DISCUSSION,
        phase="night",
        night_number=1,
        own_role="werewolf",
        recent_transcript=[{"speaker": "p1", "text": "公开发言"}],
        decision_identity=identity,
        exposure_collector=collector,
    )
    captured: dict[str, AgentContext] = {}
    build_kwargs: dict[str, object] = {}

    class _Agent:
        def act(self, context: AgentContext):
            captured["context"] = context
            return SimpleNamespace(speech="选择 p1", target_stance=None, trace=None), None

    class _Registry:
        def get_agent(self, _player_id: str):
            return _Agent()

    def _fake_build_context(*_args, **kwargs):
        build_kwargs.update(kwargs)
        return base_context

    monkeypatch.setattr(agent_wolf_actions, "build_agent_context", _fake_build_context)

    result = agent_wolf_actions.agent_wolf_discussion(
        {
            "game_state": gs,
            "wolf_discussion_round": 3,
            "wolf_team_plan": {
                "night_number": 0,
                "night_kill_primary": "p2",
                "night_kill_backup": "p1",
            },
        },
        object(),
        _Registry(),
        "w1",
        decision_identity=identity,
        exposure_collector=collector,
    )

    assert result is not None
    directive = captured["context"].strategy_directive
    assert len(directive["wolf_universal_rules"]["target_stances"]) == 12
    assert len(directive["previous_discussion"]["recent_raw"]) == 8
    assert len(directive["previous_discussion"]["older_summary"]) == 4
    assert captured["context"].recent_transcript == base_context.recent_transcript
    assert "wolf_high_priority_target" not in directive
    assert "primary=p2" in directive["wolf_plan_history"]
    assert build_kwargs["wolf_team_plan"] is None
    audit_event = collector.flush_events()[0]
    assert audit_event.type == "wolf_prompt_context_audit"
    assert audit_event.payload["context"]["raw_text_count"] == 8
    assert len(audit_event.payload["context"]["injected_event_ids"]) == 12
    serialized = str(audit_event.payload)
    assert "raw-secret" not in serialized
    assert "target_stance" not in serialized


def test_prompt_renderer_never_truncates_structured_wolf_stances() -> None:
    from werewolf_agent.runtime.wolf_discussion_directives import (
        build_layered_wolf_discussion_context,
        build_wolf_discussion_strategy_directive,
    )

    gs = _discussion_state()
    layered = build_layered_wolf_discussion_context(
        gs,
        ["w1", "w2", "w3", "w4"],
    )
    directive = build_wolf_discussion_strategy_directive(
        discussion_instruction="必须完成团队讨论。" * 100,
        round_focus="统一刀口",
        wolf_teammates=["w2", "w3", "w4"],
        previous_speeches=[],
        layered_context=layered,
    )
    context = AgentContext(
        agent_id="w1",
        task_type=TaskType.WOLF_DISCUSSION,
        phase="night",
        night_number=1,
        own_role="werewolf",
        strategy_directive=directive,
    )

    rendered = PlayerPromptBuilder(context)._build_strategy_directive()

    for index in range(12):
        assert f"g-layered:e{index:06d}" in rendered
    for index in range(12):
        assert f"raw-secret-{index}" in rendered
