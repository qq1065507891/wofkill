# -*- coding: utf-8 -*-
"""
运行时节点共享 helper 拆分后的兼容导入测试。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> pytest tests/runtime/test_shared_node_helpers.py
"""

from __future__ import annotations

from werewolf_agent.core.models import GameState, PlayerState
from werewolf_agent.runtime.nodes import _shared
from werewolf_agent.runtime.nodes import action_audit
from werewolf_agent.runtime.nodes import node_helpers
from werewolf_agent.runtime.nodes import runtime_state


def test_runtime_state_symbols_are_reexported_from_shared_facade() -> None:
    assert _shared.RuntimeState is runtime_state.RuntimeState
    assert _shared.RULESET_PATH == runtime_state.RULESET_PATH
    assert _shared._new_engine is runtime_state._new_engine
    assert _shared._stable_seed is runtime_state._stable_seed


def test_action_audit_symbols_are_reexported_from_shared_facade() -> None:
    assert _shared._allocate_decision_identity is action_audit._allocate_decision_identity
    assert _shared._action_trace_event is action_audit._action_trace_event
    assert _shared._action_audit_events is action_audit._action_audit_events
    assert _shared._private_vote_audit_payload is action_audit._private_vote_audit_payload
    assert _shared._public_vote_reason is action_audit._public_vote_reason
    assert _shared._with_vote_target_in_trace is action_audit._with_vote_target_in_trace


def test_node_helper_symbols_are_reexported_from_shared_facade() -> None:
    assert _shared._player_ids is node_helpers._player_ids
    assert _shared._alive_wolves is node_helpers._alive_wolves
    assert _shared._alive_non_wolves is node_helpers._alive_non_wolves
    assert _shared._build_wolf_team_plan is node_helpers._build_wolf_team_plan
    assert _shared._planned_wolf_kill is node_helpers._planned_wolf_kill
    assert _shared._jb is node_helpers._jb


def test_node_helpers_keep_wolf_team_plan_behavior() -> None:
    state = GameState(
        game_id="g_shared",
        night_number=2,
        players={
            "w1": PlayerState(id="w1", role="werewolf", alive=True),
            "w2": PlayerState(id="w2", role="werewolf", alive=True),
            "p01": PlayerState(id="p01", role="seer", alive=True),
        },
    )

    plan = node_helpers._build_wolf_team_plan(
        state,
        previous_plan={
            "night_kill_primary": "p01",
            "evidence_quality": "strong",
            "evidence_from_discussion": [{"target": "p01"}],
        },
    )

    assert plan["fake_seer"] == "w1"
    assert plan["pusher"] == "w2"
    assert plan["night_kill_primary"] == "p01"
