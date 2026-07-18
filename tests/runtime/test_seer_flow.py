# -*- coding: utf-8 -*-
"""
验证预言家指令不会鼓励使用无公开证据的身份世界作为验人理由。

作者: Project contributors
创建日期: 2026-07-13
"""

from __future__ import annotations

from werewolf_agent.core.models import GameState, PlayerState
from werewolf_agent.runtime.directives.seer import build_seer_directive
from werewolf_agent.runtime.seer_night_directives import build_seer_legal_targets


def test_seer_reason_rejects_evidence_free_world_roles_without_changing_targets() -> None:
    players = {
        "p01": PlayerState(id="p01", role="seer", alive=True),
        "p02": PlayerState(id="p02", role="werewolf", alive=True),
        "p03": PlayerState(id="p03", role="villager", alive=True),
    }
    gs = GameState(
        game_id="seer_reason_gate",
        phase="night",
        day_number=0,
        night_number=1,
        players=players,
        events=[],
    )

    before = build_seer_legal_targets(gs, seer_id="p01", counterclaiming_seers=set())
    directive = build_seer_directive(gs, "p01")
    after = build_seer_legal_targets(gs, seer_id="p01", counterclaiming_seers=set())

    text = directive["seer_speech_directive"]
    assert "没有公开事件或公开声明 ID 支撑" in text
    assert "不得猜测具体神职身份" in text
    assert before == after == ["p02", "p03"]


def test_seer_opportunity_chain_records_repair_and_private_resolution() -> None:
    from werewolf_agent.core.event_visibility import EventVisibility
    from werewolf_agent.runtime.graph import _new_engine, resolve_night
    from werewolf_agent.runtime.nodes.night_specialists import night_seer

    gs = GameState(
        game_id="seer_opportunity_repair",
        phase="night",
        night_number=1,
        players={
            "seer": PlayerState(id="seer", role="seer"),
            "wolf": PlayerState(id="wolf", role="werewolf"),
            "villager": PlayerState(id="villager", role="villager"),
        },
    )
    engine = _new_engine()
    decision = night_seer({
        "game_state": gs,
        "engine": engine,
        "seer_target_id": "seer",
    })
    assert decision["seer_target_id"] == "wolf"
    after_resolution = resolve_night({
        "game_state": decision["game_state"],
        "engine": engine,
        "wolf_kill_target_id": None,
        "use_antidote": False,
        "poison_target_id": None,
        "seer_target_id": decision["seer_target_id"],
    })["game_state"]
    events = {event.type: event for event in after_resolution.events}

    assert events["seer_check_opportunity"].visibility is EventVisibility.MODERATOR_ONLY
    assert events["seer_check_repaired"].visibility is EventVisibility.MODERATOR_ONLY
    assert events["seer_check_resolved"].visibility is EventVisibility.MODERATOR_ONLY
    assert all(event.visibility is not EventVisibility.PUBLIC for event in after_resolution.events if event.type.startswith("seer_check_"))


def test_seer_without_legal_targets_records_skipped_opportunity() -> None:
    from werewolf_agent.runtime.graph import _new_engine
    from werewolf_agent.runtime.nodes.night_specialists import night_seer

    gs = GameState(
        game_id="seer_opportunity_skip",
        phase="night",
        night_number=2,
        players={"seer": PlayerState(id="seer", role="seer")},
    )

    result = night_seer({"game_state": gs, "engine": _new_engine()})["game_state"]
    assert "seer_check_opportunity" in [event.type for event in result.events]
    assert "seer_check_skipped" in [event.type for event in result.events]
