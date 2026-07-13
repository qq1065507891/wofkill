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
