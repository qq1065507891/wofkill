# -*- coding: utf-8 -*-
"""
测试白天公开发言阶段的策略指令辅助函数。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.runtime.day_speech_directives import build_day_speech_base_directive
    >>> build_day_speech_base_directive("")
"""

from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.runtime.day_speech_directives import (
    build_day_speech_base_directive,
    build_empty_day_speech_fallback,
    build_sanitized_seer_claim_fallback,
    build_sheriff_election_record,
    build_sheriff_speech_directive,
    build_torn_badge_speech_state,
    collect_sheriff_election_speeches,
)


def _make_game_state() -> GameState:
    players = {
        "p01": PlayerState(id="p01", role="villager", alive=True),
        "p02": PlayerState(id="p02", role="seer", alive=True),
        "p03": PlayerState(id="p03", role="werewolf", alive=True),
    }
    return GameState(
        game_id="day_speech_directives",
        phase="day",
        day_number=1,
        players=players,
        events=[
            GameEvent(type="sheriff_speech", payload={"speaker": "p02", "text": "我是预言家，警徽流先p01后p03。"}),
            GameEvent(type="sheriff_speech", payload={"speaker": "p03", "text": "我认为p02逻辑不稳。"}),
        ],
    )


def test_build_day_speech_base_directive_contains_originality_and_style() -> None:
    """基础白天发言指令包含原创性、平安夜规则和风格提示。"""
    directive = build_day_speech_base_directive("发言短促，重视票型。")

    assert "anti_following_and_peace_night_rule" in directive
    assert "【发言原创性要求】" in directive["speech_originality"]
    assert "发言短促，重视票型。" in directive["speech_originality"]


def test_build_sheriff_speech_directive_distinguishes_silent_and_active() -> None:
    """警长无法发言和正常归票时使用不同提示。"""
    silent = build_sheriff_speech_directive(
        is_silenced=True,
        alive_others=["p02", "p03"],
    )
    active = build_sheriff_speech_directive(
        is_silenced=False,
        alive_others=["p02", "p03"],
    )

    assert "sheriff_silent" in silent
    assert "仍需提交 vote action" in silent["sheriff_silent"]
    assert active["sheriff_alive_others"] == ["p02", "p03"]
    assert "警长归票是核心职责" in active["sheriff_vote_push"]


def test_build_torn_badge_speech_state_states_no_sheriff() -> None:
    """撕徽后所有玩家都应知道本局无警长。"""
    assert build_torn_badge_speech_state() == "本局无警长；本轮发言顺序随机；无归票人。"


def test_collect_and_render_sheriff_election_record() -> None:
    """警上竞选发言记录会被收集并截断成摘要。"""
    speeches = collect_sheriff_election_speeches(_make_game_state())
    record = build_sheriff_election_record(speeches)

    assert speeches == [
        {"speaker": "p02", "text": "我是预言家，警徽流先p01后p03。"},
        {"speaker": "p03", "text": "我认为p02逻辑不稳。"},
    ]
    assert "以下是警上竞选环节各候选人发言的摘要" in record
    assert "[p02]" in record


def test_build_sheriff_election_record_returns_empty_when_missing() -> None:
    """没有警上发言时不注入摘要文本。"""
    assert build_sheriff_election_record([]) == ""


def test_day_speech_fallbacks_keep_existing_wording() -> None:
    """空发言和违规预言家宣称兜底文案保持稳定。"""
    assert build_empty_day_speech_fallback("p01", "p02") == (
        "我是p01，我认为目前场上信息不够明确。"
        "我关注p02的发言，需要更多信息来判断。"
    )
    assert build_sanitized_seer_claim_fallback("p01", "p02") == (
        "我是p01，目前信息不足，我需要先观察其他玩家的发言再做判断。"
        "我会重点关注p02的站边和投票倾向。"
    )
