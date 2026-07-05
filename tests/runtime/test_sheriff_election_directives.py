# -*- coding: utf-8 -*-
"""
测试警长竞选发言阶段的策略指令构建函数。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.runtime.sheriff_election_directives import sheriff_uses_seer_protocol
    >>> sheriff_uses_seer_protocol("seer", "")
"""

from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.runtime.sheriff_election_directives import (
    build_previous_sheriff_speech_instruction,
    build_seer_verification_rationale,
    build_sheriff_badge_flow_instruction,
    build_sheriff_election_speech_directive,
    build_sheriff_role_speech_hint,
    build_sheriff_seer_context,
    build_wolf_sheriff_election_directives,
    collect_previous_sheriff_speeches,
    sheriff_uses_seer_protocol,
)


def _make_game_state() -> GameState:
    players = {
        "p01": PlayerState(id="p01", role="seer", alive=True),
        "p02": PlayerState(id="p02", role="villager", alive=True),
        "p03": PlayerState(id="p03", role="werewolf", alive=True),
    }
    return GameState(
        game_id="sheriff_election_directives",
        phase="sheriff_election",
        players=players,
        events=[
            GameEvent(type="sheriff_speech", payload={"speaker": "p01", "text": "我怀疑p02，也会观察p03。"}),
            GameEvent(type="sheriff_speech", payload={"speaker": "p02", "text": "我觉得p04发言有问题。"}),
            GameEvent(type="sheriff_speech", payload={"speaker": "p03", "text": "当前候选人自己的发言不应出现。"}),
        ],
    )


def test_sheriff_uses_seer_protocol_for_true_or_fake_seer_only() -> None:
    """只有真预言家或悍跳预言家使用警徽流协议。"""
    assert sheriff_uses_seer_protocol("seer", "") is True
    assert sheriff_uses_seer_protocol("werewolf", "fake_seer") is True
    assert sheriff_uses_seer_protocol("villager", "") is False


def test_build_sheriff_badge_flow_instruction_requires_seer_protocol() -> None:
    """非预言家协议身份不应收到警徽流私有提示。"""
    assert build_sheriff_badge_flow_instruction(False) == ""
    assert "必须留两个晚上的验人对象" in build_sheriff_badge_flow_instruction(True)


def test_build_sheriff_seer_context_distinguishes_single_and_multi_claims() -> None:
    """单边预言家和多人跳预言家使用不同局势提示。"""
    assert "单边预言家" in build_sheriff_seer_context({"p01"}, uses_seer_protocol=True)
    assert "多人跳预言家" in build_sheriff_seer_context({"p01", "p02"}, uses_seer_protocol=False)
    assert build_sheriff_seer_context(set(), uses_seer_protocol=False) == ""


def test_collect_previous_sheriff_speeches_excludes_current_candidate() -> None:
    """前人发言摘要不包含当前候选人自己的历史文本。"""
    speeches = collect_previous_sheriff_speeches(_make_game_state(), "p03")

    assert speeches == [
        {"speaker": "p01", "text": "我怀疑p02，也会观察p03。"},
        {"speaker": "p02", "text": "我觉得p04发言有问题。"},
    ]


def test_build_previous_sheriff_speech_instruction_summarizes_covered_topics() -> None:
    """前人发言提示包含摘要和已覆盖玩家提醒。"""
    instruction = build_previous_sheriff_speech_instruction([
        {"speaker": "p01", "text": "我怀疑p02，也会观察p03。"},
        {"speaker": "p02", "text": "我觉得p04发言有问题。"},
    ])

    assert "【前人发言摘要】" in instruction
    assert "[p01]" in instruction
    assert "p01已分析过p02" in instruction
    assert "严禁照搬" in instruction


def test_build_previous_sheriff_speech_instruction_handles_first_speaker() -> None:
    """没有前人发言时提示当前候选人只能基于公开信息发言。"""
    instruction = build_previous_sheriff_speech_instruction([])

    assert "你是本轮第一个发言的候选人" in instruction
    assert "严禁编造" in instruction


def test_build_sheriff_election_speech_directive_merges_all_text_inputs() -> None:
    """竞选发言基础指令聚合风格、警徽流、预言家局势和前人发言提示。"""
    directive = build_sheriff_election_speech_directive(
        style_hint="强势但克制",
        task_hint="请多引用票型。",
        badge_flow_instruction="警徽流文本",
        seer_context="单边预言家文本",
        prev_speech_instruction="前人发言文本",
        other_candidates=["p02", "p03"],
    )

    assert directive["other_candidates"] == ["p02", "p03"]
    assert "强势但克制" in directive["sheriff_election_speech"]
    assert "警徽流文本" in directive["sheriff_election_speech"]
    assert "单边预言家文本" in directive["sheriff_election_speech"]
    assert "前人发言文本" in directive["sheriff_election_speech"]
    assert "禁止模板化" in directive["anti_template"]


def test_role_speech_hint_and_seer_rationale_are_role_gated() -> None:
    """角色发言提示和预言家查验理由提示按身份注入。"""
    assert "警上发言重点" in build_sheriff_role_speech_hint("witch")
    assert build_sheriff_role_speech_hint("seer") == ""
    assert "查验理由要求" in build_seer_verification_rationale("seer")
    assert build_seer_verification_rationale("villager") == ""


def test_build_wolf_sheriff_election_directives_handles_fake_seer_plan() -> None:
    """狼队警上私有指令应区分悍跳本人和暂未公开的悍跳队友。"""
    must_claim = build_wolf_sheriff_election_directives(
        wolf_assignment="fake_seer",
        wolf_plan={"fake_seer": "w1"},
        candidate_id="w1",
        fake_seer_publicly_claimed=False,
    )
    no_reveal = build_wolf_sheriff_election_directives(
        wolf_assignment="pusher",
        wolf_plan={"fake_seer": "w1"},
        candidate_id="w2",
        fake_seer_publicly_claimed=False,
    )
    already_public = build_wolf_sheriff_election_directives(
        wolf_assignment="pusher",
        wolf_plan={"fake_seer": "w1"},
        candidate_id="w2",
        fake_seer_publicly_claimed=True,
    )

    assert "wolf_sheriff_must_claim_seer" in must_claim
    assert "必须在这段发言中跳预言家" in must_claim["wolf_sheriff_must_claim_seer"]
    assert "wolf_no_reveal_seer" in no_reveal
    assert "绝不能站边TA或透露TA会跳预言家" in no_reveal["wolf_no_reveal_seer"]
    assert already_public == {}
