# -*- coding: utf-8 -*-
"""
验证 player retry helper 拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-14

使用示例:
    >>> python -m pytest tests/agents/test_player_retry.py -q
"""

from __future__ import annotations

from werewolf_agent.agents.schemas import ActionType, AgentContext, RetryInfo, TaskType


def test_player_retry_helper_detects_repeated_signature() -> None:
    from werewolf_agent.agents.player_retry import check_repeat_error_signature

    retry = RetryInfo(error_code="parse_error")
    first_should_stop, signature = check_repeat_error_signature(
        retry,
        "not json",
        1,
        None,
        structured_output_mode="json_object",
    )
    second_should_stop, signature = check_repeat_error_signature(
        retry,
        "not json",
        2,
        signature,
        structured_output_mode="json_object",
    )

    assert first_should_stop is False
    assert second_should_stop is True
    assert retry.early_exit_reason == "repeat_error_signature: parse_error on attempts 1 and 2"


def test_player_retry_builds_vote_fallback_from_explicit_strategy_target() -> None:
    from werewolf_agent.agents.player_failures import fallback_reason
    from werewolf_agent.agents.player_retry import build_fallback_action

    context = AgentContext(
        agent_id="p08",
        task_type=TaskType.VOTE,
        legal_actions=[ActionType.VOTE],
        legal_targets=["p02", "p05"],
        strategy_directive={"_vote_fallback_target": "p05"},
    )

    fallback = build_fallback_action(
        context,
        fallback_reason=fallback_reason,
        fallback_speech=lambda _context: "",
    )

    assert fallback.action_type == ActionType.VOTE
    assert fallback.target_id == "p05"
    assert "p05" not in fallback.reason


def test_player_retry_vote_fallback_uses_seer_claim_evidence() -> None:
    from werewolf_agent.agents.player_failures import fallback_reason
    from werewolf_agent.agents.player_retry import build_fallback_action

    context = AgentContext(
        agent_id="p08",
        task_type=TaskType.VOTE,
        legal_actions=[ActionType.VOTE],
        legal_targets=["p02", "p05"],
        salience_items=[
            {"type": "seer_claim", "target": "p02", "alignment": "werewolf"},
        ],
    )

    fallback = build_fallback_action(
        context,
        fallback_reason=fallback_reason,
        fallback_speech=lambda _context: "",
    )

    assert fallback.target_id == "p02"


def test_player_retry_speech_fallback_uses_injected_speech_builder() -> None:
    from werewolf_agent.agents.player_failures import fallback_reason
    from werewolf_agent.agents.player_retry import build_fallback_action

    context = AgentContext(
        agent_id="p08",
        task_type=TaskType.WOLF_DISCUSSION,
        legal_actions=[ActionType.SPEECH],
        legal_targets=[],
    )

    fallback = build_fallback_action(
        context,
        fallback_reason=fallback_reason,
        fallback_speech=lambda _context: "狼队夜间兜底发言",
    )

    assert fallback.speech == "狼队夜间兜底发言"


def test_player_retry_public_speech_fallback_matches_real_hidden_flow() -> None:
    from werewolf_agent.agents.player_failures import fallback_reason
    from werewolf_agent.agents.player_retry import build_fallback_action

    context = AgentContext(
        agent_id="p08",
        task_type=TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH],
        legal_targets=["p02"],
    )

    speech_calls = []
    fallback = build_fallback_action(
        context,
        fallback_reason=fallback_reason,
        fallback_speech=lambda _context: speech_calls.append(_context) or "不应公开",
    )

    assert fallback.speech == ""
    assert speech_calls == []


def test_semantic_repair_uses_complete_authoritative_public_claim_ledger() -> None:
    from werewolf_agent.agents.schemas import SpeechPlayerAction
    from werewolf_agent.agents.semantic_repair_audit import build_semantic_repair_audit

    context = AgentContext(
        agent_id="p08", task_type=TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH], legal_targets=["p02"],
        recent_transcript=[], public_summary="",
        public_claim_ledger=[{
            "event_index": 0, "speaker": "p05", "text": "我是预言家",
        }],
    )
    source = SpeechPlayerAction(
        action_type=ActionType.SPEECH, target_id="p02",
        speech="p05声称自己是预言家，我怀疑p02。", reason="公开引用",
        confidence=0.5,
    )
    final = source.model_copy(update={"speech": "p05声称自己是预言家，我仍怀疑p02。"})

    audit = build_semantic_repair_audit(context, source, final, success=True)

    assert audit["verified_claim_count"] == 1
    assert audit["retained_verified_claim_count"] == 1


def test_semantic_repair_gate_rejects_changed_target_with_partial_claim_retention() -> None:
    """成功门必须同时保持源目标和全部已验证论点。"""
    from werewolf_agent.agents.schemas import SpeechPlayerAction
    from werewolf_agent.agents.semantic_repair_audit import (
        semantic_repair_retains_verified_claim,
    )

    context = AgentContext(
        agent_id="p08", task_type=TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH], legal_targets=["p02", "p04"],
        public_claim_ledger=[
            {"speaker": "p05", "text": "我是预言家"},
            {"speaker": "p06", "text": "我是女巫"},
        ],
    )
    source = SpeechPlayerAction(
        target_id="p02",
        speech="p05声称自己是预言家，p06声称自己是女巫，我怀疑p02。",
        reason="公开引用",
        confidence=0.5,
    )
    changed = source.model_copy(update={
        "target_id": "p04",
        "speech": "p05声称自己是预言家，我怀疑p04。",
    })

    assert semantic_repair_retains_verified_claim(context, source, changed) is False


def test_semantic_repair_gate_rejects_introduced_claim() -> None:
    """即使保留全部已验证论点，修复结果也不得新增事实 claim。"""
    from werewolf_agent.agents.schemas import SpeechPlayerAction
    from werewolf_agent.agents.semantic_repair_audit import (
        semantic_repair_retains_verified_claim,
    )

    context = AgentContext(
        agent_id="p08", task_type=TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH], legal_targets=["p02"],
        public_claim_ledger=[
            {"speaker": "p05", "text": "我是预言家"},
            {"speaker": "p06", "text": "我是女巫"},
        ],
    )
    source = SpeechPlayerAction(
        target_id="p02",
        speech="p05声称自己是预言家，p06声称自己是女巫，我怀疑p02。",
        reason="公开引用",
        confidence=0.5,
    )
    introduced = source.model_copy(update={
        "speech": (
            "p05声称自己是预言家，p06声称自己是女巫，"
            "p07声称自己是猎人，我怀疑p02。"
        ),
    })

    assert semantic_repair_retains_verified_claim(context, source, introduced) is False


def test_semantic_terminal_fallback_preserves_all_verified_claims() -> None:
    """终态 fallback 保持目标、全部已验证论点且不引入新 claim。"""
    from werewolf_agent.agents.schemas import FallbackAction, SpeechPlayerAction
    from werewolf_agent.agents.semantic_repair_audit import (
        build_semantic_repair_audit,
        preserve_verified_claim_in_fallback,
    )

    context = AgentContext(
        agent_id="p08", task_type=TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH], legal_targets=["p02"],
        public_claim_ledger=[
            {"speaker": "p05", "text": "我是预言家"},
            {"speaker": "p06", "text": "我是女巫"},
        ],
    )
    source = SpeechPlayerAction(
        target_id="p02",
        speech="p05声称自己是预言家，p06声称自己是女巫，我怀疑p02。",
        reason="公开引用",
        confidence=0.5,
    )

    fallback = preserve_verified_claim_in_fallback(
        context,
        source,
        FallbackAction(action_type=ActionType.SPEECH),
    )
    audit = build_semantic_repair_audit(context, source, fallback, success=False)

    assert audit["target_preserved"] is True
    assert audit["introduced_claim_count"] == 0
    assert audit["retained_verified_claim_count"] == audit["verified_claim_count"] == 2


def test_generic_fallback_classification_uses_actual_template_family() -> None:
    from werewolf_agent.agents.player_fallback_speech import (
        build_fallback_speech,
        generic_fallback_speech_used,
    )

    targeted = AgentContext(
        agent_id="p08", task_type=TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH], legal_targets=["p02"],
    )
    defense = targeted.model_copy(update={"task_type": TaskType.DEFENSE_SPEECH})
    wolf_discussion = targeted.model_copy(update={
        "task_type": TaskType.WOLF_DISCUSSION,
        "own_role": "werewolf",
    })
    sheriff = targeted.model_copy(update={
        "task_type": TaskType.SHERIFF_SPEECH,
        "legal_targets": [],
    })
    generic = targeted.model_copy(update={"legal_targets": []})

    assert generic_fallback_speech_used(targeted, build_fallback_speech(targeted)) is False
    assert generic_fallback_speech_used(defense, build_fallback_speech(defense)) is False
    assert (
        generic_fallback_speech_used(
            wolf_discussion, build_fallback_speech(wolf_discussion)
        )
        is False
    )
    assert generic_fallback_speech_used(sheriff, build_fallback_speech(sheriff)) is False
    assert generic_fallback_speech_used(generic, build_fallback_speech(generic)) is True
    assert generic_fallback_speech_used(generic, "") is False
    assert generic_fallback_speech_used(defense, "信息不足，继续观察。") is True
    assert generic_fallback_speech_used(targeted, "我是好人，我怀疑p02。") is False
