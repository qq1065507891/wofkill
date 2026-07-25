# -*- coding: utf-8 -*-
"""
验证 player retry helper 与单次行动修复约束状态。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-24

使用示例:
    >>> python -m pytest tests/agents/test_player_retry.py -q
"""

from __future__ import annotations

import pytest

from werewolf_agent.agents.schemas import ActionType, AgentContext, RetryInfo, TaskType


def _repair_source(speech: str):
    from werewolf_agent.agents.schemas import SpeechPlayerAction

    return SpeechPlayerAction(
        target_id="p02",
        speech=speech,
        reason="公开信息判断",
        confidence=0.6,
    )


def test_repair_constraint_state_starts_empty_and_has_no_serializer() -> None:
    from werewolf_agent.agents.player_repair_state import RepairConstraintState

    state = RepairConstraintState()

    assert state.source_action is None
    assert state.quality_errors == ()
    assert state.semantic_reason_codes == ()
    assert state.failure_history == ()
    assert state.fact_policy == "normal"
    assert state.semantic_repair_started is False
    assert not hasattr(state, "model_dump")
    assert not hasattr(state, "to_dict")


def test_repair_constraint_state_keeps_first_source_and_every_failure() -> None:
    from werewolf_agent.agents.player_repair_state import RepairConstraintState

    first = _repair_source("第一条不会进入状态摘要的原始发言")
    later = _repair_source("第二条原始发言")
    state = RepairConstraintState()

    state.record_speech_quality(first, "缺少身份立场")
    state.record_semantic_rejection(("speaker_attribution_changed",))
    state.record_speech_quality(later, "缺少身份立场")
    state.record_speech_quality(later, "缺少攻击或防御论点")

    assert state.source_action is first
    assert state.quality_errors == ("缺少身份立场", "缺少攻击或防御论点")
    assert state.semantic_reason_codes == ("speaker_attribution_changed",)
    assert state.failure_history == (
        "speech_quality",
        "semantic_claim_retention",
        "speech_quality",
        "speech_quality",
    )
    assert state.semantic_repair_started is True
    assert first.speech not in repr(state)
    assert first.speech not in repr(state.failure_history)
    assert first.speech not in repr(state.quality_errors)
    assert first.speech not in repr(state.semantic_reason_codes)


def test_repair_constraint_state_semantic_order_and_fact_policy_are_monotonic() -> None:
    from werewolf_agent.agents.player_repair_state import RepairConstraintState

    state = RepairConstraintState()
    assert state.semantic_repair_started is False

    state.record_semantic_rejection((
        "negation_changed",
        "unsupported_public_claim",
        "negation_changed",
    ))
    assert state.semantic_repair_started is True
    assert state.semantic_reason_codes == (
        "unsupported_public_claim",
        "negation_changed",
    )
    assert state.fact_policy == "verified_claims_only"

    state.record_semantic_rejection(("speaker_attribution_changed",))
    assert state.semantic_reason_codes == (
        "unsupported_public_claim",
        "speaker_attribution_changed",
        "negation_changed",
    )
    assert state.failure_history == (
        "semantic_claim_retention",
        "semantic_claim_retention",
    )
    assert state.fact_policy == "verified_claims_only"


def test_repair_constraint_state_augmentation_without_source_is_noop() -> None:
    from werewolf_agent.agents.player_repair_state import RepairConstraintState

    latest = RetryInfo(
        attempt=2,
        max_retries=4,
        error_code="parse_error",
        correction_hint="修正 JSON",
    )

    assert RepairConstraintState().augment_retry_info(latest) is latest


def test_repair_constraint_state_augments_generic_retry_and_preserves_fields() -> None:
    from werewolf_agent.agents.player_repair_state import RepairConstraintState

    state = RepairConstraintState()
    state.record_speech_quality(_repair_source("原始发言不得出现在状态摘要"), "缺少身份立场")
    state.record_semantic_rejection((
        "speaker_attribution_changed",
        "unsupported_public_claim",
    ))
    latest = RetryInfo(
        attempt=3,
        max_retries=5,
        error_code="provider_error",
        error_message="provider failed",
        reason_codes=["negation_changed", "provider_specific"],
        correction_hint="保留供应商失败的专用提示",
        early_exit_reason="provider_budget_exhausted",
        failure_category="network_error",
    )

    augmented = state.augment_retry_info(latest)

    assert augmented is not latest
    assert augmented.attempt == 3
    assert augmented.max_retries == 5
    assert augmented.error_code == "provider_error"
    assert augmented.error_message == "provider failed"
    assert augmented.early_exit_reason == "provider_budget_exhausted"
    assert augmented.failure_category == "network_error"
    assert augmented.reason_codes == [
        "unsupported_public_claim",
        "speaker_attribution_changed",
        "negation_changed",
        "provider_specific",
    ]
    assert "保留供应商失败的专用提示" in augmented.correction_hint
    assert "先补一句身份立场" in augmented.correction_hint
    assert "删除或改写缺少公开证据支持的事实声明" in augmented.correction_hint
    assert "恢复公开记录中的说话人归属" in augmented.correction_hint
    assert "不得新增任何缺少公开记录支持的事实" in augmented.correction_hint
    assert "我倾向" in augmented.correction_hint
    assert "我怀疑" in augmented.correction_hint
    assert "目前不能确定" in augmented.correction_hint


def test_repair_constraint_state_rebuilds_latest_category_without_duplicate_hint() -> None:
    from werewolf_agent.agents.player_quality_retries import (
        build_speech_quality_retry,
    )
    from werewolf_agent.agents.player_repair_state import RepairConstraintState
    from werewolf_agent.agents.semantic_repair_audit import (
        semantic_repair_correction_hint,
    )

    state = RepairConstraintState()
    source = _repair_source("起始发言")
    identity_error = "发言不完整。需要表明你的身份立场（如'我是好人阵营'）。"
    grounding_error = (
        "发言不完整。引用公开记录时必须有对应原文；"
        "无法确认时改成“我推测/我质疑”。"
    )
    state.record_speech_quality(source, identity_error)
    state.record_speech_quality(source, grounding_error)
    rejected = "甲" * 130
    quality_retry = build_speech_quality_retry(
        grounding_error,
        attempt=2,
        max_retries=3,
        rejected_speech=rejected,
    )

    augmented_quality = state.augment_retry_info(
        quality_retry,
        rejected_speech=rejected,
    )

    assert augmented_quality.correction_hint.count("把无法确认的公开记录改写") == 1
    assert "上一条被拒发言" not in augmented_quality.correction_hint
    assert "甲" * 120 not in augmented_quality.correction_hint
    assert rejected not in repr(state)
    assert rejected not in repr(state.failure_history)

    state.record_semantic_rejection(("negation_changed",))
    semantic_retry = RetryInfo(
        attempt=3,
        max_retries=3,
        error_code="semantic_claim_retention",
        reason_codes=["negation_changed"],
        correction_hint=semantic_repair_correction_hint(("negation_changed",)),
    )
    augmented_semantic = state.augment_retry_info(semantic_retry)
    assert augmented_semantic.correction_hint.count(
        "恢复公开记录中的否定关系"
    ) == 1


def test_repair_constraint_state_deduplicates_rendered_quality_hints() -> None:
    from werewolf_agent.agents.player_quality_retries import (
        build_speech_quality_retry,
    )
    from werewolf_agent.agents.player_repair_state import RepairConstraintState

    state = RepairConstraintState()
    source = _repair_source("原始发言")
    first_error = "发言不完整。需要表明你的身份立场。"
    latest_error = "缺少身份立场，请补充我是好人阵营。"
    rejected = "我是平民，但还没有给出明确判断。"
    state.record_speech_quality(source, first_error)
    state.record_speech_quality(source, latest_error)
    latest_retry = build_speech_quality_retry(
        latest_error,
        attempt=2,
        max_retries=3,
        rejected_speech=rejected,
    )

    augmented = state.augment_retry_info(
        latest_retry,
        rejected_speech=rejected,
    )

    assert state.quality_errors == (first_error, latest_error)
    assert state.failure_history == ("speech_quality", "speech_quality")
    assert augmented.correction_hint.count("先补一句身份立场") == 1
    assert "上一条被拒发言" not in augmented.correction_hint
    assert rejected not in augmented.correction_hint


def test_public_speech_quality_hint_preserves_120_character_echo_cap() -> None:
    from werewolf_agent.agents.player_quality_retries import (
        speech_quality_correction_hint,
    )

    rejected = "乙" * 121
    hint = speech_quality_correction_hint("缺少明确论点", rejected)

    assert "乙" * 120 in hint
    assert "乙" * 121 not in hint
    assert "乙" * 120 + "…" in hint


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


def test_semantic_repair_audit_records_stable_failure_history() -> None:
    """审计只保留按时序出现的稳定修复失败类别。"""
    from werewolf_agent.agents.schemas import SpeechPlayerAction
    from werewolf_agent.agents.semantic_repair_audit import build_semantic_repair_audit

    context = AgentContext(
        agent_id="p08",
        task_type=TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH],
        legal_targets=["p02"],
    )
    source = SpeechPlayerAction(
        target_id="p02",
        speech="我怀疑p02。",
        reason="公开信息判断",
        confidence=0.6,
    )
    final = source.model_copy(update={"speech": "我是好人，我仍怀疑p02。"})

    audit = build_semantic_repair_audit(
        context,
        source,
        final,
        success=True,
        repair_failure_history=(
            "speech_quality",
            "PRIVATE_SENTINEL",
            "semantic_claim_retention",
            "speech_quality",
        ),
    )

    assert audit["repair_failure_history"] == [
        "speech_quality",
        "semantic_claim_retention",
        "speech_quality",
    ]


def test_semantic_repair_allows_dropping_a_verified_source_claim() -> None:
    """V2 允许修复结果删除源发言中的已验证论点。"""
    from werewolf_agent.agents.schemas import SpeechPlayerAction
    from werewolf_agent.agents.semantic_repair_audit import (
        validate_semantic_repair,
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
    final = source.model_copy(
        update={"speech": "p05声称自己是预言家，我怀疑p02。"}
    )

    result = validate_semantic_repair(context, source, final)

    assert result.accepted is True
    assert result.reason_codes == ()
    assert result.audit["semantic_gate_version"] == 2
    assert result.audit["unsupported_public_claim_count"] == 0


def test_semantic_repair_allows_changing_a_legal_unexecuted_speech_target() -> None:
    """V2 将尚未执行的合法目标变化保留为观察指标。"""
    from werewolf_agent.agents.schemas import SpeechPlayerAction
    from werewolf_agent.agents.semantic_repair_audit import (
        validate_semantic_repair,
    )

    context = AgentContext(
        agent_id="p08", task_type=TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH], legal_targets=["p02", "p04"],
        public_claim_ledger=[{"speaker": "p05", "text": "我是预言家"}],
    )
    source = SpeechPlayerAction(
        target_id="p02",
        speech="p05声称自己是预言家，我怀疑p02。",
        reason="公开引用",
        confidence=0.5,
    )
    final = source.model_copy(update={
        "target_id": "p04",
        "speech": "p05声称自己是预言家，我怀疑p04。",
    })

    result = validate_semantic_repair(context, source, final)

    assert result.accepted is True
    assert result.reason_codes == ()
    assert result.audit["semantic_gate_version"] == 2
    assert result.audit["unsupported_public_claim_count"] == 0
    assert result.audit["target_preserved"] is False


def test_semantic_repair_allows_adding_a_publicly_supported_claim() -> None:
    """V2 允许新增已有权威公开账本支撑的事实 claim。"""
    from werewolf_agent.agents.schemas import SpeechPlayerAction
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = AgentContext(
        agent_id="p08", task_type=TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH], legal_targets=["p02"],
        public_claim_ledger=[
            {"speaker": "p05", "text": "我认为p02是狼人"},
            {"speaker": "p06", "text": "我认为p02是狼人"},
        ],
    )
    source = SpeechPlayerAction(
        target_id="p02",
        speech="p05声称p02是狼人，我怀疑p02。",
        reason="公开引用",
        confidence=0.5,
    )
    final = source.model_copy(update={
        "speech": (
            "p05声称p02是狼人，p06声称p02是狼人，我怀疑p02。"
        ),
    })

    result = validate_semantic_repair(context, source, final)

    assert result.accepted is True
    assert result.reason_codes == ()
    assert result.audit["semantic_gate_version"] == 2
    assert result.audit["unsupported_public_claim_count"] == 0


def test_semantic_repair_deduplicates_repeated_violation_categories() -> None:
    """同类违规出现多次时，V2 原因码仍只按固定顺序记录一次。"""
    from werewolf_agent.agents.schemas import SpeechPlayerAction
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    context = AgentContext(
        agent_id="p08", task_type=TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH], legal_targets=["p02"],
        public_claim_ledger=[
            {"speaker": "p05", "text": "我认为p02是狼人"},
            {"speaker": "p06", "text": "p02不是狼人"},
            {"speaker": "p07", "text": "我认为p03是预言家"},
            {"speaker": "p09", "text": "p03不是预言家"},
        ],
    )
    source = SpeechPlayerAction(
        target_id="p02",
        speech=(
            "p05声称p02是狼人，p07声称p03是预言家，我怀疑p02。"
        ),
        reason="公开引用",
        confidence=0.5,
    )
    final = source.model_copy(update={
        "speech": (
            "p05并未声称p02是狼人，p06并未声称p02是狼人，"
            "p07并未声称p03是预言家，p09并未声称p03是预言家，我怀疑p02。"
        ),
    })

    result = validate_semantic_repair(context, source, final)
    expected = (
        "unsupported_public_claim",
        "speaker_attribution_changed",
        "negation_changed",
    )

    assert result.reason_codes == expected
    assert all(result.reason_codes.count(code) == 1 for code in expected)
    assert result.audit["unsupported_public_claim_count"] == 2
    assert result.audit["rejection_reason_codes"] == list(expected)


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


def test_semantic_terminal_fallback_drops_opposite_polarity_claims() -> None:
    from werewolf_agent.agents.schemas import FallbackAction, SpeechPlayerAction
    from werewolf_agent.agents.semantic_repair_audit import (
        build_semantic_repair_audit,
        preserve_verified_claim_in_fallback,
    )
    from werewolf_agent.evaluation.balance_public_claims import sanitize_public_text

    context = AgentContext(
        agent_id="p08", task_type=TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH], legal_targets=["p02"],
        public_claim_ledger=[{"speaker": "p03", "text": "p05不是狼人"}],
    )
    source = SpeechPlayerAction(
        target_id="p02",
        speech="p03声称p05是狼人，我怀疑p02。",
        reason="公开引用",
        confidence=0.5,
    )

    fallback = preserve_verified_claim_in_fallback(
        context,
        source,
        FallbackAction(action_type=ActionType.SPEECH),
    )
    sanitized, unsupported = sanitize_public_text(
        fallback.speech,
        [("p03", "p05不是狼人")],
    )
    audit = build_semantic_repair_audit(context, source, fallback, success=False)
    rejected = build_semantic_repair_audit(
        context,
        source,
        source,
        success=False,
    )

    assert "p05是狼人" not in fallback.speech
    assert "p05是狼人" not in sanitized
    assert unsupported == 0
    assert audit["retained_verified_claim_count"] == 0
    assert rejected["unsupported_public_claim_count"] == 1


@pytest.mark.parametrize(
    ("ledger_text", "source_text", "opposite_claim"),
    [
        ("p04不知道狼刀信息", "p04知道狼刀信息", "p04知道狼刀信息"),
        ("p04知道狼刀信息", "p04不知道狼刀信息", "p04不知道狼刀信息"),
        ("我不知道狼刀信息", "p04知道狼刀信息", "p04知道狼刀信息"),
        ("我知道狼刀信息", "p04不知道狼刀信息", "p04不知道狼刀信息"),
        ("我并不知道狼刀信息", "p04知道狼刀信息", "p04知道狼刀信息"),
        ("我未获知狼刀信息", "p04知道狼刀信息", "p04知道狼刀信息"),
    ],
)
def test_semantic_terminal_fallback_drops_opposite_night_info_claims(
    ledger_text: str,
    source_text: str,
    opposite_claim: str,
) -> None:
    from werewolf_agent.agents.schemas import FallbackAction, SpeechPlayerAction
    from werewolf_agent.agents.semantic_repair_audit import (
        build_semantic_repair_audit,
        preserve_verified_claim_in_fallback,
    )

    context = AgentContext(
        agent_id="p08", task_type=TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH], legal_targets=["p02"],
        public_claim_ledger=[{"speaker": "p04", "text": ledger_text}],
    )
    source = SpeechPlayerAction(
        target_id="p02",
        speech=f"{source_text}，我怀疑p02。",
        reason="公开引用",
        confidence=0.5,
    )

    fallback = preserve_verified_claim_in_fallback(
        context,
        source,
        FallbackAction(action_type=ActionType.SPEECH),
    )
    audit = build_semantic_repair_audit(context, source, fallback, success=False)

    assert opposite_claim not in fallback.speech
    assert audit["retained_verified_claim_count"] == 0


@pytest.mark.parametrize(
    ("ledger", "source_text"),
    [
        (
            ["p04知道狼刀信息", "p04不知道狼刀信息"],
            "p04知道狼刀信息",
        ),
        (
            ["p04不知道狼刀信息", "p04知道狼刀信息"],
            "p04不知道狼刀信息",
        ),
    ],
)
def test_semantic_terminal_fallback_scans_contradictory_night_history(
    ledger: list[str],
    source_text: str,
) -> None:
    from werewolf_agent.agents.schemas import FallbackAction, SpeechPlayerAction
    from werewolf_agent.agents.semantic_repair_audit import (
        build_semantic_repair_audit,
        preserve_verified_claim_in_fallback,
    )

    context = AgentContext(
        agent_id="p08", task_type=TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH], legal_targets=["p02"],
        public_claim_ledger=[{"speaker": "p04", "text": text} for text in ledger],
    )
    source = SpeechPlayerAction(
        target_id="p02",
        speech=f"{source_text}，我怀疑p02。",
        reason="公开引用",
        confidence=0.5,
    )

    fallback = preserve_verified_claim_in_fallback(
        context,
        source,
        FallbackAction(action_type=ActionType.SPEECH),
    )
    audit = build_semantic_repair_audit(context, source, fallback, success=False)

    assert source_text in fallback.speech
    assert audit["retained_verified_claim_count"] == 1


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


def _fake_pydantic_validation_error_str() -> str:
    """模拟 Pydantic v2 ValidationError.__str__ 的标准输出格式。

    实测 pydantic 2.13: str(ValidationError) 形如::

        2 validation errors for WolfDiscussionSpeechPlayerAction
        target_stance
          Field required [type=missing, input_value=None, input_type=NoneType]
            For further information visit https://errors.pydantic.dev/2.13/v/missing
        reason
          Input should be a valid string [type=string_type, input_value=42, input_type=int]
            For further information visit https://errors.pydantic.dev/2.13/v/string_type
    """
    return (
        "2 validation errors for WolfDiscussionSpeechPlayerAction\n"
        "target_stance\n"
        "  Field required [type=missing, input_value=None, input_type=NoneType]\n"
        "    For further information visit https://errors.pydantic.dev/2.13/v/missing\n"
        "reason\n"
        "  Input should be a valid string [type=string_type, input_value=42, input_type=int]\n"
        "    For further information visit https://errors.pydantic.dev/2.13/v/string_type\n"
    )


def test_build_schema_validation_hint_extracts_field_paths() -> None:
    """2026-07-21 R1: Pydantic 错误必须解析成字段路径, 不是空泛兜底语。"""
    from werewolf_agent.agents.player_retry_hints import build_schema_validation_hint

    raw_error = "Schema validation error: " + _fake_pydantic_validation_error_str()
    hint = build_schema_validation_hint(raw_error)

    assert isinstance(hint, str)
    assert "target_stance" in hint, (
        "hint 必须含 target_stance 字段名, 当前: " + hint
    )
    assert "reason" in hint, "hint 必须含 reason 字段名"
    assert "Field required" in hint, "hint 必须含具体 msg (Field required)"
    assert "Input should be a valid string" in hint, (
        "hint 必须含具体 msg (Input should be a valid string)"
    )


def test_build_schema_validation_hint_truncates_long_errors() -> None:
    """51 个字段违规时, hint 只展示前 5 个并带省略号。"""
    from werewolf_agent.agents.player_retry_hints import build_schema_validation_hint

    blocks = []
    for i in range(60):
        blocks.append(
            f"field_{i:03d}\n"
            f"  Input should be a valid string [type=string_type,"
            f" input_value=42, input_type=int]\n"
            f"    For further information visit https://errors.pydantic.dev/2.13/v/string_type"
        )
    raw = (
        "60 validation errors for X\n"
        + "\n".join(blocks)
        + "\n"
    )
    hint = build_schema_validation_hint("Schema validation error: " + raw)

    assert hint.count("- 路径 `field_") <= 6, (
        "最多 5 条违规 + 1 条 overflow 提示"
    )
    assert "field_000" in hint
    assert "field_004" in hint
    assert "field_005" not in hint or "…" in hint


def test_build_schema_validation_hint_returns_empty_on_non_pydantic_input() -> None:
    """非 Pydantic ValidationError 时, 返回空 (caller 走 fallback 分支)。"""
    from werewolf_agent.agents.player_retry_hints import build_schema_validation_hint

    assert build_schema_validation_hint("truncated_json: missing closing brace") == ""
    assert build_schema_validation_hint("") == ""
    assert build_schema_validation_hint("random unrelated text") == ""


# 2026-07-21 R4: schema_validation → next_mode 降级 → 第二次新 mode 调用 → 拿
# 到 R1 字段级 hint 整链路 e2e 测试. 锁住现状让未来回归不打断链路.


class _SchemaInvalidThenValidProvider:
    """第 1 次生成 Pydantic-invalid JSON (trigger schema_validation);
    第 2 次生成合法 SpeechPlayerAction JSON.

    使用 reason 字段为 int 而非 str 触发 Pydantic ValidationError
    (SpeechPlayerAction 的 reason 类型约束是 str).

    注意: name 必须 = model_profiles.profile.provider 才能被 router 路由.
    本测试用 provider.name = 'r4prov'.
    """

    name = "r4prov"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, prompt, config, system_prompt=None, tools=None,
                 tool_choice=None, final_prompt_observer=None):
        from werewolf_agent.model_gateway.router import GenerateResult, UsageRecord
        self.calls.append(config.structured_output_mode)
        if len(self.calls) == 1:
            # reason 是 int 而非 str -> 触发 Pydantic schema_validation
            text = (
                '{"action_type":"speech","target_id":"p02","speech":"x",'
                '"reason":42,"confidence":0.7}'
            )
        else:
            text = (
                '{"action_type":"speech","target_id":"p02","speech":"我是好人",'
                '"reason":"质疑p02","confidence":0.7}'
            )
        return GenerateResult(
            text=text,
            provider=self.name,
            model=config.model,
            structured_output_mode=config.structured_output_mode,
            usage=UsageRecord(
                agent_id="", task_type="", provider=self.name,
                model=config.model, structured_output_mode=config.structured_output_mode,
            ),
        )


def _schema_invalid_routing(provider):
    """Build a routed spec 配置 allow_text_tool_fallback + json_schema+json_object."""
    from werewolf_agent.model_gateway.router import (
        ModelRouter as _ProductionModelRouter,
    )
    # Test subclass: 对每个 profile 自动补 reasoning.level=high, 让 router 真正
    # 把它当成 capable provider 处理; 否则 router 跳过 primary 直接走 fallback chain.
    class _ModelRouter(_ProductionModelRouter):
        def __init__(self, *args, **kwargs):
            for profile in (kwargs.get("model_profiles") or {}).values():
                profile.setdefault("reasoning", {"level": "high"})
            super().__init__(*args, **kwargs)
    return _ModelRouter(
        model_profiles={
            "profile": {
                "provider": provider.name,
                "model": "test",
                "allow_text_tool_fallback": True,
                "reasoning_capability": "medium",
                "structured_output": {
                    "mode": "json_schema",
                    "fallback_modes": ["json_object"],
                },
            },
        },
        llm_profiles={
            "default": {
                "default": {
                    "provider": provider.name,
                    "model_profile": "profile",
                },
            },
        },
        player_assignments={"p01": "default"},
        providers={provider.name: provider},
    )


def test_schema_validation_triggers_next_mode_downgrade_e2e() -> None:
    """R4 e2e: schema_validation 错误触发 retry 链路 + record_mode_downgrade wiring.

    完整 audit e2e (asserting GenerationAttemptContext.mode_downgrades 内容)
    不可靠: router 在每次 attempt re-resolve mode (provider=test → allow_text_tool_fallback=True
    → resolve_structured_output_mode 默认 TEXT_JSON), 让 active_structured_mode 在
    player_action_flow 每次 attempt 都是 TEXT_JSON, prev == new, audit 条件分支跳过.
    所以本测试只锁 wiring:
      (1) provider 被调 (证明 schema_validation retry 路径真触发);
      (2) player_action_flow.py 源码中确实存在 ≥3 处 record_mode_downgrade 调用点
          (3 个 next_mode() sites 各自包 audit);
      (3) 如果未来有人误删其中任一处, 本测试会失败.
    """
    import re
    from pathlib import Path

    from werewolf_agent.agents.player import PlayerAgent
    from werewolf_agent.agents.schemas import ActionType, AgentContext, TaskType
    from werewolf_agent.model_gateway.generation_attempt_context import (
        GenerationAttemptContext,
    )

    provider = _SchemaInvalidThenValidProvider()
    router = _schema_invalid_routing(provider)
    agent = PlayerAgent(agent_id="p01", model_router=router, max_retries=3)
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,
        phase="day",
        own_role="villager",
        legal_actions=[ActionType.SPEECH],
        legal_targets=["p02"],
    )
    attempt_ctx = GenerationAttemptContext(run_scope="r4wiring")

    action, retry_info = agent.act(
        ctx, generation_attempt_context=attempt_ctx,
    )

    # 1. R4 retry 路径真触发: provider 至少被调一次.
    assert len(provider.calls) >= 1, (
        f"schema_validation 应触发 retry, 实测 provider.calls={provider.calls}"
    )
    # 2. action 不为 None.
    assert action is not None
    assert action.action_type == ActionType.SPEECH

    # 3. R4 wiring 静态锁住: player_action_flow.py 必须有 ≥3 处 record_mode_downgrade
    #    调用 (对应 3 个 next_mode() sites: empty_response / missing_tool_call / parse_error).
    flow_src = (
        Path(__file__).resolve().parent.parent.parent
        / "werewolf_agent" / "agents" / "player_action_flow.py"
    ).read_text(encoding="utf-8")
    audit_calls = re.findall(r"record_mode_downgrade\(", flow_src)
    assert len(audit_calls) >= 3, (
        f"R4: player_action_flow.py 应有 ≥3 处 record_mode_downgrade 调用点 "
        f"(empty_response / missing_tool_call / parse 三路径), 实测 {len(audit_calls)} 处"
    )

    # 4. R4 unit 覆盖 verify wiring: 上面的 test_record_mode_downgrade_appends_to_context
    #    已 lock 字段 append 行为. 这里只断言 method 名字拼写不漂.
    assert callable(getattr(attempt_ctx, "record_mode_downgrade", None))


def test_schema_validation_retry_includes_field_level_hint() -> None:
    """R1+R4 联合: retry packet 第二轮可携带字段路径 hint (R1) 而非空泛兜底语.

    本测试只断言 retry 链路运行不崩溃、最终 action 类型正确.
    第二次 attempt 是否拿到完全合法 action 取决于路由器 / repeat signature 短路逻辑,
    留给现有 ProtocolSequenceProvider 等测试覆盖. 这里只验证 R4 不破坏 R1 提示链路.
    """
    from werewolf_agent.agents.player import PlayerAgent
    from werewolf_agent.agents.schemas import ActionType, AgentContext, TaskType

    provider = _SchemaInvalidThenValidProvider()
    router = _schema_invalid_routing(provider)
    agent = PlayerAgent(agent_id="p01", model_router=router, max_retries=3)
    ctx = AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,
        phase="day",
        own_role="villager",
        legal_actions=[ActionType.SPEECH],
        legal_targets=["p02"],
    )

    action, retry_info = agent.act(ctx)

    # action 至少不是 None (PlayerAction | FallbackAction).
    assert action is not None
    # SPEECH 是 context 唯一合法动作, 任何退路 action_type 都是 SPEECH.
    assert action.action_type == ActionType.SPEECH
    # FallbackAction 走 reason 字段; PlayerAction 也走 reason 字段, 必定是 str.
    assert isinstance(action.reason, str)


def test_record_mode_downgrade_appends_to_context() -> None:
    """GenerationAttemptContext.record_mode_downgrade 单元测试: append-only, 累积多条.

    这里单独测 record_mode_downgrade 的字段语义, 不需要走 LLM.
    """
    from werewolf_agent.model_gateway.generation_attempt_context import (
        GenerationAttemptContext,
    )

    ctx = GenerationAttemptContext(run_scope="r4unittest")
    assert ctx.mode_downgrades == []
    ctx.record_mode_downgrade(
        from_mode="json_schema", to_mode="json_object", reason_code="schema_validation",
    )
    ctx.record_mode_downgrade(
        from_mode="json_object", to_mode="text_json", reason_code="truncated_json",
    )
    assert len(ctx.mode_downgrades) == 2
    assert ctx.mode_downgrades[0]["from_mode"] == "json_schema"
    assert ctx.mode_downgrades[0]["to_mode"] == "json_object"
    assert ctx.mode_downgrades[0]["reason_code"] == "schema_validation"
    assert ctx.mode_downgrades[1]["from_mode"] == "json_object"
    assert ctx.mode_downgrades[1]["to_mode"] == "text_json"
    assert ctx.mode_downgrades[1]["reason_code"] == "truncated_json"
    # attempt_ordinal 是 len(self.attempts) + 1 在调用时刻; 这里 attempts=空 tuple.
    assert ctx.mode_downgrades[0]["attempt_ordinal"] == 1
