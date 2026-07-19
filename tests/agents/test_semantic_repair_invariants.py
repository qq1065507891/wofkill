# -*- coding: utf-8 -*-
"""
验证语义修复 V2 只拒绝不受支持的公开声明及语义关系篡改。

作者: Project contributors
创建日期: 2026-07-14
修改日期: 2026-07-19
"""

from __future__ import annotations

import pytest

from werewolf_agent.agents.schemas import (
    ActionType,
    AgentContext,
    SpeechPlayerAction,
    TaskType,
)


def _context() -> AgentContext:
    return AgentContext(
        agent_id="p08",
        task_type=TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH],
        legal_targets=["p02"],
        public_claim_ledger=[
            {"speaker": "p05", "text": "我认为p02是狼人"},
            {"speaker": "p06", "text": "我认为p02是狼人"},
        ],
    )


def _action(speech: str) -> SpeechPlayerAction:
    return SpeechPlayerAction(
        target_id="p02",
        speech=speech,
        reason="引用公开发言",
        confidence=0.5,
    )


def test_claim_keys_keep_speakers_independent_for_the_same_target() -> None:
    from werewolf_agent.evaluation.balance_public_claims import (
        public_claim_audit_keys,
    )

    claims, _ = public_claim_audit_keys(
        "p05声称p02是狼人；p06声称p02是狼人。",
        [("p05", "我认为p02是狼人"), ("p06", "我认为p02是狼人")],
    )

    assert len(claims) == 2
    assert {claim.target for claim in claims} == {"p02"}
    assert {claim.speaker_attribution for claim in claims} == {"p05", "p06"}


def test_semantic_repair_rejects_changed_existing_speaker_attribution() -> None:
    from werewolf_agent.agents.semantic_repair_audit import (
        validate_semantic_repair,
    )

    source = _action("p05声称p02是狼人，我怀疑p02。")
    changed = _action("p06声称p02是狼人，我怀疑p02。")

    result = validate_semantic_repair(_context(), source, changed)

    assert result.accepted is False
    assert result.reason_codes == ("speaker_attribution_changed",)
    assert result.audit["speaker_attribution_preserved"] is False


def test_semantic_repair_rejects_changed_negation_relation() -> None:
    from werewolf_agent.agents.semantic_repair_audit import (
        validate_semantic_repair,
    )

    source = _action("p05声称p02是狼人，我怀疑p02。")
    changed = _action("p05并未声称p02是狼人，我怀疑p02。")

    result = validate_semantic_repair(_context(), source, changed)

    assert result.accepted is False
    assert result.reason_codes == ("negation_changed",)
    assert result.audit["negation_preserved"] is False


def test_claim_key_records_prefix_negation_without_flipping_double_negation() -> None:
    from werewolf_agent.evaluation.balance_public_claims import (
        public_claim_audit_keys,
    )

    negated, _ = public_claim_audit_keys(
        "不能说p05声称p02是狼人。", [("p05", "我认为p02是狼人")]
    )
    assert {claim.negated for claim in negated} == {True}

    assertive, _ = public_claim_audit_keys(
        "不能否认p05声称p02是狼人。", [("p05", "我认为p02是狼人")]
    )
    assert {claim.negated for claim in assertive} == {False}


def test_semantic_repair_allows_supported_missing_attribution_completion() -> None:
    from werewolf_agent.agents.semantic_repair_audit import (
        validate_semantic_repair,
    )

    source = _action("有人声称p02是狼人，我怀疑p02。")
    attributed = _action("p05声称p02是狼人，我怀疑p02。")

    result = validate_semantic_repair(_context(), source, attributed)

    assert result.accepted is True
    assert result.reason_codes == ()
    assert result.audit["speaker_attribution_preserved"] is True
    assert result.audit["negation_preserved"] is True
    assert result.audit["introduced_claim_count"] == 0
    assert result.audit["semantic_gate_version"] == 2
    assert result.audit["unsupported_public_claim_count"] == 0


def test_semantic_repair_rejects_unsupported_public_claim() -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    source = _action("p05声称p02是狼人，我怀疑p02。")
    unsupported = _action(
        "p05声称p02是狼人，p07声称自己是猎人，我怀疑p02。"
    )

    result = validate_semantic_repair(_context(), source, unsupported)

    assert result.accepted is False
    assert result.reason_codes == ("unsupported_public_claim",)
    assert result.audit["unsupported_public_claim_count"] == 1


def test_semantic_repair_reports_all_rejection_reasons_in_stable_order() -> None:
    from werewolf_agent.agents.semantic_repair_audit import validate_semantic_repair

    source = _action("p05声称p02是狼人，我怀疑p02。")
    changed = _action(
        "p06并未声称p02是狼人，p07声称自己是猎人，我怀疑p02。"
    )

    result = validate_semantic_repair(_context(), source, changed)

    assert result.accepted is False
    assert result.reason_codes == (
        "unsupported_public_claim",
        "speaker_attribution_changed",
        "negation_changed",
    )
    assert result.audit["rejection_reason_codes"] == list(result.reason_codes)


@pytest.mark.parametrize(
    ("reason_code", "safe_explanation"),
    [
        ("unsupported_public_claim", "公开证据"),
        ("speaker_attribution_changed", "说话人归属"),
        ("negation_changed", "否定关系"),
    ],
)
def test_semantic_repair_fixed_messages_explain_each_reason_safely(
    reason_code: str,
    safe_explanation: str,
) -> None:
    from werewolf_agent.agents.semantic_repair_audit import (
        semantic_repair_correction_hint,
        semantic_repair_rejection_message,
    )

    message = semantic_repair_rejection_message((reason_code,))
    hint = semantic_repair_correction_hint((reason_code,))

    assert safe_explanation in message
    assert safe_explanation in hint
    assert hint.startswith("请")


@pytest.mark.parametrize(
    ("reason_codes", "included_terms", "excluded_terms"),
    [
        (
            ("unsupported_public_claim", "negation_changed"),
            ("公开证据", "否定关系"),
            ("说话人归属",),
        ),
        (
            ("speaker_attribution_changed",),
            ("说话人归属",),
            ("公开证据", "否定关系"),
        ),
    ],
)
def test_semantic_repair_fixed_messages_disclose_only_requested_reasons(
    reason_codes: tuple[str, ...],
    included_terms: tuple[str, ...],
    excluded_terms: tuple[str, ...],
) -> None:
    from werewolf_agent.agents.semantic_repair_audit import (
        semantic_repair_correction_hint,
        semantic_repair_rejection_message,
    )

    text = " ".join((
        semantic_repair_rejection_message(reason_codes),
        semantic_repair_correction_hint(reason_codes),
    ))

    assert all(term in text for term in included_terms)
    assert all(term not in text for term in excluded_terms)
    assert "SPEECH_SENTINEL_SHOULD_NOT_LEAK" not in text
    assert "p08是狼人" not in text
    assert "provider_error" not in text
    assert "不得新增任何事实" not in text
    assert "保留全部论点" not in text
