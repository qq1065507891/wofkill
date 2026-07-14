# -*- coding: utf-8 -*-
"""
验证语义修复不会改变公开声明的说话者归属或否定关系。

作者: Project contributors
创建日期: 2026-07-14
"""

from __future__ import annotations

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
        build_semantic_repair_audit,
        semantic_repair_retains_verified_claim,
    )

    source = _action("p05声称p02是狼人，我怀疑p02。")
    changed = _action("p06声称p02是狼人，我怀疑p02。")

    audit = build_semantic_repair_audit(_context(), source, changed, success=True)

    assert audit["speaker_attribution_preserved"] is False
    assert semantic_repair_retains_verified_claim(_context(), source, changed) is False


def test_semantic_repair_rejects_changed_negation_relation() -> None:
    from werewolf_agent.agents.semantic_repair_audit import (
        build_semantic_repair_audit,
        semantic_repair_retains_verified_claim,
    )

    source = _action("p05声称p02是狼人，我怀疑p02。")
    changed = _action("p05并未声称p02是狼人，我怀疑p02。")

    audit = build_semantic_repair_audit(_context(), source, changed, success=True)

    assert audit["negation_preserved"] is False
    assert semantic_repair_retains_verified_claim(_context(), source, changed) is False


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
        build_semantic_repair_audit,
        semantic_repair_retains_verified_claim,
    )

    source = _action("有人声称p02是狼人，我怀疑p02。")
    attributed = _action("p05声称p02是狼人，我怀疑p02。")

    audit = build_semantic_repair_audit(_context(), source, attributed, success=True)

    assert audit["speaker_attribution_preserved"] is True
    assert audit["negation_preserved"] is True
    assert audit["introduced_claim_count"] == 0
    assert semantic_repair_retains_verified_claim(_context(), source, attributed) is True
