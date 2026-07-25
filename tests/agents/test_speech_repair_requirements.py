# -*- coding: utf-8 -*-
"""
验证公开发言修复约束的不可变合并与安全提示投影。

作者: Project contributors
创建日期: 2026-07-25

使用示例:
    >>> python -m pytest tests/agents/test_speech_repair_requirements.py -q
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from werewolf_agent.agents.speech_repair_requirements import (
    SpeechClaimKey,
    SpeechRepairRequirements,
)
from werewolf_agent.agents.player_repair_state import RepairConstraintState
from werewolf_agent.agents.schemas import ActionType, AgentContext, PlayerAction, TaskType


def _claim(
    target: str,
    *,
    role: str = "预言家",
    speaker: str = "p03",
    negated: bool = False,
) -> SpeechClaimKey:
    return SpeechClaimKey(
        claim_type="player_claim",
        target=target,
        role=role,
        support_kind="role_assignment",
        speaker_attribution=speaker,
        negated=negated,
    )


def test_empty_requirements_are_frozen_and_use_immutable_collections() -> None:
    requirements = SpeechRepairRequirements.empty()

    assert requirements.missing_requirements == ()
    assert requirements.required_target is None
    assert requirements.forbidden_claim_keys == ()
    assert requirements.preserve_negations == ()
    assert requirements.allowlisted_evidence_refs == ()
    with pytest.raises(FrozenInstanceError):
        requirements.required_target = "p05"  # type: ignore[misc]


def test_merge_is_monotonic_deduplicated_and_deterministically_ordered() -> None:
    p07_claim = _claim("p07", role="猎人", speaker="p07")
    p03_claim = _claim("p03")
    initial = SpeechRepairRequirements.empty().merge(
        missing_requirements=("vote_leaning", "evidence_basis"),
        required_target="p05",
        preserve_negations=(p07_claim,),
        allowlisted_evidence_refs=(p03_claim,),
    )

    merged = initial.merge(
        missing_requirements=("identity_stance", "evidence_basis"),
        forbidden_claim_keys=(p07_claim, p03_claim, p07_claim),
        preserve_negations=(p03_claim,),
        allowlisted_evidence_refs=(p07_claim, p03_claim),
    )

    assert initial.missing_requirements == ("evidence_basis", "vote_leaning")
    assert merged.missing_requirements == (
        "evidence_basis",
        "identity_stance",
        "vote_leaning",
    )
    assert merged.required_target == "p05"
    assert merged.forbidden_claim_keys == (p03_claim, p07_claim)
    assert merged.preserve_negations == (p03_claim, p07_claim)
    assert merged.allowlisted_evidence_refs == (p03_claim, p07_claim)


def test_merge_rejects_conflicting_required_targets_instead_of_overwriting() -> None:
    requirements = SpeechRepairRequirements.empty().merge(required_target="p05")

    with pytest.raises(ValueError, match="conflicting required targets"):
        requirements.merge(required_target="p07")


def test_prompt_projection_is_compact_structured_and_allowlist_only() -> None:
    supported = _claim("p03")
    forbidden = _claim("p07", role="猎人", speaker="p07")
    requirements = SpeechRepairRequirements.empty().merge(
        missing_requirements=("evidence_basis",),
        required_target="p05",
        forbidden_claim_keys=(forbidden,),
        preserve_negations=(supported,),
        allowlisted_evidence_refs=(supported,),
    )

    projection = requirements.to_prompt_projection()

    assert projection == {
        "missing": ["evidence_basis"],
        "required_target": "p05",
        "forbidden_claims": [
            ["player_claim", "p07", "猎人", "role_assignment", "p07", False]
        ],
        "preserve_negations": [
            ["player_claim", "p03", "预言家", "role_assignment", "p03", False]
        ],
        "allowed_evidence": [
            ["player_claim", "p03", "预言家", "role_assignment", "p03", False]
        ],
    }
    assert "history" not in projection
    assert "speech" not in projection
    assert "provider" not in projection


def test_repair_state_projects_only_structured_constraints_after_rejection() -> None:
    """公开发言修复提示不回显被拒原文，只投影可验证约束。"""
    state = RepairConstraintState()
    context = AgentContext(
        agent_id="p01",
        task_type=TaskType.SPEECH,
        legal_actions=[ActionType.SPEECH],
        legal_targets=["p05"],
        public_claim_ledger=[{"speaker": "p03", "text": "我是预言家"}],
    )
    source = PlayerAction(
        action_type=ActionType.SPEECH,
        target_id="p05",
        speech="p03声称自己是预言家，我怀疑p05。",
        reason="公开记录",
        confidence=0.5,
    )
    rejected = source.model_copy(
        update={"speech": "p07声称自己是猎人，REJECTED_SPEECH_SENTINEL"}
    )

    state.record_speech_quality(source, "缺少身份立场")
    state.record_semantic_rejection(
        ("unsupported_public_claim",),
        context=context,
        rejected_action=rejected,
    )

    projection = state.requirements.to_prompt_projection()

    assert projection["required_target"] == "p05"
    assert projection["missing"] == ["speech_quality"]
    assert projection["forbidden_claims"]
    assert projection["allowed_evidence"]
    assert "REJECTED_SPEECH_SENTINEL" not in repr(projection)


def test_repair_state_keeps_the_first_repair_target() -> None:
    """后续草稿不能把首轮公开发言目标改写为冲突约束。"""
    state = RepairConstraintState()
    first = PlayerAction(
        action_type=ActionType.SPEECH,
        target_id="p05",
        speech="我怀疑p05。",
        reason="公开记录",
        confidence=0.5,
    )
    later = first.model_copy(update={"target_id": "p07"})

    state.record_speech_quality(first, "缺少身份立场")
    state.record_speech_quality(later, "缺少投票倾向")

    assert state.requirements.required_target == "p05"
