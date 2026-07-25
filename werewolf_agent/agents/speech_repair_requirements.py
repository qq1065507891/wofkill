# -*- coding: utf-8 -*-
"""
定义公开发言修复期间单调累积的结构化约束。

作者: Project contributors
创建日期: 2026-07-25

使用示例:
    >>> requirements = SpeechRepairRequirements.empty()
    >>> requirements.merge(required_target="p05").required_target
    'p05'
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, order=True)
class SpeechClaimKey:
    """保存可安全进入提示词的事实声明结构键，不携带原始发言。"""

    claim_type: str
    target: str
    role: str
    support_kind: str
    speaker_attribution: str
    negated: bool

    def to_prompt_value(self) -> list[str | bool]:
        """返回紧凑且顺序稳定的提示词值。"""
        return [
            self.claim_type,
            self.target,
            self.role,
            self.support_kind,
            self.speaker_attribution,
            self.negated,
        ]


@dataclass(frozen=True)
class SpeechRepairRequirements:
    """保存一次公开发言修复中不可丢失的累计约束。"""

    missing_requirements: tuple[str, ...] = ()
    required_target: str | None = None
    forbidden_claim_keys: tuple[SpeechClaimKey, ...] = ()
    preserve_negations: tuple[SpeechClaimKey, ...] = ()
    allowlisted_evidence_refs: tuple[SpeechClaimKey, ...] = ()

    def __post_init__(self) -> None:
        """把直接构造输入也规范化为不可变、确定顺序的值。"""
        object.__setattr__(
            self,
            "missing_requirements",
            tuple(sorted(set(self.missing_requirements))),
        )
        for field_name in (
            "forbidden_claim_keys",
            "preserve_negations",
            "allowlisted_evidence_refs",
        ):
            object.__setattr__(
                self,
                field_name,
                tuple(sorted(set(getattr(self, field_name)))),
            )

    @classmethod
    def empty(cls) -> SpeechRepairRequirements:
        """返回不施加任何修复条件的初始值。"""
        return cls()

    def merge(
        self,
        other: SpeechRepairRequirements | None = None,
        *,
        missing_requirements: Iterable[str] = (),
        required_target: str | None = None,
        forbidden_claim_keys: Iterable[SpeechClaimKey] = (),
        preserve_negations: Iterable[SpeechClaimKey] = (),
        allowlisted_evidence_refs: Iterable[SpeechClaimKey] = (),
    ) -> SpeechRepairRequirements:
        """单调合并约束；冲突目标必须显式失败，不能静默覆盖。"""
        incoming = other or SpeechRepairRequirements(
            missing_requirements=tuple(missing_requirements),
            required_target=required_target,
            forbidden_claim_keys=tuple(forbidden_claim_keys),
            preserve_negations=tuple(preserve_negations),
            allowlisted_evidence_refs=tuple(allowlisted_evidence_refs),
        )
        if (
            self.required_target
            and incoming.required_target
            and self.required_target != incoming.required_target
        ):
            raise ValueError(
                "conflicting required targets: "
                f"{self.required_target!r} != {incoming.required_target!r}"
            )
        return SpeechRepairRequirements(
            missing_requirements=(
                *self.missing_requirements,
                *incoming.missing_requirements,
            ),
            required_target=self.required_target or incoming.required_target,
            forbidden_claim_keys=(
                *self.forbidden_claim_keys,
                *incoming.forbidden_claim_keys,
            ),
            preserve_negations=(
                *self.preserve_negations,
                *incoming.preserve_negations,
            ),
            allowlisted_evidence_refs=(
                *self.allowlisted_evidence_refs,
                *incoming.allowlisted_evidence_refs,
            ),
        )

    def to_prompt_projection(self) -> dict[str, Any]:
        """仅投影稳定结构字段，禁止携带原文、供应商错误或自由文本历史。"""
        return {
            "missing": list(self.missing_requirements),
            "required_target": self.required_target,
            "forbidden_claims": [
                claim.to_prompt_value() for claim in self.forbidden_claim_keys
            ],
            "preserve_negations": [
                claim.to_prompt_value() for claim in self.preserve_negations
            ],
            "allowed_evidence": [
                claim.to_prompt_value()
                for claim in self.allowlisted_evidence_refs
            ],
        }


__all__ = ["SpeechClaimKey", "SpeechRepairRequirements"]
