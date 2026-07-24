# -*- coding: utf-8 -*-
"""
在单次玩家行动内累积发言质量与语义修复约束。

作者: Project contributors
创建日期: 2026-07-24

使用示例:
    >>> state = RepairConstraintState()
    >>> state.semantic_repair_started
    False
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from werewolf_agent.agents.player_quality_retries import (
    speech_quality_correction_hint,
)
from werewolf_agent.agents.schemas import PlayerAction, RetryInfo
from werewolf_agent.agents.semantic_repair_audit import (
    ordered_semantic_repair_reason_codes,
    semantic_repair_correction_hint,
)


FactPolicy = Literal["normal", "verified_claims_only"]
FailureCategory = Literal["speech_quality", "semantic_claim_retention"]

_NORMAL_FACT_POLICY_HINT = (
    "事实约束：事实声明应以公开记录为依据；可以使用主观判断表达当前立场。"
)
_VERIFIED_CLAIMS_ONLY_HINT = (
    "事实约束：不得新增任何缺少公开记录支持的事实；"
    "可以继续使用“我倾向”“我怀疑”“目前不能确定”等主观表达。"
)


@dataclass(repr=False)
class RepairConstraintState:
    """保存一次行动重试期间的累计修复约束，不执行外部读写。"""

    _source_action: PlayerAction | None = field(default=None, init=False)
    _quality_errors: list[str] = field(default_factory=list, init=False)
    _semantic_reason_codes: list[str] = field(default_factory=list, init=False)
    _failure_history: list[FailureCategory] = field(default_factory=list, init=False)
    _fact_policy: FactPolicy = field(default="normal", init=False)
    _semantic_repair_started: bool = field(default=False, init=False)

    @property
    def source_action(self) -> PlayerAction | None:
        """返回首次可修复发言；仅供当前行动内显式访问。"""
        return self._source_action

    @property
    def quality_errors(self) -> tuple[str, ...]:
        """按首次出现顺序返回去重后的质量约束。"""
        return tuple(self._quality_errors)

    @property
    def semantic_reason_codes(self) -> tuple[str, ...]:
        """按语义门固定顺序返回去重后的原因码。"""
        return tuple(self._semantic_reason_codes)

    @property
    def failure_history(self) -> tuple[FailureCategory, ...]:
        """返回保留重复项的时序失败类别。"""
        return tuple(self._failure_history)

    @property
    def fact_policy(self) -> FactPolicy:
        """返回当前单调事实约束策略。"""
        return self._fact_policy

    @property
    def semantic_repair_started(self) -> bool:
        """返回本次行动是否曾进入语义修复。"""
        return self._semantic_repair_started

    def record_speech_quality(self, source: PlayerAction, error: str) -> None:
        """记录一次质量失败，保留首个来源并去重约束文本。"""
        if self._source_action is None:
            self._source_action = source
        if error not in self._quality_errors:
            self._quality_errors.append(error)
        self._failure_history.append("speech_quality")

    def record_semantic_rejection(self, reason_codes: Iterable[str]) -> None:
        """记录一次语义拒绝，并按固定顺序合并原因码。"""
        self._semantic_repair_started = True
        self._semantic_reason_codes[:] = ordered_semantic_repair_reason_codes(
            (*self._semantic_reason_codes, *reason_codes)
        )
        self._failure_history.append("semantic_claim_retention")
        if "unsupported_public_claim" in self._semantic_reason_codes:
            self._fact_policy = "verified_claims_only"

    def augment_retry_info(
        self,
        latest_retry: RetryInfo,
        *,
        rejected_speech: str = "",
    ) -> RetryInfo:
        """保留最新失败元数据，并追加本次行动的全部累计约束。"""
        if self._source_action is None:
            return latest_retry

        ordered_semantic_codes = ordered_semantic_repair_reason_codes(
            (
                *latest_retry.reason_codes,
                *self._semantic_reason_codes,
            )
        )
        nonsemantic_codes = tuple(
            dict.fromkeys(
                code
                for code in latest_retry.reason_codes
                if code not in ordered_semantic_codes
            )
        )
        merged_reason_codes = (*ordered_semantic_codes, *nonsemantic_codes)
        correction_parts: list[str] = []
        if (
            latest_retry.error_code
            not in {"speech_quality", "semantic_claim_retention"}
            and latest_retry.correction_hint
        ):
            correction_parts.append(latest_retry.correction_hint)

        last_quality_index = len(self._quality_errors) - 1
        rendered_quality_parts: list[str] = []
        quality_part_index: dict[str, int] = {}
        for index, error in enumerate(self._quality_errors):
            canonical_hint = speech_quality_correction_hint(error)
            rendered_hint = canonical_hint
            if index == last_quality_index and rejected_speech:
                rendered_hint = speech_quality_correction_hint(error, rejected_speech)
            existing_index = quality_part_index.get(canonical_hint)
            if existing_index is None:
                quality_part_index[canonical_hint] = len(rendered_quality_parts)
                rendered_quality_parts.append(rendered_hint)
            elif index == last_quality_index and rejected_speech:
                # 保留首次约束位置，但让最终被拒原文只进入最新同类提示。
                rendered_quality_parts[existing_index] = rendered_hint
        correction_parts.extend(rendered_quality_parts)

        if self._semantic_reason_codes:
            correction_parts.append(
                semantic_repair_correction_hint(self._semantic_reason_codes)
            )
        correction_parts.append(
            _VERIFIED_CLAIMS_ONLY_HINT
            if self._fact_policy == "verified_claims_only"
            else _NORMAL_FACT_POLICY_HINT
        )

        return RetryInfo(
            attempt=latest_retry.attempt,
            max_retries=latest_retry.max_retries,
            error_code=latest_retry.error_code,
            error_message=latest_retry.error_message,
            reason_codes=list(merged_reason_codes),
            correction_hint="\n".join(correction_parts),
            early_exit_reason=latest_retry.early_exit_reason,
            failure_category=latest_retry.failure_category,
        )


__all__ = ["RepairConstraintState"]
