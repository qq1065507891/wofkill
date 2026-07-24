# -*- coding: utf-8 -*-
"""
维护一次玩家动作生成跨结构化修复轮次的连续证据链。

作者: Project contributors
创建日期: 2026-07-13
修改日期: 2026-07-24
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import uuid

from werewolf_agent.model_gateway.execution_records import (
    AttemptExecutionRecord, AttemptOutcome, EvidenceKind, OpaqueRequestId,
    ReasoningLevel, ReasoningStatus, RootCause, RouteKind,
)


@dataclass
class GenerationAttemptContext:
    """由 router 与 action parser 共享，但不保存 prompt 或 provider 原始响应。"""

    run_scope: str = "game"
    opaque_request_id: OpaqueRequestId = field(init=False)
    attempts: tuple[AttemptExecutionRecord, ...] = ()
    next_route_kind: RouteKind = RouteKind.PRIMARY
    terminal_failure_reason: str | None = None
    # 2026-07-21 R4: 累积所有 mode 降级事件 (json_schema -> json_object ->
    # text_json 等). append-only; 每条 dict 含 from_mode, to_mode, reason_code,
    # attempt_ordinal. soak run 用此字段观测 schema_validation 触发降级的频率.
    mode_downgrades: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        scope = self.run_scope if len(self.run_scope) >= 4 else "game"
        self.opaque_request_id = OpaqueRequestId.new(scope, uuid.uuid4().hex[:16])

    def accept(self, attempts: tuple[AttemptExecutionRecord, ...]) -> None:
        self.attempts = attempts

    def reject_latest_output(self, root_cause: RootCause = RootCause.INVALID_OUTPUT) -> None:
        """把最新 action 轮次投影为失败，保留 provider/模型/推理证据事实。"""
        if not self.attempts:
            raise ValueError("cannot reject an empty generation attempt chain")
        latest = self.attempts[-1]
        self.attempts = (*self.attempts[:-1], replace(
            latest,
            attempt_outcome=AttemptOutcome.FAILURE,
            root_cause=root_cause,
        ))
        self.next_route_kind = RouteKind.REPAIR

    def record_mode_downgrade(
        self,
        *,
        from_mode: str,
        to_mode: str,
        reason_code: str,
    ) -> None:
        """2026-07-21 R4: 把 N→N+1 mode 降级事件写入 current attempt 上下文.

        append-only. PlayerAgent.act() 触发 schema_validation / parse_error
        失败后调用, 让 soak run 能区分"协议降级 vs 同 mode 修复".
        """
        self.mode_downgrades.append({
            "from_mode": from_mode,
            "to_mode": to_mode,
            "reason_code": reason_code,
            "attempt_ordinal": len(self.attempts) + 1,
        })

    def append_terminal_fallback(self, failure_reason: str | None = None) -> None:
        """为 action 层确定性 fallback 添加可翻译的终止边界。"""
        if failure_reason is not None:
            self.terminal_failure_reason = failure_reason
        if not self.attempts or self.attempts[-1].route_kind is RouteKind.SAFE_FALLBACK:
            return
        latest = self.attempts[-1]
        requested = latest.requested_reasoning_level
        self.attempts = (*self.attempts, replace(
            latest,
            ordinal=len(self.attempts) + 1,
            route_kind=RouteKind.SAFE_FALLBACK,
            provider_attempted=False,
            root_cause=RootCause.INVALID_OUTPUT,
            attempt_outcome=AttemptOutcome.FAILURE,
            normalized_reasoning_status=(
                ReasoningStatus.NOT_REQUESTED
                if requested is ReasoningLevel.NONE
                else ReasoningStatus.FALLBACK_DISABLED
            ),
            reasoning_token_count=0,
            evidence_kind=(
                EvidenceKind.NONE
                if requested is ReasoningLevel.NONE
                else EvidenceKind.FALLBACK_DISABLED
            ),
        ))


__all__ = ["GenerationAttemptContext"]
