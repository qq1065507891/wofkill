# -*- coding: utf-8 -*-
"""
提供代理动作审计轨迹、私有意图和重试元数据 schema。

作者: Project contributors
创建日期: 2026-07-07
修改日期: 2026-07-23

使用示例:
    >>> from werewolf_agent.agents.trace_schemas import ActionTrace
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator
from werewolf_agent.model_gateway.execution_records import AttemptExecutionRecord
from werewolf_agent.runtime.decision_outcomes import summarize_attempt_counts


class FactionGoal(str, Enum):
    PUSH_GOOD_PLAYER_OUT = "push_good_player_out"
    PROTECT_TEAMMATE = "protect_teammate"
    FIND_WOLVES = "find_wolves"
    SURVIVE = "survive"
    HELP_MASTER_FACTION = "help_master_faction"
    CONFUSE_GOOD = "confuse_good"
    DEEP_HOOK = "deep_hook"
    AGGRESSIVE_PUSH = "aggressive_push"


class RiskFlag(str, Enum):
    AVOID_NIGHT_KILL_LEAK = "avoid_night_kill_leak"
    AVOID_TEAMMATE_EXPOSURE = "avoid_teammate_exposure"
    HIGH_VISIBILITY = "high_visibility"
    LOW_TRUST = "low_trust"
    SUSPECTED = "suspected"


# ---------------------------------------------------------------------------
# Private intent — not written to public timeline
# ---------------------------------------------------------------------------

class PrivateIntent(BaseModel):
    """Agent's private strategic snapshot. Only enters debug/audit views."""
    # P1-1: reject unknown fields. The LLM was stuffing extra keys
    # (e.g., leaked secrets, defensive fields) into private_intent and
    # the audit log happily accepted them. With extra="forbid" the
    # retry loop can surface the parse error and the LLM learns to
    # stop filling fields the prompt never requested.
    model_config = ConfigDict(extra="forbid")
    true_role: str = Field(..., description="Agent's actual role")
    faction_goal: FactionGoal = Field(..., description="Current faction objective")
    # P1-3: enforce enum. P0-S7 added the prompt-side constraint, but
    # the schema still accepted any string. Game trace g_3528592081
    # showed wolves writing `claimed_view: "我是好人，混水摸鱼"` (a
    # natural-language strategy note) and the audit log recording it.
    # `Literal` over the 7 documented identity-perspective values is
    # the same approach used by the example-renderer in
    # prompt_builder._format_examples; both stay in sync.
    claimed_view: Literal[
        "good_player_without_night_info",
        "seer",
        "witch",
        "hunter",
        "idiot",
        "hybrid",
        "werewolf",
    ] = Field(
        ..., description="Identity perspective the agent is claiming publicly"
    )
    pressure_target: str | None = Field(
        None, description="Player the agent intends to pressure"
    )
    risk_flags: list[RiskFlag] = Field(
        default_factory=list, description="Active risk markers"
    )


# ---------------------------------------------------------------------------
# Player action output — schema-constrained
# ---------------------------------------------------------------------------

class ActionTrace(BaseModel):
    """Moderator/audit trace for a model action attempt."""
    raw_text: str = ""
    parsed_action: dict[str, Any] | None = None
    # 非语义终退保留被拒输出审计；带稳定 reason_codes 的语义终退会脱敏。
    final_action: dict[str, Any] | None = None
    final_action_type: str = ""
    legal_actions: list[str] = Field(default_factory=list)
    legal_targets: list[str] = Field(default_factory=list)
    retry: dict[str, Any] | None = None
    fallback_reason: str | None = None
    # Task 1: Track whether a fallback target was used (decoupled from reason string)
    fallback_target_used: bool = False
    fallback_target_id: str | None = None
    # Task 9: Structured output metadata
    tool_call_required: bool = False
    tool_call_received: bool = False
    tool_call_name: str = ""
    parse_success: bool = False
    parse_error: str | None = None
    attempt_count: int = 0
    retry_count: int = 0
    provider_fallback_count: int = 0
    runtime_timeout_count: int = Field(default=0, ge=0, strict=True)
    generated_by: Literal[
        "model", "repair", "provider_fallback", "terminal_fallback"
    ] | None = None
    terminal_failure_code: str | None = None
    original_failure_code: str | None = None
    failure_stage: str | None = None
    fallback_kind: str | None = None
    structured_failure_reason: str | None = None
    structured_output_mode: str = ""
    structured_failure_stage: str | None = None
    total_retry_count_until_success: int = 0
    world_model_audit: dict[str, Any] = Field(default_factory=dict)
    execution_attempts: tuple[AttemptExecutionRecord, ...] = ()
    decision_outcome: str | None = None
    semantic_repair_audit: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _derive_runtime_timeout_count(self) -> "ActionTrace":
        """尝试记录是超时计数的唯一权威来源，兼容旧 payload 的缺失字段。"""
        derived_count = summarize_attempt_counts(
            self.execution_attempts
        ).runtime_timeout_count
        if "runtime_timeout_count" in self.model_fields_set:
            if self.runtime_timeout_count != derived_count:
                raise ValueError(
                    "runtime_timeout_count disagrees with execution attempts"
                )
        else:
            # 保留旧输入的字段存在性；exclude_unset 不应把回填值当作显式声明。
            object.__setattr__(self, "runtime_timeout_count", derived_count)
        return self

    @model_serializer(mode="wrap")
    def _serialize_v2(self, handler: Any) -> dict[str, Any]:
        """新 trace 不写旧计数字段；显式读入的 V1 值仍可只读访问。"""
        payload = handler(self)
        if "total_retry_count_until_success" not in self.model_fields_set:
            payload.pop("total_retry_count_until_success", None)
        return payload


# ---------------------------------------------------------------------------
# Retry / fallback metadata
# ---------------------------------------------------------------------------

class RetryInfo(BaseModel):
    """Tracks retry attempts for illegal/invalid outputs."""
    # P2-1: populated by upstream code, but without the strict field
    # guard a typo or future regression silently writes an unknown
    # key that downstream consumers won't notice.
    model_config = ConfigDict(extra="forbid")
    attempt: int = 1
    max_retries: int = 3
    error_code: str | None = None
    error_message: str | None = None
    reason_codes: list[str] = Field(default_factory=list)
    correction_hint: str | None = None
    # Pipeline-optimization Task 1: set when the retry loop short-circuits
    # because two consecutive attempts produced the same (error_code,
    # raw_text[:50]) signature. Saves wasted LLM calls when the model is
    # stuck repeating the same broken output.
    early_exit_reason: str | None = None
    # Pipeline-optimization Task 3: attribution for empty_response — one of
    # "timeout", "token_limit", "provider_error", "network_error", "unknown".
    # None when the response was not empty or the cause could not be inferred.
    failure_category: str | None = None


__all__ = [
    "ActionTrace",
    "FactionGoal",
    "PrivateIntent",
    "RetryInfo",
    "RiskFlag",
]
