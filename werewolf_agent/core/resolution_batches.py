# -*- coding: utf-8 -*-
"""
集中定义死亡结算批次 V2，并提供 V1/V2 解析与 JSON 安全序列化。

作者: Project contributors
创建日期: 2026-07-15

使用示例:
    >>> parse_resolution_batch("day_2_vote").batch
    ResolutionBatchV2(phase='day', number=2, cause='vote')
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

ResolutionPhase: TypeAlias = Literal["day", "night"]
ResolutionCause: TypeAlias = Literal[
    "vote",
    "self_destruct",
    "wolf_kill",
    "witch_poison",
    "hunter_shot",
    "rule_effect",
    "unknown",
]

_PHASES = frozenset({"day", "night"})
_CAUSES = frozenset(
    {
        "vote",
        "self_destruct",
        "wolf_kill",
        "witch_poison",
        "hunter_shot",
        "rule_effect",
        "unknown",
    }
)
_LEGACY_BATCH_RE = re.compile(
    r"(day|night)_(\d+)(?:_(vote|self_destruct|wolf_kill|witch_poison|hunter_shot|rule_effect))?"
)


@dataclass(frozen=True)
class ResolutionBatchV2:
    """结构化死亡结算批次。"""

    phase: ResolutionPhase
    number: int
    cause: ResolutionCause

    def __post_init__(self) -> None:
        if self.phase not in _PHASES:
            raise ValueError(f"invalid resolution phase: {self.phase!r}")
        if isinstance(self.number, bool) or not isinstance(self.number, int) or self.number < 0:
            raise ValueError(f"invalid resolution number: {self.number!r}")
        if self.cause not in _CAUSES:
            raise ValueError(f"invalid resolution cause: {self.cause!r}")


@dataclass(frozen=True)
class ResolutionBatchParseResult:
    """保留规范化批次、原值和失败标记的审计结果。"""

    batch: ResolutionBatchV2 | None
    raw_value: str | None
    batch_parse_failed: bool


def _stable_mapping_json(value: Mapping[str, object]) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        # 非 JSON 值仍需稳定且不可丢失失败原因。
        safe = {str(key): repr(item) for key, item in value.items()}
        return json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parse_resolution_batch(
    value: ResolutionBatchV2 | str | Mapping[str, object],
) -> ResolutionBatchParseResult:
    """解析 V2、JSON mapping 或只读兼容的 legacy string。"""
    if isinstance(value, ResolutionBatchV2):
        return ResolutionBatchParseResult(value, None, False)
    if isinstance(value, Mapping):
        raw = _stable_mapping_json(value)
        if set(value) != {"phase", "number", "cause"}:
            return ResolutionBatchParseResult(None, raw, True)
        try:
            batch = ResolutionBatchV2(
                phase=cast(ResolutionPhase, value["phase"]),
                number=cast(int, value["number"]),
                cause=cast(ResolutionCause, value["cause"]),
            )
        except (TypeError, ValueError):
            return ResolutionBatchParseResult(None, raw, True)
        return ResolutionBatchParseResult(batch, None, False)
    if isinstance(value, str):
        match = _LEGACY_BATCH_RE.fullmatch(value)
        if match is None:
            return ResolutionBatchParseResult(None, value, True)
        phase, number, suffix = match.groups()
        return ResolutionBatchParseResult(
            ResolutionBatchV2(
                phase=cast(ResolutionPhase, phase),
                number=int(number),
                cause=cast(ResolutionCause, suffix or "unknown"),
            ),
            value,
            False,
        )
    return ResolutionBatchParseResult(None, repr(value), True)


def serialize_resolution_batch(
    value: ResolutionBatchV2 | str | Mapping[str, object],
) -> tuple[dict[str, object] | str, bool]:
    """输出 JSON-safe 批次值及解析失败标记。"""
    result = parse_resolution_batch(value)
    if result.batch is None:
        return result.raw_value or "", True
    return {
        "phase": result.batch.phase,
        "number": result.batch.number,
        "cause": result.batch.cause,
    }, False


def normalize_resolution_batch_fields(data: Mapping[str, Any]) -> dict[str, Any]:
    """规范化持久化 Death 字段，并保留解析失败标记。"""
    normalized = dict(data)
    result = parse_resolution_batch(normalized.get("resolution_batch", ""))
    normalized["resolution_batch"] = (
        result.batch if result.batch is not None else result.raw_value or ""
    )
    normalized["resolution_batch_parse_failed"] = bool(
        normalized.get("resolution_batch_parse_failed", False)
        or result.batch_parse_failed
    )
    return normalized


def serialize_resolution_batch_fields(data: Mapping[str, Any]) -> dict[str, Any]:
    """把 Death 字段中的批次转换成唯一 JSON-safe 形态。"""
    serialized = dict(data)
    batch, parse_failed = serialize_resolution_batch(
        serialized.get("resolution_batch", "")
    )
    serialized["resolution_batch"] = batch
    serialized["resolution_batch_parse_failed"] = bool(
        serialized.get("resolution_batch_parse_failed", False) or parse_failed
    )
    return serialized


def same_resolution_batch(
    left: ResolutionBatchV2 | str | Mapping[str, object],
    right: ResolutionBatchV2 | str | Mapping[str, object],
) -> bool:
    """按 phase 与 number 比较同一结算链，任一解析失败即不匹配。"""
    left_result = parse_resolution_batch(left)
    right_result = parse_resolution_batch(right)
    if left_result.batch is None or right_result.batch is None:
        return False
    return (
        left_result.batch.phase == right_result.batch.phase
        and left_result.batch.number == right_result.batch.number
    )


__all__ = [
    "ResolutionBatchParseResult",
    "ResolutionBatchV2",
    "normalize_resolution_batch_fields",
    "parse_resolution_batch",
    "same_resolution_batch",
    "serialize_resolution_batch",
    "serialize_resolution_batch_fields",
]
