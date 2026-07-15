# -*- coding: utf-8 -*-
"""
集中定义死亡结算批次 V2，并提供 V1/V2 解析及嵌套 JSON 安全序列化。

作者: Project contributors
创建日期: 2026-07-15
修改日期: 2026-07-16

使用示例:
    >>> parse_resolution_batch("day_2_vote").batch
    ResolutionBatchV2(phase='day', number=2, cause='vote')
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import islice
from typing import Any, Literal, Protocol, TypeAlias, cast

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


class ResolutionBatchCarrier(Protocol):
    """提供结算批次值及其持久化失败标记的对象。"""

    @property
    def resolution_batch(self) -> ResolutionBatchV2 | str | Mapping[str, object]: ...

    @property
    def resolution_batch_parse_failed(self) -> bool: ...


def valid_resolution_batch(
    value: ResolutionBatchV2 | str | Mapping[str, object],
    *,
    parse_failed: bool = False,
) -> ResolutionBatchV2 | None:
    """仅在持久化标记和当前解析都成功时返回结构化批次。"""
    if parse_failed:
        return None
    result = parse_resolution_batch(value)
    if result.batch_parse_failed:
        return None
    return result.batch


def valid_carrier_resolution_batch(
    carrier: ResolutionBatchCarrier,
) -> ResolutionBatchV2 | None:
    """读取带失败标记对象的批次；任一失败信号都按失败关闭。"""
    return valid_resolution_batch(
        carrier.resolution_batch,
        parse_failed=carrier.resolution_batch_parse_failed,
    )


_SANITIZER_MAX_DEPTH = 8
_SANITIZER_MAX_ITEMS = 64
_SANITIZER_MAX_STRING_LENGTH = 512


def _type_label(value: object) -> str:
    value_type = type(value)
    label = f"{value_type.__module__}.{value_type.__qualname__}"
    if len(label) <= 256:
        return label
    digest = hashlib.sha256(label.encode("utf-8", errors="replace")).hexdigest()
    return f"type_sha256:{digest}"


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_order_token(value: object) -> str:
    """为异构 JSON 值提供不依赖 Python 比较规则的稳定顺序。"""
    if value is None:
        rank = 0
    elif isinstance(value, bool):
        rank = 1
    elif isinstance(value, (int, float)):
        rank = 2
    elif isinstance(value, str):
        rank = 3
    elif isinstance(value, list):
        rank = 4
    else:
        rank = 5
    return f"{rank}:{_canonical_json(value)}"


def _mapping_read_failure(value: Mapping[object, object]) -> dict[str, object]:
    return {"$unreadable_mapping": _type_label(value)}


def _bounded_mapping_items(
    value: Mapping[object, object],
) -> tuple[list[tuple[object, object]] | dict[str, object], bool]:
    """最多读取 MAX+1 个条目，并校验 Mapping 报告的长度。"""
    try:
        reported_size = len(value)
    except Exception:
        return _mapping_read_failure(value), False
    if reported_size > _SANITIZER_MAX_ITEMS:
        return {
            "$truncated_mapping": {
                "reported_size": reported_size,
                "type": _type_label(value),
            }
        }, False
    try:
        items = list(islice(iter(value.items()), _SANITIZER_MAX_ITEMS + 1))
    except Exception:
        return _mapping_read_failure(value), False
    if len(items) > _SANITIZER_MAX_ITEMS:
        return {
            "$truncated_mapping": {
                "observed_at_least": _SANITIZER_MAX_ITEMS + 1,
                "reported_size": reported_size,
                "type": _type_label(value),
            }
        }, False
    if len(items) != reported_size:
        return {
            "$mapping_size_mismatch": {
                "observed_size": len(items),
                "reported_size": reported_size,
                "type": _type_label(value),
            }
        }, False
    return items, True


def _canonical_sanitize_mapping(
    value: Mapping[object, object],
    *,
    depth: int,
    active_ids: set[int],
) -> tuple[object, bool]:
    """有界读取并规范化 Mapping，同时返回根结构是否完整可信。"""
    value_id = id(value)
    active_ids.add(value_id)
    try:
        bounded, complete = _bounded_mapping_items(value)
        if not complete:
            return bounded, False
        items = bounded
        if all(
            isinstance(key, str) and len(key) <= _SANITIZER_MAX_STRING_LENGTH
            for key, _ in items
        ):
            return {
                key: _canonical_sanitize(
                    item,
                    depth=depth + 1,
                    active_ids=active_ids,
                )
                for key, item in sorted(items, key=lambda pair: pair[0])
            }, True

        canonical_items: list[tuple[str, object, object]] = []
        for key, item in items:
            safe_key = _canonical_sanitize(
                key,
                depth=depth + 1,
                active_ids=active_ids,
            )
            canonical_items.append(
                (
                    _canonical_order_token(safe_key),
                    safe_key,
                    _canonical_sanitize(
                        item,
                        depth=depth + 1,
                        active_ids=active_ids,
                    ),
                )
            )
        canonical_items.sort(key=lambda entry: entry[0])
        for index in range(1, len(canonical_items)):
            if canonical_items[index - 1][0] == canonical_items[index][0]:
                return {"$mapping_key_collision": canonical_items[index][1]}, False
        return {
            "$mapping": [
                [safe_key, safe_item]
                for _, safe_key, safe_item in canonical_items
            ]
        }, True
    except Exception:
        return _mapping_read_failure(value), False
    finally:
        active_ids.remove(value_id)


def _canonical_sanitize(
    value: object,
    *,
    depth: int = 0,
    active_ids: set[int] | None = None,
) -> object:
    """递归生成有界、确定且不暴露对象内部状态的 JSON-safe 值。"""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"$non_finite_float": str(value)}
    if isinstance(value, str):
        if len(value) <= _SANITIZER_MAX_STRING_LENGTH:
            return value
        digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()
        return {"$long_string": {"length": len(value), "sha256": digest}}
    if isinstance(value, bytes):
        return {
            "$bytes": {
                "length": len(value),
                "sha256": hashlib.sha256(value).hexdigest(),
            }
        }
    if depth >= _SANITIZER_MAX_DEPTH:
        return {"$max_depth": _type_label(value)}

    active_ids = active_ids if active_ids is not None else set()
    value_id = id(value)
    if value_id in active_ids:
        return {"$cycle": _type_label(value)}

    if isinstance(value, Mapping):
        sanitized, _complete = _canonical_sanitize_mapping(
            value,
            depth=depth,
            active_ids=active_ids,
        )
        return sanitized

    if isinstance(value, (list, tuple)):
        active_ids.add(value_id)
        try:
            if len(value) > _SANITIZER_MAX_ITEMS:
                return {
                    "$truncated_sequence": {
                        "length": len(value),
                        "type": _type_label(value),
                    }
                }
            return [
                _canonical_sanitize(
                    item,
                    depth=depth + 1,
                    active_ids=active_ids,
                )
                for item in value
            ]
        finally:
            active_ids.remove(value_id)

    if isinstance(value, (set, frozenset)):
        active_ids.add(value_id)
        try:
            if type(value) not in {set, frozenset}:
                return {
                    "$set_summary": {
                        "size": None,
                        "type": _type_label(value),
                    }
                }
            try:
                size = len(value)
            except Exception:
                size = None
            if size is None:
                return {
                    "$set_summary": {
                        "size": size,
                        "type": _type_label(value),
                    }
                }
            if size > _SANITIZER_MAX_ITEMS:
                return {
                    "$set_summary": {
                        "size": size,
                        "type": _type_label(value),
                    }
                }
            items = [
                _canonical_sanitize(
                    item,
                    depth=depth + 1,
                    active_ids=active_ids,
                )
                for item in value
            ]
            items.sort(key=_canonical_order_token)
            return {"$set": items}
        finally:
            active_ids.remove(value_id)

    return {"$unsupported": _type_label(value)}


def _stable_mapping_json(value: Mapping[object, object]) -> tuple[str, bool]:
    sanitized, complete = _canonical_sanitize_mapping(
        value,
        depth=0,
        active_ids=set(),
    )
    return _canonical_json(sanitized), complete


def parse_resolution_batch(
    value: ResolutionBatchV2 | str | Mapping[str, object],
) -> ResolutionBatchParseResult:
    """解析 V2、JSON mapping 或只读兼容的 legacy string。"""
    if isinstance(value, ResolutionBatchV2):
        return ResolutionBatchParseResult(value, None, False)
    if isinstance(value, Mapping):
        raw, mapping_complete = _stable_mapping_json(value)
        if not mapping_complete:
            return ResolutionBatchParseResult(None, raw, True)
        try:
            if len(value) != 3:
                return ResolutionBatchParseResult(None, raw, True)
            batch = ResolutionBatchV2(
                phase=cast(ResolutionPhase, value["phase"]),
                number=cast(int, value["number"]),
                cause=cast(ResolutionCause, value["cause"]),
            )
        except Exception:
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
    return ResolutionBatchParseResult(
        None,
        _canonical_json(_canonical_sanitize(value)),
        True,
    )


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
    *,
    left_parse_failed: bool = False,
    right_parse_failed: bool = False,
) -> bool:
    """按 phase 与 number 比较同一结算链，任一解析失败即不匹配。"""
    left_batch = valid_resolution_batch(left, parse_failed=left_parse_failed)
    right_batch = valid_resolution_batch(right, parse_failed=right_parse_failed)
    if left_batch is None or right_batch is None:
        return False
    return (
        left_batch.phase == right_batch.phase
        and left_batch.number == right_batch.number
    )


def carrier_matches_resolution_batch(
    carrier: ResolutionBatchCarrier,
    other: ResolutionBatchV2 | str | Mapping[str, object],
    *,
    other_parse_failed: bool = False,
) -> bool:
    """比较对象批次，同时尊重对象和对端的失败标记。"""
    return same_resolution_batch(
        carrier.resolution_batch,
        other,
        left_parse_failed=carrier.resolution_batch_parse_failed,
        right_parse_failed=other_parse_failed,
    )


def serialize_resolution_batches_in_value(value: Any) -> Any:
    """递归转换容器中的 ResolutionBatchV2，不处理其他业务数据类。"""
    if isinstance(value, ResolutionBatchV2):
        return serialize_resolution_batch(value)[0]
    if isinstance(value, Mapping):
        return {
            key: serialize_resolution_batches_in_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [serialize_resolution_batches_in_value(item) for item in value]
    return value


__all__ = [
    "ResolutionBatchParseResult",
    "ResolutionBatchV2",
    "carrier_matches_resolution_batch",
    "normalize_resolution_batch_fields",
    "parse_resolution_batch",
    "same_resolution_batch",
    "serialize_resolution_batch",
    "serialize_resolution_batches_in_value",
    "serialize_resolution_batch_fields",
    "valid_carrier_resolution_batch",
    "valid_resolution_batch",
]
