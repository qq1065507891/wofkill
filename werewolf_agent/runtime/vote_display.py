# -*- coding: utf-8 -*-
"""
集中处理投票内部单位、展示值格式化与版本化事件载荷解码。

作者: Project contributors
创建日期: 2026-07-25

使用示例:
    >>> vote_units_to_display(3, base_vote_weight=2)
    Decimal('1.5')
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping


class VotePayloadError(ValueError):
    """表示投票事件载荷无法被安全解码。"""


@dataclass(frozen=True)
class DecodedVoteTally:
    """法官唱票载荷的规范化结果。"""

    format_version: int
    base_vote_weight: int | None
    tally_units: dict[str, int]
    sheriff_weight_units: int
    tally_display: dict[str, Decimal] | None
    sheriff_weight_display: Decimal | None
    display_supported: bool
    unsupported_reason: str | None = None


@dataclass(frozen=True)
class DecodedVoteResolved:
    """投票结算载荷的规范化结果。"""

    format_version: int
    base_vote_weight: int | None
    weighted_tally_units: dict[str, int]
    vote_weight_units: dict[str, int]
    weighted_tally_display: dict[str, Decimal] | None
    vote_weights_display: dict[str, Decimal] | None
    display_supported: bool
    unsupported_reason: str | None = None


def vote_units_to_display(units: int, *, base_vote_weight: int) -> Decimal:
    """把非负整数票权单位精确转换为实际票数。"""
    if type(units) is not int or type(base_vote_weight) is not int:
        raise TypeError("vote units and base_vote_weight must be integers")
    if units < 0 or base_vote_weight <= 0:
        raise ValueError(
            "vote units must be non-negative and base_vote_weight positive"
        )
    return Decimal(units) / Decimal(base_vote_weight)


def format_vote_count(value: Decimal) -> str:
    """稳定格式化实际票数，不保留无意义的小数零。"""
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
        raise ValueError("vote display value must be a non-negative finite Decimal")
    normalized = (
        value.quantize(Decimal("1"))
        if value == value.to_integral()
        else value.normalize()
    )
    return format(normalized, "f")


def vote_display_to_json_number(value: Decimal) -> int | float:
    """把精确的整数或半整数票数转换为 JSON 安全数值。"""
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
        raise ValueError("vote display value must be a non-negative finite Decimal")
    if value == value.to_integral():
        return int(value)
    if value * 2 == (value * 2).to_integral():
        # 二进制浮点可精确表示 0.5，因而不会损失半票值。
        return float(value)
    raise ValueError("vote display value must be an integer or half-integer")


def decode_vote_tally_payload(
    payload: Mapping[str, Any],
    *,
    ruleset_base_vote_weight: int | None = None,
) -> DecodedVoteTally:
    """解码 V1/V2 的 ``judge_broadcast`` 唱票载荷。"""
    data = _validate_payload(payload)
    version = _payload_version(data)
    if version == 1:
        tally = _unit_map(data, "tally")
        sheriff_weight = _unit_value(data, "sheriff_weight")
        base = _v1_base_vote_weight(data, ruleset_base_vote_weight)
        if base is None:
            return DecodedVoteTally(
                format_version=1,
                base_vote_weight=None,
                tally_units=tally,
                sheriff_weight_units=sheriff_weight,
                tally_display=None,
                sheriff_weight_display=None,
                display_supported=False,
                unsupported_reason="base_vote_weight_unknown",
            )
        return DecodedVoteTally(
            format_version=1,
            base_vote_weight=base,
            tally_units=tally,
            sheriff_weight_units=sheriff_weight,
            tally_display=_display_map_from_units(tally, base),
            sheriff_weight_display=vote_units_to_display(
                sheriff_weight,
                base_vote_weight=base,
            ),
            display_supported=True,
        )

    _require_keys(
        data,
        "base_vote_weight",
        "tally",
        "sheriff_weight",
        "tally_units",
        "sheriff_weight_units",
        "tally_display",
        "sheriff_weight_display",
    )
    base = _positive_int(data, "base_vote_weight")
    tally_alias = _unit_map(data, "tally")
    tally_units = _unit_map(data, "tally_units")
    sheriff_alias = _unit_value(data, "sheriff_weight")
    sheriff_units = _unit_value(data, "sheriff_weight_units")
    _require_alias_match("tally", tally_alias, "tally_units", tally_units)
    _require_alias_match(
        "sheriff_weight",
        sheriff_alias,
        "sheriff_weight_units",
        sheriff_units,
    )
    tally_display = _display_map(data, "tally_display")
    sheriff_display = _display_value(data, "sheriff_weight_display")
    _require_display_match(
        "tally_display",
        tally_display,
        _display_map_from_units(tally_units, base),
    )
    _require_display_match(
        "sheriff_weight_display",
        sheriff_display,
        vote_units_to_display(sheriff_units, base_vote_weight=base),
    )
    return DecodedVoteTally(
        format_version=2,
        base_vote_weight=base,
        tally_units=tally_units,
        sheriff_weight_units=sheriff_units,
        tally_display=tally_display,
        sheriff_weight_display=sheriff_display,
        display_supported=True,
    )


def decode_vote_resolved_payload(
    payload: Mapping[str, Any],
    *,
    ruleset_base_vote_weight: int | None = None,
) -> DecodedVoteResolved:
    """解码 V1/V2 的 ``vote_resolved`` 载荷。"""
    data = _validate_payload(payload)
    version = _payload_version(data)
    if version == 1:
        tally = _unit_map(data, "weighted_tally")
        weights = _unit_map(data, "vote_weights")
        base = _v1_base_vote_weight(data, ruleset_base_vote_weight)
        if base is None:
            return DecodedVoteResolved(
                format_version=1,
                base_vote_weight=None,
                weighted_tally_units=tally,
                vote_weight_units=weights,
                weighted_tally_display=None,
                vote_weights_display=None,
                display_supported=False,
                unsupported_reason="base_vote_weight_unknown",
            )
        return DecodedVoteResolved(
            format_version=1,
            base_vote_weight=base,
            weighted_tally_units=tally,
            vote_weight_units=weights,
            weighted_tally_display=_display_map_from_units(tally, base),
            vote_weights_display=_display_map_from_units(weights, base),
            display_supported=True,
        )

    _require_keys(
        data,
        "base_vote_weight",
        "weighted_tally",
        "vote_weights",
        "weighted_tally_units",
        "vote_weight_units",
        "weighted_tally_display",
        "vote_weights_display",
    )
    base = _positive_int(data, "base_vote_weight")
    tally_alias = _unit_map(data, "weighted_tally")
    weights_alias = _unit_map(data, "vote_weights")
    tally_units = _unit_map(data, "weighted_tally_units")
    weight_units = _unit_map(data, "vote_weight_units")
    _require_alias_match(
        "weighted_tally",
        tally_alias,
        "weighted_tally_units",
        tally_units,
    )
    _require_alias_match(
        "vote_weights",
        weights_alias,
        "vote_weight_units",
        weight_units,
    )
    tally_display = _display_map(data, "weighted_tally_display")
    weights_display = _display_map(data, "vote_weights_display")
    _require_display_match(
        "weighted_tally_display",
        tally_display,
        _display_map_from_units(tally_units, base),
    )
    _require_display_match(
        "vote_weights_display",
        weights_display,
        _display_map_from_units(weight_units, base),
    )
    return DecodedVoteResolved(
        format_version=2,
        base_vote_weight=base,
        weighted_tally_units=tally_units,
        vote_weight_units=weight_units,
        weighted_tally_display=tally_display,
        vote_weights_display=weights_display,
        display_supported=True,
    )


def _validate_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise VotePayloadError("vote payload must be a mapping")
    return payload


def _payload_version(payload: Mapping[str, Any]) -> int:
    if "vote_weight_format_version" not in payload:
        return 1
    version = payload["vote_weight_format_version"]
    if type(version) is not int or version != 2:
        raise VotePayloadError(
            f"unsupported vote_weight_format_version: {version!r}"
        )
    return version


def _v1_base_vote_weight(
    payload: Mapping[str, Any],
    ruleset_base_vote_weight: int | None,
) -> int | None:
    payload_base = (
        _positive_int(payload, "base_vote_weight")
        if "base_vote_weight" in payload
        else None
    )
    ruleset_base = _optional_positive_int(
        ruleset_base_vote_weight,
        "ruleset_base_vote_weight",
    )
    if (
        payload_base is not None
        and ruleset_base is not None
        and payload_base != ruleset_base
    ):
        raise VotePayloadError(
            "payload base_vote_weight conflicts with ruleset_base_vote_weight"
        )
    return payload_base if payload_base is not None else ruleset_base


def _require_keys(payload: Mapping[str, Any], *keys: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise VotePayloadError(f"missing canonical vote fields: {', '.join(missing)}")


def _positive_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload[key]
    checked = _optional_positive_int(value, key)
    if checked is None:
        raise VotePayloadError(f"{key} must be a positive integer")
    return checked


def _optional_positive_int(value: Any, key: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise VotePayloadError(f"{key} must be a positive integer")
    return value


def _unit_value(payload: Mapping[str, Any], key: str) -> int:
    if key not in payload:
        raise VotePayloadError(f"missing vote field: {key}")
    value = payload[key]
    if type(value) is not int or value < 0:
        raise VotePayloadError(f"{key} must be a non-negative integer")
    return value


def _unit_map(payload: Mapping[str, Any], key: str) -> dict[str, int]:
    if key not in payload:
        raise VotePayloadError(f"missing vote field: {key}")
    value = payload[key]
    if not isinstance(value, Mapping):
        raise VotePayloadError(f"{key} must be a mapping")
    result: dict[str, int] = {}
    for item_key, item_value in value.items():
        if not isinstance(item_key, str):
            raise VotePayloadError(f"{key} keys must be strings")
        if type(item_value) is not int or item_value < 0:
            raise VotePayloadError(
                f"{key}[{item_key!r}] must be a non-negative integer"
            )
        result[item_key] = item_value
    return result


def _display_value(payload: Mapping[str, Any], key: str) -> Decimal:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise VotePayloadError(f"{key} must be a non-negative finite number")
    display = Decimal(str(value))
    if not display.is_finite() or display < 0:
        raise VotePayloadError(f"{key} must be a non-negative finite number")
    return display


def _display_map(payload: Mapping[str, Any], key: str) -> dict[str, Decimal]:
    value = payload[key]
    if not isinstance(value, Mapping):
        raise VotePayloadError(f"{key} must be a mapping")
    result: dict[str, Decimal] = {}
    for item_key in value:
        if not isinstance(item_key, str):
            raise VotePayloadError(f"{key} keys must be strings")
        result[item_key] = _display_value(value, item_key)
    return result


def _display_map_from_units(
    units: Mapping[str, int],
    base_vote_weight: int,
) -> dict[str, Decimal]:
    return {
        key: vote_units_to_display(value, base_vote_weight=base_vote_weight)
        for key, value in units.items()
    }


def _require_alias_match(
    alias_name: str,
    alias_value: Any,
    unit_name: str,
    unit_value: Any,
) -> None:
    if alias_value != unit_value:
        raise VotePayloadError(f"{alias_name} conflicts with {unit_name}")


def _require_display_match(
    display_name: str,
    actual: Any,
    expected: Any,
) -> None:
    if actual != expected:
        raise VotePayloadError(f"{display_name} conflicts with canonical vote units")
