# -*- coding: utf-8 -*-
"""
验证投票内部单位、展示值与版本化事件载荷的转换契约。

作者: Project contributors
创建日期: 2026-07-25
"""

from decimal import Decimal

import pytest

from werewolf_agent.runtime.vote_display import (
    VotePayloadError,
    decode_vote_resolved_payload,
    decode_vote_tally_payload,
    vote_display_to_json_number,
    format_vote_count,
    vote_units_to_display,
)


@pytest.mark.parametrize(
    ("units", "expected"),
    [
        (2, Decimal("1")),
        (3, Decimal("1.5")),
        (21, Decimal("10.5")),
    ],
)
def test_vote_units_to_display_is_exact(units: int, expected: Decimal) -> None:
    assert vote_units_to_display(units, base_vote_weight=2) == expected


@pytest.mark.parametrize(
    ("units", "base_vote_weight", "error_type"),
    [
        (True, 2, TypeError),
        (2.0, 2, TypeError),
        (2, False, TypeError),
        (2, 2.0, TypeError),
        (-1, 2, ValueError),
        (2, 0, ValueError),
        (2, -1, ValueError),
    ],
)
def test_vote_units_to_display_rejects_invalid_values(
    units: object,
    base_vote_weight: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        vote_units_to_display(units, base_vote_weight=base_vote_weight)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Decimal("1"), "1"),
        (Decimal("1.0"), "1"),
        (Decimal("1.5"), "1.5"),
        (Decimal("10.5"), "10.5"),
    ],
)
def test_format_vote_count_is_stable(value: Decimal, expected: str) -> None:
    assert format_vote_count(value) == expected


def test_vote_display_to_json_number_preserves_integral_and_half_types() -> None:
    integral = vote_display_to_json_number(Decimal("1"))
    half = vote_display_to_json_number(Decimal("1.5"))

    assert integral == 1
    assert type(integral) is int
    assert half == 1.5
    assert type(half) is float


@pytest.mark.parametrize(
    "value",
    [
        Decimal("NaN"),
        Decimal("Infinity"),
        Decimal("-Infinity"),
        Decimal("-0.5"),
        Decimal("1.25"),
    ],
)
def test_vote_display_to_json_number_rejects_unsafe_values(
    value: Decimal,
) -> None:
    with pytest.raises(ValueError):
        vote_display_to_json_number(value)


def test_decode_v1_vote_tally_with_known_ruleset_base() -> None:
    decoded = decode_vote_tally_payload(
        {"tally": {"p02": 3}, "sheriff_weight": 3},
        ruleset_base_vote_weight=2,
    )

    assert decoded.format_version == 1
    assert decoded.tally_units == {"p02": 3}
    assert decoded.sheriff_weight_units == 3
    assert decoded.tally_display == {"p02": Decimal("1.5")}
    assert decoded.sheriff_weight_display == Decimal("1.5")
    assert decoded.display_supported is True
    assert decoded.unsupported_reason is None


def test_decode_v1_vote_resolved_without_known_base_is_unsupported() -> None:
    decoded = decode_vote_resolved_payload(
        {
            "weighted_tally": {"p02": 3},
            "vote_weights": {"p01": 3},
        }
    )

    assert decoded.format_version == 1
    assert decoded.weighted_tally_units == {"p02": 3}
    assert decoded.vote_weight_units == {"p01": 3}
    assert decoded.weighted_tally_display is None
    assert decoded.vote_weights_display is None
    assert decoded.display_supported is False
    assert decoded.unsupported_reason == "base_vote_weight_unknown"


def test_decode_complete_v2_vote_tally() -> None:
    decoded = decode_vote_tally_payload(
        {
            "vote_weight_format_version": 2,
            "base_vote_weight": 2,
            "tally": {"p02": 21},
            "sheriff_weight": 3,
            "tally_units": {"p02": 21},
            "sheriff_weight_units": 3,
            "tally_display": {"p02": 10.5},
            "sheriff_weight_display": 1.5,
        }
    )

    assert decoded.format_version == 2
    assert decoded.base_vote_weight == 2
    assert decoded.tally_units == {"p02": 21}
    assert decoded.tally_display == {"p02": Decimal("10.5")}
    assert decoded.sheriff_weight_display == Decimal("1.5")
    assert decoded.display_supported is True


def test_decode_complete_v2_vote_resolved() -> None:
    decoded = decode_vote_resolved_payload(
        {
            "vote_weight_format_version": 2,
            "base_vote_weight": 2,
            "weighted_tally": {"p02": 21},
            "vote_weights": {"p04": 3},
            "weighted_tally_units": {"p02": 21},
            "vote_weight_units": {"p04": 3},
            "weighted_tally_display": {"p02": 10.5},
            "vote_weights_display": {"p04": 1.5},
        }
    )

    assert decoded.format_version == 2
    assert decoded.base_vote_weight == 2
    assert decoded.weighted_tally_units == {"p02": 21}
    assert decoded.vote_weight_units == {"p04": 3}
    assert decoded.weighted_tally_display == {"p02": Decimal("10.5")}
    assert decoded.vote_weights_display == {"p04": Decimal("1.5")}
    assert decoded.display_supported is True


def test_decode_v2_rejects_alias_unit_conflict() -> None:
    payload = {
        "vote_weight_format_version": 2,
        "base_vote_weight": 2,
        "weighted_tally": {"p02": 20},
        "vote_weights": {"p04": 3},
        "weighted_tally_units": {"p02": 21},
        "vote_weight_units": {"p04": 3},
        "weighted_tally_display": {"p02": 10.5},
        "vote_weights_display": {"p04": 1.5},
    }

    with pytest.raises(VotePayloadError, match="conflicts"):
        decode_vote_resolved_payload(payload)


@pytest.mark.parametrize(
    ("decoder", "payload"),
    [
        (
            decode_vote_tally_payload,
            {
                "vote_weight_format_version": 2,
                "base_vote_weight": 2,
                "tally": {},
                "sheriff_weight": 3,
                "sheriff_weight_units": 3,
                "tally_display": {},
                "sheriff_weight_display": 1.5,
            },
        ),
        (
            decode_vote_resolved_payload,
            {
                "vote_weight_format_version": 2,
                "base_vote_weight": 2,
                "weighted_tally": {},
                "vote_weights": {},
                "weighted_tally_units": {},
                "vote_weight_units": {},
                "weighted_tally_display": {},
            },
        ),
    ],
)
def test_decode_v2_rejects_missing_canonical_fields(
    decoder: object,
    payload: dict[str, object],
) -> None:
    with pytest.raises(VotePayloadError, match="missing"):
        decoder(payload)  # type: ignore[operator]


@pytest.mark.parametrize("version", [True, "2", 3])
def test_decode_rejects_unknown_or_invalid_version(version: object) -> None:
    with pytest.raises(VotePayloadError, match="version"):
        decode_vote_resolved_payload({"vote_weight_format_version": version})


@pytest.mark.parametrize(
    "payload",
    [
        {"weighted_tally": {"p02": True}, "vote_weights": {}},
        {"weighted_tally": {"p02": -1}, "vote_weights": {}},
        {"weighted_tally": [], "vote_weights": {}},
        {"weighted_tally": {}, "vote_weights": {"p01": 1.5}},
    ],
)
def test_decode_rejects_invalid_unit_types(payload: dict[str, object]) -> None:
    with pytest.raises(VotePayloadError):
        decode_vote_resolved_payload(payload, ruleset_base_vote_weight=2)
