"""Tests for customization YAML validators."""

from __future__ import annotations

import pytest

from werewolf_agent.customization import validators
from werewolf_agent.customization.validators import (
    _UNICODE_SUSPICIOUS_RANGES,
    validate_persona_pack_yaml,
    validate_ruleset_yaml,
)


# ---------------------------------------------------------------------------
# C1 (post-review-v2): _UNICODE_SUSPICIOUS_RANGES 应含 zero-width /
# bidirectional 字符。 这些字符是 Unicode 同形/混淆攻击的常见载体。
# ---------------------------------------------------------------------------


def test_unicode_suspicious_ranges_contains_required_chars() -> None:
    """C1: 关键可疑 Unicode 字符必须存在于 _UNICODE_SUSPICIOUS_RANGES。"""
    required = {
        "​",  # zero-width space
        "‌",  # zero-width non-joiner
        "‍",  # zero-width joiner
        "﻿",  # BOM
        "‪",  # LRM
        "‫",  # RLM
        "‬",  # ALM (deprecated in 2024)
        "‭",  # LRO
        "‮",  # RLO
    }
    missing = []
    for c in required:
        if c not in _UNICODE_SUSPICIOUS_RANGES:
            missing.append(f"U+{ord(c):04X}")
    assert not missing, (
        f"C1: _UNICODE_SUSPICIOUS_RANGES missing chars: {missing}; "
        f"got: {[hex(ord(c)) for c in _UNICODE_SUSPICIOUS_RANGES]!r}"
    )


def test_unicode_suspicious_ranges_triggers_validation_error() -> None:
    """C1: 包含 ZWSP 的 ruleset YAML 应被拒绝（integration check）。"""
    bad_yaml = """
ruleset_id: test_zwsp
player_count: 12
roles:
  villager:
    count: 12
constraints: {}
fields:
  suspicious: "value​with_zwsp"
"""
    result = validate_ruleset_yaml(bad_yaml)
    assert not result.valid, (
        f"C1: zero-width-space payload should be rejected, got valid={result.valid}"
    )
    codes = {e.code for e in result.errors}
    assert "unicode_injection" in codes, (
        f"C1: expected unicode_injection error code, got {codes!r}"
    )
