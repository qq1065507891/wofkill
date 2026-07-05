# -*- coding: utf-8 -*-
"""
功能描述：**：定制化校验结果的共享数据结构（dataclass schema）。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ValidationIssue:
    field: str
    message: str
    code: str = "invalid"


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    summary: dict[str, Any] = field(default_factory=dict)
    normalized: dict[str, Any] = field(default_factory=dict)
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    diff_against_default: list[dict[str, Any]] = field(default_factory=list)
