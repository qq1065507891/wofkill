"""Shared schemas for customization validation results."""

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
