"""Safe YAML validators for uploaded customization templates."""

from __future__ import annotations

from typing import Any

import yaml

from werewolf_agent.customization.compatibility import build_compatibility_matrix
from werewolf_agent.customization.schemas import ValidationIssue, ValidationResult


ALLOWED_RULESET_FIELDS = frozenset(
    {
        "ruleset_id",
        "id",
        "name",
        "version",
        "description",
        "player_count",
        "roles",
        "abilities",
        "night_order",
        "victory",
        "constraints",
    }
)

ALLOWED_PERSONA_PACK_FIELDS = frozenset(
    {
        "profile_pack_id",
        "id",
        "name",
        "version",
        "description",
        "players",
    }
)

PERSONA_REQUIRED_FIELDS = frozenset(
    {
        "seat",
        "name",
        "archetype",
        "speech_style",
        "risk_tolerance",
        "deception",
        "cooperation",
        "aggression",
        "memory_focus",
        "logic_focus",
        "emotionality",
    }
)

PERSONA_OPTIONAL_FIELDS = frozenset({"preferred_roles", "catchphrases"})
PERSONA_ALLOWED_FIELDS = PERSONA_REQUIRED_FIELDS | PERSONA_OPTIONAL_FIELDS
PERSONA_LEVEL_FIELDS = frozenset(
    {
        "risk_tolerance",
        "deception",
        "cooperation",
        "aggression",
        "memory_focus",
        "logic_focus",
        "emotionality",
    }
)
PERSONA_LEVELS = frozenset({"low", "medium", "high"})
PERSONA_ALLOWED_ROLES = frozenset(
    {
        "werewolf",
        "villager",
        "seer",
        "witch",
        "hunter",
        "idiot",
        "hybrid",
    }
)

ALLOWED_CONSTRAINTS = frozenset(
    {
        "witch_can_self_save",
        "witch_can_use_both_potions_same_night",
        "werewolf_can_no_kill",
        "wolf_timeout_default",
        "hybrid_enabled",
    }
)

DEFAULT_RULESET_NORMALIZED = {
    "constraints": {
        "witch_can_self_save": False,
        "witch_can_use_both_potions_same_night": False,
        "werewolf_can_no_kill": True,
        "wolf_timeout_default": "no_kill",
        "hybrid_enabled": True,
    }
}

PROMPT_INJECTION_MARKERS = (
    "ignore previous instructions",
    "system prompt",
    "<script",
    "${",
    "{{",
    "rm -rf",
    "powershell",
)

# Unicode 同形字符检测：零宽字符和混淆字符
_UNICODE_SUSPICIOUS_RANGES = (
    "​",  # 零宽空格
    "‌",  # 零宽非连接符
    "‍",  # 零宽连接符
    "﻿",  # BOM / 零宽不换行空格
    "‪",  # 从左到右嵌入
    "‫",  # 从右到左嵌入
    "‬",  # 弹出方向格式
    "‭",  # 从左到右覆盖
    "‮",  # 从右到左覆盖
)


def validate_ruleset_yaml(text: str) -> ValidationResult:
    """Parse and validate a custom ruleset YAML document as plain data."""

    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return ValidationResult(
            valid=False,
            errors=[ValidationIssue(field="yaml", message=f"invalid YAML: {exc}", code="yaml_parse_error")],
        )

    if not isinstance(loaded, dict):
        return ValidationResult(
            valid=False,
            errors=[ValidationIssue(field="yaml", message="ruleset template must be a mapping")],
        )

    unknown_fields = sorted(set(loaded) - ALLOWED_RULESET_FIELDS)
    for field in unknown_fields:
        errors.append(ValidationIssue(field=field, message=f"unknown field: {field}", code="unknown_field"))

    ruleset_id = str(loaded.get("ruleset_id") or loaded.get("id") or "").strip()
    if not ruleset_id:
        errors.append(ValidationIssue(field="ruleset_id", message="ruleset_id is required"))

    roles = loaded.get("roles")
    if not isinstance(roles, dict) or not roles:
        errors.append(ValidationIssue(field="roles", message="roles must be a non-empty mapping"))
        roles = {}

    player_count = _int_or_none(loaded.get("player_count"))
    if player_count is None:
        errors.append(ValidationIssue(field="player_count", message="player_count must be an integer"))
        player_count = 0

    role_count_sum = 0
    for role_id, cfg in roles.items():
        if not isinstance(cfg, dict):
            errors.append(ValidationIssue(field=f"roles.{role_id}", message="role config must be a mapping"))
            continue
        count = _int_or_none(cfg.get("count"))
        if count is None or count < 0:
            errors.append(ValidationIssue(field=f"roles.{role_id}.count", message="role count must be non-negative"))
            continue
        role_count_sum += count

    if player_count and role_count_sum != player_count:
        errors.append(
            ValidationIssue(
                field="player_count",
                message=f"player_count mismatch: expected {player_count}, role counts sum to {role_count_sum}",
                code="role_count_mismatch",
            )
        )

    constraints = loaded.get("constraints") or {}
    if not isinstance(constraints, dict):
        errors.append(ValidationIssue(field="constraints", message="constraints must be a mapping"))
        constraints = {}
    for constraint in sorted(set(constraints) - ALLOWED_CONSTRAINTS):
        errors.append(
            ValidationIssue(
                field=f"constraints.{constraint}",
                message=f"unknown constraint: {constraint}",
                code="unknown_constraint",
            )
        )

    _scan_text("", loaded, errors)

    normalized = _normalize_ruleset(loaded, ruleset_id, player_count, roles, constraints)
    compatibility = build_compatibility_matrix(normalized)
    normalized["status"] = compatibility.status
    normalized["unsupported_roles"] = compatibility.unsupported_roles
    normalized["missing_abilities"] = compatibility.missing_abilities
    for warning in compatibility.warnings:
        warnings.append(ValidationIssue(field="compatibility", message=warning, code="display_only"))

    diff = _diff_against_default(normalized)
    summary = {
        "ruleset_id": ruleset_id,
        "player_count": player_count,
        "role_count_sum": role_count_sum,
        "status": compatibility.status,
    }

    return ValidationResult(
        valid=not errors,
        summary=summary,
        normalized=normalized,
        errors=errors,
        warnings=warnings,
        diff_against_default=diff,
    )


def validate_persona_pack_yaml(text: str, *, expected_player_count: int = 12) -> ValidationResult:
    """Parse and validate a user-facing persona pack."""

    errors: list[ValidationIssue] = []
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return ValidationResult(
            valid=False,
            errors=[ValidationIssue(field="yaml", message=f"invalid YAML: {exc}", code="yaml_parse_error")],
        )

    if not isinstance(loaded, dict):
        return ValidationResult(
            valid=False,
            errors=[ValidationIssue(field="yaml", message="persona pack template must be a mapping")],
        )

    for field in sorted(set(loaded) - ALLOWED_PERSONA_PACK_FIELDS):
        errors.append(ValidationIssue(field=field, message=f"unknown field: {field}", code="unknown_field"))

    pack_id = str(loaded.get("profile_pack_id") or loaded.get("id") or "").strip()
    if not pack_id:
        errors.append(ValidationIssue(field="profile_pack_id", message="profile_pack_id is required"))

    players = loaded.get("players")
    if not isinstance(players, list):
        errors.append(ValidationIssue(field="players", message="players must be a list"))
        players = []

    if len(players) != expected_player_count:
        errors.append(ValidationIssue(field="players", message=f"persona pack must contain exactly {expected_player_count} players"))

    seen_seats: set[int] = set()
    normalized_players: list[dict[str, Any]] = []
    for index, player in enumerate(players):
        field_prefix = f"players.{index}"
        if not isinstance(player, dict):
            errors.append(ValidationIssue(field=field_prefix, message="player must be a mapping"))
            continue

        missing = sorted(PERSONA_REQUIRED_FIELDS - set(player))
        for field in missing:
            errors.append(ValidationIssue(field=f"{field_prefix}.{field}", message=f"{field} is required"))
        for field in sorted(set(player) - PERSONA_ALLOWED_FIELDS):
            errors.append(ValidationIssue(field=f"{field_prefix}.{field}", message=f"unknown field: {field}"))

        seat = _int_or_none(player.get("seat"))
        if seat is None or seat < 1 or seat > expected_player_count:
            errors.append(ValidationIssue(field=f"{field_prefix}.seat", message="seat must be between 1 and 12"))
        elif seat in seen_seats:
            errors.append(ValidationIssue(field=f"{field_prefix}.seat", message=f"duplicate seat: {seat}"))
        else:
            seen_seats.add(seat)

        for field in PERSONA_LEVEL_FIELDS:
            value = str(player.get(field, "")).strip()
            if value not in PERSONA_LEVELS:
                errors.append(ValidationIssue(field=f"{field_prefix}.{field}", message=f"{field} must be low, medium, or high"))

        preferred_roles = player.get("preferred_roles", [])
        if preferred_roles and not isinstance(preferred_roles, list):
            errors.append(ValidationIssue(field=f"{field_prefix}.preferred_roles", message="preferred_roles must be a list"))
            preferred_roles = []
        for role in preferred_roles:
            if str(role) not in PERSONA_ALLOWED_ROLES:
                errors.append(ValidationIssue(field=f"{field_prefix}.preferred_roles", message=f"unsupported preferred role: {role}"))

        _validate_persona_lengths(field_prefix, player, errors)
        _scan_text(field_prefix, player, errors)

        normalized_players.append(_normalize_persona_player(player))

    normalized = {
        "profile_pack_id": pack_id,
        "name": loaded.get("name", ""),
        "version": loaded.get("version", 1),
        "players": sorted(normalized_players, key=lambda item: item.get("seat", 0)),
    }
    return ValidationResult(
        valid=not errors,
        summary={"profile_pack_id": pack_id, "player_count": len(players)},
        normalized=normalized,
        errors=errors,
        diff_against_default=[],
    )


def _normalize_ruleset(
    loaded: dict[str, Any],
    ruleset_id: str,
    player_count: int,
    roles: dict[str, Any],
    constraints: dict[str, Any],
) -> dict[str, Any]:
    normalized_constraints = dict(DEFAULT_RULESET_NORMALIZED["constraints"])
    for key, value in constraints.items():
        if key in ALLOWED_CONSTRAINTS:
            normalized_constraints[key] = value
    return {
        "ruleset_id": ruleset_id,
        "name": loaded.get("name", ""),
        "version": loaded.get("version", 1),
        "player_count": player_count,
        "roles": roles,
        "abilities": loaded.get("abilities", []),
        "night_order": loaded.get("night_order", []),
        "victory": loaded.get("victory", {}),
        "constraints": normalized_constraints,
    }


def _diff_against_default(normalized: dict[str, Any]) -> list[dict[str, Any]]:
    diff: list[dict[str, Any]] = []
    default_constraints = DEFAULT_RULESET_NORMALIZED["constraints"]
    constraints = normalized.get("constraints", {})
    for key, default_value in default_constraints.items():
        uploaded = constraints.get(key)
        if uploaded != default_value:
            diff.append(
                {
                    "path": f"constraints.{key}",
                    "default": default_value,
                    "uploaded": uploaded,
                }
            )
    return diff


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validate_persona_lengths(prefix: str, player: dict[str, Any], errors: list[ValidationIssue]) -> None:
    if len(str(player.get("name", ""))) > 24:
        errors.append(ValidationIssue(field=f"{prefix}.name", message="name is too long"))
    if len(str(player.get("speech_style", ""))) > 80:
        errors.append(ValidationIssue(field=f"{prefix}.speech_style", message="speech_style is too long"))
    catchphrases = player.get("catchphrases", [])
    if catchphrases and not isinstance(catchphrases, list):
        errors.append(ValidationIssue(field=f"{prefix}.catchphrases", message="catchphrases must be a list"))
        return
    for phrase in catchphrases:
        if len(str(phrase)) > 60:
            errors.append(ValidationIssue(field=f"{prefix}.catchphrases", message="catchphrase is too long"))


def _normalize_persona_player(player: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for field in sorted(PERSONA_ALLOWED_FIELDS):
        if field in player:
            normalized[field] = player[field]
    if "preferred_roles" not in normalized:
        normalized["preferred_roles"] = []
    if "catchphrases" not in normalized:
        normalized["catchphrases"] = []
    return normalized


def _scan_text(path: str, value: Any, errors: list[ValidationIssue]) -> None:
    if isinstance(value, str):
        lowered = value.lower()
        for marker in PROMPT_INJECTION_MARKERS:
            if marker in lowered:
                errors.append(
                    ValidationIssue(
                        field=path or "text",
                        message="uploaded text contains a forbidden instruction or executable marker",
                        code="forbidden_text",
                    )
                )
                return
        # Unicode 同形字符注入检测
        for suspicious_char in _UNICODE_SUSPICIOUS_RANGES:
            if suspicious_char in value:
                errors.append(
                    ValidationIssue(
                        field=path or "text",
                        message="uploaded text contains suspicious Unicode characters (zero-width or bidirectional override)",
                        code="unicode_injection",
                    )
                )
                return
    elif isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            _scan_text(child_path, child, errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{path}.{index}" if path else str(index)
            _scan_text(child_path, child, errors)
