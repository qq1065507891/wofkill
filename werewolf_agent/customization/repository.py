"""Local repository for validated customization configs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class CustomConfigRecord:
    config_id: str
    config_type: str
    raw_yaml: str
    normalized: dict[str, Any]
    validation_result: dict[str, Any]
    content_hash: str
    status: str
    version: str
    maturity: str
    compatibility_matrix: dict[str, Any]
    diff_against_default: list[dict[str, Any]]
    creator_id: str
    created_at: str
    updated_at: str


@dataclass
class InMemoryCustomizationRepository:
    """内存中的自定义配置仓库，使用私有 _records 字典存储。"""
    _records: dict[str, CustomConfigRecord] = field(default_factory=dict)
    # P-A2: simple in-memory ruleset cache, keyed by ruleset_id. Holds raw
    # dict payloads (post-validation normalized form) so save/load round
    # trips work without going through the heavier CustomConfigRecord path.
    _rulesets: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def records(self) -> dict[str, CustomConfigRecord]:
        """只读访问内部记录。"""
        return dict(self._records)

    def list_records(self, config_type: str | None = None) -> list[CustomConfigRecord]:
        """列出所有配置记录，可选按 config_type 过滤。"""
        result = list(self._records.values())
        if config_type is not None:
            result = [r for r in result if r.config_type == config_type]
        return result

    def save(
        self,
        *,
        config_type: str,
        raw_yaml: str,
        normalized: dict[str, Any],
        validation_result: dict[str, Any],
        creator_id: str,
    ) -> CustomConfigRecord:
        content_hash = hashlib.sha256(raw_yaml.encode("utf-8")).hexdigest()
        config_id = f"{config_type}_{content_hash[:12]}"
        now = datetime.now(timezone.utc).isoformat()
        record = CustomConfigRecord(
            config_id=config_id,
            config_type=config_type,
            raw_yaml=raw_yaml,
            normalized=normalized,
            validation_result=validation_result,
            content_hash=content_hash,
            status=str(normalized.get("status", validation_result.get("summary", {}).get("status", "draft"))),
            version=str(normalized.get("version", "1")),
            maturity="validated",
            compatibility_matrix={
                "status": str(normalized.get("status", "")),
                "unsupported_roles": list(normalized.get("unsupported_roles", [])),
                "missing_abilities": list(normalized.get("missing_abilities", [])),
            },
            diff_against_default=list(validation_result.get("diff_against_default", [])),
            creator_id=creator_id,
            created_at=now,
            updated_at=now,
        )
        self._records[config_id] = record
        return record

    # P-A2: minimal ruleset save/load pair. Used by the in-memory backend
    # for symmetry with the durable PostgresGameRepository custom_config
    # path. Returns None when the ruleset id is unknown.

    def save_ruleset(self, ruleset_id: str, config: dict[str, Any]) -> None:
        self._rulesets[ruleset_id] = config

    def load_ruleset(self, ruleset_id: str) -> dict[str, Any] | None:
        return self._rulesets.get(ruleset_id)
