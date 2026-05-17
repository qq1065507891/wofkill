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
    records: dict[str, CustomConfigRecord] = field(default_factory=dict)

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
        self.records[config_id] = record
        return record
