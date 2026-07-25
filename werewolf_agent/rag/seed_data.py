# -*- coding: utf-8 -*-
"""
功能描述：从 YAML 文件加载 RAG 种子数据并校验，替代硬编码种子逻辑。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-26
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypeVar

import yaml

from werewolf_agent.rag.schemas import (
    CaseMetadata,
    CaseType,
    QualityGrade,
    RAGEntry,
    ReviewStatus,
    SourceMetadata,
    SourceType,
    VisibilityBoundary,
)
from werewolf_agent.rag.ingestion import CaseIngester

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Enum resolution maps (YAML string value -> Python enum member)
# ---------------------------------------------------------------------------

_CASE_TYPE_MAP: dict[str, CaseType] = {m.value: m for m in CaseType}
_QUALITY_GRADE_MAP: dict[str, QualityGrade] = {m.value: m for m in QualityGrade}
_REVIEW_STATUS_MAP: dict[str, ReviewStatus] = {m.value: m for m in ReviewStatus}
_SOURCE_TYPE_MAP: dict[str, SourceType] = {m.value: m for m in SourceType}
_VISIBILITY_MAP: dict[str, VisibilityBoundary] = {m.value: m for m in VisibilityBoundary}

_SEED_YAML_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "rag_seeds" / "seed_entries.yaml"


def _resolve_enum(
    name: str,
    value: str | None,
    mapping: dict[str, T],
    field_label: str,
) -> T | None:
    """Resolve a string value to an enum member, or return *None* if value is empty."""
    if not value:
        return None
    member = mapping.get(value)
    if member is None:
        raise ValueError(f"Unknown {field_label} value '{value}' for entry '{name}'")
    return member


def _build_source(raw: dict[str, Any], entry_id: str) -> SourceMetadata:
    """Build a ``SourceMetadata`` from the YAML ``source`` sub-dict."""
    source_type = _resolve_enum(entry_id, raw.get("source_type"), _SOURCE_TYPE_MAP, "source_type")
    if source_type is None:
        raise ValueError(f"Entry '{entry_id}' is missing source_type in source metadata")
    return SourceMetadata(
        source_type=source_type,
        source_url=raw.get("source_url", ""),
        source_title=raw.get("source_title", ""),
        source_author=raw.get("source_author", ""),
        publish_date=raw.get("publish_date", ""),
        collected_at="",  # let CaseIngester auto-timestamp
    )


def _build_metadata(raw: dict[str, Any], entry_id: str) -> CaseMetadata:
    """Build ``CaseMetadata`` from the YAML ``metadata`` sub-dict."""
    case_type = _resolve_enum(entry_id, raw.get("case_type"), _CASE_TYPE_MAP, "case_type")
    if case_type is None:
        raise ValueError(f"Entry '{entry_id}' is missing case_type")

    quality_grade = _resolve_enum(entry_id, raw.get("quality_grade"), _QUALITY_GRADE_MAP, "quality_grade")
    review_status = _resolve_enum(entry_id, raw.get("review_status"), _REVIEW_STATUS_MAP, "review_status")
    visibility = _resolve_enum(entry_id, raw.get("visibility_boundary"), _VISIBILITY_MAP, "visibility_boundary")

    source_raw = raw.get("source", {})
    if not isinstance(source_raw, dict):
        raise ValueError(f"Entry '{entry_id}' has invalid source metadata (expected dict)")

    return CaseMetadata(
        case_type=case_type,
        quality_grade=quality_grade or QualityGrade.UNREVIEWED,
        review_status=review_status or ReviewStatus.PENDING,
        reviewer=raw.get("reviewer", ""),
        ruleset_id=raw.get("ruleset_id", ""),
        player_count=raw.get("player_count", 12),
        phase=raw.get("phase", ""),
        role_perspective=raw.get("role_perspective", ""),
        visibility_boundary=visibility or VisibilityBoundary.PLAYER_PERSPECTIVE,
        source=_build_source(source_raw, entry_id),
        tags=raw.get("tags", []),
    )


def _build_entry(raw: dict[str, Any]) -> RAGEntry:
    """Build a single ``RAGEntry`` from one YAML item."""
    entry_id = raw.get("entry_id", "")
    if not entry_id:
        raise ValueError("Found a seed entry without entry_id")
    metadata = _build_metadata(raw.get("metadata", {}), entry_id)
    return RAGEntry(
        schema_version=raw.get("schema_version", 1),
        entry_id=entry_id,
        title=raw.get("title", ""),
        summary=raw.get("summary", ""),
        content_type=raw.get("content_type", "strategy"),
        tactical_frame=raw.get("tactical_frame"),
        key_decisions=raw.get("key_decisions", []),
        short_quotes=raw.get("short_quotes", []),
        metadata=metadata,
    )


def _load_yaml(path: Path) -> list[dict[str, Any]]:
    """Load and return the raw list from the YAML file."""
    if not path.exists():
        raise FileNotFoundError(f"Seed YAML not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, list):
        raise TypeError(f"Expected a list in {path}, got {type(data).__name__}")
    return data


def create_seed_entries(
    yaml_path: str | Path | None = None,
) -> list[RAGEntry]:
    """Load seed entries from YAML and validate each through ``CaseIngester``.

    This is the public API and replaces the old hard-coded function in
    ``ingestion.py``.  Every entry is ingested (validated + stored) so the
    result is identical to the previous implementation.
    """
    path = Path(yaml_path) if yaml_path is not None else _SEED_YAML_PATH
    raw_entries = _load_yaml(path)

    ingester = CaseIngester()
    entries: list[RAGEntry] = []

    for raw in raw_entries:
        entry = _build_entry(raw)
        ingester.ingest(entry)  # full validation
        entries.append(entry)

    logger.info("Loaded %d seed entries from %s", len(entries), path)
    return entries
