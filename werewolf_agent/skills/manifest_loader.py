# -*- coding: utf-8 -*-
"""
从技能目录的 SKILL.md 文件加载技能定义。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.skills.manifest_loader import load_manifests
    >>> load_manifests()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from werewolf_agent.skills.schemas import (
    SkillDefinition,
    SkillFaction,
    SkillName,
)

logger = logging.getLogger(__name__)


def parse_skill_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """解析 SKILL.md 的 YAML frontmatter 和正文。"""
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text.strip()
    yaml_block, body = parts[1], parts[2]
    meta = yaml.safe_load(yaml_block) or {}
    return meta, body.strip()


def load_manifests(root: Path | None = None) -> list[SkillDefinition]:
    """从指定根目录加载所有技能 manifest。"""
    if root is None:
        root = Path(__file__).resolve().parent
    result: list[SkillDefinition] = []
    for skill_dir in sorted(root.iterdir()):
        if (
            not skill_dir.is_dir()
            or skill_dir.name.startswith("_")
            or skill_dir.name.startswith(".")
        ):
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        data, body = parse_skill_frontmatter(skill_md.read_text(encoding="utf-8"))
        if not data:
            continue
        try:
            result.append(_definition_from_manifest(data, body))
        except (KeyError, ValueError) as exc:
            logger.warning("Failed to load skill %s: %s", skill_dir.name, exc)
    return result


def _definition_from_manifest(data: dict[str, Any], body: str) -> SkillDefinition:
    return SkillDefinition(
        name=SkillName(data["name"]),
        display_name=data.get("display_name", ""),
        description=data.get("description", ""),
        applicable_roles=data.get("applicable_roles", []),
        applicable_phases=data.get("applicable_phases", []),
        applies_to_task_types=data.get("applies_to_task_types", []),
        faction=SkillFaction(data.get("faction", "common")),
        tags=data.get("tags", []),
        body=body,
    )
