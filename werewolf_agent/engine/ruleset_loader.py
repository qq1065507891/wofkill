# -*- coding: utf-8 -*-
"""
规则集数据结构与 YAML 加载入口。

作者: Project contributors
创建日期: 2026-07-07

使用示例:
    >>> ruleset = load_ruleset_from_yaml("config/rulesets/pre_witch_hunter_idiot_mixed.yaml")
    >>> ruleset.player_count
    12
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Ruleset:
    raw: dict[str, Any]

    @property
    def player_count(self) -> int:
        return int(self.raw["player_count"])


def load_ruleset_from_yaml(path: str | Path) -> Ruleset:
    """从 YAML 文件读取规则集并包装为 Ruleset。"""
    ruleset_path = Path(path)
    data = yaml.safe_load(ruleset_path.read_text(encoding="utf-8"))
    return Ruleset(raw=data)


__all__ = ["Ruleset", "load_ruleset_from_yaml"]
