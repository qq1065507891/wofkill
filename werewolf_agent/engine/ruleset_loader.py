# -*- coding: utf-8 -*-
"""
规则集数据结构与 YAML 加载入口。

作者: Project contributors
创建日期: 2026-07-07
修改日期: 2026-07-16

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

    @property
    def max_consecutive_pre_resolution_no_kill(self) -> int:
        """返回结算前连续空刀阈值。"""
        constraints = self.raw.get("constraints") or {}
        game_rules = self.raw.get("game_rules") or {}
        value = constraints.get(
            "max_consecutive_pre_resolution_no_kill",
            game_rules.get("max_consecutive_pre_resolution_no_kill", 2),
        )
        _validate_no_kill_threshold(value)
        return value


def load_ruleset_from_yaml(path: str | Path) -> Ruleset:
    """从 YAML 文件读取规则集并包装为 Ruleset。"""
    ruleset_path = Path(path)
    data = yaml.safe_load(ruleset_path.read_text(encoding="utf-8"))
    ruleset = Ruleset(raw=data)
    ruleset.max_consecutive_pre_resolution_no_kill
    return ruleset


def _validate_no_kill_threshold(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            "max_consecutive_pre_resolution_no_kill must be a positive integer"
        )


__all__ = ["Ruleset", "load_ruleset_from_yaml"]
