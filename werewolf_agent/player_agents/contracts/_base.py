# -*- coding: utf-8 -*-
"""
提供玩家智能体契约共享的严格不可变模型、标识符、哈希类型和 JSON 冻结工具。

作者: Project contributors
创建日期: 2026-07-29
修改日期: 2026-07-29
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Annotated, Any, Self, TypeVar

from pydantic import BaseModel, ConfigDict, StringConstraints

NonEmptyId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=128),
]
ContentHash = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]


class StrictFrozenModel(BaseModel):
    """拒绝额外字段、隐式类型转换和实例修改的基础模型。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        """复制模型，并对所有更新重新执行完整契约校验。"""
        if not update:
            return super().model_copy(deep=deep)
        data = self.model_dump(round_trip=True)
        data.update(update)
        return type(self).model_validate(data)

    def copy(
        self,
        *,
        include: Any = None,
        exclude: Any = None,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        """将旧复制 API 委托给受校验路径，并拒绝不完整副本。"""
        if include is not None or exclude is not None:
            raise TypeError("partial copies are not supported for strict contracts")
        return self.model_copy(update=update, deep=deep)


T = TypeVar("T")


def require_unique(values: Iterable[T], *, field_name: str) -> tuple[T, ...]:
    """返回稳定元组，并拒绝会使引用语义产生歧义的重复项。"""
    items = tuple(values)
    if len(items) != len(set(items)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return items


def _freeze_json_value(value: Any) -> Any:
    """递归冻结有限 JSON 值，防止模型内部容器被外部修改。"""
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return MappingProxyType({
            key: _freeze_json_value(item)
            for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("value must be finite JSON data")


def _freeze_json_object(value: Any) -> Mapping[str, Any]:
    """冻结并校验 JSON 对象根值。"""
    frozen = _freeze_json_value(value)
    if not isinstance(frozen, Mapping):
        raise TypeError("value must be a JSON object")
    return frozen


def _thaw_json_value(value: Any) -> Any:
    """将冻结的 JSON 值恢复为可序列化的普通 Python 容器。"""
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value
