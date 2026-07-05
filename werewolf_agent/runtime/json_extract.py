# -*- coding: utf-8 -*-
"""
从文本中提取第一个完整 JSON 对象。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.runtime.json_extract import extract_first_balanced_json_object
    >>> extract_first_balanced_json_object('prefix {"a": 1}')
"""

from __future__ import annotations

import json
from typing import Any


def extract_first_balanced_json_object(text: str) -> Any:
    """扫描第一个括号平衡的 JSON 对象，解析失败或不存在时返回 None。"""
    start: int | None = None
    depth = 0
    in_str = False
    escape = False
    for idx, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:idx + 1])
                except json.JSONDecodeError:
                    start = None
    return None
