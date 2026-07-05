# -*- coding: utf-8 -*-
"""
功能描述：**：提供token化正则（ASCII单词字符/单个CJK字符），供evaluation.attribution和memory.reflection共享，避免循环导入
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9_]+|[一-鿿]")


def tokenize(text: str) -> set[str]:
    """Token set: lowercase ASCII runs + individual CJK characters."""
    return set(_TOKEN_RE.findall(str(text or "").lower()))


def jaccard(left: str, right: str) -> float:
    """Jaccard similarity over token sets; 0.0 if either side is empty."""
    a = tokenize(left)
    b = tokenize(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
