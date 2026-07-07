# -*- coding: utf-8 -*-
"""
RAG 查询文本处理工具，提供轻量 token 化能力。

作者: Project contributors
创建日期: 2026-07-07
修改日期: 2026-07-07

使用示例:
    >>> from werewolf_agent.rag.query_processing import _tokenize
    >>> _tokenize("Seer 查验")
    ['seer', '查', '验']
"""

from __future__ import annotations

import re


def _tokenize(text: str) -> list[str]:
    """按英文数字词和单个 CJK 字符切分查询文本。"""
    tokens: list[str] = []
    for part in re.findall(r"[一-鿿]|[a-zA-Z0-9]+", text.lower()):
        tokens.append(part)
    return tokens
