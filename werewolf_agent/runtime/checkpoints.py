# -*- coding: utf-8 -*-
"""Checkpoint and pause/resume support using langgraph-checkpoint.
    作者: Mike
    创建日期: 2025-01-15
    修改日期: 2026-07-05
    使用示例: 内部模块，无对外接口
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver


def make_checkpointer() -> MemorySaver:
    return MemorySaver()
