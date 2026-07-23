# -*- coding: utf-8 -*-
"""
提供 deprecated/inert 的 Agent 超时配置兼容对象。

这些数值仅供旧导入方读取，不控制 provider HTTP 调用或 Runtime deadline。

作者: Mike
创建日期: 2025-01-15
修改日期: 2026-07-23
使用示例: 内部模块，无对外接口
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentTimeouts:
    """保留旧字段形状的惰性兼容配置。"""

    wolf_discussion_per_player: float = 180.0
    wolf_discussion_total: float = 600.0
    wolf_consensus: float = 180.0
    seer_check: float = 360.0
    witch_action: float = 360.0
    # 保留旧 seer/witch 字段名，避免外部导入方在迁移期间中断。
    seer: float = 360.0
    witch: float = 360.0
    day_speech: float = 240.0
    day_vote: float = 180.0
    hunter_shot: float = 120.0


AGENT_TIMEOUTS = AgentTimeouts()
