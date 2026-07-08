# -*- coding: utf-8 -*-
"""
定义法官 HITL 命令的数据结构和解析逻辑。

作者: Project contributors
创建日期: 2026-07-08
修改日期: 2026-07-08

使用示例:
    >>> from werewolf_agent.agents.judge_hitl_commands import HITLCommand
    >>> HITLCommand.parse("resume 2").command
    'resume'
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class HITLCommand:
    command: str
    args: list[str] = field(default_factory=list)
    timestamp: float = 0.0

    @classmethod
    def parse(cls, raw: str) -> "HITLCommand":
        parts = raw.strip().split()
        if not parts:
            return cls(command="")
        return cls(
            command=parts[0].lower(),
            args=parts[1:] if len(parts) > 1 else [],
            timestamp=time.time(),
        )


__all__ = ["HITLCommand"]
