# -*- coding: utf-8 -*-
"""
定义 provider 最终请求中 system 内容的只读观察协议与安全通知入口。

作者: Project contributors
创建日期: 2026-07-13
修改日期: 2026-07-13

使用示例:
    >>> proof = FinalPromptAssembly(b"rules", "system", None, "anthropic", "m")
    >>> proof.final_system_location
    'system'
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FinalPromptAssembly:
    """仅在 HTTP 调用前短暂传递最终 system 字节，不负责持久化原文。"""

    system_bytes: bytes
    final_system_location: str
    final_system_message_index: int | None
    provider: str
    model: str
    attempt_kind: str = ""
    attempt_ordinal: int | None = None


FinalPromptObserver = Callable[[FinalPromptAssembly], None]


def bind_attempt(
    observer: FinalPromptObserver | None,
    *,
    attempt_kind: str,
    attempt_ordinal: int,
) -> FinalPromptObserver | None:
    """把 router 的逐尝试身份绑定到 provider 产生的真实 payload 证明。"""
    if observer is None:
        return None

    def _observe(assembly: FinalPromptAssembly) -> None:
        observer(replace(
            assembly,
            attempt_kind=attempt_kind,
            attempt_ordinal=attempt_ordinal,
        ))

    return _observe


def notify_final_prompt_observer(
    observer: FinalPromptObserver | None,
    assembly: FinalPromptAssembly,
) -> None:
    """隔离观察回调异常，避免监控故障改变 provider 请求语义。"""
    if observer is None:
        return
    try:
        observer(assembly)
    except Exception as exc:  # noqa: BLE001 - 观察器不得阻断模型调用
        _LOGGER.warning(
            "final prompt observer failed stage=final_payload error_type=%s",
            type(exc).__name__,
        )


__all__ = [
    "FinalPromptAssembly",
    "FinalPromptObserver",
    "bind_attempt",
    "notify_final_prompt_observer",
]
