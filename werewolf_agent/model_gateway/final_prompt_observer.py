# -*- coding: utf-8 -*-
"""
定义 provider 最终请求中 system 内容的只读观察协议与安全通知入口。

作者: Project contributors
创建日期: 2026-07-13
修改日期: 2026-07-18

使用示例:
    >>> proof = FinalPromptAssembly(b"rules", "system", None, "anthropic", "m")
    >>> proof.final_system_location
    'system'
"""

from __future__ import annotations

import logging
import json
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
    provider_payload_bytes: bytes = b""


@dataclass(frozen=True)
class FinalPromptContract:
    """定义 provider 调用前必须出现在最终 system 字节中的稳定区块。"""

    contract_id: str
    version: str
    required_sections: tuple[tuple[str, bytes], ...]


class FinalPromptContractError(RuntimeError):
    """最终 system 缺少合同区块，必须在 provider HTTP 前终止调用。"""



class RouterPromptContractCompatibilityError(RuntimeError):
    """不支持最终 prompt 观察器的旧 provider 不能绕过送达前合同。"""


def canonical_provider_payload(payload: object) -> bytes:
    """稳定序列化实际 provider 请求体，供 HMAC 证明且不持久化原文。"""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def validate_final_prompt_contract(
    assembly: FinalPromptAssembly,
    contract: FinalPromptContract,
) -> dict[str, bool]:
    """只检查真实 provider payload 的 system 字节，不持久化提示词原文。"""
    ordered_offsets: list[int] = []
    confirmations: dict[str, bool] = {}
    for section_id, marker in contract.required_sections:
        first = assembly.system_bytes.find(marker)
        duplicate = first >= 0 and assembly.system_bytes.find(marker, first + 1) >= 0
        confirmations[section_id] = first >= 0 and not duplicate
        if section_id != "persona":
            ordered_offsets.append(first)
    missing = [
        section_id for section_id, confirmed in confirmations.items() if not confirmed
    ]
    if missing or ordered_offsets != sorted(ordered_offsets):
        detail = missing or ["section_order"]
        raise FinalPromptContractError(
            "final prompt contract missing required sections: " + ",".join(detail)
        )
    return confirmations


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
    except FinalPromptContractError:
        # 合同缺失是请求本身不安全，而不是监控故障；必须在 HTTP 前失败。
        raise
    except Exception as exc:  # noqa: BLE001 - 普通监控故障不得阻断模型调用
        _LOGGER.warning(
            "final prompt observer failed stage=final_payload error_type=%s",
            type(exc).__name__,
        )


__all__ = [
    "FinalPromptAssembly",
    "FinalPromptContract",
    "FinalPromptContractError",
    "RouterPromptContractCompatibilityError",
    "FinalPromptObserver",
    "bind_attempt",
    "canonical_provider_payload",
    "notify_final_prompt_observer",
    "validate_final_prompt_contract",
]
