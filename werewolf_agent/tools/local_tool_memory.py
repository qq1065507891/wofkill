# -*- coding: utf-8 -*-
"""
本地工具的记忆查询与复盘写入 helper。

作者: Project contributors
创建日期: 2026-07-08

使用示例:
    >>> from werewolf_agent.tools.local_tool_memory import query_cognition_matrix
    >>> query_cognition_matrix("p01", "p02", memory_store=store)
"""

from __future__ import annotations

from typing import Any

_DEFAULT_MEMORY_STORE: Any = None


def get_default_memory_store() -> Any:
    """返回模块级默认 MemoryStore，首次使用时延迟创建。"""
    global _DEFAULT_MEMORY_STORE
    if _DEFAULT_MEMORY_STORE is None:
        from werewolf_agent.memory.store import MemoryStore

        _DEFAULT_MEMORY_STORE = MemoryStore()
    return _DEFAULT_MEMORY_STORE


def query_cognition_matrix(
    viewer_id: str,
    target_id: str,
    memory_store: Any | None = None,
) -> dict[str, Any]:
    """从 MemoryStore 查询指定 viewer 对 target 的认知矩阵条目。"""
    store = memory_store or get_default_memory_store()
    response: dict[str, Any] = {
        "viewer_id": viewer_id,
        "target_id": target_id,
    }
    matrix = _matrix_for_viewer(store, viewer_id, target_id)
    entry = matrix.get(target_id) if matrix is not None else None
    if entry is None:
        response["available"] = False
        response["note"] = f"no cognition entry for {target_id}"
        return response

    response["available"] = True
    response["faction_read"] = entry.faction_read
    response["trust"] = entry.trust
    response["key_evidence"] = [
        evidence.to_dict() if hasattr(evidence, "to_dict") else evidence
        for evidence in entry.key_evidence
    ]
    response["open_questions"] = list(entry.open_questions)
    response["role_probabilities"] = dict(entry.role_probabilities)
    return response


def write_review(
    game_id: str,
    player_id: str,
    review_data: dict[str, Any],
    memory_store: Any | None = None,
) -> dict[str, Any]:
    """把复盘数据写入 MemoryStore，返回持久化结果。"""
    store = memory_store or get_default_memory_store()
    try:
        review_id = store.save_review(
            game_id=game_id,
            player_id=player_id,
            review_data=review_data,
        )
        return {"persisted": True, "review_id": review_id}
    except Exception as exc:  # noqa: BLE001 - 工具调用方需要看到失败原因
        return {"persisted": False, "error": str(exc)}


def _matrix_for_viewer(
    memory_store: Any,
    viewer_id: str,
    target_id: str,
) -> Any | None:
    matrix = memory_store.get_matrix(viewer_id)
    if matrix is not None:
        return matrix
    try:
        memory_store.init_matrix(viewer_id, sorted({viewer_id, target_id}))
        return memory_store.get_matrix(viewer_id)
    except Exception:
        return None


__all__ = [
    "get_default_memory_store",
    "query_cognition_matrix",
    "write_review",
]
