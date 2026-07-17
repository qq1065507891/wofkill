# -*- coding: utf-8 -*-
"""
定义狼人 system prompt 的关键语义条款及其严格送达合同标记。

作者: Project contributors
创建日期: 2026-07-18
修改日期: 2026-07-18

使用示例:
    >>> len(WEREWOLF_CRITICAL_SEMANTIC_CLAUSES)
    4
"""

from __future__ import annotations


WEREWOLF_TARGET_SEMANTICS_HEADER = "\u3010\u72fc\u4eba\u591c\u95f4\u76ee\u6807\u8bed\u4e49\u3011"
WEREWOLF_CRITICAL_SEMANTIC_CLAUSES: tuple[tuple[str, str], ...] = (
    ("backup_not_second_kill", "\u5907\u5200\u4e0d\u662f\u5973\u5deb\u6551\u4eba\u540e\u7684\u7b2c\u4e8c\u5200"),
    ("dead_player_not_target", "\u6b7b\u4ea1\u73a9\u5bb6\u4e0d\u53ef\u4f5c\u4e3a\u51fb\u6740\u76ee\u6807"),
    ("candidate_list_not_fact", "\u7cfb\u7edf\u63d0\u4f9b\u7684\u5019\u9009\u5217\u8868\u4e0d\u662f\u5c40\u5185\u4e8b\u5b9e"),
    ("captain_support_requires_source", "\u961f\u957f\u4e0d\u5f97\u4f2a\u9020\u652f\u6301\u8005\uff1b\u53ea\u6709\u5e26 source_event_id \u7684\u672c\u591c\u7ed3\u6784\u5316 stance \u624d\u80fd\u4f5c\u4e3a\u961f\u53cb\u652f\u6301\u8bc1\u636e"),
)

WEREWOLF_ROLE_GUIDE = (
    "\u72fc\u4eba\u89c4\u5219\uff1a\u591c\u95f4\u4e0e\u961f\u53cb\u8ba8\u8bba\u51fb\u6740\u76ee\u6807\uff0c\u53ef\u6309\u5408\u6cd5\u884c\u52a8\u9009\u62e9\u51fb\u6740\u3001\u81ea\u5200\u6216\u7a7a\u5200\u3002"
    "\u767d\u5929\u53ef\u5728\u89c4\u5219\u5141\u8bb8\u65f6\u81ea\u7206\uff1b\u81ea\u7206\u540e\u7acb\u5373\u51fa\u5c40\u3001\u65e0\u9057\u8a00\uff0c\u5e76\u4e2d\u65ad\u5f53\u524d\u767d\u5929\u3002"
    "\u53ef\u4ee5\u608d\u8df3\u9884\u8a00\u5bb6\u4e0a\u8b66\u5bf9\u6297\u771f\u9884\u8a00\u5bb6\u3002\n"
    + WEREWOLF_TARGET_SEMANTICS_HEADER
    + "\uff1b"
    + "\uff1b".join(clause for _section_id, clause in WEREWOLF_CRITICAL_SEMANTIC_CLAUSES)
    + "\u3002"
)


__all__ = [
    "WEREWOLF_CRITICAL_SEMANTIC_CLAUSES",
    "WEREWOLF_ROLE_GUIDE",
    "WEREWOLF_TARGET_SEMANTICS_HEADER",
]
