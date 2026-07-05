# -*- coding: utf-8 -*-
"""
测试遗言阶段的策略指令构建函数。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.runtime.last_words_directives import build_last_words_directive
    >>> build_last_words_directive("seer")
"""

from werewolf_agent.runtime.last_words_directives import (
    build_exile_last_words_strategy,
    build_hunter_last_words_directive,
    build_hybrid_last_words_directive,
    build_last_words_directive,
)


def test_build_last_words_directive_mentions_public_role() -> None:
    """基础遗言指令应说明身份已公开并列出各身份重点。"""
    directive = build_last_words_directive("seer")

    assert "身份已公开（seer）" in directive
    assert "预言家：交代你所有的验人结果" in directive


def test_build_hunter_last_words_directive_lists_alive_targets() -> None:
    """猎人遗言指令应列出当前可带走目标。"""
    directive = build_hunter_last_words_directive(["p02", "p03"])

    assert "你有权开枪带走一名玩家" in directive
    assert "当前存活玩家（不含你）: ['p02', 'p03']" in directive


def test_build_hybrid_last_words_directive_blocks_identity_leak() -> None:
    """混血儿遗言指令应禁止泄漏身份和主人信息。"""
    directive = build_hybrid_last_words_directive()

    assert "严禁泄漏混血儿身份" in directive
    assert "不要透露你的主人是谁" in directive


def test_build_exile_last_words_strategy_adds_role_specific_sections() -> None:
    """完整遗言策略应按角色追加专属段落。"""
    hunter = build_exile_last_words_strategy("hunter", ["p02"])
    hybrid = build_exile_last_words_strategy("hybrid", ["p02"])
    villager = build_exile_last_words_strategy("villager", ["p02"])

    assert "last_words" in hunter
    assert "hunter_last_words" in hunter
    assert "hybrid_last_words" in hybrid
    assert set(villager) == {"last_words"}
