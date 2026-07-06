# -*- coding: utf-8 -*-
"""
验证 context strategy directive 辅助函数拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/runtime/test_context_strategy_directives.py -q
"""

from __future__ import annotations


def test_strategy_directive_helpers_remain_compatibly_importable() -> None:
    from werewolf_agent.runtime import context
    from werewolf_agent.runtime import context_strategy_directives

    assert context._merge_strategy_directive is context_strategy_directives._merge_strategy_directive
    assert context._directive_size is context_strategy_directives._directive_size
    assert context._cap_strategy_directive is context_strategy_directives._cap_strategy_directive
    assert context._MAX_STRATEGY_DIRECTIVE_TOKENS == context_strategy_directives._MAX_STRATEGY_DIRECTIVE_TOKENS
    assert context._ROUND_SPECIFIC_DROP_KEYS is context_strategy_directives._ROUND_SPECIFIC_DROP_KEYS
