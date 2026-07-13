# -*- coding: utf-8 -*-
"""
验证 reflection sanitization helper 拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-13

使用示例:
    >>> python -m pytest tests/memory/test_reflection_sanitization.py -q
"""

from __future__ import annotations


def test_reflection_sanitization_helpers_remain_compatibly_importable() -> None:
    from werewolf_agent.memory import reflection
    from werewolf_agent.memory import reflection_sanitization

    assert reflection._iter_section_items is reflection_sanitization._iter_section_items
    assert reflection._scrub_ids is reflection_sanitization._scrub_ids
    assert reflection._cap_source_text is reflection_sanitization._cap_source_text


def test_reflection_sanitization_anonymizes_prior_game_ids_to_stable_labels() -> None:
    from werewolf_agent.memory.reflection_sanitization import anonymize_player_ids

    text = anonymize_player_ids("p01 应先质疑 p02，player_12 也要复核")

    assert text == "历史玩家A 应先质疑 历史玩家B，历史玩家C 也要复核"
    assert "p01" not in text


def test_reflection_sanitization_scrubs_player_ids() -> None:
    from werewolf_agent.memory.reflection_sanitization import _scrub_ids

    assert _scrub_ids("p07 和 player_12 都需要隐藏") == "[玩家ID已省略] 和 [玩家ID已省略] 都需要隐藏"
