# -*- coding: utf-8 -*-
"""
验证 ReflectionMemory repository 拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-13

使用示例:
    >>> python -m pytest tests/memory/test_reflection_repository.py -q
"""

from __future__ import annotations


def test_reflection_memory_remains_compatibly_importable() -> None:
    from werewolf_agent.memory import reflection
    from werewolf_agent.memory import reflection_repository

    assert reflection.ReflectionMemory is reflection_repository.ReflectionMemory
    assert reflection._LOG is reflection_repository._LOG


def test_query_live_returns_anonymized_verified_abstraction_only() -> None:
    from werewolf_agent.memory.reflection_repository import ReflectionMemory
    from werewolf_agent.memory.schemas import (
        CrossGameQuery, ReflectionEntryV2, ReflectionPromptCard,
        ReflectionQualityStatus, ReflectionSource,
    )

    entry = ReflectionEntryV2(
        entry_id="r1", game_id="old-game", player_id="p01", role="seer",
        quality_status=ReflectionQualityStatus.APPROVED,
        prompt_card=ReflectionPromptCard(
            theme="核验", lesson="p01 投 p02 前应核验证据", recommended_action="复核",
            misuse_risk="不映射身份", auto_verified=True,
        ),
        source=ReflectionSource(llm_self_review="原始草稿 p01 投 p02"),
    )
    memory = ReflectionMemory()
    memory.store_v2(entry)

    found = memory.query_live(CrossGameQuery(player_id="p01"))[0]

    assert found.game_id == "old-game"
    assert found.player_id == "历史玩家本人"
    assert found.prompt_card.lesson == "历史玩家A 投 历史玩家B 前应核验证据"
    assert found.source.llm_self_review == ""
