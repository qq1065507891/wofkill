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

import pytest


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


def test_query_live_uses_one_recursive_player_label_map_per_entry() -> None:
    from werewolf_agent.memory.reflection_repository import ReflectionMemory
    from werewolf_agent.memory.schemas import (
        CrossGameQuery, ReflectionEntryV2, ReflectionMistakePattern,
        ReflectionPromptCard, ReflectionQualityStatus,
    )

    entry = ReflectionEntryV2(
        entry_id="r-map", game_id="old-game", player_id="p01", role="seer",
        quality_status=ReflectionQualityStatus.APPROVED,
        mistake_patterns=[ReflectionMistakePattern(
            category="vote_mistake", trigger="p01 质疑 p02",
            wrong_action="p02 跟随 p03", better_action="p01 核验 p03",
        )],
        prompt_card=ReflectionPromptCard(
            theme="p03 票型", lesson="p01 复核 p02", trigger_signals=["p03 发言"],
            recommended_action="p02 先问 p01", misuse_risk="不映射 p03",
            auto_verified=True,
        ),
    )
    memory = ReflectionMemory()
    memory.store_v2(entry)

    found = memory.query_live(CrossGameQuery(player_id="p01"))[0]
    dump = found.model_dump(mode="json")

    assert dump["mistake_patterns"][0]["trigger"] == "历史玩家A 质疑 历史玩家B"
    assert dump["mistake_patterns"][0]["wrong_action"] == "历史玩家B 跟随 历史玩家C"
    assert dump["mistake_patterns"][0]["better_action"] == "历史玩家A 核验 历史玩家C"
    assert dump["prompt_card"]["lesson"] == "历史玩家A 复核 历史玩家B"
    assert dump["prompt_card"]["trigger_signals"] == ["历史玩家C 发言"]


@pytest.mark.parametrize("raise_on_failure", [False, True])
def test_store_v2_is_transactional_when_repository_write_fails(raise_on_failure) -> None:
    from werewolf_agent.memory.reflection_repository import ReflectionMemory
    from werewolf_agent.memory.schemas import ReflectionEntryV2, ReflectionPromptCard

    class BrokenRepository:
        def load_all_reflections(self):
            return []

        def save_reflection(self, data):
            raise OSError("storage unavailable")

    memory = ReflectionMemory(BrokenRepository())
    entry = ReflectionEntryV2(
        entry_id="r-fail", game_id="g1", player_id="p01", role="seer",
        prompt_card=ReflectionPromptCard(
            theme="核验", lesson="先核验", recommended_action="复核",
            misuse_risk="不映射历史身份",
        ),
    )

    if raise_on_failure:
        with pytest.raises(OSError, match="storage unavailable"):
            memory.store_v2(entry, raise_on_failure=True)
    else:
        memory.store_v2(entry, raise_on_failure=False)

    assert memory.all_v2_entries() == []
