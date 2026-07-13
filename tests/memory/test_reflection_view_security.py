# -*- coding: utf-8 -*-
"""
验证跨局反思视图保留可聚合信息且不可泄露历史玩家标识。

作者: Project contributors
创建日期: 2026-07-13
"""

from __future__ import annotations

import json

from werewolf_agent.memory.reflection_repository import ReflectionMemory
from werewolf_agent.memory.schemas import (
    CrossGameQuery,
    ReflectionEntryV2,
    ReflectionMistakePattern,
    ReflectionPreservedStrength,
    ReflectionPromptCard,
    ReflectionQualityStatus,
)


def test_query_live_preserves_anonymized_patterns_and_strengths_for_aggregation() -> None:
    memory = ReflectionMemory()
    memory.store_v2(ReflectionEntryV2(
        entry_id="reflection_oldgame_p01", game_id="oldgame", player_id="p01", role="seer",
        quality_status=ReflectionQualityStatus.APPROVED,
        mistake_patterns=[ReflectionMistakePattern(
            category="vote_mistake", trigger="p02 发言后", wrong_action="p01 投给 p03",
            better_action="先核验 p02 的公开证据",
        )],
        preserved_strengths=[ReflectionPreservedStrength(
            category="evidence_check", behavior="p01 复核了 p02", reuse_condition="再次遇到 p03",
        )],
        prompt_card=ReflectionPromptCard(
            theme="核验", lesson="先核验证据", recommended_action="复核", misuse_risk="不映射身份",
            auto_verified=True,
        ),
    ))

    found = memory.query_live(CrossGameQuery(player_id="p01"))[0]
    second_memory = ReflectionMemory()
    second_memory.store_v2(memory.all_v2_entries()[0])
    second_view = second_memory.query_live(CrossGameQuery(player_id="p01"))[0]
    serialized = json.dumps(found.model_dump(mode="json"), ensure_ascii=False)
    pattern = memory.live_error_pattern("p01", role="seer")

    assert found.mistake_patterns and found.preserved_strengths
    assert "p01" not in serialized and "p02" not in serialized and "p03" not in serialized
    assert "oldgame" not in found.entry_id
    assert second_view.entry_id != found.entry_id
    assert pattern["top_mistakes"] == [("vote_mistake", 1)]
    assert pattern["preserved_strength_count"] == 1
