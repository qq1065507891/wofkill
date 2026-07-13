# -*- coding: utf-8 -*-
"""
验证 ReflectionSynthesizer 拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-13

使用示例:
    >>> python -m pytest tests/memory/test_reflection_synthesis.py -q
"""

from __future__ import annotations

from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.memory.reflection_synthesis import (
    ReflectionClaim,
    ReflectionDraft,
    ReflectionLesson,
    verify_reflection_draft,
)


def test_reflection_synthesizer_remains_compatibly_importable() -> None:
    from werewolf_agent.memory import reflection
    from werewolf_agent.memory import reflection_synthesis

    assert reflection.ReflectionSynthesizer is reflection_synthesis.ReflectionSynthesizer


def _state() -> GameState:
    return GameState(
        game_id="g1",
        players={
            "p01": PlayerState(id="p01", role="seer", alive=True),
            "p02": PlayerState(id="p02", role="werewolf", alive=False),
            "p03": PlayerState(id="p03", role="witch", alive=True),
        },
        events=[
            GameEvent(type="role_revealed", payload={"player_id": "p01", "role": "seer"}),
            GameEvent(type="vote", payload={"voter": "p01", "target": "p02"}),
            GameEvent(type="player_died", payload={"player_id": "p02", "reason": "exile"}),
            GameEvent(type="witch_poison_used", payload={"target_id": "p02"}),
        ],
    )


def _draft(claim: ReflectionClaim) -> ReflectionDraft:
    return ReflectionDraft(
        claims=[claim],
        lessons=[ReflectionLesson(
            lesson_id="l1",
            abstraction="投票前先核对公开证据链，不把历史身份映射到新局。",
            claim_dependencies=[claim.claim_id],
        )],
    )


def test_verify_reflection_rejects_wrong_role_vote_death_and_potion_facts() -> None:
    claims = [
        ReflectionClaim(claim_id="c1", event_ref="g1:0", claim_type="role", subject_id="p01", value="werewolf"),
        ReflectionClaim(claim_id="c2", event_ref="g1:1", claim_type="vote", subject_id="p01", target_id="p03"),
        ReflectionClaim(claim_id="c3", event_ref="g1:2", claim_type="death", subject_id="p02", value="wolf_kill"),
        ReflectionClaim(claim_id="c4", event_ref="g1:3", claim_type="potion", subject_id="p03", target_id="p01", value="poison"),
    ]
    for claim in claims:
        result = verify_reflection_draft(_draft(claim), _state())
        assert result.verified_lessons == []
        assert result.rejected_fact_count == 1
        assert result.rejected_lesson_count == 1


def test_verify_reflection_rejects_unknown_ref_type_and_partially_false_lesson() -> None:
    valid = ReflectionClaim(claim_id="valid", event_ref="g1:1", claim_type="vote", subject_id="p01", target_id="p02")
    unknown_ref = ReflectionClaim(claim_id="bad-ref", event_ref="g1:99", claim_type="vote", subject_id="p01", target_id="p02")
    wrong_type = ReflectionClaim(claim_id="bad-type", event_ref="g1:2", claim_type="vote", subject_id="p01", target_id="p02")
    draft = ReflectionDraft(
        claims=[valid, unknown_ref, wrong_type],
        lessons=[
            ReflectionLesson(lesson_id="partial", abstraction="先核验再投票。", claim_dependencies=["valid", "bad-ref"]),
            ReflectionLesson(lesson_id="valid-lesson", abstraction="投票前核验公开票型。", claim_dependencies=["valid"]),
        ],
    )
    result = verify_reflection_draft(draft, _state())
    assert result.rejected_fact_count == 2
    assert result.rejected_lesson_count == 1
    assert [lesson.lesson_id for lesson in result.verified_lessons] == ["valid-lesson"]
