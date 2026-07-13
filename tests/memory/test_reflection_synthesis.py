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

import pytest
from pydantic import ValidationError

from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.memory.reflection_synthesis import (
    ReflectionClaim,
    ReflectionDraft,
    ReflectionLesson,
    ReflectionVerification,
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


def test_event_registry_verifies_real_seer_victory_hunter_and_sheriff_vote() -> None:
    state = GameState(
        game_id="g-rich", phase="finished", winning_faction="good",
        players={
            "p01": PlayerState(id="p01", role="seer"),
            "p02": PlayerState(id="p02", role="werewolf", alive=False),
            "p03": PlayerState(id="p03", role="hunter", alive=False),
            "p04": PlayerState(id="p04", role="hybrid"),
            "p05": PlayerState(id="p05", role="villager"),
        },
        hybrid_master_id="p05", hybrid_master_faction="good",
        events=[
            GameEvent(type="roles_assigned", payload={}),
            GameEvent(type="seer_check", payload={
                "target_id": "p02", "alignment": "werewolf",
            }),
            GameEvent(type="sheriff_vote_record", payload={
                "votes": [{"voter": "p01", "target": "p03"}],
            }),
            GameEvent(type="hunter_shot_public", payload={
                "hunter_id": "p03", "target_id": "p02",
            }),
            GameEvent(type="victory", payload={
                "winner": "good", "winning_faction": "good", "reason": "all_werewolves_out",
            }),
            GameEvent(type="hybrid_master_chosen", payload={
                "hybrid_id": "p04", "master_id": "p05",
            }),
        ],
    )
    claims = [
        ReflectionClaim(claim_id="faction", event_ref="g-rich:0", claim_type="faction", subject_id="p01", value="good"),
        ReflectionClaim(claim_id="seer", event_ref="g-rich:1", claim_type="seer_check", subject_id="p01", target_id="p02", value="werewolf"),
        ReflectionClaim(claim_id="sheriff", event_ref="g-rich:2", claim_type="vote", subject_id="p01", target_id="p03"),
        ReflectionClaim(claim_id="hunter", event_ref="g-rich:3", claim_type="skill", subject_id="p03", target_id="p02", value="hunter_shot"),
        ReflectionClaim(claim_id="win", event_ref="g-rich:4", claim_type="victory", subject_id="good", value="all_werewolves_out"),
        ReflectionClaim(claim_id="hybrid", event_ref="g-rich:5", claim_type="skill", subject_id="p04", target_id="p05", value="hybrid_bind"),
    ]

    result = verify_reflection_draft(ReflectionDraft(claims=claims), state)

    assert result.verified_claims == claims
    assert result.rejected_fact_count == 0


def test_event_registry_rejects_mismatched_supported_fact() -> None:
    state = GameState(
        game_id="g-seer",
        players={
            "p01": PlayerState(id="p01", role="seer"),
            "p02": PlayerState(id="p02", role="werewolf"),
        },
        events=[GameEvent(type="seer_check", payload={
            "seer_id": "p01", "target_id": "p02", "alignment": "werewolf",
        })],
    )
    wrong = ReflectionClaim(
        claim_id="wrong", event_ref="g-seer:0", claim_type="seer_check",
        subject_id="p01", target_id="p02", value="good",
    )

    result = verify_reflection_draft(ReflectionDraft(claims=[wrong]), state)

    assert result.verified_claims == []
    assert result.rejected_fact_count == 1


@pytest.mark.parametrize(
    "abstraction",
    [
        "证据不足时降低结论强度并比较替代方案。",
        "投票前先复核公开证据链。",
        "查验前比较多个候选方案。",
    ],
)
def test_fact_independent_general_strategy_accepts_target_free_methods(
    abstraction: str,
) -> None:
    safe = ReflectionLesson(
        lesson_id="general", abstraction=abstraction,
        claim_dependencies=[], lesson_kind="general_strategy",
    )

    result = verify_reflection_draft(ReflectionDraft(lessons=[safe]), _state())

    assert result.verified_lessons == [safe]
    assert result.rejected_lesson_count == 0


def test_fact_independent_general_strategy_rejects_concrete_fact() -> None:
    unsafe = ReflectionLesson(
        lesson_id="masked-fact", abstraction="预言家是 p01，狼人阵营获胜。",
        claim_dependencies=[], fact_independent=True,
    )

    result = verify_reflection_draft(ReflectionDraft(lessons=[unsafe]), _state())

    assert result.verified_lessons == []
    assert result.rejected_lesson_count == 1


@pytest.mark.parametrize(
    "abstraction",
    [
        "3号玩家是狼人，下局直接投他。",
        "3号是预言家，应当相信他。",
        "player_3 是女巫。",
        "狼人阵营获胜，下次优先找猎人。",
        "投他。",
        "投3号。",
        "查验结果是狼人。",
        "毒了p02。",
        "救了3号。",
        "某人死亡。",
        "狼人获胜。",
        "玩家死亡后判断胜负。",
        "参考 g-final:42 的结论。",
    ],
)
def test_fact_independent_lesson_rejects_specific_entities_roles_actions_and_refs(
    abstraction: str,
) -> None:
    lesson = ReflectionLesson(
        lesson_id="unsafe", abstraction=abstraction,
        claim_dependencies=[], lesson_kind="general_strategy",
    )

    result = verify_reflection_draft(ReflectionDraft(lessons=[lesson]), _state())

    assert result.verified_lessons == []
    assert result.rejected_lesson_count == 1


@pytest.mark.parametrize(
    "abstraction",
    [
        "投票前先复核公开证据链。",
        "查验前比较多个候选方案。",
        "证据不足时应降低结论强度。",
        "需要在证据冲突时比较替代解释。",
        "保持结论可修正并避免单线推断。",
    ],
)
def test_fact_independent_contract_accepts_depersonalized_normative_language(
    abstraction: str,
) -> None:
    lesson = ReflectionLesson(
        lesson_id="normative", abstraction=abstraction,
        claim_dependencies=[], lesson_kind="general_strategy",
    )

    result = verify_reflection_draft(ReflectionDraft(lessons=[lesson]), _state())

    assert result.verified_lessons == [lesson]


@pytest.mark.parametrize(
    "abstraction",
    [
        "我本局投给了3号。",
        "本人曾经救下了某位玩家。",
        "我们这局毒死了狼人。",
        "已经查过他。",
        "刀过某人后获胜。",
        "我应先复核证据。",
        "某人需要先比较证据。",
        "投给他前先核验证据。",
        "天气很好。",
    ],
)
def test_fact_independent_contract_rejects_personal_narrative_or_non_normative_language(
    abstraction: str,
) -> None:
    lesson = ReflectionLesson(
        lesson_id="narrative", abstraction=abstraction,
        claim_dependencies=[], lesson_kind="general_strategy",
    )

    result = verify_reflection_draft(ReflectionDraft(lessons=[lesson]), _state())

    assert result.verified_lessons == []


@pytest.mark.parametrize(
    "abstraction",
    [
        "面对预言家对跳时先比较公开查验链。",
        "保持自我修正并避免单线推断。",
        "发言时优先区分阵营推测与公开事实。",
    ],
)
def test_fact_independent_contract_allows_generic_role_terms_and_self_correction(
    abstraction: str,
) -> None:
    lesson = ReflectionLesson(
        lesson_id="generic-role", abstraction=abstraction,
        claim_dependencies=[], lesson_kind="general_strategy",
    )

    result = verify_reflection_draft(ReflectionDraft(lessons=[lesson]), _state())

    assert result.verified_lessons == [lesson]


@pytest.mark.parametrize("abstraction", ["当前信息不足。", "时间线很乱。"])
def test_fact_independent_contract_rejects_single_character_marker_false_positives(
    abstraction: str,
) -> None:
    lesson = ReflectionLesson(
        lesson_id="not-normative", abstraction=abstraction,
        claim_dependencies=[], lesson_kind="general_strategy",
    )

    result = verify_reflection_draft(ReflectionDraft(lessons=[lesson]), _state())

    assert result.verified_lessons == []


def test_fact_independent_contract_accepts_generic_player_process_rule() -> None:
    lesson = ReflectionLesson(
        lesson_id="generic-player",
        abstraction="玩家发言时应当区分公开事实与推测。",
        claim_dependencies=[], lesson_kind="general_strategy",
    )

    result = verify_reflection_draft(ReflectionDraft(lessons=[lesson]), _state())

    assert result.verified_lessons == [lesson]


@pytest.mark.parametrize("abstraction", ["公开信息比较混乱。", "复核失败。"])
def test_fact_independent_contract_rejects_bare_process_verbs(
    abstraction: str,
) -> None:
    lesson = ReflectionLesson(
        lesson_id="bare-verb", abstraction=abstraction,
        claim_dependencies=[], lesson_kind="general_strategy",
    )

    result = verify_reflection_draft(ReflectionDraft(lessons=[lesson]), _state())

    assert result.verified_lessons == []


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (ReflectionClaim, {"claim_id": "c", "event_ref": "g1:0", "claim_type": "vote", "subject_id": 1}),
        (ReflectionLesson, {"lesson_id": "l", "abstraction": "先核验", "claim_dependencies": ("c",)}),
        (ReflectionDraft, {"claims": (), "lessons": []}),
        (ReflectionVerification, {"rejected_fact_count": "1"}),
    ],
)
def test_reflection_contracts_are_strict_and_forbid_coercion(model, payload) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(payload)
