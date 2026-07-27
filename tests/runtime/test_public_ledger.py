# -*- coding: utf-8 -*-
"""
验证公开账本与公开发言快照的可见性边界。

作者: Project contributors
创建日期: 2026-07-15
修改日期: 2026-07-27

兼容边界：legacy 可见性的空字符串按未设置处理，必须回退 payload。
"""

from __future__ import annotations

from types import SimpleNamespace

from werewolf_agent.core.event_visibility import EventVisibility
from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.evaluation.balance_public_claims import (
    public_claim_audit_keys,
    public_speech_history,
    sanitize_public_text,
)
from werewolf_agent.runtime.public_ledger import (
    build_public_claim_text_ledger,
    build_public_ledger,
)


def test_public_ledger_extracts_role_claims_check_claims_and_badge_flow() -> None:
    gs = GameState(events=[
        GameEvent(type="speech", payload={
            "speaker": "p03",
            "day_number": 1,
            "text": "我是预言家，昨晚验p08查杀，警徽流p05 p07",
        }),
    ])

    ledger = build_public_ledger(gs)

    assert ledger["role_claims"] == [
        {"day": 1, "speaker": "p03", "role": "seer", "source_event": "speech"}
    ]
    assert ledger["seer_check_claims"] == [
        {"day": 1, "speaker": "p03", "target": "p08", "result": "wolf", "source_event": "speech"}
    ]
    assert ledger["badge_flow_claims"] == [
        {"day": 1, "speaker": "p03", "targets": ["p05", "p07"], "source_event": "speech"}
    ]


def test_public_ledger_ignores_private_and_moderator_only_events() -> None:
    gs = GameState(events=[
        GameEvent(type="wolf_discussion", payload={
            "wolf_id": "p01",
            "text": "刀p08",
            "visibility": "werewolf_team_only",
        }),
        GameEvent(type="action_trace_audit", payload={
            "player_id": "p02",
            "visibility": "moderator_only",
            "action_trace": {"parsed_action": {"true_role": "seer"}},
        }),
    ])

    ledger = build_public_ledger(gs)

    assert ledger["role_claims"] == []
    assert ledger["seer_check_claims"] == []
    assert ledger["vote_records"] == []


def test_public_ledger_extracts_vote_records_and_last_words() -> None:
    gs = GameState(events=[
        GameEvent(type="vote_resolved", payload={
            "day_number": 2,
            "exiled": "p08",
            "reason": "highest_votes",
            "votes": [
                {"voter": "p01", "target": "p08", "reason": "跟预言家查杀"},
            ],
        }),
        GameEvent(type="exile_last_words", payload={
            "speaker": "p08",
            "day_number": 2,
            "text": "我不是狼，重点看p03。",
        }),
    ])

    ledger = build_public_ledger(gs)

    assert ledger["vote_records"] == [
        {"day": 2, "voter": "p01", "target": "p08", "reason": "跟预言家查杀", "source_event": "vote_resolved"}
    ]
    assert ledger["last_words"] == [
        {"day": 2, "speaker": "p08", "text": "我不是狼，重点看p03。", "source_event": "exile_last_words"}
    ]


def test_public_ledger_extracts_night_death_last_words_player_claim() -> None:
    gs = GameState(game_id="g42_action_claim", events=[GameEvent(
        type="night_death_last_words",
        payload={
            "speaker": "p07",
            "day_number": 1,
            "text": "我是猎人，现在开枪带走p01。",
        },
        visibility="public",
    )])

    ledger = build_public_ledger(gs)

    assert len(ledger["action_claims"]) == 1
    assert ledger["action_claims"][0]["authority"] == "player_claim"
    assert ledger["action_claims"][0]["target"] == "p01"
    assert ledger["confirmed_actions"] == []


def test_public_ledger_does_not_expose_real_seer_check() -> None:
    gs = GameState(events=[
        GameEvent(type="seer_check", payload={
            "seer_id": "p03",
            "target_id": "p08",
            "alignment": "werewolf",
            "visibility": "seer_private",
        }),
    ])

    ledger = build_public_ledger(gs)

    assert ledger["seer_check_claims"] == []


def test_public_ledger_excludes_wolf_plan_and_action_trace_private_role() -> None:
    gs = GameState(events=[
        GameEvent(type="wolf_team_plan", payload={
            "night_kill_primary": "p08",
            "visibility": "werewolf_team_only",
        }),
        GameEvent(type="wolf_discussion", payload={
            "wolf_id": "p01",
            "text": "刀p08",
            "visibility": "werewolf_team_only",
        }),
        GameEvent(type="action_trace_audit", payload={
            "player_id": "p02",
            "visibility": "moderator_only",
            "action_trace": {"parsed_action": {"private_intent": {"true_role": "seer"}}},
        }),
    ])

    ledger = build_public_ledger(gs)

    assert "p08" not in str(ledger)
    assert "true_role" not in str(ledger)
    assert "private_intent" not in str(ledger)


def test_public_claim_text_ledger_is_complete_ordered_and_public_only() -> None:
    events = [
        GameEvent(type="speech", payload={"speaker": "p01", "text": "最早公开声明"}),
        *[
            GameEvent(type="speech", payload={"speaker": "p02", "text": f"公开发言{i}"})
            for i in range(12)
        ],
        GameEvent(type="speech", payload={
            "speaker": "p03", "text": "私密声明", "visibility": "moderator_only",
        }),
    ]
    ledger = build_public_claim_text_ledger(GameState(game_id="g1", events=events))

    assert len(ledger) == 13
    assert ledger[0] == {
        "event_index": 0, "speaker": "p01", "text": "最早公开声明",
    }
    assert ledger[-1]["event_index"] == 12
    assert "私密声明" not in str(ledger)


def test_public_evidence_snapshots_match_and_exclude_private_role_claims() -> None:
    game_state = GameState(game_id="g-public-parity", events=[
        GameEvent(type="speech", payload={
            "speaker": "p05", "text": "我是预言家",
        }),
        GameEvent(
            type="speech",
            payload={
                "speaker": "p06",
                "text": "我是狼人，真实身份已确认",
                "role_truth": "werewolf",
            },
            visibility=EventVisibility.MODERATOR_ONLY,
            schema_version="2",
        ),
        GameEvent(type="sheriff_speech", payload={
            "speaker": "p07", "text": "警长竞选发言",
        }),
        GameEvent(type="sheriff_pk_speech", payload={
            "speaker": "p08", "text": "PK 发言",
        }),
        GameEvent(type="exile_last_words", payload={
            "speaker": "p09", "text": "最后遗言",
        }),
        GameEvent(type="tie_pk_speech", payload={
            "speaker": "p10", "text": "平票 PK 发言",
        }),
        GameEvent(type="night_death_last_words", payload={
            "speaker": "p11", "text": "夜亡遗言",
        }),
    ])
    claim_ledger = build_public_claim_text_ledger(game_state)
    history = public_speech_history(game_state.events)
    candidate = "p05自认预言家，p06自认狼人。"

    assert [(item["speaker"], item["text"]) for item in claim_ledger] == history
    assert history == [
        ("p05", "我是预言家"),
        ("p07", "警长竞选发言"),
        ("p08", "PK 发言"),
        ("p09", "最后遗言"),
        ("p10", "平票 PK 发言"),
        ("p11", "夜亡遗言"),
    ]
    assert "p06" not in str(claim_ledger)
    assert "狼人" not in str(claim_ledger)

    ledger_speeches = [
        (item["speaker"], item["text"])
        for item in claim_ledger
    ]
    claims, supported_claims = public_claim_audit_keys(candidate, ledger_speeches)
    assert {claim.target for claim in claims} == {"p05", "p06"}
    assert {claim.target for claim in supported_claims} == {"p05"}
    assert sanitize_public_text(candidate, ledger_speeches) == (
        "p05自认预言家，对p06的身份声明暂不采信，需继续核验。",
        1,
    )


def test_public_speech_history_respects_legacy_mapping_visibility() -> None:
    events = [
        {
            "type": "speech",
            "visibility": "public",
            "payload": {"speaker": "p01", "text": "顶层公开"},
        },
        {
            "type": "tie_pk_speech",
            "payload": {
                "speaker": "p02",
                "text": "payload 公开",
                "visibility": "public",
            },
        },
        {
            "type": "night_death_last_words",
            "visibility": "moderator_only",
            "payload": {"speaker": "p03", "text": "顶层私密"},
        },
        {
            "type": "speech",
            "payload": {
                "speaker": "p04",
                "text": "payload 私密",
                "visibility": "moderator_only",
            },
        },
        {
            "type": "speech",
            "visibility": "",
            "payload": {
                "speaker": "p08",
                "text": "顶层空字符串不能泄露",
                "visibility": "moderator_only",
            },
        },
        {
            "type": "sheriff_speech",
            "payload": {"speaker": "p05", "text": "缺失可见性默认公开"},
        },
        {
            "type": "speech",
            "visibility": None,
            "payload": {
                "speaker": "p06",
                "text": "顶层 None 不能泄露",
                "visibility": "moderator_only",
            },
        },
        {
            "type": "speech",
            "visibility": "public",
            "payload": {
                "speaker": "p07",
                "text": "顶层公开优先",
                "visibility": "moderator_only",
            },
        },
    ]

    assert public_speech_history(events) == [
        ("p01", "顶层公开"),
        ("p02", "payload 公开"),
        ("p05", "缺失可见性默认公开"),
        ("p07", "顶层公开优先"),
    ]
    typed_private = GameEvent(type="speech", payload={
        "speaker": "p06",
        "text": "typed 私密",
        "visibility": "moderator_only",
    })
    legacy_private = {
        "type": "speech",
        "visibility": None,
        "payload": {
            "speaker": "p06",
            "text": "legacy 私密",
            "visibility": "moderator_only",
        },
    }
    assert public_speech_history([typed_private]) == public_speech_history([
        legacy_private,
    ]) == []
    typed_empty_public = GameEvent(
        type="speech",
        visibility="",
        payload={"speaker": "p11", "text": "typed 空值公开"},
    )
    typed_empty_private = GameEvent(
        type="speech",
        visibility="",
        payload={
            "speaker": "p12",
            "text": "typed 空值私密",
            "visibility": "moderator_only",
        },
    )
    legacy_empty_public = {
        "type": "speech",
        "visibility": "",
        "payload": {"speaker": "p11", "text": "legacy 空值公开"},
    }
    legacy_empty_private = {
        "type": "speech",
        "visibility": "",
        "payload": {
            "speaker": "p12",
            "text": "legacy 空值私密",
            "visibility": "moderator_only",
        },
    }
    assert public_speech_history([
        typed_empty_public,
        typed_empty_private,
    ]) == [("p11", "typed 空值公开")]
    assert public_speech_history([
        legacy_empty_public,
        legacy_empty_private,
    ]) == [("p11", "legacy 空值公开")]
    event_like_public = SimpleNamespace(
        type="speech",
        payload={"speaker": "p09", "text": "属性型公开发言"},
        visibility=None,
    )
    event_like_private = SimpleNamespace(
        type="speech",
        payload={
            "speaker": "p10",
            "text": "属性型私密发言",
            "visibility": "moderator_only",
        },
        visibility=None,
    )
    assert public_speech_history([
        event_like_public,
        event_like_private,
    ]) == [("p09", "属性型公开发言")]


def test_public_ledger_uses_v2_top_level_visibility_without_payload_marker() -> None:
    state = GameState(events=[GameEvent(
        type="speech",
        payload={"speaker": "p01", "text": "我是预言家"},
        visibility=EventVisibility.MODERATOR_ONLY,
        schema_version="2",
    )])

    ledger = build_public_ledger(state)

    assert ledger["role_claims"] == []
