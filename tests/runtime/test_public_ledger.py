from __future__ import annotations

from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.runtime.public_ledger import build_public_ledger


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
    from werewolf_agent.runtime.public_ledger import build_public_claim_text_ledger

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
