from werewolf_agent.agents.schemas import TaskType
from werewolf_agent.core.models import Death, GameEvent, GameState, PlayerState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.agent_adapter import build_agent_context
from werewolf_agent.runtime.visible_state import (
    build_public_summary,
    build_visible_player_state,
)


def test_build_visible_player_state_contains_shared_timeline_and_public_fields() -> None:
    gs = GameState(
        game_id="visible_state",
        phase="day",
        day_number=1,
        night_number=1,
        players={
            "p01": PlayerState(id="p01", role="seer", alive=True),
            "p02": PlayerState(id="p02", role="werewolf", alive=False),
        },
        deaths=[Death(
            player_id="p02",
            reason="wolf_kill",
            timing="night",
            resolution_batch="n1",
        )],
        sheriff_id="p01",
        sheriff_badge_state="active",
    )

    state = build_visible_player_state(gs)

    assert state["phase_label"] == "D1 / 第一天"
    assert state["timeline_facts"]["previous_phase_label"] == "N1 / 首夜"
    assert state["alive_players"] == ["p01"]
    assert state["dead_players"] == [{"id": "p02", "reason": "wolf_kill"}]
    assert state["sheriff_id"] == "p01"


def test_build_public_summary_uses_same_timeline_note() -> None:
    gs = GameState(
        game_id="visible_summary",
        phase="day",
        day_number=1,
        night_number=1,
        players={"p01": PlayerState(id="p01", role="villager", alive=True)},
    )

    summary = build_public_summary(gs)

    assert "N1 首夜 -> D1 第一天" in summary
    assert "D1 / 第一天" in summary
    assert "N1 / 首夜" in summary


def test_visible_state_includes_public_ledger() -> None:
    gs = GameState(events=[
        GameEvent(type="speech", payload={
            "speaker": "p03",
            "day_number": 1,
            "text": "我是预言家，验p08查杀",
        }),
    ])

    visible = build_visible_player_state(gs)

    assert visible["public_ledger"]["role_claims"][0]["speaker"] == "p03"
    assert visible["public_ledger"]["seer_check_claims"][0]["target"] == "p08"


def test_agent_context_shares_same_public_ledger_across_roles() -> None:
    gs = GameState(
        game_id="shared_public_ledger",
        phase="day",
        day_number=1,
        players={
            "p01": PlayerState(id="p01", role="villager"),
            "p02": PlayerState(id="p02", role="werewolf"),
        },
        events=[
            GameEvent(type="speech", payload={
                "speaker": "p01",
                "day_number": 1,
                "text": "我是村民，今天重点看p02。",
            }),
            GameEvent(type="wolf_discussion", payload={
                "wolf_id": "p02",
                "text": "今晚刀p01",
                "visibility": "werewolf_team_only",
            }),
            GameEvent(type="seer_check", payload={
                "seer_id": "p03",
                "target_id": "p02",
                "alignment": "werewolf",
                "visibility": "seer_private",
            }),
        ],
    )

    engine = RuleEngine.from_yaml("config/rulesets/pre_witch_hunter_idiot_mixed.yaml")
    villager_context = build_agent_context(engine, gs, "p01", TaskType.SPEECH)
    wolf_context = build_agent_context(engine, gs, "p02", TaskType.SPEECH)

    villager_ledger = villager_context.visible_world_state["public_ledger"]
    wolf_ledger = wolf_context.visible_world_state["public_ledger"]
    assert villager_ledger == wolf_ledger
    assert villager_ledger["role_claims"] == [
        {"day": 1, "speaker": "p01", "role": "villager", "source_event": "speech"}
    ]
    assert "今晚刀p01" not in str(villager_ledger)
    assert "alignment" not in str(villager_ledger)
