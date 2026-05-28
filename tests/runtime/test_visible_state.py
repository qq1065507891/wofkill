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
        events=[
            GameEvent(type="judge_broadcast", payload={"phase": "death_announce", "message": "昨夜死亡: p02", "visibility": "public"}),
        ],
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
    assert state["dead_players"] == [{"id": "p02", "reason": "night"}]
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


def test_agent_context_includes_only_viewers_private_memory() -> None:
    engine = RuleEngine.from_yaml("config/rulesets/pre_witch_hunter_idiot_mixed.yaml")
    gs = GameState(
        game_id="private_memory",
        phase="day",
        day_number=2,
        players={
            "p01": PlayerState(id="p01", role="villager"),
            "p02": PlayerState(id="p02", role="werewolf"),
        },
        events=[
            GameEvent(type="action_trace_audit", payload={
                "player_id": "p01",
                "visibility": "moderator_only",
                "private_vote_thought": {
                    "target": "p02",
                    "standing_with_seer": "p03",
                    "suspect_reason": "p02的逻辑漏洞是没有解释自己为什么站边p03",
                    "not_voting_reason": "p04这一点说得合理，暂时不投",
                    "private_reason": "我心里更信p03，所以准备投p02",
                },
            }),
            GameEvent(type="action_trace_audit", payload={
                "player_id": "p02",
                "visibility": "moderator_only",
                "private_vote_thought": {
                    "target": "p01",
                    "suspect_reason": "p01秘密怀疑点",
                    "private_reason": "我是狼，想抗推p01",
                },
            }),
        ],
    )

    p01_context = build_agent_context(engine, gs, "p01", TaskType.SPEECH)
    p02_context = build_agent_context(engine, gs, "p02", TaskType.SPEECH)

    p01_memory = p01_context.visible_world_state["private_memory"]
    p02_memory = p02_context.visible_world_state["private_memory"]
    assert "p02的逻辑漏洞" in str(p01_memory)
    assert "p04这一点说得合理" in str(p01_memory)
    assert "p01秘密怀疑点" not in str(p01_memory)
    assert "p01秘密怀疑点" in str(p02_memory)
    assert "private_memory" not in str(p01_context.visible_world_state["public_ledger"])


def test_agent_context_private_memory_keeps_heard_public_speech_points() -> None:
    engine = RuleEngine.from_yaml("config/rulesets/pre_witch_hunter_idiot_mixed.yaml")
    gs = GameState(
        game_id="heard_public_points",
        phase="day",
        day_number=2,
        players={
            "p01": PlayerState(id="p01", role="villager"),
            "p02": PlayerState(id="p02", role="werewolf"),
        },
        events=[
            GameEvent(type="speech", payload={
                "speaker": "p02",
                "day_number": 1,
                "text": "完整原文唯一标记XYZ。p03的逻辑漏洞是没有解释警徽流。p04这一点说得合理。",
            }),
            GameEvent(type="speech", payload={
                "speaker": "p02",
                "day_number": 1,
                "visibility": "werewolf_team_only",
                "text": "秘密狼队逻辑漏洞",
            }),
        ],
    )

    context = build_agent_context(engine, gs, "p01", TaskType.SPEECH)

    memory = context.visible_world_state["private_memory"]
    assert "p03的逻辑漏洞是没有解释警徽流" in str(memory)
    assert "p04这一点说得合理" in str(memory)
    assert "完整原文唯一标记XYZ" not in context.public_summary
    assert "完整原文唯一标记XYZ" not in str(memory)
    assert "秘密狼队逻辑漏洞" not in str(memory)


def test_agent_context_uses_private_memory_not_prior_day_full_speech_text() -> None:
    engine = RuleEngine.from_yaml("config/rulesets/pre_witch_hunter_idiot_mixed.yaml")
    gs = GameState(
        game_id="prior_day_notes",
        phase="day",
        day_number=2,
        players={
            "p01": PlayerState(id="p01", role="villager"),
            "p02": PlayerState(id="p02", role="werewolf"),
        },
        events=[
            GameEvent(type="day_announce", payload={"day": 1}),
            GameEvent(type="speech", payload={
                "speaker": "p01",
                "day_number": 1,
                "text": "完整原文唯一标记ABC。p02的逻辑漏洞是没有解释自己为什么站边p03。",
            }),
            GameEvent(type="action_trace_audit", payload={
                "player_id": "p01",
                "visibility": "moderator_only",
                "private_vote_thought": {
                    "target": "p02",
                    "suspect_reason": "p02的逻辑漏洞是没有解释自己为什么站边p03",
                    "private_reason": "我记住这个漏洞，明天继续压p02",
                },
            }),
        ],
    )

    context = build_agent_context(engine, gs, "p01", TaskType.SPEECH)

    assert "完整原文唯一标记ABC" not in context.public_summary
    assert "逻辑漏洞" not in context.public_summary
    assert "逻辑漏洞" in str(context.visible_world_state["private_memory"])
    assert "p02" in str(context.visible_world_state["private_memory"])
