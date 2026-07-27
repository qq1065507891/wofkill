# -*- coding: utf-8 -*-
"""
验证玩家可见状态、公开摘要和 JSON-safe 赛后摘要。

作者: Project contributors
修改日期: 2026-07-27
"""

from werewolf_agent.agents.schemas import TaskType
from werewolf_agent.core.models import Death, GameEvent, GameState, PlayerState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.agent_adapter import build_agent_context
from werewolf_agent.runtime.visible_state import (
    build_post_game_summary,
    build_public_summary,
    build_visible_player_state,
)


def test_post_game_summary_serializes_v2_resolution_batch() -> None:
    import json

    from werewolf_agent.core.resolution_batches import ResolutionBatchV2

    gs = GameState(
        game_id="post-game-v2",
        players={"p01": PlayerState(id="p01", role="villager", alive=False)},
        deaths=[
            Death(
                "p01",
                "exile",
                "day_vote",
                ResolutionBatchV2("day", 2, "vote"),
            )
        ],
    )

    summary = build_post_game_summary(gs, "p01")

    assert json.loads(json.dumps(summary))["deaths"][0]["batch"] == {
        "phase": "day",
        "number": 2,
        "cause": "vote",
    }
    assert summary["deaths"][0]["resolution_batch_parse_failed"] is False


def test_post_game_summary_preserves_unknown_raw_with_failure_marker() -> None:
    import json

    gs = GameState(
        game_id="post-game-unknown",
        players={"p01": PlayerState(id="p01", role="villager", alive=False)},
        deaths=[Death("p01", "rule_effect", "day", "day_SECRET")],
    )

    summary = build_post_game_summary(gs, "p01")
    encoded = json.loads(json.dumps(summary))

    assert encoded["deaths"][0]["batch"] == "day_SECRET"
    assert encoded["deaths"][0]["resolution_batch_parse_failed"] is True


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
            GameEvent(type="judge_broadcast", payload={
                "phase": "death_announce",
                "day_number": 1,
                "message": "昨夜死亡: p02",
                "visibility": "public",
            }),
        ],
        deaths=[Death(
            player_id="p02",
            reason="wolf_kill",
            timing="night",
            resolution_batch="night_1",
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


def test_visible_state_hides_new_night_death_until_matching_day_announcement() -> None:
    gs = GameState(
        game_id="cross_day_death_visibility",
        phase="day",
        day_number=2,
        night_number=2,
        players={
            "p01": PlayerState(id="p01", role="villager", alive=True),
            "p02": PlayerState(id="p02", role="werewolf", alive=False),
            "p03": PlayerState(id="p03", role="seer", alive=False),
        },
        events=[GameEvent(type="judge_broadcast", payload={
            "phase": "death_announce",
            "day_number": 1,
            "message": "首夜死亡: p02",
            "visibility": "public",
        })],
        deaths=[
            Death("p02", "wolf_kill", "night", "night_1"),
            Death("p03", "wolf_kill", "night", "night_2"),
        ],
    )

    state = build_visible_player_state(gs)

    assert state["dead_players"] == [{"id": "p02", "reason": "wolf_kill"}]


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


def test_visible_state_keeps_public_ledger_shape_without_private_action_audit() -> None:
    gs = GameState(
        players={"p05": PlayerState(id="p05", role="witch")},
        events=[
            GameEvent(type="speech", payload={
                "speaker": "p05",
                "day_number": 1,
                "text": "昨晚解药救了p04。",
            }),
            GameEvent(type="witch_antidote_used", payload={
                "target_id": "p04",
                "visibility": "witch_private",
            }),
            GameEvent(type="hunter_shot_selected", payload={
                "actor_id": "p07",
                "target_id": "p01",
                "visibility": "moderator_only",
            }),
        ],
    )

    visible = build_visible_player_state(gs)
    engine = RuleEngine.from_yaml("config/rulesets/pre_witch_hunter_idiot_mixed.yaml")
    context = build_agent_context(engine, gs, "p05", TaskType.SPEECH)
    context_ledger = context.visible_world_state["public_ledger"]

    assert set(visible["public_ledger"]) == {
        "role_claims",
        "seer_check_claims",
        "badge_flow_claims",
        "vote_records",
        "last_words",
        "badge_events",
        "action_claims",
        "confirmed_actions",
        "claim_conflicts",
    }
    assert visible["public_ledger"]["confirmed_actions"] == []
    assert visible["public_ledger"]["claim_conflicts"] == []
    assert "moderator_only" not in str(visible)
    assert "status" not in str(context_ledger)
    assert "visibility" not in str(context_ledger)
    assert "witch_antidote_used" not in str(visible)
    assert "hunter_shot_selected" not in str(visible)


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


def test_agent_context_private_memory_ignores_other_players_public_speech_points() -> None:
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

    memory = context.visible_world_state.get("private_memory", {})
    assert "p03的逻辑漏洞是没有解释警徽流" not in str(memory)
    assert "p04这一点说得合理" not in str(memory)
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


def test_public_summary_marks_silent_speech() -> None:
    engine = RuleEngine.from_yaml("config/rulesets/pre_witch_hunter_idiot_mixed.yaml")
    gs = GameState(
        game_id="silent_speech_test",
        phase="day",
        day_number=1,
        players={
            "p01": PlayerState(id="p01", role="villager"),
            "p02": PlayerState(id="p02", role="werewolf"),
        },
        events=[
            GameEvent(type="speech", payload={
                "speaker": "p02",
                "day_number": 1,
                "text": "未发表有效言论",
            }),
        ],
    )

    context = build_agent_context(engine, gs, "p01", TaskType.SPEECH)
    assert "[沉默]" in context.public_summary
    assert "未发表任何有效言论" in context.public_summary
    # Should NOT extract seer check claims from silent speech
    assert "seer_check_claims" not in str(
        context.visible_world_state.get("public_ledger", {})
    ) or not context.visible_world_state.get("public_ledger", {}).get("seer_check_claims")


def test_public_summary_marks_empty_speech_as_silent() -> None:
    engine = RuleEngine.from_yaml("config/rulesets/pre_witch_hunter_idiot_mixed.yaml")
    gs = GameState(
        game_id="empty_speech_test",
        phase="day",
        day_number=1,
        players={
            "p01": PlayerState(id="p01", role="villager"),
            "p02": PlayerState(id="p02", role="werewolf"),
        },
        events=[
            GameEvent(type="speech", payload={
                "speaker": "p02",
                "day_number": 1,
                "text": "   ",
            }),
        ],
    )

    context = build_agent_context(engine, gs, "p01", TaskType.SPEECH)
    assert "[沉默]" in context.public_summary


def test_visible_state_preserves_death_reasons() -> None:
    gs = GameState(
        game_id="death_reasons_test",
        phase="day",
        day_number=2,
        night_number=1,
        players={
            "p01": PlayerState(id="p01", role="villager", alive=True),
            "p02": PlayerState(id="p02", role="werewolf", alive=False),
            "p03": PlayerState(id="p03", role="seer", alive=False),
        },
        events=[
            GameEvent(type="judge_broadcast", payload={
                "phase": "death_announce",
                "day_number": 1,
            }),
        ],
        deaths=[
            Death(player_id="p02", reason="wolf_kill", timing="night", resolution_batch="night_1"),
            Death(player_id="p03", reason="witch_poison", timing="night", resolution_batch="night_1"),
        ],
    )

    state = build_visible_player_state(gs)
    reasons = {d["id"]: d["reason"] for d in state["dead_players"]}
    assert reasons["p02"] == "wolf_kill"
    assert reasons["p03"] == "witch_poison"


def test_death_reason_labels_in_broadcast() -> None:
    from werewolf_agent.runtime.nodes.day import _death_reason_label

    assert _death_reason_label("wolf_kill") == "狼杀"
    assert _death_reason_label("witch_poison") == "毒杀"
    assert _death_reason_label("hunter_shot") == "猎人开枪"
    assert _death_reason_label("exile") == "放逐"
    assert _death_reason_label("self_destruct") == "自爆"
    assert _death_reason_label("unknown") == "原因不明"


def test_sheriff_id_hidden_after_death() -> None:
    """Design doc §visibility: when the sheriff has died but badge
    transfer has not yet executed, the public view must hide the dead
    sheriff's id. Agents must not reference a dead player as '在场'.

    Applies to BOTH build_visible_player_state and build_public_summary
    so player contexts never leak a dead sheriff's id.
    """
    gs = GameState(
        game_id="sheriff_dead_visibility",
        phase="day",
        day_number=2,
        night_number=1,
        players={
            "p01": PlayerState(id="p01", role="seer", alive=True),
            # Sheriff died (poisoned/hunter-shot); badge not yet transferred
            "p02": PlayerState(id="p02", role="villager", alive=False),
        },
        sheriff_id="p02",
        sheriff_badge_state="active",
    )

    # build_visible_player_state: sheriff_id field must be None
    visible = build_visible_player_state(gs)
    assert visible["sheriff_id"] is None, (
        f"Visible state must hide dead sheriff id; got {visible['sheriff_id']!r}"
    )
    assert visible["badge_state"] is None

    # build_public_summary: must not mention dead sheriff by id
    summary = build_public_summary(gs)
    assert "警长" not in summary, (
        f"Public summary must not mention dead sheriff; got: {summary!r}"
    )
    assert "p02" not in summary, (
        f"Public summary must not leak dead sheriff id p02; got: {summary!r}"
    )
