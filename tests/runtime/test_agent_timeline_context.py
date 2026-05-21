from werewolf_agent.agents.schemas import TaskType
from werewolf_agent.core.models import GameState, PlayerState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.agent_adapter import build_agent_context


def test_agent_context_uses_explicit_timeline_label_and_correct_night_number() -> None:
    players = {
        "p01": PlayerState(id="p01", role="villager", alive=True),
        "p02": PlayerState(id="p02", role="werewolf", alive=True),
    }
    gs = GameState(
        game_id="timeline_ctx",
        players=players,
        phase="night",
        day_number=0,
        night_number=1,
    )

    ctx = build_agent_context(RuleEngine({}), gs, "p01", TaskType.NIGHT_ACTION)

    assert ctx.night_number == 1
    assert ctx.day_number == 0
    assert ctx.visible_world_state["phase_label"] == "N1 / 首夜"
    assert ctx.visible_world_state["timeline_facts"]["current_phase_label"] == "N1 / 首夜"
    assert ctx.visible_world_state["timeline_facts"]["first_night_before_first_day"] is True
    assert "N1 首夜 -> D1 第一天" in ctx.visible_world_state["timeline_note"]
    assert "首夜发生在第一天之前" in ctx.visible_world_state["timeline_note"]


def test_agent_context_day_one_is_first_day_after_first_night() -> None:
    players = {
        "p01": PlayerState(id="p01", role="villager", alive=True),
        "p02": PlayerState(id="p02", role="werewolf", alive=True),
    }
    gs = GameState(
        game_id="timeline_day_ctx",
        players=players,
        phase="day",
        day_number=1,
        night_number=1,
    )

    ctx = build_agent_context(RuleEngine({}), gs, "p01", TaskType.SPEECH)

    assert ctx.visible_world_state["phase_label"] == "D1 / 第一天"
    assert ctx.visible_world_state["timeline_facts"]["day_one_definition"] == "D1 是首夜 N1 结算后的第一个白天"
    assert ctx.visible_world_state["timeline_facts"]["previous_phase_label"] == "N1 / 首夜"
    assert "N1 首夜 -> D1 第一天" in ctx.public_summary
    assert "第1天" not in ctx.public_summary
