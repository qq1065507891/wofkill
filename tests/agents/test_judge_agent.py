"""Tests for Judge Agent broadcasts and agent integration pipeline."""

from __future__ import annotations

from werewolf_agent.agents.schemas import (
    ActionType,
    AgentContext,
    PlayerAction,
    TaskType,
)
from werewolf_agent.agents.player import PlayerAgent
from werewolf_agent.agents.judge import JudgeAgent
from werewolf_agent.persona_runtime.router import PersonaRouter
from werewolf_agent.model_gateway.router import (
    ModelRouter,
    MockProvider,
)

from tests.agents.test_player_agent import _JsonProvider

PERSONAS_YAML = "config/personas/jingcheng_style_prototypes.yaml"
MODELS_YAML = "config/models.yaml"


class TestJudgeAgent:
    def _make_judge(self) -> JudgeAgent:
        router = ModelRouter(
            model_profiles={}, llm_profiles={},
            player_assignments={},
            providers={"mock": MockProvider()},
        )
        return JudgeAgent(model_router=router)

    def test_broadcast_night_phase(self) -> None:
        judge = self._make_judge()
        b = judge.broadcast_phase("night", night_number=2)
        assert "N2 / 第二夜" in b.message
        assert b.broadcast_type == "night"

    def test_broadcast_day_phase(self) -> None:
        judge = self._make_judge()
        b = judge.broadcast_phase("day", day_number=3)
        assert "D3 / 第三天" in b.message

    def test_broadcast_death_announcement_with_deaths(self) -> None:
        judge = self._make_judge()
        b = judge.broadcast_death_announcement(
            deaths=[{"player_id": "p03", "reason": "wolf_kill"}],
            day_number=2,
        )
        assert "p03" in b.message
        assert "倒牌" in b.message

    def test_broadcast_peace_night(self) -> None:
        judge = self._make_judge()
        b = judge.broadcast_death_announcement(deaths=[], day_number=2)
        assert "平安夜" in b.message

    def test_broadcast_vote_result_exile(self) -> None:
        judge = self._make_judge()
        b = judge.broadcast_vote_result(
            {"exiled_player_id": "p05", "reason": "majority"}
        )
        assert "p05" in b.message
        assert "放逐" in b.message

    def test_broadcast_vote_result_first_tie(self) -> None:
        judge = self._make_judge()
        b = judge.broadcast_vote_result(
            {"exiled_player_id": None, "reason": "first_tie_pk"}
        )
        assert "平票" in b.message

    def test_broadcast_vote_result_second_tie(self) -> None:
        judge = self._make_judge()
        b = judge.broadcast_vote_result(
            {"exiled_player_id": None, "reason": "second_tie_no_exile"}
        )
        assert "再次平票" in b.message

    def test_broadcast_sheriff_elected(self) -> None:
        judge = self._make_judge()
        b = judge.broadcast_sheriff_result("p03", "active")
        assert "p03" in b.message
        assert "当选" in b.message

    def test_broadcast_badge_torn(self) -> None:
        judge = self._make_judge()
        b = judge.broadcast_sheriff_result(None, "torn")
        assert "撕掉" in b.message

    def test_summarize_speech_fallback(self) -> None:
        judge = self._make_judge()
        speeches = [
            {"speaker": "p01", "text": "我觉得3号是狼人。因为他的视角有问题。"},
            {"speaker": "p02", "text": "我站边1号，归票5号。"},
        ]
        summary = judge.summarize_speech(speeches)
        assert "p01" in summary

    def test_judge_does_not_adjudicate(self) -> None:
        """Judge broadcast has no authority fields."""
        judge = self._make_judge()
        b = judge.broadcast_phase("vote")
        data = b.model_dump()
        assert "winner" not in data
        assert "ruling" not in data


class TestAgentIntegration:
    def test_full_pipeline_persona_model_agent(self) -> None:
        """End-to-end: persona -> model router -> player agent -> valid output."""
        # Load configs
        persona_router = PersonaRouter.from_yaml(PERSONAS_YAML)
        persona_router.load_assignments({"p01": "logic_leader"})

        model_router = ModelRouter.from_yaml(MODELS_YAML)
        model_router.register_provider(MockProvider("minimax"))
        model_router.register_provider(MockProvider("glm"))

        # Resolve persona
        persona = persona_router.resolve("p01", "vote")
        assert persona.profile_id == "logic_leader"

        # Resolve model config
        config, _ = model_router.resolve_config("p01", "vote")
        assert config.provider != ""

        # Create agent and act
        json_resp = '{"action_type":"vote","target_id":"p07","speech":"归7","reason":"逻辑链完整","confidence":0.85}'
        # Override provider to return valid JSON
        model_router._providers[config.provider] = _JsonProvider(json_resp)

        agent = PlayerAgent(agent_id="p01", model_router=model_router)
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            phase="day",
            day_number=1,
            own_role="villager",
            legal_actions=[ActionType.VOTE, ActionType.NO_ACTION],
            legal_targets=["p07", "p08"],
            persona_snapshot=persona.effective_params,
        )
        action, retry = agent.act(ctx)

        assert isinstance(action, PlayerAction)
        assert action.action_type == ActionType.VOTE
        assert action.target_id in ["p07", "p08"]
        assert retry.attempt == 1

        # Usage logged
        usage = model_router.get_usage_log()
        assert len(usage) >= 1
