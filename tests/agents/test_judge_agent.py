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


class TestPersonaSystemMessageInjection:
    """J-7: persona should be passed as a separate system_prompt, not concatenated to user prompt."""

    def _make_judge_with_capture(self) -> tuple[JudgeAgent, list[dict[str, object]]]:
        captured: list[dict[str, object]] = []

        class _CaptureProvider:
            name = "capture"

            def generate(self, prompt, config, system_prompt=None, tools=None, tool_choice=None):  # type: ignore[no-untyped-def]
                captured.append({"prompt": prompt, "system_prompt": system_prompt})
                from werewolf_agent.model_gateway.router import GenerateResult, UsageRecord
                return GenerateResult(
                    text="captured",
                    provider=self.name,
                    model=config.model,
                    usage=UsageRecord(
                        agent_id="judge", task_type="speech",
                        provider=self.name, model=config.model,
                        prompt_tokens=0, completion_tokens=0, latency_ms=0,
                    ),
                )

        router = ModelRouter(
            model_profiles={"cap": {"model": "capture-model"}},
            llm_profiles={
                "judge_default": {
                    "default": {"provider": "capture", "model_profile": "cap"},
                }
            },
            player_assignments={"judge": "judge_default"},
            providers={"capture": _CaptureProvider()},
        )
        # Build a judge with a profile router that yields a known persona
        from werewolf_agent.persona_runtime.judge_router import JudgeProfileRouter
        profile_router = JudgeProfileRouter(
            profiles={
                "tournament_referee": {
                    "display_name": "锦标赛裁判",
                    "tone_variant": "tournament",
                    "base": {},
                    "task_styles": {},
                    "broadcast_patterns": {},
                    "system_prompt": "你是专业的狼人杀裁判。",
                },
            }
        )
        judge = JudgeAgent(model_router=router, profile_router=profile_router)
        return judge, captured

    def test_persona_in_system_message(self) -> None:
        """J-7: when an LLM is used, persona is passed via system_prompt, not the user prompt."""
        judge, captured = self._make_judge_with_capture()
        judge.broadcast_vote_calling(
            voter_id="p01", voter_name="玩家一", candidates=["p02", "p03"],
            position=1, total=3, day_number=1,
        )
        assert len(captured) == 1, "LLM should have been invoked once"
        rec = captured[0]
        assert rec["system_prompt"] is not None
        assert "狼人杀裁判" in str(rec["system_prompt"])
        # The user prompt must NOT carry the persona prefix
        assert "狼人杀裁判" not in str(rec["prompt"])


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
        json_resp = '{"action_type":"vote","target_id":"p07","speech":"归7","reason":"逻辑链完整","confidence":0.85,"suspect_reason":"p07发言矛盾","not_voting_reason":"p08没有证据","private_reason":"我投p07"}'
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
