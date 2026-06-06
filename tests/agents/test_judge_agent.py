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

    def test_peaceful_night_broadcast_includes_empty_deaths(self) -> None:
        """J-11: peaceful night broadcast must include public_data (empty deaths explicitly)."""
        judge = self._make_judge()
        b = judge.broadcast_death_announcement(deaths=[], day_number=2)
        # public_data must be present and explicitly indicate zero deaths
        assert b.public_data, "peaceful night broadcast must include public_data"
        # Either "deaths" key with empty list, or death_count == 0
        has_empty_deaths = (
            b.public_data.get("deaths") == []
            or b.public_data.get("death_count") == 0
        )
        assert has_empty_deaths, (
            f"peaceful night must have empty deaths marker, got {b.public_data}"
        )

    def test_broadcast_sheriff_elected(self) -> None:
        # P3-5: broadcast_sheriff_result now routes through LLM.  Use
        # router=None so the function returns the hardcoded fallback
        # (which is what the test was originally asserting).
        judge = JudgeAgent(model_router=None, profile_router=None)
        b = judge.broadcast_sheriff_result("p03", "active")
        assert "p03" in b.message
        assert "当选" in b.message

    def test_broadcast_badge_torn(self) -> None:
        # P3-5: see above — use router=None for the fallback path
        judge = JudgeAgent(model_router=None, profile_router=None)
        b = judge.broadcast_sheriff_result(None, "torn")
        assert "撕掉" in b.message

    def test_judge_does_not_adjudicate(self) -> None:
        """Judge broadcast has no authority fields."""
        judge = self._make_judge()
        b = judge.broadcast_phase("vote")
        data = b.model_dump()
        assert "winner" not in data
        assert "ruling" not in data

    def test_judge_dead_methods_removed(self) -> None:
        """J-9: summarize_speech and broadcast_vote_result are dead code; must be removed."""
        judge = self._make_judge()
        assert not hasattr(judge, "summarize_speech"), "summarize_speech is dead code"
        assert not hasattr(judge, "broadcast_vote_result"), "broadcast_vote_result is dead code"


class TestJudgeProfilePublicOnlyBoundary:
    """J-13: every judge persona profile must include a uniform
    public-only broadcasting constraint in its system_prompt.

    The judge is a non-adjudicating broadcaster: it can only relay
    public information, never reveal hidden identity, role, or
    night-time private actions. The constraint must be present in
    every profile (tournament, variety_show, neutral, mystic) so
    the LLM behaves consistently regardless of which persona the
    router picked.
    """

    PROFILES_YAML = "config/personas/judge_profiles.yaml"

    def test_judge_profile_includes_public_only_constraint(self) -> None:
        """J-13: every loaded profile has a public-only boundary clause."""
        import yaml as _yaml
        from pathlib import Path as _Path
        with open(self.PROFILES_YAML, encoding="utf-8") as f:
            data = _yaml.safe_load(f)
        profiles = data.get("judge_profiles", {})
        assert profiles, "judge_profiles.yaml must define at least one profile"
        # The boundary phrase the prompt must carry. Phrasing is
        # intentionally close to the existing profile style; we
        # only require the public-info intent, not exact wording.
        boundary_markers = ("公开信息", "public", "公开")
        for pname, prof in profiles.items():
            sp = (prof.get("system_prompt") or "").strip()
            assert sp, f"profile {pname!r} has empty system_prompt"
            has_marker = any(marker in sp for marker in boundary_markers)
            assert has_marker, (
                f"J-13: profile {pname!r} system_prompt must include a "
                f"public-only boundary (one of {boundary_markers!r}); got: {sp!r}"
            )

    def test_judge_profile_boundary_is_uniform(self) -> None:
        """J-13: the boundary clause itself must be uniform across profiles
        (same wording), so the LLM sees the same constraint regardless of
        which persona the router picked. The sentinel is the public-only
        contract core ("公开信息不得透露") — a substring that all 4
        profiles must carry verbatim.
        """
        import yaml as _yaml
        with open(self.PROFILES_YAML, encoding="utf-8") as f:
            data = _yaml.safe_load(f)
        profiles = data.get("judge_profiles", {})
        # Sentinel substring that must appear in every profile.
        # We split the constraint into two substrings to avoid the
        # middle verb "能" that would break a single contiguous
        # sentinel like "只播报公开信息".
        public_clause = "公开信息，不得透露"
        forbidden_leak = "真实身份或夜间私密信息"
        for pname, prof in profiles.items():
            sp = prof.get("system_prompt") or ""
            assert public_clause in sp, (
                f"J-13: profile {pname!r} missing public-only clause "
                f"{public_clause!r}; got: {sp!r}"
            )
            assert forbidden_leak in sp, (
                f"J-13: profile {pname!r} missing forbidden-leak clause "
                f"{forbidden_leak!r}; got: {sp!r}"
            )

    def test_judge_profile_router_returns_constrained_snapshots(self) -> None:
        """J-13: JudgeProfileRouter snapshots also carry the constraint."""
        from werewolf_agent.persona_runtime.judge_router import JudgeProfileRouter
        router = JudgeProfileRouter.from_yaml(self.PROFILES_YAML)
        public_clause = "公开信息，不得透露"
        for pname in router.list_profiles():
            snap = router.resolve(pname)
            assert public_clause in snap.system_prompt, (
                f"J-13: snapshot for {pname!r} missing boundary sentinel "
                f"{public_clause!r}; got: {snap.system_prompt!r}"
            )


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


# ---------------------------------------------------------------------------
# Phase-1 audit: judge jitter (P1-31) — task_styles in system_prompt
# reverted (P1-4) per Phase 1 self-audit
# ---------------------------------------------------------------------------


def test_judge_broadcasts_use_zero_jitter():
    """Phase-1 P1-31: all 4 judge LLM calls (vote_calling, skill_guide,
    vote_tally, exile) must request ``jitter_seconds=(0.0, 0.0)`` since
    judge calls are serial and do not contend with the 12-player burst.

    Implementation note: the router applies jitter via ``time.sleep``
    BEFORE calling the provider (router.py:417-419), and the provider
    itself does not receive ``jitter_seconds``.  We mock
    ``time.sleep`` to capture the values that the router would have
    slept.
    """
    import time as _time
    from werewolf_agent.persona_runtime.judge_router import JudgeProfileRouter
    from werewolf_agent.agents.judge import JudgeAgent

    sleep_calls: list[float] = []
    original_sleep = _time.sleep

    def _capture_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        # No actual sleep in tests.

    _time.sleep = _capture_sleep  # type: ignore[assignment]
    try:
        captured: list[dict[str, object]] = []

        class _CaptureProvider:
            name = "capture"

            def generate(self, prompt, config, system_prompt=None, tools=None,
                         tool_choice=None, **kwargs):  # type: ignore[no-untyped-def]
                captured.append({"prompt": prompt, "system_prompt": system_prompt})
                from werewolf_agent.model_gateway.router import GenerateResult, UsageRecord
                return GenerateResult(
                    text="ok",
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
        profile_router = JudgeProfileRouter(
            profiles={
                "neutral_arbiter": {
                    "display_name": "中立",
                    "tone_variant": "neutral",
                    "base": {},
                    "task_styles": {},
                    "broadcast_patterns": {},
                    "system_prompt": "你是中立的狼人杀仲裁者。",
                },
            }
        )
        judge = JudgeAgent(model_router=router, profile_router=profile_router)
        # All 4 LLM-call broadcasts in one test to amortize setup
        judge.broadcast_vote_calling(
            voter_id="p01", voter_name="玩家一", candidates=["p02"],
            position=1, total=1, day_number=1,
        )
        judge.guide_skill_use(
            role="witch", player_id="p11", player_name="玩家十一",
            available_actions=["use_antidote"],
        )
        judge.announce_vote_tally(
            tally={"p05": 5.0}, player_names={"p05": "玩家五"},
            sheriff_id=None, sheriff_weight=1.5, day_number=1,
        )
        judge.announce_exile_result(
            exiled_player_id="p05", exiled_player_name="玩家五",
            reason="", tied_player_ids=[], day_number=1,
        )
        assert len(captured) == 4, f"expected 4 judge LLM calls, got {len(captured)}"
        # The router should NOT have slept before any judge call (jitter
        # is (0,0)).  When jitter is (0, 0), router.py:418 short-circuits
        # the sleep entirely (R3-MG-3).  So sleep_calls should be empty
        # OR every sleep should be 0.0.
        nonzero_sleeps = [s for s in sleep_calls if s > 0.0]
        assert not nonzero_sleeps, (
            f"judge calls must use jitter_seconds=(0,0) → no nonzero "
            f"sleeps in router. Got sleeps={sleep_calls!r}"
        )
    finally:
        _time.sleep = original_sleep  # type: ignore[assignment]
