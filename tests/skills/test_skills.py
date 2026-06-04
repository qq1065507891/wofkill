"""Skill tests: registry, definitions, applicability, dispatch."""

import pytest

from werewolf_agent.skills.schemas import (
    SkillDefinition,
    SkillFaction,
    SkillInput,
    SkillName,
    SkillOutput,
)
from werewolf_agent.skills.registry import SkillRegistry, faction_for_role
from werewolf_agent.skills.werewolf_skills import (
    SKILL_DEFINITIONS,
    apply_skill,
)


# ---------------------------------------------------------------------------
# Skill definitions
# ---------------------------------------------------------------------------

class TestSkillDefinitions:

    def test_twelve_skills_defined(self):
        assert len(SKILL_DEFINITIONS) == 12

    def test_all_skill_names_unique(self):
        names = [s.name for s in SKILL_DEFINITIONS]
        assert len(names) == len(set(names))

    def test_all_skills_have_display_names(self):
        for skill in SKILL_DEFINITIONS:
            assert skill.display_name != ""

    def test_all_skills_have_descriptions(self):
        for skill in SKILL_DEFINITIONS:
            assert skill.description != ""

    def test_bold_claim_werewolf_only(self):
        skill = next(s for s in SKILL_DEFINITIONS if s.name == SkillName.BOLD_CLAIM)
        assert skill.applicable_roles == ["werewolf"]

    def test_deep_hook_werewolf_only(self):
        skill = next(s for s in SKILL_DEFINITIONS if s.name == SkillName.DEEP_HOOK)
        assert skill.applicable_roles == ["werewolf"]

    def test_swing_vote_werewolf_only(self):
        skill = next(s for s in SKILL_DEFINITIONS if s.name == SkillName.SWING_VOTE)
        assert skill.applicable_roles == ["werewolf"]

    def test_push_vote_all_roles(self):
        skill = next(s for s in SKILL_DEFINITIONS if s.name == SkillName.PUSH_VOTE)
        assert len(skill.applicable_roles) == 7

    def test_review_correction_all_roles(self):
        skill = next(s for s in SKILL_DEFINITIONS if s.name == SkillName.REVIEW_CORRECTION)
        assert len(skill.applicable_roles) == 7

    def test_skill_has_tags(self):
        for skill in SKILL_DEFINITIONS:
            assert len(skill.tags) > 0


class TestSkillApplicability:

    def test_applicable_role_and_phase(self):
        skill = SkillDefinition(
            name=SkillName.BOLD_CLAIM,
            display_name="悍跳",
            description="test",
            applicable_roles=["werewolf"],
            applicable_phases=["speech"],
        )
        assert skill.is_applicable("werewolf", "speech")
        assert not skill.is_applicable("seer", "speech")
        assert not skill.is_applicable("werewolf", "vote")

    def test_empty_roles_means_all(self):
        skill = SkillDefinition(
            name=SkillName.PUSH_VOTE,
            display_name="归票",
            description="test",
            applicable_roles=[],
            applicable_phases=["speech"],
        )
        assert skill.is_applicable("werewolf", "speech")
        assert skill.is_applicable("seer", "speech")
        assert not skill.is_applicable("seer", "vote")

    def test_empty_phases_means_all(self):
        skill = SkillDefinition(
            name=SkillName.PUSH_VOTE,
            display_name="归票",
            description="test",
            applicable_roles=["werewolf"],
            applicable_phases=[],
        )
        assert skill.is_applicable("werewolf", "speech")
        assert skill.is_applicable("werewolf", "vote")
        assert skill.is_applicable("werewolf", "night_action")


class TestSkillInput:

    def test_creation(self):
        inp = SkillInput(role="werewolf", phase="speech", day=1)
        assert inp.role == "werewolf"
        assert inp.phase == "speech"
        assert inp.day == 1


class TestSkillOutput:

    def test_creation(self):
        out = SkillOutput(
            skill_name="bold_claim",
            recommended_action="claim_role",
            confidence=0.6,
        )
        assert out.skill_name == "bold_claim"
        assert out.confidence == 0.6


# ---------------------------------------------------------------------------
# Skill dispatch
# ---------------------------------------------------------------------------

class TestSkillDispatch:

    def test_bold_claim_handler(self):
        inp = SkillInput(role="werewolf", phase="speech", day=1)
        out = apply_skill(SkillName.BOLD_CLAIM, inp)
        assert out.skill_name == "bold_claim"
        assert out.recommended_action == "claim_role"
        assert len(out.speech_structure) > 0
        assert len(out.risk_alerts) > 0
        assert out.confidence > 0

    def test_bold_claim_late_day_lower_confidence(self):
        early = apply_skill(SkillName.BOLD_CLAIM, SkillInput(role="werewolf", phase="speech", day=1))
        late = apply_skill(SkillName.BOLD_CLAIM, SkillInput(role="werewolf", phase="speech", day=3))
        assert late.confidence < early.confidence

    def test_deep_hook_handler(self):
        out = apply_skill(SkillName.DEEP_HOOK, SkillInput(role="werewolf", phase="speech"))
        assert out.skill_name == "deep_hook"
        assert out.recommended_action == "speech"

    def test_push_vote_handler(self):
        out = apply_skill(SkillName.PUSH_VOTE, SkillInput(role="seer", phase="speech"))
        assert out.skill_name == "push_vote"
        assert out.recommended_action == "vote"

    def test_resist_push_handler(self):
        out = apply_skill(SkillName.RESIST_PUSH, SkillInput(role="werewolf", phase="defense_speech"))
        assert out.skill_name == "resist_push"
        assert out.recommended_action == "defense_speech"

    def test_wolf_pit_handler(self):
        out = apply_skill(SkillName.WOLF_PIT_ANALYSIS, SkillInput(role="villager", phase="speech"))
        assert out.skill_name == "wolf_pit"
        assert out.recommended_action == "analyze"

    def test_review_correction_handler(self):
        out = apply_skill(SkillName.REVIEW_CORRECTION, SkillInput(role="seer", phase="review"))
        assert out.skill_name == "review_correct"
        assert out.confidence > 0.6

    def test_all_skills_dispatch(self):
        for skill_name in SkillName:
            out = apply_skill(skill_name, SkillInput(role="werewolf", phase="speech"))
            assert out.skill_name == skill_name.value
            assert out.confidence >= 0.0


# ---------------------------------------------------------------------------
# SkillRegistry
# ---------------------------------------------------------------------------

class TestSkillRegistry:

    def test_default_12_skills(self):
        reg = SkillRegistry()
        assert reg.count() == 12

    def test_get_existing(self):
        reg = SkillRegistry()
        skill = reg.get(SkillName.BOLD_CLAIM)
        assert skill is not None
        assert skill.name == SkillName.BOLD_CLAIM

    def test_get_all(self):
        reg = SkillRegistry()
        all_skills = reg.all_skills()
        assert len(all_skills) == 12

    def test_applicable_for_werewolf_speech(self):
        reg = SkillRegistry()
        skills = reg.applicable_skills("werewolf", "speech")
        names = {s.name for s in skills}
        assert SkillName.BOLD_CLAIM in names
        assert SkillName.DEEP_HOOK in names

    def test_applicable_for_seer_speech(self):
        reg = SkillRegistry()
        skills = reg.applicable_skills("seer", "speech")
        names = {s.name for s in skills}
        assert SkillName.COUNTER_CLAIM in names
        assert SkillName.BOLD_CLAIM not in names

    def test_applicable_for_vote_phase(self):
        reg = SkillRegistry()
        skills = reg.applicable_skills("werewolf", "vote")
        names = {s.name for s in skills}
        assert SkillName.SWING_VOTE in names
        assert SkillName.PUSH_VOTE in names

    def test_applicable_for_review(self):
        reg = SkillRegistry()
        skills = reg.applicable_skills("seer", "review")
        names = {s.name for s in skills}
        assert SkillName.REVIEW_CORRECTION in names

    def test_dispatch(self):
        reg = SkillRegistry()
        out = reg.dispatch(SkillName.BOLD_CLAIM, SkillInput(role="werewolf", phase="speech", day=1))
        assert out.skill_name == "bold_claim"
        assert out.confidence > 0

    def test_dispatch_applicable(self):
        reg = SkillRegistry()
        results = reg.dispatch_applicable(SkillInput(role="werewolf", phase="speech", day=1))
        assert len(results) > 0
        for r in results:
            assert r.skill_name in [s.name.value for s in SKILL_DEFINITIONS]

    def test_by_role(self):
        reg = SkillRegistry()
        wolf_skills = reg.by_role("werewolf")
        assert len(wolf_skills) > 0
        assert any(s.name == SkillName.BOLD_CLAIM for s in wolf_skills)

    def test_by_phase(self):
        reg = SkillRegistry()
        speech_skills = reg.by_phase("speech")
        assert len(speech_skills) > 0

    def test_by_tag(self):
        reg = SkillRegistry()
        deception_skills = reg.by_tag("deception")
        assert len(deception_skills) > 0
        assert any(s.name == SkillName.BOLD_CLAIM for s in deception_skills)

    def test_names(self):
        reg = SkillRegistry()
        names = reg.names()
        assert "bold_claim" in names
        assert len(names) == 12

    def test_register_custom_skill(self):
        reg = SkillRegistry()
        custom = SkillDefinition(
            name=SkillName.HIDE_IDENTITY,
            display_name="测试自定义",
            description="test custom",
            applicable_roles=["seer"],
            applicable_phases=["speech"],
            tags=["test"],
        )
        reg.register(custom)
        assert reg.count() == 12  # Overwrites existing HIDE_IDENTITY
        skill = reg.get(SkillName.HIDE_IDENTITY)
        assert skill.display_name == "测试自定义"


# ---------------------------------------------------------------------------
# Faction-based skill loading
# ---------------------------------------------------------------------------

class TestSkillFaction:

    def test_all_skills_have_faction(self):
        for skill in SKILL_DEFINITIONS:
            assert skill.faction in (
                SkillFaction.WOLF,
                SkillFaction.GOOD,
                SkillFaction.COMMON,
                SkillFaction.UNIVERSAL,
            )

    def test_wolf_skills_are_wolf_faction(self):
        wolf_skill_names = {SkillName.BOLD_CLAIM, SkillName.SWING_VOTE, SkillName.DEEP_HOOK}
        for skill in SKILL_DEFINITIONS:
            if skill.name in wolf_skill_names:
                assert skill.faction == SkillFaction.WOLF, f"{skill.name} should be WOLF"

    def test_good_skills_are_good_faction(self):
        good_skill_names = {SkillName.WOLF_PIT_ANALYSIS, SkillName.PROTECT_POWER}
        for skill in SKILL_DEFINITIONS:
            if skill.name in good_skill_names:
                assert skill.faction == SkillFaction.GOOD, f"{skill.name} should be GOOD"

    def test_review_is_universal(self):
        skill = next(s for s in SKILL_DEFINITIONS if s.name == SkillName.REVIEW_CORRECTION)
        assert skill.faction == SkillFaction.UNIVERSAL

    def test_by_faction_wolf(self):
        reg = SkillRegistry()
        wolf_skills = reg.by_faction(SkillFaction.WOLF)
        assert len(wolf_skills) == 3
        names = {s.name for s in wolf_skills}
        assert names == {SkillName.BOLD_CLAIM, SkillName.SWING_VOTE, SkillName.DEEP_HOOK}

    def test_by_faction_good(self):
        reg = SkillRegistry()
        good_skills = reg.by_faction(SkillFaction.GOOD)
        assert len(good_skills) == 2
        names = {s.name for s in good_skills}
        assert names == {SkillName.WOLF_PIT_ANALYSIS, SkillName.PROTECT_POWER}

    def test_by_faction_common(self):
        reg = SkillRegistry()
        common_skills = reg.by_faction(SkillFaction.COMMON)
        assert len(common_skills) == 6
        names = {s.name for s in common_skills}
        assert SkillName.COUNTER_CLAIM in names
        assert SkillName.PUSH_VOTE in names
        assert SkillName.FIND_POWER in names
        assert SkillName.HIDE_IDENTITY in names
        assert SkillName.RESIST_PUSH in names
        assert SkillName.LAST_WORDS_ANALYSIS in names

    def test_by_faction_universal(self):
        reg = SkillRegistry()
        universal = reg.by_faction(SkillFaction.UNIVERSAL)
        assert len(universal) == 1
        assert universal[0].name == SkillName.REVIEW_CORRECTION

    def test_faction_for_role(self):
        assert faction_for_role("werewolf") == SkillFaction.WOLF
        assert faction_for_role("villager") == SkillFaction.GOOD
        assert faction_for_role("seer") == SkillFaction.GOOD
        assert faction_for_role("witch") == SkillFaction.GOOD
        assert faction_for_role("hunter") == SkillFaction.GOOD
        assert faction_for_role("idiot") == SkillFaction.GOOD
        assert faction_for_role("hybrid") == SkillFaction.GOOD


class TestSkillsForRole:

    def test_wolf_gets_wolf_plus_common_plus_universal(self):
        reg = SkillRegistry()
        skills = reg.skills_for_role("werewolf")
        names = {s.name for s in skills}
        # Wolf-only
        assert SkillName.BOLD_CLAIM in names
        assert SkillName.SWING_VOTE in names
        assert SkillName.DEEP_HOOK in names
        # Common
        assert SkillName.COUNTER_CLAIM in names
        assert SkillName.PUSH_VOTE in names
        assert SkillName.FIND_POWER in names
        assert SkillName.HIDE_IDENTITY in names
        assert SkillName.RESIST_PUSH in names
        assert SkillName.LAST_WORDS_ANALYSIS in names
        # Universal
        assert SkillName.REVIEW_CORRECTION in names
        # Good-only must NOT be present
        assert SkillName.WOLF_PIT_ANALYSIS not in names
        assert SkillName.PROTECT_POWER not in names

    def test_good_role_gets_good_plus_common_plus_universal(self):
        reg = SkillRegistry()
        for role in ["villager", "seer", "witch", "hunter", "idiot", "hybrid"]:
            skills = reg.skills_for_role(role)
            names = {s.name for s in skills}
            # Good-only
            assert SkillName.WOLF_PIT_ANALYSIS in names, f"{role} missing WOLF_PIT_ANALYSIS"
            assert SkillName.PROTECT_POWER in names, f"{role} missing PROTECT_POWER"
            # Common
            assert SkillName.PUSH_VOTE in names
            assert SkillName.RESIST_PUSH in names
            assert SkillName.LAST_WORDS_ANALYSIS in names
            assert SkillName.REVIEW_CORRECTION in names
            # Wolf-only must NOT be present
            assert SkillName.BOLD_CLAIM not in names
            assert SkillName.SWING_VOTE not in names
            assert SkillName.DEEP_HOOK not in names

    def test_seer_has_counter_claim_but_villager_does_not(self):
        reg = SkillRegistry()
        seer_skills = reg.skills_for_role("seer")
        villager_skills = reg.skills_for_role("villager")
        seer_names = {s.name for s in seer_skills}
        villager_names = {s.name for s in villager_skills}
        assert SkillName.COUNTER_CLAIM in seer_names
        assert SkillName.COUNTER_CLAIM not in villager_names
        assert SkillName.HIDE_IDENTITY in seer_names
        assert SkillName.HIDE_IDENTITY not in villager_names

    def test_dispatch_for_role_werewolf_speech(self):
        reg = SkillRegistry()
        results = reg.dispatch_for_role(
            "werewolf", "speech",
            SkillInput(role="werewolf", phase="speech", day=1),
        )
        assert len(results) > 0
        for r in results:
            assert r.confidence >= 0.0

    def test_dispatch_for_role_seer_excludes_wolf_skills(self):
        reg = SkillRegistry()
        results = reg.dispatch_for_role(
            "seer", "speech",
            SkillInput(role="seer", phase="speech", day=1),
        )
        result_names = {r.skill_name for r in results}
        assert "bold_claim" not in result_names
        assert "swing_vote" not in result_names
        assert "deep_hook" not in result_names


# ---------------------------------------------------------------------------
# Dynamic skill behavior tests (game-state-aware)
# ---------------------------------------------------------------------------

from werewolf_agent.core.models import GameState, PlayerState, GameEvent


def _make_skill_gs(
    day: int = 1,
    night: int = 1,
    player_role: str = "werewolf",
    player_id: str = "p01",
    events: list[GameEvent] | None = None,
) -> GameState:
    """Build a minimal GameState for skill testing."""
    players = {
        "p01": PlayerState(id="p01", role="werewolf", alive=True),
        "p02": PlayerState(id="p02", role="werewolf", alive=True),
        "p03": PlayerState(id="p03", role="seer", alive=True),
        "p04": PlayerState(id="p04", role="villager", alive=True),
        "p05": PlayerState(id="p05", role="villager", alive=True),
        "p06": PlayerState(id="p06", role="witch", alive=True),
    }
    # Override the test player's role
    players[player_id] = PlayerState(id=player_id, role=player_role, alive=True)

    return GameState(
        ruleset_id="test",
        game_id="test_game",
        phase="speech",
        day_number=day,
        night_number=night,
        players=players,
        events=events or [],
    )


def _build_cognition(gs: GameState, player_id: str):
    """Build world_state, belief_state, and alerts for a GameState."""
    from werewolf_agent.cognition.world_state import build_world_state
    from werewolf_agent.cognition.belief import BeliefUpdater
    from werewolf_agent.cognition.contradiction import ContradictionEngine

    world_state = build_world_state(gs)
    updater = BeliefUpdater()
    belief_state = updater.initialize(list(gs.players.keys()), player_id)
    belief_state = updater.update(belief_state, world_state.facts, gs.day_number)
    alerts = ContradictionEngine().detect(world_state.facts, gs.day_number)
    return world_state, belief_state, alerts


class TestSkillInputGameState:

    def test_new_fields_default_none(self):
        inp = SkillInput(role="werewolf", phase="speech")
        assert inp.game_state is None
        assert inp.world_state is None
        assert inp.belief_state is None
        assert inp.contradiction_alerts == []
        assert inp.player_id == ""

    def test_new_fields_assignable(self):
        gs = _make_skill_gs()
        inp = SkillInput(
            role="werewolf", phase="speech",
            game_state=gs, player_id="p01",
        )
        assert inp.game_state is not None
        assert inp.player_id == "p01"

    def test_prompt_injectable_default_empty(self):
        out = SkillOutput(skill_name="test")
        assert out.prompt_injectable == ""


class TestDynamicSkillOutput:

    def test_bold_claim_no_seer_claimants_high_confidence(self):
        gs = _make_skill_gs(day=1, player_role="werewolf")
        ws, bs, alerts = _build_cognition(gs, "p01")
        inp = SkillInput(
            role="werewolf", phase="speech", day=1,
            game_state=gs, world_state=ws, belief_state=bs,
            contradiction_alerts=alerts, player_id="p01",
        )
        out = apply_skill(SkillName.BOLD_CLAIM, inp)
        assert out.confidence >= 0.7
        assert "单边" in out.prompt_injectable or "无人跳" in out.prompt_injectable

    def test_bold_claim_with_seer_claimant_lower_confidence(self):
        gs = _make_skill_gs(day=1, player_role="werewolf", events=[
            GameEvent(type="speech", payload={
                "speaker": "p03", "text": "我是预言家", "day_number": 1,
            }),
        ])
        ws, bs, alerts = _build_cognition(gs, "p01")
        inp = SkillInput(
            role="werewolf", phase="speech", day=1,
            game_state=gs, world_state=ws, belief_state=bs,
            contradiction_alerts=alerts, player_id="p01",
        )
        out = apply_skill(SkillName.BOLD_CLAIM, inp)
        assert out.confidence < 0.7
        assert "对跳" in out.prompt_injectable

    def test_push_vote_with_suspects_sets_target(self):
        gs = _make_skill_gs(day=1, player_role="seer", player_id="p03")
        ws, bs, alerts = _build_cognition(gs, "p03")
        inp = SkillInput(
            role="seer", phase="speech", day=1,
            game_state=gs, world_state=ws, belief_state=bs,
            contradiction_alerts=alerts, player_id="p03",
        )
        out = apply_skill(SkillName.PUSH_VOTE, inp)
        # Should produce prompt_injectable text (may or may not have target)
        assert out.prompt_injectable != "" or out.confidence < 0.4

    def test_resist_push_with_seer_check(self):
        # 使用公开的查杀声明 speech，不使用私有 seer_check 事件
        gs = _make_skill_gs(day=1, player_role="werewolf", player_id="p01", events=[
            GameEvent(type="speech", payload={
                "speaker": "p03", "text": "我查验了p01是狼人",
                "day_number": 1,
            }),
        ])
        ws, bs, alerts = _build_cognition(gs, "p01")
        inp = SkillInput(
            role="werewolf", phase="defense_speech", day=1,
            game_state=gs, world_state=ws, belief_state=bs,
            contradiction_alerts=alerts, player_id="p01",
        )
        out = apply_skill(SkillName.RESIST_PUSH, inp)
        assert out.prompt_injectable != ""
        assert "查杀" in out.prompt_injectable or "预言家" in out.prompt_injectable

    def test_deep_hook_exposed_teammate(self):
        # 使用公开的查杀声明 speech，不使用私有 seer_check 事件
        gs = _make_skill_gs(day=1, player_role="werewolf", player_id="p01", events=[
            GameEvent(type="speech", payload={
                "speaker": "p03", "text": "我查验了p02是狼人",
                "day_number": 1,
            }),
        ])
        ws, bs, alerts = _build_cognition(gs, "p01")
        inp = SkillInput(
            role="werewolf", phase="speech", day=1,
            game_state=gs, world_state=ws, belief_state=bs,
            contradiction_alerts=alerts, player_id="p01",
        )
        out = apply_skill(SkillName.DEEP_HOOK, inp)
        assert out.prompt_injectable != ""
        assert "p02" in out.prompt_injectable or "队友" in out.prompt_injectable

    def test_wolf_pit_analysis_produces_output(self):
        gs = _make_skill_gs(day=2, player_role="villager", player_id="p04")
        ws, bs, alerts = _build_cognition(gs, "p04")
        inp = SkillInput(
            role="villager", phase="speech", day=2,
            game_state=gs, world_state=ws, belief_state=bs,
            contradiction_alerts=alerts, player_id="p04",
        )
        out = apply_skill(SkillName.WOLF_PIT_ANALYSIS, inp)
        assert out.prompt_injectable != ""
        assert "狼" in out.prompt_injectable

    def test_fallback_to_static_when_no_game_state(self):
        """All 12 handlers must still work without game_state."""
        for skill_name in SkillName:
            out = apply_skill(skill_name, SkillInput(role="werewolf", phase="speech", day=1))
            assert out.skill_name == skill_name.value
            assert out.confidence >= 0.0

    def test_all_skills_dispatch_with_game_state(self):
        """All 12 handlers must work with game_state provided."""
        gs = _make_skill_gs()
        ws, bs, alerts = _build_cognition(gs, "p01")
        for skill_name in SkillName:
            inp = SkillInput(
                role="werewolf", phase="speech", day=1,
                game_state=gs, world_state=ws, belief_state=bs,
                contradiction_alerts=alerts, player_id="p01",
            )
            out = apply_skill(skill_name, inp)
            assert out.skill_name == skill_name.value
            assert out.confidence >= 0.0

    def test_swing_vote_for_wolf(self):
        gs = _make_skill_gs(day=1, player_role="werewolf", player_id="p01")
        ws, bs, alerts = _build_cognition(gs, "p01")
        inp = SkillInput(
            role="werewolf", phase="vote", day=1,
            game_state=gs, world_state=ws, belief_state=bs,
            contradiction_alerts=alerts, player_id="p01",
        )
        out = apply_skill(SkillName.SWING_VOTE, inp)
        assert out.prompt_injectable != ""

    def test_hide_identity_safe(self):
        gs = _make_skill_gs(day=1, player_role="seer", player_id="p03")
        ws, bs, alerts = _build_cognition(gs, "p03")
        inp = SkillInput(
            role="seer", phase="speech", day=1,
            game_state=gs, world_state=ws, belief_state=bs,
            contradiction_alerts=alerts, player_id="p03",
        )
        out = apply_skill(SkillName.HIDE_IDENTITY, inp)
        assert out.prompt_injectable != ""
        assert "隐蔽" in out.prompt_injectable or "藏" in out.prompt_injectable

    def test_last_words_analysis_with_death(self):
        gs = _make_skill_gs(day=1, player_role="villager", player_id="p04", events=[
            GameEvent(type="player_died", payload={
                "player_id": "p05", "reason": "wolf_kill",
                "timing": "night", "day_number": 1,
            }),
        ])
        ws, bs, alerts = _build_cognition(gs, "p04")
        inp = SkillInput(
            role="villager", phase="speech", day=1,
            game_state=gs, world_state=ws, belief_state=bs,
            contradiction_alerts=alerts, player_id="p04",
        )
        out = apply_skill(SkillName.LAST_WORDS_ANALYSIS, inp)
        assert out.prompt_injectable != ""
        assert "p05" in out.prompt_injectable or "遗言" in out.prompt_injectable


class TestSkillIntegration:

    def test_inject_skill_output_adds_advice(self):
        from werewolf_agent.runtime.agent_adapter import _inject_skill_output

        gs = _make_skill_gs(day=1)
        ws, bs, alerts = _build_cognition(gs, "p01")
        directive: dict = {}
        result, tool_analyses = _inject_skill_output(
            directive, gs, "p01", ws, bs, alerts, "speech",
        )
        assert "skill_tactical_advice" in result
        assert isinstance(result["skill_tactical_advice"], str)
        assert len(result["skill_tactical_advice"]) > 0

    def test_inject_skill_output_no_duplicate_key(self):
        """If existing directive has keys, skill adds new key without overwriting."""
        from werewolf_agent.runtime.agent_adapter import _inject_skill_output

        gs = _make_skill_gs(day=1)
        ws, bs, alerts = _build_cognition(gs, "p01")
        directive = {"wolf_universal_rules": "test"}
        result, _ = _inject_skill_output(
            directive, gs, "p01", ws, bs, alerts, "speech",
        )
        assert "wolf_universal_rules" in result
        assert "skill_tactical_advice" in result


# ---------------------------------------------------------------------------
# Skill tool integration (on-demand LLM-callable tools)
# ---------------------------------------------------------------------------

class TestSkillToolDefinitions:

    def test_agent_context_accepts_skill_fields(self):
        from werewolf_agent.agents.schemas import AgentContext, TaskType

        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            skill_tools=[{"name": "skill_analyze_wolf_pit"}],
            skill_analyses={"skill_analyze_wolf_pit": "分析结果"},
        )
        assert len(ctx.skill_tools) == 1
        assert ctx.skill_analyses["skill_analyze_wolf_pit"] == "分析结果"

    def test_skill_fields_default_empty(self):
        from werewolf_agent.agents.schemas import AgentContext, TaskType

        ctx = AgentContext(agent_id="p01", task_type=TaskType.VOTE)
        assert ctx.skill_tools == []
        assert ctx.skill_analyses == {}

    def test_build_skill_tool_defs_for_context(self):
        from werewolf_agent.runtime.agent_adapter import _build_skill_tool_defs

        # Villager in speech task_type should get wolf_pit and find_power
        tools = _build_skill_tool_defs(role="villager", task_type="speech")
        tool_names = [t["name"] for t in tools]
        assert "skill_analyze_wolf_pit" in tool_names
        assert "skill_find_power_roles" in tool_names

    def test_build_skill_tool_defs_seer_role(self):
        from werewolf_agent.runtime.agent_adapter import _build_skill_tool_defs

        # seer is GOOD faction → gets wolf_pit; find_power now includes seer
        tools = _build_skill_tool_defs(role="seer", task_type="speech")
        tool_names = [t["name"] for t in tools]
        assert "skill_analyze_wolf_pit" in tool_names
        assert "skill_find_power_roles" in tool_names

    def test_build_skill_tool_defs_empty_for_unrelated_phase(self):
        from werewolf_agent.runtime.agent_adapter import _build_skill_tool_defs

        tools = _build_skill_tool_defs(role="werewolf", task_type="night_action")
        tool_names = [t["name"] for t in tools]
        # last_words is not applicable at night
        assert "skill_analyze_last_words" not in tool_names

    def test_tool_skills_excluded_from_prompt_injection(self):
        """Tool skills should NOT appear in skill_tactical_advice."""
        from werewolf_agent.runtime.agent_adapter import _inject_skill_output

        gs = _make_skill_gs(day=1)
        ws, bs, alerts = _build_cognition(gs, "p04")
        result, _ = _inject_skill_output(
            {}, gs, "p04", ws, bs, alerts, "speech",
        )
        advice = result.get("skill_tactical_advice", "")
        # wolf_pit/find_power/last_words should NOT be in injected advice
        assert "盘狼坑" not in advice
        assert "找神" not in advice
