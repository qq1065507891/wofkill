"""Skill tests: registry, definitions, applicability, dispatch."""

import pytest

from werewolf_agent.skills.schemas import (
    SkillDefinition,
    SkillInput,
    SkillName,
    SkillOutput,
)
from werewolf_agent.skills.registry import SkillRegistry
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
