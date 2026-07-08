"""Schema validation tests for WolfTeamPlan.

Game-state validation (alive werewolf membership) is tested in
test_agent_wolf_team_plan.py — this file only covers Pydantic schema
rules (field types, length bounds, duplicates, overlap).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from werewolf_agent.agents.schemas import TaskType, WolfTeamPlan


def test_wolf_team_plan_schema_is_split_from_action_schemas_facade():
    from werewolf_agent.agents import action_schemas, wolf_team_plan_schema

    assert action_schemas.WolfTeamPlan is wolf_team_plan_schema.WolfTeamPlan


def _make_plan(**overrides):
    base = dict(
        night_number=1,
        night_kill_primary="p01",
        night_kill_backup=None,
        fake_seer="p04",
        pusher="p05",
        hooker="p08",
        deep_cover="p10",
        public_story="执行夜刀共识,白天抗推可疑预言家",
        evidence_quality="strong",
        reasoning="队友夜聊一致同意刀 p01,p04 自荐悍跳并由其余 3 狼配合",
    )
    base.update(overrides)
    return base


class TestWolfTeamPlanHappyPath:
    def test_full_valid_plan(self):
        plan = WolfTeamPlan.model_validate(_make_plan())
        assert plan.fake_seer == "p04"
        assert plan.evidence_quality == "strong"

    def test_all_roles_nullable(self):
        plan = WolfTeamPlan.model_validate(
            _make_plan(
                fake_seer=None, pusher=None, hooker=None, deep_cover=None,
            )
        )
        assert plan.fake_seer is None

    def test_night_kill_primary_null_means_no_kill(self):
        plan = WolfTeamPlan.model_validate(
            _make_plan(night_kill_primary=None, night_kill_backup=None)
        )
        assert plan.night_kill_primary is None


class TestWolfTeamPlanDuplicateRoles:
    def test_two_roles_same_wolf_rejected(self):
        with pytest.raises(ValidationError, match="必须互不重复"):
            WolfTeamPlan.model_validate(
                _make_plan(fake_seer="p04", pusher="p04")
            )

    def test_three_roles_same_wolf_rejected(self):
        with pytest.raises(ValidationError, match="必须互不重复"):
            WolfTeamPlan.model_validate(
                _make_plan(fake_seer="p04", pusher="p04", hooker="p04")
            )

    def test_none_does_not_count_as_duplicate(self):
        plan = WolfTeamPlan.model_validate(
            _make_plan(fake_seer=None, pusher=None, hooker="p08", deep_cover="p10")
        )
        assert plan.hooker == "p08"


class TestWolfTeamPlanKillRoleOverlap:
    def test_kill_target_equal_to_fake_seer_rejected(self):
        with pytest.raises(ValidationError, match="击杀目标不能是狼队角色"):
            WolfTeamPlan.model_validate(
                _make_plan(night_kill_primary="p04", fake_seer="p04")
            )

    def test_backup_kill_overlap_with_pusher_rejected(self):
        with pytest.raises(ValidationError, match="击杀目标不能是狼队角色"):
            WolfTeamPlan.model_validate(
                _make_plan(
                    night_kill_primary="p01",
                    night_kill_backup="p05",
                    pusher="p05",
                )
            )


class TestWolfTeamPlanFieldBounds:
    def test_public_story_too_long_rejected(self):
        with pytest.raises(ValidationError):
            WolfTeamPlan.model_validate(_make_plan(public_story="a" * 121))

    def test_public_story_empty_rejected(self):
        with pytest.raises(ValidationError):
            WolfTeamPlan.model_validate(_make_plan(public_story=""))

    def test_reasoning_too_long_rejected(self):
        with pytest.raises(ValidationError):
            WolfTeamPlan.model_validate(_make_plan(reasoning="b" * 201))

    def test_reasoning_empty_rejected(self):
        with pytest.raises(ValidationError):
            WolfTeamPlan.model_validate(_make_plan(reasoning=""))

    def test_night_number_zero_rejected(self):
        with pytest.raises(ValidationError):
            WolfTeamPlan.model_validate(_make_plan(night_number=0))

    def test_night_number_negative_rejected(self):
        with pytest.raises(ValidationError):
            WolfTeamPlan.model_validate(_make_plan(night_number=-1))

    def test_evidence_quality_invalid_enum_rejected(self):
        with pytest.raises(ValidationError):
            WolfTeamPlan.model_validate(_make_plan(evidence_quality="medium"))


class TestWolfTeamPlanExtraForbid:
    def test_unknown_field_rejected(self):
        bad = _make_plan()
        bad["secret_strategy"] = "ambush p06 next night"
        with pytest.raises(ValidationError, match="extra"):
            WolfTeamPlan.model_validate(bad)

    def test_typo_field_rejected(self):
        bad = _make_plan()
        bad["pussher"] = "p05"  # typo
        with pytest.raises(ValidationError, match="extra"):
            WolfTeamPlan.model_validate(bad)


class TestWolfTeamPlanTaskTypeEnum:
    def test_wolf_team_plan_task_type_exists(self):
        assert TaskType.WOLF_TEAM_PLAN.value == "wolf_team_plan"

    def test_wolf_team_plan_distinct_from_wolf_discussion(self):
        assert TaskType.WOLF_TEAM_PLAN != TaskType.WOLF_DISCUSSION


class TestWolfTeamPlanModelDump:
    def test_dict_round_trip(self):
        original = WolfTeamPlan.model_validate(_make_plan())
        dumped = original.model_dump()
        restored = WolfTeamPlan.model_validate(dumped)
        assert restored == original

    def test_dump_excludes_no_field_by_default(self):
        plan = WolfTeamPlan.model_validate(_make_plan())
        dumped = plan.model_dump()
        for key in (
            "night_number", "night_kill_primary", "night_kill_backup",
            "fake_seer", "pusher", "hooker", "deep_cover",
            "public_story", "evidence_quality", "reasoning",
        ):
            assert key in dumped
