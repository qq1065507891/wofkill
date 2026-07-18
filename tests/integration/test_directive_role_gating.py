"""Strategy-directive role-gating tests (P0-I1, P0-I3).

The existing ``_assert_no_forbidden_info`` in
``tests/integration/test_e2e_info_leak.py`` only inspects
``ctx.visible_world_state``.  It does not check
``ctx.strategy_directive``, which is rendered into the LLM prompt and
could carry role-gated information (e.g. ``wolf_speech_directive`` in a
villager's directive, ``witch_night_action`` in a seer's directive).

This file adds two complementary test classes:

* ``TestDirectiveRoleGating`` (P0-I1) — for every role we build the
  full ``strategy_directive`` via the real ``agent_*`` entry points and
  assert that the role's own keys ARE present and that other roles'
  keys are NOT present.
* ``TestDirectiveWolfPrivateNoLeak`` (P0-I3) — for every non-werewolf
  role we assert that the specific wolf-private directive keys
  (``wolf_fake_seer_teammate``, ``wolf_day_push_target``,
  ``wolf_plan_target``, ``wolf_teammate_exposed``,
  ``wolf_high_priority_target``) are NOT in either
  ``strategy_directive`` or ``visible_world_state``.

Both test classes use the same ``CaptureRegistry`` / ``CaptureAgent``
pattern as ``tests/runtime/test_strategy_directives.py``.
"""

from __future__ import annotations

from dataclasses import replace

from werewolf_agent.agents.schemas import (
    ActionType, AgentContext, PlayerAction, RetryInfo, TaskType,
)
from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.runtime.graph import RuntimeState, _new_engine

RULESET_PATH = "config/rulesets/pre_witch_hunter_idiot_mixed.yaml"


# ---------------------------------------------------------------------------
# Canonical 12-player roster reused by every test in this file.
# ---------------------------------------------------------------------------

def _players() -> dict[str, PlayerState]:
    return {
        "w1": PlayerState(id="w1", role="werewolf"),
        "w2": PlayerState(id="w2", role="werewolf"),
        "w3": PlayerState(id="w3", role="werewolf"),
        "w4": PlayerState(id="w4", role="werewolf"),
        "v1": PlayerState(id="v1", role="villager"),
        "v2": PlayerState(id="v2", role="villager"),
        "v3": PlayerState(id="v3", role="villager"),
        "witch": PlayerState(id="witch", role="witch"),
        "seer": PlayerState(id="seer", role="seer"),
        "hunter": PlayerState(id="hunter", role="hunter"),
        "idiot": PlayerState(id="idiot", role="idiot"),
        "hybrid": PlayerState(id="hybrid", role="hybrid"),
    }


def _make_state(
    *,
    phase: str = "day",
    day_number: int = 1,
    night_number: int = 1,
    extra_events: list | None = None,
    hybrid_master_id: str | None = None,
    hybrid_master_faction: str | None = None,
    wolf_team_plan: dict | None = None,
) -> RuntimeState:
    """Build a ``RuntimeState`` ready for any of the ``agent_*`` callers."""
    gs_kwargs: dict = {
        "game_id": "directive_role_gating_test",
        "players": _players(),
        "phase": phase,
        "day_number": day_number,
        "night_number": night_number,
        "events": list(extra_events or []),
    }
    if hybrid_master_id:
        gs_kwargs["hybrid_master_id"] = hybrid_master_id
    if hybrid_master_faction:
        gs_kwargs["hybrid_master_faction"] = hybrid_master_faction
    gs = GameState(**gs_kwargs)
    state: RuntimeState = {
        "game_state": gs,
        "engine": _new_engine(),
        "wolf_kill_target_id": None,
        "use_antidote": False,
        "poison_target_id": None,
        "seer_target_id": None,
        "hybrid_master_target_id": None,
        "self_destruct_wolf_id": None,
        "exile_votes": {},
        "revote": False,
        "sheriff_candidates": [],
        "sheriff_votes": {},
        "sheriff_withdrawing": [],
        "badge_decision": "tear",
        "badge_target_id": None,
        "hunter_shot_target_id": None,
    }
    if wolf_team_plan is not None:
        state["wolf_team_plan"] = wolf_team_plan
    return state


def _make_capture_registry(
    speaker_id: str,
    action_type: ActionType,
    target_id: str = "v1",
    speech: str = "test",
) -> tuple:
    """Return a registry whose agent captures the last ``AgentContext``."""

    class CaptureAgent:
        last_context: AgentContext | None = None

        def act(self, context):
            self.last_context = context
            return (
                PlayerAction(
                    action_type=action_type,
                    target_id=target_id,
                    reason="test",
                    speech=speech,
                ),
                RetryInfo(),
            )

    class CaptureRegistry:
        def __init__(self):
            self.agent = CaptureAgent()

        def get_agent(self, player_id):
            return self.agent if player_id == speaker_id else None

    return CaptureRegistry(), CaptureAgent


# ---------------------------------------------------------------------------
# Role-key taxonomy.  Every key listed here is one of:
#   * a base key added to *every* role's directive (subset of PRESENCE)
#   * a role-specific key injected by ``agent_day_speech`` or a night
#     action (subset of PRESENCE)
#   * a foreign-role key that MUST NOT appear in this role's directive
#     (subset of ABSENCE)
# ---------------------------------------------------------------------------

# Keys that the role's *own* code paths are expected to inject.  A test
# fails if any of these is missing.
PRESENCE: dict[str, set[str]] = {
    "villager": {
        "anti_following_and_peace_night_rule",
        "speech_originality",
        "villager_speech_directive",
    },
    "seer": {
        "anti_following_and_peace_night_rule",
        "speech_originality",
        "seer_speech_directive",
    },
    "witch": {
        "anti_following_and_peace_night_rule",
        "speech_originality",
        "witch_speech_directive",
    },
    "hunter": {
        "anti_following_and_peace_night_rule",
        "speech_originality",
        "hunter_speech_directive",
    },
    "idiot": {
        "anti_following_and_peace_night_rule",
        "speech_originality",
        "idiot_speech_directive",
    },
    "hybrid": {
        "anti_following_and_peace_night_rule",
        "speech_originality",
        "hybrid_speech_directive",
    },
    "werewolf": {
        "anti_following_and_peace_night_rule",
        "speech_originality",
        "wolf_speech_directive",
        "wolf_universal_rules",
    },
}

# Foreign-role keys that MUST NOT appear in this role's directive.
# This is the core gap the test is closing: the rendered prompt should
# never show a villager a wolf-only directive, a seer a witch-only
# directive, etc.
ABSENCE: dict[str, set[str]] = {
    "villager": {
        # wolf private
        "wolf_speech_directive", "wolf_universal_rules",
        "wolf_fake_seer_teammate", "wolf_fake_seer_execution",
        "wolf_day_push_target", "wolf_plan_target",
        "wolf_teammate_exposed", "wolf_high_priority_target",
        "wolf_kill_instruction", "wolf_vote_strategy",
        "wolf_vote_role_hint", "wolf_vote_target",
        "wolf_team_discussion", "round_focus",
        # other role day-speech
        "seer_speech_directive", "witch_speech_directive",
        "hunter_speech_directive", "idiot_speech_directive",
        "hybrid_speech_directive", "master_behavior_summary",
        # night-action keys for other roles
        "witch_night_action", "witch_strategy_hint",
        "witch_poison_deterrent", "witch_poison_threshold",
        "seer_night_check", "check_value_assessment",
        "save_value_assessment", "first_night_killed",
        "witch_pressure", "required_evaluation",
        "hybrid_master_choice", "master_assessment",
        "hybrid_master_dead", "hybrid_vote_strategy",
        "badge_flow_plan", "excluded_counterclaiming_seers",
    },
    "seer": {
        "wolf_speech_directive", "wolf_universal_rules",
        "wolf_fake_seer_teammate", "wolf_fake_seer_execution",
        "wolf_day_push_target", "wolf_plan_target",
        "wolf_teammate_exposed", "wolf_high_priority_target",
        "wolf_kill_instruction", "wolf_vote_strategy",
        "wolf_vote_role_hint", "wolf_vote_target",
        "wolf_team_discussion", "round_focus",
        "witch_speech_directive", "hunter_speech_directive",
        "idiot_speech_directive", "hybrid_speech_directive",
        "villager_speech_directive", "master_behavior_summary",
        "witch_night_action", "witch_strategy_hint",
        "witch_poison_deterrent", "witch_poison_threshold",
        "save_value_assessment", "first_night_killed",
        "witch_pressure", "required_evaluation",
        "hybrid_master_choice", "master_assessment",
        "hybrid_master_dead", "hybrid_vote_strategy",
    },
    "witch": {
        "wolf_speech_directive", "wolf_universal_rules",
        "wolf_fake_seer_teammate", "wolf_fake_seer_execution",
        "wolf_day_push_target", "wolf_plan_target",
        "wolf_teammate_exposed", "wolf_high_priority_target",
        "wolf_kill_instruction", "wolf_vote_strategy",
        "wolf_vote_role_hint", "wolf_vote_target",
        "wolf_team_discussion", "round_focus",
        "seer_speech_directive", "hunter_speech_directive",
        "idiot_speech_directive", "hybrid_speech_directive",
        "villager_speech_directive", "master_behavior_summary",
        "seer_night_check", "check_value_assessment",
        "hybrid_master_choice", "master_assessment",
        "hybrid_master_dead", "hybrid_vote_strategy",
        "badge_flow_plan", "excluded_counterclaiming_seers",
    },
    "hunter": {
        "wolf_speech_directive", "wolf_universal_rules",
        "wolf_fake_seer_teammate", "wolf_fake_seer_execution",
        "wolf_day_push_target", "wolf_plan_target",
        "wolf_teammate_exposed", "wolf_high_priority_target",
        "wolf_kill_instruction", "wolf_vote_strategy",
        "wolf_vote_role_hint", "wolf_vote_target",
        "wolf_team_discussion", "round_focus",
        "seer_speech_directive", "witch_speech_directive",
        "idiot_speech_directive", "hybrid_speech_directive",
        "villager_speech_directive", "master_behavior_summary",
        "witch_night_action", "witch_strategy_hint",
        "witch_poison_deterrent", "witch_poison_threshold",
        "seer_night_check", "check_value_assessment",
        "save_value_assessment", "first_night_killed",
        "witch_pressure", "required_evaluation",
        "hybrid_master_choice", "master_assessment",
        "hybrid_master_dead", "hybrid_vote_strategy",
        "badge_flow_plan", "excluded_counterclaiming_seers",
    },
    "idiot": {
        "wolf_speech_directive", "wolf_universal_rules",
        "wolf_fake_seer_teammate", "wolf_fake_seer_execution",
        "wolf_day_push_target", "wolf_plan_target",
        "wolf_teammate_exposed", "wolf_high_priority_target",
        "wolf_kill_instruction", "wolf_vote_strategy",
        "wolf_vote_role_hint", "wolf_vote_target",
        "wolf_team_discussion", "round_focus",
        "seer_speech_directive", "witch_speech_directive",
        "hunter_speech_directive", "hybrid_speech_directive",
        "villager_speech_directive", "master_behavior_summary",
        "witch_night_action", "witch_strategy_hint",
        "witch_poison_deterrent", "witch_poison_threshold",
        "seer_night_check", "check_value_assessment",
        "save_value_assessment", "first_night_killed",
        "witch_pressure", "required_evaluation",
        "hybrid_master_choice", "master_assessment",
        "hybrid_master_dead", "hybrid_vote_strategy",
        "badge_flow_plan", "excluded_counterclaiming_seers",
    },
    "hybrid": {
        "wolf_speech_directive", "wolf_universal_rules",
        "wolf_fake_seer_teammate", "wolf_fake_seer_execution",
        "wolf_day_push_target", "wolf_plan_target",
        "wolf_teammate_exposed", "wolf_high_priority_target",
        "wolf_kill_instruction", "wolf_vote_strategy",
        "wolf_vote_role_hint", "wolf_vote_target",
        "wolf_team_discussion", "round_focus",
        "seer_speech_directive", "witch_speech_directive",
        "hunter_speech_directive", "idiot_speech_directive",
        "villager_speech_directive",
        "witch_night_action", "witch_strategy_hint",
        "witch_poison_deterrent", "witch_poison_threshold",
        "seer_night_check", "check_value_assessment",
        "save_value_assessment", "first_night_killed",
        "witch_pressure", "required_evaluation",
        "badge_flow_plan", "excluded_counterclaiming_seers",
    },
    "werewolf": {
        "seer_speech_directive", "witch_speech_directive",
        "hunter_speech_directive", "idiot_speech_directive",
        "hybrid_speech_directive", "villager_speech_directive",
        "master_behavior_summary",
        "witch_night_action", "witch_strategy_hint",
        "witch_poison_deterrent", "witch_poison_threshold",
        "seer_night_check", "check_value_assessment",
        "save_value_assessment", "first_night_killed",
        "witch_pressure", "required_evaluation",
        "hybrid_master_choice", "master_assessment",
        "hybrid_master_dead", "hybrid_vote_strategy",
        "badge_flow_plan", "excluded_counterclaiming_seers",
    },
}


# Map each role to the player_id used in tests.
PLAYER_ID_BY_ROLE: dict[str, str] = {
    "villager": "v1",
    "seer": "seer",
    "witch": "witch",
    "hunter": "hunter",
    "idiot": "idiot",
    "hybrid": "hybrid",
    "werewolf": "w1",
}


def _drive_day_speech(speaker_id: str) -> AgentContext:
    """Drive ``agent_day_speech`` for ``speaker_id`` and return its context."""
    from werewolf_agent.runtime.agent_adapter import agent_day_speech
    state = _make_state(phase="day", day_number=1)
    registry, _ = _make_capture_registry(
        speaker_id, ActionType.SPEECH, target_id="v1", speech="speech",
    )
    agent_day_speech(state, state["engine"], registry, speaker_id)
    return registry.agent.last_context


# ---------------------------------------------------------------------------
# P0-I1 — TestDirectiveRoleGating
# ---------------------------------------------------------------------------


class TestDirectiveRoleGating:
    """For every role, the rendered day-speech ``strategy_directive`` must
    include the role's own keys and exclude every other role's keys.

    This is the gap the existing ``_assert_no_forbidden_info`` does not
    cover (it only inspects ``visible_world_state``).
    """

    def test_villager_directive_isolated(self) -> None:
        ctx = _drive_day_speech(PLAYER_ID_BY_ROLE["villager"])
        d = ctx.strategy_directive
        for key in PRESENCE["villager"]:
            assert key in d, f"villager missing own directive key: {key}"
        for key in ABSENCE["villager"]:
            assert key not in d, f"villager must not see {key} directive"

    def test_seer_directive_isolated(self) -> None:
        ctx = _drive_day_speech(PLAYER_ID_BY_ROLE["seer"])
        d = ctx.strategy_directive
        for key in PRESENCE["seer"]:
            assert key in d, f"seer missing own directive key: {key}"
        for key in ABSENCE["seer"]:
            assert key not in d, f"seer must not see {key} directive"

    def test_witch_directive_isolated(self) -> None:
        ctx = _drive_day_speech(PLAYER_ID_BY_ROLE["witch"])
        d = ctx.strategy_directive
        for key in PRESENCE["witch"]:
            assert key in d, f"witch missing own directive key: {key}"
        for key in ABSENCE["witch"]:
            assert key not in d, f"witch must not see {key} directive"

    def test_hunter_directive_isolated(self) -> None:
        ctx = _drive_day_speech(PLAYER_ID_BY_ROLE["hunter"])
        d = ctx.strategy_directive
        for key in PRESENCE["hunter"]:
            assert key in d, f"hunter missing own directive key: {key}"
        for key in ABSENCE["hunter"]:
            assert key not in d, f"hunter must not see {key} directive"

    def test_idiot_directive_isolated(self) -> None:
        ctx = _drive_day_speech(PLAYER_ID_BY_ROLE["idiot"])
        d = ctx.strategy_directive
        for key in PRESENCE["idiot"]:
            assert key in d, f"idiot missing own directive key: {key}"
        for key in ABSENCE["idiot"]:
            assert key not in d, f"idiot must not see {key} directive"

    def test_hybrid_directive_isolated(self) -> None:
        # The hybrid needs a chosen master to render a meaningful
        # directive, but the *gating* contract holds either way.
        state = _make_state(
            phase="day",
            day_number=1,
            hybrid_master_id="v1",
            hybrid_master_faction="good",
        )
        from werewolf_agent.runtime.agent_adapter import agent_day_speech
        registry, _ = _make_capture_registry(
            "hybrid", ActionType.SPEECH, target_id="v1", speech="speech",
        )
        agent_day_speech(state, state["engine"], registry, "hybrid")
        d = registry.agent.last_context.strategy_directive
        for key in PRESENCE["hybrid"]:
            assert key in d, f"hybrid missing own directive key: {key}"
        for key in ABSENCE["hybrid"]:
            assert key not in d, f"hybrid must not see {key} directive"

    def test_werewolf_directive_isolated(self) -> None:
        # Werewolves see wolf speech directives on day and night
        # separately.  Day-speech is what the rest of the tests use, so
        # we keep the same surface for apples-to-apples comparison.
        state = _make_state(
            phase="day",
            day_number=1,
            wolf_team_plan={
                "day_push_target": "v1",
                "night_kill_primary": "seer",
            },
        )
        from werewolf_agent.runtime.agent_adapter import agent_day_speech
        registry, _ = _make_capture_registry(
            "w1", ActionType.SPEECH, target_id="v1", speech="speech",
        )
        agent_day_speech(state, state["engine"], registry, "w1")
        d = registry.agent.last_context.strategy_directive
        for key in PRESENCE["werewolf"]:
            assert key in d, f"werewolf missing own directive key: {key}"
        for key in ABSENCE["werewolf"]:
            assert key not in d, f"werewolf must not see {key} directive"

    def test_no_role_receives_moderator_only_directive_keys(self) -> None:
        """Sanity guard — keys reserved for the moderator/judge layer
        must never reach a player's day-speech directive."""
        reserved = {"moderator_full", "moderator_only", "all_player_roles"}
        for role, player_id in PLAYER_ID_BY_ROLE.items():
            ctx = _drive_day_speech(player_id) if role != "hybrid" else (
                _drive_day_speech("hybrid")
            )
            d = ctx.strategy_directive
            for key in reserved:
                assert key not in d, f"{role} sees moderator-only key: {key}"


# ---------------------------------------------------------------------------
# P0-I3 — TestDirectiveWolfPrivateNoLeak
# ---------------------------------------------------------------------------


# Specific wolf-private keys that must never leak to a non-wolf role.
# Listed explicitly per the P0-I3 task description.
WOLF_PRIVATE_KEYS: set[str] = {
    "wolf_fake_seer_teammate",
    "wolf_day_push_target",
    "wolf_plan_target",
    "wolf_teammate_exposed",
    "wolf_high_priority_target",
}

NON_WOLF_ROLES: tuple[str, ...] = (
    "villager", "seer", "witch", "hunter", "idiot", "hybrid",
)


def _drive_day_speech_with_teammate_events(
    speaker_id: str,
    speaker_role: str,
) -> AgentContext:
    """Drive ``agent_day_speech`` for a non-wolf role with a rich event
    timeline that would, in the werewolf pipeline, surface wolf-private
    directive keys (claimed seer, fake seer teammate, plan target, etc.).
    """
    events = [
        GameEvent(type="speech", payload={
            "speaker": "w1", "day_number": 1,
            "text": "我是预言家，昨晚查了v1是好人，警徽流查v2。",
        }),
        GameEvent(type="speech", payload={
            "speaker": "seer", "day_number": 1,
            "text": "我才是真预言家，昨晚查了w1是狼人。",
        }),
    ]
    state = _make_state(
        phase="day",
        day_number=1,
        extra_events=events,
        wolf_team_plan={
            "fake_seer": "w1",
            "day_push_target": "v1",
            "night_kill_primary": "seer",
        },
    )
    from werewolf_agent.runtime.agent_adapter import agent_day_speech
    registry, _ = _make_capture_registry(
        speaker_id, ActionType.SPEECH, target_id="v1", speech="speech",
    )
    agent_day_speech(state, state["engine"], registry, speaker_id)
    return registry.agent.last_context


class TestDirectiveWolfPrivateNoLeak:
    """For every non-werewolf role, the wolf-private directive keys
    ``{wolf_fake_seer_teammate, wolf_day_push_target, wolf_plan_target,
    wolf_teammate_exposed, wolf_high_priority_target}`` must be absent
    from BOTH ``strategy_directive`` and ``visible_world_state``.
    """

    def test_villager_no_wolf_private_directive(self) -> None:
        ctx = _drive_day_speech_with_teammate_events("v1", "villager")
        for key in WOLF_PRIVATE_KEYS:
            assert key not in ctx.strategy_directive, (
                f"villager strategy_directive leaks wolf-private key: {key}"
            )
            assert key not in ctx.visible_world_state, (
                f"villager visible_world_state leaks wolf-private key: {key}"
            )

    def test_seer_no_wolf_private_directive(self) -> None:
        ctx = _drive_day_speech_with_teammate_events("seer", "seer")
        for key in WOLF_PRIVATE_KEYS:
            assert key not in ctx.strategy_directive, (
                f"seer strategy_directive leaks wolf-private key: {key}"
            )
            assert key not in ctx.visible_world_state, (
                f"seer visible_world_state leaks wolf-private key: {key}"
            )

    def test_witch_no_wolf_private_directive(self) -> None:
        ctx = _drive_day_speech_with_teammate_events("witch", "witch")
        for key in WOLF_PRIVATE_KEYS:
            assert key not in ctx.strategy_directive, (
                f"witch strategy_directive leaks wolf-private key: {key}"
            )
            assert key not in ctx.visible_world_state, (
                f"witch visible_world_state leaks wolf-private key: {key}"
            )

    def test_hunter_no_wolf_private_directive(self) -> None:
        ctx = _drive_day_speech_with_teammate_events("hunter", "hunter")
        for key in WOLF_PRIVATE_KEYS:
            assert key not in ctx.strategy_directive, (
                f"hunter strategy_directive leaks wolf-private key: {key}"
            )
            assert key not in ctx.visible_world_state, (
                f"hunter visible_world_state leaks wolf-private key: {key}"
            )

    def test_idiot_no_wolf_private_directive(self) -> None:
        ctx = _drive_day_speech_with_teammate_events("idiot", "idiot")
        for key in WOLF_PRIVATE_KEYS:
            assert key not in ctx.strategy_directive, (
                f"idiot strategy_directive leaks wolf-private key: {key}"
            )
            assert key not in ctx.visible_world_state, (
                f"idiot visible_world_state leaks wolf-private key: {key}"
            )

    def test_hybrid_no_wolf_private_directive(self) -> None:
        state = _make_state(
            phase="day",
            day_number=1,
            extra_events=[
                GameEvent(type="speech", payload={
                    "speaker": "w1", "day_number": 1,
                    "text": "我是预言家，昨晚查了v1是好人",
                }),
            ],
            hybrid_master_id="v1",
            hybrid_master_faction="good",
            wolf_team_plan={
                "fake_seer": "w1",
                "day_push_target": "v1",
                "night_kill_primary": "seer",
            },
        )
        from werewolf_agent.runtime.agent_adapter import agent_day_speech
        registry, _ = _make_capture_registry(
            "hybrid", ActionType.SPEECH, target_id="v1", speech="speech",
        )
        agent_day_speech(state, state["engine"], registry, "hybrid")
        ctx = registry.agent.last_context
        for key in WOLF_PRIVATE_KEYS:
            assert key not in ctx.strategy_directive, (
                f"hybrid strategy_directive leaks wolf-private key: {key}"
            )
            assert key not in ctx.visible_world_state, (
                f"hybrid visible_world_state leaks wolf-private key: {key}"
            )


def test_skill_opportunity_actor_projection_does_not_enter_other_role_contexts() -> None:
    from werewolf_agent.cognition.visibility import VisibilityPolicy
    from werewolf_agent.cognition.world_state import build_world_state
    from werewolf_agent.runtime.skill_opportunity_events import build_private_skill_event

    state = _make_state()
    game_state = replace(
        state["game_state"],
        events=list(build_private_skill_event(
            "self_destruct_opportunity",
            actor_id="w1",
            day_number=1,
            available_actions=["self_destruct", "continue"],
        )),
    )
    world = build_world_state(game_state)
    policy = VisibilityPolicy()

    wolf = policy.filter_visible_facts(world, viewer_id="w1", viewer_role="werewolf")
    villager = policy.filter_visible_facts(world, viewer_id="v1", viewer_role="villager")

    assert [fact.fact_type for fact in wolf] == ["self_destruct_opportunity_actor_view"]
    assert villager == []
