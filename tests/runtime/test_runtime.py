"""Runtime tests: scripted non-LLM games through the LangGraph graph."""

from __future__ import annotations

import pytest
from dataclasses import replace

from typing import Any

from werewolf_agent.core.models import Death, GameState, PlayerState, GameEvent
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.agents.schemas import (
    ActionType, AgentContext, PlayerAction, RetryInfo, FallbackAction,
    TaskType,
)
from werewolf_agent.runtime.graph import (
    RuntimeState,
    build_game_graph,
    build_game_graph_with_checkpoint,
    _new_engine,
    _alive_wolves,
    _alive_non_wolves,
    _find_role,
    _stable_seed,
    check_victory,
    free_discussion,
    wolf_consensus,
    route_after_resolve_night,
    route_after_hunter_shot,
    route_after_post_exile,
    _sheriff_died_this_batch,
    _route_after_badge_transfer,
    _action_trace_event,
)
from werewolf_agent.runtime.agent_adapter import _single_wolf_vote
from werewolf_agent.runtime.replay import replay_from_events, extract_event_log
from werewolf_agent.runtime.checkpoints import make_checkpointer


def _last_non_broadcast_event(gs: GameState) -> GameEvent:
    return next(e for e in reversed(gs.events) if e.type != "judge_broadcast")

def _make_scripted_state(
    *,
    wolf_kill_targets: list[str | None],
    exile_votes_list: list[dict[str, str]] | None = None,
) -> RuntimeState:
    return {
        "game_state": GameState(game_id="scripted01"),
        "engine": _new_engine(),
        "wolf_kill_targets": wolf_kill_targets,
        "wolf_kill_target_id": wolf_kill_targets[0] if wolf_kill_targets else None,
        "use_antidote": False,
        "poison_target_id": None,
        "seer_target_id": None,
        "hybrid_master_target_id": None,
        "self_destruct_wolf_id": None,
        "exile_votes": exile_votes_list[0] if exile_votes_list else {},
        "revote": False,
        "sheriff_candidates": [],
        "sheriff_votes": {},
        "sheriff_withdrawing": [],
        "badge_decision": "tear",
        "badge_target_id": None,
        "hunter_shot_target_id": None,
    }



# ---------------------------------------------------------------------------
# Backward-compat re-imports: all classes and standalone tests moved to
# domain-specific test files.
# ---------------------------------------------------------------------------

# test_night_flow.py
from tests.runtime.test_night_flow import (
    TestNightHunterIdiotStatusNode,
    TestSeerNightResolution,
    test_seer_night_targets_exclude_counterclaiming_seers,
    test_all_players_on_sheriff_announces_no_sheriff_before_speeches,
    test_resolve_night_node_kills_target,
    test_resolve_night_node_keeps_witch_potion_events_in_timeline,
    test_announce_deaths_increments_day,
)

# test_wolf_flow.py
from tests.runtime.test_wolf_flow import (
    TestWolfDiscussionLoop,
    TestWolfFallbackVoteNoTeammate,
    TestWolfPlanDedup,
    test_wolf_consensus_timeout_defaults_to_no_kill_event,
    test_wolf_consensus_explicit_no_kill_records_declared_event,
    test_wolf_consensus_kill_records_selected_target,
    test_wolf_consensus_prefers_planned_primary_then_backup_target,
    test_wolf_discussion_timer_expiration_forces_no_kill_timeout,
    test_first_night_wolf_discussion_runs_three_rounds_and_builds_team_plan,
    test_later_night_wolf_discussion_runs_two_rounds_and_revises_plan,
    test_wolf_discussion_drops_stale_targets_without_current_discussion_evidence,
)

# test_witch_flow.py
from tests.runtime.test_witch_flow import (
    TestWitchDecisionFlow,
    TestWitchPoisonPressureContext,
)

# test_hunter_flow.py
from tests.runtime.test_hunter_flow import (
    TestHunterShotTiming,
    TestHunterShotOrdering,
    TestHunterShotResolution,
    TestHunterShotPublicEvent,
)

# test_sheriff_flow.py
from tests.runtime.test_sheriff_flow import (
    TestSheriffBadgeAfterNightDeath,
    TestSheriffBadgeNightDeathRouting,
    TestSheriffElectionSpeechFallback,
    test_sheriff_vote_ignores_candidates_and_withdrew_voters,
    test_sheriff_speech_calls_candidate_agents_and_keeps_trace_private,
)

# test_vote_flow.py
from tests.runtime.test_vote_flow import (
    TestVoteLifecycle,
    TestAntiStallPolicy,
    test_resolve_vote_keeps_action_traces_out_of_public_result,
    test_resolve_vote_records_sheriff_weighted_tally,
    test_resolve_vote_first_tie_emits_pk_broadcast,
    test_vote_action_trace_audit_exposes_structured_private_vote_thought_to_moderator_only,
    test_resolve_vote_records_vote_reasons_for_public_ledger,
    test_resolve_vote_uses_fallback_reason_for_public_ledger,
    test_agent_day_vote_excludes_voter_from_legal_targets,
    test_day_vote_announces_vote_collection_and_end,
)

# test_event_sourcing.py
from tests.runtime.test_event_sourcing import (
    TestPauseResumeEventSourcing,
    TestStartGameEventSourcing,
)

# test_judge_flow.py
from tests.runtime.test_judge_flow import (
    TestJudgeControlsNightRoleSequence,
    TestJudgeControlsDaySequence,
)

# test_strategy_directives.py
from tests.runtime.test_strategy_directives import (
    TestWitchStrategyHints,
    TestSeerStrategyDirectives,
    TestHunterStrategyDirectives,
    TestHybridStrategyDirectives,
    TestVillagerStrategyDirectives,
    TestIdiotStrategyDirectives,
    TestWolfStrategyDirectives,
    TestEmptySpeechGuard,
)

# test_graph_lifecycle.py
from tests.runtime.test_graph_lifecycle import (
    test_graph_compiles,
    test_graph_compiles_with_checkpoint,
    test_setup_and_assign_roles,
    test_scripted_peace_game_does_not_forge_winner,
    test_scripted_game_with_wolf_kill_does_not_forge_winner,
    test_setup_game_node,
    test_assign_roles_node,
    test_enter_night_increments_night,
    test_reflection_node_persists_entries_to_repository,
    test_single_wolf_vote_uses_global_agent_timeout,
    test_dispatch_agent_direct_call_when_timeout_zero,
    test_manual_timer_expiration_is_deterministic,
    test_action_trace_audit_flags_timeline_confusion,
    test_check_victory_good_wins,
    test_check_victory_no_winner_yet,
    test_check_victory_does_not_force_scripted_day_limit_winner,
    test_stable_seed_is_deterministic_for_same_parts,
    test_route_victory_finishes_when_won,
    test_route_victory_continues_when_no_winner,
    test_route_after_vote_exile,
    test_route_after_vote_tie,
    test_route_after_announce_night1_goes_to_free_discussion,
    test_route_after_announce_day2_discussion,
    test_route_after_free_discussion_continues_until_speech_queue_done,
    test_replay_from_events_matches_state,
    test_extract_event_log,
    test_phase1_rule_tests_still_pass,
)

# test_day_discussion.py
from tests.runtime.test_day_discussion import (
    test_free_discussion_speech_timeout_records_event,
    test_free_discussion_speech_timeout_advances_speech_queue,
    test_free_discussion_timer_expiration_records_timeout,
    test_free_discussion_normal_speech_advances_speech_queue,
    test_free_discussion_keeps_action_trace_out_of_public_speech,
    test_day_speech_passes_wolf_team_plan_to_werewolf_agent,
    test_day_speech_requires_speech_action_from_agent,
    test_announce_deaths_skips_increment_when_phase_is_day,
    test_free_discussion_routes_to_vote_after_last_normal_speech,
    test_free_discussion_announces_speech_order_and_discussion_end,
    test_night_death_last_words_has_public_broadcast,
    test_night_death_last_words_broadcasts_skip_when_no_eligible_players,
)
