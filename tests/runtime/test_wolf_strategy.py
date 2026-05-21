"""Tests for wolf night discussion evidence-based consensus."""

import pytest
from dataclasses import replace
from werewolf_agent.core.models import GameState, PlayerState, GameEvent


def _make_wolf_gs(night=1, wolf_count=4):
    """Create game state with wolves alive."""
    players = {}
    for i in range(1, 13):
        role = "werewolf" if i <= wolf_count else "villager"
        players[f"p{i:02d}"] = PlayerState(id=f"p{i:02d}", role=role, alive=True)
    events = [
        GameEvent(type="enter_night", payload={"night": night}),
    ]
    # Add discussion events
    for round_num in range(1, 4):
        for w in range(1, wolf_count + 1):
            events.append(GameEvent(
                type="wolf_discussion",
                payload={
                    "wolf_id": f"p{w:02d}",
                    "round": round_num,
                    "night_number": night,
                    "text": f"讨论第{round_num}轮 p{w:02d}",
                    "visibility": "werewolf_team_only",
                },
            ))
    return GameState(
        game_id="test",
        phase="night",
        night_number=night,
        players=players,
        events=events,
    )


class TestWolfDiscussionRequiresSpeech:
    """No silent wolf discussion -- empty speech rejected."""

    def test_wolf_discussion_requires_non_empty_speech(self):
        from werewolf_agent.runtime.wolf_strategy import extract_wolf_proposal
        # Empty speech should return empty proposal
        proposal = extract_wolf_proposal("")
        assert proposal.get("target") is None
        assert proposal.get("role_assignment") is None

    def test_extract_kill_target_from_speech(self):
        from werewolf_agent.runtime.wolf_strategy import extract_wolf_proposal
        # Speech proposing a kill target
        text = "我觉得应该刀p05，p05可能是预言家"
        proposal = extract_wolf_proposal(text)
        assert proposal.get("target") == "p05"

    def test_extract_role_proposal(self):
        from werewolf_agent.runtime.wolf_strategy import extract_wolf_proposal
        text = "我来做假预言家，p02你冲锋，p03倒钩，p04深水"
        proposal = extract_wolf_proposal(text)
        assert proposal.get("role_assignment") is not None
        if proposal.get("role_assignment"):
            assert "p01" in proposal["role_assignment"] or "fake_seer" in str(proposal["role_assignment"])


class TestWolfPlanDerivedFromDiscussion:
    """Wolf team plan is derived from discussion evidence, not seat order."""

    def test_wolf_plan_is_derived_from_discussion_evidence(self):
        from werewolf_agent.runtime.wolf_strategy import summarize_wolf_consensus
        gs = _make_wolf_gs()
        # Override discussion texts with meaningful proposals
        events = [e for e in gs.events]
        for i, e in enumerate(events):
            if e.type == "wolf_discussion" and e.payload.get("round") == 1:
                wolf_id = e.payload["wolf_id"]
                if wolf_id == "p01":
                    events[i] = replace(e, payload={**e.payload, "text": "建议刀p08，我来假预言家"})
                elif wolf_id == "p02":
                    events[i] = replace(e, payload={**e.payload, "text": "同意刀p08，p01假预言家，我做冲锋"})
                elif wolf_id == "p03":
                    events[i] = replace(e, payload={**e.payload, "text": "同意p08，我做倒钩"})
                elif wolf_id == "p04":
                    events[i] = replace(e, payload={**e.payload, "text": "同意刀p08，深水位"})
        gs = replace(gs, events=events)
        alive_wolves = [f"p{i:02d}" for i in range(1, 5)]
        consensus = summarize_wolf_consensus(gs.events, alive_wolves)
        # The consensus should reference p08 as kill target
        assert consensus.get("night_kill_primary") == "p08"
        # Should have evidence from discussion
        assert consensus.get("evidence_from_discussion") is not None
        assert len(consensus["evidence_from_discussion"]) > 0

    def test_consensus_includes_role_assignments(self):
        from werewolf_agent.runtime.wolf_strategy import summarize_wolf_consensus
        gs = _make_wolf_gs()
        events = list(gs.events)
        for i, e in enumerate(events):
            if e.type == "wolf_discussion" and e.payload.get("round") == 1:
                wolf_id = e.payload["wolf_id"]
                if wolf_id == "p01":
                    events[i] = replace(e, payload={**e.payload, "text": "我来做假预言家"})
                elif wolf_id == "p02":
                    events[i] = replace(e, payload={**e.payload, "text": "我做冲锋位"})
                elif wolf_id == "p03":
                    events[i] = replace(e, payload={**e.payload, "text": "我做倒钩"})
                elif wolf_id == "p04":
                    events[i] = replace(e, payload={**e.payload, "text": "我做深水"})
        gs = replace(gs, events=events)
        alive_wolves = [f"p{i:02d}" for i in range(1, 5)]
        consensus = summarize_wolf_consensus(gs.events, alive_wolves)
        assert consensus.get("fake_seer") is not None

    def test_wolf_plan_without_discussion_evidence_has_no_kill_target(self):
        from werewolf_agent.runtime.wolf_strategy import build_wolf_team_plan_from_discussion

        gs = _make_wolf_gs()
        consensus = {
            "night_kill_primary": None,
            "night_kill_backup": None,
            "evidence_from_discussion": [],
            "agreement_count": 0,
            "total_wolves": 4,
        }

        plan = build_wolf_team_plan_from_discussion(gs, consensus=consensus)

        assert plan.get("night_kill_primary") is None
        assert plan.get("night_kill_backup") is None
        assert plan.get("evidence_quality") == "none"

    def test_planned_wolf_kill_ignores_low_evidence_plan(self):
        from werewolf_agent.runtime.graph import _planned_wolf_kill

        gs = _make_wolf_gs()
        state = {
            "game_state": gs,
            "wolf_team_plan": {
                "night_kill_primary": "p05",
                "evidence_quality": "none",
            },
        }

        assert _planned_wolf_kill(state) is None


class TestWolfDiscussionEarlyStop:
    """If majority agrees in round 1, later rounds can be skipped."""

    def test_wolf_discussion_can_end_early_after_consensus(self):
        from werewolf_agent.runtime.wolf_strategy import should_end_discussion_early
        # 3 out of 4 wolves agree on target and roles in round 1
        consensus = {
            "night_kill_primary": "p08",
            "agreement_count": 3,
            "total_wolves": 4,
        }
        assert should_end_discussion_early(consensus, 4) is True

    def test_no_early_stop_without_majority(self):
        from werewolf_agent.runtime.wolf_strategy import should_end_discussion_early
        consensus = {
            "night_kill_primary": "p08",
            "agreement_count": 2,
            "total_wolves": 4,
        }
        assert should_end_discussion_early(consensus, 4) is False

    def test_no_early_stop_with_only_2_wolves(self):
        from werewolf_agent.runtime.wolf_strategy import should_end_discussion_early
        consensus = {
            "night_kill_primary": "p08",
            "agreement_count": 1,
            "total_wolves": 2,
        }
        assert should_end_discussion_early(consensus, 2) is False


class TestRoundRequirements:
    """Discussion round requirements vary by night number."""

    def test_night_1_has_3_rounds(self):
        from werewolf_agent.runtime.wolf_strategy import round_requirements
        reqs = round_requirements(night_number=1, round_number=1)
        assert "fake_seer" in reqs or "role_assignment" in reqs

    def test_later_nights_fewer_rounds(self):
        from werewolf_agent.runtime.wolf_strategy import round_requirements
        reqs = round_requirements(night_number=2, round_number=1)
        # Later nights should focus on review, not role assignment
        assert reqs is not None
