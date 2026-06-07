"""Tests for sheriff election policy: voter eligibility, all-on-sheriff, direct election."""

from dataclasses import replace

import pytest

from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.runtime.sheriff_policy import (
    choose_no_sheriff_speech_order,
    choose_sheriff_led_speech_order,
    eligible_sheriff_voters,
    filter_sheriff_votes_to_eligible,
    is_all_players_on_sheriff,
    resolve_no_vote_sheriff_reason,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gs(
    alive_count: int = 12,
    candidates: list[str] | None = None,
) -> GameState:
    players = {
        f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True)
        for i in range(1, alive_count + 1)
    }
    return GameState(
        game_id="test",
        phase="day",
        day_number=1,
        players=players,
        sheriff_candidates=candidates or [],
    )


# ---------------------------------------------------------------------------
# Test 1: all alive players on sheriff means no election
# ---------------------------------------------------------------------------

class TestAllPlayersOnSheriff:
    def test_all_players_on_sheriff_loses_badge(self):
        """12 players all alive, all 12 register as sheriff candidates.
        Result: sheriff_no_election with reason all_players_on_sheriff."""
        all_ids = [f"p{i:02d}" for i in range(1, 13)]
        gs = _make_gs(alive_count=12, candidates=all_ids)

        assert is_all_players_on_sheriff(gs, all_ids) is True
        reason = resolve_no_vote_sheriff_reason(gs, candidates=all_ids, voters=[])
        assert reason == "all_players_on_sheriff"

    def test_not_all_on_sheriff_when_one_missing(self):
        ids_11 = [f"p{i:02d}" for i in range(1, 12)]  # p12 not a candidate
        gs = _make_gs(alive_count=12, candidates=ids_11)
        assert is_all_players_on_sheriff(gs, ids_11) is False

    def test_dead_players_not_counted(self):
        """Dead player not in candidates should not trigger all-on-sheriff."""
        all_ids = [f"p{i:02d}" for i in range(1, 13)]
        gs = _make_gs(alive_count=12, candidates=all_ids)
        # Kill p12
        dead_p12 = replace(gs.players["p12"], alive=False)
        gs = replace(gs, players={**gs.players, "p12": dead_p12})
        # Now only 11 alive, all 11 are candidates => still all-on-sheriff
        assert is_all_players_on_sheriff(gs, all_ids) is True


# ---------------------------------------------------------------------------
# Test 2: only off-sheriff players vote
# ---------------------------------------------------------------------------

class TestEligibleVoters:
    def test_only_off_sheriff_players_vote(self):
        """12 players, candidates = [p01, p02, p03].
        Only the remaining 9 players are eligible voters."""
        candidates = ["p01", "p02", "p03"]
        gs = _make_gs(alive_count=12, candidates=candidates)
        voters = eligible_sheriff_voters(gs, candidates)
        # p01, p02, p03 should NOT be in voters
        assert "p01" not in voters
        assert "p02" not in voters
        assert "p03" not in voters
        # All other 9 should be present
        assert len(voters) == 9
        for i in range(4, 13):
            assert f"p{i:02d}" in voters

    def test_withdrew_players_still_cannot_vote(self):
        """Players who went on sheriff and then withdrew still cannot vote."""
        candidates = ["p01", "p02", "p03"]
        withdrew = ["p02", "p03"]
        gs = _make_gs(alive_count=12, candidates=candidates)
        voters = eligible_sheriff_voters(gs, candidates, withdrew=withdrew)
        # p01, p02, p03 are all excluded
        assert "p01" not in voters
        assert "p02" not in voters
        assert "p03" not in voters
        assert len(voters) == 9

    def test_dead_players_cannot_vote(self):
        """Dead players should not appear in voters even if off-sheriff."""
        candidates = ["p01", "p02", "p03"]
        gs = _make_gs(alive_count=12, candidates=candidates)
        dead_p04 = replace(gs.players["p04"], alive=False)
        gs = replace(gs, players={**gs.players, "p04": dead_p04})
        voters = eligible_sheriff_voters(gs, candidates)
        assert "p04" not in voters
        assert len(voters) == 8

    def test_invalid_sheriff_votes_from_candidates_and_withdrew_players_are_filtered(self):
        """Only players who never went on sheriff can cast sheriff-election votes."""
        candidates = ["p01", "p02"]
        withdrew = ["p03"]
        gs = _make_gs(alive_count=12, candidates=candidates)
        votes = {
            "p01": "p01",  # active candidate, invalid voter
            "p03": "p01",  # withdrew from sheriff, invalid voter
            "p04": "p02",  # off-sheriff voter, valid
        }

        filtered = filter_sheriff_votes_to_eligible(
            gs,
            votes,
            candidates=candidates,
            withdrew=withdrew,
        )

        assert filtered == {"p04": "p02"}


# ---------------------------------------------------------------------------
# Test 3: one remaining candidate after withdrawal elected directly
# ---------------------------------------------------------------------------

class TestDirectElection:
    def test_one_remaining_candidate_elected_directly(self):
        """After withdrawal, only 1 candidate remains -> elected without vote."""
        # This tests the policy helper: with 1 remaining candidate,
        # the graph node should elect directly. We verify the precondition.
        candidates = ["p01", "p02", "p03"]
        withdrew = ["p02", "p03"]
        remaining = [c for c in candidates if c not in withdrew]
        assert remaining == ["p01"]
        assert len(remaining) == 1
        # is_all_players_on_sheriff should be False here
        gs = _make_gs(alive_count=12, candidates=candidates)
        assert is_all_players_on_sheriff(gs, remaining) is False


# ---------------------------------------------------------------------------
# Test 4: no candidates means no sheriff
# ---------------------------------------------------------------------------

class TestNoCandidates:
    def test_no_candidates_means_no_sheriff(self):
        """No candidates registered -> no election with reason no_candidates."""
        gs = _make_gs(alive_count=12, candidates=[])
        reason = resolve_no_vote_sheriff_reason(gs, candidates=[], voters=[])
        assert reason == "no_candidates"

    def test_all_withdrawn_means_no_candidates(self):
        """All candidates withdraw -> no candidates remaining."""
        gs = _make_gs(alive_count=12, candidates=["p01", "p02"])
        reason = resolve_no_vote_sheriff_reason(
            gs, candidates=[], voters=[]
        )
        assert reason == "no_candidates"


# ---------------------------------------------------------------------------
# Test 5: normal sheriff vote by off-sheriff voters
# ---------------------------------------------------------------------------

class TestNormalSheriffVote:
    def test_normal_sheriff_vote_by_off_sheriff_voters(self):
        """candidates = [p01, p02, p03], off-sheriff voters vote majority for p01.
        p01 becomes sheriff. This verifies voter eligibility for the normal case."""
        candidates = ["p01", "p02", "p03"]
        gs = _make_gs(alive_count=12, candidates=candidates)
        voters = eligible_sheriff_voters(gs, candidates)

        # Simulate majority voting for p01
        votes = {voter: "p01" for voter in voters}
        # Verify no candidate is in the voter set
        for c in candidates:
            assert c not in votes, f"Candidate {c} should not be a voter"

        # is_all_players_on_sheriff should be False
        assert is_all_players_on_sheriff(gs, candidates) is False

        # resolve_no_vote_sheriff_reason should not return all_players_on_sheriff
        # when there are voters and candidates
        reason = resolve_no_vote_sheriff_reason(gs, candidates, voters)
        # With normal candidates and voters, reason depends on vote outcome
        # But the function should at least not return no_candidates or all_players_on_sheriff
        assert reason in ("vote_tie", "all_players_on_sheriff", "no_candidates")
        # In this case with 9 voters, it's not all_on_sheriff or no_candidates
        assert reason != "all_players_on_sheriff"
        assert reason != "no_candidates"
        # With a clear majority, the RuleEngine resolves it (not this function's job)


# ===================================================================
# No-sheriff speech order
# ===================================================================


class TestNoSheriffSpeechOrder:
    def test_judge_selects_deterministic_speech_start(self):
        """When no sheriff exists, judge creates deterministic random speech order from seed."""
        gs = _make_gs()
        order1 = choose_no_sheriff_speech_order(gs, seed=42)
        order2 = choose_no_sheriff_speech_order(gs, seed=42)
        assert order1 == order2  # deterministic
        assert set(order1) == {f"p{i:02d}" for i in range(1, 13)}  # all alive players

    def test_different_seeds_different_orders(self):
        gs = _make_gs()
        order1 = choose_no_sheriff_speech_order(gs, seed=42)
        order2 = choose_no_sheriff_speech_order(gs, seed=99)
        assert order1 != order2  # different seeds produce different orders

    def test_empty_game(self):
        gs = GameState(game_id="test", players={})
        assert choose_no_sheriff_speech_order(gs) == []


# ===================================================================
# Sheriff-led speech order
# ===================================================================


class TestSheriffLedSpeechOrder:
    def test_sheriff_places_focus_players_early_and_self_last(self):
        """Sheriff controls order: focus players early, sheriff last."""
        gs = _make_gs()
        sheriff_id = "p01"
        focus = ["p05", "p07"]  # counterclaim seers
        order = choose_sheriff_led_speech_order(gs, sheriff_id, focus_players=focus)
        assert order[-1] == sheriff_id  # sheriff last
        assert order[0] == "p05"  # focus first
        assert order[1] == "p07"  # focus second
        assert set(order) == {f"p{i:02d}" for i in range(1, 13)}  # all alive

    def test_no_focus_players(self):
        gs = _make_gs()
        sheriff_id = "p06"
        order = choose_sheriff_led_speech_order(gs, sheriff_id)
        assert order[-1] == sheriff_id
        # Others should be in seat order
        others = order[:-1]
        assert others == sorted(others)  # ascending seat order

    def test_counterclockwise_direction(self):
        gs = _make_gs()
        sheriff_id = "p06"
        order_cw = choose_sheriff_led_speech_order(
            gs, sheriff_id, direction="clockwise"
        )
        order_ccw = choose_sheriff_led_speech_order(
            gs, sheriff_id, direction="counterclockwise"
        )
        # Non-focus, non-sheriff players should be reversed
        assert order_cw[-1] == sheriff_id
        assert order_ccw[-1] == sheriff_id


# ===================================================================
# Sheriff candidate filter
# ===================================================================


def test_filter_removes_template_matching_speeches():
    from werewolf_agent.runtime.sheriff_policy import filter_sheriff_candidates
    gs = GameState(game_id="test")
    candidates = ["p03", "p04"]
    speeches = {
        "p03": "我这轮先把视角压到p01身上。依据是p04最近发言：我是村民...",
        "p04": "我是预言家，昨晚查验了p01是好人。我的警徽流先留p08，第二夜验p12。请大家支持我当警长。",
    }
    result = filter_sheriff_candidates(gs, candidates, speeches)
    assert result == ["p04"]


def test_filter_removes_short_speeches():
    from werewolf_agent.runtime.sheriff_policy import filter_sheriff_candidates
    gs = GameState(game_id="test")
    candidates = ["p03", "p04"]
    speeches = {"p03": "我上警", "p04": "我是预言家，昨晚查验了p01是好人，警徽流留p08"}
    result = filter_sheriff_candidates(gs, candidates, speeches)
    assert "p03" not in result


def test_filter_keeps_all_when_valid():
    from werewolf_agent.runtime.sheriff_policy import filter_sheriff_candidates
    gs = GameState(game_id="test")
    candidates = ["p03", "p04"]
    speeches = {
        "p03": "我是预言家，昨晚查验p01是好人，警徽流留p08，验人动机是他在警下靠前位置",
        "p04": "我是村民，上警是为了分享我对当前局势的分析和判断，重点关注后置位的发言逻辑。",
    }
    result = filter_sheriff_candidates(gs, candidates, speeches)
    assert result == ["p03", "p04"]


def test_sheriff_policy_stable_seed_is_shared():
    """P-U3: sheriff_policy._stable_seed 应是 _shared._stable_seed 的同一对象。"""
    from werewolf_agent.runtime.sheriff_policy import _stable_seed
    from werewolf_agent.runtime.nodes._shared import _stable_seed as shared_seed
    assert _stable_seed is shared_seed, (
        f"sheriff_policy has its own _stable_seed implementation; should import from _shared"
    )
