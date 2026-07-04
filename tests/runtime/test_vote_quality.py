"""Tests for evidence-based vote quality validation."""

import pytest
from werewolf_agent.runtime.vote_quality import (
    extract_vote_basis,
    validate_vote_reason,
    validate_structured_vote_action,
    build_day_discussion_summary,
    build_vote_pressure_context,
)


class TestVoteBasisValidator:
    """Vote reason must include at least one logic basis."""

    def test_vote_reason_requires_logic_basis(self):
        """A vote reason with no basis is rejected."""
        action = {
            "action_type": "vote",
            "target_id": "p03",
            "reason": "感觉不太好",
            "speech": "",
        }
        result = validate_vote_reason(action, context={})
        assert result["valid"] is False
        assert result["missing_basis"]

    def test_seer_check_is_valid_basis(self):
        """Reference to seer check result is a valid basis."""
        action = {
            "action_type": "vote",
            "target_id": "p03",
            "reason": "p03被预言家查杀，投票放逐",
            "speech": "",
        }
        result = validate_vote_reason(action, context={"has_seer_claims": True})
        assert result["valid"] is True

    def test_counterclaim_is_valid_basis(self):
        """Reference to counterclaim is a valid basis."""
        action = {
            "action_type": "vote",
            "target_id": "p05",
            "reason": "p05和p01对跳预言家，p01更可信",
            "speech": "",
        }
        result = validate_vote_reason(action, context={"has_counterclaims": True})
        assert result["valid"] is True

    def test_speech_quote_is_valid_basis(self):
        """Reference to prior speech is a valid basis."""
        action = {
            "action_type": "vote",
            "target_id": "p07",
            "reason": "p07刚才说自己是预言家但警徽流不合理",
            "speech": "",
        }
        result = validate_vote_reason(action, context={"has_speeches": True})
        assert result["valid"] is True

    def test_vote_tally_is_valid_basis(self):
        """Reference to vote tally is a valid basis."""
        action = {
            "action_type": "vote",
            "target_id": "p03",
            "reason": "上轮投票p03票数最高，今天继续推",
            "speech": "",
        }
        result = validate_vote_reason(action, context={"has_vote_tally": True})
        assert result["valid"] is True

    def test_contradiction_is_valid_basis(self):
        """Reference to contradiction is a valid basis."""
        action = {
            "action_type": "vote",
            "target_id": "p03",
            "reason": "p03的发言前后矛盾",
            "speech": "",
        }
        result = validate_vote_reason(action, context={"has_contradictions": True})
        assert result["valid"] is True

    def test_structured_vote_rejects_unexplained_fields(self):
        action = {
            "action_type": "vote",
            "target_id": "p03",
            "reason": "未说明",
            "speech": "",
            "seer_stance": "undecided",
            "vote_basis": "fallback",
            "standing_with_seer": "",
            "suspect_reason": "未说明",
            "not_voting_reason": "未说明",
            "private_reason": "未说明",
        }

        result = validate_structured_vote_action(action)

        assert result["valid"] is False
        assert result["error_code"] == "vote_quality"
        assert "投票理由" in result["hint"]

    def test_structured_vote_accepts_seer_stance_and_vote_basis(self):
        action = {
            "action_type": "vote",
            "target_id": "p07",
            "reason": "p08查杀p07，p07没有回应核心问题",
            "speech": "",
            "seer_stance": "trust",
            "vote_basis": "seer_check",
            "standing_with_seer": "p08",
            "suspect_reason": "p07被p08查杀后没有回应查杀逻辑",
            "not_voting_reason": "p06发言虽弱，但没有查验压力",
            "private_reason": "我更信p08的预言家线，所以投p07。",
        }

        result = validate_structured_vote_action(action)

        assert result["valid"] is True
        assert result["detected_bases"] == ["seer_check"]

    def test_structured_vote_rejects_template_candidate_reason(self):
        action = {
            "action_type": "vote",
            "target_id": "p07",
            "reason": "p07是当前合法投票候选，需要基于发言、票型和站边继续施压",
            "speech": "",
            "seer_stance": "undecided",
            "vote_basis": "fallback",
            "standing_with_seer": "",
            "suspect_reason": "p07是当前合法投票候选，需要基于发言、票型和站边继续施压",
            "not_voting_reason": "其他人证据更弱",
            "private_reason": "保守跟票",
        }

        result = validate_structured_vote_action(action)

        assert result["valid"] is False
        assert result["error_code"] == "vote_quality"
        assert result["missing_field"] == "reason"
        assert "候选不是证据" in result["hint"]


class TestExtractVoteBasis:
    """Extract the type of logic basis from a vote reason."""

    def test_extracts_seer_check_basis(self):
        result = extract_vote_basis("p03被预言家查杀")
        assert "seer_check" in result

    def test_extracts_counterclaim_basis(self):
        result = extract_vote_basis("p05和p01对跳预言家")
        assert "counterclaim" in result

    def test_extracts_badge_flow_basis(self):
        result = extract_vote_basis("p03警徽流不合理")
        assert "badge_flow" in result

    def test_extracts_contradiction_basis(self):
        result = extract_vote_basis("p03发言前后矛盾")
        assert "contradiction" in result

    def test_extracts_vote_tally_basis(self):
        result = extract_vote_basis("上轮投票p03票数最高")
        assert "vote_tally" in result

    def test_extracts_stance_reversal_basis(self):
        result = extract_vote_basis("p03立场反复")
        assert "stance_reversal" in result

    def test_extracts_speech_quote_basis(self):
        result = extract_vote_basis("p03刚才说自己是预言家")
        assert "speech_quote" in result

    def test_no_basis_returns_empty(self):
        result = extract_vote_basis("感觉不太好")
        assert result == []


class TestDayDiscussionSummary:
    """Vote context must include full current-day discussion summary."""

    def test_vote_context_contains_full_day_discussion_summary(self):
        """build_day_discussion_summary returns all speeches for current day."""
        from werewolf_agent.core.models import GameState, PlayerState, GameEvent

        players = {f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True) for i in range(1, 13)}
        gs = GameState(
            game_id="test",
            phase="day",
            day_number=2,
            players=players,
            events=[
                GameEvent(type="speech", payload={"speaker": "p01", "day_number": 2, "text": "我是预言家"}),
                GameEvent(type="speech", payload={"speaker": "p02", "day_number": 2, "text": "我同意p01"}),
                GameEvent(type="speech", payload={"speaker": "p03", "day_number": 1, "text": "昨天的话"}),
            ],
        )
        summary = build_day_discussion_summary(gs, day=2)
        assert len(summary) == 2  # Only day 2 speeches
        assert summary[0]["speaker"] == "p01"
        assert summary[1]["speaker"] == "p02"

    def test_empty_day_returns_empty(self):
        from werewolf_agent.core.models import GameState
        gs = GameState(game_id="test", day_number=3)
        summary = build_day_discussion_summary(gs, day=3)
        assert summary == []


class TestVotePressureContext:
    """Wolf rush vote opportunity is optional strategy information."""

    def test_build_vote_pressure_context(self):
        """Context includes vote pressure for strategic voting."""
        from werewolf_agent.core.models import GameState, PlayerState, GameEvent

        players = {f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True) for i in range(1, 13)}
        gs = GameState(
            game_id="test",
            phase="day",
            day_number=2,
            players=players,
            events=[
                GameEvent(type="vote_resolved", payload={"exiled": None, "reason": "first_tie_pk", "tied": ["p03", "p07"]}),
            ],
        )
        ctx = build_vote_pressure_context(gs, voter_id="p01", pk_candidates=["p03", "p07"])
        assert "pk_candidates" in ctx
        assert ctx["pk_candidates"] == ["p03", "p07"]


class TestValidateStructuredVoteAction:
    """Task 2: Relaxed basis detection — no regex basis defaults to fallback."""

    def test_missing_basis_defaults_to_fallback_not_error(self):
        """When basis regex finds nothing, default to fallback basis (no error).

        Regression for Issue 5: 6/6 fallback votes in g_3528592081 stemmed from
        vote_quality / empty_response retries. The strict regex caused LLM
        retries to repeat the same mistake and triggered fallback. We now
        default missing basis to "fallback" and let the vote through.
        """
        from werewolf_agent.runtime.vote_quality import validate_structured_vote_action

        action = {
            "action_type": "vote",
            "target_id": "p07",
            "speech": "我跟p07的票",
            "reason": "我没看出什么明显理由",
            "confidence": 0.5,
            "seer_stance": "undecided",
            "vote_basis": "fallback",
            "standing_with_seer": "",
            "suspect_reason": "p07的发言节奏有点奇怪",
            "not_voting_reason": "其他人证据更弱",
            "private_reason": "保守票",
        }
        result = validate_structured_vote_action(action)
        assert result.get("valid") is True
        assert result.get("vote_basis") in ("fallback", "speech_logic")
        assert result.get("seer_stance") in ("undecided", "no_claim")

    def test_missing_basis_with_empty_seer_stance_defaults_to_no_claim(self):
        """When seer_stance is missing/empty, default to no_claim on no basis."""
        from werewolf_agent.runtime.vote_quality import validate_structured_vote_action

        action = {
            "action_type": "vote",
            "target_id": "p03",
            "speech": "我跟p03的票",
            "reason": "没看出依据",
            "confidence": 0.4,
            "seer_stance": "",
            "vote_basis": "fallback",
            "standing_with_seer": "",
            "suspect_reason": "p03的立场模糊",
            "not_voting_reason": "其他人证据更弱",
            "private_reason": "保守票",
        }
        result = validate_structured_vote_action(action)
        assert result.get("valid") is True
        assert result.get("seer_stance") in ("no_claim", "undecided")


class TestNormalizeMultiBasis:
    """D-14: normalize_vote_basis must rank multi-basis evidence by
    evidentiary weight, not just return the first detector hit."""

    def test_normalize_multi_basis(self) -> None:
        from werewolf_agent.runtime.vote_quality import normalize_vote_basis

        # Multi-basis: seer_check + counterclaim + vote_tally. The
        # strongest evidence (seer_check) should win, NOT the first
        # detector in iteration order.
        result = normalize_vote_basis(["vote_tally", "counterclaim", "seer_check"])
        assert result == "seer_check", (
            f"strongest basis (seer_check) should win; got: {result}"
        )

    def test_normalize_picks_counterclaim_when_no_seer_check(self) -> None:
        from werewolf_agent.runtime.vote_quality import normalize_vote_basis

        result = normalize_vote_basis(["vote_tally", "counterclaim"])
        assert result == "seer_siding"

    def test_normalize_picks_vote_tally_alone(self) -> None:
        from werewolf_agent.runtime.vote_quality import normalize_vote_basis

        result = normalize_vote_basis(["vote_tally"])
        assert result == "vote_pattern"

    def test_normalize_empty_falls_back(self) -> None:
        from werewolf_agent.runtime.vote_quality import normalize_vote_basis

        result = normalize_vote_basis([])
        assert result == "fallback"


class TestVoteFallbackTarget:
    """Fallback vote target selection should use public evidence when possible."""

    def test_vote_fallback_does_not_always_pick_first_legal_target(self):
        from werewolf_agent.core.models import GameEvent, GameState, PlayerState
        from werewolf_agent.runtime.vote_quality import choose_vote_fallback_target

        players = {
            f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True)
            for i in range(1, 5)
        }
        gs = GameState(
            game_id="g_vote",
            day_number=1,
            players=players,
            events=[
                GameEvent(type="speech", payload={
                    "speaker": "p03",
                    "day_number": 1,
                    "text": "我刚才说我是预言家，但警徽流前后矛盾，逻辑不通。",
                }),
            ],
        )

        target = choose_vote_fallback_target(gs, "p01", ["p02", "p03", "p04"])

        assert target == "p03"

    def test_vote_fallback_returns_none_when_evidence_is_required_but_absent(self):
        from werewolf_agent.core.models import GameState, PlayerState
        from werewolf_agent.runtime.vote_quality import choose_vote_fallback_target

        players = {
            f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True)
            for i in range(1, 5)
        }
        gs = GameState(game_id="g_vote_no_evidence", day_number=1, players=players)

        target = choose_vote_fallback_target(
            gs,
            "p01",
            ["p02", "p03", "p04"],
            require_evidence=True,
        )

        assert target is None

    def test_vote_fallback_uses_public_evidence_when_required(self):
        from werewolf_agent.core.models import GameEvent, GameState, PlayerState
        from werewolf_agent.runtime.vote_quality import choose_vote_fallback_target

        players = {
            f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True)
            for i in range(1, 5)
        }
        gs = GameState(
            game_id="g_vote_evidence_required",
            day_number=1,
            players=players,
            events=[
                GameEvent(type="speech", payload={
                    "speaker": "p04",
                    "day_number": 1,
                    "text": "p03发言前后矛盾，逻辑不通，我今天会投p03。",
                }),
            ],
        )

        target = choose_vote_fallback_target(
            gs,
            "p01",
            ["p02", "p03", "p04"],
            require_evidence=True,
        )

        assert target == "p03"
