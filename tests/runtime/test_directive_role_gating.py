"""Tests for role gating on directive builders and adapter handlers.

D-1, D-4, D-10: witch/hunter directive module API and placeholder safety.
"""

from __future__ import annotations

import pytest


class TestWitchDirectiveModule:
    """D-1: witch day-speech must be extracted into a dedicated module."""

    def test_witch_directive_module_exists(self) -> None:
        """werewolf_agent.runtime.directives.witch must expose build_witch_directive()."""
        from werewolf_agent.runtime.directives.witch import build_witch_directive

        result = build_witch_directive(
            gs=None, witch_id="witch01",  # type: ignore[arg-type]
        )
        assert isinstance(result, dict)
        assert "witch_speech_directive" in result

    def test_witch_directive_referenced_from_adapter(self) -> None:
        """agent_adapter must re-export build_witch_directive for backward compat."""
        from werewolf_agent.runtime import agent_adapter

        assert hasattr(agent_adapter, "_build_witch_day_speech_directive")

    def test_witch_directive_uses_evaluate_death_cause_claims(self) -> None:
        """D-7: witch directive should call evaluate_death_cause_claims with witch role
        and include a `witch_death_cause_evaluations` key."""
        from werewolf_agent.core.models import GameEvent, GameState, PlayerState
        from werewolf_agent.runtime.directives.witch import build_witch_directive

        players = {
            "witch": PlayerState(id="witch", role="witch"),
            "v1": PlayerState(id="v1", role="villager"),
            "w1": PlayerState(id="w1", role="werewolf"),
        }
        events = [
            GameEvent(type="speech", payload={
                "speaker": "v1", "text": "我是女巫，我毒了p01", "day_number": 2,
            }),
        ]
        gs = GameState(
            game_id="witch_dcc_test",
            players=players,
            phase="day",
            day_number=2,
            events=events,
            poison_used=True,
        )
        result = build_witch_directive(gs, "witch")
        # Should contain the death cause evaluations as a strategy hint
        assert "witch_death_cause_evaluations" in result
        # And at least one evaluation should mention the poison claim
        evals = result["witch_death_cause_evaluations"]
        assert isinstance(evals, list)
        assert any("p01" in e for e in evals)


class TestHunterDirectiveReturnsDict:
    """D-4: build_hunter_directive must return dict[str, Any], not str."""

    def test_hunter_directive_returns_dict(self) -> None:
        from werewolf_agent.core.models import GameState, PlayerState
        from werewolf_agent.runtime.directives.hunter import build_hunter_directive

        gs = GameState(
            game_id="hunter_dict_test",
            players={
                "hunter": PlayerState(id="hunter", role="hunter"),
                "v1": PlayerState(id="v1", role="villager"),
            },
            phase="day",
            day_number=2,
        )
        result = build_hunter_directive(gs, "hunter")
        assert isinstance(result, dict)
        # The directive text is exposed under hunter_speech_directive
        assert "hunter_speech_directive" in result
        assert isinstance(result["hunter_speech_directive"], str)
        # Pre-fix returned a bare str; new API wraps in dict
        assert "不要暴露" in result["hunter_speech_directive"] or "隐藏" in result["hunter_speech_directive"]


class TestWolfDirectiveNoUnfilledPlaceholders:
    """D-10: wolf_universal_rules must not contain raw `{fake_seer}` placeholders."""

    def test_wolf_directive_no_unfilled_placeholders(self) -> None:
        from werewolf_agent.core.models import GameState, PlayerState
        from werewolf_agent.runtime.directives.wolf import build_wolf_directive

        players = {
            "w1": PlayerState(id="w1", role="werewolf"),
            "w2": PlayerState(id="w2", role="werewolf"),
            "seer": PlayerState(id="seer", role="seer"),
            "v1": PlayerState(id="v1", role="villager"),
        }
        events = [
            GameEvent_p := __import__(
                "werewolf_agent.core.models", fromlist=["GameEvent"]
            ).GameEvent(
                type="speech",
                payload={"speaker": "w2", "text": "我是预言家，查杀了v1", "day_number": 1},
            ),
        ]
        gs = GameState(
            game_id="wolf_placeholder_test",
            players=players,
            phase="day",
            day_number=2,
            events=events,
        )
        plan = {"fake_seer": "w2", "pusher": "w1"}
        result = build_wolf_directive(gs, "w1", plan)
        rules = result["wolf_universal_rules"]
        # After fix, no raw Python format placeholders should leak into the rendered string
        assert "{fake_seer}" not in rules
        # The teammate name must be substituted in
        assert "w2" in rules


class TestDeathCauseLabelD5:
    """D-5: death-cause evaluation label is `[公开判断]` (not `[需判断]`)
    for villager / hunter / idiot / hybrid.  Villager branch must
    explicitly say the player has no private info."""

    def test_villager_death_cause_uses_public_label(self) -> None:
        from werewolf_agent.core.models import GameEvent, GameState, PlayerState
        from werewolf_agent.runtime.strategy.death import evaluate_death_cause_claims

        players = {
            "v1": PlayerState(id="v1", role="villager"),
            "p01": PlayerState(id="p01", role="werewolf"),
            "witch": PlayerState(id="witch", role="witch"),
        }
        events = [
            GameEvent(type="speech", payload={
                "speaker": "witch", "text": "我毒了p01", "day_number": 2,
            }),
        ]
        gs = GameState(
            game_id="villager_death_label_test",
            players=players,
            phase="day",
            day_number=2,
            events=events,
            poison_used=True,
        )
        evals = evaluate_death_cause_claims(gs, "v1", "villager")
        assert evals
        # Pre-fix used `[需判断]`; new label is `[公开判断]`.
        assert any("[公开判断]" in e for e in evals)
        # Villager branch must explicitly say no private info.
        joined = " ".join(evals)
        assert "无" in joined and ("信息" in joined or "私有" in joined)


class TestNegationExcludedFromClaimedSeerD6:
    """D-6: denial-of-seer claims must NOT count as a public seer claim."""

    def test_negation_excluded_from_claimed_seer(self) -> None:
        from werewolf_agent.core.models import GameEvent, GameState, PlayerState
        from werewolf_agent.runtime.strategy.wolf import (
            has_publicly_claimed_seer,
            evaluate_wolf_kill_target,
        )

        players = {
            "p01": PlayerState(id="p01", role="werewolf"),
            "p02": PlayerState(id="p02", role="werewolf"),
            "p03": PlayerState(id="p03", role="werewolf"),
            "p04": PlayerState(id="p04", role="werewolf"),
            "v1": PlayerState(id="v1", role="villager"),
            "seer": PlayerState(id="seer", role="seer"),
        }
        events = [
            GameEvent(type="speech", payload={
                "speaker": "v1", "text": "我不是预言家", "day_number": 1,
            }),
        ]
        gs = GameState(
            game_id="negation_test",
            players=players,
            phase="day",
            day_number=2,
            events=events,
        )
        # v1 explicitly DENIED being seer — must NOT count as a claim.
        assert has_publicly_claimed_seer(gs, "v1") is False

        # And the wolf kill scorer must not score v1 as claimed_seer.
        result = evaluate_wolf_kill_target(gs, "p01", ["v1", "seer"])
        assert result is not None
        v1_entry = next(
            (t for t in result["ranked_targets"] if t["target"] == "v1"), None
        )
        assert v1_entry is not None
        assert "claimed_seer" not in v1_entry["signals"], (
            "negated seer claim must not score v1 as claimed_seer; "
            f"got signals: {v1_entry['signals']}"
        )

    def test_third_party_recap_excluded_from_claimed_seer(self) -> None:
        from werewolf_agent.core.models import GameEvent, GameState, PlayerState
        from werewolf_agent.runtime.strategy.seer import public_seer_claimants
        from werewolf_agent.runtime.strategy.wolf import has_publicly_claimed_seer

        players = {
            "p01": PlayerState(id="p01", role="villager"),
            "p02": PlayerState(id="p02", role="seer"),
            "p09": PlayerState(id="p09", role="werewolf"),
            "p11": PlayerState(id="p11", role="villager"),
        }
        gs = GameState(
            game_id="third_party_seer_recap_test",
            players=players,
            phase="day",
            day_number=2,
            events=[
                GameEvent(type="speech", payload={
                    "speaker": "p02",
                    "text": "我是预言家，昨晚查了p01是好人。",
                    "day_number": 1,
                }),
                GameEvent(type="speech", payload={
                    "speaker": "p11",
                    "text": "p02报p01金水，p09悍跳预言家，我会继续对比。",
                    "day_number": 2,
                }),
            ],
        )

        assert has_publicly_claimed_seer(gs, "p11") is False
        assert public_seer_claimants(gs) == {"p02"}

    def test_affirmative_seer_claim_still_counts(self) -> None:
        """Sanity guard: an affirmative claim still triggers claimed_seer."""
        from werewolf_agent.core.models import GameEvent, GameState, PlayerState
        from werewolf_agent.runtime.strategy.wolf import has_publicly_claimed_seer

        players = {
            "seer": PlayerState(id="seer", role="seer"),
            "v1": PlayerState(id="v1", role="villager"),
        }
        events = [
            GameEvent(type="speech", payload={
                "speaker": "seer", "text": "我是预言家，昨晚查了v1是好人", "day_number": 1,
            }),
        ]
        gs = GameState(
            game_id="affirm_test",
            players=players,
            phase="day",
            day_number=2,
            events=events,
        )
        assert has_publicly_claimed_seer(gs, "seer") is True


class TestStrategyDirectiveTokenCapD9:
    """D-9: _MAX_STRATEGY_DIRECTIVE_TOKENS caps the merged strategy_directive
    size; when the cap is exceeded, the oldest round-specific blocks
    are dropped first (structural / role-critical keys are preserved)."""

    def test_strategy_directive_within_token_cap(self) -> None:
        from werewolf_agent.runtime.context import (
            _cap_strategy_directive,
            _MAX_STRATEGY_DIRECTIVE_TOKENS,
        )

        # Build a directive that exceeds the cap by a wide margin
        directive: dict = {
            "seer_speech_directive": "structural — must be kept",
            "wolf_speech_directive": "structural — must be kept",
            "must_address_alerts": ["must survive"],
            "role_alerts": ["must survive"],
            "vote_pressure": "must vote",
        }
        # Add round-specific blocks whose total size > cap.
        big_text = "X" * (_MAX_STRATEGY_DIRECTIVE_TOKENS * 2)
        for k in (
            "sheriff_election_record",
            "day_discussion_summary",
            "vote_pressure_context",
            "skill_tactical_advice",
            "death_cause_evaluation",
        ):
            directive[k] = big_text

        capped = _cap_strategy_directive(directive)
        # Cap must be respected.
        size = sum(len(str(v)) for v in capped.values()) // 2
        assert size <= _MAX_STRATEGY_DIRECTIVE_TOKENS, (
            f"capped directive exceeds token cap: {size} > {_MAX_STRATEGY_DIRECTIVE_TOKENS}"
        )
        # Structural keys must survive.
        assert "seer_speech_directive" in capped
        assert "wolf_speech_directive" in capped
        assert "must_address_alerts" in capped
        assert "role_alerts" in capped
        assert "vote_pressure" in capped
        # Round-specific blocks must be dropped.
        assert "sheriff_election_record" not in capped
        assert "day_discussion_summary" not in capped

    def test_strategy_directive_under_cap_is_unchanged(self) -> None:
        from werewolf_agent.runtime.context import _cap_strategy_directive

        small = {"seer_speech_directive": "tiny"}
        assert _cap_strategy_directive(small) is small or (
            _cap_strategy_directive(small) == small
        )

    def test_unknown_current_turn_keys_survive_before_reference_noise(self) -> None:
        from werewolf_agent.runtime.context import _cap_strategy_directive

        directive: dict = {
            "seer_speech_directive": "structural — must be kept",
            "wolf_speech_directive": "structural — must be kept",
            "gold_water_duty": "current-turn role fact " * 200,
            "unreported_checks": "current-turn role fact " * 200,
            "my_check_history": "current-turn role fact " * 200,
            "day_discussion_summary": "reference noise " * 1200,
            "vote_pressure_context": "reference noise " * 1200,
        }

        capped = _cap_strategy_directive(directive)

        assert "seer_speech_directive" in capped
        assert "wolf_speech_directive" in capped
        assert "gold_water_duty" in capped
        assert "unreported_checks" in capped
        assert "my_check_history" in capped
        assert "day_discussion_summary" not in capped or "vote_pressure_context" not in capped
