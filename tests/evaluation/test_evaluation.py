"""Evaluation tests: schemas, metrics, runner, reports.

Covers per §16 test plan:
- Single game result stats
- Multi-game win rates
- Persona/model/RAG comparisons
- Leakage rate
- Illegal action rate
- Growth curves
- Leaderboard JSON output
- Replayability verification
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.engine.rule_engine import RuleEngine, Ruleset
from werewolf_agent.evaluation.metrics import MetricsAggregator
from werewolf_agent.evaluation.reports import ReportGenerator
from werewolf_agent.evaluation.runner import BatchRunner
from werewolf_agent.evaluation.schemas import (
    ActionRecord,
    ActionVerdict,
    BatchConfig,
    CostMetrics,
    CostRecord,
    ExperimentComparison,
    ExperimentDimension,
    FactionMetrics,
    GameResult,
    GrowthPoint,
    LeaderboardEntry,
    LeaderboardReport,
    LeakageRecord,
    MetricsSnapshot,
    PlayerMetrics,
    QualityMetrics,
    ReplayRecord,
    RoleMetrics,
    SafetyMetrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RULESET_PATH = Path(__file__).parent.parent.parent / "config" / "rulesets" / "pre_witch_hunter_idiot_mixed.yaml"


def _make_engine() -> RuleEngine:
    return RuleEngine.from_yaml(RULESET_PATH)


def _make_game_result(
    game_id: str = "test_game_001",
    seed: int = 42,
    winning_faction: str = "good",
    player_roles: dict[str, str] | None = None,
    player_factions: dict[str, str] | None = None,
    deaths: list[dict] | None = None,
    action_records: list[ActionRecord] | None = None,
    leakage_records: list[LeakageRecord] | None = None,
    cost_records: list[CostRecord] | None = None,
    event_log: list[dict] | None = None,
    reviews: list[dict] | None = None,
    cognition_snapshots: dict | None = None,
) -> GameResult:
    if player_roles is None:
        player_roles = {
            "player_01": "werewolf", "player_02": "werewolf",
            "player_03": "werewolf", "player_04": "werewolf",
            "player_05": "villager", "player_06": "villager", "player_07": "villager",
            "player_08": "seer", "player_09": "witch",
            "player_10": "hunter", "player_11": "idiot", "player_12": "hybrid",
        }
    if player_factions is None:
        player_factions = {
            "player_01": "werewolf", "player_02": "werewolf",
            "player_03": "werewolf", "player_04": "werewolf",
            "player_05": "good", "player_06": "good", "player_07": "good",
            "player_08": "good", "player_09": "good",
            "player_10": "good", "player_11": "good", "player_12": "good",
        }
    return GameResult(
        game_id=game_id,
        initial_seed=seed,
        ruleset_id="pre_witch_hunter_idiot_mixed",
        ruleset_snapshot={},
        winning_faction=winning_faction,
        player_roles=player_roles,
        player_factions=player_factions,
        deaths=deaths or [],
        event_log=event_log or [],
        action_records=action_records or [],
        leakage_records=leakage_records or [],
        cost_records=cost_records or [],
        reviews=reviews or [],
        cognition_snapshots=cognition_snapshots or {},
    )


# ===========================================================================
# Schema tests
# ===========================================================================


class TestSchemas:
    def test_action_record_legal(self):
        ar = ActionRecord(player_id="p1", action_type="vote", target_id="p2")
        assert ar.verdict == ActionVerdict.LEGAL

    def test_action_record_illegal(self):
        ar = ActionRecord(player_id="p1", action_type="vote", verdict=ActionVerdict.ILLEGAL, illegal_reason="target already dead")
        assert ar.verdict == ActionVerdict.ILLEGAL
        assert ar.illegal_reason == "target already dead"

    def test_leakage_record(self):
        lr = LeakageRecord(game_id="g1", player_id="p1", leaked_info_type="wolf_kill_target")
        assert lr.leaked_info_type == "wolf_kill_target"

    def test_cost_record(self):
        cr = CostRecord(game_id="g1", player_id="p1", task_type="speech", provider="mock", model="test", estimated_cost=0.01)
        assert cr.estimated_cost == 0.01

    def test_game_result_to_dict(self):
        gr = _make_game_result()
        d = gr.to_dict()
        assert d["game_id"] == "test_game_001"
        assert d["initial_seed"] == 42
        assert d["winning_faction"] == "good"

    def test_batch_config(self):
        bc = BatchConfig(
            batch_id="test_batch",
            num_games=5,
            seed_set=[1, 2, 3, 4, 5],
            experiment_dimension=ExperimentDimension.MODEL,
        )
        assert bc.num_games == 5
        assert bc.experiment_dimension == ExperimentDimension.MODEL

    def test_replay_record_round_trip(self):
        replay = ReplayRecord(
            game_id="g1",
            initial_seed=42,
            ruleset_snapshot={"player_count": 12},
            event_log=[{"type": "test", "payload": {}}],
        )
        d = replay.to_dict()
        restored = ReplayRecord.from_dict(d)
        assert restored.game_id == "g1"
        assert restored.initial_seed == 42
        assert len(restored.event_log) == 1

    def test_faction_metrics_defaults(self):
        fm = FactionMetrics()
        assert fm.good_win_rate == 0.0
        assert fm.total_games == 0

    def test_leaderboard_entry(self):
        le = LeaderboardEntry(rank=1, player_id="p1", overall_score=0.85, win_rate=0.8)
        assert le.rank == 1

    def test_experiment_comparison(self):
        ec = ExperimentComparison(
            dimension="model", label_a="gpt4", label_b="claude",
            metric_name="win_rate", value_a=0.6, value_b=0.7, delta=0.1,
        )
        assert ec.delta == 0.1

    def test_growth_point(self):
        gp = GrowthPoint(game_number=5, metric_name="win_rate", value=0.6)
        assert gp.game_number == 5

    def test_metrics_snapshot_defaults(self):
        ms = MetricsSnapshot(batch_id="test")
        assert ms.total_games == 0
        assert ms.faction_metrics.good_win_rate == 0.0

    def test_leaderboard_report_to_json_dict(self):
        report = LeaderboardReport(
            report_id="r1",
            entries=[LeaderboardEntry(rank=1, player_id="p1")],
            total_games=10,
        )
        d = report.to_json_dict()
        assert d["report_id"] == "r1"
        assert len(d["entries"]) == 1

    def test_experiment_dimension_enum(self):
        assert ExperimentDimension.MODEL == "model"
        assert ExperimentDimension.PERSONA == "persona"
        assert ExperimentDimension.RAG_STRATEGY == "rag_strategy"


# ===========================================================================
# Metrics aggregator tests
# ===========================================================================


class TestMetricsAggregator:
    def test_empty_results(self):
        agg = MetricsAggregator()
        snap = agg.compute_snapshot()
        assert snap.total_games == 0
        assert snap.faction_metrics.good_win_rate == 0.0

    def test_single_game_good_win(self):
        agg = MetricsAggregator()
        agg.add_result(_make_game_result(winning_faction="good"))
        snap = agg.compute_snapshot()
        assert snap.total_games == 1
        assert snap.faction_metrics.good_win_rate == 1.0
        assert snap.faction_metrics.werewolf_win_rate == 0.0

    def test_single_game_wolf_win(self):
        agg = MetricsAggregator()
        agg.add_result(_make_game_result(winning_faction="werewolf"))
        snap = agg.compute_snapshot()
        assert snap.faction_metrics.werewolf_win_rate == 1.0

    def test_multi_game_faction_rates(self):
        agg = MetricsAggregator()
        agg.add_results([
            _make_game_result(game_id="g1", winning_faction="good"),
            _make_game_result(game_id="g2", winning_faction="good"),
            _make_game_result(game_id="g3", winning_faction="werewolf"),
            _make_game_result(game_id="g4", winning_faction="good"),
        ])
        snap = agg.compute_snapshot()
        assert snap.total_games == 4
        assert snap.faction_metrics.good_win_rate == 0.75
        assert snap.faction_metrics.werewolf_win_rate == 0.25

    def test_player_metrics(self):
        agg = MetricsAggregator()
        agg.add_result(_make_game_result(winning_faction="good"))
        snap = agg.compute_snapshot()
        # All good players should have win_rate 1.0, werewolves 0.0
        assert "player_05" in snap.player_metrics
        assert snap.player_metrics["player_05"].win_rate == 1.0
        assert snap.player_metrics["player_01"].win_rate == 0.0

    def test_role_metrics(self):
        agg = MetricsAggregator()
        agg.add_result(_make_game_result(winning_faction="good"))
        snap = agg.compute_snapshot()
        assert "werewolf" in snap.role_metrics
        assert snap.role_metrics["werewolf"].win_rate == 0.0
        assert "villager" in snap.role_metrics
        assert snap.role_metrics["villager"].win_rate == 1.0

    def test_safety_metrics_no_leaks(self):
        agg = MetricsAggregator()
        agg.add_result(_make_game_result())
        snap = agg.compute_snapshot()
        assert snap.safety_metrics.leakage_rate == 0.0
        assert snap.safety_metrics.illegal_action_rate == 0.0

    def test_safety_metrics_with_leaks(self):
        agg = MetricsAggregator()
        result = _make_game_result()
        result.leakage_records.append(LeakageRecord(
            game_id="test_game_001", player_id="player_05",
            leaked_info_type="wolf_kill_target",
        ))
        agg.add_result(result)
        snap = agg.compute_snapshot()
        # 1 leak / 12 players = ~0.083
        assert snap.safety_metrics.leakage_rate == pytest.approx(1 / 12, abs=0.01)

    def test_safety_metrics_with_illegal_actions(self):
        agg = MetricsAggregator()
        result = _make_game_result(action_records=[
            ActionRecord(player_id="p1", action_type="vote", verdict=ActionVerdict.LEGAL),
            ActionRecord(player_id="p2", action_type="vote", verdict=ActionVerdict.ILLEGAL, illegal_reason="dead player"),
            ActionRecord(player_id="p3", action_type="vote", verdict=ActionVerdict.RETRY_RECOVERED),
        ])
        agg.add_result(result)
        snap = agg.compute_snapshot()
        assert snap.safety_metrics.illegal_action_rate == pytest.approx(1 / 3, abs=0.01)
        assert snap.safety_metrics.retry_recovery_rate == pytest.approx(1 / 3, abs=0.01)

    def test_cost_metrics(self):
        agg = MetricsAggregator()
        result = _make_game_result(cost_records=[
            CostRecord(game_id="g1", player_id="p1", task_type="speech", provider="mock", model="test", estimated_cost=0.01, latency_ms=100),
            CostRecord(game_id="g1", player_id="p2", task_type="vote", provider="mock", model="test", estimated_cost=0.02, latency_ms=200),
        ])
        agg.add_result(result)
        snap = agg.compute_snapshot()
        assert snap.cost_metrics.total_cost == 0.03
        assert snap.cost_metrics.avg_cost_per_game == 0.03
        assert snap.cost_metrics.avg_latency_ms == 150

    def test_cost_metrics_by_provider(self):
        agg = MetricsAggregator()
        result = _make_game_result(cost_records=[
            CostRecord(game_id="g1", player_id="p1", task_type="speech", provider="openai", model="gpt4", estimated_cost=0.05),
            CostRecord(game_id="g1", player_id="p2", task_type="speech", provider="local", model="ollama", estimated_cost=0.0),
        ])
        agg.add_result(result)
        snap = agg.compute_snapshot()
        assert "openai" in snap.cost_metrics.by_provider
        assert snap.cost_metrics.by_provider["openai"] == 0.05

    def test_growth_curve_two_games(self):
        agg = MetricsAggregator()
        agg.add_results([
            _make_game_result(game_id="g1", winning_faction="good"),
            _make_game_result(game_id="g2", winning_faction="werewolf"),
        ])
        snap = agg.compute_snapshot()
        assert len(snap.growth_curve) > 0
        # First game: good_win_rate=1.0, second: good_win_rate=0.5
        good_curve = [p for p in snap.growth_curve if p.metric_name == "good_win_rate"]
        assert len(good_curve) == 2
        assert good_curve[0].value == 1.0
        assert good_curve[1].value == 0.5

    def test_quality_metrics_vote_accuracy(self):
        agg = MetricsAggregator()
        result = _make_game_result(
            winning_faction="good",
            deaths=[
                {"player_id": "player_01", "reason": "exile", "timing": "day_vote", "resolution_batch": "day_1"},
            ],
        )
        agg.add_result(result)
        snap = agg.compute_snapshot()
        # One exiled wolf → vote_accuracy = 1/1 = 1.0
        assert snap.quality_metrics.vote_accuracy == 1.0

    def test_quality_metrics_hybrid_co_win(self):
        agg = MetricsAggregator()
        result = _make_game_result(winning_faction="good")
        result.cognition_snapshots = {}
        agg.add_result(result)
        snap = agg.compute_snapshot()
        # Hybrid is good faction, good won → co-win rate should be 1.0
        assert snap.quality_metrics.hybrid_co_win_rate == 1.0

    def test_extract_replay(self):
        agg = MetricsAggregator()
        result = _make_game_result(event_log=[{"type": "test", "payload": {}}])
        agg.add_result(result)
        replay = agg.extract_replay(result)
        assert replay.game_id == "test_game_001"
        assert replay.initial_seed == 42
        assert len(replay.event_log) == 1

    def test_extract_all_replays(self):
        agg = MetricsAggregator()
        agg.add_results([
            _make_game_result(game_id="g1"),
            _make_game_result(game_id="g2"),
        ])
        replays = agg.extract_all_replays()
        assert len(replays) == 2

    def test_compare_snapshots(self):
        agg_a = MetricsAggregator(BatchConfig(batch_id="a"))
        agg_a.add_results([_make_game_result(winning_faction="good")])

        agg_b = MetricsAggregator(BatchConfig(batch_id="b"))
        agg_b.add_results([_make_game_result(winning_faction="werewolf")])

        snap_a = agg_a.compute_snapshot()
        snap_b = agg_b.compute_snapshot()

        comparisons = MetricsAggregator.compare_snapshots(
            snap_a, snap_b,
            dimension="model", label_a="model_a", label_b="model_b",
        )
        assert len(comparisons) > 0
        # good_win_rate: A=1.0, B=0.0, delta=-1.0
        gwr = next(c for c in comparisons if c["metric_name"] == "good_win_rate")
        assert gwr["value_a"] == 1.0
        assert gwr["value_b"] == 0.0

    def test_add_results(self):
        agg = MetricsAggregator()
        agg.add_results([_make_game_result(game_id="g1"), _make_game_result(game_id="g2")])
        assert len(agg.results) == 2


# ===========================================================================
# Batch runner tests
# ===========================================================================


class TestBatchRunner:
    def test_generate_seed_set_explicit(self):
        engine = _make_engine()
        config = BatchConfig(batch_id="test", seed_set=[100, 200, 300], num_games=3)
        runner = BatchRunner(engine, config)
        seeds = runner.generate_seed_set()
        assert seeds == [100, 200, 300]

    def test_generate_seed_set_auto(self):
        engine = _make_engine()
        config = BatchConfig(batch_id="test_auto", num_games=5)
        runner = BatchRunner(engine, config)
        seeds = runner.generate_seed_set()
        assert len(seeds) == 5
        # Deterministic from batch_id
        seeds2 = runner.generate_seed_set()
        assert seeds == seeds2

    def test_run_single_game(self):
        engine = _make_engine()
        config = BatchConfig(batch_id="single", seed_set=[42], num_games=1)
        runner = BatchRunner(engine, config)
        result = runner.run_game(42, game_index=0)
        assert result.game_id == "single_game_0000"
        assert result.initial_seed == 42
        assert result.ruleset_id == "pre_witch_hunter_idiot_mixed"
        assert result.winning_faction in ("good", "werewolf")
        assert len(result.player_roles) == 12

    def test_run_batch(self):
        engine = _make_engine()
        config = BatchConfig(batch_id="batch_test", seed_set=[1, 2, 3], num_games=3)
        runner = BatchRunner(engine, config)
        results = runner.run_batch()
        assert len(results) == 3
        assert all(r.winning_faction in ("good", "werewolf") for r in results)

    def test_game_result_has_ruleset_snapshot(self):
        engine = _make_engine()
        config = BatchConfig(batch_id="snap_test", seed_set=[42], num_games=1)
        runner = BatchRunner(engine, config)
        result = runner.run_game(42)
        assert result.ruleset_snapshot != {}
        assert result.ruleset_snapshot.get("player_count") == 12

    def test_game_result_has_event_log(self):
        engine = _make_engine()
        config = BatchConfig(batch_id="event_test", seed_set=[42], num_games=1)
        runner = BatchRunner(engine, config)
        result = runner.run_game(42)
        assert len(result.event_log) > 0
        event_types = {e["type"] for e in result.event_log}
        assert "hybrid_master_chosen" in event_types
        assert "player_died" in event_types
        assert "victory" in event_types

    def test_hybrid_result_uses_final_winning_faction(self):
        engine = _make_engine()
        config = BatchConfig(batch_id="hybrid_result", seed_set=[42], num_games=1)
        runner = BatchRunner(engine, config)
        result = runner.run_game(42)

        if result.hybrid_master_faction == result.winning_faction:
            assert result.hybrid_result == "win"
        else:
            assert result.hybrid_result == "lose"

    def test_game_result_has_deaths(self):
        engine = _make_engine()
        config = BatchConfig(batch_id="death_test", seed_set=[42], num_games=1)
        runner = BatchRunner(engine, config)
        result = runner.run_game(42)
        # Game should have some deaths unless all wolves die immediately
        assert isinstance(result.deaths, list)

    def test_add_leakage_record(self):
        engine = _make_engine()
        config = BatchConfig(batch_id="leak_test", seed_set=[42], num_games=1)
        runner = BatchRunner(engine, config)
        result = runner.run_game(42)
        runner.add_leakage_record(
            game_id=result.game_id,
            player_id="player_05",
            leaked_info_type="wolf_kill_target",
            detail="accidental reveal in speech",
        )
        assert len(runner.results[0].leakage_records) == 1
        assert runner.results[0].leakage_records[0].leaked_info_type == "wolf_kill_target"

    def test_add_action_record(self):
        engine = _make_engine()
        config = BatchConfig(batch_id="action_test", seed_set=[42], num_games=1)
        runner = BatchRunner(engine, config)
        result = runner.run_game(42)
        runner.add_action_record(
            game_id=result.game_id,
            record=ActionRecord(player_id="p1", action_type="vote", verdict=ActionVerdict.ILLEGAL),
        )
        assert any(a.verdict == ActionVerdict.ILLEGAL for a in runner.results[0].action_records)

    def test_add_cost_record(self):
        engine = _make_engine()
        config = BatchConfig(batch_id="cost_test", seed_set=[42], num_games=1)
        runner = BatchRunner(engine, config)
        result = runner.run_game(42)
        runner.add_cost_record(
            game_id=result.game_id,
            record=CostRecord(game_id=result.game_id, player_id="p1", task_type="speech", provider="mock", model="test", estimated_cost=0.01),
        )
        assert any(c.estimated_cost == 0.01 for c in runner.results[0].cost_records)

    def test_deterministic_replay(self):
        """Same seed produces identical results."""
        engine = _make_engine()
        config = BatchConfig(batch_id="replay_det", seed_set=[42], num_games=1)
        runner = BatchRunner(engine, config)
        r1 = runner.run_game(42)
        r2 = runner.run_game(42)
        assert r1.player_roles == r2.player_roles
        assert r1.winning_faction == r2.winning_faction

    def test_replay_from_event_log(self):
        """Verify replay from initial_seed + ruleset_snapshot + event_log."""
        engine = _make_engine()
        config = BatchConfig(batch_id="replay_verify", seed_set=[42], num_games=1)
        runner = BatchRunner(engine, config)
        result = runner.run_game(42)

        # Extract replay record
        agg = MetricsAggregator()
        replay = agg.extract_replay(result)

        # Replay through the engine
        replayed_state = BatchRunner.verify_replay(replay, engine)

        # The replayed state should match the original game's final state
        # (same players alive/dead, same winning condition derivable)
        original_alive = {pid for pid, p in result.player_roles.items()}
        assert len(replayed_state.players) == 12
        assert replayed_state.winning_faction == result.winning_faction
        assert replayed_state.hybrid_result == result.hybrid_result
        assert len(replayed_state.deaths) == len(result.deaths)
        for replayed_death, recorded_death in zip(replayed_state.deaths, result.deaths):
            assert replayed_death.resolution_batch == recorded_death["resolution_batch"]

    def test_verify_replay_uses_ruleset_snapshot(self):
        engine = _make_engine()
        snapshot = dict(engine.ruleset.raw)
        snapshot["id"] = "custom_ruleset_from_snapshot"
        snapshot["roles"] = dict(snapshot["roles"])
        snapshot["roles"]["werewolf"] = dict(snapshot["roles"]["werewolf"])
        snapshot["roles"]["werewolf"]["count"] = 0
        replay = ReplayRecord(
            game_id="snapshot_replay",
            initial_seed=42,
            ruleset_snapshot=snapshot,
            event_log=[],
        )

        replayed_state = BatchRunner.verify_replay(replay, engine)

        assert all(player.role != "werewolf" for player in replayed_state.players.values())
        assert replayed_state.ruleset_id == "custom_ruleset_from_snapshot"

    def test_evaluation_does_not_mutate_ruleset(self):
        """Verify evaluation never mutates the original ruleset."""
        engine = _make_engine()
        original_raw = dict(engine.ruleset.raw)
        config = BatchConfig(batch_id="no_mutate", seed_set=[42], num_games=1)
        runner = BatchRunner(engine, config)
        runner.run_batch()
        # Ruleset should be unchanged
        assert engine.ruleset.raw == original_raw


# ===========================================================================
# Report generator tests
# ===========================================================================


class TestReportGenerator:
    def test_generate_leaderboard(self):
        gen = ReportGenerator()
        agg = MetricsAggregator(BatchConfig(batch_id="lb_test"))
        agg.add_results([
            _make_game_result(game_id="g1", winning_faction="good"),
            _make_game_result(game_id="g2", winning_faction="werewolf"),
        ])
        snap = agg.compute_snapshot()
        gen.add_snapshot(snap)

        report = gen.generate_leaderboard(
            player_model_map={"player_01": "gpt4"},
            player_persona_map={"player_01": "aggressive"},
        )
        assert report.total_games == 2
        assert len(report.entries) > 0
        # Check model/persona mapping
        p01 = next(e for e in report.entries if e.player_id == "player_01")
        assert p01.model == "gpt4"
        assert p01.persona == "aggressive"

    def test_leaderboard_ranking(self):
        gen = ReportGenerator()
        agg = MetricsAggregator(BatchConfig(batch_id="rank_test"))
        # Good win → good players win, wolves lose
        agg.add_result(_make_game_result(winning_faction="good"))
        snap = agg.compute_snapshot()
        gen.add_snapshot(snap)

        report = gen.generate_leaderboard()
        # Entries should be sorted by overall_score descending
        scores = [e.overall_score for e in report.entries]
        assert scores == sorted(scores, reverse=True)
        # Ranks should be sequential
        ranks = [e.rank for e in report.entries]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_compare_experiments(self):
        gen = ReportGenerator()
        agg_a = MetricsAggregator(BatchConfig(batch_id="exp_a"))
        agg_a.add_results([_make_game_result(winning_faction="good")])
        snap_a = agg_a.compute_snapshot()

        agg_b = MetricsAggregator(BatchConfig(batch_id="exp_b"))
        agg_b.add_results([_make_game_result(winning_faction="werewolf")])
        snap_b = agg_b.compute_snapshot()

        comparisons = gen.compare_experiments(
            snap_a, snap_b,
            dimension="model", label_a="claude", label_b="gpt4",
        )
        assert len(comparisons) > 0
        gwr = next(c for c in comparisons if c.metric_name == "good_win_rate")
        assert gwr.value_a == 1.0
        assert gwr.value_b == 0.0
        assert gwr.delta == -1.0
        assert gwr.dimension == "model"

    def test_build_growth_curves(self):
        gen = ReportGenerator()
        agg = MetricsAggregator(BatchConfig(batch_id="gc"))
        agg.add_results([
            _make_game_result(game_id="g1", winning_faction="good"),
            _make_game_result(game_id="g2", winning_faction="werewolf"),
        ])
        snap = agg.compute_snapshot()

        curves = gen.build_growth_curves([snap], labels=["test_exp"])
        assert len(curves) > 0
        # Should have good_win_rate and werewolf_win_rate curves
        assert any("good_win_rate" in key for key in curves)

    def test_leaderboard_json_round_trip(self):
        gen = ReportGenerator()
        agg = MetricsAggregator(BatchConfig(batch_id="json_test"))
        agg.add_result(_make_game_result())
        snap = agg.compute_snapshot()
        gen.add_snapshot(snap)

        report = gen.generate_leaderboard()
        json_str = ReportGenerator.to_json(report)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["report_id"] == report.report_id
        assert "entries" in parsed

        # Round trip
        restored = ReportGenerator.from_json(json_str)
        assert restored.report_id == report.report_id
        assert len(restored.entries) == len(report.entries)
        assert restored.total_games == report.total_games

    def test_leaderboard_json_has_required_fields(self):
        gen = ReportGenerator()
        agg = MetricsAggregator(BatchConfig(batch_id="fields_test"))
        agg.add_result(_make_game_result())
        snap = agg.compute_snapshot()
        gen.add_snapshot(snap)

        report = gen.generate_leaderboard()
        d = report.to_json_dict()

        # Per §14: leaderboard must include these dimensions
        assert "entries" in d
        if d["entries"]:
            entry = d["entries"][0]
            required = [
                "rank", "player_id", "model", "persona", "overall_score",
                "win_rate", "good_win_rate", "werewolf_win_rate",
                "anti_push_rate", "lie_detection_rate", "stance_accuracy",
                "illegal_action_rate", "avg_cost_per_game", "avg_latency_ms",
                "games_played",
            ]
            for field in required:
                assert field in entry, f"Missing field: {field}"


# ===========================================================================
# Integration: full batch → metrics → leaderboard
# ===========================================================================


class TestEvaluationIntegration:
    def test_full_pipeline(self):
        """End-to-end: batch run → metrics → leaderboard."""
        engine = _make_engine()
        config = BatchConfig(
            batch_id="integration",
            seed_set=[1, 2, 3, 4, 5],
            num_games=5,
            experiment_dimension=ExperimentDimension.MODEL,
            experiment_label="test_model",
        )
        runner = BatchRunner(engine, config)
        results = runner.run_batch()
        assert len(results) == 5

        # Add some leakage and cost records
        for i, result in enumerate(results):
            if i % 2 == 0:
                runner.add_leakage_record(
                    result.game_id, "player_05", "wolf_kill_target",
                    detail="test leak",
                )
            runner.add_cost_record(
                result.game_id,
                CostRecord(
                    game_id=result.game_id,
                    player_id="player_01",
                    task_type="speech",
                    provider="mock",
                    model="test",
                    estimated_cost=0.01,
                    latency_ms=100,
                ),
            )

        # Compute metrics
        agg = MetricsAggregator(config)
        agg.add_results(runner.results)
        snap = agg.compute_snapshot()

        assert snap.total_games == 5
        assert 0.0 <= snap.faction_metrics.good_win_rate <= 1.0
        assert 0.0 <= snap.faction_metrics.werewolf_win_rate <= 1.0
        assert snap.faction_metrics.good_win_rate + snap.faction_metrics.werewolf_win_rate == pytest.approx(1.0)
        assert snap.cost_metrics.total_cost > 0

        # Generate leaderboard
        gen = ReportGenerator()
        gen.add_snapshot(snap)
        report = gen.generate_leaderboard(
            player_model_map={"player_01": "test_model"},
        )
        assert report.total_games == 5

        # Verify JSON output
        json_str = ReportGenerator.to_json(report)
        parsed = json.loads(json_str)
        assert len(parsed["entries"]) > 0

    def test_multi_experiment_comparison(self):
        """Compare two experiments with different seeds."""
        engine = _make_engine()

        # Experiment A
        config_a = BatchConfig(batch_id="exp_a", seed_set=[10, 20, 30], num_games=3)
        runner_a = BatchRunner(engine, config_a)
        results_a = runner_a.run_batch()
        agg_a = MetricsAggregator(config_a)
        agg_a.add_results(results_a)
        snap_a = agg_a.compute_snapshot()

        # Experiment B
        config_b = BatchConfig(batch_id="exp_b", seed_set=[40, 50, 60], num_games=3)
        runner_b = BatchRunner(engine, config_b)
        results_b = runner_b.run_batch()
        agg_b = MetricsAggregator(config_b)
        agg_b.add_results(results_b)
        snap_b = agg_b.compute_snapshot()

        # Compare
        gen = ReportGenerator()
        comparisons = gen.compare_experiments(
            snap_a, snap_b,
            dimension="model", label_a="claude", label_b="gpt4",
        )
        assert len(comparisons) > 0
        for c in comparisons:
            assert c.dimension == "model"
            assert c.label_a == "claude"
            assert c.label_b == "gpt4"

    def test_growth_curve_across_games(self):
        """Growth curves should show metric evolution over sequential games."""
        engine = _make_engine()
        config = BatchConfig(batch_id="growth", seed_set=list(range(1, 11)), num_games=10)
        runner = BatchRunner(engine, config)
        results = runner.run_batch()

        agg = MetricsAggregator(config)
        agg.add_results(results)
        snap = agg.compute_snapshot()

        good_curve = [p for p in snap.growth_curve if p.metric_name == "good_win_rate"]
        assert len(good_curve) == 10
        # Each point should have an increasing game_number
        assert [p.game_number for p in good_curve] == list(range(1, 11))
        # Values should be between 0 and 1
        assert all(0.0 <= p.value <= 1.0 for p in good_curve)

    def test_replay_preserves_player_roles(self):
        """Replaying from initial_seed must produce same role assignment."""
        engine = _make_engine()
        config = BatchConfig(batch_id="replay_roles", seed_set=[42], num_games=1)
        runner = BatchRunner(engine, config)
        result = runner.run_game(42)

        # Replay
        agg = MetricsAggregator()
        replay = agg.extract_replay(result)
        replayed_state = BatchRunner.verify_replay(replay, engine)

        # Role assignment must match
        for pid, p in replayed_state.players.items():
            assert p.role == result.player_roles[pid]

    def test_safety_metrics_across_batch(self):
        """Safety metrics aggregate correctly across batch games."""
        engine = _make_engine()
        config = BatchConfig(batch_id="safety_batch", seed_set=[1, 2, 3], num_games=3)
        runner = BatchRunner(engine, config)
        results = runner.run_batch()

        # Add some illegal actions and leaks
        runner.add_action_record(results[0].game_id, ActionRecord(
            player_id="p1", action_type="vote", verdict=ActionVerdict.ILLEGAL,
        ))
        runner.add_leakage_record(results[1].game_id, "p1", "seer_check")

        agg = MetricsAggregator(config)
        agg.add_results(runner.results)
        snap = agg.compute_snapshot()

        assert snap.safety_metrics.illegal_action_count >= 1
        assert snap.safety_metrics.leakage_count >= 1
        assert snap.safety_metrics.illegal_action_rate > 0.0


# ===========================================================================
# Advanced quality metrics — data-backed computation
# ===========================================================================


class TestAdvancedQualityMetrics:
    """Test that previously-placeholder metrics now compute from real data."""

    def test_lie_detection_from_claims(self):
        """Lie detection rate from claim_role events + cognition snapshots."""
        agg = MetricsAggregator()
        result = _make_game_result(
            winning_faction="good",
            event_log=[
                {"type": "claim_role", "payload": {"player_id": "player_01", "claimed_role": "seer"}},
                {"type": "claim_role", "payload": {"player_id": "player_03", "claimed_role": "witch"}},
            ],
            cognition_snapshots={
                "player_05": {
                    "entries": {
                        "player_01": {"top_role_guess": "werewolf"},
                        "player_03": {"top_role_guess": "werewolf"},
                    },
                },
            },
        )
        agg.add_result(result)
        snap = agg.compute_snapshot()
        # 2 lies (wolf claiming power role), both detected by player_05
        assert snap.quality_metrics.lie_detection_rate == pytest.approx(1.0)

    def test_lie_detection_partial(self):
        """Lie detection rate with partial detection."""
        agg = MetricsAggregator()
        result = _make_game_result(
            winning_faction="good",
            event_log=[
                {"type": "claim_role", "payload": {"player_id": "player_01", "claimed_role": "seer"}},
                {"type": "claim_role", "payload": {"player_id": "player_03", "claimed_role": "witch"}},
            ],
            cognition_snapshots={
                "player_05": {
                    "entries": {
                        "player_01": {"top_role_guess": "werewolf"},
                        "player_03": {"top_role_guess": "villager"},
                    },
                },
            },
        )
        agg.add_result(result)
        snap = agg.compute_snapshot()
        # 2 lies, 1 detected (player_01)
        assert snap.quality_metrics.lie_detection_rate == pytest.approx(0.5)

    def test_stance_accuracy_from_votes(self):
        """Stance accuracy from vote action records — good voters only."""
        agg = MetricsAggregator()
        result = _make_game_result(
            winning_faction="good",
            action_records=[
                ActionRecord(player_id="player_05", action_type="vote", target_id="player_01"),  # good→wolf: correct
                ActionRecord(player_id="player_06", action_type="vote", target_id="player_01"),  # good→wolf: correct
                ActionRecord(player_id="player_07", action_type="vote", target_id="player_05"),  # good→good: incorrect
            ],
        )
        agg.add_result(result)
        snap = agg.compute_snapshot()
        # 2 correct out of 3 good-player votes
        assert snap.quality_metrics.stance_accuracy == pytest.approx(2 / 3, abs=0.01)

    def test_bold_claim_success_rate(self):
        """Bold claim success from claim events + survival/outcome."""
        agg = MetricsAggregator()
        result = _make_game_result(
            winning_faction="werewolf",
            event_log=[
                {"type": "claim_role", "payload": {"player_id": "player_01", "claimed_role": "seer"}},
                {"type": "claim_role", "payload": {"player_id": "player_02", "claimed_role": "seer"}},
            ],
            deaths=[
                {"player_id": "player_01", "reason": "exile", "timing": "day_vote", "resolution_batch": "day_1"},
            ],
        )
        agg.add_result(result)
        snap = agg.compute_snapshot()
        # player_01: bold claim + exiled → failed
        # player_02: bold claim + survived + wolf won → success
        assert 0.0 < snap.quality_metrics.bold_claim_success_rate <= 1.0

    def test_hybrid_master_choice_benefit(self):
        """Hybrid master choice benefit = hybrid co-win fraction."""
        agg = MetricsAggregator()
        r1 = _make_game_result(game_id="g1", winning_faction="good")
        r1.hybrid_result = "win"
        r2 = _make_game_result(game_id="g2", winning_faction="werewolf")
        r2.hybrid_result = "lose"
        r2.player_factions["player_12"] = "good"
        agg.add_results([r1, r2])
        snap = agg.compute_snapshot()
        assert snap.quality_metrics.hybrid_master_choice_benefit == pytest.approx(0.5, abs=0.01)

    def test_witch_potion_benefit(self):
        """Witch potion benefit from antidote/poison events."""
        agg = MetricsAggregator()
        result = _make_game_result(
            winning_faction="good",
            event_log=[
                {"type": "antidote_used", "payload": {"witch_id": "player_09", "target_id": "player_05"}},
                {"type": "poison_used", "payload": {"witch_id": "player_09", "target_id": "player_01"}},
            ],
        )
        agg.add_result(result)
        snap = agg.compute_snapshot()
        # antidote saved good villager → beneficial; poison killed wolf → beneficial
        assert snap.quality_metrics.witch_potion_benefit == pytest.approx(1.0)

    def test_witch_potion_partial(self):
        """Witch potion benefit when some potions are misused."""
        agg = MetricsAggregator()
        result = _make_game_result(
            winning_faction="good",
            event_log=[
                {"type": "antidote_used", "payload": {"witch_id": "player_09", "target_id": "player_01"}},  # saved wolf
                {"type": "poison_used", "payload": {"witch_id": "player_09", "target_id": "player_05"}},  # killed good
            ],
        )
        agg.add_result(result)
        snap = agg.compute_snapshot()
        # Both misused → 0.0
        assert snap.quality_metrics.witch_potion_benefit == pytest.approx(0.0)

    def test_seer_badge_flow_quality(self):
        """Seer badge-flow quality from seer checks + subsequent exile."""
        agg = MetricsAggregator()
        result = _make_game_result(
            winning_faction="good",
            event_log=[
                {"type": "seer_check", "payload": {"seer_id": "player_08", "target_id": "player_01", "alignment": "wolf"}},
            ],
            deaths=[
                {"player_id": "player_01", "reason": "exile", "timing": "day_vote", "resolution_batch": "day_1"},
            ],
        )
        agg.add_result(result)
        snap = agg.compute_snapshot()
        # Seer checked wolf, wolf was exiled → quality > 0
        assert snap.quality_metrics.seer_badge_flow_quality > 0.0

    def test_wolf_consensus_quality(self):
        """Wolf consensus quality: targeting power roles is strategic."""
        agg = MetricsAggregator()
        result = _make_game_result(
            winning_faction="werewolf",
            event_log=[
                {"type": "wolf_kill", "payload": {"target_id": "player_08"}},
            ],
            deaths=[
                {"player_id": "player_08", "reason": "wolf_kill", "timing": "night", "resolution_batch": "night_1"},
            ],
        )
        agg.add_result(result)
        snap = agg.compute_snapshot()
        # Killed seer (power role) → strategic → quality > 0
        assert snap.quality_metrics.wolf_consensus_quality > 0.0

    def test_contradiction_adopted_rate(self):
        """Contradiction adopted rate from reviews."""
        agg = MetricsAggregator()
        result = _make_game_result(
            winning_faction="good",
            reviews=[
                {
                    "contradiction_alerts": [
                        {"player_id": "player_01", "type": "stance_reversal"},
                        {"player_id": "player_02", "type": "claim_conflict"},
                    ],
                    "contradiction_adopted": [
                        {"player_id": "player_01", "alert_type": "stance_reversal"},
                    ],
                },
            ],
        )
        agg.add_result(result)
        snap = agg.compute_snapshot()
        # 2 alerts, 1 adopted → 0.5
        assert snap.quality_metrics.contradiction_adopted_rate == pytest.approx(0.5, abs=0.01)

    def test_badge_decision_quality(self):
        """Badge transfer/tear decision quality from events."""
        agg = MetricsAggregator()
        result = _make_game_result(
            winning_faction="good",
            event_log=[
                {"type": "badge_transfer", "payload": {"from_id": "player_08", "to_id": "player_05"}},
            ],
        )
        agg.add_result(result)
        snap = agg.compute_snapshot()
        assert snap.quality_metrics.badge_decision_quality >= 0.0


# ===========================================================================
# Metric provenance tests
# ===========================================================================


class TestMetricProvenance:
    """Test that every advanced metric has provenance tracking."""

    def test_provenance_keys_cover_quality_metrics(self):
        from werewolf_agent.evaluation.schemas import MetricProvenance
        agg = MetricsAggregator()
        result = _make_game_result(
            winning_faction="good",
            event_log=[
                {"type": "claim_role", "payload": {"player_id": "player_01", "claimed_role": "seer"}},
                {"type": "seer_check", "payload": {"seer_id": "player_08", "target_id": "player_01", "alignment": "wolf"}},
            ],
            action_records=[
                ActionRecord(player_id="player_05", action_type="vote", target_id="player_01"),
            ],
            cognition_snapshots={
                "player_05": {"entries": {"player_01": {"top_role_guess": "werewolf"}}},
            },
        )
        agg.add_result(result)
        snap = agg.compute_snapshot()
        assert isinstance(snap.provenance, dict)
        # Key metrics should have provenance entries
        for key in ("lie_detection_rate", "stance_accuracy", "vote_accuracy"):
            assert key in snap.provenance, f"Missing provenance for {key}"
            p = snap.provenance[key]
            assert isinstance(p, MetricProvenance)
            assert p.source_count > 0
            assert len(p.contributing_games) > 0

    def test_provenance_records_source_types(self):
        agg = MetricsAggregator()
        result = _make_game_result(
            winning_faction="good",
            event_log=[
                {"type": "claim_role", "payload": {"player_id": "player_01", "claimed_role": "seer"}},
            ],
            cognition_snapshots={
                "player_05": {"entries": {"player_01": {"top_role_guess": "werewolf"}}},
            },
        )
        agg.add_result(result)
        snap = agg.compute_snapshot()
        p = snap.provenance.get("lie_detection_rate")
        assert p is not None
        assert "event_log" in p.source_types
        assert "cognition_snapshots" in p.source_types

    def test_provenance_game_ids(self):
        agg = MetricsAggregator()
        agg.add_result(_make_game_result(game_id="g_abc", winning_faction="good"))
        snap = agg.compute_snapshot()
        for p in snap.provenance.values():
            assert "g_abc" in p.contributing_games


# ===========================================================================
# Report export tests
# ===========================================================================


class TestReportExport:
    """Test JSON report export for observer UI."""

    def test_metrics_snapshot_json_export(self):
        agg = MetricsAggregator()
        agg.add_result(_make_game_result())
        snap = agg.compute_snapshot()
        d = snap.to_json_dict()
        assert isinstance(d, dict)
        assert "batch_id" in d
        assert "faction_metrics" in d
        assert "quality_metrics" in d
        assert "safety_metrics" in d
        assert "cost_metrics" in d
        assert "provenance" in d
        # Round-trip via JSON
        json_str = json.dumps(d, ensure_ascii=False)
        parsed = json.loads(json_str)
        assert parsed["batch_id"] == snap.batch_id

    def test_full_evaluation_report(self):
        from werewolf_agent.evaluation.schemas import FullEvaluationReport
        agg = MetricsAggregator(BatchConfig(batch_id="report_test"))
        agg.add_result(_make_game_result())
        snap = agg.compute_snapshot()

        gen = ReportGenerator()
        gen.add_snapshot(snap)
        report = gen.generate_leaderboard()

        full = FullEvaluationReport(
            report_id="full_001",
            batch_id="report_test",
            metrics=snap.to_json_dict(),
            leaderboard=report.to_json_dict(),
        )
        d = full.to_json_dict()
        assert d["report_id"] == "full_001"
        assert "metrics" in d
        assert "leaderboard" in d
        json_str = json.dumps(d, ensure_ascii=False)
        parsed = json.loads(json_str)
        assert parsed["report_id"] == "full_001"

    def test_report_generator_export_full_report(self):
        gen = ReportGenerator()
        agg = MetricsAggregator(BatchConfig(batch_id="export_test"))
        agg.add_results([
            _make_game_result(game_id="g1", winning_faction="good"),
            _make_game_result(game_id="g2", winning_faction="werewolf"),
        ])
        snap = agg.compute_snapshot()
        gen.add_snapshot(snap)

        full_dict = gen.export_full_report(snap)
        assert "report_id" in full_dict
        assert "metrics" in full_dict
        assert "leaderboard" in full_dict
        assert "provenance" in full_dict.get("metrics", {})
        # Must be JSON-serializable
        json_str = json.dumps(full_dict, ensure_ascii=False)
        assert len(json_str) > 0

    def test_compare_snapshots_includes_new_metrics(self):
        agg_a = MetricsAggregator(BatchConfig(batch_id="cmp_a"))
        agg_a.add_result(_make_game_result(
            winning_faction="good",
            event_log=[
                {"type": "claim_role", "payload": {"player_id": "player_01", "claimed_role": "seer"}},
            ],
            cognition_snapshots={
                "player_05": {"entries": {"player_01": {"top_role_guess": "werewolf"}}},
            },
        ))
        agg_b = MetricsAggregator(BatchConfig(batch_id="cmp_b"))
        agg_b.add_result(_make_game_result(
            winning_faction="werewolf",
            event_log=[
                {"type": "claim_role", "payload": {"player_id": "player_01", "claimed_role": "seer"}},
            ],
        ))

        snap_a = agg_a.compute_snapshot()
        snap_b = agg_b.compute_snapshot()
        comparisons = MetricsAggregator.compare_snapshots(snap_a, snap_b)
        metric_names = {c["metric_name"] for c in comparisons}
        # Should include new metrics in comparison
        assert "lie_detection_rate" in metric_names
        assert "stance_accuracy" in metric_names
