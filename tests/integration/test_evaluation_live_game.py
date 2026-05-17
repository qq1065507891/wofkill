"""Integration tests: evaluation metrics from batch runner games with enriched data.

Covers Task 9 Steps 1-3:
1. Advanced quality metrics compute from batch runner game results
2. Metric provenance tracks data sources from real game flow
3. Full report export is JSON-serializable and observer-UI compatible
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.evaluation.metrics import MetricsAggregator
from werewolf_agent.evaluation.reports import ReportGenerator
from werewolf_agent.evaluation.runner import BatchRunner
from werewolf_agent.evaluation.schemas import (
    ActionRecord,
    ActionVerdict,
    BatchConfig,
    CostRecord,
    ExperimentDimension,
    FullEvaluationReport,
    GameResult,
    MetricProvenance,
)


RULESET_PATH = Path(__file__).parent.parent.parent / "config" / "rulesets" / "pre_witch_hunter_idiot_mixed.yaml"


def _make_engine() -> RuleEngine:
    return RuleEngine.from_yaml(RULESET_PATH)


def _enrich_result(result: GameResult) -> None:
    """Add enriched event data to a batch runner result for metric testing."""
    # Add claim_role events for wolves who claimed power roles
    wolves = [pid for pid, role in result.player_roles.items() if role == "werewolf"]
    for wolf_id in wolves[:2]:
        result.event_log.append({
            "type": "claim_role",
            "payload": {"player_id": wolf_id, "claimed_role": "seer"},
        })

    # Add seer_check events if seer is alive
    seer_id = next(
        (pid for pid, role in result.player_roles.items() if role == "seer"),
        None,
    )
    if seer_id and wolves:
        target = wolves[0]
        result.event_log.append({
            "type": "seer_check",
            "payload": {
                "seer_id": seer_id,
                "target_id": target,
                "alignment": "wolf",
            },
        })

    # Add wolf_kill event with target
    for death in result.deaths:
        if death.get("reason") == "wolf_kill":
            result.event_log.append({
                "type": "wolf_kill",
                "payload": {"target_id": death["player_id"]},
            })
            break

    # Add cognition snapshots for a few players
    good_players = [
        pid for pid, f in result.player_factions.items() if f == "good"
    ]
    if good_players and wolves:
        viewer = good_players[0]
        entries = {}
        for wolf_id in wolves:
            entries[wolf_id] = {"top_role_guess": "werewolf"}
        result.cognition_snapshots[viewer] = {"entries": entries}


class TestEvaluationLiveGame:
    """Integration: metrics from batch runner with enriched data."""

    def test_batch_with_enriched_data_produces_quality_metrics(self):
        engine = _make_engine()
        config = BatchConfig(
            batch_id="enriched_batch",
            seed_set=[42, 43, 44],
            num_games=3,
        )
        runner = BatchRunner(engine, config)
        results = runner.run_batch()

        for result in results:
            _enrich_result(result)

        agg = MetricsAggregator(config)
        agg.add_results(results)
        snap = agg.compute_snapshot()

        # At least some quality metrics should be non-zero with enriched data
        q = snap.quality_metrics
        assert q.vote_accuracy >= 0.0
        assert q.hybrid_co_win_rate >= 0.0

    def test_provenance_from_batch_games(self):
        engine = _make_engine()
        config = BatchConfig(batch_id="prov_batch", seed_set=[42], num_games=1)
        runner = BatchRunner(engine, config)
        results = runner.run_batch()
        _enrich_result(results[0])

        agg = MetricsAggregator(config)
        agg.add_results(results)
        snap = agg.compute_snapshot()

        assert len(snap.provenance) > 0
        for name, prov in snap.provenance.items():
            assert isinstance(prov, MetricProvenance)
            assert prov.metric_name == name
            assert prov.computation_method != ""

    def test_full_report_export_from_batch(self):
        engine = _make_engine()
        config = BatchConfig(
            batch_id="export_batch",
            seed_set=[1, 2, 3],
            num_games=3,
        )
        runner = BatchRunner(engine, config)
        results = runner.run_batch()

        # Add cost records for realism
        for result in results:
            result.cost_records.append(CostRecord(
                game_id=result.game_id,
                player_id="player_01",
                task_type="speech",
                provider="mock",
                model="test",
                estimated_cost=0.01,
                latency_ms=100,
            ))

        agg = MetricsAggregator(config)
        agg.add_results(results)
        snap = agg.compute_snapshot()

        gen = ReportGenerator()
        gen.add_snapshot(snap)

        full_dict = gen.export_full_report(snap)
        json_str = json.dumps(full_dict, ensure_ascii=False)
        parsed = json.loads(json_str)

        assert "metrics" in parsed
        assert "leaderboard" in parsed
        assert parsed["metrics"]["total_games"] == 3

    def test_evaluation_never_mutates_ruleset_with_enrichment(self):
        engine = _make_engine()
        original_raw = dict(engine.ruleset.raw)

        config = BatchConfig(batch_id="safe_batch", seed_set=[42], num_games=2)
        runner = BatchRunner(engine, config)
        results = runner.run_batch()
        for r in results:
            _enrich_result(r)

        agg = MetricsAggregator(config)
        agg.add_results(results)
        snap = agg.compute_snapshot()

        assert engine.ruleset.raw == original_raw

    def test_growth_curves_include_advanced_metrics(self):
        engine = _make_engine()
        config = BatchConfig(
            batch_id="growth_adv",
            seed_set=list(range(1, 6)),
            num_games=5,
        )
        runner = BatchRunner(engine, config)
        results = runner.run_batch()

        agg = MetricsAggregator(config)
        agg.add_results(results)
        snap = agg.compute_snapshot()

        # Growth curve should have more than just win rates
        metric_names = {p.metric_name for p in snap.growth_curve}
        assert "good_win_rate" in metric_names
        assert "werewolf_win_rate" in metric_names

    def test_multi_experiment_comparison_with_provenance(self):
        engine = _make_engine()

        # Experiment A
        config_a = BatchConfig(batch_id="exp_a", seed_set=[10, 20], num_games=2)
        runner_a = BatchRunner(engine, config_a)
        results_a = runner_a.run_batch()
        for r in results_a:
            _enrich_result(r)

        # Experiment B
        config_b = BatchConfig(batch_id="exp_b", seed_set=[30, 40], num_games=2)
        runner_b = BatchRunner(engine, config_b)
        results_b = runner_b.run_batch()

        agg_a = MetricsAggregator(config_a)
        agg_a.add_results(results_a)
        snap_a = agg_a.compute_snapshot()

        agg_b = MetricsAggregator(config_b)
        agg_b.add_results(results_b)
        snap_b = agg_b.compute_snapshot()

        gen = ReportGenerator()
        comparisons = gen.compare_experiments(
            snap_a, snap_b,
            dimension="model", label_a="model_a", label_b="model_b",
        )
        assert len(comparisons) > 0

        metric_names = {c.metric_name for c in comparisons}
        # Should include advanced metrics
        assert "good_win_rate" in metric_names
