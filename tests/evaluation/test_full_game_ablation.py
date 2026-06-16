from __future__ import annotations

import pytest

from werewolf_agent.evaluation.schemas import GameResult


def _config(**overrides):
    from werewolf_agent.evaluation.full_game_ablation import FullGameAblationConfig

    values = {
        "batch_id": "b1",
        "seed_set": [1, 2],
        "removed_modules": ["rag"],
        "player_count": 12,
        "ruleset_id": "pre_witch_hunter_idiot_mixed",
        "ruleset_snapshot": {"id": "rules"},
        "agent_mode": "deterministic_fake",
        "model_config_snapshot": {"provider": "fake"},
        "baseline_storage_namespace": "baseline",
        "ablated_storage_namespace": "ablated",
        "replay_policy": "deterministic_fallback_only",
    }
    values.update(overrides)
    return FullGameAblationConfig(**values)


def _result(game_id: str, *, winning_faction: str, illegal_actions: int = 0) -> GameResult:
    events = [
        {"type": "illegal_action", "payload": {"i": i}}
        for i in range(illegal_actions)
    ]
    return GameResult(
        game_id=game_id,
        initial_seed=0,
        ruleset_id="pre_witch_hunter_idiot_mixed",
        player_roles={"p01": "villager"},
        player_factions={"p01": "good"},
        event_log=events,
        winning_faction=winning_faction,
    )


def test_full_game_config_includes_replay_and_namespace_fields() -> None:
    config = _config(
        candidate_storage_namespace="candidate_draft",
        replay_capture_ref="captures/b1.json",
        replay_match_key="trace_id",
    )

    assert config.baseline_storage_namespace == "baseline"
    assert config.ablated_storage_namespace == "ablated"
    assert config.candidate_storage_namespace == "candidate_draft"
    assert config.replay_capture_ref == "captures/b1.json"
    assert config.replay_match_key == "trace_id"


def test_live_model_without_replay_reports_unsupported_causal_metrics() -> None:
    from werewolf_agent.evaluation.full_game_ablation import FullGameAblationRunner

    runner = FullGameAblationRunner(game_runner_factory=None)
    report = runner.run(_config(
        seed_set=[1],
        agent_mode="live_model",
        replay_policy="unsupported_live_model",
    ))

    assert report.mode == "full_game"
    assert report.agent_mode == "live_model"
    assert report.pair_count == 0
    assert report.metric_deltas == {}
    assert report.unsupported_metrics["live_win_rate_delta"] == "fresh_live_model_without_replay"
    assert report.unsupported_metrics["causal_decision_delta"] == "fresh_live_model_without_replay"


def test_deterministic_fake_report_uses_baseline_minus_ablated_deltas() -> None:
    from werewolf_agent.evaluation.full_game_ablation import FullGameAblationRunner

    calls: list[dict] = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        namespace = kwargs["storage_namespace"]
        seed = kwargs["seed"]
        if namespace == "baseline":
            return _result(f"baseline_{seed}", winning_faction="good", illegal_actions=1)
        return _result(f"ablated_{seed}", winning_faction="werewolf", illegal_actions=3)

    report = FullGameAblationRunner(game_runner_factory=fake_runner).run(_config(seed_set=[7]))

    assert report.pair_count == 1
    assert report.metric_deltas["good_win_rate"].baseline == 1.0
    assert report.metric_deltas["good_win_rate"].ablated == 0.0
    assert report.metric_deltas["good_win_rate"].delta == 1.0
    assert report.metric_deltas["illegal_action_rate"].baseline == 1.0
    assert report.metric_deltas["illegal_action_rate"].ablated == 3.0
    assert report.metric_deltas["illegal_action_rate"].delta == -2.0
    assert calls[0]["removed_modules"] == []
    assert calls[1]["removed_modules"] == ["rag"]
    assert calls[0]["ruleset_snapshot"] == {"id": "rules"}
    assert calls[1]["mode"] == "deterministic_fake"


def test_replay_matcher_missing_trace_id_fails_closed() -> None:
    from werewolf_agent.evaluation.replay import ReplayArtifact, ReplayMatcher, ReplayRecord

    matcher = ReplayMatcher(ReplayArtifact(records=[
        ReplayRecord(trace_id="t1", output={"action_type": "vote"}, event_index=0),
    ]))

    assert matcher.match("missing", event_index=1, match_key="trace_id") is None
    assert matcher.unsupported_reason == "missing_replay_output"


def test_replay_matcher_event_order_requires_exact_order() -> None:
    from werewolf_agent.evaluation.replay import ReplayArtifact, ReplayMatcher, ReplayRecord

    matcher = ReplayMatcher(ReplayArtifact(records=[
        ReplayRecord(trace_id="t1", output={"action_type": "vote"}, event_index=0),
    ]))

    assert matcher.match("ignored", event_index=1, match_key="event_order") is None
    assert matcher.unsupported_reason == "event_order_mismatch"


def test_full_game_ablation_report_rejects_fresh_replay_fallback() -> None:
    from werewolf_agent.evaluation.full_game_ablation import FullGameAblationRunner
    from werewolf_agent.evaluation.replay import ReplayArtifact, ReplayRecord

    report = FullGameAblationRunner(
        game_runner_factory=lambda **kwargs: pytest.fail("must not call fresh runner"),
        replay_artifact=ReplayArtifact(records=[
            ReplayRecord(trace_id="other", output={"action_type": "vote"}, event_index=0),
        ]),
    ).run(_config(
        seed_set=[1],
        agent_mode="replay",
        replay_policy="strict_replay",
        replay_capture_ref="capture.json",
    ))

    assert report.pair_count == 0
    assert report.unsupported_metrics["replay"] == "missing_replay_output"
