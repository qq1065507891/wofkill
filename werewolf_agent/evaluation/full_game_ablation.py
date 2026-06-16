"""Reproducible full-game ablation contracts and runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from werewolf_agent.evaluation.schemas import GameResult
from werewolf_agent.evaluation.replay import ReplayArtifact, ReplayMatcher

AgentMode = Literal["deterministic_fake", "replay", "live_model"]
ReplayPolicy = Literal[
    "strict_replay",
    "deterministic_fallback_only",
    "unsupported_live_model",
]
ReplayMatchKey = Literal["trace_id", "event_order"]


@dataclass(frozen=True)
class FullGameAblationConfig:
    batch_id: str
    seed_set: list[int]
    removed_modules: list[str]
    player_count: int
    ruleset_id: str
    ruleset_snapshot: dict[str, Any]
    agent_mode: AgentMode
    model_config_snapshot: dict[str, Any]
    baseline_storage_namespace: str
    ablated_storage_namespace: str
    candidate_storage_namespace: str = ""
    replay_policy: ReplayPolicy = "deterministic_fallback_only"
    replay_capture_ref: str = ""
    replay_match_key: ReplayMatchKey = "trace_id"


@dataclass(frozen=True)
class FullGameMetricDelta:
    metric: str
    baseline: float
    ablated: float
    delta: float


@dataclass(frozen=True)
class FullGameAblationPair:
    seed: int
    baseline_game_id: str
    ablated_game_id: str
    baseline_metrics: dict[str, float]
    ablated_metrics: dict[str, float]


@dataclass(frozen=True)
class FullGameAblationReport:
    batch_id: str
    mode: str
    agent_mode: str
    removed_modules: list[str]
    pair_count: int
    metric_deltas: dict[str, FullGameMetricDelta]
    unsupported_metrics: dict[str, str]
    pairs: list[FullGameAblationPair] = field(default_factory=list)


class FullGameAblationRunner:
    def __init__(
        self,
        game_runner_factory: Callable[..., GameResult] | None = None,
        *,
        replay_artifact: ReplayArtifact | None = None,
    ) -> None:
        self._game_runner_factory = game_runner_factory
        self._replay_artifact = replay_artifact

    def run(self, config: FullGameAblationConfig) -> FullGameAblationReport:
        if config.agent_mode == "live_model" and not config.replay_capture_ref:
            return _unsupported_live_model_report(config)

        if config.agent_mode == "replay":
            unsupported = self._validate_replay(config)
            if unsupported:
                return FullGameAblationReport(
                    batch_id=config.batch_id,
                    mode="full_game",
                    agent_mode=config.agent_mode,
                    removed_modules=list(config.removed_modules),
                    pair_count=0,
                    metric_deltas={},
                    unsupported_metrics={"replay": unsupported},
                    pairs=[],
                )

        if self._game_runner_factory is None:
            return FullGameAblationReport(
                batch_id=config.batch_id,
                mode="full_game",
                agent_mode=config.agent_mode,
                removed_modules=list(config.removed_modules),
                pair_count=0,
                metric_deltas={},
                unsupported_metrics={"runner": "game_runner_factory_required"},
                pairs=[],
            )

        pairs: list[FullGameAblationPair] = []
        for seed in config.seed_set:
            baseline = self._run_game(
                config,
                seed=seed,
                storage_namespace=config.baseline_storage_namespace,
                removed_modules=[],
            )
            ablated = self._run_game(
                config,
                seed=seed,
                storage_namespace=config.ablated_storage_namespace,
                removed_modules=config.removed_modules,
            )
            pairs.append(FullGameAblationPair(
                seed=seed,
                baseline_game_id=baseline.game_id,
                ablated_game_id=ablated.game_id,
                baseline_metrics=_game_metrics(baseline),
                ablated_metrics=_game_metrics(ablated),
            ))

        return FullGameAblationReport(
            batch_id=config.batch_id,
            mode="full_game",
            agent_mode=config.agent_mode,
            removed_modules=list(config.removed_modules),
            pair_count=len(pairs),
            metric_deltas=_metric_deltas(pairs),
            unsupported_metrics={},
            pairs=pairs,
        )

    def _run_game(
        self,
        config: FullGameAblationConfig,
        *,
        seed: int,
        storage_namespace: str,
        removed_modules: list[str],
    ) -> GameResult:
        assert self._game_runner_factory is not None
        return self._game_runner_factory(
            seed=seed,
            storage_namespace=storage_namespace,
            removed_modules=list(removed_modules),
            ruleset_snapshot=dict(config.ruleset_snapshot),
            ruleset_id=config.ruleset_id,
            player_count=config.player_count,
            mode=config.agent_mode,
            model_config_snapshot=dict(config.model_config_snapshot),
        )

    def _validate_replay(self, config: FullGameAblationConfig) -> str:
        if self._replay_artifact is None:
            return "missing_replay_capture"
        matcher = ReplayMatcher(self._replay_artifact)
        if config.replay_match_key == "trace_id":
            matcher.match(
                f"{config.batch_id}:seed:{config.seed_set[0] if config.seed_set else 0}",
                event_index=0,
                match_key="trace_id",
            )
        else:
            matcher.match("", event_index=0, match_key="event_order")
        return matcher.unsupported_reason


def _unsupported_live_model_report(
    config: FullGameAblationConfig,
) -> FullGameAblationReport:
    return FullGameAblationReport(
        batch_id=config.batch_id,
        mode="full_game",
        agent_mode=config.agent_mode,
        removed_modules=list(config.removed_modules),
        pair_count=0,
        metric_deltas={},
        unsupported_metrics={
            "live_win_rate_delta": "fresh_live_model_without_replay",
            "causal_decision_delta": "fresh_live_model_without_replay",
        },
        pairs=[],
    )


def _game_metrics(result: GameResult) -> dict[str, float]:
    return {
        "good_win_rate": 1.0 if result.winning_faction == "good" else 0.0,
        "werewolf_win_rate": 1.0 if result.winning_faction == "werewolf" else 0.0,
        "illegal_action_rate": float(
            sum(1 for event in result.event_log if _event_type(event) == "illegal_action")
        ),
    }


def _metric_deltas(
    pairs: list[FullGameAblationPair],
) -> dict[str, FullGameMetricDelta]:
    if not pairs:
        return {}
    metrics = sorted({
        metric
        for pair in pairs
        for metric in set(pair.baseline_metrics) | set(pair.ablated_metrics)
    })
    result: dict[str, FullGameMetricDelta] = {}
    for metric in metrics:
        baseline = sum(pair.baseline_metrics.get(metric, 0.0) for pair in pairs) / len(pairs)
        ablated = sum(pair.ablated_metrics.get(metric, 0.0) for pair in pairs) / len(pairs)
        result[metric] = FullGameMetricDelta(
            metric=metric,
            baseline=baseline,
            ablated=ablated,
            delta=baseline - ablated,
        )
    return result


def _event_type(event: Any) -> str:
    if isinstance(event, dict):
        return str(event.get("type") or "")
    return str(getattr(event, "type", "") or "")
