"""Evaluation schemas: structured types for metrics, batch runs, and reports.

Design doc §14 defines all evaluation metrics and experiment dimensions.
Every game result is replayable from initial_seed + ruleset_snapshot + event_log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Single action record — per-action tracking within a game
# ---------------------------------------------------------------------------

class ActionVerdict(str, Enum):
    LEGAL = "legal"
    ILLEGAL = "illegal"
    RETRY_RECOVERED = "retry_recovered"
    FALBACK = "fallback"


@dataclass(frozen=True)
class ActionRecord:
    player_id: str
    action_type: str
    target_id: str | None = None
    verdict: ActionVerdict = ActionVerdict.LEGAL
    phase: str = ""
    day_number: int = 0
    night_number: int = 0
    illegal_reason: str | None = None


# ---------------------------------------------------------------------------
# Leakage record — information boundary violation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LeakageRecord:
    game_id: str
    player_id: str
    leaked_info_type: str
    leaked_to: str = ""
    phase: str = ""
    day_number: int = 0
    detail: str = ""


# ---------------------------------------------------------------------------
# Cost / latency record — per-call LLM usage
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CostRecord:
    game_id: str
    player_id: str
    task_type: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    estimated_cost: float = 0.0
    fallback: bool = False


# ---------------------------------------------------------------------------
# Game result — outcome of a single completed game
# ---------------------------------------------------------------------------

@dataclass
class GameResult:
    game_id: str
    initial_seed: int
    ruleset_id: str
    ruleset_snapshot: dict[str, Any] = field(default_factory=dict)
    winning_faction: str | None = None
    hybrid_master_id: str | None = None
    hybrid_master_faction: str | None = None
    hybrid_result: str | None = None
    victory_reason: str | None = None
    total_days: int = 0
    total_nights: int = 0
    player_roles: dict[str, str] = field(default_factory=dict)
    player_factions: dict[str, str] = field(default_factory=dict)
    deaths: list[dict[str, Any]] = field(default_factory=list)
    event_log: list[dict[str, Any]] = field(default_factory=list)
    action_records: list[ActionRecord] = field(default_factory=list)
    leakage_records: list[LeakageRecord] = field(default_factory=list)
    cost_records: list[CostRecord] = field(default_factory=list)
    reviews: list[dict[str, Any]] = field(default_factory=list)
    cognition_snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Experiment metadata
    persona_config_snapshot: dict[str, Any] = field(default_factory=dict)
    model_config_snapshot: dict[str, Any] = field(default_factory=dict)
    rag_config_snapshot: dict[str, Any] = field(default_factory=dict)
    strategy_config_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "initial_seed": self.initial_seed,
            "ruleset_id": self.ruleset_id,
            "ruleset_snapshot": dict(self.ruleset_snapshot),
            "winning_faction": self.winning_faction,
            "hybrid_master_id": self.hybrid_master_id,
            "hybrid_master_faction": self.hybrid_master_faction,
            "hybrid_result": self.hybrid_result,
            "victory_reason": self.victory_reason,
            "total_days": self.total_days,
            "total_nights": self.total_nights,
            "player_roles": dict(self.player_roles),
            "player_factions": dict(self.player_factions),
            "deaths": list(self.deaths),
            "event_log": list(self.event_log),
            "persona_config_snapshot": dict(self.persona_config_snapshot),
            "model_config_snapshot": dict(self.model_config_snapshot),
            "rag_config_snapshot": dict(self.rag_config_snapshot),
            "strategy_config_snapshot": dict(self.strategy_config_snapshot),
        }


# ---------------------------------------------------------------------------
# Batch configuration — how to run a set of games
# ---------------------------------------------------------------------------

class ExperimentDimension(str, Enum):
    MODEL = "model"
    PERSONA = "persona"
    RAG_STRATEGY = "rag_strategy"
    COGNITION_PIPELINE = "cognition_pipeline"
    SALIENCE_STRATEGY = "salience_strategy"
    MEMORY_STRATEGY = "memory_strategy"
    TASK_ROUTING = "task_routing"
    PERSONA_ROUTING = "persona_routing"


@dataclass(frozen=True)
class BatchConfig:
    batch_id: str
    ruleset_id: str = "pre_witch_hunter_idiot_mixed"
    seed_set: list[int] = field(default_factory=list)
    num_games: int = 10
    experiment_dimension: ExperimentDimension = ExperimentDimension.MODEL
    experiment_label: str = ""
    persona_config_path: str = ""
    model_config_path: str = ""
    rag_config_path: str = ""
    strategy_config_path: str = ""
    player_count: int = 12
    # Metadata for reproducibility
    description: str = ""


# ---------------------------------------------------------------------------
# Metrics snapshot — aggregated statistics over a set of games
# ---------------------------------------------------------------------------

@dataclass
class FactionMetrics:
    good_win_rate: float = 0.0
    werewolf_win_rate: float = 0.0
    good_wins: int = 0
    werewolf_wins: int = 0
    total_games: int = 0


@dataclass
class RoleMetrics:
    role: str = ""
    win_rate: float = 0.0
    games: int = 0
    wins: int = 0


@dataclass
class PlayerMetrics:
    player_id: str = ""
    win_rate: float = 0.0
    games: int = 0
    wins: int = 0
    role_metrics: dict[str, RoleMetrics] = field(default_factory=dict)


@dataclass
class QualityMetrics:
    anti_push_rate: float = 0.0
    lie_detection_rate: float = 0.0
    stance_accuracy: float = 0.0
    vote_accuracy: float = 0.0
    identity_disguise_rate: float = 0.0
    bold_claim_success_rate: float = 0.0
    deep_hook_benefit: float = 0.0
    hybrid_master_choice_benefit: float = 0.0
    hybrid_co_win_rate: float = 0.0
    witch_potion_benefit: float = 0.0
    seer_badge_flow_quality: float = 0.0
    badge_decision_quality: float = 0.0
    speech_order_utilization: float = 0.0
    wolf_consensus_quality: float = 0.0
    contradiction_hit_rate: float = 0.0
    contradiction_adopted_rate: float = 0.0
    speech_influence_rate: float = 0.0
    cognitive_compression_rate: float = 0.0


@dataclass
class SafetyMetrics:
    leakage_rate: float = 0.0
    leakage_count: int = 0
    illegal_action_rate: float = 0.0
    illegal_action_count: int = 0
    retry_recovery_rate: float = 0.0
    fallback_rate: float = 0.0


@dataclass
class CostMetrics:
    total_cost: float = 0.0
    avg_cost_per_game: float = 0.0
    avg_cost_per_player: float = 0.0
    avg_latency_ms: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    by_provider: dict[str, float] = field(default_factory=dict)
    by_task_type: dict[str, float] = field(default_factory=dict)
    by_player: dict[str, float] = field(default_factory=dict)


@dataclass
class GrowthPoint:
    game_number: int
    metric_name: str
    value: float


@dataclass
class MetricProvenance:
    metric_name: str
    computation_method: str
    source_types: list[str] = field(default_factory=list)
    source_count: int = 0
    contributing_games: list[str] = field(default_factory=list)
    sample_entries: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class MetricsSnapshot:
    batch_id: str
    faction_metrics: FactionMetrics = field(default_factory=FactionMetrics)
    player_metrics: dict[str, PlayerMetrics] = field(default_factory=dict)
    role_metrics: dict[str, RoleMetrics] = field(default_factory=dict)
    quality_metrics: QualityMetrics = field(default_factory=QualityMetrics)
    safety_metrics: SafetyMetrics = field(default_factory=SafetyMetrics)
    cost_metrics: CostMetrics = field(default_factory=CostMetrics)
    growth_curve: list[GrowthPoint] = field(default_factory=list)
    provenance: dict[str, MetricProvenance] = field(default_factory=dict)
    total_games: int = 0

    def to_json_dict(self) -> dict[str, Any]:
        def _dataclass_to_dict(obj: Any) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _dataclass_to_dict(v) for k, v in obj.__dict__.items()}
            if isinstance(obj, dict):
                return {k: _dataclass_to_dict(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_dataclass_to_dict(v) for v in obj]
            return obj

        return {
            "batch_id": self.batch_id,
            "faction_metrics": _dataclass_to_dict(self.faction_metrics),
            "player_metrics": _dataclass_to_dict(self.player_metrics),
            "role_metrics": _dataclass_to_dict(self.role_metrics),
            "quality_metrics": _dataclass_to_dict(self.quality_metrics),
            "safety_metrics": _dataclass_to_dict(self.safety_metrics),
            "cost_metrics": _dataclass_to_dict(self.cost_metrics),
            "growth_curve": _dataclass_to_dict(self.growth_curve),
            "provenance": _dataclass_to_dict(self.provenance),
            "total_games": self.total_games,
        }


# ---------------------------------------------------------------------------
# Leaderboard — standard JSON report per §14
# ---------------------------------------------------------------------------

@dataclass
class LeaderboardEntry:
    rank: int
    player_id: str
    model: str = ""
    persona: str = ""
    overall_score: float = 0.0
    win_rate: float = 0.0
    good_win_rate: float = 0.0
    werewolf_win_rate: float = 0.0
    anti_push_rate: float = 0.0
    lie_detection_rate: float = 0.0
    stance_accuracy: float = 0.0
    illegal_action_rate: float = 0.0
    avg_cost_per_game: float = 0.0
    avg_latency_ms: int = 0
    games_played: int = 0


@dataclass
class ExperimentComparison:
    dimension: str
    label_a: str
    label_b: str
    metric_name: str
    value_a: float = 0.0
    value_b: float = 0.0
    delta: float = 0.0
    games_a: int = 0
    games_b: int = 0


@dataclass
class LeaderboardReport:
    report_id: str
    batch_ids: list[str] = field(default_factory=list)
    entries: list[LeaderboardEntry] = field(default_factory=list)
    comparisons: list[ExperimentComparison] = field(default_factory=list)
    growth_curves: dict[str, list[GrowthPoint]] = field(default_factory=dict)
    generated_at: str = ""
    total_games: int = 0

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "batch_ids": list(self.batch_ids),
            "entries": [
                {
                    "rank": e.rank,
                    "player_id": e.player_id,
                    "model": e.model,
                    "persona": e.persona,
                    "overall_score": round(e.overall_score, 4),
                    "win_rate": round(e.win_rate, 4),
                    "good_win_rate": round(e.good_win_rate, 4),
                    "werewolf_win_rate": round(e.werewolf_win_rate, 4),
                    "anti_push_rate": round(e.anti_push_rate, 4),
                    "lie_detection_rate": round(e.lie_detection_rate, 4),
                    "stance_accuracy": round(e.stance_accuracy, 4),
                    "illegal_action_rate": round(e.illegal_action_rate, 4),
                    "avg_cost_per_game": round(e.avg_cost_per_game, 4),
                    "avg_latency_ms": e.avg_latency_ms,
                    "games_played": e.games_played,
                }
                for e in self.entries
            ],
            "comparisons": [
                {
                    "dimension": c.dimension,
                    "label_a": c.label_a,
                    "label_b": c.label_b,
                    "metric_name": c.metric_name,
                    "value_a": round(c.value_a, 4),
                    "value_b": round(c.value_b, 4),
                    "delta": round(c.delta, 4),
                    "games_a": c.games_a,
                    "games_b": c.games_b,
                }
                for c in self.comparisons
            ],
            "growth_curves": {
                key: [{"game_number": p.game_number, "metric_name": p.metric_name, "value": round(p.value, 4)} for p in points]
                for key, points in self.growth_curves.items()
            },
            "generated_at": self.generated_at,
            "total_games": self.total_games,
        }


# ---------------------------------------------------------------------------
# Replay record — sufficient to reproduce a game deterministically
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReplayRecord:
    game_id: str
    initial_seed: int
    ruleset_snapshot: dict[str, Any]
    event_log: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "initial_seed": self.initial_seed,
            "ruleset_snapshot": dict(self.ruleset_snapshot),
            "event_log": list(self.event_log),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReplayRecord":
        return cls(
            game_id=data["game_id"],
            initial_seed=data["initial_seed"],
            ruleset_snapshot=data["ruleset_snapshot"],
            event_log=data["event_log"],
        )


# ---------------------------------------------------------------------------
# Full evaluation report — observer-UI-ready JSON bundle
# ---------------------------------------------------------------------------

@dataclass
class FullEvaluationReport:
    report_id: str
    batch_id: str
    metrics: dict[str, Any]
    leaderboard: dict[str, Any] | None = None
    generated_at: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "batch_id": self.batch_id,
            "metrics": self.metrics,
            "leaderboard": self.leaderboard,
            "generated_at": self.generated_at,
        }
