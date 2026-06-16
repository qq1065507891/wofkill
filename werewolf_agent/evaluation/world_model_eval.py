"""Offline world-model true-world rank metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from werewolf_agent.evaluation.feedback_schemas import EvaluationTrace, ModuleExposure
from werewolf_agent.evaluation.schemas import GameResult


@dataclass(frozen=True)
class WorldRankSample:
    trace_id: str
    support: str
    true_world_rank: int = 0
    top_world_score: float = 0.0
    unsupported_reason: str = ""


@dataclass(frozen=True)
class WorldModelRankMetrics:
    supported_count: int = 0
    unsupported_count: int = 0
    true_world_top1_rate: float = 0.0
    true_world_top3_rate: float = 0.0
    avg_true_world_rank: float = 0.0
    overconfidence_rate: float = 0.0
    samples: list[WorldRankSample] = field(default_factory=list)


def compute_world_model_rank_metrics(
    result: GameResult,
    traces: list[EvaluationTrace],
) -> WorldModelRankMetrics:
    """Compute top-k-only true-world rank from sanitized trace assignments."""
    samples = [
        _sample_from_trace(result, trace)
        for trace in traces
        if any(exposure.module == "possible_worlds" for exposure in trace.module_exposures)
    ]
    supported_samples = [sample for sample in samples if sample.support == "supported"]
    supported_count = len(supported_samples)
    unsupported_count = sum(1 for sample in samples if sample.support == "unsupported")
    ranks = [sample.true_world_rank for sample in supported_samples if sample.true_world_rank > 0]
    top1_hits = sum(1 for rank in ranks if rank == 1)
    top3_hits = sum(1 for rank in ranks if rank <= 3)
    overconfident = sum(
        1
        for sample in supported_samples
        if sample.true_world_rank > 1 and sample.top_world_score > 0.0
    )
    return WorldModelRankMetrics(
        supported_count=supported_count,
        unsupported_count=unsupported_count,
        true_world_top1_rate=_rate(top1_hits, supported_count),
        true_world_top3_rate=_rate(top3_hits, supported_count),
        avg_true_world_rank=sum(ranks) / len(ranks) if ranks else 0.0,
        overconfidence_rate=_rate(overconfident, supported_count),
        samples=samples,
    )


def _sample_from_trace(result: GameResult, trace: EvaluationTrace) -> WorldRankSample:
    worlds = sorted(
        (
            exposure
            for exposure in trace.module_exposures
            if exposure.module == "possible_worlds"
        ),
        key=lambda exposure: exposure.rank if exposure.rank > 0 else 9999,
    )
    comparable_seen = False
    top_world_score = worlds[0].score if worlds else 0.0
    for fallback_rank, exposure in enumerate(worlds, start=1):
        assignments = _assignments(exposure)
        comparable = {
            player_id: _normalize_role(role)
            for player_id, role in assignments.items()
            if player_id in result.player_roles
        }
        if not comparable:
            continue
        comparable_seen = True
        if all(
            _normalize_role(result.player_roles[player_id]) == role
            for player_id, role in comparable.items()
        ):
            return WorldRankSample(
                trace_id=trace.trace_id,
                support="supported",
                true_world_rank=exposure.rank if exposure.rank > 0 else fallback_rank,
                top_world_score=top_world_score,
            )
    if comparable_seen:
        return WorldRankSample(
            trace_id=trace.trace_id,
            support="supported",
            true_world_rank=len(worlds) + 1,
            top_world_score=top_world_score,
        )
    return WorldRankSample(
        trace_id=trace.trace_id,
        support="unsupported",
        unsupported_reason="no_comparable_assignments",
        top_world_score=top_world_score,
    )


def _assignments(exposure: ModuleExposure) -> dict[str, Any]:
    raw = exposure.metadata.get("key_assignments")
    if not isinstance(raw, dict):
        return {}
    return {str(player_id): role for player_id, role in raw.items()}


def _normalize_role(value: Any) -> str:
    role = str(value or "").strip().lower()
    if role in {"wolf", "wolves", "werewolves"}:
        return "werewolf"
    return role


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator
