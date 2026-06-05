"""Per-player failure profile metrics.

Each player has a fixed model + persona (their "brain"). This module
aggregates per-attempt failure data so developers can identify which
players' prompts need tuning — NOT to switch models (the brain is
immutable), but to refine the persona's prompt template.

Memory-only: not persisted across sessions. Used for offline prompt
tuning and for surfacing failure hotspots in audit reports.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlayerFailureProfile:
    player_id: str
    sample_count: int = 0
    fallback_count: int = 0
    error_code_counts: dict[str, int] = field(default_factory=dict)
    per_task_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def fallback_rate(self) -> float:
        return self.fallback_count / self.sample_count if self.sample_count else 0.0


class MetricsCollector:
    """Per-player failure aggregator. Thread-safe — record/get operations
    take a single ``threading.Lock`` so concurrent game threads updating
    the same profile do not lose counts (R3-MG-4).
    """

    def __init__(self) -> None:
        self._profiles: dict[str, PlayerFailureProfile] = {}
        self._lock = threading.Lock()

    def record(
        self,
        *,
        player_id: str,
        task_type: str,
        error_code: str | None,
        fallback_used: bool,
        retry_count: int,
    ) -> None:
        with self._lock:
            profile = self._profiles.setdefault(player_id, PlayerFailureProfile(player_id=player_id))
            profile.sample_count += 1
            if fallback_used:
                profile.fallback_count += 1
            if error_code:
                profile.error_code_counts[error_code] = profile.error_code_counts.get(error_code, 0) + 1

            task_stats = profile.per_task_breakdown.setdefault(task_type, {
                "sample_count": 0,
                "fallback_count": 0,
                "error_code_counts": {},
            })
            task_stats["sample_count"] += 1
            if fallback_used:
                task_stats["fallback_count"] += 1
            if error_code:
                task_stats["error_code_counts"][error_code] = task_stats["error_code_counts"].get(error_code, 0) + 1

    def get_profile(self, player_id: str) -> PlayerFailureProfile:
        with self._lock:
            return self._profiles.get(player_id, PlayerFailureProfile(player_id=player_id))

    def get_top_failures(self, *, n: int = 5) -> list[PlayerFailureProfile]:
        with self._lock:
            candidates = [p for p in self._profiles.values() if p.sample_count > 0]
            candidates.sort(key=lambda p: (p.fallback_rate, p.sample_count), reverse=True)
            return candidates[:n]

    def export_report(self) -> dict[str, Any]:
        with self._lock:
            return {
                pid: {
                    "sample_count": p.sample_count,
                    "fallback_rate": p.fallback_rate,
                    "error_code_counts": p.error_code_counts,
                    "per_task_breakdown": p.per_task_breakdown,
                }
                for pid, p in self._profiles.items()
            }
