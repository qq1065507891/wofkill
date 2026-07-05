# -*- coding: utf-8 -*-
"""
功能描述：人设路由器——按玩家、任务和上下文解析运行时人设配置。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_INCOMPATIBLE_GOOD_ROLE_STYLE_MARKERS = (
    "fake",
    "deceiv",
    "deception",
    "manipulat",
    "hook",
    "infiltrat",
    "mislead",
    "appears_good",
    "redirect_blame",
    "seer_claim",
    "悍跳",
    "倒钩",
    "欺骗",
    "操纵",
    "冒充",
)

_ROLE_NEUTRAL_TASK_STYLES = {
    "speech": "evidence_based_expression",
    "vote": "independent_evidence_vote",
    "night_action": "deliberate_legal_action",
    "deception": "role_consistent_expression",
    "sheriff_speech": "evidence_based_sheriff_speech",
    "defense_speech": "evidence_based_defense",
    "last_words": "concise_factual_summary",
    "reflection": "factual_self_review",
}


def _has_incompatible_good_role_style(value: Any) -> bool:
    text = str(value or "").lower()
    return any(marker in text for marker in _INCOMPATIBLE_GOOD_ROLE_STYLE_MARKERS)


def sanitize_persona_snapshot(
    snapshot: dict[str, Any],
    *,
    own_role: str,
    task_type: str,
) -> dict[str, Any]:
    """Remove identity strategy from a non-wolf persona snapshot.

    Persona may shape expression, but it must not assign fake claims,
    manipulation plans, or wolf-team duties to a role that cannot legally
    rely on those strategies.
    """
    sanitized = dict(snapshot)
    if not own_role or own_role == "werewolf":
        return sanitized

    if _has_incompatible_good_role_style(sanitized.get("personality")):
        sanitized["personality"] = "role_consistent_player"
    if _has_incompatible_good_role_style(sanitized.get("speech_style")):
        sanitized["speech_style"] = "role_consistent_expression"
    if _has_incompatible_good_role_style(sanitized.get("task_style")):
        sanitized["task_style"] = _ROLE_NEUTRAL_TASK_STYLES.get(
            task_type,
            "role_consistent_expression",
        )

    for field_name in ("effective_params", "dynamic_adjustments", "base_params"):
        params = sanitized.get(field_name)
        if isinstance(params, dict):
            sanitized[field_name] = {
                key: value
                for key, value in params.items()
                if key != "deception_skill"
            }

    for field_name in ("tone", "style", "phrase_style"):
        if _has_incompatible_good_role_style(sanitized.get(field_name)):
            sanitized.pop(field_name, None)

    return sanitized


@dataclass(frozen=True)
class PersonaSnapshot:
    """Stable runtime snapshot of a persona for one agent call."""
    agent_id: str
    profile_id: str
    display_name: str
    personality: str
    speech_style: str
    base_params: dict[str, float]
    task_style: str
    dynamic_adjustments: dict[str, float]
    effective_params: dict[str, float]


@dataclass(frozen=True)
class GameContext:
    """Minimal game context for dynamic persona adjustments."""
    phase: str = ""
    day_number: int = 0
    night_number: int = 0
    player_is_suspected: bool = False
    teammate_exiled: bool = False
    trusted_by_good: bool = False
    has_badge: bool = False
    own_role: str = ""
    alive: bool = True


class PersonaRouter:
    """Runtime persona resolution engine."""

    def __init__(
        self,
        profiles: dict[str, dict[str, Any]],
        player_assignments: dict[str, str],
    ) -> None:
        self._profiles = profiles
        self._assignments = player_assignments

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PersonaRouter":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        profiles = data.get("persona_profiles", {})
        # Player assignments come from models.yaml, not persona config
        return cls(profiles=profiles, player_assignments={})

    def load_assignments(self, player_assignments: dict[str, str]) -> None:
        self._assignments = player_assignments

    def resolve(
        self,
        agent_id: str,
        task_type: str,
        game_context: GameContext | None = None,
    ) -> PersonaSnapshot:
        """Resolve a persona snapshot for this agent+task+context combination."""
        profile_id = self._assignments.get(agent_id, "")
        profile = self._profiles.get(profile_id, {})

        if not profile:
            return PersonaSnapshot(
                agent_id=agent_id,
                profile_id="default",
                display_name="default",
                personality="",
                speech_style="",
                base_params={},
                task_style="default",
                dynamic_adjustments={},
                effective_params={},
            )

        base = dict(profile.get("base", {}))
        base_params = {
            k: float(v) for k, v in base.items()
            if isinstance(v, (int, float))
        }

        task_styles = profile.get("task_styles", {})
        task_style = task_styles.get(task_type, "default")

        dynamic_adj = self._compute_dynamic_adjustments(
            profile, game_context
        )

        effective = dict(base_params)
        for k, delta in dynamic_adj.items():
            effective[k] = max(0.0, min(1.0, effective.get(k, 0.5) + delta))

        snapshot_data = sanitize_persona_snapshot(
            {
                "agent_id": agent_id,
                "profile_id": profile_id,
                "display_name": profile.get("display_name", profile_id),
                "personality": str(base.get("personality", "")),
                "speech_style": str(base.get("speech_style", "")),
                "base_params": base_params,
                "task_style": task_style,
                "dynamic_adjustments": dynamic_adj,
                "effective_params": effective,
            },
            own_role=game_context.own_role if game_context else "",
            task_type=task_type,
        )

        return PersonaSnapshot(
            agent_id=snapshot_data["agent_id"],
            profile_id=snapshot_data["profile_id"],
            display_name=snapshot_data["display_name"],
            personality=snapshot_data["personality"],
            speech_style=snapshot_data["speech_style"],
            base_params=snapshot_data["base_params"],
            task_style=snapshot_data["task_style"],
            dynamic_adjustments=snapshot_data["dynamic_adjustments"],
            effective_params=snapshot_data["effective_params"],
        )

    def get_profile_for_agent(self, agent_id: str) -> str:
        return self._assignments.get(agent_id, "")

    def _compute_dynamic_adjustments(
        self,
        profile: dict[str, Any],
        context: GameContext | None,
    ) -> dict[str, float]:
        if context is None:
            return {}

        policy = profile.get("dynamic_policy", {})
        adjustments: dict[str, float] = {}

        if context.player_is_suspected and "when_suspected" in policy:
            for k, v in policy["when_suspected"].items():
                key = k.replace("_delta", "")
                adjustments[key] = adjustments.get(key, 0.0) + float(v)

        if context.teammate_exiled and "when_teammate_exiled" in policy:
            for k, v in policy["when_teammate_exiled"].items():
                key = k.replace("_delta", "")
                adjustments[key] = adjustments.get(key, 0.0) + float(v)

        if context.trusted_by_good and "when_trusted_by_good_players" in policy:
            for k, v in policy["when_trusted_by_good_players"].items():
                key = k.replace("_delta", "")
                adjustments[key] = adjustments.get(key, 0.0) + float(v)

        return adjustments
