"""Judge persona router — resolves judge tone/style profiles for broadcasts.

Layer 3: Follows PersonaRouter pattern (YAML loading, profile resolution) but
specialized for judge-specific dimensions and broadcast styles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class JudgePersonaSnapshot:
    """Immutable runtime snapshot of a judge persona for one broadcast."""

    profile_id: str
    display_name: str = ""
    tone_variant: str = "neutral"
    base_params: dict[str, float] = field(default_factory=dict)
    task_styles: dict[str, str] = field(default_factory=dict)
    broadcast_patterns: dict[str, str] = field(default_factory=dict)
    system_prompt: str = ""


class JudgeProfileRouter:
    """Loads judge persona profiles from YAML, resolves by profile_id or tone."""

    def __init__(self, profiles: dict[str, dict[str, Any]]) -> None:
        self._profiles: dict[str, dict[str, Any]] = profiles
        self._tone_index: dict[str, str] = {}
        for pid, prof in profiles.items():
            tone = prof.get("tone_variant", "")
            if tone:
                self._tone_index[tone] = pid

    @classmethod
    def from_yaml(cls, path: str | Path) -> JudgeProfileRouter:
        """Load judge profiles from a YAML file."""
        p = Path(path)
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        profiles = data.get("judge_profiles", {})
        return cls(profiles=profiles)

    def resolve(
        self,
        profile_id: str = "neutral_arbiter",
        task_type: str = "judge_phase",
    ) -> JudgePersonaSnapshot:
        """Resolve a judge persona snapshot for the given profile and task.

        Falls back to ``neutral_arbiter`` when profile_id is not found.
        """
        prof = self._profiles.get(profile_id)
        if prof is None:
            profile_id = "neutral_arbiter"
            prof = self._profiles.get(profile_id, {})
        if not prof:
            return JudgePersonaSnapshot(profile_id="fallback")

        base = prof.get("base", {})
        task_styles = prof.get("task_styles", {})
        # P5 (post-review-v2): 把 task_styles[task_type] 拼进 system_prompt。
        # 这覆盖了 P1-4 的 revert：J-7 之后 system_prompt 的「稳定半」只指
        # base 的 prompt 模板；per-task_type 的风格提示属于同一 system
        # message 的可读扩展，不影响下游 prompt 缓存键（base 仍稳定）。
        base_system = prof.get("system_prompt", "")
        task_style_text = (task_styles or {}).get(task_type, "")
        if task_style_text:
            system_prompt = f"{base_system}\n\n[TASK STYLE: {task_style_text}]"
        else:
            system_prompt = base_system
        return JudgePersonaSnapshot(
            profile_id=profile_id,
            display_name=prof.get("display_name", profile_id),
            tone_variant=prof.get("tone_variant", "neutral"),
            base_params=dict(base),
            task_styles=dict(task_styles),
            broadcast_patterns=dict(prof.get("broadcast_patterns", {})),
            system_prompt=system_prompt,
        )

    def resolve_by_tone(
        self,
        tone_variant: str = "neutral",
        task_type: str = "judge_phase",
    ) -> JudgePersonaSnapshot:
        """Resolve by tone variant name (e.g. 'tournament', 'variety_show')."""
        profile_id = self._tone_index.get(tone_variant, "neutral_arbiter")
        return self.resolve(profile_id=profile_id, task_type=task_type)

    def list_profiles(self) -> list[str]:
        """Return available profile IDs."""
        return list(self._profiles.keys())

    def list_tones(self) -> list[str]:
        """Return available tone variants."""
        return list(self._tone_index.keys())
