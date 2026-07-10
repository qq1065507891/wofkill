# -*- coding: utf-8 -*-
"""
功能描述：技能注册中心，根据角色和阶段查找、派发适用技能，并生成调用监控摘要。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-10
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.skills.schemas import (
    SkillDefinition,
    SkillFaction,
    SkillInput,
    SkillName,
    SkillOutput,
)
from werewolf_agent.skills.werewolf_skills import (
    SKILL_DEFINITIONS,
    _cap_prompt_injectable,
    apply_skill,
)


# Roles that belong to the good faction
_GOOD_ROLES = {"villager", "seer", "witch", "hunter", "idiot"}
_WOLF_ROLES = {"werewolf"}


def faction_for_role(role: str, gs: Any | None = None) -> SkillFaction:
    """Return the faction a role belongs to.

    Most roles are statically mapped (villager→GOOD, werewolf→WOLF).
    Hybrid is neutral for agent-facing skill selection because the role does
    not know its master's hidden faction.
    """
    if role in _WOLF_ROLES:
        return SkillFaction.WOLF
    if role == "hybrid":
        return SkillFaction.NEUTRAL
    return SkillFaction.GOOD


class SkillRegistry:
    """Central registry for werewolf agent skills."""

    def __init__(self) -> None:
        self._skills: dict[SkillName, SkillDefinition] = {}
        for skill in SKILL_DEFINITIONS:
            self._skills[skill.name] = skill

    def register(self, skill: SkillDefinition) -> None:
        self._skills[skill.name] = skill

    def get(self, name: SkillName) -> SkillDefinition | None:
        return self._skills.get(name)

    def all_skills(self) -> list[SkillDefinition]:
        return list(self._skills.values())

    def count(self) -> int:
        return len(self._skills)

    def applicable_skills(self, role: str, phase: str) -> list[SkillDefinition]:
        return [s for s in self._skills.values() if s.is_applicable(role, phase)]

    def dispatch(self, name: SkillName, skill_input: SkillInput) -> SkillOutput:
        """Apply a skill and return structured output."""
        return apply_skill(name, skill_input)

    def dispatch_applicable(
        self,
        skill_input: SkillInput,
    ) -> list[SkillOutput]:
        """Dispatch all applicable skills for the given input."""
        applicable = self.applicable_skills(skill_input.role, skill_input.phase)
        return [
            self.dispatch(s.name, skill_input)
            for s in applicable
        ]

    def by_role(self, role: str) -> list[SkillDefinition]:
        return [
            s for s in self._skills.values()
            if not s.applicable_roles or role in s.applicable_roles
        ]

    def by_phase(self, phase: str) -> list[SkillDefinition]:
        return [
            s for s in self._skills.values()
            if not s.applicable_phases or phase in s.applicable_phases
        ]

    def by_tag(self, tag: str) -> list[SkillDefinition]:
        return [s for s in self._skills.values() if tag in s.tags]

    def by_faction(self, faction: SkillFaction) -> list[SkillDefinition]:
        return [s for s in self._skills.values() if s.faction == faction]

    def skills_for_role(self, role: str, gs: Any | None = None) -> list[SkillDefinition]:
        """Return skills available to a role based on its faction.

        Loading rule:
        - WOLF roles get: WOLF + COMMON + UNIVERSAL
        - GOOD roles get: GOOD + COMMON + UNIVERSAL

        Hybrid receives only COMMON and UNIVERSAL skills.
        """
        role_faction = faction_for_role(role, gs=gs)
        allowed = {SkillFaction.COMMON, SkillFaction.UNIVERSAL}
        if role_faction != SkillFaction.NEUTRAL:
            allowed.add(role_faction)
        return [
            s for s in self._skills.values()
            if s.faction in allowed
            and (not s.applicable_roles or role in s.applicable_roles)
        ]

    def dispatch_for_role(
        self,
        role: str,
        phase: str,
        skill_input: SkillInput,
        task_type: str = "",
        gs: Any | None = None,
        audit_records: list[dict[str, Any]] | None = None,
    ) -> list[SkillOutput]:
        """Dispatch all faction-applicable skills for a role in a given phase.

        P0-K2: when `task_type` is provided, the dispatch is filtered by
        `SkillDefinition.applies_to_task_types` (in addition to the
        existing `applicable_phases` / `applicable_roles` checks).

        Hybrid receives only faction-neutral skills because its master's
        faction is hidden from the player.
        """
        role_faction = faction_for_role(role, gs=gs)
        effective_role = role
        allowed = {SkillFaction.COMMON, SkillFaction.UNIVERSAL}
        if role_faction != SkillFaction.NEUTRAL:
            allowed.add(role_faction)
        skills = [
            s for s in self._skills.values()
            if s.faction in allowed
            and s.is_applicable(effective_role, phase, task_type=task_type)
        ]
        outputs: list[SkillOutput] = []
        # P-SK1: 把 SKILL.md 正文（markdown body）追加到每个
        # SkillOutput.prompt_injectable 末尾的 "## 技能说明" 段。
        # 散文形式的设计/使用/注意事项由此真正进入 LLM 的视野 —
        # markdown-driven 设计的"驱动"语义由此字段落地。无 body
        # 的 skill（如纯 Python handler 自带完整 advice 的）保持
        # 现状不变。
        for skill_def in skills:
            try:
                out = self.dispatch(skill_def.name, skill_input)
            except Exception as exc:
                if audit_records is None:
                    raise
                audit_records.append(
                    _skill_call_audit_error(skill_def, skill_input, exc)
                )
                continue
            if skill_def.body and out.prompt_injectable:
                out.prompt_injectable = _cap_prompt_injectable(
                    out.prompt_injectable
                    + f"\n\n## 技能说明\n{skill_def.body}\n"
                )
            else:
                out.prompt_injectable = _cap_prompt_injectable(out.prompt_injectable)
            outputs.append(out)
            if audit_records is not None:
                audit_records.append(
                    _skill_call_audit_success(skill_def, skill_input, out)
                )
        return outputs

    def names(self) -> list[str]:
        return [s.name.value for s in self._skills.values()]


def _skill_input_summary(skill_input: SkillInput) -> dict[str, Any]:
    return {
        "role": skill_input.role,
        "phase": skill_input.phase,
        "task_type": skill_input.task_type,
        "day": skill_input.day,
        "legal_target_count": len(skill_input.legal_targets),
        "has_wolf_team_plan": bool(skill_input.extra.get("wolf_team_plan")),
    }


def _skill_call_base(
    skill_def: SkillDefinition,
    skill_input: SkillInput,
) -> dict[str, Any]:
    return {
        "call_kind": "skill",
        "call_name": skill_def.name.value,
        "skill_name": skill_def.name.value,
        "input_summary": _skill_input_summary(skill_input),
    }


def _skill_call_audit_success(
    skill_def: SkillDefinition,
    skill_input: SkillInput,
    output: SkillOutput,
) -> dict[str, Any]:
    prompt_visible = bool(output.prompt_injectable)
    evidence_refs = (output.metadata or {}).get("evidence_refs", []) or []
    return {
        **_skill_call_base(skill_def, skill_input),
        "status": "success",
        "success": True,
        "prompt_visible": prompt_visible,
        "result_available_to_decision": prompt_visible,
        "decision_usage": "prompt_injected" if prompt_visible else "result_not_prompt_visible",
        "output_summary": {
            "confidence": float(output.confidence),
            "has_prompt_injectable": prompt_visible,
            "risk_alert_count": len(output.risk_alerts),
            "evidence_ref_count": len(evidence_refs),
        },
    }


def _skill_call_audit_error(
    skill_def: SkillDefinition,
    skill_input: SkillInput,
    exc: Exception,
) -> dict[str, Any]:
    return {
        **_skill_call_base(skill_def, skill_input),
        "status": "error",
        "success": False,
        "prompt_visible": False,
        "result_available_to_decision": False,
        "decision_usage": "not_available_error",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }
