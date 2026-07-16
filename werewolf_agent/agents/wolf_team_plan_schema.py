# -*- coding: utf-8 -*-
"""
定义狼队夜间团队计划的结构化输出 schema。

作者: Project contributors
创建日期: 2026-07-08
修改日期: 2026-07-16

使用示例:
    >>> from werewolf_agent.agents.wolf_team_plan_schema import WolfTeamPlan
    >>> WolfTeamPlan.model_validate({"night_number": 1, "public_story": "白天统一口径", "reasoning": "夜聊共识"})
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WolfTeamPlan(BaseModel):
    """狼队队长每夜产出的结构化团队计划。"""

    model_config = ConfigDict(extra="forbid")

    night_number: int = Field(..., ge=1, description="本夜编号")
    night_kill_primary: str | None = Field(
        None,
        description="本夜首选击杀目标 player_id; None 表示主动空刀",
    )
    night_kill_backup: str | None = Field(
        None,
        description="备选击杀目标 (primary 已死或不合法时启用); None 表示无备选",
    )
    fake_seer: str | None = Field(
        None, description="悍跳预言家位 (alive werewolf player_id 或 None)"
    )
    pusher: str | None = Field(
        None, description="冲票位 (alive werewolf player_id 或 None)"
    )
    hooker: str | None = Field(
        None, description="倒钩位 (alive werewolf player_id 或 None)"
    )
    deep_cover: str | None = Field(
        None, description="深水位 (alive werewolf player_id 或 None)"
    )
    public_story: str = Field(
        ...,
        min_length=1,
        max_length=120,
        description="白天对外统一口径 / 抗推叙事",
    )
    evidence_quality: Literal["strong", "weak", "none"] = Field(
        "weak",
        description="队长对夜聊共识度的展示评估，不参与 V2 执行判定",
    )
    reasoning: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="队长决策依据 (审计用; werewolf_team_only 边界, 禁止公开)",
    )

    @model_validator(mode="after")
    def _no_duplicate_role_assignments(self) -> "WolfTeamPlan":
        roles = [self.fake_seer, self.pusher, self.hooker, self.deep_cover]
        assigned = [role for role in roles if role is not None]
        if len(assigned) != len(set(assigned)):
            raise ValueError(
                "WolfTeamPlan: 4 角色(fake_seer/pusher/hooker/deep_cover)"
                f"中非 None 字段必须互不重复, 当前: {roles}"
            )
        return self

    @model_validator(mode="after")
    def _kills_not_overlap_roles(self) -> "WolfTeamPlan":
        roles = {self.fake_seer, self.pusher, self.hooker, self.deep_cover}
        roles.discard(None)
        kills = {self.night_kill_primary, self.night_kill_backup}
        kills.discard(None)
        overlap = roles & kills
        if overlap:
            raise ValueError(
                f"WolfTeamPlan: 击杀目标不能是狼队角色 {sorted(overlap)}"
            )
        return self


def wolf_team_plan_contract() -> dict[str, dict[str, Any]]:
    """从 Pydantic schema 导出 prompt/tool 共用的字段约束。"""
    properties = WolfTeamPlan.model_json_schema()["properties"]
    return {
        "public_story": {
            "min_length": properties["public_story"]["minLength"],
            "max_length": properties["public_story"]["maxLength"],
        },
        "reasoning": {
            "min_length": properties["reasoning"]["minLength"],
            "max_length": properties["reasoning"]["maxLength"],
        },
    }


__all__ = ["WolfTeamPlan", "wolf_team_plan_contract"]
