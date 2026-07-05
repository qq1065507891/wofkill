# -*- coding: utf-8 -*-
"""
功能描述：主动检测预言家受压、女巫毒药未用、猎人即将死亡、混血儿主人死亡
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from werewolf_agent.core.models import GameState


@dataclass(frozen=True)
class RoleAlert:
    alert_type: str
    severity: str  # "critical" | "warning" | "info"
    message: str
    evidence: list[str]


class RoleStateMonitor:
    """Detect critical role-specific states (seer pressure, witch poison risk, etc.).

    All detection is deterministic regex/keyword-based — no LLM calls.
    Stateless — single instance can be reused across all calls.
    """

    def assess(
        self, gs: GameState, player_id: str, role: str, phase: str,
    ) -> list[RoleAlert]:
        alerts: list[RoleAlert] = []

        if role == "seer":
            alerts.extend(self._check_seer_pressure(gs, player_id))
        elif role == "witch":
            alerts.extend(self._check_witch_poison_risk(gs, player_id))
        elif role == "hunter":
            alerts.extend(self._check_hunter_risk(gs, player_id))
        elif role == "hybrid":
            alerts.extend(self._check_hybrid_master_dead(gs, player_id))

        return alerts

    def _check_seer_pressure(self, gs: GameState, seer_id: str) -> list[RoleAlert]:
        current_day = gs.day_number
        questioning_speakers = set()
        for e in gs.events:
            if e.type not in ("speech", "sheriff_speech"):
                continue
            if e.payload.get("day_number") != current_day:
                continue
            speaker = e.payload.get("speaker", "")
            text = str(e.payload.get("text", ""))
            if self._text_targets_player(text, seer_id):
                questioning_speakers.add(speaker)

        if len(questioning_speakers) < 2:
            return []

        seer_checks = [
            e for e in gs.events
            if e.type == "seer_check"
        ]
        reported = set()
        for e in gs.events:
            if e.type not in ("speech", "sheriff_speech"):
                continue
            text = str(e.payload.get("text", ""))
            speaker = e.payload.get("speaker", "")
            if speaker != seer_id:
                continue
            for sc in seer_checks:
                target = sc.payload.get("target_id", "")
                if target and target in text:
                    reported.add(target)

        unreported = [sc for sc in seer_checks if sc.payload.get("target_id") not in reported]

        evidence = [f"{len(questioning_speakers)} players questioning seer {seer_id}"]
        if unreported:
            evidence.append(
                f"{len(unreported)} unreported checks: "
                f"{[sc.payload.get('target_id') for sc in unreported]}"
            )

        return [RoleAlert(
            alert_type="SEER_UNDER_PRESSURE",
            severity="critical",
            message=(
                f"你有{len(questioning_speakers)}个玩家在质疑你的预言家身份（"
                f"{', '.join(sorted(questioning_speakers))}）。"
                f"你必须完整列出所有夜晚查验记录（含查验理由和结果），"
                f"并明确警徽流。不能只说'按顺序验'——每夜的查验目标必须有具体动机。"
                f"{'你还有未报出的查验结果，必须在本轮全部报出！' if unreported else ''}"
                f"注意：冲票你的人中至少有一匹狼在带节奏——点出谁在踩你、他们的票型和发言"
                f"是否存在矛盾，引导好人盘冲票关系。"
            ),
            evidence=evidence,
        )]

    def _check_witch_poison_risk(self, gs: GameState, witch_id: str) -> list[RoleAlert]:
        if gs.poison_used:
            return []

        alive_count = sum(1 for p in gs.players.values() if p.alive)
        if alive_count > 8:
            return []

        current_day = gs.day_number
        targeting_speakers = set()
        for e in gs.events:
            if e.type not in ("speech", "sheriff_speech"):
                continue
            if e.payload.get("day_number") != current_day:
                continue
            text = str(e.payload.get("text", ""))
            if self._text_targets_player(text, witch_id):
                targeting_speakers.add(e.payload.get("speaker", ""))

        if len(targeting_speakers) < 2:
            return []

        return [RoleAlert(
            alert_type="WITCH_POISON_UNUSED_AT_RISK",
            severity="critical",
            message=(
                f"警告：你的毒药还未使用，但已有{len(targeting_speakers)}个玩家在质疑你。"
                f"场上仅存活{alive_count}人。如果你今晚不撒毒而被狼杀或被放逐，"
                f"毒药将彻底浪费。你必须今晚选择一个目标使用毒药——"
                f"从质疑你的玩家中挑选最可疑的。"
            ),
            evidence=[
                f"{len(targeting_speakers)} players targeting witch {witch_id}",
                f"alive={alive_count}, poison unused",
            ],
        )]

    def _check_hunter_risk(self, gs: GameState, hunter_id: str) -> list[RoleAlert]:
        alive_count = sum(1 for p in gs.players.values() if p.alive)
        if alive_count > 6:
            return []

        mentioned = False
        for e in gs.events:
            if e.type not in ("speech", "sheriff_speech"):
                continue
            text = str(e.payload.get("text", ""))
            if hunter_id in text:
                mentioned = True
                break

        if not mentioned:
            return []

        return [RoleAlert(
            alert_type="HUNTER_MAY_DIE_SOON",
            severity="warning",
            message=(
                f"场上仅存活{alive_count}人，且你的身份已被讨论。"
                f"如果今晚被狼人刀杀，你必须果断开枪。"
                f"提前思考：如果现在就要开枪，你会带走谁？"
            ),
            evidence=[f"alive={alive_count}, hunter mentioned in speeches"],
        )]

    def _check_hybrid_master_dead(self, gs: GameState, hybrid_id: str) -> list[RoleAlert]:
        master_id = gs.hybrid_master_id
        if not master_id:
            return []

        master = gs.players.get(master_id)
        if master and master.alive:
            return []

        alive_count = sum(1 for p in gs.players.values() if p.alive)
        if alive_count > 8:
            return []

        return [RoleAlert(
            alert_type="HYBRID_MASTER_DEAD",
            severity="info",
            message=(
                f"你的主人 {master_id} 已经死亡。"
                "你的胜利绑定仍按主人的原始阵营结算，但你仍不知道主人的阵营。"
                "继续回顾主人的公开发言、投票和遗言，从公开行为中判断方向。"
            ),
            evidence=[f"master {master_id} is dead, alive={alive_count}"],
        )]

    @staticmethod
    def _text_targets_player(text: str, player_id: str) -> bool:
        if player_id not in text:
            return False
        patterns = [
            rf"(?:怀疑|标狼|狼面|定狼|抗推|有问题|投票.*出|投.*出局|出)\s*{player_id}",
            rf"{player_id}.*(?:是狼|有问题|不对劲|发言矛盾|站边问题)",
        ]
        return any(re.search(p, text) for p in patterns)
