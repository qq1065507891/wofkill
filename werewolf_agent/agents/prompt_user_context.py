# -*- coding: utf-8 -*-
"""
渲染玩家 user prompt 中随回合变化的当前局上下文片段。

作者: Project contributors
创建日期: 2026-07-08
修改日期: 2026-07-09

使用示例:
    >>> from werewolf_agent.agents.prompt_user_context import PromptUserContextMixin
"""

from __future__ import annotations

import re
from typing import Any

from werewolf_agent.agents.schemas import ActionType, OutputMode

_MAX_PUBLIC_SUMMARY_CHARS = 700
_MAX_TRANSCRIPT_ITEMS = 4
_MAX_TRANSCRIPT_TEXT_CHARS = 220
_MAX_PERSONA_LINE_CHARS = 180


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_current_game_token(value: Any, *, max_chars: int) -> str:
    text = str(value or "").strip()
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text[:max_chars]


def _clean_current_game_list_items(
    value: Any,
    *,
    limit: int,
    max_chars: int,
) -> list[str]:
    """清理当前局文本列表，保留 p03 等本局玩家编号。"""
    if not isinstance(value, list):
        return []
    return [
        _clean_current_game_token(item, max_chars=max_chars)
        for item in value[:limit]
        if str(item or "").strip()
    ]


class PromptUserContextMixin:
    """构建 user prompt 中当前回合可见状态、推理辅助和 transcript 片段。"""

    def _build_phase_context(self) -> str:
        ctx = self.context
        lines = [f"当前阶段: {ctx.phase}"]
        if ctx.legal_actions:
            lines.append(f"可用操作: {[a.value for a in ctx.legal_actions]}")
        if ctx.legal_targets:
            lines.append(f"可选目标: {ctx.legal_targets}")
        output_mode = self._select_output_mode()
        is_exile_vote_context = self._is_exile_vote_context()
        if ctx.legal_actions and is_exile_vote_context:
            if ActionType.NO_ACTION not in ctx.legal_actions:
                lines.append("重要：本轮投票必须选择一名玩家放逐，不能弃票！")
            if ctx.legal_actions == [ActionType.VOTE] and ctx.legal_targets:
                lines.append("你必须投出选票，从可选目标中选择一人。")
            json_label = (
                "choice 决策JSON"
                if output_mode == OutputMode.TARGET_CHOICE
                else "JSON"
            )
            lines.append(
                f"投票时必须先在心里完成判断，并在{json_label}中额外给出这些私有字段："
                "seer_stance（枚举：trust/distrust/undecided/no_claim）、"
                "vote_basis（枚举：seer_check/seer_siding/speech_logic/vote_pattern/pressure_test/anti_herd/fallback）、"
                "standing_with_seer（你站边哪个预言家/逻辑线，没有则写空字符串）、"
                "suspect_reason（为什么怀疑最终投票对象）、"
                "not_voting_reason（为什么不投其他主要候选人）、"
                "candidate_comparison（至少两名候选人的公开证据与反证对比）、"
                "private_reason（完整内心活动：为什么投他、担心什么、最终如何决定）。"
                "这些字段不会公开发言，只给主持人审计。"
            )
            _GOOD_SIDE = {"villager", "seer", "witch", "hunter", "idiot"}
            _WOLF_SIDE = {"werewolf"}
            role = ctx.own_role or ""
            if role == "hybrid":
                is_wolf_side = False
            else:
                is_wolf_side = role in _WOLF_SIDE
            if is_wolf_side:
                lines.append(
                    "狼队抱团是正常策略；投票时跟队友一致是预期行为；"
                    "只有在倒钩场景下需独立判断。"
                    "P2-5: 悍跳狼应跟悍跳队友的归票走，而非原狼队的票型。"
                )
            else:
                lines.append(
                    "反跟票警告：不要无条件跟随任何人的归票。如果多人集中投同一人，"
                    "检查是否可能是狼人抱团。独立判断优先级：发言逻辑矛盾 > 票型异常 > 谁说了什么。"
                    "不要仅因立场反复或票型矛盾机械抗推可能神职；涉及预言家、女巫、猎人、白痴时，"
                    "先核验证据链和技能风险。"
                    "投票前至少比较两名候选人，分别写清公开证据、反证和不投其他主要候选人的理由，"
                    "不能只写跟票、感觉可疑或继续施压。"
                )
        return "\n".join(lines)

    def _build_belief_state(self) -> str:
        ctx = self.context
        if not ctx.belief_state:
            return ""
        _MAX_BELIEF_ITEMS = 3
        suspects = ctx.belief_state.get("my_suspects", [])
        trusted = ctx.belief_state.get("my_trusted", [])
        belief_lines = []
        if suspects:
            suspect_desc = ", ".join(
                f"{s['player']}(嫌疑{s['faction_lean']}, 猜{s['top_role_guess']})"
                for s in suspects[:_MAX_BELIEF_ITEMS]
            )
            belief_lines.append(f"我怀疑的玩家: {suspect_desc}")
        if trusted:
            trust_desc = ", ".join(
                f"{t['player']}(倾向{t['faction_lean']}, 信任{t['trust']})"
                for t in trusted[:_MAX_BELIEF_ITEMS]
            )
            belief_lines.append(f"我信任的玩家: {trust_desc}")
        if belief_lines:
            return "【我的判断（基于已有信息的推理，可能是错的）】" + " ".join(belief_lines)
        return ""

    def _build_contradiction_alerts(self) -> str:
        ctx = self.context
        alerts = ctx.contradiction_alerts
        if not alerts:
            return ""
        _MAX = 3
        _PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
        typed = [a for a in alerts if isinstance(a, dict)]
        if not typed:
            return ""
        typed.sort(key=lambda a: _PRIORITY_ORDER.get(str(a.get("priority", "")).lower(), 3))
        lines = ["公开矛盾点（从公开行为推出，可作为攻击/防守依据，非裁判定性）:"]
        for a in typed[:_MAX]:
            player = _clean_current_game_token(a.get("player_id") or "", max_chars=24)
            atype = _clean_current_game_token(a.get("alert_type") or "", max_chars=24)
            desc = _clean_current_game_token(a.get("description") or "", max_chars=120)
            line = f"- {player}"
            if atype:
                line += f" {atype}"
            if desc:
                line += f": {desc}"
            lines.append(line)
        return "\n".join(lines) if len(lines) > 1 else ""

    def _build_seer_credibility(self) -> str:
        ctx = self.context
        summary = ctx.seer_credibility
        if not summary:
            return ""
        lines_data = summary.get("seer_lines", [])
        if not lines_data:
            return ""
        lines = ["预言家线可信度（公开证据推断，非裁判真相，仅辅助推理）:"]
        for item in lines_data[:3]:
            claimant = _clean_current_game_token(item.get("claimant") or "", max_chars=12)
            status = _clean_current_game_token(item.get("status") or "", max_chars=12)
            score = _safe_float(item.get("score"), default=0.0)
            checks = ", ".join(
                _clean_current_game_token(c, max_chars=20)
                for c in item.get("checks", [])[:3]
            )
            evidence = ", ".join(
                _clean_current_game_token(e, max_chars=40)
                for e in (item.get("evidence") or [])[:3]
                if str(e or "").strip()
            )
            seg = [f"{claimant} {status} score={score:.2f}"]
            if checks:
                seg.append(f"查验[{checks}]")
            if evidence:
                seg.append(evidence)
            lines.append("- " + "; ".join(seg))
        return "\n".join(lines) if len(lines) > 1 else ""

    def _build_possible_worlds(self) -> str:
        ctx = self.context
        if not ctx.possible_worlds:
            return ""
        worlds = ctx.possible_worlds.get("top_worlds")
        if not isinstance(worlds, list) or not worlds:
            return ""
        warning = self._clean_prompt_text(
            ctx.possible_worlds.get(
                "warning",
                "These are hypotheses from visible evidence, not ground truth.",
            ),
            max_chars=160,
        )
        lines = [
            "可能世界假设: 以下是假设，不是裁判真相；只能用于私有推理，不能当作公开事实。"
        ]
        if warning:
            lines.append(warning)
        for idx, world in enumerate(worlds[:3], start=1):
            if not isinstance(world, dict):
                continue
            label = _clean_current_game_token(
                world.get("label") or f"World {idx}",
                max_chars=40,
            )
            probability = _safe_float(world.get("probability"), default=0.0)
            assignments = world.get("key_assignments")
            if isinstance(assignments, dict):
                assignment_text = ", ".join(
                    f"{_clean_current_game_token(pid, max_chars=16)}="
                    f"{_clean_current_game_token(role, max_chars=24)}"
                    for pid, role in sorted(assignments.items())[:4]
                )
            else:
                assignment_text = ""
            why = _clean_current_game_list_items(
                world.get("why"), limit=2, max_chars=80
            )
            watch_for = _clean_current_game_list_items(
                world.get("watch_for"), limit=2, max_chars=80
            )
            line = f"- {label}: prob={probability:.2f}"
            if assignment_text:
                line += f"; key={assignment_text}"
            if why:
                line += "; why=" + "；".join(why)
            if watch_for:
                line += "; watch=" + "；".join(watch_for)
            lines.append(line)
        return "\n".join(lines) if len(lines) > 1 else ""

    def _build_simulation_predictions(self) -> str:
        ctx = self.context
        simulation = ctx.simulation_predictions
        if not simulation or simulation.get("type") != "simulation":
            return ""
        predictions = simulation.get("predictions")
        if not isinstance(predictions, list) or not predictions:
            return ""
        warning = self._clean_prompt_text(
            simulation.get("warning", "Prediction, not fact."),
            max_chars=120,
        )
        horizon = self._clean_prompt_text(
            simulation.get("horizon", "next_turn"),
            max_chars=40,
        )
        lines = [
            f"Simulation predictions ({horizon}): use as private planning signals only."
        ]
        if warning:
            lines.append(warning)
        for item in predictions[:2]:
            if not isinstance(item, dict):
                continue
            event = _clean_current_game_token(item.get("event"), max_chars=48)
            if not event:
                continue
            probability = _safe_float(item.get("probability"), default=0.0)
            affected_raw = item.get("affected_players")
            affected = [
                _clean_current_game_token(player_id, max_chars=16)
                for player_id in (
                    affected_raw[:3] if isinstance(affected_raw, list) else []
                )
                if str(player_id or "").strip()
            ]
            rationale = _clean_current_game_token(
                item.get("rationale"),
                max_chars=100,
            )
            world_ids = _clean_current_game_list_items(
                item.get("world_ids"),
                limit=3,
                max_chars=32,
            )
            line = f"- {event}: prob={probability:.2f}"
            if affected:
                line += "; players=" + ", ".join(affected)
            if world_ids:
                line += "; worlds=" + ", ".join(world_ids)
            if rationale:
                line += "; why=" + rationale
            lines.append(line)
        return "\n".join(lines) if len(lines) > 1 else ""

    def _build_public_summary(self) -> str:
        ctx = self.context
        if not ctx.public_summary:
            return ""
        return "当前局公开事实:\n" + self._truncate_text(
            ctx.public_summary,
            _MAX_PUBLIC_SUMMARY_CHARS,
        )

    def _build_visible_state(self) -> str:
        ctx = self.context
        if not ctx.visible_world_state:
            return ""
        visible = dict(ctx.visible_world_state)
        visible.pop("private_memory", None)
        if not visible:
            return ""
        return "可见状态: " + self._compact_json(visible)

    def _build_recent_transcript(self) -> str:
        ctx = self.context
        if not ctx.recent_transcript:
            return ""
        items = sorted(
            ctx.recent_transcript[-_MAX_TRANSCRIPT_ITEMS:],
            key=lambda it: (
                it.get("day_number", it.get("day", 0)),
                it.get("phase_order", 0),
            ),
        )
        lines: list[str] = ["近期发言:"]
        for item in items:
            speaker = item.get("speaker", "?")
            text = self._truncate_text(
                str(item.get("text", "")),
                _MAX_TRANSCRIPT_TEXT_CHARS,
            )
            lines.append(f"  [{speaker}] {text}")
        return "\n".join(lines)


__all__ = [
    "PromptUserContextMixin",
    "_MAX_PERSONA_LINE_CHARS",
    "_MAX_PUBLIC_SUMMARY_CHARS",
    "_MAX_TRANSCRIPT_ITEMS",
    "_MAX_TRANSCRIPT_TEXT_CHARS",
    "_clean_current_game_list_items",
    "_clean_current_game_token",
    "_safe_float",
]
