# -*- coding: utf-8 -*-
"""
功能描述：女巫毒药候选目标提取——按公开证据强度排序，为 LLM 提供结构化候选列表。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：内部模块，无对外接口
"""
from __future__ import annotations

import re
from typing import Any

from werewolf_agent.core.models import GameState


# 查杀关键词 (公开查杀声明)
_CHECK_KILL_KEYWORDS = ("查杀", "验出狼", "是狼人", "是狼", "是坏人")

# 普通指控关键词 (比查杀弱)
_ACCUSATION_KEYWORDS = ("是狼人", "是狼", "像是狼", "像狼", "我怀疑是狼", "站边投", "投TA是狼")


def collect_witch_poison_candidates(
    gs: GameState,
    witch_id: str,
) -> list[dict[str, Any]]:
    """从公开事件中按证据强度排序收集女巫毒药候选目标。

    返回 list[dict],每个 dict 含 ``player_id`` / ``reason`` / ``score`` / ``sources``。
    排序按 score 降序;同分按 source 数量降序。

    已死玩家被自动排除;活着的非狼玩家才会被考虑 (因为公开事件是女巫唯一视野)。
    女巫自己不会被任何来源指为狼 (不会出现在候选中)。
    """
    candidates: dict[str, dict[str, Any]] = {}

    # ============ 来源 1: 公开查杀声明 (权重 +10) ============
    # 路径 A: 通过 build_world_state 的 seer_check_claim fact 提取 (仅匹配 p\d{2} 格式)
    seen_check_targets: set[str] = set()
    try:
        from werewolf_agent.cognition.world_state import build_world_state
        ws = build_world_state(gs)
        for f in ws.facts_of_type("seer_check_claim"):
            val = (f.value or "").lower()
            if not ("wolf" in val or "狼" in (f.value or "")):
                continue
            target = f.target_player
            source = f.source_player
            if not target or target == witch_id:
                continue
            if not _is_alive(gs, target):
                continue
            entry = candidates.setdefault(target, {
                "player_id": target,
                "reason": "",
                "score": 0,
                "sources": [],
            })
            entry["score"] += 10
            entry["sources"].append(f"被{source or '?'}公开查杀")
            seen_check_targets.add(target)
    except Exception:
        pass

    # 路径 B: 直接扫描 speech,捕获短 ID (w1/seer 等) 和未通过 fact extractor 的查杀声明
    # 这是路径 A 的兜底,确保测试夹具和真实游戏都能命中
    for event in gs.events:
        if event.type not in ("speech", "sheriff_speech"):
            continue
        text = event.payload.get("text", "")
        speaker = event.payload.get("speaker", "")
        if not text or speaker == witch_id:
            continue
        # 在同一段 speech 中查找所有查杀/指控目标 (用 findall 而非 search,
        # 因为玩家可能在同段话里指证多个目标)
        accused_targets: set[str] = set()
        # 模式 1: 验了 X 是狼 / 第 N 夜验了 X 是狼
        for m in re.finditer(
            r"(?:验了?|查验|验人|第\s*\d+\s*夜验了?)\s*"
            r"([a-zA-Z]+\d+)[^,。]{0,8}(?:查杀|是狼|是狼人|是坏人)",
            text,
        ):
            accused_targets.add(m.group(1))
        # 模式 2: X 查杀 / X 是狼人 / X 是狼 (短格式指控)
        for m in re.finditer(
            r"([a-zA-Z]+\d+)[^,。]{0,8}(?:查杀|是狼人|是狼)",
            text,
        ):
            accused_targets.add(m.group(1))
        for target in accused_targets:
            if target in seen_check_targets or target == witch_id:
                continue
            if not _is_alive(gs, target):
                continue
            entry = candidates.setdefault(target, {
                "player_id": target,
                "reason": "",
                "score": 0,
                "sources": [],
            })
            entry["score"] += 10
            entry["sources"].append(f"被{speaker}公开查杀 (speech 扫描)")
            seen_check_targets.add(target)

    # ============ 来源 2: 公开 speech 中的查杀/指控 (权重按人数) ============
    accusation_count: dict[str, set[str]] = {}
    for event in gs.events:
        if event.type not in ("speech", "sheriff_speech"):
            continue
        text = event.payload.get("text", "")
        speaker = event.payload.get("speaker", "")
        if not text or speaker == witch_id:
            continue
        if not any(kw in text for kw in _ACCUSATION_KEYWORDS + _CHECK_KILL_KEYWORDS):
            continue
        # 提取被指控的玩家
        # 优先匹配 "X 查杀" / "X 是狼" 模式
        accused = _extract_accused_player(text)
        if not accused or accused == witch_id:
            continue
        if not _is_alive(gs, accused):
            continue
        accusation_count.setdefault(accused, set()).add(speaker)

    for target, accusers in accusation_count.items():
        if len(accusers) >= 2:
            entry = candidates.setdefault(target, {
                "player_id": target,
                "reason": "",
                "score": 0,
                "sources": [],
            })
            # 2 人指控 = +6, 3+ 人 = +8
            entry["score"] += 6 if len(accusers) == 2 else 8
            entry["sources"].append(
                f"被{len(accusers)}人明确指控 ({', '.join(sorted(accusers))})"
            )

    # ============ 排序 + 渲染 reason ============
    sorted_cands = sorted(
        candidates.values(),
        key=lambda c: (c["score"], len(c["sources"])),
        reverse=True,
    )
    for c in sorted_cands:
        # reason 字段是渲染时用的中文描述
        primary_source = c["sources"][0] if c["sources"] else "公开证据"
        c["reason"] = primary_source
    return sorted_cands


def _extract_accused_player(text: str) -> str | None:
    """从 speech text 中提取被指控的玩家 ID。

    优先匹配 'X 查杀' / 'X 是狼' 模式,其次匹配 '查杀 X' / '是狼 X' 反向模式。
    返回第一个匹配的玩家 ID,没有则 None。
    """
    # 正向: PLAYER 查杀 / PLAYER 是狼人
    for kw in _CHECK_KILL_KEYWORDS:
        m = re.search(rf"([a-zA-Z]+\d+)[^,。]{{0,8}}{kw}", text)
        if m:
            return m.group(1)
    # 反向: 查杀 PLAYER / 是狼 PLAYER
    for kw in _CHECK_KILL_KEYWORDS:
        m = re.search(rf"{kw}[^,。]{{0,8}}([a-zA-Z]+\d+)", text)
        if m:
            return m.group(1)
    # 弱指控: PLAYER 像是狼
    for kw in _ACCUSATION_KEYWORDS:
        m = re.search(rf"([a-zA-Z]+\d+)[^,。]{{0,12}}{kw}", text)
        if m:
            return m.group(1)
    return None


def _is_alive(gs: GameState, player_id: str) -> bool:
    """Check if a player is currently alive (helper for poison candidate filter)."""
    p = gs.players.get(player_id)
    return bool(p and p.alive)
