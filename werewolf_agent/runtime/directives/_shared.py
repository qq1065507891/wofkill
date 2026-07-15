# -*- coding: utf-8 -*-
"""提供多个角色指令构建器共享的公开历史与约束 helper。

作者: Project contributors
创建日期: 2025-01-15
修改日期: 2026-07-15
使用示例: 内部模块，无对外接口
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from typing import Any

from werewolf_agent.core.models import GameState
from werewolf_agent.core.resolution_batches import parse_resolution_batch

logger = logging.getLogger(__name__)

_WARNED_BATCH_PARSE_FAILURES: set[tuple[str, str]] = set()


def collect_public_vote_history(
    gs: GameState,
    current_day: int | None = None,
) -> str:
    """Collect public vote history for villager analysis.

    M3-2: optional ``current_day`` filter.  When supplied, only
    events with ``payload.day_number <= current_day`` are
    included.  Default ``None`` (no filter) preserves the
    pre-fix behavior for back-compat callers.  By day 5 the
    LLM was reading all 5 days of vote history and losing
    focus on the current game state; this filter lets
    directives cap the helper to "up to current day" without
    changing the helper signature.
    """
    lines: list[str] = []
    for e in gs.events:
        if e.type != "vote_resolved":
            continue
        day = e.payload.get("day_number", "?")
        if (
            current_day is not None
            and isinstance(day, int)
            and day > current_day
        ):
            continue
        exiled = e.payload.get("exiled")
        tied = e.payload.get("tied", [])
        votes = e.payload.get("votes", [])
        if exiled:
            # votes is a list of {"voter": ..., "target": ..., "reason": ...}
            supporters = [
                v.get("voter", "") for v in votes
                if isinstance(v, dict) and v.get("target") == exiled
            ]
            lines.append(f"D{day}: {exiled}被放逐（投TA的: {', '.join(supporters)}）")
        elif tied:
            lines.append(f"D{day}: 平票PK {', '.join(tied)}，无人出局")
    if not lines:
        return ""
    return "\n".join(lines)


def collect_death_order(
    gs: GameState,
    current_day: int | None = None,
) -> str:
    """Collect public death order for villager analysis.

    Only exile and hunter_shot reasons are public knowledge.
    wolf_kill and witch_poison are indistinguishable to players -- both are night deaths.

    ``current_day`` 表示当前已进入 Dn；Dk 与该日清晨公布的 Nk
    都只在 ``k <= n`` 时可见。无法解析的批次和未来 day/night
    批次一律失败关闭，避免把未来死亡注入当前指令。
    """
    _public_reasons = {"exile": "放逐", "hunter_shot": "枪杀"}
    lines: list[str] = []
    for d in gs.deaths:
        if current_day is not None:
            parsed = parse_resolution_batch(d.resolution_batch)
            if parsed.batch_parse_failed:
                raw_batch = parsed.raw_value or ""
                raw_hash = hashlib.sha256(raw_batch.encode("utf-8")).hexdigest()
                warning_key = (gs.game_id, raw_hash)
                if warning_key not in _WARNED_BATCH_PARSE_FAILURES:
                    _WARNED_BATCH_PARSE_FAILURES.add(warning_key)
                    batch_type = (
                        "str"
                        if isinstance(d.resolution_batch, str)
                        else "mapping"
                        if isinstance(d.resolution_batch, Mapping)
                        else "other"
                    )
                    logger.warning(
                        "collect_death_order: malformed resolution_batch "
                        "batch_type=%s batch_hash=%s",
                        batch_type,
                        raw_hash[:12],
                    )
                # 未知批次无法证明属于当前日，必须 fail closed。
                continue
            if (
                parsed.batch is not None
                and parsed.batch.number > current_day
            ):
                continue
        label = _public_reasons.get(d.reason)
        if label:
            lines.append(f"{d.player_id}({label})")
        else:
            lines.append(d.player_id)
    if not lines:
        return ""
    return " → ".join(lines)


def build_sheriff_silent_directive(
    gs: GameState,
    sheriff_id: str | None,
    badge_state: str,
) -> dict[str, Any]:
    """Build the no-active-sheriff vote directive.

    P0-G3223805846-9: when the badge has been torn (or otherwise lost)
    there is no 归票人 (vote-pusher) in the game.  Without explicit
    guidance the LLM tends to fall back on personal whim, or to
    "follow the loudest voice", both of which are easy for the wolf
    team to exploit by simply being the loudest faction.

    This directive injects a 归票 hint that tells the model to:
      1) follow the publicly confirmed 查杀 side (jumped/fake-seer
         hunter) if one is on the table;
      2) otherwise follow a player who has demonstrated a clear
         站边 (side-taking) logic, not just loudness;
      3) avoid voting without evidence on D1/D2 — the first two
         days should be information-gathering.

    The dict key ``no_sheriff_vote_hint`` is intentionally distinct
    from the existing ``sheriff_silent`` key in ``agent_adapter.py``,
    which is reserved for the *silenced-but-alive* sheriff case.
    Naming the two directives differently prevents the two distinct
    no-归票 scenarios from being conflated by the LLM (and by the
    test suite guarding them).
    """
    parts: dict[str, Any] = {}
    if badge_state != "torn" or sheriff_id is not None:
        # Active sheriff or pre-game setup — this directive is a
        # no-op.  Callers should gate on the same condition used
        # in agent_adapter.py (``gs.sheriff_id is None and
        # gs.sheriff_badge_state == "torn"``).
        return parts
    parts["no_sheriff_vote_hint"] = (
        "【无警长归票提示 P0-G3223805846-9】本局警徽已流失，无警长归票人。"
        "投票建议：\n"
        "1) 如有公开查杀或预言家对跳，先核验预言家可信度、验人链和前后逻辑，"
        "再决定是否投查杀对象；\n"
        "2) 如无可信查杀，基于发言矛盾、票型和站边链独立归票；\n"
        "3) 在证据接近时说明取舍，避免无理由跟票。"
    )
    return parts


# ---------------------------------------------------------------------------
# NEW (v1.1.4 fallback-fix, Part A.2 + B.2)
#
# These two constants are stable MUST-text injected into the system prompt
# via ``strategy_directive`` (rendered under the 【硬约束】 section by
# ``PromptStrategyMixin._build_strategy_directive``).  The goal is to
# reduce the two largest sources of fallback observed in 7-14+ games:
#   - ``speech_quality``           (49/86 = 57% of fallbacks)
#   - ``semantic_claim_retention`` (30/86 = 35% of fallbacks)
#
# We inject these into strategy_directive rather than relying on retry
# hints because retry hints only fire AFTER the LLM has already failed
# once; the contract text below preempts the failure mode by making the
# MUST visible at every step.
#
# Refs:
#   - Part A.1 ``runtime/context.py:301-310`` (priority 门槛从 high 放宽)
#   - Part A.3 ``runtime/speech_quality.py::_required_components`` (强制 stance)
#   - Part D.1 ``agents/prompt_output.py::_build_output_contract`` (JSON 硬约束)
# ---------------------------------------------------------------------------

_SPEECH_QUALITY_HARD_CONSTRAINTS = (
    "【发言质量硬约束 / MUST】\n"
    "1) 发言开头先表明身份立场（一句话：我是好人阵营 / 我是预言家 / 我是女巫等），"
    "不要写\"按公开信息判断\"之类的占位文本。\n"
    "2) 必须给出至少 1 个具体怀疑对象（用玩家 ID 如 p05/p07），不要泛指\"某玩家\"。\n"
    "3) 必须给出投票倾向（我倾向投 pXX / 我归票 pXX / 我保留观望）。\n"
    "4) 必须引用至少 1 条公开依据（预言家查杀、金水、对跳、票型突变、警徽流、发言前后矛盾）。\n"
    "5) PK / 警徽 / 遗言阶段必须包含角色声明、对跳分析或攻击/防守论点。\n"
    "6) 不要写\"先听\"、\"再观察\"、\"信息不足\"、\"我没什么可说\"等空洞起手式——这些会被 ``validate_public_speech`` 直接判为 filler 并触发 fallback。"
)

_SPEECH_CONSISTENCY_HARD_CONSTRAINTS = (
    "【发言一致性硬约束 / MUST（适用重写场景）】\n"
    "1) 重写发言时必须保持源 target_id 不变（不得更换攻击对象）。\n"
    "2) 不得新增事实声明——所有数据点必须能在公开记录或近因发言中找到对应原文。\n"
    "3) 不能因为 retry hint 而改变行动（投谁/杀谁）——仅优化发言措辞。\n"
    "4) 若必须回应矛盾点，请基于已有公开引用，标注\"我推测/我质疑\"，不要把推断写成\"公开记录已证明\"。"
)


def build_speech_quality_hard_constraints() -> dict[str, str]:
    """返回 ``strategy_directive`` 兼容的 dict — 注入发言质量硬约束。

    Key 是 ``speech_quality_constraints``；该 key 不在 ``HARD_CONSTRAINT_KEYS`` /
    ``SUGGESTION_KEYS`` / ``REFERENCE_KEYS`` 任何白名单内，会落入 ``【参考】`` 段。
    这是有意为之：硬约束文档应与 ``must_address_alerts``（已在 HARD）
    并行渲染，在 system prompt 顶层可见。
    """
    return {"speech_quality_constraints": _SPEECH_QUALITY_HARD_CONSTRAINTS}


def build_speech_consistency_hard_constraints() -> dict[str, str]:
    """返回 ``strategy_directive`` 兼容的 dict — 注入发言一致性硬约束。

    Key 是 ``speech_consistency_constraints``；与 ``speech_quality_constraints``
    同处理（落入【参考】段）。两个 key 一起渲染保证 system prompt 顶层可见。
    """
    return {"speech_consistency_constraints": _SPEECH_CONSISTENCY_HARD_CONSTRAINTS}
