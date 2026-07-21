# -*- coding: utf-8 -*-
"""
构建狼队夜聊阶段的策略指令、队友摘要和兜底文案。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-21

使用示例:
    >>> from werewolf_agent.runtime.wolf_discussion_directives import build_wolf_discussion_instruction
    >>> build_wolf_discussion_instruction("w1", night_number=1, has_teammate_input=False, has_previous_speeches=False)
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.agents.schemas import WolfTargetStance, WolfTargetStanceAction
from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.core.event_visibility import EventVisibility
from werewolf_agent.runtime.event_metadata import validate_v2_event_identity


def living_wolf_ids(gs: GameState) -> list[str]:
    """返回当前存活狼人 ID。"""
    return [
        player_id
        for player_id, player in gs.players.items()
        if player.alive and player.role == "werewolf"
    ]


def living_wolf_teammates(gs: GameState, wolf_id: str) -> list[str]:
    """返回当前狼人之外的存活狼队队友。"""
    return [player_id for player_id in living_wolf_ids(gs) if player_id != wolf_id]


def _trusted_current_wolf_discussion_event(
    gs: GameState,
    event: GameEvent,
    *,
    wolf_ids: set[str] | None = None,
) -> bool:
    """判断事件能否作为本夜狼队 prompt 或 stance 的权威来源。"""
    if event.type != "wolf_discussion":
        return False
    try:
        validate_v2_event_identity(
            gs.game_id,
            event,
            required_visibility=EventVisibility.WEREWOLF_TEAM_ONLY,
        )
    except ValueError:
        return False
    wolf_id = event.payload.get("wolf_id")
    round_number = event.payload.get("round")
    wolf = gs.players.get(str(wolf_id))
    return bool(
        event.payload.get("night_number") == gs.night_number
        and isinstance(round_number, int)
        and not isinstance(round_number, bool)
        and round_number >= 1
        and wolf is not None
        and wolf.alive
        and wolf.role == "werewolf"
        and (wolf_ids is None or wolf_id in wolf_ids)
    )


def collect_wolf_discussion_speeches(
    gs: GameState,
    wolf_ids: list[str],
) -> list[dict[str, str]]:
    """收集狼队成员已有的夜聊发言。"""
    wolf_id_set = set(wolf_ids)
    speeches: list[dict[str, str]] = []
    for event in gs.events:
        if _trusted_current_wolf_discussion_event(
            gs,
            event,
            wolf_ids=wolf_id_set,
        ):
            speeches.append({
                "wolf_id": str(event.payload.get("wolf_id", "")),
                "round": str(event.payload.get("round", "")),
                "text": str(event.payload.get("text", "")),
            })
    return speeches


def collect_current_wolf_target_stances(gs: GameState) -> list[dict[str, Any]]:
    """读取本夜已验证的结构化 stance，不读取或解析自由文本。"""
    stances: list[dict[str, Any]] = []
    for event in gs.events:
        if not _trusted_current_wolf_discussion_event(gs, event):
            continue
        raw_stance = event.payload.get("target_stance")
        if not isinstance(raw_stance, dict):
            continue
        try:
            stance = WolfTargetStance.model_validate(raw_stance)
        except ValueError:
            continue
        wolf = gs.players.get(stance.wolf_id)
        target = gs.players.get(stance.target_id) if stance.target_id else None
        if (
            stance.source_event_id != event.event_id
            or stance.wolf_id != event.payload.get("wolf_id")
            or stance.round_number != event.payload.get("round")
            or wolf is None
            or not wolf.alive
            or wolf.role != "werewolf"
            or (
                stance.target_id is not None
                and (
                    target is None
                    or target.role == "werewolf"
                )
            )
        ):
            continue
        stances.append(stance.model_dump())
    return stances


def build_layered_wolf_discussion_context(
    gs: GameState,
    wolf_ids: list[str],
    *,
    recent_raw_limit: int = 8,
    older_summary_chars: int = 120,
) -> dict[str, Any]:
    """分离权威结构化状态与有界夜聊文本，并返回脱敏审计计数。"""
    if recent_raw_limit < 0:
        raise ValueError("recent_raw_limit must be non-negative")
    if older_summary_chars < 1:
        raise ValueError("older_summary_chars must be positive")

    wolf_id_set = set(wolf_ids)
    rows: list[dict[str, Any]] = []
    for event in gs.events:
        if not _trusted_current_wolf_discussion_event(
            gs,
            event,
            wolf_ids=wolf_id_set,
        ):
            continue
        rows.append({
            "event_id": str(event.event_id),
            "wolf_id": str(event.payload["wolf_id"]),
            "round": int(event.payload["round"]),
            "text": str(event.payload.get("text") or ""),
        })

    raw_start = max(0, len(rows) - recent_raw_limit)
    older_rows = rows[:raw_start]
    recent_rows = rows[raw_start:]
    older_summary: list[dict[str, Any]] = []
    truncated_count = 0
    for row in older_rows:
        normalized = " ".join(row["text"].split())
        truncated = len(normalized) > older_summary_chars
        if truncated:
            normalized = normalized[:older_summary_chars].rstrip() + "…"
            truncated_count += 1
        older_summary.append({
            "event_id": row["event_id"],
            "wolf_id": row["wolf_id"],
            "round": row["round"],
            "summary": normalized,
        })

    alive_wolves = sorted(
        player_id
        for player_id, player in gs.players.items()
        if player.alive and player.role == "werewolf"
    )
    alive_non_wolves = sorted(
        player_id
        for player_id, player in gs.players.items()
        if player.alive and player.role != "werewolf"
    )
    return {
        "structured": {
            "live_status": {
                "night_number": gs.night_number,
                "alive_wolves": alive_wolves,
                "alive_non_wolves": alive_non_wolves,
            },
            # stance 不受原文条数和摘要长度预算影响。
            "target_stances": collect_current_wolf_target_stances(gs),
        },
        "text": {
            "recent_raw": recent_rows,
            "older_summary": older_summary,
        },
        "audit": {
            "injected_event_ids": [row["event_id"] for row in rows],
            "raw_text_count": len(recent_rows),
            "summarized_text_count": len(older_summary),
            "truncated_text_count": truncated_count,
        },
    }


def build_validated_wolf_target_stance(
    gs: GameState,
    discussion_event: GameEvent,
    *,
    wolf_id: str,
    round_number: int,
    raw_stance: dict[str, Any] | WolfTargetStanceAction | None,
) -> WolfTargetStance:
    """结合实时存活状态，把 action 草稿补全为可审计的完整 stance。"""
    wolf = gs.players.get(wolf_id)
    if wolf is None or not wolf.alive or wolf.role != "werewolf":
        raise ValueError("wolf_id must identify a current alive werewolf")
    if not _trusted_current_wolf_discussion_event(
        gs,
        discussion_event,
        wolf_ids={wolf_id},
    ):
        raise ValueError("source_event_id must reference this night's V2 wolf_discussion")
    if (
        discussion_event.payload.get("wolf_id") != wolf_id
        or discussion_event.payload.get("round") != round_number
    ):
        raise ValueError("source wolf_id/round must match target stance")

    draft = WolfTargetStanceAction.model_validate(
        raw_stance
        or {
            "target_id": None,
            "stance": "abstain",
            "priority": "primary",
        }
    )
    if draft.target_id is not None:
        target = gs.players.get(draft.target_id)
        if target is None or not target.alive or target.role == "werewolf":
            raise ValueError("target_id must identify a current alive non-werewolf")

    return WolfTargetStance(
        wolf_id=wolf_id,
        target_id=draft.target_id,
        stance=draft.stance,
        priority=draft.priority,
        source_event_id=discussion_event.event_id,
        round_number=round_number,
    )


def teammate_discussion_speeches(
    previous_speeches: list[dict[str, str]],
    wolf_id: str,
) -> list[dict[str, str]]:
    """从夜聊历史中筛出队友发言。"""
    return [
        speech
        for speech in previous_speeches
        if speech["wolf_id"] != wolf_id
    ]


def build_wolf_discussion_instruction(
    wolf_id: str,
    *,
    night_number: int,
    has_teammate_input: bool,
    has_previous_speeches: bool,
) -> str:
    """构建当前狼人夜聊发言的核心提示。"""
    instruction = (
        "这是狼队密谈，只有狼人队友能看到。你必须以狼人视角发言，讨论狼队策略。"
        "禁止假装好人视角发言，禁止质疑或试探队友身份——你清楚知道谁是队友。"
        "必须发言，不能沉默。必须提出具体的击杀目标或战术建议。\n"
        "注意用词：被放逐或已死的队友是'队友'或'悍跳狼'，不要叫TA'预言家'。"
        "即使TA白天冒充了预言家，在狼队内部你们应该用真实身份称呼。\n"
        f"【身份约束】你的玩家ID是{wolf_id}。在发言中只能以{wolf_id}自称，"
        "绝对不能自称其他玩家的ID或使用别人的ID发言。"
    )
    if has_teammate_input:
        instruction += (
            "\n\n重要：你必须回应队友的发言！看看队友提出了什么建议，"
            "表示同意、反对或补充意见，形成真正的团队讨论，而不是自顾自发言。"
        )
    if night_number == 1 and not has_previous_speeches:
        instruction += (
            "\n\n【首夜角色分工建议】狼队可以分工配合 (字段名 ↔ 中文):\n"
            "1) fake_seer (悍跳位)——白天假装预言家争夺警徽（建议由能言善辩的队友担任）\n"
            "2) pusher (冲锋位)——为悍跳队友强力站边，质疑真预言家\n"
            "3) hooker (倒钩位)——表面上站边真预言家，暗中破坏好人节奏\n"
            "4) deep_cover (深水位)——保持低调，活到最后为团队收尾\n"
            "讨论谁适合什么角色，但不一定每局都需要悍跳。"
            "首夜只能预设悍跳/倒钩等备选分工，"
            "不要在缺少白天公开发言证据时机械悍跳；"
            "是否悍跳应等公开发言、查验冲突或警徽竞争形势出现后再决定。"
            "建议在发言里明确写出角色名 (例如 '我做悍跳' / 'p04 做倒钩'), "
            "队长会基于这些表态做最终结构化分工。"
        )
    return instruction


def build_wolf_discussion_strategy_directive(
    *,
    discussion_instruction: str,
    round_focus: str,
    wolf_teammates: list[str],
    previous_speeches: list[dict[str, str]],
    layered_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构建狼队夜聊的策略指令字典。

    2026-07-21 1a-verify 暴露推理模型在 plan 信封模式下经常省略 ``target_stance``
    字段，导致 ``_planned_wolf_kill`` 走 strategic_abstain 空刀。planning 层
    透传修复只是必要不充分；这里在 prompt 里硬性要求 LLM 必须产出该字段，
    并给出合法枚举示例，让 LLM 行为层选择输出。
    """
    directive = {
        "wolf_team_discussion": discussion_instruction,
        "round_focus": round_focus,
        "wolf_teammates": wolf_teammates,
        "previous_discussion": previous_speeches[-8:],
        # 必填项：plan 信封模式下必须产出 target_stance
        # （顶层键与 build_full_action_schema 中 WOLF_DISCUSSION 分支一致）
        "target_stance_contract": (
            "【必填 / MUST】输出 JSON 时必须包含顶级键 `target_stance`，"
            "否则本轮立场证据作废，强制 strategic_abstain 空刀。"
            "合法结构（对象或 null）：\n"
            "{"
            "\"target_id\": \"p05\" 或 null, "
            "\"stance\": \"propose\" | \"support\" | \"oppose\" | \"abstain\", "
            "\"priority\": \"primary\" | \"backup\""
            "}\n"
            "规则：\n"
            "- 若本轮想给某玩家留击杀目标 → target_id 必填，stance=propose|support，priority=primary；\n"
            "- 若有备选 → 再用 priority=backup 输出第二个 stance（同一对象里多次出现）"
            "或用对话里提到的备选 id 在 stance=support 一并列名（受 schema 限制只能输出一个对象）；\n"
            "- 若本轮真的想 abstain（如讨论中还没结论）→ target_id=null，stance=abstain；\n"
            "- 严禁整段省略 target_stance 字段。即便 target_id 仍在斟酌，"
            "也要先输出 stance=abstain 让协议层知道不是数据缺失。"
        ),
    }
    if layered_context is not None:
        directive["wolf_universal_rules"] = layered_context["structured"]
        directive["previous_discussion"] = layered_context["text"]
    return directive


def build_teammate_transcript(
    teammate_speeches: list[dict[str, str]],
) -> list[dict[str, str]]:
    """把队友夜聊发言转换为 AgentContext 的 transcript 条目。"""
    return [
        {"speaker": speech["wolf_id"], "text": speech["text"]}
        for speech in teammate_speeches[-6:]
    ]


def build_empty_wolf_discussion_fallback(
    wolf_id: str,
    required_text: str,
) -> str:
    """构建不猜测任何目标的空狼队夜聊兜底文本。"""
    return (
        f"我是{wolf_id}，本轮暂不提出击杀目标。"
        f"{required_text}请大家发表意见。"
    )
